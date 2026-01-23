#!/usr/bin/env python3
"""
Binance Futures Webhook Service
Receives TradingView webhook signals and creates Binance Futures limit orders
Optimized for low-resource servers (1 CPU, 1 GB RAM)
"""

import os
import json
import logging
import hmac
import hashlib
import time
import math
import requests
from datetime import datetime
from flask import Flask, request, jsonify
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
import threading

# Load environment variables from .env file (if present)
# This allows manual testing; systemd service uses EnvironmentFile which takes precedence
try:
    from dotenv import load_dotenv
    # Load .env from service root directory (parent of src/)
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    try:
        load_dotenv(env_path)
    except (PermissionError, IOError, FileNotFoundError):
        # .env file not readable or doesn't exist - systemd will load via EnvironmentFile
        pass
except ImportError:
    # python-dotenv not installed, skip (systemd will load via EnvironmentFile)
    pass

# Try to import Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Configure logging
# Get log directory from environment or use logs directory relative to service root
LOG_DIR = os.getenv('LOG_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs'))
LOG_FILE = os.path.join(LOG_DIR, 'webhook_service.log')

# Ensure log directory exists
try:
    os.makedirs(LOG_DIR, exist_ok=True)
except Exception:
    # If we can't create directory, try current directory
    try:
        LOG_DIR = os.path.dirname(os.path.abspath(__file__))
        LOG_FILE = os.path.join(LOG_DIR, 'webhook_service.log')
        os.makedirs(LOG_DIR, exist_ok=True)
    except Exception:
        LOG_FILE = None  # Will use StreamHandler only

# Try to create file handler, fallback to StreamHandler only if it fails
handlers = [logging.StreamHandler()]
if LOG_FILE:
    try:
        handlers.append(logging.FileHandler(LOG_FILE))
    except Exception:
        # If file logging fails, continue with StreamHandler only
        pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger(__name__)

# Log if Gemini is not available
if not GEMINI_AVAILABLE:
    logger.warning("google-generativeai not available. AI validation will be disabled. Install with: pip install google-generativeai")

app = Flask(__name__)

# Configuration from environment variables
WEBHOOK_TOKEN = os.getenv('WEBHOOK_TOKEN', 'CHANGE_ME')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
BINANCE_TESTNET = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
BINANCE_SUB_ACCOUNT_EMAIL = os.getenv('BINANCE_SUB_ACCOUNT_EMAIL', '')  # For sub-account trading
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL', '')  # Slack webhook URL for error notifications
SLACK_SIGNAL_WEBHOOK_URL = os.getenv('SLACK_SIGNAL_WEBHOOK_URL', '')  # Slack webhook URL for signal notifications (set via environment variable)

# AI Validation Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
ENABLE_AI_VALIDATION = os.getenv('ENABLE_AI_VALIDATION', 'true').lower() == 'true'
AI_VALIDATION_MIN_CONFIDENCE = float(os.getenv('AI_VALIDATION_MIN_CONFIDENCE', '50'))
ENABLE_AI_PRICE_SUGGESTIONS = os.getenv('ENABLE_AI_PRICE_SUGGESTIONS', 'true').lower() == 'true'  # Manual review mode

# Initialize Gemini API if available and configured
gemini_client = None
gemini_model_name = None
# Free tier models (in order of preference - higher limits first)
# Priority: Models with higher daily limits (RPD) first
# Based on actual available models from Google AI Studio
GEMINI_MODEL_NAMES = [
    # High limit models (1,500 RPD, 15 RPM) - BEST for free tier
    'gemini-1.5-flash-latest',  # Free tier - 1,500 RPD, 15 RPM - RECOMMENDED
    'gemini-1.5-flash',         # Alternative naming - 1,500 RPD, 15 RPM
    'gemini-1.0-pro-latest',    # Free tier - 1,500 RPD, 15 RPM
    'gemini-1.0-pro',           # Alternative naming - 1,500 RPD, 15 RPM
    
    # Medium limit models (50 RPD, 2 RPM) - Lower but still free
    'gemini-1.5-pro-latest',    # Free tier - 50 RPD, 2 RPM - more capable but slower
    'gemini-1.5-pro',           # Alternative naming - 50 RPD, 2 RPM
    
    # Alternative flash models (20 RPD, 5-10 RPM) - Different quota pools
    'gemini-2.5-flash-lite',    # Free tier - 20 RPD, 10 RPM - DIFFERENT quota from gemini-2.5-flash
    'gemini-3-flash',           # Free tier - 20 RPD, 5 RPM - DIFFERENT quota pool
    
    # Low limit models (20 RPD, 5 RPM) - Last resort only (same quota as gemini-2.5-flash)
    'gemini-2.5-flash',         # Free tier - 20 RPD, 5 RPM - very low limits (AVOID if quota exceeded)
    'gemini-2.5-flash-latest', # Alternative naming - 20 RPD, 5 RPM (AVOID if quota exceeded)
]

if GEMINI_AVAILABLE and GEMINI_API_KEY and ENABLE_AI_VALIDATION:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # First, try to get list of available models from API
        available_models = []
        try:
            models = genai.list_models()
            all_models = [m.name.split('/')[-1] for m in models if 'generateContent' in m.supported_generation_methods]
            # FILTER OUT paid models (gemini-2.5-pro, gemini-2.0-pro, ultra, etc.) - only keep FREE TIER models
            paid_model_keywords = ['2.5-pro', '2.0-pro', 'ultra', '2.5-pro-exp']  # Models that require billing
            available_models = [m for m in all_models if not any(paid in m.lower() for paid in paid_model_keywords)]
            logger.info(f"Found {len(available_models)} FREE TIER Gemini models (excluded {len(all_models) - len(available_models)} paid models): {', '.join(available_models[:5])}...")
        except Exception as e:
            logger.debug(f"Could not list available models: {e}. Will try common model names.")
        
        # Use user-specified model first, then available models, then fallback list
        user_model = os.getenv('GEMINI_MODEL', None)
        if user_model:
            model_names = [user_model]
        elif available_models:
            # EXCLUDE paid models (gemini-2.5-pro, etc.) - only use FREE TIER models
            # Free tier models: gemini-1.5-flash, gemini-1.5-pro, gemini-1.0-pro, gemini-2.5-flash (low limits)
            # EXCLUDE: gemini-2.5-pro (requires billing, not free tier)
            paid_models = ['2.5-pro', '2.0-pro', 'ultra']  # Models that require billing
            free_tier_models = [m for m in available_models if not any(paid in m.lower() for paid in paid_models)]
            
            if not free_tier_models:
                logger.warning("⚠️ No free tier models found in available models. Using fallback list.")
                model_names = GEMINI_MODEL_NAMES
            else:
                # Prefer free tier models with HIGHER limits (avoid gemini-2.5-flash which has only 20 RPD)
                # Prioritize models with 1,500 RPD over those with 20-50 RPD
                # IMPORTANT: gemini-2.5-flash-lite and gemini-3-flash have SEPARATE quotas from gemini-2.5-flash
                high_limit_models = [m for m in free_tier_models if ('1.5-flash' in m.lower() or '1.0-pro' in m.lower()) and '2.5' not in m.lower() and '3' not in m.lower()]
                medium_limit_models = [m for m in free_tier_models if '1.5-pro' in m.lower() and m not in high_limit_models]
                # Separate quota models (different from gemini-2.5-flash quota pool) - these have their own 20 RPD limit
                separate_quota_models = [m for m in free_tier_models if ('2.5-flash-lite' in m.lower() or '3-flash' in m.lower())]
                low_limit_models = [m for m in free_tier_models if '2.5-flash' in m.lower() and 'lite' not in m.lower()]  # Last resort (shared quota)
                other_free_models = [m for m in free_tier_models if m not in high_limit_models and m not in medium_limit_models and m not in low_limit_models and m not in separate_quota_models]
                # Prioritize: High limit (1,500 RPD) > Medium (50 RPD) > Separate quota models > Others > Low limit (20 RPD, shared quota)
                model_names = (high_limit_models[:3] + medium_limit_models[:2] + separate_quota_models[:2] + other_free_models[:2] + low_limit_models[:1])[:10]
                logger.info(f"✅ Selected {len(model_names)} free tier models (excluded paid models like gemini-2.5-pro)")
                if separate_quota_models:
                    logger.info(f"   📊 Models with SEPARATE quotas (can use even if gemini-2.5-flash quota exceeded): {', '.join(separate_quota_models[:2])}")
        else:
            model_names = GEMINI_MODEL_NAMES
        
        # Try to initialize with available models
        for model_name in model_names:
            if not model_name:
                continue
            try:
                gemini_client = genai.GenerativeModel(model_name)
                gemini_model_name = model_name
                logger.info(f"✅ Gemini API initialized successfully (using {model_name})")
                break
            except Exception as e:
                logger.debug(f"Failed to initialize {model_name}: {e}")
                continue
        
        # If still no client, try fallback models
        if not gemini_client:
            logger.info("Trying fallback model names...")
            for model_name in GEMINI_MODEL_NAMES:
                if model_name in model_names:  # Skip if already tried
                    continue
                try:
                    gemini_client = genai.GenerativeModel(model_name)
                    gemini_model_name = model_name
                    logger.info(f"✅ Gemini API initialized successfully (using fallback {model_name})")
                    break
                except Exception as e:
                    logger.debug(f"Failed to initialize {model_name}: {e}")
                    continue
        
        if not gemini_client:
            logger.warning(f"Failed to initialize any Gemini model. Tried: {', '.join([m for m in model_names if m])}")
            logger.warning("Tip: Check your API key and available models at https://aistudio.google.com/app/apikey")
    except Exception as e:
        logger.warning(f"Failed to configure Gemini API: {e}. AI validation will be disabled.")
        gemini_client = None
elif ENABLE_AI_VALIDATION and not GEMINI_API_KEY:
    logger.warning("ENABLE_AI_VALIDATION is true but GEMINI_API_KEY is not set. AI validation will be disabled.")
elif ENABLE_AI_VALIDATION and not GEMINI_AVAILABLE:
    logger.warning("AI validation is enabled but google-generativeai package is not installed. Install it with: pip install google-generativeai")

def send_slack_alert(error_type, message, details=None, symbol=None, severity='ERROR'):
    """
    Send a beautiful error notification to Slack webhook
    
    Args:
        error_type: Type of error (e.g., 'Binance API Error', 'Order Creation Failed')
        message: Main error message
        details: Additional details dict (optional)
        symbol: Trading symbol if applicable (optional)
        severity: ERROR, WARNING, or CRITICAL
    """
    if not SLACK_WEBHOOK_URL:
        return  # Skip if webhook URL not configured
    
    try:
        # Determine emoji based on severity
        emoji_map = {
            'ERROR': '🚨',
            'WARNING': '⚠️',
            'CRITICAL': '🔥'
        }
        emoji = emoji_map.get(severity, '🚨')
        
        # Build the message
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        environment = 'TESTNET' if BINANCE_TESTNET else 'PRODUCTION'
        
        # Format the message with beautiful structure
        slack_message = f"""{emoji} *{severity}: {error_type}*

*App:* Binance Trading Bot
*Environment:* {environment}
*Module:* Webhook Service
*Time:* {timestamp}"""
        
        if symbol:
            slack_message += f"\n*Symbol:* {symbol}"
        
        slack_message += f"\n*Message:* {message}"
        
        # Add additional details if provided
        if details:
            slack_message += "\n*Details:*"
            for key, value in details.items():
                if value is not None:
                    slack_message += f"\n  • *{key}:* {value}"
        
        # Send to Slack (non-blocking in a thread)
        def send_async():
            try:
                payload = {'text': slack_message}
                response = requests.post(
                    SLACK_WEBHOOK_URL,
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=5
                )
                response.raise_for_status()
            except Exception as e:
                # Don't log Slack errors to avoid infinite loops
                logger.debug(f"Failed to send Slack notification: {e}")
        
        # Send in background thread to avoid blocking
        thread = threading.Thread(target=send_async, daemon=True)
        thread.start()
        
    except Exception as e:
        # Silently fail - don't break the service if Slack is down
        logger.debug(f"Error preparing Slack notification: {e}")

def send_signal_rejection_notification(symbol, signal_side, timeframe, entry_price, 
                                       rejection_reason, confidence_score=None, risk_level=None, 
                                       validation_result=None):
    """
    Send a rejection notification to Slack exception channel when signal is rejected
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        signal_side: 'LONG' or 'SHORT'
        timeframe: Trading timeframe (e.g., '1H', '4H')
        entry_price: Entry price from signal
        rejection_reason: Reason for rejection
        confidence_score: AI confidence score if available
        risk_level: Risk level if available
        validation_result: Full validation result dict (optional)
    """
    if not SLACK_WEBHOOK_URL:
        return  # Skip if webhook URL not configured
    
    try:
        # Determine side emoji
        side_emoji = '📈' if signal_side == 'LONG' else '📉'
        
        # Build the message
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        environment = 'TESTNET' if BINANCE_TESTNET else 'PRODUCTION'
        
        # Format symbol consistently (remove .P suffix if present, uppercase)
        formatted_symbol = symbol.replace('.P', '').upper()
        # Format timeframe consistently (uppercase, ensure proper format)
        formatted_timeframe = timeframe.upper() if timeframe else 'N/A'
        
        # Format entry price consistently
        entry_str = f'${entry_price:,.8f}' if entry_price else 'N/A'
        
        # Build rejection message
        slack_message = f"""🚫 *SIGNAL REJECTED - AI VALIDATION FAILED*

*Symbol:* `{formatted_symbol}`
*Timeframe:* `{formatted_timeframe}`
*Side:* {side_emoji} {signal_side}
*Environment:* {environment}
*Time:* {timestamp}

*Signal Details:*
  • Entry Price: {entry_str}"""
        
        if confidence_score is not None:
            slack_message += f"\n  • Confidence Score: {confidence_score:.1f}% (Threshold: {AI_VALIDATION_MIN_CONFIDENCE}%)"
        
        if risk_level:
            risk_emoji_map = {
                'LOW': '🟢',
                'MEDIUM': '🟡',
                'HIGH': '🔴'
            }
            risk_emoji = risk_emoji_map.get(risk_level, '⚪')
            slack_message += f"\n  • Risk Level: {risk_emoji} {risk_level}"
        
        slack_message += f"\n\n*Rejection Reason:*\n{rejection_reason}"
        
        # Add AI reasoning if available (truncated)
        if validation_result and validation_result.get('reasoning'):
            reasoning = validation_result.get('reasoning', '')
            # Convert to single line
            reasoning = ' '.join(reasoning.split())
            # Truncate if too long
            if len(reasoning) > 500:
                reasoning = reasoning[:497] + "..."
            slack_message += f"\n\n*AI Analysis:* {reasoning}"
        
        # Send to Slack (non-blocking in a thread)
        def send_async():
            try:
                payload = {'text': slack_message}
                response = requests.post(
                    SLACK_WEBHOOK_URL,
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=5
                )
                response.raise_for_status()
                logger.info(f"✅ Rejection notification sent to Slack for {symbol}")
            except Exception as e:
                logger.debug(f"Failed to send Slack rejection notification: {e}")
        
        # Send in background thread to avoid blocking
        thread = threading.Thread(target=send_async, daemon=True)
        thread.start()
        
    except Exception as e:
        logger.debug(f"Error preparing Slack rejection notification: {e}")


def send_signal_notification(symbol, signal_side, timeframe, confidence_score, risk_level, 
                             entry1_price, entry2_price, stop_loss, take_profit, 
                             tp1_price=None, use_single_tp=False, validation_result=None):
    """
    Send a beautiful signal notification to Slack signal channel after order is opened
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        signal_side: 'LONG' or 'SHORT'
        timeframe: Trading timeframe (e.g., '1H', '4H')
        confidence_score: AI confidence score (0-100)
        risk_level: Risk level from AI validation ('LOW', 'MEDIUM', 'HIGH')
        entry1_price: Primary entry price
        entry2_price: DCA entry price (optional)
        stop_loss: Stop loss price
        take_profit: Take profit price
        validation_result: Full validation result dict (optional)
    """
    if not SLACK_SIGNAL_WEBHOOK_URL:
        return  # Skip if webhook URL not configured
    
    try:
        # Determine signal strength emoji and text based on confidence score
        if confidence_score >= 80:
            strength_emoji = '🔥'
            strength_text = 'VERY STRONG'
        elif confidence_score >= 65:
            strength_emoji = '✅'
            strength_text = 'STRONG'
        elif confidence_score >= 50:
            strength_emoji = '⚡'
            strength_text = 'MODERATE'
        else:
            strength_emoji = '⚠️'
            strength_text = 'WEAK'
        
        # Determine side emoji
        side_emoji = '📈' if signal_side == 'LONG' else '📉'
        
        # Determine risk level emoji
        risk_emoji_map = {
            'LOW': '🟢',
            'MEDIUM': '🟡',
            'HIGH': '🔴'
        }
        risk_emoji = risk_emoji_map.get(risk_level, '⚪')
        
        # Build the message
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        environment = 'TESTNET' if BINANCE_TESTNET else 'PRODUCTION'
        
        # Format symbol consistently (remove .P suffix if present, uppercase)
        formatted_symbol = symbol.replace('.P', '').upper()
        # Format timeframe consistently (uppercase, ensure proper format)
        formatted_timeframe = timeframe.upper() if timeframe else 'N/A'
        
        # Format prices consistently (all with same format: $X,XXX.XXXXXXXX)
        entry1_str = f'${entry1_price:,.8f}' if entry1_price else 'N/A'
        entry2_str = f'${entry2_price:,.8f}' if entry2_price else 'N/A'
        stop_loss_str = f'${stop_loss:,.8f}' if stop_loss else 'N/A'
        tp1_str = f'${tp1_price:,.8f}' if tp1_price else 'N/A'
        tp2_str = f'${take_profit:,.8f}' if take_profit else 'N/A'
        
        # Format the message with beautiful structure (consistent formatting)
        slack_message = f"""{side_emoji} *NEW {signal_side} SIGNAL - ORDER OPENED*

*Symbol:* `{formatted_symbol}`
*Timeframe:* `{formatted_timeframe}`
*Environment:* {environment}
*Time:* {timestamp}

*Signal Strength:* {strength_emoji} {strength_text} ({confidence_score:.1f}%)
*Risk Level:* {risk_emoji} {risk_level}

*Entry Prices:*
  • Entry 1: {entry1_str}
  • Entry 2: {entry2_str}

*Risk Management:*
  • Stop Loss: {stop_loss_str}"""
        
        # Add TP information based on strategy
        if use_single_tp:
            slack_message += f"""
  • Take Profit (100%): {tp2_str} - Main TP (High Confidence: {confidence_score:.1f}%)"""
        else:
            slack_message += f"""
  • Take Profit 1 (70%): {tp1_str} - {TP1_PERCENT}% profit
  • Take Profit 2 (30%): {tp2_str} - Original/AI TP (Lower Confidence: {confidence_score:.1f}%)"""
        
        slack_message += "\n\n"
        
        # Add AI reasoning if available (single line only)
        if validation_result and validation_result.get('reasoning'):
            reasoning = validation_result.get('reasoning', '')
            # Convert to single line by replacing newlines with spaces
            reasoning = ' '.join(reasoning.split())
            # Show more complete reasoning (increased from 200 to 600 chars for better context)
            if len(reasoning) > 600:
                reasoning = reasoning[:597] + "..."
            slack_message += f"*AI Analysis:* {reasoning}\n"
        
        # Send to Slack (non-blocking in a thread)
        def send_async():
            try:
                payload = {'text': slack_message}
                response = requests.post(
                    SLACK_SIGNAL_WEBHOOK_URL,
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=5
                )
                response.raise_for_status()
                logger.info(f"✅ Signal notification sent to Slack for {symbol}")
            except Exception as e:
                # Don't log Slack errors to avoid infinite loops
                logger.debug(f"Failed to send Slack signal notification: {e}")
        
        # Send in background thread to avoid blocking
        thread = threading.Thread(target=send_async, daemon=True)
        thread.start()
        
    except Exception as e:
        # Silently fail - don't break the service if Slack is down
        logger.debug(f"Error preparing Slack signal notification: {e}")

# Initialize Binance client
try:
    if BINANCE_TESTNET:
        client = Client(
            api_key=BINANCE_API_KEY,
            api_secret=BINANCE_API_SECRET,
            testnet=True
        )
        logger.info("Connected to Binance TESTNET")
    else:
        client = Client(
            api_key=BINANCE_API_KEY,
            api_secret=BINANCE_API_SECRET
        )
        logger.info("Connected to Binance LIVE")
except Exception as e:
    logger.error(f"Failed to initialize Binance client: {e}")
    send_slack_alert(
        error_type="Binance Client Initialization Failed",
        message=str(e),
        details={'API_Key_Configured': bool(BINANCE_API_KEY), 'Testnet': BINANCE_TESTNET},
        severity='CRITICAL'
    )
    client = None

# Order tracking to prevent duplicates
recent_orders = {}
ORDER_COOLDOWN = 60  # seconds

# Trading configuration
# Entry size per trade (can be overridden via ENTRY_SIZE_USD environment variable)
# TESTING MODE: Set to $5 for small timeframe testing (10-15 min charts)
# PRODUCTION: Change back to $10 after testing
ENTRY_SIZE_USD = float(os.getenv('ENTRY_SIZE_USD', '10.0'))  # Default: $10 per entry
# Leverage (can be overridden via LEVERAGE environment variable)
# Default: 20X leverage
LEVERAGE = int(os.getenv('LEVERAGE', '20'))  # Default: 20X leverage
TOTAL_ENTRIES = 2  # Primary entry + DCA entry

# TP1 and TP2 Configuration
TP1_PERCENT = float(os.getenv('TP1_PERCENT', '4.0'))  # Default: 4.0% profit for TP1 (calculated from Entry 1 only)
TP1_SPLIT = float(os.getenv('TP1_SPLIT', '70.0'))  # Default: 70% of position closes at TP1
TP2_SPLIT = float(os.getenv('TP2_SPLIT', '30.0'))  # Default: 30% of position closes at TP2 (remaining)
TP_HIGH_CONFIDENCE_THRESHOLD = float(os.getenv('TP_HIGH_CONFIDENCE_THRESHOLD', '90.0'))  # If confidence >= 90%, use single TP

# Risk Management Configuration
ENABLE_RISK_VALIDATION = os.getenv('ENABLE_RISK_VALIDATION', 'true').lower() == 'true'  # Enable/disable risk validation
MAX_RISK_PERCENT = float(os.getenv('MAX_RISK_PERCENT', '20.0'))  # Default: 20% of account (includes pending orders) - High threshold to avoid rejecting good signals
ENABLE_TRAILING_STOP_LOSS = os.getenv('ENABLE_TRAILING_STOP_LOSS', 'true').lower() == 'true'  # Enable trailing SL after TP1
TRAILING_SL_BREAKEVEN_PERCENT = float(os.getenv('TRAILING_SL_BREAKEVEN_PERCENT', '0.5'))  # Move SL to 0.5% profit after TP1

# Account balance cache (to reduce API calls)
account_balance_cache = {'balance': None, 'timestamp': 0}
BALANCE_CACHE_TTL = 60  # Cache balance for 1 minute

# Track active trades per symbol
active_trades = {}  # {symbol: {'primary_filled': bool, 'dca_filled': bool, 'position_open': bool, 
                    #           'primary_order_id': int, 'dca_order_id': int, 'tp1_order_id': int, 'tp2_order_id': int, 'sl_order_id': int,
                    #           'tp1_price': float, 'tp2_price': float, 'tp1_quantity': float, 'tp2_quantity': float,
                    #           'exit_processed': bool, 'last_exit_time': float}}

# Track recent EXIT events to prevent duplicate processing
recent_exits = {}  # {symbol: timestamp}
EXIT_COOLDOWN = 30

# Helper function to create TP orders for a symbol (can be called from anywhere)
def delayed_tp_creation(symbol, delay_seconds=5):
    """Helper function to create TP orders after a delay (allows Binance to update position)"""
    def _create():
        time.sleep(delay_seconds)
        if symbol in active_trades:
            logger.info(f"🔄 Delayed TP check for {symbol} (after {delay_seconds}s delay)")
            create_tp1_tp2_if_needed(symbol, active_trades[symbol])
    
    thread = threading.Thread(target=_create, daemon=True)
    thread.start()

def create_tp_if_needed(symbol, trade_info):
    """Create TP order if position exists and TP is stored but not created yet
    NOTE: This function does NOT call AI validation - it only creates TP orders for existing positions"""
    if 'tp_price' not in trade_info:
        return False  # No TP to create
    
    # If TP order ID exists, check if it's still valid
    if 'tp_order_id' in trade_info:
        try:
            # Verify TP order still exists
            open_orders = client.futures_get_open_orders(symbol=symbol)
            existing_tp = [o for o in open_orders if o.get('orderId') == trade_info['tp_order_id']]
            if existing_tp:
                return True  # TP order exists and is valid
            else:
                # TP order was filled or canceled, remove the ID to allow recreation
                logger.info(f"TP order {trade_info['tp_order_id']} no longer exists for {symbol}, will recreate if needed")
                del trade_info['tp_order_id']
        except Exception as e:
            logger.warning(f"Error checking TP order status for {symbol}: {e}")
    
    try:
        # Check if position exists
        positions = client.futures_position_information(symbol=symbol)
        position_to_use = None
        
        for position in positions:
            position_amt = float(position.get('positionAmt', 0))
            if abs(position_amt) > 0:
                position_to_use = position
                break
        
        if not position_to_use:
            return False  # No position exists yet
        
        # Position exists, check if TP already exists
        has_orders, open_orders = check_existing_orders(symbol)
        existing_tp = [o for o in open_orders if o.get('type') == 'TAKE_PROFIT_MARKET']
        
        if existing_tp:
            # TP already exists, store the order ID and clean up stored price
            trade_info['tp_order_id'] = existing_tp[0].get('orderId')
            logger.info(f"✅ TP order already exists for {symbol}: {trade_info['tp_order_id']}")
            if 'tp_price' in trade_info:
                del trade_info['tp_price']
            return True
        
        # Create TP order
        tp_price = trade_info['tp_price']
        tp_side = trade_info.get('tp_side', 'SELL')
        position_amt = float(position_to_use.get('positionAmt', 0))
        
        # Get symbol info
        exchange_info = client.futures_exchange_info()
        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
        if not symbol_info:
            logger.error(f"Symbol info not found for {symbol}")
            send_slack_alert(
                error_type="Symbol Info Not Found",
                message=f"Symbol {symbol} not found in Binance exchange info",
                symbol=symbol,
                severity='ERROR'
            )
            return False
            
        price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
        tick_size = float(price_filter['tickSize']) if price_filter else 0.01
        tp_price = format_price_precision(tp_price, tick_size)
        
        # Detect position mode
        try:
            is_hedge_mode = get_position_mode(symbol)
        except:
            is_hedge_mode = False
        
        position_side = position_to_use.get('positionSide', 'BOTH')
        
        # Use stored TP quantity (total of primary + DCA) or position amount as fallback
        tp_quantity = trade_info.get('tp_quantity', abs(position_amt))
        
        # Format quantity precision
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            step_size = float(lot_size_filter['stepSize'])
            tp_quantity = format_quantity_precision(tp_quantity, step_size)
        
        working_type = trade_info.get('tp_working_type', 'MARK_PRICE')
        
        logger.info(f"🔄 Creating TP order for {symbol}: price={tp_price}, qty={tp_quantity}, side={tp_side}, workingType={working_type}, positionSide={position_side}, hedgeMode={is_hedge_mode}")
        
        # Try using closePosition first (recommended by Binance for conditional orders)
        # If that fails, fall back to using quantity
        tp_params_close = {
            'symbol': symbol,
            'side': tp_side,
            'type': 'TAKE_PROFIT_MARKET',
            'timeInForce': 'GTC',
            'closePosition': True,  # Close entire position (recommended for conditional orders)
            'stopPrice': tp_price,
            'workingType': working_type,  # Use mark price for trigger (like Binance UI)
        }
        
        tp_params_quantity = {
            'symbol': symbol,
            'side': tp_side,
            'type': 'TAKE_PROFIT_MARKET',
            'timeInForce': 'GTC',
            'quantity': tp_quantity,
            'stopPrice': tp_price,
            'workingType': working_type,  # Use mark price for trigger (like Binance UI)
        }
        
        # Add positionSide only if in hedge mode
        if is_hedge_mode and position_side != 'BOTH':
            tp_params_close['positionSide'] = position_side
            tp_params_quantity['positionSide'] = position_side
        
        # Strategy: Try multiple approaches in order of preference
        # 1. closePosition=True (recommended by Binance for conditional orders)
        # 2. quantity with reduceOnly=True
        # 3. quantity without reduceOnly
        
        # Try 1: closePosition=True (no reduceOnly needed when using closePosition)
        try:
            tp_order = client.futures_create_order(**tp_params_close)
            trade_info['tp_order_id'] = tp_order.get('orderId')
            logger.info(f"✅ TP order created successfully (using closePosition): Order ID {tp_order.get('orderId')} @ {tp_price} for {symbol}")
            
            # Remove stored TP details since it's now created (but keep tp_order_id)
            if 'tp_price' in trade_info:
                del trade_info['tp_price']
            if 'tp_quantity' in trade_info:
                del trade_info['tp_quantity']
            if 'tp_working_type' in trade_info:
                del trade_info['tp_working_type']
            return True
        except BinanceAPIException as e:
            # If closePosition fails, try with quantity
            if e.code == -4120 or 'order type not supported' in str(e).lower() or 'closePosition' in str(e).lower():
                logger.info(f"closePosition approach failed for {symbol}, trying with quantity: {e.message}")
                # Try 2: quantity with reduceOnly=True
                try:
                    tp_params_qty_reduce = tp_params_quantity.copy()
                    tp_params_qty_reduce['reduceOnly'] = True
                    tp_order = client.futures_create_order(**tp_params_qty_reduce)
                    trade_info['tp_order_id'] = tp_order.get('orderId')
                    logger.info(f"✅ TP order created successfully (using quantity with reduceOnly): Order ID {tp_order.get('orderId')} @ {tp_price} for {symbol} (qty: {tp_quantity})")
                    
                    # Remove stored TP details since it's now created (but keep tp_order_id)
                    if 'tp_price' in trade_info:
                        del trade_info['tp_price']
                    if 'tp_quantity' in trade_info:
                        del trade_info['tp_quantity']
                    if 'tp_working_type' in trade_info:
                        del trade_info['tp_working_type']
                    return True
                except BinanceAPIException as e2:
                    # Try 3: quantity without reduceOnly
                    if e2.code == -1106 or 'reduceonly' in str(e2).lower():
                        logger.info(f"reduceOnly not accepted, trying without it: {e2.message}")
                        try:
                            tp_order = client.futures_create_order(**tp_params_quantity)
                            trade_info['tp_order_id'] = tp_order.get('orderId')
                            logger.info(f"✅ TP order created successfully (using quantity without reduceOnly): Order ID {tp_order.get('orderId')} @ {tp_price} for {symbol} (qty: {tp_quantity})")
                            
                            # Remove stored TP details since it's now created (but keep tp_order_id)
                            if 'tp_price' in trade_info:
                                del trade_info['tp_price']
                            if 'tp_quantity' in trade_info:
                                del trade_info['tp_quantity']
                            if 'tp_working_type' in trade_info:
                                del trade_info['tp_working_type']
                            return True
                        except BinanceAPIException as e3:
                            # All approaches failed
                            if e3.code == -4120 or 'order type not supported' in str(e3).lower():
                                logger.warning(f"⚠️ TAKE_PROFIT_MARKET orders not supported for {symbol} (Code: {e3.code}). All approaches failed. Cleaning up stored TP details.")
                                send_slack_alert(
                                    error_type="Take Profit Order Type Not Supported",
                                    message=f"TAKE_PROFIT_MARKET orders are not supported for {symbol} after trying all approaches. You may need to set TP manually in Binance UI.",
                                    details={'Error_Code': e3.code, 'TP_Price': tp_price, 'TP_Quantity': tp_quantity, 'Symbol': symbol, 'Attempts': 'closePosition, quantity+reduceOnly, quantity'},
                                    symbol=symbol,
                                    severity='WARNING'
                                )
                                # Clean up stored TP details to prevent repeated retries
                                if 'tp_price' in trade_info:
                                    del trade_info['tp_price']
                                if 'tp_quantity' in trade_info:
                                    del trade_info['tp_quantity']
                                if 'tp_working_type' in trade_info:
                                    del trade_info['tp_working_type']
                                return False
                            else:
                                raise e3
                    elif e2.code == -4120 or 'order type not supported' in str(e2).lower():
                        # Try without reduceOnly
                        try:
                            tp_order = client.futures_create_order(**tp_params_quantity)
                            trade_info['tp_order_id'] = tp_order.get('orderId')
                            logger.info(f"✅ TP order created successfully (using quantity without reduceOnly): Order ID {tp_order.get('orderId')} @ {tp_price} for {symbol} (qty: {tp_quantity})")
                            
                            # Remove stored TP details since it's now created (but keep tp_order_id)
                            if 'tp_price' in trade_info:
                                del trade_info['tp_price']
                            if 'tp_quantity' in trade_info:
                                del trade_info['tp_quantity']
                            if 'tp_working_type' in trade_info:
                                del trade_info['tp_working_type']
                            return True
                        except BinanceAPIException as e3:
                            if e3.code == -4120 or 'order type not supported' in str(e3).lower():
                                logger.warning(f"⚠️ TAKE_PROFIT_MARKET orders not supported for {symbol} (Code: {e3.code}). All approaches failed. Cleaning up stored TP details.")
                                send_slack_alert(
                                    error_type="Take Profit Order Type Not Supported",
                                    message=f"TAKE_PROFIT_MARKET orders are not supported for {symbol} after trying all approaches. You may need to set TP manually in Binance UI.",
                                    details={'Error_Code': e3.code, 'TP_Price': tp_price, 'TP_Quantity': tp_quantity, 'Symbol': symbol, 'Attempts': 'closePosition, quantity+reduceOnly, quantity'},
                                    symbol=symbol,
                                    severity='WARNING'
                                )
                                # Clean up stored TP details to prevent repeated retries
                                if 'tp_price' in trade_info:
                                    del trade_info['tp_price']
                                if 'tp_quantity' in trade_info:
                                    del trade_info['tp_quantity']
                                if 'tp_working_type' in trade_info:
                                    del trade_info['tp_working_type']
                                return False
                            else:
                                raise e3
                    else:
                        raise e2
            else:
                # Other error from closePosition - raise to be caught by outer handler
                raise
        
    except BinanceAPIException as e:
        # Check if order type is not supported (error -4120)
        if e.code == -4120 or 'order type not supported' in str(e).lower():
            logger.warning(f"⚠️ TAKE_PROFIT_MARKET orders not supported for {symbol} (Code: {e.code}). This symbol may not support conditional orders. Cleaning up stored TP details.")
            send_slack_alert(
                error_type="Take Profit Order Type Not Supported",
                message=f"TAKE_PROFIT_MARKET orders are not supported for {symbol}. You may need to set TP manually in Binance UI.",
                details={'Error_Code': e.code, 'TP_Price': tp_price, 'TP_Quantity': tp_quantity, 'Symbol': symbol},
                symbol=symbol,
                severity='WARNING'
            )
            # Clean up stored TP details to prevent repeated retries
            if 'tp_price' in trade_info:
                del trade_info['tp_price']
            if 'tp_quantity' in trade_info:
                del trade_info['tp_quantity']
            if 'tp_working_type' in trade_info:
                del trade_info['tp_working_type']
            return False
        else:
            logger.error(f"❌ Binance API error creating TP for {symbol}: {e.message} (Code: {e.code})")
            send_slack_alert(
                error_type="Take Profit Order Creation Failed",
                message=f"{e.message} (Code: {e.code})",
                details={'Error_Code': e.code, 'TP_Price': tp_price, 'TP_Quantity': tp_quantity},
                symbol=symbol,
                severity='ERROR'
            )
            return False
    except Exception as e:
        logger.error(f"❌ Error creating TP for {symbol}: {e}", exc_info=True)
        send_slack_alert(
            error_type="Take Profit Order Creation Error",
            message=str(e),
            details={'TP_Price': tp_price, 'TP_Quantity': tp_quantity},
            symbol=symbol,
            severity='ERROR'
        )
        return False


def create_single_tp_order(symbol, tp_price, tp_quantity, tp_side, trade_info, tp_number=1):
    """Helper function to create a single TP order (used for both TP1 and TP2)
    
    Args:
        symbol: Trading symbol
        tp_price: Take profit price
        tp_quantity: Quantity to close at this TP
        tp_side: 'SELL' for LONG, 'BUY' for SHORT
        trade_info: Trade info dict
        tp_number: 1 for TP1, 2 for TP2 (for logging)
    
    Returns:
        order_id if successful, None if failed
    """
    try:
        # Get symbol info
        exchange_info = client.futures_exchange_info()
        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
        if not symbol_info:
            logger.error(f"Symbol info not found for {symbol}")
            return None
            
        price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
        tick_size = float(price_filter['tickSize']) if price_filter else 0.01
        tp_price = format_price_precision(tp_price, tick_size)
        
        # Format quantity precision
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            step_size = float(lot_size_filter['stepSize'])
            tp_quantity = format_quantity_precision(tp_quantity, step_size)
        
        # Get position info
        positions = client.futures_position_information(symbol=symbol)
        position_to_use = None
        for position in positions:
            position_amt = float(position.get('positionAmt', 0))
            if abs(position_amt) > 0:
                position_to_use = position
                break
        
        if not position_to_use:
            return None  # No position exists
        
        # Detect position mode
        try:
            is_hedge_mode = get_position_mode(symbol)
        except:
            is_hedge_mode = False
        
        position_side = position_to_use.get('positionSide', 'BOTH')
        working_type = trade_info.get('tp_working_type', 'MARK_PRICE')
        
        logger.info(f"🔄 Creating TP{tp_number} order for {symbol}: price={tp_price}, qty={tp_quantity}, side={tp_side}, workingType={working_type}")
        
        # Try using quantity with reduceOnly=True (required for partial closes)
        tp_params = {
            'symbol': symbol,
            'side': tp_side,
            'type': 'TAKE_PROFIT_MARKET',
            'timeInForce': 'GTC',
            'quantity': tp_quantity,
            'stopPrice': tp_price,
            'workingType': working_type,
            'reduceOnly': True,  # Required for partial position closes
        }
        
        # Add positionSide only if in hedge mode
        if is_hedge_mode and position_side != 'BOTH':
            tp_params['positionSide'] = position_side
        
        # Try to create the order
        try:
            tp_order = client.futures_create_order(**tp_params)
            order_id = tp_order.get('orderId')
            logger.info(f"✅ TP{tp_number} order created successfully: Order ID {order_id} @ {tp_price} for {symbol} (qty: {tp_quantity})")
            return order_id
        except BinanceAPIException as e:
            if e.code == -1106 or 'reduceonly' in str(e).lower():
                # Try without reduceOnly
                try:
                    tp_params_no_reduce = tp_params.copy()
                    del tp_params_no_reduce['reduceOnly']
                    tp_order = client.futures_create_order(**tp_params_no_reduce)
                    order_id = tp_order.get('orderId')
                    logger.info(f"✅ TP{tp_number} order created successfully (without reduceOnly): Order ID {order_id} @ {tp_price} for {symbol} (qty: {tp_quantity})")
                    return order_id
                except Exception as e2:
                    logger.error(f"❌ Failed to create TP{tp_number} order for {symbol}: {e2}")
                    return None
            else:
                logger.error(f"❌ Failed to create TP{tp_number} order for {symbol}: {e.message} (Code: {e.code})")
                return None
    except Exception as e:
        logger.error(f"❌ Error creating TP{tp_number} order for {symbol}: {e}", exc_info=True)
        return None


def create_tp1_tp2_if_needed(symbol, trade_info):
    """Create TP orders if position exists and TPs are stored but not created yet
    High Confidence (>=90%): Single TP (100% of position at main TP)
    Lower Confidence (<90%): TP1 + TP2 (70% at TP1, 30% at TP2)
    """
    # Check if using single TP mode
    use_single_tp = trade_info.get('use_single_tp', False)
    
    if use_single_tp:
        # Single TP mode: Only TP2 exists (main TP)
        if 'tp2_price' not in trade_info or not trade_info.get('tp2_price'):
            return False  # TP not configured
    else:
        # TP1 + TP2 mode: Both TPs exist
        if 'tp1_price' not in trade_info or 'tp2_price' not in trade_info:
            return False  # TPs not configured
    
    # Check existing TP orders
    tp1_exists = False
    tp2_exists = False
    
    if not use_single_tp:
        # TP1 + TP2 mode: Check both
        if 'tp1_order_id' in trade_info:
            try:
                open_orders = client.futures_get_open_orders(symbol=symbol)
                existing_tp1 = [o for o in open_orders if o.get('orderId') == trade_info['tp1_order_id']]
                if existing_tp1:
                    tp1_exists = True
                else:
                    logger.info(f"TP1 order {trade_info['tp1_order_id']} no longer exists for {symbol}")
                    del trade_info['tp1_order_id']
            except Exception as e:
                logger.warning(f"Error checking TP1 order status for {symbol}: {e}")
    
    # Check TP2 (always exists, either as main TP or as TP2)
    if 'tp2_order_id' in trade_info:
        try:
            open_orders = client.futures_get_open_orders(symbol=symbol)
            existing_tp2 = [o for o in open_orders if o.get('orderId') == trade_info['tp2_order_id']]
            if existing_tp2:
                tp2_exists = True
            else:
                logger.info(f"TP2 order {trade_info['tp2_order_id']} no longer exists for {symbol}")
                del trade_info['tp2_order_id']
        except Exception as e:
            logger.warning(f"Error checking TP2 order status for {symbol}: {e}")
    
    # If all required TPs exist, we're done
    if use_single_tp:
        if tp2_exists:
            return True
    else:
        if tp1_exists and tp2_exists:
            return True
    
    # Check if position exists
    try:
        positions = client.futures_position_information(symbol=symbol)
        position_to_use = None
        for position in positions:
            position_amt = float(position.get('positionAmt', 0))
            if abs(position_amt) > 0:
                position_to_use = position
                break
        
        if not position_to_use:
            return False  # No position exists yet
        
        # Get TP details
        tp_side = trade_info.get('tp_side', 'SELL')
        
        if use_single_tp:
            # Single TP mode: Only create TP2 (main TP)
            tp2_price = trade_info['tp2_price']
            tp2_quantity = trade_info.get('tp2_quantity', 0)
            
            if not tp2_exists and tp2_price and tp2_quantity > 0:
                tp2_order_id = create_single_tp_order(symbol, tp2_price, tp2_quantity, tp_side, trade_info, tp_number=2)
                if tp2_order_id:
                    trade_info['tp2_order_id'] = tp2_order_id
                    logger.info(f"✅ Main TP order created and stored for {symbol} (Single TP mode)")
                    # Clean up stored price
                    if 'tp2_price' in trade_info:
                        del trade_info['tp2_price']
                    if 'tp2_quantity' in trade_info:
                        del trade_info['tp2_quantity']
                    return True
        else:
            # TP1 + TP2 mode: Create both
            tp1_price = trade_info.get('tp1_price')
            tp2_price = trade_info['tp2_price']
            tp1_quantity = trade_info.get('tp1_quantity', 0)
            tp2_quantity = trade_info.get('tp2_quantity', 0)
            
            # Create TP1 if it doesn't exist
            if not tp1_exists and tp1_price and tp1_quantity > 0:
                tp1_order_id = create_single_tp_order(symbol, tp1_price, tp1_quantity, tp_side, trade_info, tp_number=1)
                if tp1_order_id:
                    trade_info['tp1_order_id'] = tp1_order_id
                    logger.info(f"✅ TP1 order created and stored for {symbol}")
                else:
                    logger.warning(f"⚠️ Failed to create TP1 order for {symbol}")
            
            # Create TP2 if it doesn't exist
            if not tp2_exists and tp2_price and tp2_quantity > 0:
                tp2_order_id = create_single_tp_order(symbol, tp2_price, tp2_quantity, tp_side, trade_info, tp_number=2)
                if tp2_order_id:
                    trade_info['tp2_order_id'] = tp2_order_id
                    logger.info(f"✅ TP2 order created and stored for {symbol}")
                else:
                    logger.warning(f"⚠️ Failed to create TP2 order for {symbol}")
            
            # If both orders created successfully, clean up stored prices (but keep order IDs)
            if 'tp1_order_id' in trade_info and 'tp2_order_id' in trade_info:
                if 'tp1_price' in trade_info:
                    del trade_info['tp1_price']
                if 'tp2_price' in trade_info:
                    del trade_info['tp2_price']
                if 'tp1_quantity' in trade_info:
                    del trade_info['tp1_quantity']
                if 'tp2_quantity' in trade_info:
                    del trade_info['tp2_quantity']
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"❌ Error creating TP1/TP2 orders for {symbol}: {e}", exc_info=True)
        return False


def update_trailing_stop_loss(symbol, trade_info, position):
    """Update stop loss to breakeven/profit after TP1 is filled (trailing stop loss)
    
    Args:
        symbol: Trading symbol
        trade_info: Trade info dict
        position: Position data from Binance
    
    Returns:
        bool: True if SL was updated, False otherwise
    """
    if not ENABLE_TRAILING_STOP_LOSS:
        return False
    
    # Only apply trailing SL if TP1 was filled (for TP1+TP2 mode)
    if trade_info.get('use_single_tp', False):
        return False  # Single TP mode doesn't need trailing SL
    
    # Check if TP1 order was filled
    tp1_order_id = trade_info.get('tp1_order_id')
    if not tp1_order_id:
        return False  # TP1 not created yet
    
    # Check if TP1 order still exists (if not, it was filled)
    try:
        open_orders = client.futures_get_open_orders(symbol=symbol)
        tp1_still_exists = any(o.get('orderId') == tp1_order_id for o in open_orders)
        
        if tp1_still_exists:
            return False  # TP1 not filled yet, no trailing SL needed
        
        # TP1 was filled - check if we already moved SL
        if trade_info.get('sl_moved_to_breakeven', False):
            return False  # Already moved SL, no need to update again
        
    except Exception as e:
        logger.warning(f"Error checking TP1 status for {symbol}: {e}")
        return False
    
    # TP1 was filled - calculate new stop loss (breakeven + small profit)
    try:
        # Get position details
        position_amt = float(position.get('positionAmt', 0))
        entry_price = float(position.get('entryPrice', 0))
        mark_price = float(position.get('markPrice', 0))
        
        if abs(position_amt) == 0 or entry_price <= 0:
            return False  # Invalid position data
        
        # Get original stop loss and entry prices
        original_sl = safe_float(trade_info.get('original_stop_loss'), default=0)
        original_entry1 = safe_float(trade_info.get('original_entry1'), default=entry_price)
        original_entry2 = safe_float(trade_info.get('original_entry2'), default=entry_price)
        
        # Calculate average entry (for breakeven calculation)
        if original_entry2 and original_entry2 > 0 and original_entry2 != original_entry1:
            avg_entry = (original_entry1 + original_entry2) / 2
        else:
            avg_entry = original_entry1
        
        # Determine position side
        is_long = position_amt > 0
        
        # Calculate new stop loss: breakeven + small profit (0.5% default)
        breakeven_profit_percent = TRAILING_SL_BREAKEVEN_PERCENT / 100.0
        
        if is_long:
            new_sl_price = avg_entry * (1 + breakeven_profit_percent)
            # Ensure new SL is above current mark price (safety check)
            if new_sl_price > mark_price:
                new_sl_price = mark_price * 0.999  # Slightly below current price
        else:  # SHORT
            new_sl_price = avg_entry * (1 - breakeven_profit_percent)
            # Ensure new SL is below current mark price (safety check)
            if new_sl_price < mark_price:
                new_sl_price = mark_price * 1.001  # Slightly above current price
        
        # Get symbol info for precision
        exchange_info = client.futures_exchange_info()
        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
        if not symbol_info:
            return False
        
        price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
        tick_size = float(price_filter['tickSize']) if price_filter else 0.01
        new_sl_price = format_price_precision(new_sl_price, tick_size)
        
        # Get current stop loss order
        current_sl_id = trade_info.get('sl_order_id')
        current_sl_price = safe_float(trade_info.get('current_sl_price'), default=0)
        
        # Only update if new SL is better than current SL
        if is_long:
            sl_needs_update = new_sl_price > current_sl_price
        else:  # SHORT
            sl_needs_update = new_sl_price < current_sl_price or current_sl_price == 0
        
        if not sl_needs_update:
            return False  # Current SL is already better
        
        # Cancel old stop loss if exists
        if current_sl_id:
            try:
                client.futures_cancel_order(symbol=symbol, orderId=current_sl_id)
                logger.info(f"🔄 Cancelled old stop loss order {current_sl_id} for {symbol}")
            except Exception as e:
                logger.warning(f"Error cancelling old SL for {symbol}: {e}")
        
        # Create new stop loss at breakeven + profit
        sl_side = 'SELL' if is_long else 'BUY'
        position_side = position.get('positionSide', 'BOTH')
        
        try:
            is_hedge_mode = get_position_mode(symbol)
        except:
            is_hedge_mode = False
        
        sl_params = {
            'symbol': symbol,
            'side': sl_side,
            'type': 'STOP_MARKET',
            'timeInForce': 'GTC',
            'stopPrice': new_sl_price,
            'closePosition': True,  # Close entire remaining position
            'workingType': 'MARK_PRICE',
        }
        
        if is_hedge_mode and position_side != 'BOTH':
            sl_params['positionSide'] = position_side
        
        new_sl_order = client.futures_create_order(**sl_params)
        new_sl_id = new_sl_order.get('orderId')
        
        # Update trade info
        trade_info['sl_order_id'] = new_sl_id
        trade_info['current_sl_price'] = new_sl_price
        trade_info['sl_moved_to_breakeven'] = True
        trade_info['tp1_filled'] = True  # Mark TP1 as filled
        
        logger.info(f"✅ Trailing stop loss updated for {symbol}: Moved SL to breakeven+profit @ ${new_sl_price:,.8f} "
                   f"(Original SL: ${original_sl:,.8f}, Entry: ${avg_entry:,.8f})")
        
        # Send Slack notification
        send_slack_alert(
            error_type="Trailing Stop Loss Updated",
            message=f"TP1 filled - Stop loss moved to breakeven+{TRAILING_SL_BREAKEVEN_PERCENT}% profit",
            details={
                'Original_SL': f"${original_sl:,.8f}",
                'New_SL': f"${new_sl_price:,.8f}",
                'Entry_Price': f"${avg_entry:,.8f}",
                'TP1_Filled': True
            },
            symbol=symbol,
            severity='INFO'
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating trailing stop loss for {symbol}: {e}", exc_info=True)
        return False


# Background thread to create TP orders when positions exist
# This mimics Binance UI behavior: TP is "set" when creating limit order, but placed when order fills
# Optimized to minimize API calls and stay within Binance rate limits
def create_missing_tp_orders():
    """Background function to automatically create TP orders for positions with stored TP details
    NOTE: This background thread does NOT call AI validation - it only creates TP orders for existing positions
    Optimized to reduce API calls:
    - Only checks symbols with stored TP details (skips symbols without TP config)
    - Uses longer intervals to stay well within Binance rate limits
    - Caches results to avoid redundant checks"""
    check_count = 0
    symbols_without_tp_logged = set()  # Track symbols we've already logged warnings for
    
    while True:
        try:
            # Optimized intervals to reduce API calls while still being effective:
            # - First 4 checks (4 minutes): Every 1 minute to catch newly filled orders quickly
            # - Next 10 checks (20 minutes): Every 2 minutes for active monitoring
            # - After that: Every 5 minutes for ongoing checks
            if check_count < 4:
                sleep_time = 60  # Check every 1 minute for first 4 minutes
            elif check_count < 14:
                sleep_time = 120  # Check every 2 minutes for next 20 minutes
            else:
                sleep_time = 300  # Then check every 5 minutes (reduces API calls significantly)
            
            time.sleep(sleep_time)
            check_count += 1
            
            try:
                # Only check symbols that have stored TP details (optimization: skip symbols without TP config)
                # Check for symbols with TP configured (either single TP or TP1+TP2)
                symbols_to_check = []
                for s, info in active_trades.items():
                    if info.get('use_single_tp', False):
                        # Single TP mode: only need tp2_price
                        if 'tp2_price' in info and info.get('tp2_price'):
                            symbols_to_check.append(s)
                    else:
                        # TP1+TP2 mode: need both
                        if 'tp1_price' in info and 'tp2_price' in info:
                            symbols_to_check.append(s)
                
                if not symbols_to_check:
                    # No symbols with stored TP details - skip API calls entirely
                    continue
                
                # Get ALL open positions from Binance (single API call)
                # This single call is used for BOTH TP creation AND trailing stop loss checks
                all_positions = client.futures_position_information()
                positions_dict = {p['symbol']: p for p in all_positions if abs(float(p.get('positionAmt', 0))) > 0}
                
                # Check trailing stop loss for ALL positions (uses same position data - no extra API call!)
                # This runs BEFORE TP creation check to use the same position data
                if ENABLE_TRAILING_STOP_LOSS and positions_dict:
                    for symbol, position in positions_dict.items():
                        if symbol in active_trades:
                            try:
                                trade_info = active_trades[symbol]
                                # Only check if we have TP1 configured (TP1+TP2 mode)
                                if not trade_info.get('use_single_tp', False):
                                    update_trailing_stop_loss(symbol, trade_info, position)
                            except Exception as e:
                                logger.debug(f"Error checking trailing SL for {symbol}: {e}")
                
                if not positions_dict:
                    # No open positions - but check if there are pending orders before cleaning up TP
                    # If orders are pending, keep TP details and wait for order to fill
                    for symbol in list(symbols_to_check):
                        use_single_tp = active_trades.get(symbol, {}).get('use_single_tp', False)
                        has_tp = False
                        if use_single_tp:
                            has_tp = 'tp2_price' in active_trades.get(symbol, {}) and active_trades[symbol].get('tp2_price')
                        else:
                            has_tp = ('tp1_price' in active_trades.get(symbol, {}) or 'tp2_price' in active_trades.get(symbol, {}))
                        
                        if symbol in active_trades and has_tp:
                            # Check if there are pending orders for this symbol
                            try:
                                has_orders, open_orders = check_existing_orders(symbol, log_result=False)
                                if has_orders:
                                    # There are pending orders - keep TP details and wait for order to fill
                                    logger.debug(f"⏳ Background thread: Keeping TP1/TP2 for {symbol} (pending orders, waiting for fill)")
                                    continue  # Don't clean up - order might fill soon
                                else:
                                    # No orders and no position - clean up TP details
                                    logger.info(f"🧹 Background thread: Cleaning up stored TP1/TP2 for {symbol} (no position and no pending orders)")
                                    if 'tp1_price' in active_trades[symbol]:
                                        del active_trades[symbol]['tp1_price']
                                    if 'tp2_price' in active_trades[symbol]:
                                        del active_trades[symbol]['tp2_price']
                                    if 'tp1_quantity' in active_trades[symbol]:
                                        del active_trades[symbol]['tp1_quantity']
                                    if 'tp2_quantity' in active_trades[symbol]:
                                        del active_trades[symbol]['tp2_quantity']
                                    if 'tp_working_type' in active_trades[symbol]:
                                        del active_trades[symbol]['tp_working_type']
                            except Exception as e:
                                logger.debug(f"Error checking orders for {symbol}: {e}")
                                # On error, keep TP details (safer to keep than delete)
                                continue
                    continue
                
                # Only log if we're actually checking symbols (reduce log spam)
                if len(symbols_to_check) > 0:
                    logger.debug(f"Background thread: Checking {len(symbols_to_check)} symbol(s) with stored TP details")
                
                # Only check symbols that have stored TP details AND have open positions
                symbols_checked = 0
                for symbol in symbols_to_check:
                    if symbol not in positions_dict:
                        # Symbol has stored TP but no position - check if orders are pending
                        # Keep TP details if orders exist (waiting for fill), only clean up if no orders
                        try:
                            has_orders, open_orders = check_existing_orders(symbol, log_result=False)
                            
                            # Also check if we have tracked order IDs for this symbol
                            has_tracked_orders = False
                            if symbol in active_trades:
                                # Check if we have primary or DCA order IDs tracked
                                if 'primary_order_id' in active_trades[symbol] or 'dca_order_id' in active_trades[symbol]:
                                    # Verify these specific orders still exist
                                    tracked_order_ids = []
                                    if 'primary_order_id' in active_trades[symbol]:
                                        tracked_order_ids.append(active_trades[symbol]['primary_order_id'])
                                    if 'dca_order_id' in active_trades[symbol]:
                                        tracked_order_ids.append(active_trades[symbol]['dca_order_id'])
                                    
                                    # Check if any tracked orders still exist
                                    if tracked_order_ids:
                                        existing_order_ids = [o.get('orderId') for o in open_orders]
                                        has_tracked_orders = any(oid in existing_order_ids for oid in tracked_order_ids)
                            
                            if has_orders or has_tracked_orders:
                                # There are pending orders - keep TP details and wait for order to fill
                                logger.debug(f"⏳ Background thread: Keeping TP for {symbol} (pending orders, waiting for fill)")
                                continue  # Don't clean up - order might fill soon
                            else:
                                # No orders and no position - clean up TP details
                                logger.info(f"🧹 Background thread: Cleaning up stored TP1/TP2 for {symbol} (no position and no pending orders)")
                                if symbol in active_trades:
                                    if 'tp1_price' in active_trades[symbol]:
                                        del active_trades[symbol]['tp1_price']
                                    if 'tp2_price' in active_trades[symbol]:
                                        del active_trades[symbol]['tp2_price']
                                    if 'tp1_quantity' in active_trades[symbol]:
                                        del active_trades[symbol]['tp1_quantity']
                                    if 'tp2_quantity' in active_trades[symbol]:
                                        del active_trades[symbol]['tp2_quantity']
                                    if 'tp_working_type' in active_trades[symbol]:
                                        del active_trades[symbol]['tp_working_type']
                        except Exception as e:
                            logger.debug(f"Error checking orders for {symbol}: {e}")
                            # On error, keep TP details (safer to keep than delete)
                            continue
                        continue
                    
                    symbols_checked += 1
                    position = positions_dict[symbol]
                    position_amt = float(position.get('positionAmt', 0))
                    
                    try:
                        # Check if TP orders already exist (don't log every check to reduce spam)
                        has_orders, open_orders = check_existing_orders(symbol, log_result=False)
                        existing_tps = [o for o in open_orders if o.get('type') == 'TAKE_PROFIT_MARKET']
                        
                        use_single_tp = active_trades.get(symbol, {}).get('use_single_tp', False)
                        
                        if use_single_tp:
                            # Single TP mode: only need 1 TP order
                            if len(existing_tps) >= 1:
                                if symbol in active_trades:
                                    active_trades[symbol]['tp2_order_id'] = existing_tps[0].get('orderId')
                                    # Clean up stored TP details
                                    if 'tp2_price' in active_trades[symbol]:
                                        del active_trades[symbol]['tp2_price']
                                    if 'tp2_quantity' in active_trades[symbol]:
                                        del active_trades[symbol]['tp2_quantity']
                                    if 'tp_working_type' in active_trades[symbol]:
                                        del active_trades[symbol]['tp_working_type']
                                continue  # TP exists, skip
                        else:
                            # TP1+TP2 mode: need 2 TP orders
                            if len(existing_tps) >= 2:
                                if symbol in active_trades:
                                    if len(existing_tps) >= 1:
                                        active_trades[symbol]['tp1_order_id'] = existing_tps[0].get('orderId')
                                    if len(existing_tps) >= 2:
                                        active_trades[symbol]['tp2_order_id'] = existing_tps[1].get('orderId')
                                    # Clean up stored TP details
                                    if 'tp1_price' in active_trades[symbol]:
                                        del active_trades[symbol]['tp1_price']
                                    if 'tp2_price' in active_trades[symbol]:
                                        del active_trades[symbol]['tp2_price']
                                    if 'tp1_quantity' in active_trades[symbol]:
                                        del active_trades[symbol]['tp1_quantity']
                                    if 'tp2_quantity' in active_trades[symbol]:
                                        del active_trades[symbol]['tp2_quantity']
                                    if 'tp_working_type' in active_trades[symbol]:
                                        del active_trades[symbol]['tp_working_type']
                                continue  # Both TPs exist, skip
                        
                        # No TP orders exist - we have stored TP details, create TP1 and TP2 orders
                        trade_info = active_trades[symbol]
                        logger.info(f"🔄 Background thread: Position exists for {symbol} with stored TP details - creating TP1 and TP2 orders")
                        success = create_tp1_tp2_if_needed(symbol, trade_info)
                        if success:
                            logger.info(f"✅ Background thread: TP1 and TP2 orders created successfully for {symbol}")
                        else:
                            logger.warning(f"⚠️ Background thread: Failed to create TP1/TP2 for {symbol} (check logs for details)")
                        
                    
                    except Exception as e:
                        logger.error(f"Error processing position {symbol}: {e}", exc_info=True)
                        send_slack_alert(
                            error_type="Background TP Check Error",
                            message=str(e),
                            details={'Position_Amount': position_amt if 'position_amt' in locals() else 'Unknown'},
                            symbol=symbol,
                            severity='WARNING'
                        )
                
                # Check trailing stop loss for ALL positions (not just those with TP details)
                # This runs after TP creation check, uses same position data (no extra API call!)
                if ENABLE_TRAILING_STOP_LOSS and positions_dict:
                    for symbol, position in positions_dict.items():
                        if symbol in active_trades:
                            try:
                                trade_info = active_trades[symbol]
                                # Only check if we have TP1 configured (TP1+TP2 mode)
                                if not trade_info.get('use_single_tp', False):
                                    update_trailing_stop_loss(symbol, trade_info, position)
                            except Exception as e:
                                logger.debug(f"Error checking trailing SL for {symbol}: {e}")
                
                # Also check for positions without stored TP details - calculate TP from position
                for symbol, position in positions_dict.items():
                    if symbol not in symbols_to_check:
                        # Position exists but no stored TP details - calculate TP from current position
                        # Check if there are pending orders (entry might not be filled yet)
                        try:
                            has_orders, open_orders = check_existing_orders(symbol, log_result=False)
                            existing_tp = [o for o in open_orders if o.get('type') == 'TAKE_PROFIT_MARKET']
                            
                            if existing_tp:
                                # TP already exists, mark as processed to avoid repeated checks
                                if symbol not in symbols_without_tp_logged:
                                    symbols_without_tp_logged.add(symbol)
                                continue
                            
                            if not has_orders:
                                # No pending orders and no TP - calculate TP from position
                                position_amt = float(position.get('positionAmt', 0))
                                entry_price = float(position.get('entryPrice', 0))
                                position_side = position.get('positionSide', 'BOTH')
                                
                                if entry_price > 0 and abs(position_amt) > 0:
                                    # Calculate TP: 2.1% profit target (adjustable)
                                    DEFAULT_TP_PERCENT = 0.021  # 2.1% profit
                                    
                                    if position_amt > 0:  # LONG position
                                        tp_price = entry_price * (1 + DEFAULT_TP_PERCENT)
                                        tp_side = 'SELL'
                                    else:  # SHORT position
                                        tp_price = entry_price * (1 - DEFAULT_TP_PERCENT)
                                        tp_side = 'BUY'
                                    
                                    # Get symbol info for precision
                                    try:
                                        exchange_info = client.futures_exchange_info()
                                        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
                                        
                                        if symbol_info:
                                            # Format price precision
                                            price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
                                            if price_filter:
                                                tick_size = float(price_filter['tickSize'])
                                                tp_price = format_price_precision(tp_price, tick_size)
                                            
                                            # Format quantity precision
                                            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                                            if lot_size_filter:
                                                step_size = float(lot_size_filter['stepSize'])
                                                tp_quantity = format_quantity_precision(abs(position_amt), step_size)
                                            else:
                                                tp_quantity = abs(position_amt)
                                            
                                            # Detect position mode
                                            is_hedge_mode = get_position_mode(symbol)
                                            
                                            # Create temporary trade_info for TP creation
                                            # Note: position_side will be retrieved from the position object by create_tp_if_needed
                                            temp_trade_info = {
                                                'tp_price': tp_price,
                                                'tp_quantity': tp_quantity,
                                                'tp_side': tp_side,
                                                'tp_working_type': 'MARK_PRICE'
                                            }
                                            
                                            logger.info(f"🔄 Background thread: Creating TP for {symbol} from position (calculated: {tp_price}, qty: {tp_quantity})")
                                            success = create_tp_if_needed(symbol, temp_trade_info)
                                            
                                            if success:
                                                logger.info(f"✅ Background thread: TP order created successfully for {symbol} (calculated from position)")
                                                # Only mark as processed if TP creation succeeded
                                                if symbol not in symbols_without_tp_logged:
                                                    symbols_without_tp_logged.add(symbol)
                                            else:
                                                logger.warning(f"⚠️ Background thread: Failed to create calculated TP for {symbol} (will retry on next cycle)")
                                                # Don't add to symbols_without_tp_logged - allow retry on next cycle
                                        else:
                                            logger.debug(f"Symbol info not found for {symbol}")
                                    except Exception as e:
                                        logger.debug(f"Error getting symbol info or creating TP for {symbol}: {e}")
                                        # Don't add to symbols_without_tp_logged - allow retry on next cycle
                                else:
                                    logger.debug(f"Skipping TP calculation for {symbol}: invalid entry_price or position_amt")
                        except Exception as e:
                            logger.debug(f"Error calculating TP for {symbol}: {e}")
                            pass
                            
            except Exception as e:
                logger.error(f"Error getting positions in background thread: {e}", exc_info=True)
                send_slack_alert(
                    error_type="Background Thread Error",
                    message=f"Error getting positions: {str(e)}",
                    severity='ERROR'
                )
                
        except Exception as e:
            logger.error(f"Error in TP creation background thread: {e}", exc_info=True)
            send_slack_alert(
                error_type="TP Creation Background Thread Error",
                message=str(e),
                severity='ERROR'
            )

# Start background thread for TP creation (only if client is initialized)
if client:
    tp_thread = threading.Thread(target=create_missing_tp_orders, daemon=True)
    tp_thread.start()
    logger.info("Background TP creation thread started")
else:
    logger.warning("Binance client not initialized - TP creation thread not started")  # seconds - prevent duplicate EXIT processing


def verify_webhook_token(payload_token):
    """Verify webhook token matches configured token"""
    return payload_token == WEBHOOK_TOKEN


def safe_float(value, default=None):
    """Safely convert value to float, handling None, 'null' string, and invalid values"""
    if value is None:
        return default
    if isinstance(value, str):
        # Handle Pine Script's "null" string
        if value.lower() == 'null' or value.strip() == '':
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    try:
        result = float(value)
        # Check for invalid values (NaN, infinity, negative/zero for prices)
        if result != result or result == float('inf') or result == float('-inf'):
            return default
        return result
    except (ValueError, TypeError):
        return default


def get_order_side(signal_side):
    """Convert signal side to Binance order side"""
    signal_side = signal_side.upper()
    if signal_side == 'LONG':
        return 'BUY'
    elif signal_side == 'SHORT':
        return 'SELL'
    else:
        raise ValueError(f"Invalid signal_side: {signal_side}")


def get_position_side(signal_side):
    """Get position side for Binance Futures"""
    signal_side = signal_side.upper()
    if signal_side == 'LONG':
        return 'LONG'
    elif signal_side == 'SHORT':
        return 'SHORT'
    else:
        return 'BOTH'  # Default


def get_position_mode(symbol):
    """Detect if account is in One-Way or Hedge mode for a symbol"""
    try:
        # Try to get position mode from Binance
        position_mode = client.futures_get_position_mode()
        # If dualSidePosition is True, it's Hedge mode, else One-Way
        return position_mode.get('dualSidePosition', False)
    except Exception as e:
        logger.warning(f"Could not detect position mode: {e}, defaulting to One-Way mode")
        return False  # Default to One-Way mode


def format_symbol(trading_symbol):
    """Format symbol for Binance (e.g., ETHUSDT.P -> ETHUSDT)"""
    # Remove .P or any exchange suffix
    symbol = trading_symbol.replace('.P', '').replace('.PERP', '').upper()
    # Ensure it ends with USDT
    if not symbol.endswith('USDT'):
        # Try to extract base currency and add USDT
        if 'USD' in symbol:
            symbol = symbol.replace('USD', 'USDT')
        else:
            symbol = symbol + 'USDT'
    return symbol


def check_existing_position(symbol, signal_side):
    """Check if there's an existing open position for the symbol"""
    try:
        # Get ALL positions first, then filter by symbol (more reliable than filtering in API call)
        all_positions = client.futures_position_information()
        for position in all_positions:
            position_symbol = position.get('symbol', '')
            position_amt = float(position.get('positionAmt', 0))
            
            # Match symbol (case-insensitive) and check if position exists
            if position_symbol.upper() == symbol.upper() and abs(position_amt) > 0:
                position_side = position.get('positionSide', 'BOTH')
                # Check if position side matches (or if it's BOTH mode)
                if position_side == 'BOTH' or position_side == signal_side.upper() or signal_side.upper() == 'BOTH':
                    logger.info(f"Existing position found for {symbol}: {position_amt} @ {position.get('entryPrice')}")
                    return True, position
        return False, None
    except Exception as e:
        logger.error(f"Error checking existing position: {e}")
        return False, None


def check_existing_orders(symbol, log_result=False):
    """Check for existing open orders (limit orders) for the symbol
    Args:
        symbol: Trading symbol to check
        log_result: If True, log the result (default False to reduce log spam)
    """
    try:
        open_orders = client.futures_get_open_orders(symbol=symbol)
        if open_orders:
            if log_result:
                logger.info(f"Found {len(open_orders)} open orders for {symbol}")
            return True, open_orders
        return False, []
    except Exception as e:
        logger.error(f"Error checking existing orders: {e}")
        return False, []


def cancel_order(symbol, order_id):
    """Cancel a specific order by ID"""
    try:
        result = client.futures_cancel_order(symbol=symbol, orderId=order_id)
        logger.info(f"Canceled order {order_id} for {symbol}: {result}")
        return True
    except Exception as e:
        logger.error(f"Failed to cancel order {order_id} for {symbol}: {e}")
        return False


def cancel_all_limit_orders(symbol, side=None):
    """Cancel all limit orders for a symbol, optionally filtered by side"""
    try:
        open_orders = client.futures_get_open_orders(symbol=symbol)
        canceled_count = 0
        for order in open_orders:
            # Only cancel LIMIT orders (entry orders)
            if order.get('type') == 'LIMIT':
                if side is None or order.get('side') == side:
                    try:
                        client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
                        logger.info(f"Canceled limit order {order['orderId']} for {symbol}")
                        canceled_count += 1
                    except Exception as e:
                        logger.error(f"Failed to cancel order {order['orderId']}: {e}")
        return canceled_count
    except Exception as e:
        logger.error(f"Error canceling orders for {symbol}: {e}")
        return 0


def close_position_at_market(symbol, signal_side, is_hedge_mode=False):
    """Close position at market price"""
    try:
        # Get ALL positions first, then filter by symbol (more reliable than filtering in API call)
        all_positions = client.futures_position_information()
        position_to_close = None
        
        for position in all_positions:
            position_symbol = position.get('symbol', '')
            position_amt = float(position.get('positionAmt', 0))
            
            # Match symbol (case-insensitive) and check if position exists
            if position_symbol.upper() == symbol.upper() and abs(position_amt) > 0:
                position_side = position.get('positionSide', 'BOTH')
                # Check if position side matches (or if it's BOTH mode)
                if position_side == 'BOTH' or position_side == signal_side.upper() or signal_side.upper() == 'BOTH':
                    position_to_close = position
                    break
        
        if not position_to_close:
            logger.info(f"No open position found for {symbol} to close")
            return {'success': False, 'error': 'No position to close'}
        
        position_amt = float(position_to_close.get('positionAmt', 0))
        if abs(position_amt) == 0:
            logger.info(f"Position amount is zero for {symbol}")
            return {'success': False, 'error': 'Position amount is zero'}
        
        # Determine close side (opposite of position)
        # If position is positive (LONG), we need to SELL
        # If position is negative (SHORT), we need to BUY
        close_side = 'SELL' if position_amt > 0 else 'BUY'
        close_quantity = abs(position_amt)
        
        # Get symbol info for quantity precision
        exchange_info = client.futures_exchange_info()
        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
        if symbol_info:
            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            if lot_size_filter:
                step_size = float(lot_size_filter['stepSize'])
                close_quantity = format_quantity_precision(close_quantity, step_size)
        
        # Create market order to close position
        # Try with reduceOnly first, then without if it fails (some accounts/modes don't accept it)
        close_params = {
            'symbol': symbol,
            'side': close_side,
            'type': 'MARKET',
            'quantity': close_quantity,
        }
        
        # Only include positionSide if in Hedge mode
        if is_hedge_mode:
            position_side_str = 'LONG' if position_amt > 0 else 'SHORT'
            close_params['positionSide'] = position_side_str
        
        # Try with reduceOnly first (for safety)
        close_params_with_reduce = close_params.copy()
        close_params_with_reduce['reduceOnly'] = True
        
        logger.info(f"Closing position at market for {symbol}: {close_params_with_reduce}")
        try:
            result = client.futures_create_order(**close_params_with_reduce)
            logger.info(f"Position closed successfully: {result}")
        except BinanceAPIException as e:
            # If reduceOnly is not accepted, try without it
            if e.code == -1106 or 'reduceonly' in str(e).lower() or 'not required' in str(e).lower():
                logger.warning(f"reduceOnly not accepted for {symbol}, retrying without it: {e}")
                try:
                    result = client.futures_create_order(**close_params)
                    logger.info(f"Position closed successfully (without reduceOnly): {result}")
                except Exception as e2:
                    logger.error(f"Failed to close position even without reduceOnly: {e2}")
                    raise e2  # Re-raise the retry error
            else:
                raise  # Re-raise if it's a different error
        
        return {
            'success': True,
            'order_id': result.get('orderId'),
            'symbol': symbol,
            'quantity': close_quantity,
            'side': close_side
        }
        
    except BinanceAPIException as e:
        logger.error(f"Binance API error closing position: {e}")
        send_slack_alert(
            error_type="Position Close Failed",
            message=f"{e.message} (Code: {e.code})",
            details={'Error_Code': e.code, 'Close_Side': close_side, 'Quantity': close_quantity},
            symbol=symbol,
            severity='ERROR'
        )
        return {'success': False, 'error': f'Binance API error: {e.message}'}
    except Exception as e:
        logger.error(f"Error closing position: {e}", exc_info=True)
        send_slack_alert(
            error_type="Position Close Error",
            message=str(e),
            details={'Close_Side': close_side, 'Quantity': close_quantity},
            symbol=symbol,
            severity='ERROR'
        )
        return {'success': False, 'error': str(e)}


def cleanup_closed_positions():
    """Periodically clean up active_trades for symbols with no open positions"""
    try:
        symbols_to_remove = []
        for symbol in list(active_trades.keys()):
            # Check if position exists (check both LONG and SHORT)
            has_long, _ = check_existing_position(symbol, 'LONG')
            has_short, _ = check_existing_position(symbol, 'SHORT')
            has_position = has_long or has_short
            
            if not has_position:
                # No position exists, check if orders exist
                has_orders, _ = check_existing_orders(symbol)
                if not has_orders:
                    # No position and no orders, can clean up
                    symbols_to_remove.append(symbol)
        
        for symbol in symbols_to_remove:
            logger.info(f"Cleaning up closed position tracking for {symbol}")
            del active_trades[symbol]
    except Exception as e:
        logger.error(f"Error cleaning up closed positions: {e}")


def format_quantity_precision(quantity, step_size):
    """Format quantity to match step_size precision, removing floating point errors
    
    Handles all step_size formats:
    - Large: 1.0, 10.0, 100.0 (0 decimal places)
    - Medium: 0.1, 0.5 (1 decimal place)
    - Small: 0.01, 0.001 (2-3 decimal places)
    - Very small: 0.0001, 0.00001 (4-5 decimal places)
    - Extremely small: 0.000001 (6+ decimal places)
    """
    # Calculate decimal places from step_size
    # Convert step_size to string to count decimal places accurately
    step_size_str = f"{step_size:.10f}".rstrip('0').rstrip('.')
    
    if '.' in step_size_str:
        # Count decimal places after the decimal point
        decimal_places = len(step_size_str.split('.')[1])
    else:
        # Step size is a whole number (1.0, 10.0, etc.)
        decimal_places = 0
    
    # Round to step size: divide by step_size, round to nearest integer, multiply back
    # This ensures quantity is a multiple of step_size
    quantity = round(quantity / step_size) * step_size
    
    # Format to correct decimal places to eliminate floating point errors
    # This converts values like 15.280000000000001 to 15.28
    quantity = round(quantity, decimal_places)
    
    # Final safety check: convert to string and back to float to remove any remaining precision errors
    # Format with the exact number of decimal places needed
    if decimal_places > 0:
        quantity_str = f"{quantity:.{decimal_places}f}"
        quantity = float(quantity_str)
    else:
        # For whole numbers, ensure it's an integer
        quantity = float(int(quantity))
    
    return quantity


def format_price_precision(price, tick_size):
    """Format price to match tick_size precision, removing floating point errors
    
    Handles all tick_size formats:
    - Large: 1.0, 10.0, 100.0 (0 decimal places)
    - Medium: 0.1, 0.5 (1 decimal place)
    - Small: 0.01, 0.001 (2-3 decimal places)
    - Very small: 0.0001, 0.00001 (4-5 decimal places)
    - Extremely small: 0.000001 (6+ decimal places)
    """
    # Calculate decimal places from tick_size
    # Convert tick_size to string to count decimal places accurately
    tick_size_str = f"{tick_size:.10f}".rstrip('0').rstrip('.')
    
    if '.' in tick_size_str:
        # Count decimal places after the decimal point
        decimal_places = len(tick_size_str.split('.')[1])
    else:
        # Tick size is a whole number (1.0, 10.0, etc.)
        decimal_places = 0
    
    # Round to tick size: divide by tick_size, round to nearest integer, multiply back
    # This ensures price is a multiple of tick_size
    price = round(price / tick_size) * tick_size
    
    # Format to correct decimal places to eliminate floating point errors
    # This converts values like 0.7111000000000001 to 0.7111
    price = round(price, decimal_places)
    
    # Final safety check: convert to string and back to float to remove any remaining precision errors
    # Format with the exact number of decimal places needed
    if decimal_places > 0:
        price_str = f"{price:.{decimal_places}f}"
        price = float(price_str)
    else:
        # For whole numbers, ensure it's an integer
        price = float(int(price))
    
    return price


def get_account_balance(cached=True):
    """Get account balance with caching to reduce API calls
    
    Args:
        cached: If True, use cached balance if available and fresh
    
    Returns:
        float: Account balance in USD, or None if error
    """
    global account_balance_cache
    
    current_time = time.time()
    
    # Check cache if enabled
    if cached and account_balance_cache['balance'] is not None:
        if current_time - account_balance_cache['timestamp'] < BALANCE_CACHE_TTL:
            return account_balance_cache['balance']
    
    try:
        account = client.futures_account()
        balance = float(account.get('totalWalletBalance', 0))
        
        # Update cache
        account_balance_cache['balance'] = balance
        account_balance_cache['timestamp'] = current_time
        
        return balance
    except Exception as e:
        logger.warning(f"Failed to get account balance: {e}")
        # Return cached value if available, even if stale
        if account_balance_cache['balance'] is not None:
            logger.info(f"Using cached account balance: ${account_balance_cache['balance']:,.2f}")
            return account_balance_cache['balance']
        return None


def calculate_trade_risk(entry_price, stop_loss, quantity, signal_side):
    """Calculate risk for a single trade
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        quantity: Position quantity
        signal_side: 'LONG' or 'SHORT'
    
    Returns:
        float: Risk in USD
    """
    if not entry_price or not stop_loss or not quantity:
        return 0.0
    
    if signal_side == 'LONG':
        risk_per_unit = abs(entry_price - stop_loss)
    else:  # SHORT
        risk_per_unit = abs(stop_loss - entry_price)
    
    risk_usd = risk_per_unit * quantity
    return risk_usd


def calculate_total_risk_including_pending(symbol, entry_price, stop_loss, quantity, signal_side):
    """Calculate total risk including current trade and all pending orders
    
    Args:
        symbol: Trading symbol (for current trade)
        entry_price: Entry price for current trade
        stop_loss: Stop loss for current trade
        quantity: Quantity for current trade
        signal_side: 'LONG' or 'SHORT' for current trade
    
    Returns:
        dict: {
            'current_trade_risk': float,
            'pending_orders_risk': float,
            'total_risk': float,
            'account_balance': float,
            'risk_percent': float,
            'pending_orders': list
        }
    """
    try:
        # Get account balance (cached)
        account_balance = get_account_balance()
        if account_balance is None or account_balance <= 0:
            logger.warning("Cannot calculate risk: Account balance not available")
            return None
        
        # Calculate current trade risk
        current_trade_risk = calculate_trade_risk(entry_price, stop_loss, quantity, signal_side)
        
        # Get all open orders to calculate pending risk
        pending_orders_risk = 0.0
        pending_orders = []
        
        try:
            all_orders = client.futures_get_open_orders()
            
            for order in all_orders:
                order_symbol = order.get('symbol', '')
                order_side = order.get('side', '')
                order_type = order.get('type', '')
                order_price = safe_float(order.get('price'), default=0)
                order_qty = safe_float(order.get('origQty'), default=0)
                
                # Only count LIMIT orders (pending entries)
                if order_type == 'LIMIT' and order_price > 0 and order_qty > 0:
                    # Get stop loss for this order from active_trades
                    order_risk = 0.0
                    if order_symbol in active_trades:
                        trade_info = active_trades[order_symbol]
                        sl_price = safe_float(trade_info.get('stop_loss'), default=0)
                        if sl_price > 0:
                            # Determine side from order
                            order_signal_side = 'LONG' if order_side == 'BUY' else 'SHORT'
                            order_risk = calculate_trade_risk(order_price, sl_price, order_qty, order_signal_side)
                    
                    if order_risk > 0:
                        pending_orders_risk += order_risk
                        pending_orders.append({
                            'symbol': order_symbol,
                            'side': order_side,
                            'price': order_price,
                            'quantity': order_qty,
                            'risk': order_risk
                        })
        except Exception as e:
            logger.warning(f"Error calculating pending orders risk: {e}")
        
        # Calculate total risk
        total_risk = current_trade_risk + pending_orders_risk
        risk_percent = (total_risk / account_balance) * 100 if account_balance > 0 else 0
        
        return {
            'current_trade_risk': current_trade_risk,
            'pending_orders_risk': pending_orders_risk,
            'total_risk': total_risk,
            'account_balance': account_balance,
            'risk_percent': risk_percent,
            'pending_orders': pending_orders
        }
    except Exception as e:
        logger.error(f"Error calculating total risk: {e}", exc_info=True)
        return None


def validate_risk_per_trade(symbol, entry_price, stop_loss, quantity, signal_side):
    """Validate if trade risk is within acceptable limits
    
    Args:
        symbol: Trading symbol
        entry_price: Entry price
        stop_loss: Stop loss price
        quantity: Position quantity
        signal_side: 'LONG' or 'SHORT'
    
    Returns:
        tuple: (is_valid: bool, risk_info: dict or None, error_message: str or None)
    """
    risk_info = calculate_total_risk_including_pending(symbol, entry_price, stop_loss, quantity, signal_side)
    
    if risk_info is None:
        # If we can't calculate risk, allow trade (fail-open)
        logger.warning(f"Could not calculate risk for {symbol}, allowing trade (fail-open)")
        return True, None, None
    
    risk_percent = risk_info['risk_percent']
    
    if risk_percent > MAX_RISK_PERCENT:
        error_msg = (f"Trade risk {risk_percent:.2f}% exceeds maximum {MAX_RISK_PERCENT}% "
                    f"(Current trade: ${risk_info['current_trade_risk']:.2f}, "
                    f"Pending orders: ${risk_info['pending_orders_risk']:.2f}, "
                    f"Total: ${risk_info['total_risk']:.2f} of ${risk_info['account_balance']:.2f} account)")
        logger.warning(f"🚫 Risk validation REJECTED for {symbol}: {error_msg}")
        return False, risk_info, error_msg
    
    logger.info(f"✅ Risk validation PASSED for {symbol}: {risk_percent:.2f}% risk "
               f"(Current: ${risk_info['current_trade_risk']:.2f}, "
               f"Pending: ${risk_info['pending_orders_risk']:.2f}, "
               f"Total: ${risk_info['total_risk']:.2f} of ${risk_info['account_balance']:.2f} account)")
    return True, risk_info, None


def calculate_quantity(entry_price, symbol_info):
    """Calculate quantity based on entry size and leverage"""
    # Position value = Entry size * Leverage (e.g., $10 * 20X = $200)
    position_value = ENTRY_SIZE_USD * LEVERAGE
    
    # Quantity = Position value / Entry price
    quantity = position_value / entry_price
    
    # Get step size and min quantity from symbol info
    lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
    step_size = float(lot_size_filter['stepSize']) if lot_size_filter else 0.001
    min_qty = float(lot_size_filter['minQty']) if lot_size_filter else 0.001
    
    # Format quantity with proper precision
    quantity = format_quantity_precision(quantity, step_size)
    
    # Ensure minimum quantity
    if quantity < min_qty:
        quantity = min_qty
        # Re-format after setting to min_qty to ensure precision
        quantity = format_quantity_precision(quantity, step_size)
    
    # Binance requires minimum notional of $100 (quantity * price >= 100)
    min_notional = 100.0
    actual_notional = quantity * entry_price
    if actual_notional < min_notional:
        # Adjust quantity to meet minimum notional - round UP to ensure we exceed minimum
        required_quantity = min_notional / entry_price
        # Round UP to next step_size increment (always round up, never down)
        quantity = math.ceil(required_quantity / step_size) * step_size
        # Ensure still meets min_qty
        if quantity < min_qty:
            quantity = min_qty
        # Format to ensure proper precision (but don't round down)
        # Use ceil again after formatting to ensure we don't lose precision
        formatted_qty = format_quantity_precision(quantity, step_size)
        formatted_notional = formatted_qty * entry_price
        if formatted_notional < min_notional:
            # If formatted quantity still doesn't meet minimum, add one more step
            quantity = formatted_qty + step_size
            quantity = format_quantity_precision(quantity, step_size)
        else:
            quantity = formatted_qty
        final_notional = quantity * entry_price
        logger.warning(f"Adjusted quantity to meet minimum notional: ${actual_notional:.2f} -> ${final_notional:.2f} (quantity: {quantity})")
    
    logger.info(f"Calculated quantity: {quantity} (Position value: ${position_value} @ ${entry_price}, Notional: ${quantity * entry_price:.2f}, step_size: {step_size})")
    return quantity


# Validation cache to avoid duplicate API calls
validation_cache = {}
VALIDATION_CACHE_TTL = 600  # 10 minutes (increased to reduce API calls and stay within free tier limits)


def validate_signal_with_ai(signal_data):
    """
    Validate trading signal using AI (Google Gemini API)
    
    Args:
        signal_data: Dictionary containing signal information
        
    Returns:
        dict: {
            'is_valid': bool,
            'confidence_score': float (0-100),
            'reasoning': str,
            'risk_level': str (LOW/MEDIUM/HIGH),
            'error': str (if validation failed)
        }
    """
    # Check if AI validation is enabled
    if not ENABLE_AI_VALIDATION:
        logger.debug("AI validation is disabled, skipping validation")
        return {
            'is_valid': True,
            'confidence_score': 100.0,
            'reasoning': 'AI validation disabled',
            'risk_level': 'UNKNOWN'
        }
    
    # Check if Gemini client is available
    if not gemini_client:
        logger.warning("Gemini client not available, proceeding without AI validation (fail-open)")
        return {
            'is_valid': True,
            'confidence_score': 100.0,
            'reasoning': 'AI validation unavailable, proceeding',
            'risk_level': 'UNKNOWN'
        }
    
    # Extract signal details
    symbol = format_symbol(signal_data.get('symbol', ''))
    signal_side = signal_data.get('signal_side', '').upper()
    entry_price = safe_float(signal_data.get('entry_price'), default=None)
    second_entry_price = safe_float(signal_data.get('second_entry_price'), default=None)
    stop_loss = safe_float(signal_data.get('stop_loss'), default=None)
    take_profit = safe_float(signal_data.get('take_profit'), default=None)
    timeframe = signal_data.get('timeframe', 'Unknown')
    
    # Extract indicator values from TradingView script (if provided)
    indicators = signal_data.get('indicators', {})
    
    # Check cache first
    cache_key = f"{symbol}_{signal_side}_{entry_price}_{stop_loss}_{take_profit}"
    current_time = time.time()
    if cache_key in validation_cache:
        cached_result, cache_time = validation_cache[cache_key]
        if current_time - cache_time < VALIDATION_CACHE_TTL:
            logger.debug(f"Using cached validation result for {symbol}")
            return cached_result
    
    # Validate required fields
    if not entry_price or entry_price <= 0:
        logger.warning(f"Cannot validate signal: invalid entry_price")
        return {
            'is_valid': True,  # Fail-open: proceed if we can't validate
            'confidence_score': 50.0,
            'reasoning': 'Invalid entry price, proceeding without validation',
            'risk_level': 'MEDIUM'
        }
    
    # Calculate risk/reward ratio
    risk_reward_ratio = None
    if stop_loss and take_profit:
        if signal_side == 'LONG':
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
        else:  # SHORT
            risk = abs(stop_loss - entry_price)
            reward = abs(entry_price - take_profit)
        
        if risk > 0:
            risk_reward_ratio = reward / risk
    
    # Fetch real-time market data for technical analysis
    market_data = {}
    try:
        if client:
            # Get current market price
            ticker = client.futures_symbol_ticker(symbol=symbol)
            current_price = float(ticker.get('price', 0))
            market_data['current_price'] = current_price
            
            # Calculate price distance from entry
            if current_price > 0:
                price_distance_pct = abs((entry_price - current_price) / current_price) * 100
                market_data['price_distance_pct'] = price_distance_pct
                market_data['entry_vs_current'] = 'ABOVE' if entry_price > current_price else 'BELOW'
            
            # Get recent candles for technical analysis (last 50 candles)
            # Map timeframe to Binance interval
            timeframe_map = {
                '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
                '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'
            }
            interval = timeframe_map.get(timeframe.lower(), '1h')  # Default to 1h
            
            try:
                klines = client.futures_klines(symbol=symbol, interval=interval, limit=100)  # Increased to 100 for better analysis
                if klines:
                    # Extract OHLCV data
                    opens = [float(k[1]) for k in klines]
                    highs = [float(k[2]) for k in klines]
                    lows = [float(k[3]) for k in klines]
                    closes = [float(k[4]) for k in klines]
                    volumes = [float(k[5]) for k in klines]
                    
                    # Store raw candle data for AI analysis
                    market_data['candle_count'] = len(klines)
                    market_data['recent_candles'] = {
                        'last_10_closes': closes[-10:] if len(closes) >= 10 else closes,
                        'last_10_highs': highs[-10:] if len(highs) >= 10 else highs,
                        'last_10_lows': lows[-10:] if len(lows) >= 10 else lows,
                        'last_10_volumes': volumes[-10:] if len(volumes) >= 10 else volumes
                    }
                    
                    # Fetch lower timeframe data for tighter TP/SL analysis (15m and 1h)
                    # This helps find the nearest realistic reversal points
                    lower_tf_data = {}
                    try:
                        # Get 15m data for near-term support/resistance
                        klines_15m = client.futures_klines(symbol=symbol, interval='15m', limit=50)
                        if klines_15m:
                            highs_15m = [float(k[2]) for k in klines_15m]
                            lows_15m = [float(k[3]) for k in klines_15m]
                            closes_15m = [float(k[4]) for k in klines_15m]
                            lower_tf_data['15m'] = {
                                'recent_high': max(highs_15m[-20:]) if len(highs_15m) >= 20 else max(highs_15m) if highs_15m else None,
                                'recent_low': min(lows_15m[-20:]) if len(lows_15m) >= 20 else min(lows_15m) if lows_15m else None,
                                'resistance_levels': sorted(set(highs_15m[-30:]), reverse=True)[:3] if len(highs_15m) >= 30 else [],
                                'support_levels': sorted(set(lows_15m[-30:]))[:3] if len(lows_15m) >= 30 else []
                            }
                    except Exception as e:
                        logger.debug(f"Could not fetch 15m data for {symbol}: {e}")
                    
                    try:
                        # Get 1h data for short-term support/resistance
                        klines_1h = client.futures_klines(symbol=symbol, interval='1h', limit=50)
                        if klines_1h:
                            highs_1h = [float(k[2]) for k in klines_1h]
                            lows_1h = [float(k[3]) for k in klines_1h]
                            closes_1h = [float(k[4]) for k in klines_1h]
                            lower_tf_data['1h'] = {
                                'recent_high': max(highs_1h[-20:]) if len(highs_1h) >= 20 else max(highs_1h) if highs_1h else None,
                                'recent_low': min(lows_1h[-20:]) if len(lows_1h) >= 20 else min(lows_1h) if lows_1h else None,
                                'resistance_levels': sorted(set(highs_1h[-30:]), reverse=True)[:3] if len(highs_1h) >= 30 else [],
                                'support_levels': sorted(set(lows_1h[-30:]))[:3] if len(lows_1h) >= 30 else []
                            }
                    except Exception as e:
                        logger.debug(f"Could not fetch 1h data for {symbol}: {e}")
                    
                    if lower_tf_data:
                        market_data['lower_timeframe_levels'] = lower_tf_data
                    
                    # Calculate technical indicators for AI's own analysis
                    # Trend analysis (multiple timeframes)
                    if len(closes) >= 5:
                        short_trend = ((closes[-1] - closes[-5]) / closes[-5]) * 100  # Last 5 candles
                        market_data['short_term_trend_pct'] = short_trend
                        market_data['short_term_direction'] = 'UP' if short_trend > 0.3 else 'DOWN' if short_trend < -0.3 else 'SIDEWAYS'
                    
                    if len(closes) >= 20:
                        medium_trend = ((closes[-1] - closes[-20]) / closes[-20]) * 100  # Last 20 candles
                        market_data['medium_term_trend_pct'] = medium_trend
                        market_data['medium_term_direction'] = 'UP' if medium_trend > 0.5 else 'DOWN' if medium_trend < -0.5 else 'SIDEWAYS'
                    
                    if len(closes) >= 50:
                        long_trend = ((closes[-1] - closes[-50]) / closes[-50]) * 100  # Last 50 candles
                        market_data['long_term_trend_pct'] = long_trend
                        market_data['long_term_direction'] = 'UP' if long_trend > 1 else 'DOWN' if long_trend < -1 else 'SIDEWAYS'
                    
                    # Moving averages (for AI's own trend analysis)
                    if len(closes) >= 20:
                        market_data['sma_20'] = sum(closes[-20:]) / 20
                    if len(closes) >= 50:
                        market_data['sma_50'] = sum(closes[-50:]) / 50
                    
                    # Support/Resistance levels (multiple levels)
                    if len(highs) >= 20:
                        recent_highs = highs[-20:]
                        recent_lows = lows[-20:]
                        market_data['resistance_level'] = max(recent_highs)
                        market_data['support_level'] = min(recent_lows)
                        market_data['recent_high'] = max(recent_highs)
                        market_data['recent_low'] = min(recent_lows)
                    
                    # Volume analysis (comprehensive)
                    if len(volumes) >= 20:
                        avg_volume = sum(volumes[-20:]) / 20
                    current_volume = volumes[-1] if volumes else 0
                    market_data['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1.0
                    market_data['volume_status'] = 'HIGH' if market_data['volume_ratio'] > 1.5 else 'NORMAL' if market_data['volume_ratio'] > 0.5 else 'LOW'
                    # Volume trend
                    recent_vol_avg = sum(volumes[-5:]) / 5
                    older_vol_avg = sum(volumes[-20:-5]) / 15 if len(volumes) >= 20 else avg_volume
                    market_data['volume_trend'] = 'INCREASING' if recent_vol_avg > older_vol_avg * 1.2 else 'DECREASING' if recent_vol_avg < older_vol_avg * 0.8 else 'STABLE'
                    
                    # Price position relative to recent range
                    if market_data.get('recent_high', 0) > market_data.get('recent_low', 0):
                        price_position = ((current_price - market_data['recent_low']) / 
                                         (market_data['recent_high'] - market_data['recent_low'])) * 100
                        market_data['price_position_in_range'] = price_position
                        if price_position > 80:
                            market_data['price_level'] = 'NEAR_RESISTANCE'
                        elif price_position < 20:
                            market_data['price_level'] = 'NEAR_SUPPORT'
                        else:
                            market_data['price_level'] = 'MID_RANGE'
                    
                    # Volatility (ATR-like calculation)
                    if len(highs) >= 14 and len(lows) >= 14:
                        true_ranges = [highs[i] - lows[i] for i in range(max(0, len(highs)-14), len(highs))]
                        avg_true_range = sum(true_ranges) / len(true_ranges) if true_ranges else 0
                        volatility_pct = (avg_true_range / current_price) * 100 if current_price > 0 else 0
                        market_data['volatility_pct'] = volatility_pct
                        market_data['volatility_status'] = 'HIGH' if volatility_pct > 2 else 'MODERATE' if volatility_pct > 1 else 'LOW'
                    
                    # Price momentum (rate of change)
                    if len(closes) >= 10:
                        momentum = ((closes[-1] - closes[-10]) / closes[-10]) * 100
                        market_data['momentum_pct'] = momentum
                        market_data['momentum_direction'] = 'BULLISH' if momentum > 0.5 else 'BEARISH' if momentum < -0.5 else 'NEUTRAL'
                    
                    # Price action patterns (simplified)
                    if len(closes) >= 3:
                        # Check for higher highs/lower lows pattern
                        if len(highs) >= 3 and len(lows) >= 3:
                            recent_highs_3 = highs[-3:]
                            recent_lows_3 = lows[-3:]
                            if recent_highs_3[-1] > recent_highs_3[0] and recent_lows_3[-1] > recent_lows_3[0]:
                                market_data['price_pattern'] = 'HIGHER_HIGHS_HIGHER_LOWS'  # Bullish
                            elif recent_highs_3[-1] < recent_highs_3[0] and recent_lows_3[-1] < recent_lows_3[0]:
                                market_data['price_pattern'] = 'LOWER_HIGHS_LOWER_LOWS'  # Bearish
                            else:
                                market_data['price_pattern'] = 'MIXED'
                    
                    # Legacy fields for backward compatibility
                    recent_closes = closes[-20:] if len(closes) >= 20 else closes
                    if len(recent_closes) >= 2:
                        price_change = ((recent_closes[-1] - recent_closes[0]) / recent_closes[0]) * 100
                        market_data['recent_trend_pct'] = price_change
                        market_data['trend_direction'] = 'UP' if price_change > 0.5 else 'DOWN' if price_change < -0.5 else 'SIDEWAYS'
                    
            except Exception as e:
                logger.debug(f"Could not fetch klines for {symbol}: {e}")
                market_data['klines_error'] = str(e)
    except Exception as e:
        logger.debug(f"Could not fetch market data for {symbol}: {e}")
        market_data['error'] = str(e)
    
    # Build indicator values section for AI prompt
    indicator_info = ""
    if indicators:
        rsi_val = safe_float(indicators.get('rsi'), default=None)
        macd_line = safe_float(indicators.get('macd_line'), default=None)
        macd_signal = safe_float(indicators.get('macd_signal'), default=None)
        macd_hist = safe_float(indicators.get('macd_histogram'), default=None)
        stoch_k = safe_float(indicators.get('stoch_k'), default=None)
        stoch_d = safe_float(indicators.get('stoch_d'), default=None)
        ema200_val = safe_float(indicators.get('ema200'), default=None)
        atr_val = safe_float(indicators.get('atr'), default=None)
        bb_upper = safe_float(indicators.get('bb_upper'), default=None)
        bb_basis = safe_float(indicators.get('bb_basis'), default=None)
        bb_lower = safe_float(indicators.get('bb_lower'), default=None)
        smv_norm = safe_float(indicators.get('smv_normalized'), default=None)
        cum_smv = safe_float(indicators.get('cum_smv'), default=None)
        supertrend_val = safe_float(indicators.get('supertrend'), default=None)
        supertrend_bull = indicators.get('supertrend_bull', False)
        obv_val = safe_float(indicators.get('obv'), default=None)
        rel_vol_pct = safe_float(indicators.get('relative_volume_percentile'), default=None)
        mfi_val = safe_float(indicators.get('mfi'), default=None)
        vol_ratio = safe_float(indicators.get('volume_ratio'), default=None)
        has_bull_div = indicators.get('has_bullish_divergence', False)
        has_bear_div = indicators.get('has_bearish_divergence', False)
        at_bottom = indicators.get('at_bottom', False)
        at_top = indicators.get('at_top', False)
        smart_money_buy = indicators.get('smart_money_buying', False)
        smart_money_sell = indicators.get('smart_money_selling', False)
        price_above_ema200 = indicators.get('price_above_ema200', False)
        price_below_ema200 = indicators.get('price_below_ema200', False)
        
        indicator_info = f"""
TRADINGVIEW INDICATOR VALUES (from your script):
- RSI: {(f'{rsi_val:.2f}' if rsi_val is not None else 'N/A')} (Oversold: <30, Overbought: >85)
- MACD Line: {(f'{macd_line:.4f}' if macd_line is not None else 'N/A')}
- MACD Signal: {(f'{macd_signal:.4f}' if macd_signal is not None else 'N/A')}
- MACD Histogram: {(f'{macd_hist:.4f}' if macd_hist is not None else 'N/A')} (Positive = bullish momentum)
- Stochastic K: {(f'{stoch_k:.2f}' if stoch_k is not None else 'N/A')} (Oversold: <20, Overbought: >80)
- Stochastic D: {(f'{stoch_d:.2f}' if stoch_d is not None else 'N/A')}
- EMA 200: ${(f'{ema200_val:,.8f}' if ema200_val is not None else 'N/A')} (Price {'ABOVE' if price_above_ema200 else 'BELOW'} EMA200 = {'BULLISH' if price_above_ema200 else 'BEARISH'} trend)
- ATR: {(f'{atr_val:.8f}' if atr_val is not None else 'N/A')} (Volatility measure)
- Bollinger Bands: Upper=${(f'{bb_upper:,.8f}' if bb_upper is not None else 'N/A')}, Basis=${(f'{bb_basis:,.8f}' if bb_basis is not None else 'N/A')}, Lower=${(f'{bb_lower:,.8f}' if bb_lower is not None else 'N/A')}
- Smart Money Volume (Normalized): {(f'{smv_norm:.2f}' if smv_norm is not None else 'N/A')} (Positive = buying pressure)
- Cumulative SMV: {(f'{cum_smv:.2f}' if cum_smv is not None else 'N/A')} ({'BUYING' if smart_money_buy else 'SELLING' if smart_money_sell else 'NEUTRAL'} pressure)
- Supertrend: ${(f'{supertrend_val:,.8f}' if supertrend_val is not None else 'N/A')} ({'BULLISH' if supertrend_bull else 'BEARISH'})
- OBV: {(f'{obv_val:.2f}' if obv_val is not None else 'N/A')} (Rising = buying pressure)
- Relative Volume Percentile: {(f'{rel_vol_pct:.1f}' if rel_vol_pct is not None else 'N/A')}% (High: >70%, Low: <30%)
- MFI (Money Flow Index): {(f'{mfi_val:.2f}' if mfi_val is not None else 'N/A')} (Oversold: <20, Overbought: >80)
- Volume Ratio: {(f'{vol_ratio:.2f}' if vol_ratio is not None else 'N/A')}x (vs average)
- Bullish Divergence: {'YES ✅' if has_bull_div else 'NO'}
- Bearish Divergence: {'YES ✅' if has_bear_div else 'NO'}
- At Bottom/Top: {'BOTTOM ✅' if at_bottom else 'TOP ✅' if at_top else 'MID-RANGE'}"""
    
    # Build prompt for AI - Enhanced with real market data AND indicator values for technical analysis
    market_info = ""
    if market_data.get('current_price'):
        market_info = f"""
═══════════════════════════════════════════════════════════════
REAL-TIME MARKET DATA (from Binance API) - FOR YOUR INDEPENDENT ANALYSIS:
═══════════════════════════════════════════════════════════════
CURRENT MARKET CONDITIONS:
- Current Market Price: ${market_data['current_price']:,.8f}
- Entry Price (from signal): ${entry_price:,.8f}
- Entry Price vs Current: Entry is {market_data.get('entry_vs_current', 'N/A')} current price
- Price Distance: {market_data.get('price_distance_pct', 0):.2f}% from current price

TREND ANALYSIS (Multiple Timeframes):
- Short-term Trend (last 5 candles): {market_data.get('short_term_direction', market_data.get('trend_direction', 'N/A'))} ({market_data.get('short_term_trend_pct', market_data.get('recent_trend_pct', 0)):+.2f}%)
- Medium-term Trend (last 20 candles): {market_data.get('medium_term_direction', market_data.get('trend_direction', 'N/A'))} ({market_data.get('medium_term_trend_pct', market_data.get('recent_trend_pct', 0)):+.2f}%)
- Long-term Trend (last 50 candles): {market_data.get('long_term_direction', market_data.get('trend_direction', 'N/A'))} ({market_data.get('long_term_trend_pct', market_data.get('recent_trend_pct', 0)):+.2f}%)
- Overall Trend ({timeframe}): {market_data.get('trend_direction', 'N/A')} ({market_data.get('recent_trend_pct', 0):+.2f}%)

MOVING AVERAGES:
- SMA 20: ${(f"{market_data.get('sma_20'):,.8f}" if market_data.get('sma_20') is not None else 'N/A')}
- SMA 50: ${(f"{market_data.get('sma_50'):,.8f}" if market_data.get('sma_50') is not None else 'N/A')}
- Price vs SMA 20: {'ABOVE' if market_data.get('sma_20') and market_data['current_price'] > market_data['sma_20'] else 'BELOW' if market_data.get('sma_20') else 'N/A'}
- Price vs SMA 50: {'ABOVE' if market_data.get('sma_50') and market_data['current_price'] > market_data['sma_50'] else 'BELOW' if market_data.get('sma_50') else 'N/A'}

SUPPORT & RESISTANCE LEVELS:
- Resistance Level: ${market_data.get('resistance_level', market_data.get('recent_high', 0)):,.8f}
- Support Level: ${market_data.get('support_level', market_data.get('recent_low', 0)):,.8f}
- Price Range (last 20 candles): ${market_data.get('recent_low', 0):,.8f} - ${market_data.get('recent_high', 0):,.8f}
- Current Price Position: {market_data.get('price_level', 'N/A')} ({market_data.get('price_position_in_range', 0):.1f}% of range)

VOLUME ANALYSIS:
- Volume Status: {market_data.get('volume_status', 'N/A')} (current/avg ratio: {market_data.get('volume_ratio', 1):.2f}x)
- Volume Trend: {market_data.get('volume_trend', 'N/A')} (increasing/decreasing/stable)

MOMENTUM & VOLATILITY:
- Price Momentum: {market_data.get('momentum_direction', 'N/A')} ({market_data.get('momentum_pct', 0):+.2f}%)
- Volatility: {market_data.get('volatility_status', 'N/A')} ({market_data.get('volatility_pct', 0):.2f}%)

PRICE ACTION PATTERNS:
- Pattern: {market_data.get('price_pattern', 'N/A')} (Higher Highs/Higher Lows = Bullish, Lower Highs/Lower Lows = Bearish)

═══════════════════════════════════════════════════════════════"""
    
    prompt = f"""You are an INSTITUTIONAL CRYPTO TRADER and WHALE with 20+ years of elite professional trading experience.
You have achieved consistent 200% monthly returns (2x per month) through:
- Deep understanding of crypto market structure, order flow, and institutional/whale behavior
- Mastery of multi-timeframe analysis (HTF → LTF) and market structure (BOS, CHoCH)
- Ability to identify liquidity pools, stop-hunt zones, order blocks, and fair value gaps
- Understanding of volume & open interest behavior, funding rates, and market microstructure
- Experience trading with institutional capital - you think like a whale, not retail
- Mastery of risk management with minimum RR ≥ 1:3 for all trades
- Ability to identify high-probability institutional liquidity zones for entries
- Institutional-level understanding of market manipulation, smart money flow, and retail vs. professional behavior

YOUR TRADING PHILOSOPHY (INSTITUTIONAL/WHALE PERSPECTIVE):
- You think like a whale/institution, not retail - you see what institutions see
- You understand market structure, liquidity, and order flow at the deepest level
- You know that entries must be at institutional liquidity zones, not arbitrary price levels
- You understand that markets move based on liquidity grabs, stop hunts, and order blocks
- You combine multi-timeframe analysis with market structure for superior edge
- You trust your institutional-level analysis but verify with multiple confirmations
- You've seen every market scenario - bull markets, bear markets, crashes, pumps, liquidations
- You only take trades with minimum RR ≥ 1:3 - anything less is retail behavior

YOUR ANALYSIS APPROACH (INSTITUTIONAL METHODOLOGY):
1. FIRST: Multi-timeframe analysis (HTF → LTF) - understand the dominant trend and market structure
2. SECOND: Market structure analysis (BOS, CHoCH) - identify break of structure and change of character
3. THIRD: Liquidity analysis - identify liquidity pools, stop-hunt zones, and institutional entry zones
4. FOURTH: Order flow analysis - identify order blocks, fair value gaps, and optimal entry points
5. FIFTH: Volume & OI analysis - understand institutional accumulation/distribution and funding rates
6. SIXTH: Validate signal against YOUR institutional analysis - does it align with whale behavior?
7. SEVENTH: Optimize entries ONLY at high-probability institutional liquidity zones
8. EIGHTH: Combine YOUR institutional analysis with TradingView indicators - both must align

IMPORTANT CONTEXT:
- This signal comes from a TradingView indicator that already filters signals (65% win rate)
- Your job is to VALIDATE signals, not reject everything - the script already did filtering
- You have REAL-TIME market data - use it to assess trend, volume, and price position
- Think practically: Does this signal have merit based on indicators and market data?
- Focus on TradingView indicators (PRIMARY) and basic market data (SECONDARY)
- Accept RR ≥ 1:1 (good setups), prefer RR ≥ 1:2 (better), excellent if RR ≥ 1:3
- Only reject if MULTIPLE red flags: poor R/R (< 0.5) AND weak indicators AND strong counter-trend

═══════════════════════════════════════════════════════════════
STEP 1: INSTITUTIONAL MULTI-TIMEFRAME & MARKET STRUCTURE ANALYSIS
═══════════════════════════════════════════════════════════════

Before looking at the signal, analyze the market using YOUR INSTITUTIONAL/WHALE METHODOLOGY.
Think like the whale/institution you are - see liquidity, order flow, and market structure.

1. MULTI-TIMEFRAME ANALYSIS (HTF → LTF - Institutional Approach):
   - HIGHER TIMEFRAME (HTF) - 4H/1D: What is the DOMINANT TREND? (This is your primary filter)
   - CURRENT TIMEFRAME ({timeframe}): What is the INTERMEDIATE TREND? (Does it align with HTF?)
   - LOWER TIMEFRAME (LTF) - 15m/5m: What is the SHORT-TERM MOMENTUM? (For entry timing)
   - TREND ALIGNMENT: Are all timeframes aligned? (HTF + CTF + LTF = HIGH PROBABILITY)
   - TREND CONFLICT: If timeframes conflict, is this a REVERSAL or CORRECTION? (Evaluate carefully)
   - Moving Averages: Are they aligned across timeframes? (Bullish alignment = price > SMA20 > SMA50)
   - Price vs MAs: Is price respecting or rejecting key levels? (Institutional behavior)

2. MARKET STRUCTURE ANALYSIS (BOS, CHoCH - Institutional Methodology):
   - BREAK OF STRUCTURE (BOS): Has there been a BOS? (Bullish BOS = higher high breaks previous high)
   - CHANGE OF CHARACTER (CHoCH): Has there been a CHoCH? (Bullish CHoCH = higher low after lower low)
   - MARKET STRUCTURE: Is structure BULLISH, BEARISH, or NEUTRAL? (This determines bias)
   - STRUCTURE BREAKS: Are we in a structure break or consolidation? (Structure breaks = high probability)
   - SWING POINTS: Identify key swing highs/lows (these are institutional levels)
   - STRUCTURE ALIGNMENT: Does the signal align with market structure? (Must align for approval)

3. LIQUIDITY ANALYSIS (Institutional/Whale Perspective):
   - LIQUIDITY POOLS: Where are the liquidity zones? (Above resistance for longs, below support for shorts)
   - STOP-HUNT ZONES: Where are retail stops likely placed? (Institutions hunt these)
   - LIQUIDITY GRABS: Has price grabbed liquidity? (Liquidity grab = potential reversal)
   - INSTITUTIONAL ENTRY ZONES: Where would institutions enter? (At liquidity zones, not random prices)
   - ORDER BLOCKS: Are there order blocks (institutional entry zones) near the signal?
   - FAIR VALUE GAPS (FVG): Are there FVGs that need to be filled? (These are entry zones)

4. SUPPORT & RESISTANCE (Institutional Level Identification):
   - REAL SUPPORT: Where will INSTITUTIONAL buyers step in? (Not just recent lows)
   - REAL RESISTANCE: Where will INSTITUTIONAL sellers step in? (Not just recent highs)
   - ORDER BLOCKS: Identify bullish/bearish order blocks (institutional entry zones)
   - LIQUIDITY LEVELS: Identify liquidity pools above/below key levels
   - PSYCHOLOGICAL LEVELS: Round numbers, previous swing points (institutions use these)
   - IS PRICE AT INSTITUTIONAL ZONE? (Order block, liquidity pool, FVG = high probability entry)

5. VOLUME & OPEN INTEREST ANALYSIS (Institutional Order Flow):
   - VOLUME PROFILE: Is volume INCREASING on moves in trend direction? (Institutional accumulation)
   - VOLUME DIVERGENCE: Is volume CONFIRMING or DIVERGING from price? (Divergence = warning)
   - OPEN INTEREST (OI): Is OI increasing or decreasing? (Increasing OI = institutional interest)
   - FUNDING RATES: What are funding rates? (Extreme funding = potential reversal)
   - SMART MONEY: Are institutions buying or selling? (Use Smart Money indicators)
   - ACCUMULATION/DISTRIBUTION: Are institutions accumulating or distributing? (This determines direction)

6. MOMENTUM & VOLATILITY (Institutional Perspective):
   - MOMENTUM STRENGTH: Is momentum STRONG or WEAK? (Strong momentum = institutional participation)
   - VOLATILITY EXPANSION: Is volatility EXPANDING or CONTRACTING? (Expanding = big move coming)
   - PRICE PATTERNS: Higher highs/higher lows (bullish) vs Lower highs/lower lows (bearish)
   - DIVERGENCE: Price vs indicators divergence? (Divergence = potential reversal)
   - MARKET TEMPERATURE: Is the market ready for a move? (Hot = high probability, Cold = low probability)

7. YOUR MARKET DIRECTION PREDICTION (Based on Available Data):
   Based on AVAILABLE MARKET DATA (trend, volume, price position, support/resistance):
   - What direction is the market MOST LIKELY to move? (UP/DOWN/SIDEWAYS)
   - How CONFIDENT are you? (High/Medium/Low) - Be honest based on available data
   - What are the KEY FACTORS? (Trend alignment, volume confirmation, price position, support/resistance)
   - What are the RISKS? (Counter-trend, weak volume, unfavorable price position)
   - Does this signal align with market direction? (Yes/Partial/No)
   - What is the Risk/Reward ratio? (Accept if ≥ 1:1, good if ≥ 1:2, excellent if ≥ 1:3)

═══════════════════════════════════════════════════════════════
STEP 2: INSTITUTIONAL ENTRY & TARGET OPTIMIZATION (CRITICAL - WHALE METHODOLOGY)
═══════════════════════════════════════════════════════════════

As an institutional trader/whale, you MUST optimize ALL prices based on:
- Multi-timeframe structure (HTF → LTF alignment)
- Market structure (BOS, CHoCH, swing points)
- Liquidity zones (order blocks, FVGs, stop-hunt zones)
- Institutional entry zones (NOT based on closeness to current price)
- Minimum RR ≥ 1:3 requirement (if can't achieve, modify or discard)

CRITICAL: Entry optimization is based on TECHNICAL & STRUCTURAL CONFIRMATION, NOT on closeness to current price.
Entries must be placed ONLY at HIGH-PROBABILITY INSTITUTIONAL LIQUIDITY ZONES.

1. ENTRY 1 (PRIMARY INSTITUTIONAL ENTRY):
   - IDENTIFY INSTITUTIONAL LIQUIDITY ZONE: Where is the order block, FVG, or liquidity pool?
   - LONG: Entry should be at BULLISH ORDER BLOCK, FVG fill, or liquidity grab zone (NOT just below current price)
   - SHORT: Entry should be at BEARISH ORDER BLOCK, FVG fill, or liquidity grab zone (NOT just above current price)
   - EVALUATE: Is original entry at an institutional zone? (If not, REPLACE with optimal zone)
   - STRUCTURAL CONFIRMATION: Entry must align with HTF structure and market structure (BOS/CHoCH)
   - If original entry is NOT at institutional zone, REPLACE it with optimal institutional entry
   - DO NOT suggest entry based on "closeness to current price" - only suggest based on STRUCTURE

2. ENTRY 2 (CONFIRMATION OR SCALING ENTRY) - FULL TECHNICAL ANALYSIS REQUIRED:
   - CRITICAL: Perform the SAME LEVEL of technical analysis for Entry 2 as Entry 1
   - Entry 2 is NOT spacing - it requires FULL institutional analysis
   - SPACING IS THE LAST PRIORITY - only use if no institutional zones are found
   
   TECHNICAL ANALYSIS FOR ENTRY 2 (Same as Entry 1):
   - MULTI-TIMEFRAME: Analyze HTF → LTF structure to find Entry 2 institutional zones
   - MARKET STRUCTURE: Identify BOS/CHoCH levels where Entry 2 should be placed
   - LIQUIDITY ZONES: Find order blocks, FVGs, stop-hunt zones BELOW Entry 1 (LONG) or ABOVE Entry 1 (SHORT)
   - SUPPORT/RESISTANCE: Identify key support (LONG) or resistance (SHORT) levels for Entry 2
   - VOLUME ANALYSIS: Check volume profile and OI behavior at potential Entry 2 zones
   
   ENTRY 2 OPTIMIZATION (Priority Order):
   1. FIRST PRIORITY: Find institutional zones (order blocks, FVGs, support/resistance, reversal points)
   2. SECOND PRIORITY: Check if original Entry 2 is at institutional zone - if YES, KEEP it
   3. THIRD PRIORITY: If original Entry 2 is NOT at institutional zone, use BEST institutional zone from analysis
   4. LAST PRIORITY (ONLY IF NO ZONES FOUND): Use spacing calculations as fallback
   
   OPTIMIZATION RULES:
   - LONG: Should be BELOW Entry 1 (at another institutional zone or confirmation level)
   - SHORT: Should be ABOVE Entry 1 (at another institutional zone or confirmation level)
   - MUST be at an institutional liquidity zone (order block, FVG, support/resistance, reversal point)
   - ALWAYS prioritize institutional zones FIRST - spacing is LAST RESORT
   - NEVER use spacing if institutional zones are available - always use the zone
   - If Entry 2 is missing, perform FULL technical analysis to find optimal Entry 2 at institutional zone
   - SPACING (last resort only) must be realistic (1.0-2.5x ATR or 2-7% depending on timeframe)

3. STOP LOSS (INSTITUTIONAL RISK MANAGEMENT):
   - IMPORTANT: Use LOWER TIMEFRAMES (15m, 1h) for SL placement, NOT HTF levels (HTF SL would be too wide)
   - LONG: Must be BELOW Entry 1 at the NEAREST support level on 15m/1h (typically 1.5-3% from entry, max 4%)
   - SHORT: Must be ABOVE Entry 1 at the NEAREST resistance level on 15m/1h (typically 1.5-3% from entry, max 4%)
   - PLACE SL: At nearest support/resistance on LTF (15m, 1h), NOT HTF levels
   - Evaluate if SL is too tight (will get stopped out by noise) or too wide (poor R/R)
   - Suggest optimal SL based on: LTF (15m, 1h) support/resistance levels, not HTF structure

4. TAKE PROFIT (INSTITUTIONAL TARGET ALIGNMENT):
   - IMPORTANT: Use LOWER TIMEFRAMES (15m, 1h) for TP placement, NOT HTF targets (HTF targets are too far)
   - LONG: Must be ABOVE Entry 1 at the NEAREST resistance level on 15m/1h (typically 3-8% from entry, max 15%)
   - SHORT: Must be BELOW Entry 1 at the NEAREST support level on 15m/1h (typically 3-8% from entry, max 15%)
   - TARGET ALIGNMENT: TP should align with LTF support/resistance levels (not HTF structure targets which are too far)
   - MINIMUM RR ≥ 1:3 REQUIRED (if can't achieve, modify or discard signal)
   - Evaluate if TP is realistic based on: LTF (15m, 1h) resistance/support levels, not HTF targets
   - If TP is too aggressive (won't hit) or too conservative (leaves profit on table), suggest better TP at nearest LTF level

PRICE OPTIMIZATION RULES (INSTITUTIONAL METHODOLOGY):
- If prices are OPTIMAL (at institutional zones, RR ≥ 1:3): Keep original prices (set suggested_* to null)
- If prices can be IMPROVED (better institutional zones, better RR): Suggest better prices with reasoning
- If prices are INVALID (not at institutional zones, RR < 1:3): REPLACE with optimal prices or REJECT signal
- If setup is WEAK, COUNTER-TREND, or lacks INSTITUTIONAL CONFIRMATION: Modify or discard entirely

ENTRY OPTIMIZATION PRIORITY (Institutional Thinking):
1. FIRST: Identify institutional liquidity zones (order blocks, FVGs, liquidity pools)
2. SECOND: Validate entry is at institutional zone (if not, REPLACE)
3. THIRD: Ensure entry aligns with HTF structure and market structure (BOS/CHoCH)
4. FOURTH: Calculate RR ratio (must be ≥ 1:3)
5. FIFTH: If can't achieve RR ≥ 1:3 or no institutional zones, MODIFY or DISCARD signal

═══════════════════════════════════════════════════════════════
STEP 3: COMBINE YOUR ANALYSIS + TRADINGVIEW INDICATORS (BOTH DECISION MAKERS!)
═══════════════════════════════════════════════════════════════

IMPORTANT: You have TWO independent sources of analysis - use BOTH equally:

1. YOUR INDEPENDENT MARKET ANALYSIS (from Step 1):
   - Your trend analysis (short/medium/long-term)
   - Your support/resistance identification
   - Your volume and momentum assessment
   - YOUR market direction prediction

2. TRADINGVIEW INDICATORS (provided below):
   - RSI, MACD, Stochastic, EMA200, Supertrend
   - Volume indicators (OBV, Relative Volume)
   - Smart Money indicators
   - Divergence signals

DECISION PROCESS (COMBINE BOTH SOURCES):

A. Compare signal direction with YOUR market prediction:
   - If signal ALIGNS with YOUR prediction: +20-30% confidence boost
   - If signal PARTIALLY aligns: +10-20% confidence boost
   - If signal CONTRADICTS YOUR prediction: -20-30% confidence penalty

B. Analyze TradingView indicators (provided below):
   - Count indicators that SUPPORT the signal direction
   - Count indicators that CONTRADICT the signal direction
   - If 8+ indicators support: +20-30% confidence boost
   - If 6-7 indicators support: +10-20% confidence boost
   - If 4-5 indicators support: +0-10% confidence boost
   - If 2-3 indicators support: -10-20% confidence penalty
   - If 0-1 indicators support: -20-30% confidence penalty or REJECT

C. COMBINE BOTH ANALYSES:
   - Start with base confidence: 50%
   - Add/subtract based on YOUR market analysis alignment
   - Add/subtract based on TradingView indicator alignment
   - Final confidence = Base + YOUR analysis impact + Indicator impact

EXAMPLE:
- YOUR analysis: Market likely to go UP (LONG signal aligns) → +25%
- TradingView indicators: 7 indicators support LONG → +15%
- Final confidence: 50% + 25% + 15% = 90%

Remember: BOTH sources are EQUALLY IMPORTANT. Don't ignore either one!

Signal Details:
- Symbol: {symbol}
- Direction: {signal_side}
- Timeframe: {timeframe}
- Entry Price (Entry 1): ${entry_price:,.8f}
- Entry Price 2 (DCA): ${(f'{second_entry_price:,.8f}' if second_entry_price is not None and second_entry_price > 0 else 'N/A (not provided)')}
- Stop Loss: ${(f'{stop_loss:,.8f}' if stop_loss is not None and stop_loss > 0 else 'N/A (not provided)')}
- Take Profit: ${(f'{take_profit:,.8f}' if take_profit is not None and take_profit > 0 else 'N/A (not provided)')}
- Risk/Reward Ratio: {(f'{risk_reward_ratio:.2f}' if risk_reward_ratio is not None else 'N/A')}{market_info}{indicator_info}

═══════════════════════════════════════════════════════════════
STEP 4: DETAILED TRADINGVIEW INDICATOR ANALYSIS (SECOND DECISION MAKER)
═══════════════════════════════════════════════════════════════

CRITICAL: You MUST analyze EACH indicator value INDIVIDUALLY from the TradingView script.
These are REAL-TIME indicator values calculated by the Pine Script - analyze them like an institutional trader.

ANALYZE EACH INDICATOR INDEPENDENTLY - These are YOUR SECOND SOURCE OF ANALYSIS:

For EACH indicator value provided below, you MUST:
1. Read the ACTUAL VALUE (not just whether it exists)
2. Determine if it supports the signal direction? (YES/NO)
3. Assess how strong is the signal? (STRONG/MODERATE/WEAK)
4. Count total indicators that SUPPORT vs CONTRADICT
5. Consider the COMBINATION of indicators - are they aligned or conflicting?

INDICATOR ANALYSIS GUIDE (Analyze Each Value Individually):
1. RSI Analysis (Check the ACTUAL RSI value):
   - Read the RSI value from TradingView indicators
   - LONG signals: RSI < 50 is GOOD (oversold <30 is EXCELLENT) ✅
   - SHORT signals: RSI > 50 is GOOD (overbought >85 is EXCELLENT) ✅
   - RSI divergence (bullish/bearish) = STRONG confirmation ✅
   - If RSI contradicts signal direction, note it as a CONTRADICTING indicator

2. MACD Analysis (Check ALL MACD values: Line, Signal, Histogram):
   - Read MACD Line, Signal Line, and Histogram values from TradingView
   - MACD Line > Signal Line = Bullish momentum ✅
   - MACD Histogram positive = Bullish momentum ✅
   - LONG: MACD bullish (Line > Signal AND Histogram > 0) = GOOD ✅
   - SHORT: MACD bearish (Line < Signal AND Histogram < 0) = GOOD ✅
   - If MACD contradicts signal direction, note it as a CONTRADICTING indicator

3. Stochastic Analysis (Check BOTH Stoch K and Stoch D):
   - Read Stochastic K and D values from TradingView
   - LONG: Stoch K/D < 50 (oversold <20 is EXCELLENT) ✅
   - SHORT: Stoch K/D > 50 (overbought >80 is EXCELLENT) ✅
   - If Stochastic contradicts signal direction, note it as a CONTRADICTING indicator

4. Trend Filters (EMA 200 & Supertrend - Check BOTH):
   - Read EMA200 value and Supertrend value/bullish status from TradingView
   - LONG: Price above EMA200 AND Supertrend bullish = STRONG trend ✅
   - SHORT: Price below EMA200 AND Supertrend bearish = STRONG trend ✅
   - Contradicting trend = Evaluate carefully but APPROVE if other factors good
   - If both EMA200 and Supertrend contradict signal, note as CONTRADICTING

5. Volume Analysis (Check ALL volume indicators):
   - Read Relative Volume Percentile, Volume Ratio, OBV, and Smart Money indicators
   - High Relative Volume (>70%) = Strong confirmation ✅
   - Volume Ratio > 1.5x = Strong confirmation ✅
   - OBV rising = Buying pressure ✅
   - Smart Money Buying = Institutional accumulation ✅
   - If volume indicators contradict signal, note as CONTRADICTING

6. Bollinger Bands (Check ALL BB values: Upper, Basis, Lower):
   - Read BB Upper, Basis, and Lower values from TradingView
   - Compare current price to BB levels (provided in market data)
   - LONG near lower band = Good entry zone ✅
   - SHORT near upper band = Good entry zone ✅
   - Price at bands = Potential reversal ✅

7. Divergence & Reversal Signals (Check boolean flags):
   - Read Bullish Divergence, Bearish Divergence, At Bottom, At Top flags
   - Bullish Divergence + At Bottom = EXCELLENT LONG setup ✅
   - Bearish Divergence + At Top = EXCELLENT SHORT setup ✅
   - These are STRONG reversal signals - weight them heavily

8. MFI (Money Flow Index) Analysis:
   - Read MFI value from TradingView
   - LONG: MFI < 50 (oversold <20 is EXCELLENT) ✅
   - SHORT: MFI > 50 (overbought >80 is EXCELLENT) ✅
   - If MFI contradicts signal direction, note it as a CONTRADICTING indicator

9. Market Data Analysis (Combine with Indicators):
   - Trend Alignment: Use both market trend AND indicator trends (EMA200, Supertrend)
   - Price Position: Combine market support/resistance with Bollinger Bands levels
   - Volume: Use both Relative Volume Percentile AND Volume Ratio for confirmation

10. Risk/Reward: 
   - APPROVE if R/R >= 1.0 (even 1:1 is acceptable for good setups)
   - APPROVE if R/R >= 0.8 AND 4+ indicators support (acceptable R/R with good indicator alignment)
   - Only REJECT if R/R < 0.5 AND multiple indicators contradict (poor R/R + weak indicators)
   - R/R between 0.5-0.8: Approve if 5+ indicators support (good indicator alignment compensates for lower R/R)

INDICATOR COUNTING METHOD:
- Go through EACH indicator value provided above
- For each indicator, determine: SUPPORT, CONTRADICT, or NEUTRAL
- Count total: SUPPORT count vs CONTRADICT count
- If SUPPORT count > CONTRADICT count by 3+: Strong alignment ✅
- If SUPPORT count = CONTRADICT count: Mixed signals (evaluate carefully)
- If CONTRADICT count > SUPPORT count by 3+: Weak alignment (may reject) ❌

SIGNAL QUALITY SCORING:
- EXCELLENT (80-100%): Multiple indicators aligned + good R/R + volume confirmation + divergence/reversal signals
- GOOD (60-79%): Most indicators aligned + acceptable R/R + normal volume
- ACCEPTABLE (50-59%): Some indicators aligned + acceptable R/R (may have minor concerns)
- QUESTIONABLE (40-49%): Mixed signals but not clearly bad (still approve if above threshold)
- POOR (0-39%): Multiple indicators contradict signal + poor R/R + low volume

REJECTION CRITERIA (only reject if MULTIPLE red flags - be lenient):
- Risk/Reward < 0.5 AND
- Signal contradicts STRONG trend (>5% against signal direction) AND
- Price at VERY unfavorable level (LONG at strong resistance, SHORT at strong support) AND
- Multiple indicators STRONGLY contradict signal (7+ indicators against signal direction)

IMPORTANT: If TradingView script sent this signal, it likely has merit. Only reject if ALL of the above conditions are met. When in doubt, APPROVE (the script already filtered signals).

═══════════════════════════════════════════════════════════════
FINAL DECISION PROCESS (COMBINE BOTH DECISION MAKERS EQUALLY):
═══════════════════════════════════════════════════════════════

You have TWO EQUAL DECISION MAKERS - combine them:

DECISION MAKER 1: YOUR INDEPENDENT MARKET ANALYSIS (from Step 1)
- What direction did YOU predict? (UP/DOWN/SIDEWAYS)
- How confident are YOU? (High/Medium/Low)
- Does the signal align with YOUR prediction?

DECISION MAKER 2: TRADINGVIEW INDICATORS (from Step 4)
- How many indicators support the signal? (Count them)
- How many indicators contradict the signal? (Count them)
- What is the overall indicator alignment? (Strong/Moderate/Weak)

COMBINATION FORMULA (TRADINGVIEW INDICATORS ARE PRIMARY - 70% WEIGHT):
1. Start with base confidence: 60% (signals are pre-filtered by TradingView script)
2. TradingView indicator impact (PRIMARY - 70% weight):
   - 8+ indicators support: +25-35% (EXCELLENT alignment)
   - 6-7 indicators support: +15-25% (GOOD alignment)
   - 4-5 indicators support: +5-15% (ACCEPTABLE alignment)
   - 2-3 indicators support: -5-10% (WEAK alignment, but still approve if R/R is good)
   - 0-1 indicators support: -15-25% (POOR alignment, reject only if R/R < 0.5)
3. Add YOUR market analysis impact (SECONDARY - 30% weight):
   - Signal aligns with YOUR prediction: +5-10%
   - Signal partially aligns: +0-5%
   - Signal contradicts YOUR prediction: -5-15% (but don't reject if indicators are strong)
4. Final confidence = Base + TradingView indicators (70%) + Market analysis (30%)
5. Clamp final score between 0-100%

DECISION RULES (MORE LENIENT - FOCUS ON INDICATORS):
- If final confidence >= 40%: APPROVE (TradingView script already filtered signals)
- If final confidence 30-39%: APPROVE with low confidence (unless R/R < 0.5 AND indicators contradict)
- If final confidence < 30%: REJECT only if MULTIPLE red flags (poor R/R + weak indicators + strong counter-trend)

REASONING REQUIREMENT:
In your reasoning, EXPLICITLY mention:
1. YOUR market analysis conclusion
2. TradingView indicator alignment
3. How you combined both to reach final confidence

Remember: TRADINGVIEW INDICATORS ARE PRIMARY (70% weight) - they come from a proven script with 65% win rate. Market analysis is SECONDARY (30% weight) - use it to fine-tune confidence, not to reject good indicator setups.

Respond in JSON format ONLY with this exact structure:
{{
    "is_valid": true/false,
    "confidence_score": 0-100,
    "reasoning": "MUST mention: (1) TradingView indicator alignment (PRIMARY - count how many support vs contradict), (2) Market data analysis (trend, volume, price position), (3) Risk/Reward validation, (4) Final conclusion. Example: 'TradingView indicators show 7 out of 10 indicators support LONG direction. RSI at 45 indicates neutral-bullish, MACD shows bullish momentum (histogram positive), Stochastic at 35 indicates oversold recovery, Volume is high (75th percentile), Supertrend is bullish. Market data shows short-term uptrend (+2.3%), price is mid-range, volume is increasing. Risk/Reward ratio is 1:2.5 which is acceptable. Combined analysis gives 72% confidence - APPROVE.'",
    "risk_level": "LOW" or "MEDIUM" or "HIGH",
    "suggested_entry_price": <number> or null,
    "suggested_second_entry_price": <number> or null,
    "suggested_stop_loss": <number> or null,
    "suggested_take_profit": <number> or null,
    "price_suggestion_reasoning": "Why these prices are suggested (if different from original)"
}}

REASONING REQUIREMENT (Practical Analysis - Focus on Available Data):
Your reasoning should be PRACTICAL and based on AVAILABLE DATA. Mention:
1. TRADINGVIEW INDICATOR ANALYSIS (PRIMARY): "TradingView indicators show X out of Y indicators support this direction. Specifically: RSI at [value] indicates..., MACD shows..., Volume/OI behavior suggests..., [list key supporting indicators]..."
2. MARKET DATA ANALYSIS (SECONDARY): "Market data shows [trend direction], price is [position relative to support/resistance], volume is [status]. This [aligns/partially aligns/contradicts] the signal because..."
3. RISK/REWARD VALIDATION: "Risk/Reward ratio is [value]. Entry at $X, Stop Loss at $Y, Take Profit at $Z. This provides [adequate/good/excellent] risk management..."
4. FINAL CONCLUSION: "Combining TradingView indicator analysis (X indicators support) with market data validation (trend/volume/position), I conclude this signal has [confidence level]% confidence. [Approve/Reject] because..."

Think practically - TradingView script already filtered signals. Your job is to validate based on indicators and basic market data, not to require perfect institutional-level analysis that isn't available. Focus on what you CAN analyze: indicators, trend, volume, R/R ratio.

Note: If you want to suggest price optimizations (based on institutional zones, NOT closeness to price), include the suggested_* fields. Otherwise, you may omit them or set them to null.

Confidence Score Guidelines (Practical Standards - Focus on TradingView Indicators):
- 80-100: Excellent signal, 7+ indicators support, RR ≥ 1:1, strong trend alignment
- 60-79: Good signal, 5-6 indicators support, RR ≥ 1:1, acceptable trend alignment
- 50-59: Acceptable signal, 4-5 indicators support, RR ≥ 1:1, minor concerns but still valid
- 40-49: Questionable signal, 3-4 indicators support, RR ≥ 0.8, mixed signals but approve if R/R acceptable
- 30-39: Weak signal, 2-3 indicators support, RR ≥ 0.5, approve only if not strongly counter-trend
- 0-29: Poor signal, 0-1 indicators support OR RR < 0.5, reject only if multiple red flags

Remember: TradingView script already filters signals (65% win rate). Your job is to VALIDATE, not reject everything. Approve signals with 4+ indicator support and RR ≥ 1:1. Only reject if MULTIPLE red flags (poor R/R < 0.5 AND weak indicators AND strong counter-trend).

PRICE OPTIMIZATION (Institutional Methodology - Based on Structure & Liquidity):
You MUST calculate and suggest optimized prices using INSTITUTIONAL METHODOLOGY:
- Multi-timeframe structure (HTF → LTF alignment)
- Market structure (BOS, CHoCH, swing points)
- Liquidity zones (order blocks, FVGs, stop-hunt zones, liquidity pools)
- Volume & OI behavior, funding rates
- Support/resistance levels, ATR volatility, Bollinger Bands, EMA200
- Use indicator values (RSI, MACD, Stochastic, etc.) for confirmation

CRITICAL: Entry optimization is based on TECHNICAL & STRUCTURAL CONFIRMATION, NOT on closeness to current price.
Entries must be placed ONLY at HIGH-PROBABILITY INSTITUTIONAL LIQUIDITY ZONES.

OPTIMIZATION RULES (Institutional Methodology - Only suggest if BETTER than original):
CRITICAL: Signals are based on 2H/4H timeframes. 
- FIRST: Evaluate if original prices are GOOD based on technical analysis (institutional zones, support/resistance, market structure)
- IF ORIGINAL PRICES ARE GOOD: KEEP THEM (set suggested_* to null) - DO NOT change just to make a 1-2% adjustment
- IF ORIGINAL PRICES NEED IMPROVEMENT: Only suggest changes if there's a SOLID TECHNICAL REASON (better institutional zone, better support/resistance, better R/R)
- WHEN SUGGESTING CHANGES: Keep optimizations within 1-2% MAXIMUM of original to ensure orders FILL (this is a LIMIT, not a requirement)
- DO NOT suggest changes just because price is "close to entry" or to make a 1-2% adjustment - only change when technically justified

1. ENTRY PRICE (PRIMARY INSTITUTIONAL ENTRY):
   - STEP 1: EVALUATE original entry based on technical analysis:
     * Is original entry at an institutional liquidity zone? (order block, FVG, support/resistance)
     * Does original entry align with HTF structure and market structure (BOS/CHoCH)?
     * Is original entry well-positioned for the trade direction?
   
   - STEP 2: DECISION - Keep or Optimize:
     * If original entry is GOOD (at institutional zone, aligns with structure): KEEP IT (set suggested_entry_price to null)
     * If original entry is NOT at institutional zone AND there's a better zone nearby: Suggest better entry
     * DO NOT suggest changes just to make a 1-2% adjustment - only change if technically justified
   
   - STEP 3: If optimization needed (SOLID TECHNICAL REASON):
     * IDENTIFY INSTITUTIONAL LIQUIDITY ZONE: Order block, FVG, liquidity pool, or stop-hunt zone
     * LONG trades: Suggest entry at BULLISH ORDER BLOCK, FVG fill, or liquidity grab zone
     * SHORT trades: Suggest entry at BEARISH ORDER BLOCK, FVG fill, or liquidity grab zone
     * ENTRY MUST ALIGN with HTF structure and market structure (BOS/CHoCH)
     * MAXIMUM distance: Keep within 1-2% of original to ensure orders FILL (this is a LIMIT, not a requirement)
     * Only suggest if new entry is clearly better AND within 1-2% of original

2. ENTRY 2 (CONFIRMATION OR SCALING ENTRY) - VALIDATE AND OPTIMIZE INDEPENDENTLY WITH FULL TECHNICAL ANALYSIS:
   - CRITICAL: You MUST perform the SAME LEVEL of technical analysis for Entry 2 as you do for Entry 1
   - Entry 2 is NOT a spacing calculation - it requires FULL institutional analysis
   - SPACING IS THE LAST PRIORITY - only use if no institutional zones are found
   
   TECHNICAL ANALYSIS FOR ENTRY 2 (Same as Entry 1):
   - MULTI-TIMEFRAME ANALYSIS: Check HTF → LTF structure for Entry 2 location
   - MARKET STRUCTURE: Identify BOS/CHoCH levels where Entry 2 should be placed
   - LIQUIDITY ZONES: Find order blocks, FVGs, stop-hunt zones BELOW Entry 1 (LONG) or ABOVE Entry 1 (SHORT)
   - SUPPORT/RESISTANCE: Identify key support levels (LONG) or resistance levels (SHORT) for Entry 2
   - VOLUME ANALYSIS: Check volume profile and OI behavior at potential Entry 2 zones
   - STRUCTURAL CONFIRMATION: Entry 2 must align with HTF structure and market structure (BOS/CHoCH)
   
   ENTRY 2 OPTIMIZATION PROCESS (Priority Order):
   1. FIRST PRIORITY: Analyze market structure to identify ALL institutional zones below Entry 1 (LONG) or above Entry 1 (SHORT)
      - Order blocks, FVGs, support/resistance levels, reversal points
      - These are the PRIMARY candidates for Entry 2
   
   2. SECOND PRIORITY: Check if original Entry 2 is ALREADY at one of these institutional zones
      - If YES and well-positioned → KEEP it (set suggested_second_entry_price to null)
      - If NO → Find the BEST institutional zone from your analysis
   
   3. THIRD PRIORITY: If multiple institutional zones exist, choose the BEST one based on:
      - Closest to Entry 1 (but still realistic spacing)
      - Strongest support/resistance level
      - Best volume/OI confirmation
      - Best alignment with market structure
   
   4. LAST PRIORITY (ONLY IF NO INSTITUTIONAL ZONES FOUND): Use spacing calculations as fallback
      - Only use spacing if NO institutional zones are identified
      - Spacing must be realistic (not too far) - Entry 2 must fill before trade closes
      - SPACING GUIDELINES (Maximum realistic spacing - LAST RESORT):
        * 1H timeframe: 1.0-1.5x ATR spacing (tight, fills quickly)
        * 2H timeframe: 1.2-1.8x ATR spacing (moderate, still fills reliably)
        * 4H timeframe: 1.5-2.0x ATR spacing (wider but still realistic)
        * Daily timeframe: 2.0-2.5x ATR spacing (widest, but must still be fillable)
        * Percentage fallback (if ATR not available):
          - 1H: 2-3% spacing (tight)
          - 2H: 3-4% spacing (moderate)
          - 4H: 4-5% spacing (wider but realistic)
          - Daily: 5-7% spacing (widest realistic)
   
   OPTIMIZATION RULES:
   - STEP 1: EVALUATE original Entry 2 based on technical analysis:
     * Is original Entry 2 at an institutional liquidity zone? (order block, FVG, support/resistance)
     * Does original Entry 2 align with market structure and provide good spacing from Entry 1?
     * Is original Entry 2 well-positioned for the trade direction?
     * CRITICAL: Is Entry 2 DIFFERENT from Stop Loss? (Entry 2 must be between Entry 1 and SL)
     * For LONG: Entry 2 must be ABOVE SL and BELOW Entry 1
     * For SHORT: Entry 2 must be BELOW SL and ABOVE Entry 1
   
   - STEP 2: DECISION - Keep or Optimize:
     * If original Entry 2 is GOOD (at institutional zone, good spacing, different from SL): KEEP IT (set suggested_second_entry_price to null)
     * If original Entry 2 is NOT at institutional zone AND there's a better zone: Suggest better Entry 2
     * If original Entry 2 is SAME as SL: MUST suggest different Entry 2 (this is a critical error)
     * DO NOT suggest changes just to make a 1-2% adjustment - only change if technically justified
   
   - STEP 3: If optimization needed (SOLID TECHNICAL REASON):
     * LONG: Entry 2 should be BELOW Entry 1 and ABOVE SL (at another institutional zone or confirmation level)
     * SHORT: Entry 2 should be ABOVE Entry 1 and BELOW SL (at another institutional zone or confirmation level)
     * MUST be at an institutional liquidity zone (order block, FVG, support/resistance, reversal point)
     * CRITICAL VALIDATION: Entry 2 must be DIFFERENT from SL with proper spacing:
       - For LONG: Entry 2 > SL (at least 0.5% above SL, preferably 1-2% above)
       - For SHORT: Entry 2 < SL (at least 0.5% below SL, preferably 1-2% below)
     * ALWAYS prioritize institutional zones FIRST - spacing is LAST RESORT
     * MAXIMUM distance: Keep within 1-2% of original to ensure orders FILL (this is a LIMIT, not a requirement)
     * Only suggest if new Entry 2 is clearly better AND within 1-2% of original AND different from SL
     * Entry 2 can be optimized independently of Entry 1 - validate both separately with full technical analysis
     * NEVER use spacing calculations if institutional zones are available - always use the zone

3. STOP LOSS (INSTITUTIONAL RISK MANAGEMENT - EVALUATE ORIGINAL SL FIRST):
   - STEP 1: EVALUATE THE ORIGINAL SL from the signal:
     * Check if original SL is at a realistic support/resistance level (check lower timeframes: 15m, 1h)
     * Check if original SL is tight enough (typically 1.5-3% from entry, max 4% if structure requires)
     * Check if original SL provides adequate protection (below/above order block, liquidity pool)
     * Check if original SL is too tight (will get stopped out by noise) or too wide (poor R/R)
     * CRITICAL: Is SL DIFFERENT from Entry 2? (SL must be beyond Entry 2)
     * For LONG: SL must be BELOW Entry 2 (at least 0.5% below, preferably 1-2% below)
     * For SHORT: SL must be ABOVE Entry 2 (at least 0.5% above, preferably 1-2% above)
   
   - STEP 2: DECISION - Keep or Optimize:
     * If original SL is GOOD (realistic, tight, provides protection, different from Entry 2): KEEP IT (set suggested_stop_loss to null)
     * If original SL is SAME as Entry 2: MUST suggest different SL (this is a critical error - SL must be beyond Entry 2)
     * If original SL is TOO TIGHT (will get stopped out by noise): Suggest WIDER SL (only if technically justified)
     * If original SL is TOO WIDE (poor R/R, unnecessary risk): Suggest TIGHTER SL (only if technically justified)
     * DO NOT suggest changes just to make a 1-2% adjustment - only change if technically justified
   
   - STEP 3: If optimization needed (SOLID TECHNICAL REASON), use LOWER TIMEFRAMES (15m, 1h):
     * Check market_data['lower_timeframe_levels']['15m']['support_levels'] and market_data['lower_timeframe_levels']['1h']['support_levels'] for LONG
     * Check market_data['lower_timeframe_levels']['15m']['resistance_levels'] and market_data['lower_timeframe_levels']['1h']['resistance_levels'] for SHORT
     * LONG: Find nearest support level BELOW entry (within 1-2% of original SL, max 4% if structure requires)
     * SHORT: Find nearest resistance level ABOVE entry (within 1-2% of original SL, max 4% if structure requires)
     * CRITICAL VALIDATION: SL must be DIFFERENT from Entry 2 with proper spacing:
       - For LONG: SL < Entry 2 (at least 0.5% below Entry 2, preferably 1-2% below)
       - For SHORT: SL > Entry 2 (at least 0.5% above Entry 2, preferably 1-2% above)
     * SL should be TIGHT but realistic - just beyond the nearest support/resistance, NOT 5-10% away
     * DO NOT suggest wide SL (5%+) - find the nearest realistic reversal point on lower timeframes
     * MAXIMUM distance: Keep within 1-2% of original to ensure orders FILL (this is a LIMIT, not a requirement)
     * Only suggest if new SL is clearly better AND within 1-2% of original AND different from Entry 2
   
   - CRITICAL RULES:
     * ALWAYS evaluate original SL first - don't blindly suggest new SL
     * If original SL is good OR within 1-2% of optimal level, KEEP IT (set suggested_stop_loss to null)
     * Only suggest new SL if original is clearly problematic AND new SL is within 1-2% of original AND different from Entry 2
     * When suggesting new SL, explain WHY original SL needs adjustment in price_suggestion_reasoning
     * NEVER suggest SL that is the same as Entry 2 - this is a critical error

4. TAKE PROFIT (INSTITUTIONAL TARGET ALIGNMENT - EVALUATE ORIGINAL TP FIRST):
   - STEP 1: EVALUATE THE ORIGINAL TP from the signal:
     * Check if original TP is at a realistic support/resistance level (check lower timeframes: 15m, 1h)
     * Check if original TP is achievable (not too far - typically 3-15% from entry is realistic)
     * Check if original TP gives RR ≥ 1:3 (minimum requirement)
     * Check if original TP aligns with market structure and liquidity objectives
   
   - STEP 2: DECISION - Keep or Optimize:
     * If original TP is GOOD (realistic, achievable, RR ≥ 1:3, at support/resistance): KEEP IT (set suggested_take_profit to null)
     * If original TP is TOO AGGRESSIVE (won't hit, too far, unrealistic): Suggest LOWER TP (only if technically justified)
     * If original TP is TOO CONSERVATIVE (leaves profit on table, can go further): Suggest HIGHER TP (only if technically justified)
     * If original TP gives RR < 1:3: Find nearest level that achieves RR ≥ 1:3 (only if technically justified)
     * DO NOT suggest changes just to make a 1-2% adjustment - only change if technically justified
   
   - STEP 3: If optimization needed (SOLID TECHNICAL REASON):
     * MAXIMUM distance: Keep within 1-2% of original to ensure orders FILL (this is a LIMIT, not a requirement)
     * Only suggest if new TP is clearly better AND within 1-2% of original
   
   - STEP 3: If optimization needed, use LOWER TIMEFRAMES (15m, 1h):
     * Check market_data['lower_timeframe_levels']['15m']['resistance_levels'] and market_data['lower_timeframe_levels']['1h']['resistance_levels'] for LONG
     * Check market_data['lower_timeframe_levels']['15m']['support_levels'] and market_data['lower_timeframe_levels']['1h']['support_levels'] for SHORT
     * LONG: Find nearest resistance level ABOVE entry (within 1-2% of original TP, typically 3-8% from entry, max 15% if structure requires)
     * SHORT: Find nearest support level BELOW entry (within 1-2% of original TP, typically 3-8% from entry, max 15% if structure requires)
     * TP should be REALISTIC and ACHIEVABLE - not 20-30% away
     * DO NOT suggest very wide TP (15%+) unless absolutely necessary for RR ≥ 1:3
   
   - CRITICAL RULES:
     * ALWAYS evaluate original TP first - don't blindly suggest new TP
     * If original TP is good OR within 1-2% of optimal level, KEEP IT (set suggested_take_profit to null)
     * Only suggest new TP if original is clearly problematic AND new TP is within 1-2% of original
     * When suggesting new TP, explain WHY original TP needs adjustment in price_suggestion_reasoning

CALCULATION METHOD (Institutional Approach):
- IDENTIFY INSTITUTIONAL ZONES: Order blocks, FVGs, liquidity pools, stop-hunt zones
- VALIDATE STRUCTURE: HTF → LTF alignment, BOS/CHoCH, swing points (HTF for trend direction, LTF for precise levels)
- CALCULATE ENTRY: At institutional liquidity zone (order block, FVG, or liquidity pool) - can use HTF for direction
- CALCULATE SL: At NEAREST support/resistance on LTF (15m, 1h) - NOT HTF (HTF SL would be too wide)
- CALCULATE TP: At NEAREST resistance/support on LTF (15m, 1h) - NOT HTF targets (HTF targets are too far, 20-30% away)
- VALIDATE RR: Must be ≥ 1:3 (if not, modify or discard)
- CRITICAL: HTF is for TREND DIRECTION and ENTRY optimization, LTF (15m, 1h) is for SL/TP placement (tighter, more achievable)

FINAL VALIDATION (CRITICAL - CHECK BEFORE RETURNING):
Before returning your response, VALIDATE that:
1. Entry 2 is DIFFERENT from Stop Loss:
   - For LONG: Entry 2 > SL (Entry 2 must be at least 0.5% above SL)
   - For SHORT: Entry 2 < SL (Entry 2 must be at least 0.5% below SL)
   - If Entry 2 == SL, this is a CRITICAL ERROR - adjust one of them
2. All prices are properly spaced:
   - Entry 1, Entry 2, SL, TP must all be different prices
   - Entry 2 must be between Entry 1 and SL
   - SL must be beyond Entry 2 (further from Entry 1)
3. If you find Entry 2 == SL, you MUST fix it:
   - For LONG: Move SL lower or Entry 2 higher (whichever is more technically justified)
   - For SHORT: Move SL higher or Entry 2 lower (whichever is more technically justified)

If original prices are already optimal (at institutional zones, RR ≥ 1:3, Entry 2 ≠ SL), you may omit suggestion fields (they will use original).
If you suggest prices, they will be APPLIED if they improve the trade (better institutional entry, tighter SL, higher TP, better RR).
If setup is weak, counter-trend, or lacks institutional confirmation, MODIFY or DISCARD entirely."""
    
    try:
        # Call Gemini API with timeout
        logger.info(f"🤖 [AI VALIDATION] Starting validation for NEW ENTRY signal: {symbol} ({signal_side}) @ ${entry_price:,.8f}")
        logger.info(f"🤖 [AI VALIDATION] This validation ONLY runs for new ENTRY signals, NOT for order tracking or TP creation")
        
        # Check if gemini_client is available
        if not gemini_client:
            logger.warning("⚠️ Gemini client is None - AI validation will be skipped (fail-open)")
            return {
                'is_valid': True,
                'confidence_score': 100.0,
                'reasoning': 'Gemini client not initialized, proceeding (fail-open)',
                'risk_level': 'MEDIUM'
            }
        
        logger.info(f"📡 Using Gemini model: {gemini_model_name}")
        logger.debug(f"📤 AI PROMPT (full):\n{prompt}")  # Only log at DEBUG level to reduce log size
        start_time = time.time()
        
        # Use threading to implement timeout
        result_container = {'response': None, 'error': None}
        
        def call_api():
            global gemini_client, gemini_model_name
            try:
                # Try current model first
                logger.info(f"📡 Calling Gemini API with model: {gemini_model_name}")
                response = gemini_client.generate_content(prompt)
                result_container['response'] = response.text
                logger.info(f"✅ Gemini API call successful, received response (length: {len(response.text)} chars)")
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"⚠️ Gemini API call failed with model {gemini_model_name}: {error_msg}")
                
                # Check if we should try alternative models
                should_try_alternatives = False
                # Check if model requires billing (limit: 0 means not available in free tier)
                is_paid_model = 'limit: 0' in error_msg.lower() or '2.5-pro' in gemini_model_name.lower()
                
                if 'not found' in error_msg.lower() or 'not supported' in error_msg.lower():
                    logger.warning(f"🔄 Model {gemini_model_name} not available, trying alternative FREE TIER models...")
                    should_try_alternatives = True
                elif is_paid_model:
                    logger.warning(f"🔄 Model {gemini_model_name} requires billing (not free tier), trying alternative FREE TIER models...")
                    should_try_alternatives = True
                elif '429' in error_msg or 'quota' in error_msg.lower() or 'rate limit' in error_msg.lower():
                    logger.warning(f"🔄 Model {gemini_model_name} quota exceeded, trying alternative FREE TIER models with different limits...")
                    should_try_alternatives = True
                
                # Try alternative models if applicable
                if should_try_alternatives:
                    # Try other FREE TIER models in order of preference (skip current model and paid models)
                    paid_models_to_skip = ['2.5-pro', '2.0-pro', 'ultra']
                    for alt_model_name in GEMINI_MODEL_NAMES:
                        if alt_model_name == gemini_model_name:
                            continue
                        # Skip paid models
                        if any(paid in alt_model_name.lower() for paid in paid_models_to_skip):
                            logger.debug(f"⏭️ Skipping paid model: {alt_model_name}")
                            continue
                        try:
                            logger.info(f"🔄 Trying alternative FREE TIER model: {alt_model_name}")
                            alt_client = genai.GenerativeModel(alt_model_name)
                            response = alt_client.generate_content(prompt)
                            result_container['response'] = response.text
                            logger.info(f"✅ Successfully used alternative FREE TIER model: {alt_model_name}")
                            # Update global client for future use
                            gemini_client = alt_client
                            gemini_model_name = alt_model_name
                            return
                        except Exception as alt_e:
                            alt_error = str(alt_e)
                            # Check if this is a paid model error (limit: 0)
                            if 'limit: 0' in alt_error.lower():
                                logger.debug(f"⏭️ Model {alt_model_name} requires billing, skipping...")
                                continue
                            # Check if quota error - but only skip if it's the SAME model family (they share quota)
                            # Different model families have SEPARATE quotas (e.g., gemini-2.5-flash vs gemini-2.5-flash-lite)
                            if '429' in alt_error or 'quota' in alt_error.lower():
                                # Check if it's the same model family (they share quota pool)
                                current_model_family = gemini_model_name.split('-')[1] if '-' in gemini_model_name else ''
                                alt_model_family = alt_model_name.split('-')[1] if '-' in alt_model_name else ''
                                # If same family (e.g., both 2.5-flash), skip (same quota pool)
                                # If different family (e.g., 2.5-flash vs 2.5-flash-lite), try it (different quota)
                                if current_model_family == alt_model_family and 'lite' not in alt_model_name.lower() and '3-flash' not in alt_model_name.lower():
                                    logger.debug(f"⚠️ Model {alt_model_name} shares quota pool with {gemini_model_name}, skipping...")
                                else:
                                    logger.debug(f"⚠️ Model {alt_model_name} has quota issues but different quota pool, will try anyway...")
                                    # Don't skip - different quota pools might work
                            else:
                                logger.debug(f"❌ Alternative model {alt_model_name} also failed: {alt_error}")
                            continue
                
                # If no alternative worked, return original error
                result_container['error'] = error_msg
                logger.error(f"❌ All Gemini models failed. Last error: {error_msg}")
        
        api_thread = threading.Thread(target=call_api, daemon=True)
        api_thread.start()
        
        # Wait for response or error (with safety timeout to prevent infinite hangs)
        # Using 60s safety timeout - should be enough even for slow free tier
        api_thread.join(timeout=60)
        
        # Wait until we get a response or error (or safety timeout)
        while api_thread.is_alive() and not result_container['response'] and not result_container['error']:
            time.sleep(0.1)  # Small sleep to avoid busy-waiting
            # Check if we've exceeded safety timeout (60s)
            elapsed = time.time() - start_time
            if elapsed > 60:
                logger.warning(f"⏱️ AI validation safety timeout for {symbol} (60s), proceeding without validation (fail-open)")
            return {
                'is_valid': True,
                    'confidence_score': 100.0,  # High score to pass threshold - fail-open design
                    'reasoning': 'AI validation safety timeout, proceeding (fail-open)',
                'risk_level': 'MEDIUM'
            }
        
        # Check for error first - handle gracefully with fail-open design
        if result_container['error']:
            error_msg = result_container['error']
            logger.warning(f"⚠️ AI validation error for {symbol}: {error_msg}")
            
            # Check if it's a quota/rate limit error
            if '429' in error_msg or 'quota' in error_msg.lower() or 'rate limit' in error_msg.lower():
                logger.warning(f"⚠️ Gemini API quota/rate limit exceeded for {symbol}. Proceeding without validation (fail-open)")
                return {
                    'is_valid': True,
                    'confidence_score': 100.0,  # High score to pass threshold - fail-open design
                    'reasoning': f'AI validation unavailable (quota exceeded), proceeding (fail-open)',
                    'risk_level': 'MEDIUM'
                }
            
            # For other errors, also fail-open (don't block trading)
            logger.warning(f"⚠️ AI validation error for {symbol}, proceeding without validation (fail-open)")
            return {
                'is_valid': True,
                'confidence_score': 100.0,  # High score to pass threshold - fail-open design
                'reasoning': f'AI validation error: {error_msg[:200]}... Proceeding without validation (fail-open)',
                'risk_level': 'MEDIUM'
            }
        
        # Check for response
        if result_container['response']:
            logger.info(f"✅ Received AI response after {time.time() - start_time:.2f}s")
        else:
            # No response and no error - should not happen, but fail-open
            logger.warning(f"⚠️ No response or error from AI validation for {symbol}, proceeding (fail-open)")
            return {
                'is_valid': True,
                'confidence_score': 100.0,
                'reasoning': 'No response from AI, proceeding (fail-open)',
                'risk_level': 'MEDIUM'
            }
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ AI validation API call completed in {elapsed_time:.2f}s")
        
        # Parse response
        response_text = result_container['response'].strip()
        
        # Log full response for debugging
        logger.debug(f"📥 AI RESPONSE (full):\n{response_text}")  # Only log at DEBUG level to reduce log size
        
        # Try to extract JSON from response (AI might wrap it in markdown or text)
        import re
        # More flexible regex to capture full JSON including nested objects and optional fields
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"is_valid"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        else:
            # Fallback: try to find JSON block with braces
            json_match = re.search(r'\{.*"is_valid".*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
        
        # Parse JSON response
        logger.info(f"📝 Parsing AI response (length: {len(response_text)} chars)")
        try:
            validation_result = json.loads(response_text)
            logger.info(f"✅ Successfully parsed AI response JSON")
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ Failed to parse AI response as JSON: {e}")
            # Try to extract values manually if JSON parsing fails
            logger.warning(f"Failed to parse AI response as JSON, attempting manual extraction")
            is_valid = 'true' in response_text.lower() or '"is_valid": true' in response_text.lower()
            confidence_match = re.search(r'"confidence_score":\s*(\d+(?:\.\d+)?)', response_text)
            confidence_score = float(confidence_match.group(1)) if confidence_match else 50.0
            
            reasoning_match = re.search(r'"reasoning":\s*"([^"]+)"', response_text)
            reasoning = reasoning_match.group(1) if reasoning_match else "AI validation completed"
            
            risk_match = re.search(r'"risk_level":\s*"([^"]+)"', response_text)
            risk_level = risk_match.group(1) if risk_match else "MEDIUM"
            
            # Extract optional price suggestions
            suggested_entry_match = re.search(r'"suggested_entry_price":\s*([\d.]+|null)', response_text)
            suggested_entry = float(suggested_entry_match.group(1)) if suggested_entry_match and suggested_entry_match.group(1) != 'null' else None
            
            suggested_entry2_match = re.search(r'"suggested_second_entry_price":\s*([\d.]+|null)', response_text)
            suggested_entry2 = float(suggested_entry2_match.group(1)) if suggested_entry2_match and suggested_entry2_match.group(1) != 'null' else None
            
            suggested_sl_match = re.search(r'"suggested_stop_loss":\s*([\d.]+|null)', response_text)
            suggested_sl = float(suggested_sl_match.group(1)) if suggested_sl_match and suggested_sl_match.group(1) != 'null' else None
            
            suggested_tp_match = re.search(r'"suggested_take_profit":\s*([\d.]+|null)', response_text)
            suggested_tp = float(suggested_tp_match.group(1)) if suggested_tp_match and suggested_tp_match.group(1) != 'null' else None
            
            price_reasoning_match = re.search(r'"price_suggestion_reasoning":\s*"([^"]+)"', response_text)
            price_reasoning = price_reasoning_match.group(1) if price_reasoning_match else ""
            
            validation_result = {
                'is_valid': is_valid,
                'confidence_score': confidence_score,
                'reasoning': reasoning,
                'risk_level': risk_level
            }
            
            # Add price suggestions if found
            if suggested_entry or suggested_entry2 or suggested_sl or suggested_tp:
                if suggested_entry:
                    validation_result['suggested_entry_price'] = suggested_entry
                if suggested_entry2:
                    validation_result['suggested_second_entry_price'] = suggested_entry2
                if suggested_sl:
                    validation_result['suggested_stop_loss'] = suggested_sl
                if suggested_tp:
                    validation_result['suggested_take_profit'] = suggested_tp
                if price_reasoning:
                    validation_result['price_suggestion_reasoning'] = price_reasoning
        
        # Validate response structure
        if 'is_valid' not in validation_result:
            validation_result['is_valid'] = True  # Fail-open
        
        if 'confidence_score' not in validation_result:
            validation_result['confidence_score'] = 50.0
        
        if 'reasoning' not in validation_result:
            validation_result['reasoning'] = 'AI validation completed'
        
        if 'risk_level' not in validation_result:
            validation_result['risk_level'] = 'MEDIUM'
        
        # Ensure confidence_score is within valid range
        validation_result['confidence_score'] = max(0, min(100, float(validation_result['confidence_score'])))
        
        # Log validation result
        logger.info(f"📊 AI Validation Result:")
        logger.info(f"   ✅ Valid: {validation_result.get('is_valid', True)}")
        logger.info(f"   📈 Confidence: {validation_result['confidence_score']:.1f}%")
        logger.info(f"   ⚠️  Risk Level: {validation_result.get('risk_level', 'MEDIUM')}")
        # Log full reasoning (split into multiple lines if very long for readability)
        reasoning = validation_result.get('reasoning', 'N/A')
        if len(reasoning) > 500:
            # Split long reasoning into multiple log lines
            logger.info(f"   💭 Reasoning (full):")
            # Split by sentences or chunks of 500 chars
            chunks = [reasoning[i:i+500] for i in range(0, len(reasoning), 500)]
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"      [{i}/{len(chunks)}] {chunk}")
        else:
            logger.info(f"   💭 Reasoning: {reasoning}")
        
        # Extract price suggestions and apply smart optimization
        original_entry2 = safe_float(signal_data.get('second_entry_price'), default=None)
        optimized_prices = {
            'entry_price': entry_price,  # Default to original
            'stop_loss': stop_loss,      # Default to original
            'take_profit': take_profit,  # Default to original
            'second_entry_price': original_entry2,   # Default to original, will be optimized if AI suggests or Entry 1 is optimized
            'applied_optimizations': []
        }
        
        if ENABLE_AI_PRICE_SUGGESTIONS:
            suggested_entry = safe_float(validation_result.get('suggested_entry_price'), default=None)
            suggested_entry2 = safe_float(validation_result.get('suggested_second_entry_price'), default=None)
            suggested_sl = safe_float(validation_result.get('suggested_stop_loss'), default=None)
            suggested_tp = safe_float(validation_result.get('suggested_take_profit'), default=None)
            price_reasoning = validation_result.get('price_suggestion_reasoning', '')
            original_entry2 = safe_float(signal_data.get('second_entry_price'), default=None)
            
            # Smart optimization logic: Only apply if AI suggestion is BETTER
            if suggested_entry:
                if signal_side == 'LONG':
                    # For LONG: Lower entry is better (closer to support)
                    if suggested_entry < entry_price:
                        optimized_prices['entry_price'] = suggested_entry
                        optimized_prices['applied_optimizations'].append(f"Entry optimized: ${entry_price:,.8f} → ${suggested_entry:,.8f} (better entry for LONG)")
                        logger.info(f"✅ [AI OPTIMIZATION] Entry optimized for LONG: ${entry_price:,.8f} → ${suggested_entry:,.8f} (better entry)")
                    else:
                        logger.info(f"⚠️  [AI OPTIMIZATION] Entry suggestion ${suggested_entry:,.8f} is HIGHER than original ${entry_price:,.8f} - keeping original (better for LONG)")
                else:  # SHORT
                    # For SHORT: Higher entry is better (closer to resistance)
                    if suggested_entry > entry_price:
                        optimized_prices['entry_price'] = suggested_entry
                        optimized_prices['applied_optimizations'].append(f"Entry optimized: ${entry_price:,.8f} → ${suggested_entry:,.8f} (better entry for SHORT)")
                        logger.info(f"✅ [AI OPTIMIZATION] Entry optimized for SHORT: ${entry_price:,.8f} → ${suggested_entry:,.8f} (better entry)")
                    else:
                        logger.info(f"⚠️  [AI OPTIMIZATION] Entry suggestion ${suggested_entry:,.8f} is LOWER than original ${entry_price:,.8f} - keeping original (better for SHORT)")
            
            # Entry 2 optimization (independent of Entry 1)
            if suggested_entry2 and original_entry2:
                # Validate Entry 2 is in correct direction relative to Entry 1
                current_entry1 = optimized_prices.get('entry_price', entry_price)
                
                if signal_side == 'LONG':
                    # For LONG: Entry 2 should be BELOW Entry 1
                    if suggested_entry2 < current_entry1:
                        # Validate it's better than original Entry 2 (lower is better for LONG)
                        if suggested_entry2 < original_entry2:
                            optimized_prices['second_entry_price'] = suggested_entry2
                            optimized_prices['applied_optimizations'].append(f"Entry 2 optimized: ${original_entry2:,.8f} → ${suggested_entry2:,.8f} (better entry for LONG)")
                            logger.info(f"✅ [AI OPTIMIZATION] Entry 2 optimized for LONG: ${original_entry2:,.8f} → ${suggested_entry2:,.8f} (better entry)")
                        elif suggested_entry2 == original_entry2:
                            # Same as original - keep it
                            optimized_prices['second_entry_price'] = suggested_entry2
                            logger.info(f"ℹ️  [AI OPTIMIZATION] Entry 2 suggestion ${suggested_entry2:,.8f} matches original - keeping it")
                        else:
                            logger.info(f"⚠️  [AI OPTIMIZATION] Entry 2 suggestion ${suggested_entry2:,.8f} is HIGHER than original ${original_entry2:,.8f} - keeping original (better for LONG)")
                    else:
                        logger.info(f"⚠️  [AI OPTIMIZATION] Entry 2 suggestion ${suggested_entry2:,.8f} is ABOVE Entry 1 ${current_entry1:,.8f} - invalid for LONG, keeping original")
                else:  # SHORT
                    # For SHORT: Entry 2 should be ABOVE Entry 1
                    if suggested_entry2 > current_entry1:
                        # Validate it's better than original Entry 2 (higher is better for SHORT)
                        if suggested_entry2 > original_entry2:
                            optimized_prices['second_entry_price'] = suggested_entry2
                            optimized_prices['applied_optimizations'].append(f"Entry 2 optimized: ${original_entry2:,.8f} → ${suggested_entry2:,.8f} (better entry for SHORT)")
                            logger.info(f"✅ [AI OPTIMIZATION] Entry 2 optimized for SHORT: ${original_entry2:,.8f} → ${suggested_entry2:,.8f} (better entry)")
                        elif suggested_entry2 == original_entry2:
                            # Same as original - keep it
                            optimized_prices['second_entry_price'] = suggested_entry2
                            logger.info(f"ℹ️  [AI OPTIMIZATION] Entry 2 suggestion ${suggested_entry2:,.8f} matches original - keeping it")
                        else:
                            logger.info(f"⚠️  [AI OPTIMIZATION] Entry 2 suggestion ${suggested_entry2:,.8f} is LOWER than original ${original_entry2:,.8f} - keeping original (better for SHORT)")
                    else:
                        logger.info(f"⚠️  [AI OPTIMIZATION] Entry 2 suggestion ${suggested_entry2:,.8f} is BELOW Entry 1 ${current_entry1:,.8f} - invalid for SHORT, keeping original")
            elif suggested_entry2 and not original_entry2:
                # AI suggests Entry 2 but original signal didn't have one - use AI suggestion if valid
                current_entry1 = optimized_prices.get('entry_price', entry_price)
                if signal_side == 'LONG' and suggested_entry2 < current_entry1:
                    optimized_prices['second_entry_price'] = suggested_entry2
                    optimized_prices['applied_optimizations'].append(f"Entry 2 added: ${suggested_entry2:,.8f} (AI suggested, original was missing)")
                    logger.info(f"✅ [AI OPTIMIZATION] Entry 2 added for LONG: ${suggested_entry2:,.8f} (original was missing)")
                elif signal_side == 'SHORT' and suggested_entry2 > current_entry1:
                    optimized_prices['second_entry_price'] = suggested_entry2
                    optimized_prices['applied_optimizations'].append(f"Entry 2 added: ${suggested_entry2:,.8f} (AI suggested, original was missing)")
                    logger.info(f"✅ [AI OPTIMIZATION] Entry 2 added for SHORT: ${suggested_entry2:,.8f} (original was missing)")
            
            if suggested_sl and stop_loss:
                if signal_side == 'LONG':
                    # For LONG: Tighter SL (higher) is better, but must stay below entry
                    if suggested_sl > stop_loss and suggested_sl < optimized_prices['entry_price']:
                        optimized_prices['stop_loss'] = suggested_sl
                        optimized_prices['applied_optimizations'].append(f"SL optimized: ${stop_loss:,.8f} → ${suggested_sl:,.8f} (tighter risk)")
                        logger.info(f"✅ [AI OPTIMIZATION] SL optimized for LONG: ${stop_loss:,.8f} → ${suggested_sl:,.8f} (tighter)")
                    elif suggested_sl < stop_loss:
                        # Even tighter SL - apply if safe
                        optimized_prices['stop_loss'] = suggested_sl
                        optimized_prices['applied_optimizations'].append(f"SL optimized: ${stop_loss:,.8f} → ${suggested_sl:,.8f} (tighter risk)")
                        logger.info(f"✅ [AI OPTIMIZATION] SL optimized for LONG: ${stop_loss:,.8f} → ${suggested_sl:,.8f} (tighter)")
                    else:
                        logger.info(f"⚠️  [AI OPTIMIZATION] SL suggestion ${suggested_sl:,.8f} rejected (wider than original or invalid)")
                else:  # SHORT
                    # For SHORT: Tighter SL (lower) is better, but must stay above entry
                    if suggested_sl < stop_loss and suggested_sl > optimized_prices['entry_price']:
                        optimized_prices['stop_loss'] = suggested_sl
                        optimized_prices['applied_optimizations'].append(f"SL optimized: ${stop_loss:,.8f} → ${suggested_sl:,.8f} (tighter risk)")
                        logger.info(f"✅ [AI OPTIMIZATION] SL optimized for SHORT: ${stop_loss:,.8f} → ${suggested_sl:,.8f} (tighter)")
                    elif suggested_sl > stop_loss:
                        # Even tighter SL - apply if safe
                        optimized_prices['stop_loss'] = suggested_sl
                        optimized_prices['applied_optimizations'].append(f"SL optimized: ${stop_loss:,.8f} → ${suggested_sl:,.8f} (tighter risk)")
                        logger.info(f"✅ [AI OPTIMIZATION] SL optimized for SHORT: ${stop_loss:,.8f} → ${suggested_sl:,.8f} (tighter)")
                    else:
                        logger.info(f"⚠️  [AI OPTIMIZATION] SL suggestion ${suggested_sl:,.8f} rejected (wider than original or invalid)")
            
            if suggested_tp and take_profit:
                # Smart TP optimization: Use AI's analysis to determine BEST TP
                # AI analyzes resistance/support levels, reversal risk, and realistic targets
                # Trust AI's analysis - it considers if original TP might miss by 0.1-0.5% due to reversal
                tp_diff_pct = abs((suggested_tp - take_profit) / take_profit * 100) if take_profit else 0
                
                if signal_side == 'LONG':
                    # For LONG: AI analyzes resistance levels to find BEST TP
                    # Apply AI TP if:
                    # 1. Higher (more profit) OR
                    # 2. Better positioned at resistance (even if slightly lower, within 2% - avoids reversal risk)
                    if suggested_tp > take_profit:
                        # Higher TP = more profit, apply it
                        optimized_prices['take_profit'] = suggested_tp
                        optimized_prices['applied_optimizations'].append(f"TP optimized: ${take_profit:,.8f} → ${suggested_tp:,.8f} (higher profit, +{tp_diff_pct:.2f}%)")
                        logger.info(f"✅ [AI OPTIMIZATION] TP optimized for LONG: ${take_profit:,.8f} → ${suggested_tp:,.8f} (higher profit, +{tp_diff_pct:.2f}%)")
                    elif suggested_tp >= take_profit * 0.98:  # Within 2% of original (AI found better resistance level, avoids reversal)
                        # AI TP is slightly lower but better positioned at resistance - use it (avoids missing TP by 0.1-0.5%)
                        optimized_prices['take_profit'] = suggested_tp
                        optimized_prices['applied_optimizations'].append(f"TP optimized: ${take_profit:,.8f} → ${suggested_tp:,.8f} (better resistance level, avoids reversal, -{tp_diff_pct:.2f}%)")
                        logger.info(f"✅ [AI OPTIMIZATION] TP optimized for LONG: ${take_profit:,.8f} → ${suggested_tp:,.8f} (better positioned at resistance, avoids reversal risk, -{tp_diff_pct:.2f}%)")
                    else:
                        # AI TP is significantly lower (>2%) - keep original (it's better)
                        logger.info(f"⚠️  [AI OPTIMIZATION] TP suggestion ${suggested_tp:,.8f} is {tp_diff_pct:.2f}% LOWER than original ${take_profit:,.8f} - keeping original (better profit)")
                else:  # SHORT
                    # For SHORT: AI analyzes support levels to find BEST TP
                    # Apply AI TP if:
                    # 1. Lower (more profit) OR
                    # 2. Better positioned at support (even if slightly higher, within 2% - avoids reversal risk)
                    if suggested_tp < take_profit:
                        # Lower TP = more profit, apply it
                        optimized_prices['take_profit'] = suggested_tp
                        optimized_prices['applied_optimizations'].append(f"TP optimized: ${take_profit:,.8f} → ${suggested_tp:,.8f} (higher profit, -{tp_diff_pct:.2f}%)")
                        logger.info(f"✅ [AI OPTIMIZATION] TP optimized for SHORT: ${take_profit:,.8f} → ${suggested_tp:,.8f} (higher profit, -{tp_diff_pct:.2f}%)")
                    elif suggested_tp <= take_profit * 1.02:  # Within 2% of original (AI found better support level, avoids reversal)
                        # AI TP is slightly higher but better positioned at support - use it (avoids missing TP by 0.1-0.5%)
                        optimized_prices['take_profit'] = suggested_tp
                        optimized_prices['applied_optimizations'].append(f"TP optimized: ${take_profit:,.8f} → ${suggested_tp:,.8f} (better support level, avoids reversal, +{tp_diff_pct:.2f}%)")
                        logger.info(f"✅ [AI OPTIMIZATION] TP optimized for SHORT: ${take_profit:,.8f} → ${suggested_tp:,.8f} (better positioned at support, avoids reversal risk, +{tp_diff_pct:.2f}%)")
                    else:
                        # AI TP is significantly higher (>2%) - keep original (it's better)
                        logger.info(f"⚠️  [AI OPTIMIZATION] TP suggestion ${suggested_tp:,.8f} is {tp_diff_pct:.2f}% HIGHER than original ${take_profit:,.8f} - keeping original (better profit)")
            
            # Store suggestions for logging (even if not applied)
            if suggested_entry or suggested_entry2 or suggested_sl or suggested_tp:
                validation_result['price_suggestions'] = {
                    'entry_price': suggested_entry,
                    'second_entry_price': suggested_entry2,
                    'stop_loss': suggested_sl,
                    'take_profit': suggested_tp,
                    'reasoning': price_reasoning,
                    'original_entry': entry_price,
                    'original_second_entry': original_entry2,
                    'original_stop_loss': stop_loss,
                    'original_take_profit': take_profit,
                    'optimized_entry': optimized_prices['entry_price'],
                    'optimized_stop_loss': optimized_prices['stop_loss'],
                    'optimized_take_profit': optimized_prices['take_profit'],
                    'applied_optimizations': optimized_prices['applied_optimizations']
                }
                
                # Log comparison with applied optimizations
                logger.info(f"💡 [AI PRICE OPTIMIZATION] Analysis for {symbol}:")
                logger.info(f"   ┌─────────────────────────────────────────────────────────────────────────────┐")
                logger.info(f"   │ PRICE COMPARISON: Original (TradingView) vs AI Suggested vs Applied       │")
                logger.info(f"   ├─────────────────────────────────────────────────────────────────────────────┤")
                if suggested_entry:
                    diff_pct = ((suggested_entry - entry_price) / entry_price * 100) if entry_price else 0
                    applied = optimized_prices['entry_price']
                    applied_diff = ((applied - entry_price) / entry_price * 100) if entry_price else 0
                    status = "✅ APPLIED" if applied != entry_price else "❌ REJECTED (worse)"
                    logger.info(f"   │ 📍 Entry:      Original=${entry_price:,.8f}  →  AI=${suggested_entry:,.8f} ({diff_pct:+.2f}%)  →  Applied=${applied:,.8f} ({applied_diff:+.2f}%) {status} │")
                else:
                    logger.info(f"   │ 📍 Entry:      Original=${entry_price:,.8f}  →  No AI suggestion (keeping original)                    │")
                if suggested_entry2:
                    if original_entry2:
                        diff_pct = ((suggested_entry2 - original_entry2) / original_entry2 * 100) if original_entry2 else 0
                        applied = optimized_prices.get('second_entry_price', original_entry2)
                        applied_diff = ((applied - original_entry2) / original_entry2 * 100) if original_entry2 else 0
                        status = "✅ APPLIED" if applied != original_entry2 else "❌ REJECTED (worse)"
                        logger.info(f"   │ 📍 Entry 2:    Original=${original_entry2:,.8f}  →  AI=${suggested_entry2:,.8f} ({diff_pct:+.2f}%)  →  Applied=${applied:,.8f} ({applied_diff:+.2f}%) {status} │")
                    else:
                        applied = optimized_prices.get('second_entry_price')
                        status = "✅ APPLIED" if applied else "❌ REJECTED"
                        logger.info(f"   │ 📍 Entry 2:    Original=N/A (not provided)  →  AI=${suggested_entry2:,.8f}  →  Applied=${applied:,.8f if applied else 'N/A'} {status} │")
                elif original_entry2:
                    logger.info(f"   │ 📍 Entry 2:    Original=${original_entry2:,.8f}  →  No AI suggestion (keeping original)                    │")
                if suggested_sl:
                    diff_pct = ((suggested_sl - stop_loss) / stop_loss * 100) if stop_loss else 0
                    applied = optimized_prices['stop_loss']
                    applied_diff = ((applied - stop_loss) / stop_loss * 100) if stop_loss else 0
                    status = "✅ APPLIED" if applied != stop_loss else "❌ REJECTED"
                    logger.info(f"   │ 🛑 Stop Loss:   Original=${stop_loss:,.8f}  →  AI=${suggested_sl:,.8f} ({diff_pct:+.2f}%)  →  Applied=${applied:,.8f} ({applied_diff:+.2f}%) {status} │")
                else:
                    sl_display = f"${stop_loss:,.8f}" if stop_loss else "N/A"
                    logger.info(f"   │ 🛑 Stop Loss:   Original={sl_display:<15}  →  No AI suggestion (keeping original)                    │")
                if suggested_tp:
                    diff_pct = ((suggested_tp - take_profit) / take_profit * 100) if take_profit else 0
                    applied = optimized_prices['take_profit']
                    applied_diff = ((applied - take_profit) / take_profit * 100) if take_profit else 0
                    status = "✅ APPLIED" if applied != take_profit else "❌ REJECTED (worse)"
                    logger.info(f"   │ 🎯 Take Profit: Original=${take_profit:,.8f}  →  AI=${suggested_tp:,.8f} ({diff_pct:+.2f}%)  →  Applied=${applied:,.8f} ({applied_diff:+.2f}%) {status} │")
                else:
                    tp_display = f"${take_profit:,.8f}" if take_profit else "N/A"
                    logger.info(f"   │ 🎯 Take Profit: Original={tp_display:<15}  →  No AI suggestion (keeping original)                    │")
                logger.info(f"   ├─────────────────────────────────────────────────────────────────────────────┤")
                if price_reasoning:
                    reasoning_lines = [price_reasoning[i:i+70] for i in range(0, len(price_reasoning), 70)]
                    for line in reasoning_lines:
                        logger.info(f"   │ 💭 AI Reasoning: {line:<70} │")
                if optimized_prices['applied_optimizations']:
                    logger.info(f"   │ ✅ APPLIED OPTIMIZATIONS: {len(optimized_prices['applied_optimizations'])} price(s) optimized          │")
                    for opt in optimized_prices['applied_optimizations']:
                        logger.info(f"   │    • {opt:<68} │")
                else:
                    logger.info(f"   │ ⚠️  No optimizations applied (AI suggestions were worse than original)      │")
                logger.info(f"   └─────────────────────────────────────────────────────────────────────────────┘")
        
        # Store optimized prices in validation result for use in order creation
        validation_result['optimized_prices'] = optimized_prices
        
        # Cache the result
        validation_cache[cache_key] = (validation_result, current_time)
        
        # Clean up old cache entries (keep last 100)
        if len(validation_cache) > 100:
            sorted_cache = sorted(validation_cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_cache[:-100]:
                del validation_cache[key]
        
        logger.info(f"✅ AI Validation Result for {symbol}: Valid={validation_result['is_valid']}, "
                   f"Confidence={validation_result['confidence_score']:.1f}%, "
                   f"Risk={validation_result['risk_level']}, "
                   f"Reasoning={validation_result['reasoning']}")
        
        return validation_result
        
    except Exception as e:
        logger.warning(f"⚠️ AI validation error for {symbol}: {e}. Proceeding without validation (fail-open)")
        # Return high confidence score to ensure signal passes threshold (fail-open design)
        # This ensures API errors don't block legitimate trading signals
        return {
            'is_valid': True,  # Fail-open: proceed if validation fails
            'confidence_score': 100.0,  # High score to pass threshold - fail-open design
            'reasoning': f'AI validation error: {str(e)}, proceeding (fail-open)',
            'risk_level': 'MEDIUM',
            'error': str(e)
        }


def create_limit_order(signal_data):
    """Create a Binance Futures limit order with configurable entry size and leverage"""
    try:
        # Extract signal data
        token = signal_data.get('token')
        event = signal_data.get('event')
        signal_side = signal_data.get('signal_side')
        symbol = format_symbol(signal_data.get('symbol', ''))
        
        # Safely parse prices (handles None, "null" string, and invalid values)
        entry_price = safe_float(signal_data.get('entry_price'), default=None)
        stop_loss = safe_float(signal_data.get('stop_loss'), default=None)
        take_profit = safe_float(signal_data.get('take_profit'), default=None)
        second_entry_price = safe_float(signal_data.get('second_entry_price'), default=None)
        
        # Extract indicators if available (for ATR-based Entry 2 calculation)
        indicators = signal_data.get('indicators', {})
        
        reduce_only = signal_data.get('reduce_only', False)
        order_subtype = signal_data.get('order_subtype', 'primary_entry')
        
        # Verify token
        if not verify_webhook_token(token):
            logger.warning(f"Invalid webhook token received")
            return {'success': False, 'error': 'Invalid token'}
        
        # Validate required fields for ENTRY events
        if event == 'ENTRY':
            if entry_price is None or entry_price <= 0:
                logger.warning(f"Invalid or missing entry_price in webhook payload. Discarding request. entry_price={signal_data.get('entry_price')}")
                return {'success': False, 'error': 'Invalid or missing entry_price (NA/null)'}
            
            # AI Signal Validation (ONLY for NEW ENTRY signals - NOT for order tracking or TP creation)
            logger.info(f"🔍 [AI VALIDATION] Processing NEW ENTRY signal for {symbol} - AI validation will run")
            validation_result = validate_signal_with_ai(signal_data)
            
            # Check if signal is valid and meets confidence threshold
            if not validation_result.get('is_valid', True):
                rejection_reason = validation_result.get('reasoning', 'Signal validation failed - AI determined signal is invalid')
                logger.warning(f"🚫 AI Validation REJECTED signal for {symbol}: {rejection_reason}")
                
                # Send rejection notification to Slack exception channel
                send_signal_rejection_notification(
                    symbol=symbol,
                    signal_side=signal_side,
                    timeframe=timeframe,
                    entry_price=entry_price,
                    rejection_reason=f"Signal validation failed - AI determined signal is invalid.\n\n{rejection_reason}",
                    confidence_score=validation_result.get('confidence_score'),
                    risk_level=validation_result.get('risk_level'),
                    validation_result=validation_result
                )
                
                return {
                    'success': False,
                    'error': 'Signal validation failed',
                    'validation_result': validation_result
                }
            
            confidence_score = validation_result.get('confidence_score', 100.0)
            if confidence_score < AI_VALIDATION_MIN_CONFIDENCE:
                rejection_reason = f"Confidence score {confidence_score:.1f}% is below minimum threshold of {AI_VALIDATION_MIN_CONFIDENCE}%"
                logger.warning(f"🚫 AI Validation REJECTED signal for {symbol}: {rejection_reason}")
                logger.info(f"   Reasoning: {validation_result.get('reasoning', 'No reasoning provided')}")
                logger.info(f"   Risk Level: {validation_result.get('risk_level', 'UNKNOWN')}")
                
                # Send rejection notification to Slack exception channel
                full_reason = f"{rejection_reason}\n\n{validation_result.get('reasoning', 'No detailed reasoning provided')}"
                send_signal_rejection_notification(
                    symbol=symbol,
                    signal_side=signal_side,
                    timeframe=timeframe,
                    entry_price=entry_price,
                    rejection_reason=full_reason,
                    confidence_score=confidence_score,
                    risk_level=validation_result.get('risk_level'),
                    validation_result=validation_result
                )
                
                return {
                    'success': False,
                    'error': f'Signal confidence {confidence_score:.1f}% below minimum {AI_VALIDATION_MIN_CONFIDENCE}%',
                    'validation_result': validation_result
                }
            
            # Log successful validation
            logger.info(f"✅ AI Validation APPROVED signal for {symbol}: Confidence={confidence_score:.1f}%, "
                       f"Risk={validation_result.get('risk_level', 'UNKNOWN')}, "
                       f"Reasoning={validation_result.get('reasoning', 'No reasoning')}")
            
            # Apply optimized prices if available
            if 'optimized_prices' in validation_result:
                opt_prices = validation_result['optimized_prices']
                # Update prices with optimized values
                if opt_prices.get('entry_price') and opt_prices['entry_price'] != entry_price:
                    entry_price = opt_prices['entry_price']
                    logger.info(f"🔄 [PRICE UPDATE] Using AI-optimized entry price: ${entry_price:,.8f}")
                if opt_prices.get('stop_loss') and opt_prices['stop_loss'] != stop_loss:
                    stop_loss = opt_prices['stop_loss']
                    logger.info(f"🔄 [PRICE UPDATE] Using AI-optimized stop loss: ${stop_loss:,.8f}")
                if opt_prices.get('take_profit') and opt_prices['take_profit'] != take_profit:
                    take_profit = opt_prices['take_profit']
                    logger.info(f"🔄 [PRICE UPDATE] Using AI-optimized take profit: ${take_profit:,.8f}")
                
                # Calculate optimized Entry 2 (DCA) if Entry 1 was optimized
                # Keep Entry 2 close to original Entry 2, maintaining relative spacing from Entry 1
                if opt_prices.get('entry_price') and opt_prices['entry_price'] != safe_float(signal_data.get('entry_price'), default=entry_price):
                    original_entry = safe_float(signal_data.get('entry_price'), default=entry_price)
                    original_entry2 = safe_float(signal_data.get('second_entry_price'), default=None)
                    
                    if original_entry2 and original_entry:
                        # Calculate the ORIGINAL spacing between Entry 1 and Entry 2
                        if signal_side == 'LONG':
                            original_spacing = original_entry - original_entry2  # Entry 2 is below Entry 1
                        else:  # SHORT
                            original_spacing = original_entry2 - original_entry  # Entry 2 is above Entry 1
                        
                        # Calculate original spacing as percentage
                        original_spacing_pct = (original_spacing / original_entry) * 100 if original_entry > 0 else 0
                        
                        # Apply the SAME spacing percentage to the optimized Entry 1
                        # This keeps Entry 2 close to its original position relative to Entry 1
                        if signal_side == 'LONG':
                            optimized_entry2 = opt_prices['entry_price'] * (1 - original_spacing_pct / 100)
                        else:  # SHORT
                            optimized_entry2 = opt_prices['entry_price'] * (1 + original_spacing_pct / 100)
                        
                        opt_prices['second_entry_price'] = optimized_entry2
                        logger.info(f"🔄 [PRICE UPDATE] Calculated optimized Entry 2: ${optimized_entry2:,.8f} (maintaining original {original_spacing_pct:.2f}% spacing from optimized Entry 1, original Entry 2 was ${original_entry2:,.8f})")
        
        # Handle EXIT events - close position at market price and cancel all orders for symbol
        if event == 'EXIT':
            logger.info(f"Processing EXIT event for {symbol} (AI validation skipped for EXIT events)")
            
            # Check for duplicate EXIT processing
            current_time = time.time()
            if symbol in recent_exits:
                if current_time - recent_exits[symbol] < EXIT_COOLDOWN:
                    logger.warning(f"EXIT event for {symbol} already processed recently. Discarding duplicate EXIT alert.")
                    return {'success': False, 'error': 'EXIT already processed recently'}
            
            # Detect position mode
            is_hedge_mode = get_position_mode(symbol)
            
            # Cancel ALL orders for this symbol first (even if entry didn't fill)
            logger.info(f"Canceling all orders for {symbol}")
            canceled_count = cancel_all_limit_orders(symbol)
            logger.info(f"Canceled {canceled_count} limit orders for {symbol}")
            
            # Also cancel any TP/SL orders
            try:
                open_orders = client.futures_get_open_orders(symbol=symbol)
                for order in open_orders:
                    if order.get('type') in ['TAKE_PROFIT_MARKET', 'STOP_MARKET']:
                        try:
                            client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
                            logger.info(f"Canceled {order.get('type')} order {order['orderId']} for {symbol}")
                        except Exception as e:
                            logger.warning(f"Failed to cancel {order.get('type')} order: {e}")
            except Exception as e:
                logger.warning(f"Error canceling TP/SL orders: {e}")
            
            # Check for ANY open position (don't filter by signal_side - close all positions)
            # Get ALL positions first, then filter by symbol (more reliable than filtering in API call)
            try:
                all_positions = client.futures_position_information()
                positions_to_close = []
                
                for position in all_positions:
                    position_symbol = position.get('symbol', '')
                    position_amt = float(position.get('positionAmt', 0))
                    
                    # Match symbol (case-insensitive)
                    if position_symbol.upper() == symbol.upper() and abs(position_amt) > 0:
                        positions_to_close.append(position)
                        logger.info(f"Found open position for {symbol}: {position_amt} @ {position.get('entryPrice')} (side: {position.get('positionSide', 'BOTH')})")
                
                if positions_to_close:
                    # Get symbol info for quantity precision (once, outside loop)
                    exchange_info = client.futures_exchange_info()
                    symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
                    step_size = None
                    if symbol_info:
                        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                        if lot_size_filter:
                            step_size = float(lot_size_filter['stepSize'])
                    
                    # Close ALL positions for this symbol
                    logger.info(f"Closing {len(positions_to_close)} position(s) at market price for {symbol}")
                    for position in positions_to_close:
                        position_amt = float(position.get('positionAmt', 0))
                        position_side_binance = position.get('positionSide', 'BOTH')
                        
                        # Determine close side (opposite of position)
                        close_side = 'SELL' if position_amt > 0 else 'BUY'
                        close_quantity = abs(position_amt)
                        
                        # Format quantity precision if step_size available
                        if step_size:
                            close_quantity = format_quantity_precision(close_quantity, step_size)
                        
                        # Create market order to close position
                        # Try with reduceOnly first, then without if it fails (some accounts/modes don't accept it)
                        close_params = {
                            'symbol': symbol,
                            'side': close_side,
                            'type': 'MARKET',
                            'quantity': close_quantity,
                        }
                        
                        # Only include positionSide if in Hedge mode
                        if is_hedge_mode and position_side_binance != 'BOTH':
                            close_params['positionSide'] = position_side_binance
                        
                        # Try with reduceOnly first (for safety)
                        close_params_with_reduce = close_params.copy()
                        close_params_with_reduce['reduceOnly'] = True
                        
                        logger.info(f"Closing position at market for {symbol}: {close_params_with_reduce}")
                        try:
                            result = client.futures_create_order(**close_params_with_reduce)
                            logger.info(f"✅ Position closed successfully: {result}")
                        except BinanceAPIException as e:
                            # If reduceOnly is not accepted, try without it
                            if e.code == -1106 or 'reduceonly' in str(e).lower():
                                logger.warning(f"reduceOnly not accepted for {symbol}, retrying without it: {e}")
                                try:
                                    result = client.futures_create_order(**close_params)
                                    logger.info(f"✅ Position closed successfully (without reduceOnly): {result}")
                                except Exception as e2:
                                    logger.error(f"❌ Failed to close position (retry without reduceOnly): {e2}", exc_info=True)
                            else:
                                logger.error(f"❌ Failed to close position: {e}", exc_info=True)
                                send_slack_alert(
                                    error_type="Position Close Failed (EXIT)",
                                    message=str(e),
                                    details={'Position_Amount': position_amt, 'Close_Side': close_side},
                                    symbol=symbol,
                                    severity='ERROR'
                                )
                        except Exception as e:
                            logger.error(f"❌ Failed to close position: {e}", exc_info=True)
                            send_slack_alert(
                                error_type="Position Close Error (EXIT)",
                                message=str(e),
                                details={'Position_Amount': position_amt},
                                symbol=symbol,
                                severity='ERROR'
                            )
                else:
                    logger.info(f"No open position found for {symbol} - position may already be closed or entry never filled")
            except Exception as e:
                logger.error(f"Error checking/closing positions for {symbol}: {e}", exc_info=True)
                send_slack_alert(
                    error_type="EXIT Position Close Error",
                    message=str(e),
                    symbol=symbol,
                    severity='ERROR'
                )
            
            # Clean up tracking
            if symbol in active_trades:
                logger.info(f"Cleaning up closed trade tracking for {symbol}")
                del active_trades[symbol]
            
            # Track EXIT processing
            recent_exits[symbol] = current_time
            
            # Clean up old exit tracking
            if len(recent_exits) > 1000:
                sorted_exits = sorted(recent_exits.items(), key=lambda x: x[1])
                for key, _ in sorted_exits[:-1000]:
                    del recent_exits[key]
            
            return {'success': True, 'message': 'EXIT event processed'}
        
        # Only process ENTRY events
        if event != 'ENTRY':
            logger.info(f"Ignoring event: {event}")
            return {'success': False, 'error': f'Event {event} not processed'}
        
        # Get order side
        side = get_order_side(signal_side)
        position_side = get_position_side(signal_side)
        
        # Detect position mode (One-Way vs Hedge)
        is_hedge_mode = get_position_mode(symbol)
        
        # Determine which entry this is
        is_primary_entry = (order_subtype == 'primary_entry' or order_subtype not in ['dca_fill', 'second_entry'])
        is_dca_entry = (order_subtype == 'dca_fill' or order_subtype == 'second_entry')
        
        # Check for existing position and orders BEFORE processing
        has_orders, open_orders = check_existing_orders(symbol)
        has_position, position_info = check_existing_position(symbol, signal_side)
        
        # IMPORTANT: Prevent duplicate alerts when position is already open and both orders are filled
        # This handles the case where TradingView sends duplicate DCA alerts when price returns to DCA level
        if has_position:
            # Simple check: If position exists and there are no pending LIMIT orders (only TP/SL orders may exist),
            # then both entry orders are already filled - this is a duplicate alert
            pending_limit_orders = [o for o in open_orders if o.get('type') == 'LIMIT']
            
            if len(pending_limit_orders) == 0:
                # No pending limit orders = both entry orders are filled
                logger.warning(f"⚠️ Duplicate alert ignored: Position already open for {symbol} and no pending limit orders found. Both entry orders are already filled. This is likely a duplicate DCA alert from TradingView.")
                return {
                    'success': False, 
                    'error': 'Duplicate alert ignored - position already open with both orders filled',
                    'message': f'Position exists for {symbol} and both primary/DCA orders are already filled (no pending limit orders). Ignoring duplicate alert.'
                }
            
            # Additional check: Verify by checking order status in active_trades
            # This is a secondary verification method
            if symbol in active_trades:
                trade_info = active_trades[symbol]
                primary_order_id = trade_info.get('primary_order_id')
                dca_order_id = trade_info.get('dca_order_id')
                
                # Check if both orders exist and are filled
                if primary_order_id and dca_order_id:
                    try:
                        primary_order = client.futures_get_order(symbol=symbol, orderId=primary_order_id)
                        dca_order = client.futures_get_order(symbol=symbol, orderId=dca_order_id)
                        
                        if (primary_order.get('status') == 'FILLED' and 
                            dca_order.get('status') == 'FILLED'):
                            logger.warning(f"⚠️ Duplicate alert ignored: Both orders confirmed as FILLED on Binance for {symbol}. Ignoring duplicate alert.")
                            return {
                                'success': False, 
                                'error': 'Duplicate alert ignored - both orders already filled',
                                'message': f'Both primary and DCA orders are already FILLED for {symbol}. Ignoring duplicate alert.'
                            }
                    except Exception as e:
                        logger.debug(f"Could not verify order status for duplicate check: {e}")
                        # Continue processing if we can't verify (fallback to pending orders check above)
        
        # IMPORTANT: When new trade alert comes for any symbol, close old orders for that symbol first
        # (Only if position doesn't exist or orders aren't filled)
        logger.info(f"Checking for existing orders/positions for {symbol} - will close old ones if found")
        
        # If position exists but orders aren't both filled, close it (new trade replaces old trade)
        if has_position:
            logger.info(f"Existing position found for {symbol}. Closing it before creating new trade.")
            close_result = close_position_at_market(symbol, signal_side, is_hedge_mode)
            if close_result.get('success'):
                logger.info(f"Old position closed successfully")
            else:
                logger.warning(f"Failed to close old position: {close_result.get('error')}")
        
        # Cancel all existing orders for this symbol (old trade orders)
        if has_orders:
            logger.info(f"Found {len(open_orders)} existing orders for {symbol}. Canceling all orders.")
            canceled_count = cancel_all_limit_orders(symbol)
            # Also cancel TP/SL orders
            for order in open_orders:
                if order.get('type') in ['TAKE_PROFIT_MARKET', 'STOP_MARKET']:
                    try:
                        client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
                        logger.info(f"Canceled {order.get('type')} order {order['orderId']}")
                    except Exception as e:
                        logger.warning(f"Failed to cancel {order.get('type')} order: {e}")
            logger.info(f"Canceled {canceled_count} limit orders for {symbol}")
            
            # Wait a moment for Binance to process cancellations
            time.sleep(0.5)
            
            # Re-check orders to ensure they're canceled (prevent false duplicate detection)
            has_orders_after_cancel, _ = check_existing_orders(symbol)
            if has_orders_after_cancel:
                logger.warning(f"Some orders still exist after cancellation. Waiting longer...")
                time.sleep(1.0)
                # Try canceling again
                cancel_all_limit_orders(symbol)
        
        # Clean up old tracking (already done above, but ensure it's clean)
        if symbol in active_trades:
            logger.info(f"Cleaning up old trade tracking for {symbol}")
            del active_trades[symbol]
        
        # Get primary entry price - use optimized entry_price if available, otherwise original
        primary_entry_price = entry_price
        
        # Get DCA entry price (second entry) - prioritize AI-suggested Entry 2, then recalculated Entry 2, then original
        # Priority: 1) AI-suggested Entry 2, 2) Recalculated Entry 2 (if Entry 1 was optimized), 3) Original Entry 2
        if 'optimized_prices' in validation_result and validation_result.get('optimized_prices', {}).get('second_entry_price'):
            dca_entry_price = validation_result['optimized_prices']['second_entry_price']
            # Check if this is AI-suggested (from AI validation) or recalculated (from Entry 1 optimization)
            if 'suggested_second_entry_price' in validation_result.get('price_suggestions', {}):
                logger.info(f"🔄 [PRICE UPDATE] Using AI-suggested Entry 2 (DCA): ${dca_entry_price:,.8f}")
            else:
                logger.info(f"🔄 [PRICE UPDATE] Using recalculated Entry 2 (DCA): ${dca_entry_price:,.8f} (based on Entry 1 optimization)")
        else:
            dca_entry_price = second_entry_price if second_entry_price and second_entry_price > 0 else None
            if dca_entry_price:
                logger.info(f"ℹ️  [PRICE] Using original Entry 2 (DCA): ${dca_entry_price:,.8f}")
        
        # If this is a primary entry, we need both prices to create both orders
        if is_primary_entry and not dca_entry_price:
            logger.warning(f"Primary entry signal received but no second_entry_price provided. Using entry_price for both.")
            dca_entry_price = primary_entry_price  # Fallback to same price if not provided
        
        # If this is a DCA entry, use the second_entry_price from JSON
        if is_dca_entry:
            if dca_entry_price and dca_entry_price > 0:
                entry_price = dca_entry_price
                logger.info(f"Using DCA entry price from second_entry_price: {entry_price}")
            else:
                # Use entry_price as fallback for DCA
                dca_entry_price = entry_price
                logger.info(f"Using entry_price as DCA entry price: {entry_price}")
        
        # Validate entry price (should already be validated above, but double-check)
        if entry_price is None or entry_price <= 0:
            logger.warning(f"Invalid entry price after parsing: {entry_price}. Discarding request.")
            return {'success': False, 'error': 'Invalid entry price (NA/null or <= 0)'}
        
        # Get symbol info for precision
        exchange_info = client.futures_exchange_info()
        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
        
        if not symbol_info:
            return {'success': False, 'error': f'Symbol {symbol} not found'}
        
        # Set leverage
        try:
            client.futures_change_leverage(symbol=symbol, leverage=LEVERAGE)
            logger.info(f"Set leverage to {LEVERAGE}X for {symbol}")
        except Exception as e:
            logger.warning(f"Could not set leverage (may already be set): {e}")
        
        # Get price precision
        price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
        tick_size = float(price_filter['tickSize']) if price_filter else 0.01
        
        # Format prices to tick size precision (removes floating point errors)
        primary_entry_price = format_price_precision(primary_entry_price, tick_size)
        if dca_entry_price:
            dca_entry_price = format_price_precision(dca_entry_price, tick_size)
        
        # Calculate quantity based on entry size and leverage
        primary_quantity = calculate_quantity(primary_entry_price, symbol_info)
        dca_quantity = calculate_quantity(dca_entry_price, symbol_info) if dca_entry_price else primary_quantity
        
        # Risk Validation: Check if trade risk is within acceptable limits
        # This includes risk from pending orders (orders waiting to fill)
        # Can be disabled via ENABLE_RISK_VALIDATION=false if needed
        if ENABLE_RISK_VALIDATION and stop_loss and stop_loss > 0:
            is_valid_risk, risk_info, risk_error = validate_risk_per_trade(
                symbol, primary_entry_price, stop_loss, primary_quantity, signal_side
            )
            
            if not is_valid_risk:
                logger.error(f"❌ Risk validation FAILED for {symbol}: {risk_error}")
                send_slack_alert(
                    error_type="Risk Validation Failed",
                    message=risk_error,
                    details={
                        'Risk_Percent': f"{risk_info['risk_percent']:.2f}%",
                        'Max_Risk_Percent': f"{MAX_RISK_PERCENT}%",
                        'Current_Trade_Risk': f"${risk_info['current_trade_risk']:.2f}",
                        'Pending_Orders_Risk': f"${risk_info['pending_orders_risk']:.2f}",
                        'Total_Risk': f"${risk_info['total_risk']:.2f}",
                        'Account_Balance': f"${risk_info['account_balance']:.2f}",
                        'Pending_Orders_Count': len(risk_info['pending_orders'])
                    },
                    symbol=symbol,
                    severity='WARNING'
                )
                return {
                    'success': False,
                    'error': f'Risk validation failed: {risk_error}',
                    'risk_info': risk_info
                }
            
            # Log risk info if available
            if risk_info:
                logger.info(f"💰 Risk Validation: {risk_info['risk_percent']:.2f}% of account "
                          f"(${risk_info['total_risk']:.2f} total risk: ${risk_info['current_trade_risk']:.2f} current + "
                          f"${risk_info['pending_orders_risk']:.2f} pending from {len(risk_info['pending_orders'])} orders)")
        elif not ENABLE_RISK_VALIDATION:
            logger.debug(f"Risk validation disabled - skipping risk check for {symbol}")
        
        # Initialize active trades tracking
        if symbol not in active_trades:
            active_trades[symbol] = {
                'primary_filled': False, 
                'dca_filled': False, 
                'position_open': False,
                'primary_order_id': None,
                'dca_order_id': None,
                'tp_order_id': None,
                'sl_order_id': None,
                'original_stop_loss': None,
                'original_entry1': None,
                'original_entry2': None,
                'current_sl_price': None,
                'sl_moved_to_breakeven': False,
                'tp1_filled': False
            }
        
        # Store original prices for trailing stop loss
        active_trades[symbol]['original_stop_loss'] = stop_loss
        active_trades[symbol]['original_entry1'] = primary_entry_price
        active_trades[symbol]['original_entry2'] = dca_entry_price if dca_entry_price else None
        
        current_time = time.time()
        order_results = []
        
        # If this is a primary entry, create BOTH entry orders immediately
        if is_primary_entry:
            # Create primary entry order (Entry 1)
            primary_order_params = {
                'symbol': symbol,
                'side': side,
                'type': 'LIMIT',
                'timeInForce': 'GTC',
                'quantity': primary_quantity,
                'price': primary_entry_price,
            }
            if is_hedge_mode:
                primary_order_params['positionSide'] = position_side
            
            logger.info(f"Creating PRIMARY entry limit order: {primary_order_params}")
            try:
                primary_order_result = client.futures_create_order(**primary_order_params)
                order_results.append(primary_order_result)
                active_trades[symbol]['primary_order_id'] = primary_order_result.get('orderId')
                active_trades[symbol]['primary_filled'] = False  # Will be updated when filled
                active_trades[symbol]['position_open'] = True
                logger.info(f"✅ PRIMARY entry order created successfully: Order ID {primary_order_result.get('orderId')}")
                
                # Smart TP Strategy: Based on AI Confidence Score
                # High Confidence (>=90%): Use single TP (main TP from signal) - trust the signal completely
                # Lower Confidence (<90%): Use TP1 + TP2 strategy - secure profits early
                
                confidence_score = validation_result.get('confidence_score', 100.0) if validation_result else 100.0
                use_single_tp = confidence_score >= TP_HIGH_CONFIDENCE_THRESHOLD
                
                # Calculate entry price for TP calculation
                # TP1: Always use Entry 1 price only (4% from Entry 1)
                # TP2: Use average if Entry 2 provided, otherwise Entry 1 only
                entry_price_for_tp1 = primary_entry_price  # Always Entry 1 for TP1
                
                if dca_entry_price and dca_entry_price > 0:
                    avg_entry_price = (primary_entry_price + dca_entry_price) / 2
                    entry_price_for_tp2 = avg_entry_price  # Average for TP2
                else:
                    # Entry 2 not provided - use Entry 1 for TP2 as well
                    entry_price_for_tp2 = primary_entry_price
                    logger.info(f"📊 Entry 2 not provided - using Entry 1 price for TP2 calculation: ${entry_price_for_tp2:,.8f}")
                
                tp_side = 'SELL' if side == 'BUY' else 'BUY'
                total_qty = primary_quantity + (dca_quantity if dca_entry_price else 0)
                
                if use_single_tp:
                    # HIGH CONFIDENCE: Use single TP (main TP from signal)
                    if take_profit and take_profit > 0:
                        main_tp_price = format_price_precision(take_profit, tick_size)
                    else:
                        # If no TP provided, calculate a conservative TP
                        default_tp_percent = 0.05  # 5% profit for high confidence
                        if side == 'BUY':  # LONG position
                            main_tp_price = entry_price_for_tp2 * (1 + default_tp_percent)
                        else:  # SHORT position
                            main_tp_price = entry_price_for_tp2 * (1 - default_tp_percent)
                        main_tp_price = format_price_precision(main_tp_price, tick_size)
                        logger.info(f"📊 High confidence signal ({confidence_score:.1f}%) - TP not provided, calculating with {default_tp_percent*100}% profit: {main_tp_price}")
                    
                    # Store as single TP (100% of position)
                    active_trades[symbol]['tp1_price'] = None  # No TP1
                    active_trades[symbol]['tp2_price'] = main_tp_price  # Use TP2 as main TP
                    active_trades[symbol]['tp_side'] = tp_side
                    active_trades[symbol]['tp1_quantity'] = 0  # No TP1
                    active_trades[symbol]['tp2_quantity'] = total_qty  # 100% at main TP
                    active_trades[symbol]['tp_working_type'] = 'MARK_PRICE'
                    active_trades[symbol]['use_single_tp'] = True  # Flag for single TP mode
                    logger.info(f"📝 HIGH CONFIDENCE ({confidence_score:.1f}%) - Single TP configured for {symbol}:")
                    logger.info(f"   → Main TP: @ ${main_tp_price:,.8f} (closes 100% = {total_qty} of position)")
                    logger.info(f"   → Strategy: Trusting signal completely - using single TP")
                else:
                    # LOWER CONFIDENCE: Use TP1 + TP2 strategy (secure profits early)
                    # Calculate TP1: 3-4% profit from Entry 1 ONLY (not average)
                    tp1_percent = TP1_PERCENT / 100.0
                    if side == 'BUY':  # LONG position
                        tp1_price = entry_price_for_tp1 * (1 + tp1_percent)  # Use Entry 1 only
                    else:  # SHORT position
                        tp1_price = entry_price_for_tp1 * (1 - tp1_percent)  # Use Entry 1 only
                    tp1_price = format_price_precision(tp1_price, tick_size)
                    logger.info(f"📊 TP1 calculated: {TP1_PERCENT}% from Entry 1 (${primary_entry_price:,.8f}) = ${tp1_price:,.8f}")
                    
                    # TP2: Use the TP from webhook (or AI-optimized TP)
                    if take_profit and take_profit > 0:
                        tp2_price = format_price_precision(take_profit, tick_size)
                    else:
                        # If no TP provided, calculate a default TP2 (higher than TP1)
                        default_tp2_percent = 0.055  # 5.5% profit
                        if side == 'BUY':  # LONG position
                            tp2_price = entry_price_for_tp2 * (1 + default_tp2_percent)
                        else:  # SHORT position
                            tp2_price = entry_price_for_tp2 * (1 - default_tp2_percent)
                        tp2_price = format_price_precision(tp2_price, tick_size)
                        logger.info(f"📊 TP2 not provided in webhook - calculating with {default_tp2_percent*100}% profit: {tp2_price}")
                    
                    # Calculate TP1 and TP2 quantities based on split percentages
                    tp1_quantity = total_qty * (TP1_SPLIT / 100.0)  # 70% of position
                    tp2_quantity = total_qty * (TP2_SPLIT / 100.0)  # 30% of position
                    
                    # Store TP1 and TP2 details
                    active_trades[symbol]['tp1_price'] = tp1_price
                    active_trades[symbol]['tp2_price'] = tp2_price
                    active_trades[symbol]['tp_side'] = tp_side
                    active_trades[symbol]['tp1_quantity'] = tp1_quantity
                    active_trades[symbol]['tp2_quantity'] = tp2_quantity
                    active_trades[symbol]['tp_working_type'] = 'MARK_PRICE'
                    active_trades[symbol]['use_single_tp'] = False  # Flag for TP1+TP2 mode
                    logger.info(f"📝 LOWER CONFIDENCE ({confidence_score:.1f}%) - TP1 + TP2 configured for {symbol}:")
                    logger.info(f"   → TP1: {TP1_PERCENT}% profit @ ${tp1_price:,.8f} (closes {TP1_SPLIT}% = {tp1_quantity} of position)")
                    logger.info(f"   → TP2: @ ${tp2_price:,.8f} (closes {TP2_SPLIT}% = {tp2_quantity} of position)")
                    logger.info(f"   → Strategy: Securing profits early with TP1, letting TP2 run")
                
                logger.info(f"   → TPs will be created automatically when position opens (background thread checks every 1min/2min)")
                
                # Send signal notification to Slack
                try:
                    timeframe = signal_data.get('timeframe', 'Unknown')
                    confidence_score = validation_result.get('confidence_score', 100.0) if validation_result else 100.0
                    risk_level = validation_result.get('risk_level', 'MEDIUM') if validation_result else 'MEDIUM'
                    # Get TP prices for notification
                    if use_single_tp:
                        main_tp = active_trades[symbol].get('tp2_price')
                        tp1_for_notif = None
                    else:
                        main_tp = active_trades[symbol].get('tp2_price')
                        tp1_for_notif = active_trades[symbol].get('tp1_price')
                    
                    send_signal_notification(
                        symbol=symbol,
                        signal_side=signal_side,
                        timeframe=timeframe,
                        confidence_score=confidence_score,
                        risk_level=risk_level,
                        entry1_price=primary_entry_price,
                        entry2_price=dca_entry_price,
                        stop_loss=stop_loss,
                        take_profit=main_tp,  # Main TP (TP2 in dual mode, or single TP in high confidence)
                        tp1_price=tp1_for_notif,  # TP1 (only in dual mode)
                        use_single_tp=use_single_tp,  # Flag for notification formatting
                        validation_result=validation_result
                    )
                except Exception as e:
                    logger.debug(f"Failed to send signal notification: {e}")
                        
            except BinanceAPIException as e:
                logger.error(f"❌ Failed to create PRIMARY entry order: {e.message} (Code: {e.code})")
                send_slack_alert(
                    error_type="Primary Entry Order Creation Failed",
                    message=f"{e.message} (Code: {e.code})",
                    details={'Error_Code': e.code, 'Entry_Price': entry_price, 'Quantity': primary_quantity, 'Side': side},
                    symbol=symbol,
                    severity='ERROR'
                )
                return {'success': False, 'error': f'Failed to create order: {e.message}'}
            except Exception as e:
                logger.error(f"❌ Unexpected error creating PRIMARY entry order: {e}")
                send_slack_alert(
                    error_type="Primary Entry Order Creation Error",
                    message=str(e),
                    details={'Entry_Price': entry_price, 'Quantity': primary_quantity, 'Side': side},
                    symbol=symbol,
                    severity='ERROR'
                )
                return {'success': False, 'error': f'Unexpected error: {str(e)}'}
            
            # Track order
            order_key = f"{symbol}_{primary_entry_price}_{side}_PRIMARY"
            recent_orders[order_key] = current_time
            
            # Create DCA entry order (Entry 2) if price is provided
            if dca_entry_price and dca_entry_price > 0:
                dca_order_params = {
                    'symbol': symbol,
                    'side': side,
                    'type': 'LIMIT',
                    'timeInForce': 'GTC',
                    'quantity': dca_quantity,
                    'price': dca_entry_price,
                }
                if is_hedge_mode:
                    dca_order_params['positionSide'] = position_side
                
                logger.info(f"Creating DCA entry limit order: {dca_order_params}")
                try:
                    dca_order_result = client.futures_create_order(**dca_order_params)
                    order_results.append(dca_order_result)
                    active_trades[symbol]['dca_order_id'] = dca_order_result.get('orderId')
                    active_trades[symbol]['dca_filled'] = False  # Will be updated when filled
                    logger.info(f"✅ DCA entry order created successfully: Order ID {dca_order_result.get('orderId')}")
                except BinanceAPIException as e:
                    logger.error(f"❌ Failed to create DCA entry order: {e.message} (Code: {e.code})")
                    send_slack_alert(
                        error_type="DCA Entry Order Creation Failed",
                        message=f"{e.message} (Code: {e.code})",
                        details={'Error_Code': e.code, 'DCA_Price': dca_entry_price, 'Quantity': dca_quantity, 'Side': side},
                        symbol=symbol,
                        severity='WARNING'
                    )
                    # Continue with primary order even if DCA fails
                except Exception as e:
                    logger.error(f"❌ Unexpected error creating DCA entry order: {e}")
                    send_slack_alert(
                        error_type="DCA Entry Order Creation Error",
                        message=str(e),
                        details={'DCA_Price': dca_entry_price, 'Quantity': dca_quantity, 'Side': side},
                        symbol=symbol,
                        severity='WARNING'
                    )
                
                # Track order
                order_key = f"{symbol}_{dca_entry_price}_{side}_DCA"
                recent_orders[order_key] = current_time
            
            # Use primary order result for response
            order_result = primary_order_result
            entry_price = primary_entry_price
            quantity = primary_quantity
            entry_type = "PRIMARY"
            
            # Check if position exists and create TP1 and TP2 immediately (in case entry filled quickly)
            # Also schedule a delayed retry in case Binance hasn't updated position yet
            if symbol in active_trades and 'tp1_price' in active_trades[symbol] and 'tp2_price' in active_trades[symbol]:
                # Immediate check
                create_tp1_tp2_if_needed(symbol, active_trades[symbol])
                # Delayed retry (5 seconds) in case position updates are delayed
                delayed_tp_creation(symbol, delay_seconds=5)
                # Another delayed retry (15 seconds) for slower fills
                delayed_tp_creation(symbol, delay_seconds=15)
            
        else:
            # This is a DCA entry - create only DCA order
            # Ensure dca_entry_price is set (should be set above, but use entry_price as fallback)
            if not dca_entry_price:
                dca_entry_price = entry_price
            
            dca_order_params = {
                'symbol': symbol,
                'side': side,
                'type': 'LIMIT',
                'timeInForce': 'GTC',
                'quantity': dca_quantity,
                'price': dca_entry_price,
            }
            if is_hedge_mode:
                dca_order_params['positionSide'] = position_side
            
            logger.info(f"Creating DCA entry limit order: {dca_order_params}")
            try:
                order_result = client.futures_create_order(**dca_order_params)
                order_results.append(order_result)
                active_trades[symbol]['dca_order_id'] = order_result.get('orderId')
                active_trades[symbol]['dca_filled'] = False
                logger.info(f"✅ DCA entry order created successfully: Order ID {order_result.get('orderId')}")
            except BinanceAPIException as e:
                logger.error(f"❌ Failed to create DCA entry order: {e.message} (Code: {e.code})")
                send_slack_alert(
                    error_type="DCA Entry Order Creation Failed",
                    message=f"{e.message} (Code: {e.code})",
                    details={'Error_Code': e.code, 'DCA_Price': dca_price, 'Quantity': dca_qty, 'Side': side},
                    symbol=symbol,
                    severity='ERROR'
                )
                return {'success': False, 'error': f'Failed to create DCA order: {e.message}'}
            except Exception as e:
                logger.error(f"❌ Unexpected error creating DCA entry order: {e}")
                send_slack_alert(
                    error_type="DCA Entry Order Creation Error",
                    message=str(e),
                    details={'DCA_Price': dca_price, 'Quantity': dca_qty, 'Side': side},
                    symbol=symbol,
                    severity='ERROR'
                )
                return {'success': False, 'error': f'Unexpected error: {str(e)}'}
            
            # Track order
            order_key = f"{symbol}_{dca_entry_price}_{side}_DCA"
            recent_orders[order_key] = current_time
            
            entry_price = dca_entry_price
            quantity = dca_quantity
            entry_type = "DCA"
            
            # When DCA entry alert comes, check if position exists and create TP1/TP2 immediately
            # This handles the case where an entry filled and TradingView sent DCA fill alert
            if symbol in active_trades and 'tp1_price' in active_trades[symbol] and 'tp2_price' in active_trades[symbol]:
                logger.info(f"DCA entry alert received - checking for position to create TP1/TP2 immediately")
                create_tp1_tp2_if_needed(symbol, active_trades[symbol])
        
        # Cleanup closed positions periodically
        if len(active_trades) > 100:
            cleanup_closed_positions()
        
        # Clean old orders from tracking (keep last 1000)
        if len(recent_orders) > 1000:
            # Remove oldest entries
            sorted_orders = sorted(recent_orders.items(), key=lambda x: x[1])
            for key, _ in sorted_orders[:-1000]:
                del recent_orders[key]
        
        logger.info(f"{entry_type} order(s) created successfully: {order_results}")
        
        # After creating orders, check if position exists and create TP1/TP2 immediately
        # This handles cases where entry filled between webhook calls
        if symbol in active_trades and 'tp1_price' in active_trades[symbol] and 'tp2_price' in active_trades[symbol]:
            logger.info(f"Checking for position to create TP1/TP2 orders immediately")
            create_tp1_tp2_if_needed(symbol, active_trades[symbol])
        
        # For DCA entry: Create TP order if not exists (using same TP price from primary entry)
        if is_dca_entry and take_profit and take_profit > 0:
            # Check for existing TP orders
            has_orders, open_orders = check_existing_orders(symbol)
            existing_tp = [o for o in open_orders if o.get('type') == 'TAKE_PROFIT_MARKET']
            
            if not existing_tp:
                # Get current position quantity to update TP
                try:
                    positions = client.futures_position_information(symbol=symbol)
                    current_position_qty = 0
                    for pos in positions:
                        pos_amt = float(pos.get('positionAmt', 0))
                        if abs(pos_amt) > 0:
                            pos_side = pos.get('positionSide', 'BOTH')
                            if pos_side == 'BOTH' or pos_side == position_side.upper():
                                current_position_qty = abs(pos_amt)
                                break
                    
                    if current_position_qty > 0:
                        # Update TP with current position quantity
                        tp_side = 'SELL' if side == 'BUY' else 'BUY'
                        tp_price = format_price_precision(take_profit, tick_size)
                        tp_params = {
                            'symbol': symbol,
                            'side': tp_side,
                            'type': 'TAKE_PROFIT_MARKET',
                            'timeInForce': 'GTC',
                            'quantity': current_position_qty,  # Current position quantity
                            'stopPrice': tp_price
                            # Note: reduceOnly is not required for TAKE_PROFIT_MARKET orders
                        }
                        if is_hedge_mode:
                            tp_params['positionSide'] = position_side
                        tp_order = client.futures_create_order(**tp_params)
                        if symbol in active_trades:
                            active_trades[symbol]['tp_order_id'] = tp_order.get('orderId')
                        logger.info(f"Take profit order created for DCA entry: {tp_order}")
                except BinanceAPIException as e:
                    # Check if order type is not supported (error -4120)
                    if e.code == -4120 or 'order type not supported' in str(e).lower():
                        logger.warning(f"⚠️ TAKE_PROFIT_MARKET orders not supported for {symbol} (Code: {e.code}). This symbol may not support conditional orders. You may need to set TP manually in Binance UI.")
                        send_slack_alert(
                            error_type="DCA Take Profit Order Type Not Supported",
                            message=f"TAKE_PROFIT_MARKET orders are not supported for {symbol}. You may need to set TP manually in Binance UI.",
                            details={'Error_Code': e.code, 'TP_Price': tp_price, 'Symbol': symbol, 'DCA_Entry_Price': dca_entry_price if 'dca_entry_price' in locals() else 'Unknown'},
                            symbol=symbol,
                            severity='WARNING'
                        )
                    else:
                        logger.error(f"Failed to create take profit order for DCA: {e}")
                        send_slack_alert(
                            error_type="DCA Take Profit Creation Failed",
                            message=str(e),
                            details={'Error_Code': e.code, 'DCA_Entry_Price': dca_entry_price if 'dca_entry_price' in locals() else 'Unknown'},
                            symbol=symbol,
                            severity='ERROR'
                        )
                except Exception as e:
                    logger.error(f"Failed to create take profit order for DCA: {e}")
                    send_slack_alert(
                        error_type="DCA Take Profit Creation Failed",
                        message=str(e),
                        details={'DCA_Entry_Price': dca_entry_price if 'dca_entry_price' in locals() else 'Unknown'},
                        symbol=symbol,
                        severity='ERROR'
                    )
            else:
                logger.info(f"Take profit order already exists for {symbol}, skipping creation")
        
        return {
            'success': True,
            'order_id': order_result.get('orderId'),
            'symbol': symbol,
            'side': side,
            'price': entry_price,
            'quantity': quantity,
            'entry_type': entry_type,
            'leverage': LEVERAGE,
            'position_value_usd': ENTRY_SIZE_USD * LEVERAGE
        }
        
    except BinanceAPIException as e:
        logger.error(f"Binance API error: {e}")
        # symbol is always defined by this point (extracted at function start)
        send_slack_alert(
            error_type="Binance API Error",
            message=f"{e.message} (Code: {e.code})",
            details={'Error_Code': e.code, 'Event': event if 'event' in locals() else 'UNKNOWN'},
            symbol=symbol if 'symbol' in locals() else None,
            severity='ERROR'
        )
        return {'success': False, 'error': f'Binance API error: {e.message}'}
    except BinanceOrderException as e:
        logger.error(f"Binance order error: {e}")
        send_slack_alert(
            error_type="Binance Order Error",
            message=f"{e.message} (Code: {e.code})",
            details={'Error_Code': e.code, 'Event': event if 'event' in locals() else 'UNKNOWN'},
            symbol=symbol if 'symbol' in locals() else None,
            severity='ERROR'
        )
        return {'success': False, 'error': f'Binance order error: {e.message}'}
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        send_slack_alert(
            error_type="Unexpected Error",
            message=str(e),
            details={'Event': event if 'event' in locals() else 'UNKNOWN'},
            symbol=symbol if 'symbol' in locals() else None,
            severity='ERROR'
        )
        return {'success': False, 'error': str(e)}


@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle webhook requests from TradingView"""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            logger.warning("No JSON data received")
            return jsonify({'success': False, 'error': 'No data received'}), 400
        
        logger.info(f"Received webhook: {json.dumps(data, indent=2)}")
        
        # Process order in background thread to avoid blocking
        def process_order():
            result = create_limit_order(data)
            logger.info(f"Order result: {result}")
        
        thread = threading.Thread(target=process_order)
        thread.daemon = True
        thread.start()
        
        # Return immediate response
        return jsonify({'success': True, 'message': 'Webhook received, processing order'}), 200
        
    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        send_slack_alert(
            error_type="Webhook Processing Error",
            message=str(e),
            details={'Request_Method': request.method, 'Request_Path': request.path},
            severity='ERROR'
        )
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        # Check Binance connection
        if client:
            client.ping()
            binance_status = 'connected'
        else:
            binance_status = 'disconnected'
        
        return jsonify({
            'status': 'healthy',
            'binance': binance_status,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@app.route('/verify-account', methods=['GET'])
def verify_account():
    """Verify which account we're connected to and check orders"""
    try:
        if not client:
            return jsonify({'error': 'Binance client not initialized'}), 500
        
        # Get account info
        account_info = {}
        try:
            # Try to get futures account info
            futures_account = client.futures_account()
            account_info['futures_balance'] = futures_account.get('totalWalletBalance', 'N/A')
            account_info['available_balance'] = futures_account.get('availableBalance', 'N/A')
        except Exception as e:
            account_info['futures_error'] = str(e)
        
        # Get open orders for BTCUSDT
        open_orders = []
        try:
            orders = client.futures_get_open_orders(symbol='BTCUSDT')
            open_orders = [{
                'orderId': o.get('orderId'),
                'symbol': o.get('symbol'),
                'side': o.get('side'),
                'type': o.get('type'),
                'price': o.get('price'),
                'quantity': o.get('origQty'),
                'status': o.get('status'),
                'time': o.get('time')
            } for o in orders]
        except Exception as e:
            open_orders = {'error': str(e)}
        
        # Get all open orders (any symbol)
        all_orders = []
        try:
            all_orders_list = client.futures_get_open_orders()
            all_orders = [{
                'orderId': o.get('orderId'),
                'symbol': o.get('symbol'),
                'side': o.get('side'),
                'type': o.get('type'),
                'price': o.get('price'),
                'quantity': o.get('origQty'),
                'status': o.get('status')
            } for o in all_orders_list]
        except Exception as e:
            all_orders = {'error': str(e)}
        
        return jsonify({
            'account_info': account_info,
            'btcusdt_orders': open_orders,
            'all_open_orders': all_orders,
            'total_orders': len(all_orders) if isinstance(all_orders, list) else 0,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@app.route('/check-tp', methods=['GET'])
def check_tp():
    """Manual endpoint to check and create missing TP orders"""
    try:
        if not client:
            return jsonify({'error': 'Binance client not initialized'}), 500
        
        results = []
        for symbol, trade_info in list(active_trades.items()):
            if 'tp_price' in trade_info and 'tp_order_id' not in trade_info:
                # Check if position exists
                try:
                    positions = client.futures_position_information(symbol=symbol)
                    for position in positions:
                        position_amt = float(position.get('positionAmt', 0))
                        if abs(position_amt) > 0:
                            results.append({
                                'symbol': symbol,
                                'tp_price': trade_info['tp_price'],
                                'position_amt': position_amt,
                                'status': 'position_exists_ready_for_tp'
                            })
                            break
                    else:
                        results.append({
                            'symbol': symbol,
                            'tp_price': trade_info['tp_price'],
                            'status': 'no_position_yet'
                        })
                except Exception as e:
                    results.append({
                        'symbol': symbol,
                        'tp_price': trade_info.get('tp_price'),
                        'status': 'error',
                        'error': str(e)
                    })
        
        return jsonify({
            'stored_tp_orders': results,
            'total': len(results)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'service': 'Binance Futures Webhook Service',
        'version': '1.0.0',
        'endpoints': {
            'webhook': '/webhook (POST)',
            'health': '/health (GET)',
            'verify-account': '/verify-account (GET)',
            'check-tp': '/check-tp (GET)'
        }
    }), 200


if __name__ == '__main__':
    # Check configuration
    if WEBHOOK_TOKEN == 'CHANGE_ME':
        logger.warning("WEBHOOK_TOKEN is not set! Using default value.")
    
    # Log current trading configuration
    logger.info(f"💰 Trading Configuration:")
    logger.info(f"   Entry Size: ${ENTRY_SIZE_USD} per entry")
    logger.info(f"   Leverage: {LEVERAGE}X")
    logger.info(f"   Position Value: ${ENTRY_SIZE_USD * LEVERAGE} per entry (${ENTRY_SIZE_USD * LEVERAGE * 2} total for both entries)")
    is_testing = ENTRY_SIZE_USD == 5.0 and LEVERAGE == 5
    is_production = ENTRY_SIZE_USD == 10.0 and LEVERAGE == 20
    logger.info(f"   Mode: {'TESTING ($5, 5X)' if is_testing else 'PRODUCTION ($10, 20X)' if is_production else 'CUSTOM'}")
    logger.info(f"   To change: Set ENTRY_SIZE_USD and LEVERAGE environment variables")
    logger.info(f"   Example: ENTRY_SIZE_USD=10.0 LEVERAGE=20 for production")
    
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        logger.error("BINANCE_API_KEY and BINANCE_API_SECRET must be set!")
        send_slack_alert(
            error_type="Configuration Error",
            message="BINANCE_API_KEY and BINANCE_API_SECRET must be set!",
            details={'API_Key_Set': bool(BINANCE_API_KEY), 'API_Secret_Set': bool(BINANCE_API_SECRET)},
            severity='CRITICAL'
        )
        exit(1)
    
    # Run Flask app
    # Use 0.0.0.0 to listen on all interfaces
    # Use a production WSGI server like gunicorn in production
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

