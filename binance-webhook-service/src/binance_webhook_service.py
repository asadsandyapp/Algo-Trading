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

# AI Validation Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
ENABLE_AI_VALIDATION = os.getenv('ENABLE_AI_VALIDATION', 'true').lower() == 'true'
AI_VALIDATION_MIN_CONFIDENCE = float(os.getenv('AI_VALIDATION_MIN_CONFIDENCE', '50'))
ENABLE_AI_PRICE_SUGGESTIONS = os.getenv('ENABLE_AI_PRICE_SUGGESTIONS', 'true').lower() == 'true'  # Manual review mode

# Initialize Gemini API if available and configured
gemini_client = None
if GEMINI_AVAILABLE and GEMINI_API_KEY and ENABLE_AI_VALIDATION:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_client = genai.GenerativeModel('gemini-pro')
        logger.info("Gemini API initialized successfully for signal validation")
    except Exception as e:
        logger.warning(f"Failed to initialize Gemini API: {e}. AI validation will be disabled.")
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
ENTRY_SIZE_USD = float(os.getenv('ENTRY_SIZE_USD', '5.0'))  # Default: $5 for testing, change to 10.0 for production
# Leverage (can be overridden via LEVERAGE environment variable)
# TESTING MODE: Set to 5X for safer testing
# PRODUCTION: Change back to 20X after testing
LEVERAGE = int(os.getenv('LEVERAGE', '5'))  # Default: 5X for testing, change to 20 for production
TOTAL_ENTRIES = 2  # Primary entry + DCA entry

# Track active trades per symbol
active_trades = {}  # {symbol: {'primary_filled': bool, 'dca_filled': bool, 'position_open': bool, 
                    #           'primary_order_id': int, 'dca_order_id': int, 'tp_order_id': int, 'sl_order_id': int,
                    #           'exit_processed': bool, 'last_exit_time': float}}

# Track recent EXIT events to prevent duplicate processing
recent_exits = {}  # {symbol: timestamp}
EXIT_COOLDOWN = 30

# Helper function to create TP order for a symbol (can be called from anywhere)
def delayed_tp_creation(symbol, delay_seconds=5):
    """Helper function to create TP order after a delay (allows Binance to update position)"""
    def _create():
        time.sleep(delay_seconds)
        if symbol in active_trades and 'tp_price' in active_trades[symbol]:
            logger.info(f"🔄 Delayed TP check for {symbol} (after {delay_seconds}s delay)")
            create_tp_if_needed(symbol, active_trades[symbol])
    
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
            # - First 4 checks (1 minute): Every 15 seconds to catch newly filled orders quickly
            # - Next 10 checks (5 minutes): Every 30 seconds for active monitoring
            # - After that: Every 5 minutes for ongoing checks
            if check_count < 4:
                sleep_time = 15  # Check every 15 seconds for first minute
            elif check_count < 14:
                sleep_time = 30  # Check every 30 seconds for next 5 minutes
            else:
                sleep_time = 300  # Then check every 5 minutes (reduces API calls significantly)
            
            time.sleep(sleep_time)
            check_count += 1
            
            try:
                # Only check symbols that have stored TP details (optimization: skip symbols without TP config)
                symbols_to_check = [s for s, info in active_trades.items() if 'tp_price' in info]
                
                if not symbols_to_check:
                    # No symbols with stored TP details - skip API calls entirely
                    continue
                
                # Get ALL open positions from Binance (single API call)
                all_positions = client.futures_position_information()
                positions_dict = {p['symbol']: p for p in all_positions if abs(float(p.get('positionAmt', 0))) > 0}
                
                if not positions_dict:
                    # No open positions - clean up stored TP details
                    for symbol in list(symbols_to_check):
                        if symbol in active_trades and 'tp_price' in active_trades[symbol]:
                            logger.info(f"🧹 Background thread: Cleaning up stored TP for {symbol} (no open position)")
                            if 'tp_price' in active_trades[symbol]:
                                del active_trades[symbol]['tp_price']
                            if 'tp_quantity' in active_trades[symbol]:
                                del active_trades[symbol]['tp_quantity']
                            if 'tp_working_type' in active_trades[symbol]:
                                del active_trades[symbol]['tp_working_type']
                    continue
                
                # Only log if we're actually checking symbols (reduce log spam)
                if len(symbols_to_check) > 0:
                    logger.debug(f"Background thread: Checking {len(symbols_to_check)} symbol(s) with stored TP details")
                
                # Only check symbols that have stored TP details AND have open positions
                symbols_checked = 0
                for symbol in symbols_to_check:
                    if symbol not in positions_dict:
                        # Symbol has stored TP but no position - might be filled or canceled
                        # Check once and clean up if no orders exist
                        try:
                            has_orders, _ = check_existing_orders(symbol)
                            if not has_orders:
                                logger.info(f"🧹 Background thread: Cleaning up stored TP for {symbol} (no position and no orders)")
                                if symbol in active_trades:
                                    if 'tp_price' in active_trades[symbol]:
                                        del active_trades[symbol]['tp_price']
                                    if 'tp_quantity' in active_trades[symbol]:
                                        del active_trades[symbol]['tp_quantity']
                                    if 'tp_working_type' in active_trades[symbol]:
                                        del active_trades[symbol]['tp_working_type']
                        except:
                            pass
                        continue
                    
                    symbols_checked += 1
                    position = positions_dict[symbol]
                    position_amt = float(position.get('positionAmt', 0))
                    
                    try:
                        # Check if TP order already exists (don't log every check to reduce spam)
                        has_orders, open_orders = check_existing_orders(symbol, log_result=False)
                        existing_tp = [o for o in open_orders if o.get('type') == 'TAKE_PROFIT_MARKET']
                        
                        if existing_tp:
                            # TP already exists, ensure it's tracked in active_trades
                            if symbol in active_trades:
                                active_trades[symbol]['tp_order_id'] = existing_tp[0].get('orderId')
                                # Clean up stored TP details since TP is now active
                                if 'tp_price' in active_trades[symbol]:
                                    del active_trades[symbol]['tp_price']
                                if 'tp_quantity' in active_trades[symbol]:
                                    del active_trades[symbol]['tp_quantity']
                                if 'tp_working_type' in active_trades[symbol]:
                                    del active_trades[symbol]['tp_working_type']
                            continue  # TP exists, skip
                        
                        # No TP order exists - we have stored TP details, create TP order
                        trade_info = active_trades[symbol]
                        logger.info(f"🔄 Background thread: Position exists for {symbol} with stored TP details - creating TP order")
                        success = create_tp_if_needed(symbol, trade_info)
                        if success:
                            logger.info(f"✅ Background thread: TP order created successfully for {symbol}")
                        else:
                            logger.warning(f"⚠️ Background thread: Failed to create TP for {symbol} (check logs for details)")
                    
                    except Exception as e:
                        logger.error(f"Error processing position {symbol}: {e}", exc_info=True)
                        send_slack_alert(
                            error_type="Background TP Check Error",
                            message=str(e),
                            details={'Position_Amount': position_amt if 'position_amt' in locals() else 'Unknown'},
                            symbol=symbol,
                            severity='WARNING'
                        )
                
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


def calculate_quantity(entry_price, symbol_info):
    """Calculate quantity based on $10 per entry with 20X leverage"""
    # Position value = Entry size * Leverage = $10 * 20 = $200
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
    
    logger.info(f"Calculated quantity: {quantity} (Position value: ${position_value} @ ${entry_price}, step_size: {step_size})")
    return quantity


# Validation cache to avoid duplicate API calls
validation_cache = {}
VALIDATION_CACHE_TTL = 300  # 5 minutes


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
                klines = client.futures_klines(symbol=symbol, interval=interval, limit=50)
                if klines:
                    # Extract OHLCV data
                    closes = [float(k[4]) for k in klines]
                    highs = [float(k[2]) for k in klines]
                    lows = [float(k[3]) for k in klines]
                    volumes = [float(k[5]) for k in klines]
                    
                    # Calculate technical indicators
                    # Trend analysis
                    recent_closes = closes[-20:]  # Last 20 candles
                    if len(recent_closes) >= 2:
                        price_change = ((recent_closes[-1] - recent_closes[0]) / recent_closes[0]) * 100
                        market_data['recent_trend_pct'] = price_change
                        market_data['trend_direction'] = 'UP' if price_change > 0.5 else 'DOWN' if price_change < -0.5 else 'SIDEWAYS'
                    
                    # Support/Resistance levels
                    market_data['recent_high'] = max(highs[-20:]) if len(highs) >= 20 else max(highs)
                    market_data['recent_low'] = min(lows[-20:]) if len(lows) >= 20 else min(lows)
                    
                    # Volume analysis
                    avg_volume = sum(volumes[-20:]) / len(volumes[-20:]) if len(volumes) >= 20 else sum(volumes) / len(volumes)
                    current_volume = volumes[-1] if volumes else 0
                    market_data['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1.0
                    market_data['volume_status'] = 'HIGH' if market_data['volume_ratio'] > 1.5 else 'NORMAL' if market_data['volume_ratio'] > 0.5 else 'LOW'
                    
                    # Price position relative to recent range
                    if market_data['recent_high'] > market_data['recent_low']:
                        price_position = ((entry_price - market_data['recent_low']) / 
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
- RSI: {rsi_val:.2f if rsi_val else 'N/A'} (Oversold: <30, Overbought: >85)
- MACD Line: {macd_line:.4f if macd_line else 'N/A'}
- MACD Signal: {macd_signal:.4f if macd_signal else 'N/A'}
- MACD Histogram: {macd_hist:.4f if macd_hist else 'N/A'} (Positive = bullish momentum)
- Stochastic K: {stoch_k:.2f if stoch_k else 'N/A'} (Oversold: <20, Overbought: >80)
- Stochastic D: {stoch_d:.2f if stoch_d else 'N/A'}
- EMA 200: ${ema200_val:,.8f if ema200_val else 'N/A'} (Price {'ABOVE' if price_above_ema200 else 'BELOW'} EMA200 = {'BULLISH' if price_above_ema200 else 'BEARISH'} trend)
- ATR: {atr_val:.8f if atr_val else 'N/A'} (Volatility measure)
- Bollinger Bands: Upper=${bb_upper:,.8f if bb_upper else 'N/A'}, Basis=${bb_basis:,.8f if bb_basis else 'N/A'}, Lower=${bb_lower:,.8f if bb_lower else 'N/A'}
- Smart Money Volume (Normalized): {smv_norm:.2f if smv_norm else 'N/A'} (Positive = buying pressure)
- Cumulative SMV: {cum_smv:.2f if cum_smv else 'N/A'} ({'BUYING' if smart_money_buy else 'SELLING' if smart_money_sell else 'NEUTRAL'} pressure)
- Supertrend: ${supertrend_val:,.8f if supertrend_val else 'N/A'} ({'BULLISH' if supertrend_bull else 'BEARISH'})
- OBV: {obv_val:.2f if obv_val else 'N/A'} (Rising = buying pressure)
- Relative Volume Percentile: {rel_vol_pct:.1f if rel_vol_pct else 'N/A'}% (High: >70%, Low: <30%)
- MFI (Money Flow Index): {mfi_val:.2f if mfi_val else 'N/A'} (Oversold: <20, Overbought: >80)
- Volume Ratio: {vol_ratio:.2f if vol_ratio else 'N/A'}x (vs average)
- Bullish Divergence: {'YES ✅' if has_bull_div else 'NO'}
- Bearish Divergence: {'YES ✅' if has_bear_div else 'NO'}
- At Bottom/Top: {'BOTTOM ✅' if at_bottom else 'TOP ✅' if at_top else 'MID-RANGE'}"""
    
    # Build prompt for AI - Enhanced with real market data AND indicator values for technical analysis
    market_info = ""
    if market_data.get('current_price'):
        market_info = f"""
REAL-TIME MARKET DATA (from Binance API):
- Current Market Price: ${market_data['current_price']:,.8f}
- Entry Price vs Current: Entry is {market_data.get('entry_vs_current', 'N/A')} current price
- Price Distance: {market_data.get('price_distance_pct', 0):.2f}% from current price"""
        
        if market_data.get('trend_direction'):
            market_info += f"""
- Recent Trend ({timeframe}): {market_data.get('trend_direction', 'N/A')} ({market_data.get('recent_trend_pct', 0):+.2f}%)
- Price Range (last 20 candles): ${market_data.get('recent_low', 0):,.8f} - ${market_data.get('recent_high', 0):,.8f}
- Entry Position: {market_data.get('price_level', 'N/A')} ({market_data.get('price_position_in_range', 0):.1f}% of range)
- Volume Status: {market_data.get('volume_status', 'N/A')} (current/avg ratio: {market_data.get('volume_ratio', 1):.2f}x)
- Volatility: {market_data.get('volatility_status', 'N/A')} ({market_data.get('volatility_pct', 0):.2f}%)"""
    
    prompt = f"""You are an expert cryptocurrency trading analyst evaluating a trading signal from an automated system.

IMPORTANT CONTEXT:
- This signal comes from a TradingView indicator that already filters signals
- The system has a 65% win rate, so signals are generally reliable
- Your role is to catch OBVIOUSLY bad signals, not to be overly conservative
- When in doubt, APPROVE the signal (fail-open design)
- You now have REAL-TIME market data for proper technical analysis

Signal Details:
- Symbol: {symbol}
- Direction: {signal_side}
- Timeframe: {timeframe}
- Entry Price: ${entry_price:,.8f}
- Stop Loss: ${stop_loss:,.8f} (if provided)
- Take Profit: ${take_profit:,.8f} (if provided)
- Risk/Reward Ratio: {risk_reward_ratio:.2f if risk_reward_ratio else 'N/A'}{market_info}{indicator_info}

COMPREHENSIVE TECHNICAL ANALYSIS EVALUATION:
Use BOTH the real-time market data AND the TradingView indicator values above to evaluate this signal.

INDICATOR-BASED ANALYSIS:
1. RSI Analysis:
   - LONG signals: RSI < 50 is GOOD (oversold <30 is EXCELLENT) ✅
   - SHORT signals: RSI > 50 is GOOD (overbought >85 is EXCELLENT) ✅
   - RSI divergence (bullish/bearish) = STRONG confirmation ✅

2. MACD Analysis:
   - MACD Line > Signal Line = Bullish momentum ✅
   - MACD Histogram positive = Bullish momentum ✅
   - LONG: MACD bullish = GOOD ✅
   - SHORT: MACD bearish = GOOD ✅

3. Stochastic Analysis:
   - LONG: Stoch K/D < 50 (oversold <20 is EXCELLENT) ✅
   - SHORT: Stoch K/D > 50 (overbought >80 is EXCELLENT) ✅

4. Trend Filters (EMA 200 & Supertrend):
   - LONG: Price above EMA200 AND Supertrend bullish = STRONG trend ✅
   - SHORT: Price below EMA200 AND Supertrend bearish = STRONG trend ✅
   - Contradicting trend = Evaluate carefully but APPROVE if other factors good

5. Volume Analysis:
   - High Relative Volume (>70%) = Strong confirmation ✅
   - Volume Ratio > 1.5x = Strong confirmation ✅
   - OBV rising = Buying pressure ✅
   - Smart Money Buying = Institutional accumulation ✅

6. Bollinger Bands:
   - LONG near lower band = Good entry zone ✅
   - SHORT near upper band = Good entry zone ✅
   - Price at bands = Potential reversal ✅

7. Divergence & Reversal Signals:
   - Bullish Divergence + At Bottom = EXCELLENT LONG setup ✅
   - Bearish Divergence + At Top = EXCELLENT SHORT setup ✅

8. Market Data Analysis:
   - Trend Alignment: Use both market trend AND indicator trends (EMA200, Supertrend)
   - Price Position: Combine market support/resistance with Bollinger Bands levels
   - Volume: Use both Relative Volume Percentile AND Volume Ratio for confirmation

9. Risk/Reward: 
   - APPROVE if R/R >= 1.0 (even 1:1 is acceptable for good setups)
   - Only REJECT if R/R < 0.5 AND multiple indicators are bearish
   - R/R between 0.5-1.0: Evaluate based on indicator alignment above

SIGNAL QUALITY SCORING:
- EXCELLENT (80-100%): Multiple indicators aligned + good R/R + volume confirmation + divergence/reversal signals
- GOOD (60-79%): Most indicators aligned + acceptable R/R + normal volume
- ACCEPTABLE (50-59%): Some indicators aligned + acceptable R/R (may have minor concerns)
- QUESTIONABLE (40-49%): Mixed signals but not clearly bad (still approve if above threshold)
- POOR (0-39%): Multiple indicators contradict signal + poor R/R + low volume

REJECTION CRITERIA (only reject if MULTIPLE red flags):
- Risk/Reward < 0.5 AND
- Entry price >5% away from current market price AND
- Signal contradicts STRONG trend (>3% against signal direction) AND
- Price at unfavorable level (LONG at resistance, SHORT at support)

Otherwise, APPROVE with appropriate confidence score based on technical analysis.

Respond in JSON format ONLY with this exact structure:
{{
    "is_valid": true/false,
    "confidence_score": 0-100,
    "reasoning": "Brief explanation (1-2 sentences)",
    "risk_level": "LOW" or "MEDIUM" or "HIGH",
    "suggested_entry_price": <number> or null,
    "suggested_stop_loss": <number> or null,
    "suggested_take_profit": <number> or null,
    "price_suggestion_reasoning": "Why these prices are suggested (if different from original)"
}}
Note: If you want to suggest price optimizations, include the suggested_* fields. Otherwise, you may omit them or set them to null.

Confidence Score Guidelines:
- 80-100: Excellent signal, strong R/R, clear setup
- 60-79: Good signal, acceptable R/R, reasonable setup
- 50-59: Acceptable signal, may have minor concerns but still valid
- 40-49: Questionable but not clearly bad (still approve if above threshold)
- 0-39: Only for signals with clear red flags

Remember: When uncertain, err on the side of APPROVAL. This is a filter, not a gatekeeper.

PRICE OPTIMIZATION (AI will calculate optimal prices based on technical analysis):
You should calculate and suggest optimized prices using REAL market data:
- Current market price, support/resistance levels, ATR volatility, Bollinger Bands, EMA200, etc.
- Use the indicator values provided above (RSI, MACD, Stochastic, etc.)

OPTIMIZATION RULES (only suggest if BETTER than original):
1. ENTRY PRICE:
   - LONG trades: Suggest LOWER entry (closer to support) if better R/R or closer to key level
   - SHORT trades: Suggest HIGHER entry (closer to resistance) if better R/R or closer to key level
   - DO NOT suggest worse entry (LONG: don't suggest higher, SHORT: don't suggest lower)

2. STOP LOSS:
   - Suggest TIGHTER SL if ATR/volatility allows (better risk management)
   - LONG: SL should be BELOW entry (suggest lower SL if safe)
   - SHORT: SL should be ABOVE entry (suggest higher SL if safe)
   - DO NOT suggest wider SL unless absolutely necessary

3. TAKE PROFIT:
   - Suggest HIGHER TP for LONG (more profit potential at resistance)
   - Suggest LOWER TP for SHORT (more profit potential at support)
   - Consider R/R ratio - aim for at least 1.5:1 or better

4. ENTRY 2 (DCA) DISTANCE:
   - If Entry 1 is optimized, calculate Entry 2 based on optimal distance
   - LONG: Entry 2 should be LOWER than Entry 1 (typical 3-7% below)
   - SHORT: Entry 2 should be HIGHER than Entry 1 (typical 3-7% above)
   - Ensure good spacing between Entry 1 and Entry 2

CALCULATION METHOD:
- Use current market price, recent high/low, ATR, Bollinger Bands, support/resistance levels
- Calculate optimal entry based on: support/resistance + ATR + R/R ratio
- Calculate optimal SL based on: ATR × multiplier (typically 1.5-2.5x ATR)
- Calculate optimal TP based on: resistance levels + R/R ratio (aim for 1.5:1 minimum)

If original prices are already optimal, you may omit suggestion fields (they will use original).
If you suggest prices, they will be APPLIED if they improve the trade (better entry, tighter SL, higher TP)."""
    
    try:
        # Call Gemini API with timeout
        logger.info(f"🤖 [AI VALIDATION] Starting validation for NEW ENTRY signal: {symbol} ({signal_side}) @ ${entry_price:,.8f}")
        logger.info(f"🤖 [AI VALIDATION] This validation ONLY runs for new ENTRY signals, NOT for order tracking or TP creation")
        start_time = time.time()
        
        # Use threading to implement timeout
        result_container = {'response': None, 'error': None}
        
        def call_api():
            try:
                response = gemini_client.generate_content(prompt)
                result_container['response'] = response.text
            except Exception as e:
                result_container['error'] = str(e)
        
        api_thread = threading.Thread(target=call_api, daemon=True)
        api_thread.start()
        api_thread.join(timeout=5)  # 5 second timeout
        
        if api_thread.is_alive():
            logger.warning(f"AI validation timeout for {symbol}, proceeding without validation (fail-open)")
            return {
                'is_valid': True,
                'confidence_score': 50.0,
                'reasoning': 'AI validation timeout, proceeding',
                'risk_level': 'MEDIUM'
            }
        
        if result_container['error']:
            raise Exception(result_container['error'])
        
        if not result_container['response']:
            raise Exception("No response from AI")
        
        elapsed_time = time.time() - start_time
        logger.debug(f"AI validation completed in {elapsed_time:.2f}s")
        
        # Parse response
        response_text = result_container['response'].strip()
        
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
        try:
            validation_result = json.loads(response_text)
        except json.JSONDecodeError:
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
            if suggested_entry or suggested_sl or suggested_tp:
                validation_result['suggested_entry_price'] = suggested_entry
                validation_result['suggested_stop_loss'] = suggested_sl
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
        
        # Extract price suggestions and apply smart optimization
        optimized_prices = {
            'entry_price': entry_price,  # Default to original
            'stop_loss': stop_loss,      # Default to original
            'take_profit': take_profit,  # Default to original
            'second_entry_price': None,   # Will be calculated if Entry 1 is optimized
            'applied_optimizations': []
        }
        
        if ENABLE_AI_PRICE_SUGGESTIONS:
            suggested_entry = safe_float(validation_result.get('suggested_entry_price'), default=None)
            suggested_sl = safe_float(validation_result.get('suggested_stop_loss'), default=None)
            suggested_tp = safe_float(validation_result.get('suggested_take_profit'), default=None)
            price_reasoning = validation_result.get('price_suggestion_reasoning', '')
            
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
            if suggested_entry or suggested_sl or suggested_tp:
                validation_result['price_suggestions'] = {
                    'entry_price': suggested_entry,
                    'stop_loss': suggested_sl,
                    'take_profit': suggested_tp,
                    'reasoning': price_reasoning,
                    'original_entry': entry_price,
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
                   f"Reasoning={validation_result['reasoning'][:100]}...")
        
        return validation_result
        
    except Exception as e:
        logger.warning(f"⚠️ AI validation error for {symbol}: {e}. Proceeding without validation (fail-open)")
        return {
            'is_valid': True,  # Fail-open: proceed if validation fails
            'confidence_score': 50.0,
            'reasoning': f'AI validation error: {str(e)}, proceeding',
            'risk_level': 'MEDIUM',
            'error': str(e)
        }


def create_limit_order(signal_data):
    """Create a Binance Futures limit order with $10 per entry and 20X leverage"""
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
                logger.warning(f"🚫 AI Validation REJECTED signal for {symbol}: {validation_result.get('reasoning', 'No reasoning provided')}")
                return {
                    'success': False,
                    'error': 'Signal validation failed',
                    'validation_result': validation_result
                }
            
            confidence_score = validation_result.get('confidence_score', 100.0)
            if confidence_score < AI_VALIDATION_MIN_CONFIDENCE:
                logger.warning(f"🚫 AI Validation REJECTED signal for {symbol}: Confidence score {confidence_score:.1f}% is below minimum threshold of {AI_VALIDATION_MIN_CONFIDENCE}%")
                logger.info(f"   Reasoning: {validation_result.get('reasoning', 'No reasoning provided')}")
                logger.info(f"   Risk Level: {validation_result.get('risk_level', 'UNKNOWN')}")
                return {
                    'success': False,
                    'error': f'Signal confidence {confidence_score:.1f}% below minimum {AI_VALIDATION_MIN_CONFIDENCE}%',
                    'validation_result': validation_result
                }
            
            # Log successful validation
            logger.info(f"✅ AI Validation APPROVED signal for {symbol}: Confidence={confidence_score:.1f}%, "
                       f"Risk={validation_result.get('risk_level', 'UNKNOWN')}, "
                       f"Reasoning={validation_result.get('reasoning', 'No reasoning')[:150]}...")
            
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
                if opt_prices.get('entry_price') and opt_prices['entry_price'] != safe_float(signal_data.get('entry_price'), default=entry_price):
                    original_entry = safe_float(signal_data.get('entry_price'), default=entry_price)
                    original_entry2 = safe_float(signal_data.get('second_entry_price'), default=None)
                    
                    if original_entry2 and original_entry:
                        # Calculate Entry 2 based on optimized Entry 1, maintaining the same percentage distance
                        entry_diff_pct = abs((original_entry2 - original_entry) / original_entry * 100)
                        if signal_side == 'LONG':
                            # Entry 2 should be lower than Entry 1
                            optimized_entry2 = opt_prices['entry_price'] * (1 - entry_diff_pct / 100)
                            opt_prices['second_entry_price'] = optimized_entry2
                            logger.info(f"🔄 [PRICE UPDATE] Calculated optimized Entry 2: ${optimized_entry2:,.8f} (based on Entry 1 optimization, {entry_diff_pct:.2f}% below Entry 1)")
                        else:  # SHORT
                            # Entry 2 should be higher than Entry 1
                            optimized_entry2 = opt_prices['entry_price'] * (1 + entry_diff_pct / 100)
                            opt_prices['second_entry_price'] = optimized_entry2
                            logger.info(f"🔄 [PRICE UPDATE] Calculated optimized Entry 2: ${optimized_entry2:,.8f} (based on Entry 1 optimization, {entry_diff_pct:.2f}% above Entry 1)")
        
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
        
        # Get DCA entry price (second entry) - use optimized second_entry_price if available
        # Check if AI optimized Entry 2
        if 'optimized_prices' in validation_result and validation_result.get('optimized_prices', {}).get('second_entry_price'):
            dca_entry_price = validation_result['optimized_prices']['second_entry_price']
            logger.info(f"🔄 [PRICE UPDATE] Using AI-optimized Entry 2 (DCA): ${dca_entry_price:,.8f}")
        else:
            dca_entry_price = second_entry_price if second_entry_price and second_entry_price > 0 else None
        
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
        
        # Set leverage to 20X
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
        
        # Calculate quantity based on $10 per entry with 20X leverage
        primary_quantity = calculate_quantity(primary_entry_price, symbol_info)
        dca_quantity = calculate_quantity(dca_entry_price, symbol_info) if dca_entry_price else primary_quantity
        
        # Initialize active trades tracking
        if symbol not in active_trades:
            active_trades[symbol] = {
                'primary_filled': False, 
                'dca_filled': False, 
                'position_open': False,
                'primary_order_id': None,
                'dca_order_id': None,
                'tp_order_id': None,
                'sl_order_id': None
            }
        
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
                
                # Store TP order details immediately (like Binance UI - TP is "set" but pending until position exists)
                # The background thread will create the TP order as soon as the limit order fills and position opens
                DEFAULT_TP_PERCENT = 0.021  # 2.1% profit (used when TP not provided)
                
                if take_profit and take_profit > 0:
                    # Use TP from webhook
                    tp_side = 'SELL' if side == 'BUY' else 'BUY'
                    tp_price = format_price_precision(take_profit, tick_size)
                else:
                    # Calculate TP with 2.1% profit when not provided
                    if side == 'BUY':  # LONG position
                        tp_price = entry_price * (1 + DEFAULT_TP_PERCENT)
                        tp_side = 'SELL'
                    else:  # SHORT position
                        tp_price = entry_price * (1 - DEFAULT_TP_PERCENT)
                        tp_side = 'BUY'
                    tp_price = format_price_precision(tp_price, tick_size)
                    logger.info(f"📊 TP not provided in webhook - calculating with {DEFAULT_TP_PERCENT*100}% profit: {tp_price}")
                
                total_qty = primary_quantity + (dca_quantity if dca_entry_price else 0)
                
                # Store TP details in active_trades[symbol] - will be created automatically when position exists
                # TP is stored in memory: active_trades[symbol]['tp_price'], ['tp_side'], ['tp_quantity'], ['tp_working_type']
                # Background thread checks ALL positions every 15s (first 2 min) then every 2 min
                active_trades[symbol]['tp_price'] = tp_price
                active_trades[symbol]['tp_side'] = tp_side
                active_trades[symbol]['tp_quantity'] = total_qty
                active_trades[symbol]['tp_working_type'] = 'MARK_PRICE'
                logger.info(f"📝 TP order configured and stored in active_trades[{symbol}]: price={tp_price}, side={tp_side}, qty={total_qty}, workingType=MARK_PRICE")
                logger.info(f"   → TP will be created automatically when position opens (background thread checks every 15s/2min)")
                        
            except BinanceAPIException as e:
                logger.error(f"❌ Failed to create PRIMARY entry order: {e.message} (Code: {e.code})")
                send_slack_alert(
                    error_type="Primary Entry Order Creation Failed",
                    message=f"{e.message} (Code: {e.code})",
                    details={'Error_Code': e.code, 'Entry_Price': entry_price, 'Quantity': qty, 'Side': side},
                    symbol=symbol,
                    severity='ERROR'
                )
                return {'success': False, 'error': f'Failed to create order: {e.message}'}
            except Exception as e:
                logger.error(f"❌ Unexpected error creating PRIMARY entry order: {e}")
                send_slack_alert(
                    error_type="Primary Entry Order Creation Error",
                    message=str(e),
                    details={'Entry_Price': entry_price, 'Quantity': qty, 'Side': side},
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
                        details={'Error_Code': e.code, 'DCA_Price': dca_price, 'Quantity': dca_qty, 'Side': side},
                        symbol=symbol,
                        severity='WARNING'
                    )
                    # Continue with primary order even if DCA fails
                except Exception as e:
                    logger.error(f"❌ Unexpected error creating DCA entry order: {e}")
                    send_slack_alert(
                        error_type="DCA Entry Order Creation Error",
                        message=str(e),
                        details={'DCA_Price': dca_price, 'Quantity': dca_qty, 'Side': side},
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
            
            # Check if position exists and create TP immediately (in case entry filled quickly)
            # Also schedule a delayed retry in case Binance hasn't updated position yet
            if symbol in active_trades and 'tp_price' in active_trades[symbol]:
                # Immediate check
                create_tp_if_needed(symbol, active_trades[symbol])
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
            
            # When DCA entry alert comes, check if position exists and create TP immediately
            # This handles the case where an entry filled and TradingView sent DCA fill alert
            if symbol in active_trades and 'tp_price' in active_trades[symbol]:
                logger.info(f"DCA entry alert received - checking for position to create TP immediately")
                create_tp_if_needed(symbol, active_trades[symbol])
        
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
        
        # After creating orders, check if position exists and create TP immediately
        # This handles cases where entry filled between webhook calls
        if symbol in active_trades and 'tp_price' in active_trades[symbol]:
            logger.info(f"Checking for position to create TP order immediately")
            create_tp_if_needed(symbol, active_trades[symbol])
        
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
    logger.info(f"   Mode: {'TESTING ($5, 5X)' if is_testing else 'PRODUCTION/CUSTOM'}")
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

