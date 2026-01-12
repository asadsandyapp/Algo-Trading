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

#Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('webhook_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration from environment variables
WEBHOOK_TOKEN = os.getenv('WEBHOOK_TOKEN', 'CHANGE_ME')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
BINANCE_TESTNET = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
BINANCE_SUB_ACCOUNT_EMAIL = os.getenv('BINANCE_SUB_ACCOUNT_EMAIL', '')  # For sub-account trading
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL', '')  # Slack webhook URL for error notifications

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
ENTRY_SIZE_USD = 10.0  # $10 per entry
LEVERAGE = 20  # 20X leverage
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
    """Create TP order if position exists and TP is stored but not created yet"""
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
        
        # Handle EXIT events - close position at market price and cancel all orders for symbol
        if event == 'EXIT':
            logger.info(f"Processing EXIT event for {symbol}")
            
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
        
        # Get primary entry price - use entry_price from JSON
        primary_entry_price = entry_price
        
        # Get DCA entry price (second entry) - use second_entry_price from JSON
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

