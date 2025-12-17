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
from datetime import datetime
from flask import Flask, request, jsonify
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
import threading

# Configure logging
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
EXIT_COOLDOWN = 30  # seconds - prevent duplicate EXIT processing


def verify_webhook_token(payload_token):
    """Verify webhook token matches configured token"""
    return payload_token == WEBHOOK_TOKEN


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
        positions = client.futures_position_information(symbol=symbol)
        for position in positions:
            position_amt = float(position.get('positionAmt', 0))
            if abs(position_amt) > 0:  # Position exists
                position_side = position.get('positionSide', 'BOTH')
                # Check if position side matches
                if position_side == 'BOTH' or position_side == signal_side.upper():
                    logger.info(f"Existing position found for {symbol}: {position_amt} @ {position.get('entryPrice')}")
                    return True, position
        return False, None
    except Exception as e:
        logger.error(f"Error checking existing position: {e}")
        return False, None


def check_existing_orders(symbol):
    """Check for existing open orders (limit orders) for the symbol"""
    try:
        open_orders = client.futures_get_open_orders(symbol=symbol)
        if open_orders:
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
        # Get current position
        positions = client.futures_position_information(symbol=symbol)
        position_to_close = None
        
        for position in positions:
            position_amt = float(position.get('positionAmt', 0))
            if abs(position_amt) > 0:  # Position exists
                position_side = position.get('positionSide', 'BOTH')
                # Check if position side matches
                if position_side == 'BOTH' or position_side == signal_side.upper():
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
            step_size = float(lot_size_filter['stepSize']) if lot_size_filter else 0.001
            close_quantity = round(close_quantity / step_size) * step_size
        
        # Create market order to close position
        close_params = {
            'symbol': symbol,
            'side': close_side,
            'type': 'MARKET',
            'quantity': close_quantity,
            'reduceOnly': True
        }
        
        # Only include positionSide if in Hedge mode
        if is_hedge_mode:
            position_side_str = 'LONG' if position_amt > 0 else 'SHORT'
            close_params['positionSide'] = position_side_str
        
        logger.info(f"Closing position at market for {symbol}: {close_params}")
        result = client.futures_create_order(**close_params)
        logger.info(f"Position closed successfully: {result}")
        
        return {
            'success': True,
            'order_id': result.get('orderId'),
            'symbol': symbol,
            'quantity': close_quantity,
            'side': close_side
        }
        
    except BinanceAPIException as e:
        logger.error(f"Binance API error closing position: {e}")
        return {'success': False, 'error': f'Binance API error: {e.message}'}
    except Exception as e:
        logger.error(f"Error closing position: {e}", exc_info=True)
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


def calculate_quantity(entry_price, symbol_info):
    """Calculate quantity based on $10 per entry with 20X leverage"""
    # Position value = Entry size * Leverage = $10 * 20 = $200
    position_value = ENTRY_SIZE_USD * LEVERAGE
    
    # Quantity = Position value / Entry price
    quantity = position_value / entry_price
    
    # Round quantity to step size
    lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
    step_size = float(lot_size_filter['stepSize']) if lot_size_filter else 0.001
    min_qty = float(lot_size_filter['minQty']) if lot_size_filter else 0.001
    
    # Round to step size
    quantity = round(quantity / step_size) * step_size
    
    # Ensure minimum quantity
    if quantity < min_qty:
        quantity = min_qty
    
    logger.info(f"Calculated quantity: {quantity} (Position value: ${position_value} @ ${entry_price})")
    return quantity


def create_limit_order(signal_data):
    """Create a Binance Futures limit order with $10 per entry and 20X leverage"""
    try:
        # Extract signal data
        token = signal_data.get('token')
        event = signal_data.get('event')
        signal_side = signal_data.get('signal_side')
        symbol = format_symbol(signal_data.get('symbol', ''))
        entry_price = float(signal_data.get('entry_price', 0))
        stop_loss = float(signal_data.get('stop_loss', 0)) if signal_data.get('stop_loss') else None
        take_profit = float(signal_data.get('take_profit', 0)) if signal_data.get('take_profit') else None
        reduce_only = signal_data.get('reduce_only', False)
        order_subtype = signal_data.get('order_subtype', 'primary_entry')
        second_entry_price = float(signal_data.get('second_entry_price', 0)) if signal_data.get('second_entry_price') else None
        
        # Verify token
        if not verify_webhook_token(token):
            logger.warning(f"Invalid webhook token received")
            return {'success': False, 'error': 'Invalid token'}
        
        # Handle EXIT events - close position at market price and cancel unfilled orders
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
            
            # Check if position actually exists before trying to close
            has_position, position_info = check_existing_position(symbol, signal_side)
            
            if has_position:
                # Position exists - close it at market price
                logger.info(f"Closing position at market price for {symbol}")
                close_result = close_position_at_market(symbol, signal_side, is_hedge_mode)
                
                if close_result.get('success'):
                    logger.info(f"Position closed successfully: {close_result}")
                    # Track EXIT processing
                    recent_exits[symbol] = current_time
                    
                    # Clean up old exit tracking
                    if len(recent_exits) > 1000:
                        # Remove oldest entries
                        sorted_exits = sorted(recent_exits.items(), key=lambda x: x[1])
                        for key, _ in sorted_exits[:-1000]:
                            del recent_exits[key]
                else:
                    logger.error(f"Failed to close position: {close_result.get('error')}")
            else:
                logger.info(f"No open position found for {symbol} - position may already be closed")
            
            # Cancel any unfilled entry orders (Entry 2/DCA)
            if symbol in active_trades:
                trade_info = active_trades[symbol]
                dca_order_id = trade_info.get('dca_order_id')
                
                if dca_order_id:
                    # Check if DCA order still exists
                    has_orders, open_orders = check_existing_orders(symbol)
                    dca_order_exists = False
                    if has_orders:
                        dca_order_exists = any(o.get('orderId') == dca_order_id for o in open_orders)
                    
                    if dca_order_exists:
                        logger.info(f"Canceling unfilled DCA order {dca_order_id} for {symbol}")
                        cancel_order(symbol, dca_order_id)
                    else:
                        logger.info(f"DCA order {dca_order_id} already filled or canceled for {symbol}")
                
                # Clean up tracking
                logger.info(f"Cleaning up closed trade tracking for {symbol}")
                del active_trades[symbol]
            
            # Track EXIT processing even if no position was found
            recent_exits[symbol] = current_time
            
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
        
        # Check for existing open position FIRST - this is the most reliable duplicate check
        has_position, position_info = check_existing_position(symbol, signal_side)
        
        # If position exists, reject ENTRY alerts immediately (strongest duplicate prevention)
        # This handles the case where trade is already open and running
        if has_position:
            # Also check for pending limit orders - if position exists, we shouldn't create new orders
            has_orders, open_orders = check_existing_orders(symbol)
            matching_orders = [o for o in open_orders if o.get('side') == side and o.get('type') == 'LIMIT'] if has_orders else []
            
            logger.warning(f"Position already exists for {symbol} ({signal_side}). "
                          f"Found {len(matching_orders)} pending limit orders. Discarding duplicate ENTRY alert.")
            return {'success': False, 'error': f'Position already exists for {symbol} - duplicate ENTRY alert discarded'}
        
        # Check active trades tracking
        if symbol in active_trades:
            trade_info = active_trades[symbol]
            # If both entries are already filled, reject any new alerts
            if trade_info.get('primary_filled') and trade_info.get('dca_filled'):
                logger.warning(f"Both entries already filled for {symbol}. Discarding duplicate alert.")
                return {'success': False, 'error': 'Both entries already filled for this symbol'}
            
            # If primary entry is filled and this is another primary entry, reject
            if is_primary_entry and trade_info.get('primary_filled'):
                logger.warning(f"Primary entry already filled for {symbol}. Discarding duplicate primary entry alert.")
                return {'success': False, 'error': 'Primary entry already filled for this symbol'}
            
            # If DCA entry is filled and this is another DCA entry, reject
            if is_dca_entry and trade_info.get('dca_filled'):
                logger.warning(f"DCA entry already filled for {symbol}. Discarding duplicate DCA entry alert.")
                return {'success': False, 'error': 'DCA entry already filled for this symbol'}
        
        # Additional check: if we have active trades but no position, clean up
        if symbol in active_trades and not has_position:
            # Position was closed but tracking still exists - clean up
            logger.info(f"Cleaning up stale trade tracking for {symbol} (position closed)")
            del active_trades[symbol]
        
        # Check for existing open orders and clean up if position is closed
        has_orders, open_orders = check_existing_orders(symbol)
        
        # If position is closed but orders exist, clean them up
        if not has_position and has_orders and is_primary_entry:
            logger.info(f"Position closed but limit orders exist for {symbol}. Cleaning up...")
            canceled = cancel_all_limit_orders(symbol, side)
            logger.info(f"Canceled {canceled} limit orders for {symbol}")
            # Clean up tracking
            if symbol in active_trades:
                del active_trades[symbol]
        
        # Check for duplicate limit orders (even if position doesn't exist yet)
        if has_orders:
            # Check if we already have matching limit orders for this entry type
            matching_orders = [o for o in open_orders if o.get('side') == side and o.get('type') == 'LIMIT']
            if matching_orders:
                # Check if this is a duplicate primary or DCA entry
                if is_primary_entry:
                    # If we have any limit orders for primary entry, it's a duplicate
                    logger.warning(f"Primary entry limit order already exists for {symbol}. "
                                 f"Found {len(matching_orders)} matching limit orders. Discarding duplicate alert.")
                    return {'success': False, 'error': 'Primary entry order already exists for this symbol'}
                elif is_dca_entry:
                    # Check if we have a DCA order already
                    # Count limit orders - if we have 2, both entries are placed
                    if len(matching_orders) >= 2:
                        logger.warning(f"Both entry orders already exist for {symbol}. "
                                     f"Found {len(matching_orders)} limit orders. Discarding duplicate alert.")
                        return {'success': False, 'error': 'Both entry orders already exist for this symbol'}
                    # If we have 1 order and this is DCA, check if primary is filled
                    if symbol in active_trades and active_trades[symbol].get('primary_filled'):
                        # This might be a duplicate DCA alert, but allow if DCA not filled
                        if not active_trades[symbol].get('dca_filled'):
                            logger.info(f"Primary entry filled, allowing DCA entry for {symbol}")
                        else:
                            logger.warning(f"DCA entry already filled. Discarding duplicate DCA alert.")
                            return {'success': False, 'error': 'DCA entry already filled'}
                    else:
                        # We have 1 limit order but no tracking - might be a duplicate
                        logger.warning(f"Found existing limit order for {symbol} but no tracking. "
                                     f"Treating as duplicate DCA alert.")
                        return {'success': False, 'error': 'Limit order already exists for this symbol'}
        
        # Get primary entry price
        primary_entry_price = entry_price
        
        # Get DCA entry price (second entry)
        dca_entry_price = second_entry_price if second_entry_price and second_entry_price > 0 else None
        
        # If this is a primary entry, we need both prices to create both orders
        if is_primary_entry and not dca_entry_price:
            logger.warning(f"Primary entry signal received but no second_entry_price provided. Using entry_price for both.")
            dca_entry_price = primary_entry_price  # Fallback to same price if not provided
        
        # If this is a DCA entry, use the second_entry_price or entry_price
        if is_dca_entry:
            if dca_entry_price and dca_entry_price > 0:
                entry_price = dca_entry_price
                logger.info(f"Using DCA entry price from second_entry_price: {entry_price}")
            else:
                # Use entry_price as fallback for DCA
                dca_entry_price = entry_price
                logger.info(f"Using entry_price as DCA entry price: {entry_price}")
        
        # Validate entry price
        if entry_price <= 0:
            return {'success': False, 'error': 'Invalid entry price'}
        
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
        
        # Round prices to tick size
        primary_entry_price = round(primary_entry_price / tick_size) * tick_size
        if dca_entry_price:
            dca_entry_price = round(dca_entry_price / tick_size) * tick_size
        
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
            primary_order_result = client.futures_create_order(**primary_order_params)
            order_results.append(primary_order_result)
            active_trades[symbol]['primary_order_id'] = primary_order_result.get('orderId')
            active_trades[symbol]['primary_filled'] = False  # Will be updated when filled
            active_trades[symbol]['position_open'] = True
            
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
                dca_order_result = client.futures_create_order(**dca_order_params)
                order_results.append(dca_order_result)
                active_trades[symbol]['dca_order_id'] = dca_order_result.get('orderId')
                active_trades[symbol]['dca_filled'] = False  # Will be updated when filled
                
                # Track order
                order_key = f"{symbol}_{dca_entry_price}_{side}_DCA"
                recent_orders[order_key] = current_time
            
            # Use primary order result for response
            order_result = primary_order_result
            entry_price = primary_entry_price
            quantity = primary_quantity
            entry_type = "PRIMARY"
            
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
            order_result = client.futures_create_order(**dca_order_params)
            order_results.append(order_result)
            active_trades[symbol]['dca_order_id'] = order_result.get('orderId')
            active_trades[symbol]['dca_filled'] = False
            
            # Track order
            order_key = f"{symbol}_{dca_entry_price}_{side}_DCA"
            recent_orders[order_key] = current_time
            
            entry_price = dca_entry_price
            quantity = dca_quantity
            entry_type = "DCA"
        
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
        
        # Calculate total quantity for TP/SL (both entries combined)
        total_quantity = primary_quantity + dca_quantity if is_primary_entry and dca_entry_price else quantity * TOTAL_ENTRIES
        
        # Create stop loss and take profit orders (only for primary entry to avoid duplicates)
        if is_primary_entry:
            # Check for existing TP/SL orders
            has_orders, open_orders = check_existing_orders(symbol)
            existing_sl = [o for o in open_orders if o.get('type') == 'STOP_MARKET' and o.get('reduceOnly')]
            existing_tp = [o for o in open_orders if o.get('type') == 'TAKE_PROFIT_MARKET' and o.get('reduceOnly')]
            
            if stop_loss and stop_loss > 0 and not existing_sl:
                try:
                    sl_side = 'SELL' if side == 'BUY' else 'BUY'
                    sl_price = round(stop_loss / tick_size) * tick_size
                    sl_params = {
                        'symbol': symbol,
                        'side': sl_side,
                        'type': 'STOP_MARKET',
                        'timeInForce': 'GTC',
                        'quantity': total_quantity,  # Total quantity for both entries
                        'stopPrice': sl_price,
                        'reduceOnly': True
                    }
                    # Only include positionSide if in Hedge mode
                    if is_hedge_mode:
                        sl_params['positionSide'] = position_side
                    sl_order = client.futures_create_order(**sl_params)
                    active_trades[symbol]['sl_order_id'] = sl_order.get('orderId')
                    logger.info(f"Stop loss order created: {sl_order}")
                except Exception as e:
                    logger.error(f"Failed to create stop loss order: {e}")
            elif existing_sl:
                logger.info(f"Stop loss order already exists for {symbol}, skipping creation")
        
            if take_profit and take_profit > 0 and not existing_tp:
                try:
                    tp_side = 'SELL' if side == 'BUY' else 'BUY'
                    tp_price = round(take_profit / tick_size) * tick_size
                    tp_params = {
                        'symbol': symbol,
                        'side': tp_side,
                        'type': 'TAKE_PROFIT_MARKET',
                        'timeInForce': 'GTC',
                        'quantity': total_quantity,  # Total quantity for both entries
                        'stopPrice': tp_price,
                        'reduceOnly': True
                    }
                    # Only include positionSide if in Hedge mode
                    if is_hedge_mode:
                        tp_params['positionSide'] = position_side
                    tp_order = client.futures_create_order(**tp_params)
                    active_trades[symbol]['tp_order_id'] = tp_order.get('orderId')
                    logger.info(f"Take profit order created: {tp_order}")
                except Exception as e:
                    logger.error(f"Failed to create take profit order: {e}")
            elif existing_tp:
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
        return {'success': False, 'error': f'Binance API error: {e.message}'}
    except BinanceOrderException as e:
        logger.error(f"Binance order error: {e}")
        return {'success': False, 'error': f'Binance order error: {e.message}'}
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
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


@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'service': 'Binance Futures Webhook Service',
        'version': '1.0.0',
        'endpoints': {
            'webhook': '/webhook (POST)',
            'health': '/health (GET)'
        }
    }), 200


if __name__ == '__main__':
    # Check configuration
    if WEBHOOK_TOKEN == 'CHANGE_ME':
        logger.warning("WEBHOOK_TOKEN is not set! Using default value.")
    
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        logger.error("BINANCE_API_KEY and BINANCE_API_SECRET must be set!")
        exit(1)
    
    # Run Flask app
    # Use 0.0.0.0 to listen on all interfaces
    # Use a production WSGI server like gunicorn in production
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

