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


def create_limit_order(signal_data):
    """Create a Binance Futures limit order"""
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
        
        # Verify token
        if not verify_webhook_token(token):
            logger.warning(f"Invalid webhook token received")
            return {'success': False, 'error': 'Invalid token'}
        
        # Only process ENTRY events
        if event != 'ENTRY':
            logger.info(f"Ignoring event: {event}")
            return {'success': False, 'error': f'Event {event} not processed'}
        
        # Get order side
        side = get_order_side(signal_side)
        position_side = get_position_side(signal_side)
        
        # Check for duplicate orders (same symbol + price within cooldown)
        order_key = f"{symbol}_{entry_price}_{side}"
        current_time = time.time()
        if order_key in recent_orders:
            if current_time - recent_orders[order_key] < ORDER_COOLDOWN:
                logger.info(f"Duplicate order prevented: {order_key}")
                return {'success': False, 'error': 'Duplicate order prevented'}
        
        # Validate entry price
        if entry_price <= 0:
            return {'success': False, 'error': 'Invalid entry price'}
        
        # Get symbol info for precision
        exchange_info = client.futures_exchange_info()
        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
        
        if not symbol_info:
            return {'success': False, 'error': f'Symbol {symbol} not found'}
        
        # Get price precision
        price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
        tick_size = float(price_filter['tickSize']) if price_filter else 0.01
        
        # Round price to tick size
        entry_price = round(entry_price / tick_size) * tick_size
        
        # Calculate quantity (you can customize this based on your risk management)
        # For now, using a fixed quantity or from signal data
        quantity = float(signal_data.get('contract_quantity', 0))
        if quantity <= 0:
            # Calculate based on position size if available
            position_size = float(signal_data.get('position_size', 0))
            if position_size > 0 and entry_price > 0:
                quantity = position_size / entry_price
            else:
                # Default to minimum quantity
                lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                min_qty = float(lot_size_filter['minQty']) if lot_size_filter else 0.001
                quantity = min_qty
        
        # Round quantity to step size
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        step_size = float(lot_size_filter['stepSize']) if lot_size_filter else 0.001
        quantity = round(quantity / step_size) * step_size
        
        # Create limit order
        order_params = {
            'symbol': symbol,
            'side': side,
            'type': 'LIMIT',
            'timeInForce': 'GTC',  # Good Till Cancel
            'quantity': quantity,
            'price': entry_price,
            'positionSide': position_side,
        }
        
        # Add reduceOnly if it's an exit order
        if reduce_only:
            order_params['reduceOnly'] = True
        
        logger.info(f"Creating order: {order_params}")
        
        # Place order
        order_result = client.futures_create_order(**order_params)
        
        # Track order
        recent_orders[order_key] = current_time
        
        # Clean old orders from tracking
        recent_orders.clear() if len(recent_orders) > 1000 else None
        
        logger.info(f"Order created successfully: {order_result}")
        
        # Optionally create stop loss and take profit orders
        if stop_loss and stop_loss > 0:
            try:
                sl_side = 'SELL' if side == 'BUY' else 'BUY'
                sl_order = client.futures_create_order(
                    symbol=symbol,
                    side=sl_side,
                    type='STOP_MARKET',
                    timeInForce='GTC',
                    quantity=quantity,
                    stopPrice=round(stop_loss / tick_size) * tick_size,
                    positionSide=position_side,
                    reduceOnly=True
                )
                logger.info(f"Stop loss order created: {sl_order}")
            except Exception as e:
                logger.error(f"Failed to create stop loss order: {e}")
        
        if take_profit and take_profit > 0:
            try:
                tp_side = 'SELL' if side == 'BUY' else 'BUY'
                tp_order = client.futures_create_order(
                    symbol=symbol,
                    side=tp_side,
                    type='TAKE_PROFIT_MARKET',
                    timeInForce='GTC',
                    quantity=quantity,
                    stopPrice=round(take_profit / tick_size) * tick_size,
                    positionSide=position_side,
                    reduceOnly=True
                )
                logger.info(f"Take profit order created: {tp_order}")
            except Exception as e:
                logger.error(f"Failed to create take profit order: {e}")
        
        return {
            'success': True,
            'order_id': order_result.get('orderId'),
            'symbol': symbol,
            'side': side,
            'price': entry_price,
            'quantity': quantity
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

