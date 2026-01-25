"""
API routes module for Binance Webhook Service
Handles all Flask routes and endpoints
"""
import json
import threading
from datetime import datetime
from flask import request, jsonify
try:
    # Try relative import first (when imported as package)
    from ..core import app, client, logger
    from ..models.state import active_trades
    from ..notifications.slack import send_slack_alert
    from ..services.orders.order_manager import create_limit_order
except ImportError:
    # Fall back to absolute import (when src/ is in Python path)
    from core import app, client, logger
    from models.state import active_trades
    from notifications.slack import send_slack_alert
    from services.orders.order_manager import create_limit_order


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

