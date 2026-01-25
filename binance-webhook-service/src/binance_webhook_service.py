#!/usr/bin/env python3
"""
Binance Futures Webhook Service
Receives TradingView webhook signals and creates Binance Futures limit orders
Optimized for low-resource servers (1 CPU, 1 GB RAM)
"""
import os
import sys
import threading
import traceback

# Load environment variables from .env file (if present)
# This allows manual testing; systemd service uses EnvironmentFile which takes precedence
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    try:
        load_dotenv(env_path)
    except (PermissionError, IOError, FileNotFoundError):
        pass
except ImportError:
    pass

# Import core components
try:
    from core import app, client, logger
except Exception as e:
    import sys
    print(f"CRITICAL: Failed to import core module: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    raise

from config import (
    WEBHOOK_TOKEN, BINANCE_API_KEY, BINANCE_API_SECRET,
    ENTRY_SIZE_USD, LEVERAGE
)

# Import services to initialize background threads
from services.orders.order_manager import create_missing_tp_orders

# Import routes to register them with Flask app (routes are registered via @app.route decorator)
try:
    import api.routes  # This imports and registers all routes
    logger.info("Routes imported successfully")
except Exception as e:
    logger.error(f"Failed to import routes: {e}", exc_info=True)
    traceback.print_exc(file=sys.stderr)
    raise

# Import notifications for main block
from notifications.slack import send_slack_alert

# Ensure app is available at module level for gunicorn
# The app is imported from core module above: from core import app
# This makes 'app' available in this module's namespace
# Gunicorn accesses it via: binance_webhook_service:app
# The app object is a Flask WSGI application and is ready to use

# Final verification that app is accessible (for debugging)
try:
    if app is None:
        raise RuntimeError("Flask app is None")
    if not hasattr(app, 'wsgi_app'):
        raise RuntimeError("Flask app missing wsgi_app method")
    logger.info(f"Flask app verified: {app.name}, routes: {len(list(app.url_map.iter_rules()))}")
except Exception as e:
    logger.error(f"App verification failed: {e}", exc_info=True)
    traceback.print_exc(file=sys.stderr)
    raise

# Start background thread for TP creation (only if client is initialized)
# This runs when the module is imported (including by gunicorn workers)
# Moved after app verification to ensure app is available first
if client:
    try:
        tp_thread = threading.Thread(target=create_missing_tp_orders, daemon=True)
        tp_thread.start()
        logger.info("Background TP creation thread started")
    except Exception as e:
        logger.error(f"Failed to start background TP thread: {e}", exc_info=True)
else:
    logger.warning("Binance client not initialized - TP creation thread not started")

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
