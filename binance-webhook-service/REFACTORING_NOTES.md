# Refactoring Notes

## New Directory Structure

The codebase has been refactored into a production-ready modular structure:

```
src/
├── binance_webhook_service.py  # Main entry point (Flask app)
├── api/                        # API routes
│   └── __init__.py            # Flask routes (webhook, health, verify-account, check-tp, index)
├── config/                     # Configuration
│   └── __init__.py            # Environment variables and config
├── core/                       # Core initialization
│   └── __init__.py            # Flask app, Binance client, Gemini client, logging
├── models/                     # State management
│   └── __init__.py            # active_trades, recent_orders, caches
├── notifications/              # Notifications
│   └── __init__.py            # Slack notifications (alerts, signals, exits)
├── services/
│   ├── ai_validation/         # AI validation service
│   │   └── __init__.py        # Gemini API integration, signal validation
│   ├── orders/                 # Order management service
│   │   └── __init__.py        # Order creation, TP orders, position management
│   └── risk/                   # Risk management service
│       └── __init__.py        # Risk validation, calculations, volatility checks
└── utils/                      # Utility functions
    └── __init__.py            # Formatting, validation, Binance helpers
```

## Status

The following modules have been created:
- ✅ `models/__init__.py` - State management
- ✅ `utils/__init__.py` - Utility functions
- ✅ `notifications/__init__.py` - Slack notifications
- ✅ `services/risk/__init__.py` - Risk management
- ✅ `api/__init__.py` - API routes
- ⚠️ `services/ai_validation/__init__.py` - Placeholder (needs code extraction)
- ⚠️ `services/orders/__init__.py` - Placeholder (needs code extraction)

## Next Steps

1. Extract AI validation functions from `binance_webhook_service.py` to `services/ai_validation/__init__.py`
2. Extract order management functions from `binance_webhook_service.py` to `services/orders/__init__.py`
3. Update `binance_webhook_service.py` to import from modules
4. Test the service to ensure all functionality works

## Important Notes

- The systemd service uses `binance_webhook_service:app`, so the main file must export `app`
- All imports must use relative imports (e.g., `from ..core import app`)
- The original file has been backed up as `binance_webhook_service.py.backup`

