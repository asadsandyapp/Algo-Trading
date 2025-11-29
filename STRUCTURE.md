# Project Structure

## 📁 Directory Layout

```
Algo-Trading/
│
├── binance-webhook-service/          # Binance Futures Webhook Service
│   ├── src/                          # Source code
│   │   └── binance_webhook_service.py
│   │
│   ├── config/                       # Configuration files
│   │   └── webhook_service.service   # Systemd service file
│   │
│   ├── scripts/                      # Utility scripts
│   │   ├── install.sh                # Installation script
│   │   └── test_webhook.sh           # Test script
│   │
│   ├── docs/                         # Documentation
│   │   └── README_WEBHOOK_SERVICE.md # Detailed documentation
│   │
│   ├── logs/                         # Log files (created at runtime)
│   │
│   ├── .env.example                  # Environment variables template
│   ├── .gitignore                    # Git ignore rules
│   ├── requirements.txt              # Python dependencies
│   └── README.md                     # Service README
│
├── tradingview-indicators/           # TradingView Pine Script indicators
│   └── Target.pine                   # Main trading indicator
│
├── .gitignore                        # Root git ignore
├── README.md                         # Main project README
└── STRUCTURE.md                      # This file
```

## 🔧 Component Details

### Binance Webhook Service
- **Location**: `binance-webhook-service/`
- **Purpose**: Receives TradingView webhooks and creates Binance Futures orders
- **Technology**: Python 3.8+, Flask, python-binance
- **Deployment**: Systemd service on Ubuntu server

### TradingView Indicators
- **Location**: `tradingview-indicators/`
- **Purpose**: Pine Script indicators that generate trading signals
- **Technology**: Pine Script v5
- **Usage**: Import into TradingView charts

## 📝 File Descriptions

### Service Files
- `binance_webhook_service.py` - Main Flask application
- `webhook_service.service` - Systemd service configuration
- `requirements.txt` - Python package dependencies
- `.env.example` - Environment variables template

### Scripts
- `install.sh` - Automated installation script
- `test_webhook.sh` - Webhook testing script

### Documentation
- `README.md` - Quick start guide
- `README_WEBHOOK_SERVICE.md` - Detailed service documentation
- `STRUCTURE.md` - This file

## 🚀 Deployment Paths

### Development
- Service runs from: `Algo-Trading/binance-webhook-service/`
- Virtual environment: `binance-webhook-service/venv/`

### Production
- Service installed to: `/opt/algo-trading/binance-webhook-service/`
- Virtual environment: `/opt/algo-trading/binance-webhook-service/venv/`
- Systemd service: `/etc/systemd/system/binance-webhook.service`
- Logs: `/opt/algo-trading/binance-webhook-service/logs/`

## 🔐 Security Files (Not in Git)

- `.env` - Contains API keys and secrets (gitignored)
- `logs/*.log` - Application logs (gitignored)
- `venv/` - Python virtual environment (gitignored)

## 📦 Installation Flow

1. Clone repository
2. Navigate to `binance-webhook-service/`
3. Run `scripts/install.sh`
4. Configure `.env` file
5. Start systemd service

## 🔄 Update Flow

1. Pull latest changes from git
2. Update dependencies: `pip install -r requirements.txt --upgrade`
3. Restart service: `sudo systemctl restart binance-webhook`

