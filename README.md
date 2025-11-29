# Algo-Trading

Automated trading system integrating TradingView Pine Script indicators with Binance Futures API.

## 📦 Repository Structure

```
Algo-Trading/
├── binance-webhook-service/     # Binance Futures webhook service
│   ├── src/                     # Source code
│   ├── config/                  # Configuration files
│   ├── scripts/                 # Installation and utility scripts
│   ├── docs/                    # Documentation
│   └── logs/                    # Log files
├── tradingview-indicators/      # TradingView Pine Script indicators
│   └── Target.pine              # Main trading indicator
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. TradingView Indicator
- Located in `tradingview-indicators/Target.pine`
- Configure webhook URL in the indicator settings
- Set `alertWebhookToken` to match your service configuration

### 2. Webhook Service
- Located in `binance-webhook-service/`
- See [binance-webhook-service/README.md](binance-webhook-service/README.md) for detailed setup
- Quick install: `cd binance-webhook-service && ./scripts/install.sh`

## 🔧 Components

### Binance Webhook Service
Production-ready Python service that:
- Receives TradingView webhook signals
- Creates Binance Futures limit orders
- Manages stop loss and take profit orders
- Optimized for low-resource servers

### TradingView Indicators
Pine Script indicators that:
- Generate trading signals
- Calculate entry/exit points
- Send webhook alerts to the service

## 📚 Documentation

- [Webhook Service Documentation](binance-webhook-service/README.md)
- [Detailed Service Guide](binance-webhook-service/docs/README_WEBHOOK_SERVICE.md)

## ⚙️ Configuration

1. **Binance API Setup**
   - Create API key with Futures trading enabled
   - Disable withdrawals for security
   - Add IP whitelist

2. **Webhook Service**
   - Configure `.env` file with API credentials
   - Set webhook token
   - Setup systemd service

3. **TradingView**
   - Add indicator to chart
   - Configure webhook URL
   - Set alert conditions

## 🔒 Security

- Use HTTPS for webhook endpoints
- Store API keys securely
- Never commit credentials to git
- Use IP whitelisting on Binance
- Test with Binance Testnet first

## 📝 License

This project is for educational and personal use.

## ⚠️ Disclaimer

Trading cryptocurrencies involves substantial risk. Use this software at your own risk. Always test thoroughly before using real funds.

