# Algo Trading - Binance Futures Webhook Service

Automated trading service that receives TradingView webhook signals and executes Binance Futures limit orders.

## 🚀 Features

- ✅ Receives webhook signals from TradingView Pine Script
- ✅ Creates Binance Futures limit orders automatically
- ✅ Supports stop loss and take profit orders
- ✅ Prevents duplicate orders
- ✅ Lightweight (optimized for 1 CPU, 1 GB RAM servers)
- ✅ Health check endpoint
- ✅ Comprehensive logging
- ✅ Systemd service integration

## 📋 Prerequisites

- Ubuntu Server (or similar Linux distribution)
- Python 3.8+
- Binance Futures API Key and Secret
- TradingView webhook token

## 🛠️ Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/asadsandyapp/Algo-Trading.git
cd Algo-Trading

# Run installation script
chmod +x install.sh
./install.sh
```

### Manual Installation

See [README_WEBHOOK_SERVICE.md](README_WEBHOOK_SERVICE.md) for detailed installation instructions.

## ⚙️ Configuration

1. **Set up environment variables:**

```bash
cd /opt/binance-webhook
nano .env
```

Add your credentials:
```
WEBHOOK_TOKEN=your_webhook_token_from_pine_script
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
BINANCE_TESTNET=false
```

2. **Update systemd service:**

```bash
sudo nano /etc/systemd/system/binance-webhook.service
```

Update the environment variables in the service file.

3. **Start the service:**

```bash
sudo systemctl start binance-webhook
sudo systemctl enable binance-webhook
```

## 🔗 TradingView Integration

In your Pine Script, set the webhook URL:

```pine
webhook_url = "http://your-server-ip:5000/webhook"
// Or with domain: "https://your-domain.com/webhook"
```

Make sure the `alertWebhookToken` in your Pine Script matches the `WEBHOOK_TOKEN` in the service.

## 📡 API Endpoints

- `POST /webhook` - Receives TradingView webhook signals
- `GET /health` - Health check endpoint
- `GET /` - Service information

## 🔒 Security

- Use HTTPS in production (setup with Let's Encrypt)
- Configure firewall (only allow necessary ports)
- Store API keys securely (never commit to git)
- Use strong webhook tokens
- Enable IP whitelisting on Binance API

## 📝 Files

- `binance_webhook_service.py` - Main service application
- `requirements.txt` - Python dependencies
- `webhook_service.service` - Systemd service file
- `install.sh` - Automated installation script
- `test_webhook.sh` - Test script
- `README_WEBHOOK_SERVICE.md` - Detailed documentation

## 🧪 Testing

```bash
# Test webhook endpoint
./test_webhook.sh http://localhost:5000/webhook YOUR_TOKEN

# Check health
curl http://localhost:5000/health
```

## 📊 Monitoring

```bash
# Service status
sudo systemctl status binance-webhook

# View logs
sudo journalctl -u binance-webhook -f
tail -f /opt/binance-webhook/webhook_service.log
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## ⚠️ Disclaimer

This software is for educational purposes only. Trading cryptocurrencies involves substantial risk. Use at your own risk.

## 📧 Contact

For issues or questions, please open an issue on GitHub.

