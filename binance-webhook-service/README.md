# Binance Futures Webhook Service

A production-ready Python service that receives TradingView webhook signals and automatically creates Binance Futures limit orders.

## 🚀 Features

- ✅ Receives webhook signals from TradingView Pine Script
- ✅ Creates Binance Futures limit orders automatically
- ✅ Supports stop loss and take profit orders
- ✅ Prevents duplicate orders with cooldown mechanism
- ✅ Lightweight and optimized for low-resource servers (1 CPU, 1 GB RAM)
- ✅ Health check endpoint for monitoring
- ✅ Comprehensive logging and error handling
- ✅ Thread-safe order processing
- ✅ Production-ready with systemd service

## 📁 Project Structure

```
binance-webhook-service/
├── src/
│   └── binance_webhook_service.py  # Main service application
├── config/
│   └── webhook_service.service      # Systemd service file
├── scripts/
│   ├── install.sh                   # Installation script
│   └── test_webhook.sh              # Test script
├── docs/
│   └── README_WEBHOOK_SERVICE.md   # Detailed documentation
├── logs/                            # Log files directory
├── .env.example                     # Environment variables template
├── .gitignore                       # Git ignore rules
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🛠️ Installation

### Quick Install

```bash
cd binance-webhook-service
chmod +x scripts/install.sh
./scripts/install.sh
```

### Manual Install

1. **Install system dependencies:**
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv nginx
```

2. **Setup Python environment:**
```bash
cd binance-webhook-service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Configure environment:**
```bash
cp .env.example .env
nano .env  # Edit with your credentials
```

4. **Setup systemd service:**
```bash
sudo cp config/webhook_service.service /etc/systemd/system/binance-webhook.service
sudo nano /etc/systemd/system/binance-webhook.service  # Update environment variables
sudo systemctl daemon-reload
sudo systemctl enable binance-webhook
sudo systemctl start binance-webhook
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the service directory:

```bash
WEBHOOK_TOKEN=your_webhook_token_from_pine_script
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
BINANCE_TESTNET=false
SERVICE_PORT=5000
SERVICE_HOST=0.0.0.0
LOG_LEVEL=INFO
```

### Binance API Setup

1. Go to [Binance API Management](https://www.binance.com/en/my/settings/api-management)
2. Create new API key
3. Enable **"Enable Futures"** permission
4. **⚠️ IMPORTANT:** Do NOT enable "Enable Withdrawals"
5. Add IP whitelist (your server IP) for extra security
6. Copy API Key and Secret

### TradingView Configuration

In your Pine Script, set the webhook URL:

```pine
alertWebhookToken = "your_webhook_token_here"
webhook_url = "http://your-server-ip:5000/webhook"
// Or with domain: "https://your-domain.com/webhook"
```

## 📡 API Endpoints

### `POST /webhook`
Receives TradingView webhook signals and creates orders.

**Request Body:**
```json
{
  "payload_version": 1,
  "token": "your_webhook_token",
  "event": "ENTRY",
  "signal_side": "LONG",
  "symbol": "ETHUSDT",
  "entry_price": 2000.0,
  "stop_loss": 1950.0,
  "take_profit": 2100.0,
  "contract_quantity": 0.05
}
```

### `GET /health`
Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "binance": "connected",
  "timestamp": "2024-01-01T00:00:00"
}
```

### `GET /`
Service information endpoint.

## 🧪 Testing

### Test with curl

```bash
cd scripts
./test_webhook.sh http://localhost:5000/webhook YOUR_TOKEN
```

### Manual test

```bash
curl -X POST http://localhost:5000/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "payload_version": 1,
    "token": "YOUR_TOKEN",
    "event": "ENTRY",
    "signal_side": "LONG",
    "symbol": "ETHUSDT",
    "entry_price": 2000.0,
    "stop_loss": 1950.0,
    "take_profit": 2100.0,
    "contract_quantity": 0.05
  }'
```

## 📊 Monitoring

### Check service status
```bash
sudo systemctl status binance-webhook
```

### View logs
```bash
# Systemd logs
sudo journalctl -u binance-webhook -f

# Application logs
tail -f logs/webhook_service.log
```

### Health check
```bash
curl http://localhost:5000/health
```

## 🔒 Security Best Practices

1. **Use HTTPS** - Setup SSL with Let's Encrypt
2. **Firewall** - Only allow necessary ports (80, 443, 22)
3. **API Keys** - Store securely, never commit to git
4. **Webhook Token** - Use strong, unique tokens
5. **IP Whitelisting** - Whitelist your server IP on Binance
6. **No Withdrawals** - Never enable withdrawal permission on Binance API

## 🐛 Troubleshooting

### Service won't start
- Check logs: `sudo journalctl -u binance-webhook -n 50`
- Verify environment variables in service file
- Check Python path: `which python3`

### Orders not creating
- Verify Binance API key permissions
- Check webhook token matches
- Review application logs: `tail -f logs/webhook_service.log`
- Test Binance connection: `curl http://localhost:5000/health`

### High memory usage
- Reduce gunicorn workers in service file
- Check for memory leaks in logs
- Monitor with: `htop` or `free -h`

## 📝 License

This project is part of the Algo-Trading repository.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For issues or questions:
1. Check the logs first
2. Verify configuration
3. Test with curl
4. Check Binance API status

---

**⚠️ Disclaimer:** Trading cryptocurrencies involves risk. Use this service at your own risk. Always test thoroughly with Binance Testnet before going live.

