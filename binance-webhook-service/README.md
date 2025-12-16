# Binance Futures Webhook Service - Complete Guide

A production-ready Python service that receives TradingView webhook signals and automatically creates Binance Futures orders with intelligent duplicate prevention, position management, and risk controls.

## 🚀 Features

### Core Functionality
- ✅ Receives webhook signals from TradingView Pine Script
- ✅ Creates Binance Futures limit orders automatically
- ✅ **Creates both Entry 1 and Entry 2 orders simultaneously** when signal arrives
- ✅ Supports stop loss and take profit orders
- ✅ **Closes positions at market price** on EXIT signals
- ✅ **Intelligent duplicate prevention** - handles multiple alerts like a pro
- ✅ **$10 per entry with 20X leverage** (configurable)
- ✅ Lightweight and optimized for low-resource servers (1 CPU, 1 GB RAM)
- ✅ Health check endpoint for monitoring
- ✅ Comprehensive logging and error handling
- ✅ Thread-safe order processing
- ✅ Production-ready with systemd service

### Advanced Features
- ✅ **Duplicate Detection**: Prevents duplicate trades even with multiple alerts
- ✅ **Position Checking**: Verifies positions exist before closing
- ✅ **Order Cleanup**: Cancels unfilled Entry 2 orders when trade closes
- ✅ **Active Trade Tracking**: Monitors all open positions and orders
- ✅ **EXIT Cooldown**: Prevents duplicate EXIT processing
- ✅ **Automatic Leverage**: Sets 20X leverage automatically
- ✅ **One-Way & Hedge Mode**: Supports both position modes

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Binance API Setup](#binance-api-setup)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [TradingView Setup](#tradingview-setup)
6. [Trading Configuration](#trading-configuration)
7. [API Endpoints](#api-endpoints)
8. [Webhook Payload Format](#webhook-payload-format)
9. [How It Works](#how-it-works)
10. [Testing](#testing)
11. [Monitoring](#monitoring)
12. [Security](#security)
13. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Ubuntu Server** (18.04+) or similar Linux distribution
- **Python 3.8+**
- **Binance Futures account** with API access
- **TradingView account** with Pine Script
- **Domain name** (optional, for HTTPS)

---

## Binance API Setup

### Step 1: Create Binance API Key

1. **Log in to Binance**
   - Go to [https://www.binance.com](https://www.binance.com)
   - Log in to your account

2. **Navigate to API Management**
   - Click on your profile icon (top right)
   - Go to **"API Management"** or **"API"**
   - Or go directly to: [https://www.binance.com/en/my/settings/api-management](https://www.binance.com/en/my/settings/api-management)

3. **Create New API Key**
   - Click **"Create API"** button
   - Choose **"System generated"** (recommended for security)
   - Enter a label/name (e.g., "Trading Bot" or "Webhook Service")
   - Complete security verification (email, SMS, 2FA)

### Step 2: Configure API Permissions

**⚠️ CRITICAL: Only enable these permissions:**

✅ **Enable Futures** - REQUIRED (for futures trading)
✅ **Enable Reading** - REQUIRED (to check positions and orders)

❌ **DO NOT Enable Withdrawals** - NEVER enable this for security!

**Optional (for better security):**
- ✅ **Restrict access to trusted IPs only** - Add your server IP address

### Step 3: Save Your Credentials

After creating the API key, Binance will show you:

1. **API Key** - Copy this immediately (you'll see it only once)
2. **Secret Key** - Copy this immediately (you'll see it only once)

**⚠️ IMPORTANT:** 
- Save these keys securely
- You cannot view the Secret Key again after closing the window
- If you lose it, you'll need to delete and create a new API key

---

## Installation

### Quick Install

```bash
cd /home/lenovo/Pine/Algo-Trading/binance-webhook-service
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
cd /home/lenovo/Pine/Algo-Trading/binance-webhook-service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Configure environment:**
```bash
nano .env  # Create and edit with your credentials
```

4. **Setup systemd service:**
```bash
sudo cp config/webhook_service.service /etc/systemd/system/binance-webhook.service
sudo nano /etc/systemd/system/binance-webhook.service  # Update environment variables
sudo systemctl daemon-reload
sudo systemctl enable binance-webhook
sudo systemctl start binance-webhook
```

---

## Configuration

### Environment Variables

Create a `.env` file in the service directory:

```bash
# ============================================
# Binance API Credentials
# ============================================
# Get these from: https://www.binance.com/en/my/settings/api-management
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_secret_key_here

# ============================================
# Webhook Security Token
# ============================================
# Generate with: openssl rand -hex 32
# Must match the token in your TradingView Pine Script
WEBHOOK_TOKEN=your_secure_token_here

# ============================================
# Trading Mode
# ============================================
# true = Testnet (testing, no real money)
# false = Live Trading (real money, be careful!)
BINANCE_TESTNET=false

# ============================================
# Service Configuration
# ============================================
SERVICE_PORT=5000
SERVICE_HOST=0.0.0.0
LOG_LEVEL=INFO
```

### Generate Secure Webhook Token

```bash
# Generate a secure random token
openssl rand -hex 32
```

**Example tokens:**
- `a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6`
- `my_secure_trading_token_2024_xyz123`

### Configuration Options

#### Option 1: Using .env File (Recommended)

```bash
cd /home/lenovo/Pine/Algo-Trading/binance-webhook-service
nano .env
```

Add your credentials as shown above.

#### Option 2: Using Environment Variables

```bash
export BINANCE_API_KEY="your_api_key_here"
export BINANCE_API_SECRET="your_secret_key_here"
export WEBHOOK_TOKEN="your_secure_token_here"
export BINANCE_TESTNET="false"
```

#### Option 3: Using Systemd Service File

```bash
sudo nano /etc/systemd/system/binance-webhook.service
```

Add under `[Service]` section:

```ini
Environment="BINANCE_API_KEY=your_api_key_here"
Environment="BINANCE_API_SECRET=your_secret_key_here"
Environment="WEBHOOK_TOKEN=your_secure_token_here"
Environment="BINANCE_TESTNET=false"
```

---

## Trading Configuration

### Current Settings

The service is configured with:

- **Entry Size**: $10 per entry
- **Leverage**: 20X
- **Total Entries**: 2 (Primary Entry + DCA Entry)
- **Total Investment**: $20 ($10 × 2 entries)

### How Position Size is Calculated

For each entry:
- Position Value = Entry Size × Leverage = $10 × 20 = **$200**
- Quantity = Position Value / Entry Price

**Example:**
- Entry Price: $50,000 (BTCUSDT)
- Position Value: $200
- Quantity: $200 / $50,000 = 0.004 BTC

### Modifying Trading Settings

To change these settings, edit `src/binance_webhook_service.py`:

```python
# Trading configuration
ENTRY_SIZE_USD = 10.0  # Change this to your desired entry size
LEVERAGE = 20  # Change this to your desired leverage
TOTAL_ENTRIES = 2  # Primary entry + DCA entry
```

---

## TradingView Setup

### 1. Add Indicator to TradingView

1. Open TradingView
2. Go to Pine Editor
3. Copy your Pine Script code
4. Click "Save" and "Add to Chart"

### 2. Configure Webhook Token

In your Pine Script, set the webhook token to match your `.env` file:

```pine
alertWebhookToken = input.string("your_secure_token_here", "Webhook Token", 
    group="Alert Settings", 
    tooltip="Must match the token in .env file")
```

**⚠️ IMPORTANT:** The token must be EXACTLY the same in both places!

### 3. Configure Webhook URL

Set the webhook URL in your Pine Script:

```pine
// Option 1: Using server IP
webhook_url = "http://your-server-ip:5000/webhook"

// Option 2: Using domain name (recommended for production)
webhook_url = "https://your-domain.com/webhook"

// Option 3: Local testing
webhook_url = "http://localhost:5000/webhook"
```

### 4. Set Up Alerts

1. Right-click on your chart
2. Select "Add Alert"
3. Configure alert conditions
4. In "Webhook URL" field, use: `{{webhook_url}}`
5. In alert message, include your JSON payload

---

## API Endpoints

### `POST /webhook`

Receives TradingView webhook signals and creates orders.

**Request Body:**
```json
{
  "payload_version": 1,
  "token": "your_webhook_token",
  "event": "ENTRY",
  "order_subtype": "primary_entry",
  "signal_side": "LONG",
  "symbol": "ASTERUSDT.P",
  "timeframe": "2h",
  "entry_price": 0.80397281,
  "second_entry_price": 0.76377417,
  "stop_loss": 0.71677773,
  "take_profit": 0.81912082,
  "reduce_only": false
}
```

**Response:**
```json
{
  "success": true,
  "message": "Webhook received, processing order"
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

**Response:**
```json
{
  "service": "Binance Futures Webhook Service",
  "version": "1.0.0",
  "endpoints": {
    "webhook": "/webhook (POST)",
    "health": "/health (GET)"
  }
}
```

---

## Webhook Payload Format

### ENTRY Event

```json
{
  "payload_version": 1,
  "token": "your_webhook_token",
  "event": "ENTRY",
  "order_subtype": "primary_entry",
  "signal_side": "LONG",
  "symbol": "ASTERUSDT.P",
  "timeframe": "2h",
  "entry_price": 0.80397281,
  "average_entry_price": 0.78387349,
  "second_entry_price": 0.76377417,
  "second_entry_filled": false,
  "stop_loss": 0.71677773,
  "take_profit": 0.81912082,
  "reduce_only": false
}
```

**Fields:**
- `event`: `"ENTRY"` for opening trades
- `order_subtype`: `"primary_entry"` or `"dca_fill"` or `"second_entry"`
- `signal_side`: `"LONG"` or `"SHORT"`
- `symbol`: Trading pair (e.g., `"ASTERUSDT.P"`)
- `entry_price`: Primary entry price
- `second_entry_price`: DCA entry price (optional)
- `stop_loss`: Stop loss price
- `take_profit`: Take profit price

### EXIT Event

```json
{
  "payload_version": 1,
  "token": "your_webhook_token",
  "event": "EXIT",
  "signal_side": "LONG",
  "symbol": "ASTERUSDT.P",
  "timeframe": "2h"
}
```

**Fields:**
- `event`: `"EXIT"` for closing trades
- `signal_side`: `"LONG"` or `"SHORT"`
- `symbol`: Trading pair to close

---

## How It Works

### ENTRY Signal Processing

1. **Signal Received**: TradingView sends ENTRY signal
2. **Duplicate Check**: 
   - Checks if position already exists → Reject if yes
   - Checks active trades tracking → Reject if entries filled
   - Checks existing limit orders → Reject if orders exist
3. **Order Creation**:
   - **Primary Entry**: Creates Entry 1 limit order
   - **DCA Entry**: Creates Entry 2 limit order (if `second_entry_price` provided)
   - **TP/SL Orders**: Creates Take Profit and Stop Loss orders (total quantity)
4. **Leverage**: Automatically sets 20X leverage
5. **Tracking**: Records all order IDs for management

### EXIT Signal Processing

1. **Signal Received**: TradingView sends EXIT signal
2. **Duplicate Check**: Checks if EXIT was processed recently (30s cooldown)
3. **Position Check**: Verifies position exists before closing
4. **Close Position**: Closes position at market price
5. **Order Cleanup**: Cancels unfilled Entry 2 limit orders
6. **Tracking Cleanup**: Removes trade from active tracking

### Duplicate Prevention

The service uses multiple layers of duplicate prevention:

1. **Position Check**: If position exists, reject immediately
2. **Active Trades Tracking**: Tracks which entries are filled
3. **Order Check**: Checks for existing limit orders
4. **EXIT Cooldown**: Prevents duplicate EXIT processing (30 seconds)

### Order Management

- **Entry 1 & Entry 2**: Both created simultaneously on primary entry signal
- **TP/SL Orders**: Created with total quantity (Entry 1 + Entry 2)
- **Unfilled Orders**: Automatically canceled when trade closes
- **Order Tracking**: All order IDs stored for management

---

## Testing

### Test Service Health

```bash
curl http://localhost:5000/health
```

### Test Webhook (Manual)

```bash
curl -X POST http://localhost:5000/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "payload_version": 1,
    "token": "YOUR_TOKEN",
    "event": "ENTRY",
    "order_subtype": "primary_entry",
    "signal_side": "LONG",
    "symbol": "BTCUSDT",
    "entry_price": 50000.0,
    "second_entry_price": 49000.0,
    "stop_loss": 48000.0,
    "take_profit": 52000.0
  }'
```

### Test Webhook (Script)

```bash
cd scripts
./test_webhook.sh http://localhost:5000/webhook YOUR_TOKEN
```

### Test EXIT Signal

```bash
curl -X POST http://localhost:5000/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "payload_version": 1,
    "token": "YOUR_TOKEN",
    "event": "EXIT",
    "signal_side": "LONG",
    "symbol": "BTCUSDT"
  }'
```

---

## Monitoring

### Check Service Status

```bash
sudo systemctl status binance-webhook
```

### View Logs

```bash
# Systemd logs (real-time)
sudo journalctl -u binance-webhook -f

# Application logs
tail -f logs/webhook_service.log

# Last 100 lines
tail -n 100 logs/webhook_service.log
```

### Health Check

```bash
curl http://localhost:5000/health
```

### Monitor Active Trades

Check the logs for active trade tracking:
```bash
grep "active_trades" logs/webhook_service.log
```

---

## Security

### Best Practices

1. **Use HTTPS**: Setup SSL with Let's Encrypt for production
2. **Firewall**: Only allow necessary ports (80, 443, 22)
3. **API Keys**: Store securely, never commit to git
4. **Webhook Token**: Use strong, unique tokens
5. **IP Whitelisting**: Whitelist your server IP on Binance
6. **No Withdrawals**: Never enable withdrawal permission on Binance API
7. **Testnet First**: Always test on Testnet before going live
8. **Monitor Logs**: Regularly check logs for suspicious activity

### Environment File Security

```bash
# Set proper permissions on .env file
chmod 600 .env

# Never commit .env to git
echo ".env" >> .gitignore
```

### Nginx Reverse Proxy (Recommended)

For production, use Nginx as a reverse proxy with SSL:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## Troubleshooting

### Service Won't Start

**Check logs:**
```bash
sudo journalctl -u binance-webhook -n 50
```

**Common issues:**
- Environment variables not set correctly
- Python path incorrect in service file
- Port 5000 already in use
- Missing dependencies

**Solution:**
```bash
# Check Python path
which python3

# Check if port is in use
sudo netstat -tulpn | grep 5000

# Reinstall dependencies
pip install -r requirements.txt
```

### Orders Not Creating

**Check:**
1. Binance API key permissions (must have "Enable Futures")
2. Webhook token matches between service and TradingView
3. Application logs: `tail -f logs/webhook_service.log`
4. Binance connection: `curl http://localhost:5000/health`

**Common errors:**
- `"Invalid API-key"` → API key is incorrect
- `"Signature for this request is not valid"` → Secret key is incorrect
- `"IP address is not in whitelist"` → Add server IP to Binance whitelist
- `"Position already exists"` → Duplicate prevention working (this is normal)

### Duplicate Orders Being Created

**This should NOT happen** - the service has multiple duplicate prevention layers.

**If it does:**
1. Check logs for duplicate detection messages
2. Verify position checking is working: `curl http://localhost:5000/health`
3. Check active trades tracking in logs
4. Restart service: `sudo systemctl restart binance-webhook`

### EXIT Not Closing Position

**Check:**
1. Position actually exists on Binance
2. EXIT signal format is correct
3. Webhook token matches
4. Check logs for errors

**Common issues:**
- Position already closed manually
- EXIT signal format incorrect
- Binance API error (check logs)

### High Memory Usage

**Solutions:**
- Reduce tracking data (cleanup happens automatically)
- Check for memory leaks in logs
- Monitor with: `htop` or `free -h`
- Restart service periodically if needed

### "Failed to initialize Binance client"

**Check:**
- API key and secret are correct
- API key has "Enable Futures" permission
- IP whitelist is not blocking your server
- Testnet vs Live mode setting

---

## Project Structure

```
binance-webhook-service/
├── src/
│   └── binance_webhook_service.py  # Main service application
├── config/
│   └── webhook_service.service     # Systemd service file
├── scripts/
│   ├── install.sh                  # Installation script
│   └── test_webhook.sh             # Test script
├── logs/                           # Log files directory
├── .env                            # Environment variables (create this)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Important Reminders

1. **API Key** = Your username (can be seen again)
2. **Secret Key** = Your password (only shown once, save it!)
3. **Webhook Token** = Security token (create your own, must match TradingView)
4. **Testnet** = Testing mode (no real money) - Set `BINANCE_TESTNET=true`
5. **Live Trading** = Real money (be very careful!) - Set `BINANCE_TESTNET=false`
6. **Entry Size** = $10 per entry (configurable in code)
7. **Leverage** = 20X (configurable in code)
8. **Total Investment** = $20 per trade ($10 × 2 entries)

---

## Support & Links

- **Binance API Management**: [https://www.binance.com/en/my/settings/api-management](https://www.binance.com/en/my/settings/api-management)
- **Binance API Documentation**: [https://binance-docs.github.io/apidocs/futures/en/](https://binance-docs.github.io/apidocs/futures/en/)
- **Binance Testnet**: [https://testnet.binancefuture.com/](https://testnet.binancefuture.com/)

---

## Disclaimer

**⚠️ Trading cryptocurrencies involves significant risk. This service automates trading based on TradingView signals. Use at your own risk.**

- Always test thoroughly with Binance Testnet before going live
- Start with small position sizes
- Monitor your trades regularly
- Never invest more than you can afford to lose
- The service creators are not responsible for any financial losses

---

**Version**: 1.0.0  
**Last Updated**: December 2024

