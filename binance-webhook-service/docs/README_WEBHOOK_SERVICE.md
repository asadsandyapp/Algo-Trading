# Binance Futures Webhook Service

A lightweight Python service that receives TradingView webhook signals and creates Binance Futures limit orders.

## Features

- ✅ Receives webhook signals from TradingView Pine Script
- ✅ Creates Binance Futures limit orders automatically
- ✅ Supports stop loss and take profit orders
- ✅ Prevents duplicate orders
- ✅ Lightweight (optimized for 1 CPU, 1 GB RAM servers)
- ✅ Health check endpoint
- ✅ Comprehensive logging

## Prerequisites

- Ubuntu Server
- Python 3.8+
- Binance Futures API Key and Secret
- TradingView webhook token

## Installation

### 1. Install Python and pip

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
```

### 2. Create service directory

```bash
sudo mkdir -p /opt/binance-webhook
sudo chown $USER:$USER /opt/binance-webhook
cd /opt/binance-webhook
```

### 3. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file or set environment variables:

```bash
export WEBHOOK_TOKEN="your_webhook_token_from_pine_script"
export BINANCE_API_KEY="your_binance_api_key"
export BINANCE_API_SECRET="your_binance_api_secret"
export BINANCE_TESTNET="false"  # Set to "true" for testnet
```

### 6. Test the service

```bash
python3 binance_webhook_service.py
```

The service should start on port 5000. Test with:

```bash
curl http://localhost:5000/health
```

## Systemd Service Setup

### 1. Copy service file

```bash
sudo cp webhook_service.service /etc/systemd/system/binance-webhook.service
```

### 2. Edit service file

```bash
sudo nano /etc/systemd/system/binance-webhook.service
```

Update the environment variables:
- `WEBHOOK_TOKEN`: Your webhook token (must match Pine Script)
- `BINANCE_API_KEY`: Your Binance API key
- `BINANCE_API_SECRET`: Your Binance API secret
- `BINANCE_TESTNET`: "true" for testnet, "false" for live

### 3. Reload systemd and start service

```bash
sudo systemctl daemon-reload
sudo systemctl enable binance-webhook
sudo systemctl start binance-webhook
```

### 4. Check service status

```bash
sudo systemctl status binance-webhook
```

### 5. View logs

```bash
# Service logs
sudo journalctl -u binance-webhook -f

# Application logs
tail -f /opt/binance-webhook/webhook_service.log
```

## Nginx Reverse Proxy (Optional but Recommended)

### 1. Install Nginx

```bash
sudo apt install nginx -y
```

### 2. Create Nginx configuration

```bash
sudo nano /etc/nginx/sites-available/binance-webhook
```

Add:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3. Enable site

```bash
sudo ln -s /etc/nginx/sites-available/binance-webhook /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 4. Setup SSL with Let's Encrypt (Recommended)

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```

## TradingView Webhook Configuration

In your Pine Script, update the webhook URL:

```pine
// In your alert settings
alertWebhookToken = "your_webhook_token_here"

// Webhook URL format
webhook_url = "https://your-domain.com/webhook"
// Or if using IP: "http://your-server-ip:5000/webhook"
```

## Webhook Payload Format

The service expects JSON payloads in this format (from your Pine Script):

```json
{
  "payload_version": 1,
  "token": "your_webhook_token",
  "event": "ENTRY",
  "order_type": "LIMIT",
  "order_subtype": "primary_entry",
  "signal_side": "LONG",
  "reduce_only": false,
  "symbol": "ETHUSDT.P",
  "entry_price": 3902.59,
  "stop_loss": 3724.27,
  "take_profit": 4036.33,
  "position_size": 100.0,
  "contract_quantity": 0.025,
  "timestamp": 1698768000
}
```

## API Endpoints

### POST /webhook
Receives TradingView webhook signals and creates orders.

### GET /health
Health check endpoint. Returns service status.

### GET /
Service information endpoint.

## Security Considerations

1. **Use HTTPS**: Always use HTTPS in production (setup with Let's Encrypt)
2. **Firewall**: Only allow necessary ports (80, 443, 22)
3. **API Keys**: Store API keys securely, never commit to git
4. **Webhook Token**: Use a strong, unique token
5. **Binance API Permissions**: Only enable Futures trading permissions, disable withdrawals

## Binance API Setup

1. Go to Binance → API Management
2. Create new API key
3. Enable "Enable Futures" permission
4. **IMPORTANT**: Do NOT enable "Enable Withdrawals"
5. Add IP whitelist (your server IP) for extra security
6. Copy API Key and Secret

## Testing

### Test with curl

```bash
curl -X POST http://localhost:5000/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "payload_version": 1,
    "token": "your_webhook_token",
    "event": "ENTRY",
    "signal_side": "LONG",
    "symbol": "ETHUSDT",
    "entry_price": 2000.0,
    "stop_loss": 1950.0,
    "take_profit": 2100.0,
    "contract_quantity": 0.01
  }'
```

## Troubleshooting

### Service won't start
- Check logs: `sudo journalctl -u binance-webhook -n 50`
- Verify environment variables are set
- Check Python path in service file

### Orders not creating
- Check Binance API key permissions
- Verify webhook token matches
- Check application logs: `tail -f webhook_service.log`
- Test Binance connection: `curl http://localhost:5000/health`

### High memory usage
- Reduce gunicorn workers (change `--workers 2` to `--workers 1`)
- Check for memory leaks in logs

## Monitoring

Monitor the service:

```bash
# Service status
sudo systemctl status binance-webhook

# Real-time logs
sudo journalctl -u binance-webhook -f

# Application logs
tail -f /opt/binance-webhook/webhook_service.log

# Check if service is responding
curl http://localhost:5000/health
```

## Updates

To update the service:

```bash
cd /opt/binance-webhook
source venv/bin/activate
pip install -r requirements.txt --upgrade
sudo systemctl restart binance-webhook
```

## Support

For issues or questions:
1. Check logs first
2. Verify configuration
3. Test with curl
4. Check Binance API status

