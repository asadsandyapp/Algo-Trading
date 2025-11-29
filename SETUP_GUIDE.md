# Setup Guide - Algo Trading System

Complete setup guide for the Algo Trading system with Binance Futures integration.

## 🎯 Overview

This system consists of:
1. **TradingView Indicator** - Generates trading signals
2. **Webhook Service** - Receives signals and creates Binance orders

## 📋 Prerequisites

- Ubuntu Server (18.04+)
- Python 3.8+
- Binance Futures account
- TradingView account
- Domain name (optional, for HTTPS)

## 🔧 Step 1: Binance API Setup

1. Go to [Binance API Management](https://www.binance.com/en/my/settings/api-management)
2. Click "Create API"
3. Enable **"Enable Futures"** permission
4. **⚠️ CRITICAL:** Do NOT enable "Enable Withdrawals"
5. Add IP whitelist (your server's public IP)
6. Save API Key and Secret Key securely

## 🖥️ Step 2: Server Setup

### Clone Repository

```bash
cd /opt
sudo git clone https://github.com/asadsandyapp/Algo-Trading.git
sudo chown -R $USER:$USER Algo-Trading
cd Algo-Trading
```

### Install Webhook Service

```bash
cd binance-webhook-service
chmod +x scripts/install.sh
./scripts/install.sh
```

### Configure Environment

```bash
nano /opt/algo-trading/binance-webhook-service/.env
```

Add your credentials:
```bash
WEBHOOK_TOKEN=your_secure_token_here
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
BINANCE_TESTNET=false  # Set to true for testing
```

### Start Service

```bash
sudo systemctl start binance-webhook
sudo systemctl enable binance-webhook
sudo systemctl status binance-webhook
```

### Test Service

```bash
# Health check
curl http://localhost:5000/health

# Test webhook (use your token)
cd binance-webhook-service/scripts
./test_webhook.sh http://localhost:5000/webhook YOUR_TOKEN
```

## 📊 Step 3: TradingView Setup

### Add Indicator

1. Open TradingView
2. Go to Pine Editor
3. Copy content from `tradingview-indicators/Target.pine`
4. Click "Save" and "Add to Chart"

### Configure Webhook

1. In indicator settings, find "Alert Settings"
2. Set `alertWebhookToken` to match your `.env` file
3. Set webhook URL:
   - Local testing: `http://your-server-ip:5000/webhook`
   - Production: `https://your-domain.com/webhook`

### Create Alert

1. Right-click on chart → "Add Alert"
2. Condition: Select your indicator
3. Webhook URL: Your service URL
4. Message: Leave default (JSON payload)
5. Save alert

## 🌐 Step 4: Nginx Setup (Optional but Recommended)

### Install Nginx

```bash
sudo apt install nginx -y
```

### Create Nginx Config

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

### Enable Site

```bash
sudo ln -s /etc/nginx/sites-available/binance-webhook /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Setup SSL (Recommended)

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```

## ✅ Step 5: Verification

### Check Service Status

```bash
sudo systemctl status binance-webhook
```

### View Logs

```bash
# Systemd logs
sudo journalctl -u binance-webhook -f

# Application logs
tail -f /opt/algo-trading/binance-webhook-service/logs/webhook_service.log
```

### Test End-to-End

1. Trigger a signal in TradingView
2. Check service logs for received webhook
3. Verify order created on Binance (Testnet first!)

## 🔒 Security Checklist

- [ ] HTTPS enabled (Let's Encrypt)
- [ ] Firewall configured (only 22, 80, 443 open)
- [ ] API keys stored in `.env` (not in code)
- [ ] Webhook token is strong and unique
- [ ] Binance IP whitelist configured
- [ ] Withdrawals disabled on Binance API
- [ ] Service runs as non-root user
- [ ] Logs are monitored

## 🧪 Testing with Binance Testnet

1. Set `BINANCE_TESTNET=true` in `.env`
2. Get Testnet API keys from [Binance Testnet](https://testnet.binancefuture.com/)
3. Update `.env` with testnet credentials
4. Restart service: `sudo systemctl restart binance-webhook`
5. Test thoroughly before going live

## 📈 Monitoring

### Health Check

```bash
curl https://your-domain.com/health
```

### Service Monitoring

```bash
# Check if service is running
systemctl is-active binance-webhook

# View recent logs
journalctl -u binance-webhook --since "1 hour ago"
```

## 🐛 Troubleshooting

### Service won't start
```bash
sudo journalctl -u binance-webhook -n 50
# Check for Python errors or missing dependencies
```

### Orders not creating
1. Check Binance API permissions
2. Verify webhook token matches
3. Check application logs
4. Test with curl manually

### High memory usage
- Reduce gunicorn workers in service file
- Monitor with `htop`

## 📚 Additional Resources

- [Service Documentation](binance-webhook-service/README.md)
- [Detailed Service Guide](binance-webhook-service/docs/README_WEBHOOK_SERVICE.md)
- [Project Structure](STRUCTURE.md)

## ⚠️ Important Notes

1. **Always test with Binance Testnet first**
2. **Start with small position sizes**
3. **Monitor logs regularly**
4. **Keep API keys secure**
5. **Never enable withdrawals on Binance API**

---

**Ready to trade?** Make sure you've completed all steps and tested thoroughly!

