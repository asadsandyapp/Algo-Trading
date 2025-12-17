#!/bin/bash
# Complete fix and setup script for binance-webhook service
# This script will:
# - Update the systemd service file with correct paths
# - Create logs directory if missing
# - Set proper file permissions
# - Reload systemd and start the service

set -e

echo "🔧 Fixing Binance Webhook Service..."
echo ""

# 1. Copy updated service file
echo "1. Updating systemd service file..."
sudo cp /home/lenovo/Pine/Algo-Trading/binance-webhook-service/config/webhook_service.service /etc/systemd/system/binance-webhook.service

# 2. Create logs directory if it doesn't exist
echo "2. Checking logs directory..."
if [ ! -d "/opt/Algo-Trading/binance-webhook-service/logs" ]; then
    echo "   Creating logs directory..."
    sudo mkdir -p /opt/Algo-Trading/binance-webhook-service/logs
fi

# 3. Set proper permissions
echo "3. Setting file permissions..."
# Make sure www-data can read the files and execute gunicorn
sudo chown -R ubuntu:www-data /opt/Algo-Trading/binance-webhook-service/
sudo chmod -R 755 /opt/Algo-Trading/binance-webhook-service/
sudo chmod 644 /opt/Algo-Trading/binance-webhook-service/.env
sudo chmod 755 /opt/Algo-Trading/binance-webhook-service/venv/bin/gunicorn
sudo chmod 755 /opt/Algo-Trading/binance-webhook-service/src/binance_webhook_service.py
sudo chmod 777 /opt/Algo-Trading/binance-webhook-service/logs

# 4. Reload systemd
echo "4. Reloading systemd daemon..."
sudo systemctl daemon-reload

# 5. Verify service file
echo ""
echo "5. Verifying service configuration..."
echo "   Working Directory:"
grep "WorkingDirectory" /etc/systemd/system/binance-webhook.service
echo "   ExecStart (first 100 chars):"
grep "ExecStart" /etc/systemd/system/binance-webhook.service | cut -c1-100
echo ""

# 6. Start the service
echo "6. Starting service..."
sudo systemctl start binance-webhook

# 7. Wait a moment and check status
sleep 2
echo ""
echo "7. Service status:"
sudo systemctl status binance-webhook --no-pager -l | head -20

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Useful commands:"
echo "   View logs: sudo journalctl -u binance-webhook -f"
echo "   Check status: sudo systemctl status binance-webhook"
echo "   Restart: sudo systemctl restart binance-webhook"

