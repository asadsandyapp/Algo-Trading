#!/bin/bash
# Setup Nginx Reverse Proxy for Binance Webhook Service
# This allows TradingView to send webhooks to port 80 (required)

set -e

echo "🔧 Setting up Nginx reverse proxy for webhook service..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run as root (use sudo)"
    exit 1
fi

# Install nginx if not installed
if ! command -v nginx &> /dev/null; then
    echo "📦 Installing nginx..."
    apt update
    apt install -y nginx
fi

# Copy nginx configuration
CONFIG_DIR="/opt/Algo-Trading/binance-webhook-service/config"
NGINX_SITE="/etc/nginx/sites-available/binance-webhook"
NGINX_ENABLED="/etc/nginx/sites-enabled/binance-webhook"

echo "📝 Creating nginx configuration..."
cp "$CONFIG_DIR/nginx-webhook.conf" "$NGINX_SITE"

# Create symlink to enable site
if [ -L "$NGINX_ENABLED" ]; then
    echo "⚠️  Site already enabled, removing old symlink..."
    rm "$NGINX_ENABLED"
fi

ln -s "$NGINX_SITE" "$NGINX_ENABLED"

# Remove default nginx site if it conflicts
if [ -L "/etc/nginx/sites-enabled/default" ]; then
    echo "⚠️  Disabling default nginx site..."
    rm /etc/nginx/sites-enabled/default
fi

# Test nginx configuration
echo "🧪 Testing nginx configuration..."
nginx -t

if [ $? -eq 0 ]; then
    echo "✅ Nginx configuration is valid"
    
    # Reload nginx
    echo "🔄 Reloading nginx..."
    systemctl reload nginx
    
    # Enable nginx to start on boot
    systemctl enable nginx
    
    echo ""
    echo "✅ Nginx reverse proxy setup complete!"
    echo ""
    echo "📋 Summary:"
    echo "   - Nginx listening on port 80"
    echo "   - Forwarding to Flask service on port 5000"
    echo "   - Webhook URL: http://YOUR_SERVER_IP/webhook"
    echo ""
    echo "🔍 Test the setup:"
    echo "   curl http://localhost/health"
    echo ""
else
    echo "❌ Nginx configuration test failed!"
    exit 1
fi






















