#!/bin/bash
# Installation script for Binance Webhook Service

set -e

echo "Installing Binance Webhook Service..."

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
INSTALL_DIR="/opt/algo-trading/binance-webhook-service"

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo "Please do not run as root. The script will use sudo when needed."
   exit 1
fi

# Install system dependencies
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y python3 python3-pip python3-venv nginx

# Create service directory
echo "Creating service directory..."
sudo mkdir -p "$INSTALL_DIR"
sudo chown $USER:$USER "$INSTALL_DIR"

# Copy files
echo "Copying service files..."
cp -r "$SERVICE_DIR/src" "$INSTALL_DIR/"
cp "$SERVICE_DIR/requirements.txt" "$INSTALL_DIR/"
cp "$SERVICE_DIR/.env.example" "$INSTALL_DIR/.env.example" 2>/dev/null || true
sudo mkdir -p "$INSTALL_DIR/logs"
sudo chown $USER:$USER "$INSTALL_DIR/logs"

# Create virtual environment
echo "Setting up Python virtual environment..."
cd "$INSTALL_DIR"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Setup environment variables
echo "Setting up environment variables..."
if [ ! -f .env ]; then
    echo "Creating .env file from example..."
    cp .env.example .env 2>/dev/null || cat > .env << EOF
WEBHOOK_TOKEN=CHANGE_ME
BINANCE_API_KEY=
BINANCE_API_SECRET=
BINANCE_TESTNET=false
SERVICE_PORT=5000
SERVICE_HOST=0.0.0.0
LOG_LEVEL=INFO
LOG_FILE=logs/webhook_service.log
EOF
    echo "⚠️  Please edit $INSTALL_DIR/.env and add your credentials"
fi

# Setup systemd service
echo "Setting up systemd service..."
sudo cp "$SERVICE_DIR/config/webhook_service.service" /etc/systemd/system/binance-webhook.service
sudo systemctl daemon-reload

echo ""
echo "✅ Installation complete!"
echo ""
echo "📝 Next steps:"
echo "1. Edit $INSTALL_DIR/.env with your credentials:"
echo "   - WEBHOOK_TOKEN (must match TradingView)"
echo "   - BINANCE_API_KEY"
echo "   - BINANCE_API_SECRET"
echo "   - BINANCE_TESTNET (true for testing, false for live)"
echo ""
echo "2. Start the service:"
echo "   sudo systemctl start binance-webhook"
echo ""
echo "3. Enable auto-start:"
echo "   sudo systemctl enable binance-webhook"
echo ""
echo "4. Check status:"
echo "   sudo systemctl status binance-webhook"
echo ""
echo "5. View logs:"
echo "   sudo journalctl -u binance-webhook -f"
echo "   tail -f $INSTALL_DIR/logs/webhook_service.log"

