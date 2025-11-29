#!/bin/bash
# Installation script for Binance Webhook Service

set -e

echo "Installing Binance Webhook Service..."

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
sudo mkdir -p /opt/binance-webhook
sudo chown $USER:$USER /opt/binance-webhook

# Copy files
echo "Copying service files..."
cp binance_webhook_service.py /opt/binance-webhook/
cp requirements.txt /opt/binance-webhook/

# Create virtual environment
echo "Setting up Python virtual environment..."
cd /opt/binance-webhook
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Setup environment variables
echo "Setting up environment variables..."
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
WEBHOOK_TOKEN=CHANGE_ME
BINANCE_API_KEY=
BINANCE_API_SECRET=
BINANCE_TESTNET=false
EOF
    echo "Please edit /opt/binance-webhook/.env and add your credentials"
fi

# Setup systemd service
echo "Setting up systemd service..."
sudo cp webhook_service.service /etc/systemd/system/binance-webhook.service
echo "Please edit /etc/systemd/system/binance-webhook.service and update environment variables"

# Reload systemd
sudo systemctl daemon-reload

echo ""
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Edit /opt/binance-webhook/.env with your credentials"
echo "2. Edit /etc/systemd/system/binance-webhook.service with your credentials"
echo "3. Start the service: sudo systemctl start binance-webhook"
echo "4. Enable auto-start: sudo systemctl enable binance-webhook"
echo "5. Check status: sudo systemctl status binance-webhook"

