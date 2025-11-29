#!/bin/bash
# Test script for Binance Webhook Service

WEBHOOK_URL="${1:-http://localhost:5000/webhook}"
WEBHOOK_TOKEN="${2:-CHANGE_ME}"

echo "Testing webhook service at: $WEBHOOK_URL"
echo "Using token: $WEBHOOK_TOKEN"
echo ""

# Test LONG entry
echo "Testing LONG entry..."
curl -X POST "$WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"payload_version\": 1,
    \"token\": \"$WEBHOOK_TOKEN\",
    \"event\": \"ENTRY\",
    \"order_type\": \"LIMIT\",
    \"order_subtype\": \"primary_entry\",
    \"signal_side\": \"LONG\",
    \"reduce_only\": false,
    \"symbol\": \"ETHUSDT\",
    \"entry_price\": 2000.0,
    \"stop_loss\": 1950.0,
    \"take_profit\": 2100.0,
    \"position_size\": 100.0,
    \"contract_quantity\": 0.05,
    \"timestamp\": $(date +%s)
  }"

echo ""
echo ""
echo "Testing health endpoint..."
curl -X GET "${WEBHOOK_URL%/webhook}/health"

echo ""
echo ""
echo "Done!"

