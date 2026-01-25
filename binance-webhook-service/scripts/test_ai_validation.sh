#!/bin/bash
# Test script for AI Signal Validation
# This script sends test signals to verify AI validation is working

WEBHOOK_URL="${1:-http://localhost:5000/webhook}"
WEBHOOK_TOKEN="${2:-CHANGE_ME}"

echo "=========================================="
echo "Testing AI Signal Validation"
echo "=========================================="
echo "Webhook URL: $WEBHOOK_URL"
echo ""

# Test 1: Send a valid ENTRY signal (should be validated by AI)
echo "Test 1: Sending ENTRY signal (should trigger AI validation)..."
curl -X POST "$WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"payload_version\": 1,
    \"token\": \"$WEBHOOK_TOKEN\",
    \"event\": \"ENTRY\",
    \"order_subtype\": \"primary_entry\",
    \"signal_side\": \"LONG\",
    \"symbol\": \"BTCUSDT\",
    \"timeframe\": \"1h\",
    \"entry_price\": 50000.0,
    \"second_entry_price\": 49500.0,
    \"stop_loss\": 48000.0,
    \"take_profit\": 52000.0
  }"

echo ""
echo ""
echo "Check the logs for AI validation messages:"
echo "  - Look for: '🤖 Validating signal with AI'"
echo "  - Look for: '✅ AI Validation APPROVED' or '🚫 AI Validation REJECTED'"
echo ""
echo "To view logs, run:"
echo "  sudo journalctl -u binance-webhook -f | grep -E 'AI|Validation|🤖|✅|🚫'"
echo ""
echo "=========================================="

















