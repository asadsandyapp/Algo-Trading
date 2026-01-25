#!/usr/bin/env python3
"""Test script to verify imports work correctly"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all imports work"""
    print("Testing import structure...")
    print("=" * 60)
    
    tests = []
    
    # Test 1: Core imports
    try:
        from core import app, client, logger
        tests.append(("Core imports", True, f"app={type(app).__name__}, client={'initialized' if client else 'None'}"))
    except Exception as e:
        tests.append(("Core imports", False, str(e)))
    
    # Test 2: Config imports
    try:
        from config import WEBHOOK_TOKEN, BINANCE_API_KEY, ENTRY_SIZE_USD, LEVERAGE
        tests.append(("Config imports", True, f"ENTRY_SIZE_USD={ENTRY_SIZE_USD}, LEVERAGE={LEVERAGE}"))
    except Exception as e:
        tests.append(("Config imports", False, str(e)))
    
    # Test 3: Routes import (this registers routes)
    try:
        import api.routes
        # Check if routes are registered
        route_count = len([rule for rule in app.url_map.iter_rules()])
        tests.append(("Routes import", True, f"{route_count} routes registered"))
    except Exception as e:
        tests.append(("Routes import", False, str(e)))
    
    # Test 4: Services imports
    try:
        from services.orders.order_manager import create_missing_tp_orders
        tests.append(("Order manager import", True, "create_missing_tp_orders imported"))
    except Exception as e:
        tests.append(("Order manager import", False, str(e)))
    
    try:
        from services.ai_validation.validator import validate_signal_with_ai
        tests.append(("AI validator import", True, "validate_signal_with_ai imported"))
    except Exception as e:
        tests.append(("AI validator import", False, str(e)))
    
    try:
        from services.risk.risk_manager import validate_risk_per_trade
        tests.append(("Risk manager import", True, "validate_risk_per_trade imported"))
    except Exception as e:
        tests.append(("Risk manager import", False, str(e)))
    
    # Test 5: Notifications
    try:
        from notifications.slack import send_slack_alert
        tests.append(("Notifications import", True, "send_slack_alert imported"))
    except Exception as e:
        tests.append(("Notifications import", False, str(e)))
    
    # Test 6: Utils
    try:
        from utils.helpers import format_symbol, safe_float
        tests.append(("Utils import", True, "format_symbol, safe_float imported"))
    except Exception as e:
        tests.append(("Utils import", False, str(e)))
    
    # Test 7: Models
    try:
        from models.state import active_trades, recent_orders
        tests.append(("Models import", True, "active_trades, recent_orders imported"))
    except Exception as e:
        tests.append(("Models import", False, str(e)))
    
    # Print results
    print("\nTest Results:")
    print("-" * 60)
    passed = 0
    failed = 0
    
    for name, success, details in tests:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
        if not success:
            print(f"    Error: {details}")
            failed += 1
        else:
            print(f"    {details}")
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Summary: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n✅ All imports successful! Structure is correct.")
        return True
    else:
        print(f"\n❌ {failed} import(s) failed. Check errors above.")
        return False

if __name__ == '__main__':
    success = test_imports()
    sys.exit(0 if success else 1)
