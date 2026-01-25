"""
State management module for Binance Webhook Service
Manages global state like active_trades, recent_orders, etc.
"""
import time
from typing import Dict, Any

# Track active trades per symbol
active_trades: Dict[str, Dict[str, Any]] = {}

# Track recent orders to prevent duplicates
recent_orders: Dict[str, float] = {}
ORDER_COOLDOWN = 60  # seconds

# Track recent EXIT events to prevent duplicate processing
recent_exits: Dict[str, float] = {}
EXIT_COOLDOWN = 30  # seconds

# Account balance cache (to reduce API calls)
account_balance_cache: Dict[str, Any] = {'balance': None, 'timestamp': 0}
BALANCE_CACHE_TTL = 60  # Cache balance for 1 minute

# Validation cache
validation_cache: Dict[str, tuple] = {}
VALIDATION_CACHE_TTL = 600  # Cache validation for 10 minutes

