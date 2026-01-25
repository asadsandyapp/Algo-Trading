"""
Risk management service for Binance Webhook Service
Handles risk validation, calculations, and volatility checks
"""
import time
from typing import Tuple, Optional, Dict
try:
    # Try relative import first (when imported as package)
    from ...config import MAX_RISK_PERCENT
    from ...core import client, logger
    from ...models.state import account_balance_cache, BALANCE_CACHE_TTL, active_trades
    from ...utils.helpers import safe_float
except ImportError:
    # Fall back to absolute import (when src/ is in Python path)
    from config import MAX_RISK_PERCENT
    from core import client, logger
    from models.state import account_balance_cache, BALANCE_CACHE_TTL, active_trades
    from utils.helpers import safe_float


def get_account_balance(cached=True):
    """Get account balance with caching to reduce API calls
    
    Args:
        cached: If True, use cached balance if available and fresh
    
    Returns:
        float: Account balance in USD, or None if error
    """
    current_time = time.time()
    
    # Check cache if enabled
    if cached and account_balance_cache['balance'] is not None:
        if current_time - account_balance_cache['timestamp'] < BALANCE_CACHE_TTL:
            return account_balance_cache['balance']
    
    try:
        if not client:
            return None
        account = client.futures_account()
        balance = float(account.get('totalWalletBalance', 0))
        
        # Update cache
        account_balance_cache['balance'] = balance
        account_balance_cache['timestamp'] = current_time
        
        return balance
    except Exception as e:
        logger.warning(f"Failed to get account balance: {e}")
        # Return cached value if available, even if stale
        if account_balance_cache['balance'] is not None:
            logger.info(f"Using cached account balance: ${account_balance_cache['balance']:,.2f}")
            return account_balance_cache['balance']
        return None


def calculate_trade_risk(entry_price, stop_loss, quantity, signal_side):
    """Calculate risk for a single trade
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        quantity: Position quantity
        signal_side: 'LONG' or 'SHORT'
    
    Returns:
        float: Risk in USD
    """
    if not entry_price or not stop_loss or not quantity:
        return 0.0
    
    if signal_side == 'LONG':
        risk_per_unit = abs(entry_price - stop_loss)
    else:  # SHORT
        risk_per_unit = abs(stop_loss - entry_price)
    
    risk_usd = risk_per_unit * quantity
    return risk_usd


def calculate_total_risk_including_pending(symbol, entry_price, stop_loss, quantity, signal_side):
    """Calculate total risk including current trade and all pending orders
    
    Args:
        symbol: Trading symbol (for current trade)
        entry_price: Entry price for current trade
        stop_loss: Stop loss for current trade
        quantity: Quantity for current trade
        signal_side: 'LONG' or 'SHORT' for current trade
    
    Returns:
        dict: {
            'current_trade_risk': float,
            'pending_orders_risk': float,
            'total_risk': float,
            'account_balance': float,
            'risk_percent': float,
            'pending_orders': list
        }
    """
    try:
        # Get account balance (cached)
        account_balance = get_account_balance()
        if account_balance is None or account_balance <= 0:
            logger.warning("Cannot calculate risk: Account balance not available")
            return None
        
        # Calculate current trade risk
        current_trade_risk = calculate_trade_risk(entry_price, stop_loss, quantity, signal_side)
        
        # Get all open orders to calculate pending risk
        pending_orders_risk = 0.0
        pending_orders = []
        
        try:
            if not client:
                return None
            all_orders = client.futures_get_open_orders()
            
            for order in all_orders:
                order_symbol = order.get('symbol', '')
                order_side = order.get('side', '')
                order_type = order.get('type', '')
                order_price = safe_float(order.get('price'), default=0)
                order_qty = safe_float(order.get('origQty'), default=0)
                
                # Only count LIMIT orders (pending entries)
                if order_type == 'LIMIT' and order_price > 0 and order_qty > 0:
                    # Get stop loss for this order from active_trades
                    order_risk = 0.0
                    if order_symbol in active_trades:
                        trade_info = active_trades[order_symbol]
                        sl_price = safe_float(trade_info.get('stop_loss'), default=0)
                        if sl_price > 0:
                            # Determine side from order
                            order_signal_side = 'LONG' if order_side == 'BUY' else 'SHORT'
                            order_risk = calculate_trade_risk(order_price, sl_price, order_qty, order_signal_side)
                    
                    if order_risk > 0:
                        pending_orders_risk += order_risk
                        pending_orders.append({
                            'symbol': order_symbol,
                            'side': order_side,
                            'price': order_price,
                            'quantity': order_qty,
                            'risk': order_risk
                        })
        except Exception as e:
            logger.warning(f"Error calculating pending orders risk: {e}")
        
        # Calculate total risk
        total_risk = current_trade_risk + pending_orders_risk
        risk_percent = (total_risk / account_balance) * 100 if account_balance > 0 else 0
        
        return {
            'current_trade_risk': current_trade_risk,
            'pending_orders_risk': pending_orders_risk,
            'total_risk': total_risk,
            'account_balance': account_balance,
            'risk_percent': risk_percent,
            'pending_orders': pending_orders
        }
    except Exception as e:
        logger.error(f"Error calculating total risk: {e}", exc_info=True)
        return None


def validate_risk_per_trade(symbol, entry_price, stop_loss, quantity, signal_side):
    """Validate if trade risk is within acceptable limits
    
    Args:
        symbol: Trading symbol
        entry_price: Entry price
        stop_loss: Stop loss price
        quantity: Position quantity
        signal_side: 'LONG' or 'SHORT'
    
    Returns:
        tuple: (is_valid: bool, risk_info: dict or None, error_message: str or None)
    """
    risk_info = calculate_total_risk_including_pending(symbol, entry_price, stop_loss, quantity, signal_side)
    
    if risk_info is None:
        # If we can't calculate risk, allow trade (fail-open)
        logger.warning(f"Could not calculate risk for {symbol}, allowing trade (fail-open)")
        return True, None, None
    
    risk_percent = risk_info['risk_percent']
    
    if risk_percent > MAX_RISK_PERCENT:
        error_msg = (f"Trade risk {risk_percent:.2f}% exceeds maximum {MAX_RISK_PERCENT}% "
                    f"(Current trade: ${risk_info['current_trade_risk']:.2f}, "
                    f"Pending orders: ${risk_info['pending_orders_risk']:.2f}, "
                    f"Total: ${risk_info['total_risk']:.2f} of ${risk_info['account_balance']:.2f} account)")
        logger.warning(f"🚫 Risk validation REJECTED for {symbol}: {error_msg}")
        return False, risk_info, error_msg
    
    logger.info(f"✅ Risk validation PASSED for {symbol}: {risk_percent:.2f}% risk "
               f"(Current: ${risk_info['current_trade_risk']:.2f}, "
               f"Pending: ${risk_info['pending_orders_risk']:.2f}, "
               f"Total: ${risk_info['total_risk']:.2f} of ${risk_info['account_balance']:.2f} account)")
    return True, risk_info, None


def check_recent_price_volatility(symbol, days=7):
    """Check if price has moved significantly in recent days
    
    Args:
        symbol: Trading symbol
        days: Number of days to check (default 7)
    
    Returns:
        tuple: (has_high_volatility: bool, price_change_pct: float)
    """
    try:
        if not client:
            return False, 0.0
        
        # Get daily candles for the last N days
        interval = '1d'
        limit = days + 1  # +1 to get enough candles
        
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        if len(klines) < 2:
            return False, 0.0
        
        # Get oldest and newest close prices
        oldest_close = float(klines[0][4])  # Close price of oldest candle
        newest_close = float(klines[-1][4])  # Close price of newest candle
        
        if oldest_close <= 0:
            return False, 0.0
        
        # Calculate percentage change
        price_change_pct = abs((newest_close - oldest_close) / oldest_close) * 100
        
        # Consider high volatility if price moved >10% in recent days
        has_high_volatility = price_change_pct > 10.0
        
        logger.info(f"📊 Recent price volatility for {symbol}: {price_change_pct:.2f}% over {days} days (High volatility: {has_high_volatility})")
        return has_high_volatility, price_change_pct
    except Exception as e:
        logger.warning(f"Error checking recent price volatility for {symbol}: {e}")
        return False, 0.0

