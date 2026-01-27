"""
Utility functions for Binance Webhook Service
"""
import re
import math
from typing import Optional, Tuple, Dict
try:
    # Try relative import first (when imported as package)
    from ..config import WEBHOOK_TOKEN
    from ..core import client, logger
except ImportError:
    # Fall back to absolute import (when src/ is in Python path)
    from config import WEBHOOK_TOKEN
    from core import client, logger


def verify_webhook_token(payload_token: str) -> bool:
    """Verify webhook token matches configured token"""
    return payload_token == WEBHOOK_TOKEN


def safe_float(value, default=None):
    """Safely convert value to float, handling None, 'null' string, and invalid values"""
    if value is None:
        return default
    if isinstance(value, str):
        # Handle Pine Script's "null" string
        if value.lower() == 'null' or value.strip() == '':
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    try:
        result = float(value)
        # Check for invalid values (NaN, infinity, negative/zero for prices)
        if result != result or result == float('inf') or result == float('-inf'):
            return default
        return result
    except (ValueError, TypeError):
        return default


def get_order_side(signal_side: str) -> str:
    """Convert signal side to Binance order side"""
    signal_side = signal_side.upper()
    if signal_side == 'LONG':
        return 'BUY'
    elif signal_side == 'SHORT':
        return 'SELL'
    else:
        raise ValueError(f"Invalid signal_side: {signal_side}")


def get_position_side(signal_side: str) -> str:
    """Get position side for Binance Futures"""
    signal_side = signal_side.upper()
    if signal_side == 'LONG':
        return 'LONG'
    elif signal_side == 'SHORT':
        return 'SHORT'
    else:
        return 'BOTH'  # Default


def get_position_mode(symbol: str) -> bool:
    """Detect if account is in One-Way or Hedge mode for a symbol"""
    try:
        if not client:
            return False
        # Try to get position mode from Binance
        position_mode = client.futures_get_position_mode()
        # If dualSidePosition is True, it's Hedge mode, else One-Way
        return position_mode.get('dualSidePosition', False)
    except Exception as e:
        logger.warning(f"Could not detect position mode: {e}, defaulting to One-Way mode")
        return False  # Default to One-Way mode


def format_symbol(trading_symbol: str) -> str:
    """Format symbol for Binance (e.g., ETHUSDT.P -> ETHUSDT)"""
    # Remove .P or any exchange suffix
    symbol = trading_symbol.replace('.P', '').replace('.PERP', '').upper()
    # Ensure it ends with USDT
    if not symbol.endswith('USDT'):
        # Try to extract base currency and add USDT
        if 'USD' in symbol:
            symbol = symbol.replace('USD', 'USDT')
        else:
            symbol = symbol + 'USDT'
    return symbol


def check_existing_position(symbol: str, signal_side: str) -> Tuple[bool, Optional[Dict]]:
    """Check if there's an existing open position for the symbol"""
    try:
        if not client:
            return False, None
        # Get ALL positions first, then filter by symbol (more reliable than filtering in API call)
        all_positions = client.futures_position_information()
        for position in all_positions:
            position_symbol = position.get('symbol', '')
            position_amt = float(position.get('positionAmt', 0))
            
            # Match symbol (case-insensitive) and check if position exists
            if position_symbol.upper() == symbol.upper() and abs(position_amt) > 0:
                position_side = position.get('positionSide', 'BOTH')
                # Check if position side matches (or if it's BOTH mode)
                if position_side == 'BOTH' or position_side == signal_side.upper() or signal_side.upper() == 'BOTH':
                    logger.info(f"Existing position found for {symbol}: {position_amt} @ {position.get('entryPrice')}")
                    return True, position
        return False, None
    except Exception as e:
        logger.error(f"Error checking existing position: {e}")
        return False, None


def check_existing_orders(symbol: str, log_result: bool = False) -> Tuple[bool, list]:
    """Check for existing open orders (limit orders) for the symbol"""
    try:
        if not client:
            return False, []
        open_orders = client.futures_get_open_orders(symbol=symbol)
        if open_orders:
            if log_result:
                logger.info(f"Found {len(open_orders)} open orders for {symbol}")
            return True, open_orders
        return False, []
    except Exception as e:
        logger.error(f"Error checking existing orders: {e}")
        return False, []


def cancel_order(symbol: str, order_id: int) -> bool:
    """Cancel a specific order by ID"""
    try:
        if not client:
            return False
        result = client.futures_cancel_order(symbol=symbol, orderId=order_id)
        logger.info(f"Canceled order {order_id} for {symbol}: {result}")
        return True
    except Exception as e:
        logger.error(f"Failed to cancel order {order_id} for {symbol}: {e}")
        return False


def cancel_all_limit_orders(symbol: str, side: Optional[str] = None) -> int:
    """Cancel all limit orders for a symbol, optionally filtered by side"""
    try:
        if not client:
            return 0
        open_orders = client.futures_get_open_orders(symbol=symbol)
        canceled_count = 0
        for order in open_orders:
            # Only cancel LIMIT orders (entry orders)
            if order.get('type') == 'LIMIT':
                if side is None or order.get('side') == side:
                    try:
                        client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
                        logger.info(f"Canceled limit order {order['orderId']} for {symbol}")
                        canceled_count += 1
                    except Exception as e:
                        logger.error(f"Failed to cancel order {order['orderId']}: {e}")
        return canceled_count
    except Exception as e:
        logger.error(f"Error canceling orders for {symbol}: {e}")
        return 0


def format_quantity_precision(quantity: float, step_size: float) -> float:
    """Format quantity to match Binance step size precision, removing floating point errors
    
    Handles all step_size formats:
    - Large: 1.0, 10.0, 100.0 (0 decimal places)
    - Medium: 0.1, 0.5 (1 decimal place)
    - Small: 0.01, 0.001 (2-3 decimal places)
    - Very small: 0.0001, 0.00001 (4-5 decimal places)
    - Extremely small: 0.000001 (6+ decimal places)
    """
    if step_size <= 0:
        return quantity
    
    # Calculate decimal places from step_size
    # Convert step_size to string to count decimal places accurately
    step_size_str = f"{step_size:.10f}".rstrip('0').rstrip('.')
    
    if '.' in step_size_str:
        # Count decimal places after the decimal point
        decimal_places = len(step_size_str.split('.')[1])
    else:
        # Step size is a whole number (1.0, 10.0, etc.)
        decimal_places = 0
    
    # Round down to nearest step_size: divide by step_size, floor, multiply back
    # This ensures quantity is a multiple of step_size
    quantity = math.floor(quantity / step_size) * step_size
    
    # Format to correct decimal places to eliminate floating point errors
    # This converts values like 15.280000000000001 to 15.28
    quantity = round(quantity, decimal_places)
    
    # Final safety check: convert to string and back to float to remove any remaining precision errors
    # Format with the exact number of decimal places needed
    if decimal_places > 0:
        quantity_str = f"{quantity:.{decimal_places}f}"
        quantity = float(quantity_str)
    else:
        # For whole numbers, ensure it's an integer
        quantity = float(int(quantity))
    
    return quantity


def format_price_precision(price: float, tick_size: float) -> float:
    """Format price to match Binance tick size precision, removing floating point errors
    
    Handles all tick_size formats:
    - Large: 1.0, 10.0, 100.0 (0 decimal places)
    - Medium: 0.1, 0.5 (1 decimal place)
    - Small: 0.01, 0.001 (2-3 decimal places)
    - Very small: 0.0001, 0.00001 (4-5 decimal places)
    - Extremely small: 0.000001 (6+ decimal places)
    """
    if tick_size <= 0:
        return price
    
    # Calculate decimal places from tick_size
    # Convert tick_size to string to count decimal places accurately
    tick_size_str = f"{tick_size:.10f}".rstrip('0').rstrip('.')
    
    if '.' in tick_size_str:
        # Count decimal places after the decimal point
        decimal_places = len(tick_size_str.split('.')[1])
    else:
        # Tick size is a whole number (1.0, 10.0, etc.)
        decimal_places = 0
    
    # Round to tick size: divide by tick_size, round to nearest integer, multiply back
    # This ensures price is a multiple of tick_size
    price = round(price / tick_size) * tick_size
    
    # Format to correct decimal places to eliminate floating point errors
    # This converts values like 0.7111000000000001 to 0.7111
    price = round(price, decimal_places)
    
    # Final safety check: convert to string and back to float to remove any remaining precision errors
    # Format with the exact number of decimal places needed
    if decimal_places > 0:
        price_str = f"{price:.{decimal_places}f}"
        price = float(price_str)
    else:
        # For whole numbers, ensure it's an integer
        price = float(int(price))
    
    return price


def cleanup_closed_positions():
    """Periodically clean up active_trades for symbols with no open positions"""
    try:
        try:
            from ..models import active_trades
        except ImportError:
            from models import active_trades
        from . import check_existing_position, check_existing_orders
        
        symbols_to_remove = []
        for symbol in list(active_trades.keys()):
            # Check if position exists (check both LONG and SHORT)
            has_long, _ = check_existing_position(symbol, 'LONG')
            has_short, _ = check_existing_position(symbol, 'SHORT')
            has_position = has_long or has_short
            
            if not has_position:
                # No position exists, check if orders exist
                has_orders, _ = check_existing_orders(symbol)
                if not has_orders:
                    # No position and no orders, can clean up
                    symbols_to_remove.append(symbol)
        
        for symbol in symbols_to_remove:
            logger.info(f"Cleaning up closed position tracking for {symbol}")
            del active_trades[symbol]
    except Exception as e:
        logger.error(f"Error cleaning up closed positions: {e}")

