"""
Order management service for Binance Webhook Service
Handles order creation, TP orders, position management
"""
import time
import math
import threading
from typing import Dict, Any, Optional

# Import dependencies
# Try absolute import first (when src/ is in Python path - gunicorn case)
try:
    from core import client, logger
    from config import (
        ENTRY_SIZE_USD, LEVERAGE, TP1_PERCENT, TP1_SPLIT, TP2_SPLIT,
        TP_HIGH_CONFIDENCE_THRESHOLD, ENABLE_TRAILING_STOP_LOSS,
        TRAILING_SL_BREAKEVEN_PERCENT, ENABLE_RISK_VALIDATION,
        AI_VALIDATION_MIN_CONFIDENCE, ENABLE_AI_PRICE_SUGGESTIONS
    )
    from models.state import active_trades, recent_orders, recent_exits, ORDER_COOLDOWN, EXIT_COOLDOWN
    from utils.helpers import (
        format_symbol, safe_float, get_order_side, get_position_side,
        get_position_mode, check_existing_position, check_existing_orders,
        cancel_order, cancel_all_limit_orders, format_quantity_precision,
        format_price_precision, verify_webhook_token
    )
    from services.risk.risk_manager import validate_risk_per_trade, check_recent_price_volatility
    from services.ai_validation.validator import (
        validate_signal_with_ai, validate_entry2_standalone_with_ai,
        parse_entry_analysis_from_reasoning, analyze_symbol_for_opportunities
    )
    from notifications.slack import send_slack_alert, send_signal_notification, send_exit_notification, send_signal_rejection_notification
except ImportError:
    # Fall back to relative import (when imported as package)
    from ...core import client, logger
    from ...config import (
        ENTRY_SIZE_USD, LEVERAGE, TP1_PERCENT, TP1_SPLIT, TP2_SPLIT,
        TP_HIGH_CONFIDENCE_THRESHOLD, ENABLE_TRAILING_STOP_LOSS,
        TRAILING_SL_BREAKEVEN_PERCENT, ENABLE_RISK_VALIDATION,
        AI_VALIDATION_MIN_CONFIDENCE, ENABLE_AI_PRICE_SUGGESTIONS
    )
    from ...models.state import active_trades, recent_orders, recent_exits, ORDER_COOLDOWN, EXIT_COOLDOWN
    from ...utils.helpers import (
        format_symbol, safe_float, get_order_side, get_position_side,
        get_position_mode, check_existing_position, check_existing_orders,
        cancel_order, cancel_all_limit_orders, format_quantity_precision,
        format_price_precision, verify_webhook_token
    )
    from ...services.risk.risk_manager import validate_risk_per_trade, check_recent_price_volatility
    from ...services.ai_validation.validator import (
        validate_signal_with_ai, validate_entry2_standalone_with_ai,
        parse_entry_analysis_from_reasoning, analyze_symbol_for_opportunities
    )
    from ...notifications.slack import send_slack_alert, send_signal_notification, send_exit_notification, send_signal_rejection_notification
from binance.exceptions import BinanceAPIException, BinanceOrderException

def calculate_quantity(entry_price, symbol_info, entry_size_usd=None):
    """Calculate quantity based on entry size and leverage
    
    Args:
        entry_price: Entry price for the order
        symbol_info: Symbol information from Binance
        entry_size_usd: Custom entry size in USD (defaults to ENTRY_SIZE_USD if not provided)
    """
    # Use custom entry size if provided, otherwise use default
    size_usd = entry_size_usd if entry_size_usd is not None else ENTRY_SIZE_USD
    # Position value = Entry size * Leverage (e.g., $10 * 20X = $200)
    position_value = size_usd * LEVERAGE
    
    # Quantity = Position value / Entry price
    quantity = position_value / entry_price
    
    # Get step size and min quantity from symbol info
    lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
    step_size = float(lot_size_filter['stepSize']) if lot_size_filter else 0.001
    min_qty = float(lot_size_filter['minQty']) if lot_size_filter else 0.001
    
    # Format quantity with proper precision
    quantity = format_quantity_precision(quantity, step_size)
    
    # Ensure minimum quantity
    if quantity < min_qty:
        quantity = min_qty
        # Re-format after setting to min_qty to ensure precision
        quantity = format_quantity_precision(quantity, step_size)
    
    # Binance requires minimum notional of $100 (quantity * price >= 100)
    min_notional = 100.0
    actual_notional = quantity * entry_price
    if actual_notional < min_notional:
        # Adjust quantity to meet minimum notional - round UP to ensure we exceed minimum
        required_quantity = min_notional / entry_price
        # Round UP to next step_size increment (always round up, never down)
        quantity = math.ceil(required_quantity / step_size) * step_size
        # Ensure still meets min_qty
        if quantity < min_qty:
            quantity = min_qty
        # Format to ensure proper precision (but don't round down)
        # Use ceil again after formatting to ensure we don't lose precision
        formatted_qty = format_quantity_precision(quantity, step_size)
        formatted_notional = formatted_qty * entry_price
        if formatted_notional < min_notional:
            # If formatted quantity still doesn't meet minimum, add one more step
            quantity = formatted_qty + step_size
            quantity = format_quantity_precision(quantity, step_size)
        else:
            quantity = formatted_qty
        final_notional = quantity * entry_price
        logger.warning(f"Adjusted quantity to meet minimum notional: ${actual_notional:.2f} -> ${final_notional:.2f} (quantity: {quantity})")
    
    logger.info(f"Calculated quantity: {quantity} (Position value: ${position_value} @ ${entry_price}, Notional: ${quantity * entry_price:.2f}, step_size: {step_size})")
    return quantity


# Validation cache to avoid duplicate API calls
def delayed_tp_creation(symbol, delay_seconds=5):
    """Helper function to create TP orders after a delay (allows Binance to update position)"""
    def _create():
        time.sleep(delay_seconds)
        if symbol in active_trades:
            logger.info(f"🔄 Delayed TP check for {symbol} (after {delay_seconds}s delay)")
            create_tp1_tp2_if_needed(symbol, active_trades[symbol])
    
    thread = threading.Thread(target=_create, daemon=True)
    thread.start()

def create_tp_if_needed(symbol, trade_info):
    """Create TP order if position exists and TP is stored but not created yet
    NOTE: This function does NOT call AI validation - it only creates TP orders for existing positions"""
    if 'tp_price' not in trade_info:
        return False  # No TP to create
    
    # If TP order ID exists, check if it's still valid
    if 'tp_order_id' in trade_info:
        try:
            # Verify TP order still exists
            open_orders = client.futures_get_open_orders(symbol=symbol)
            existing_tp = [o for o in open_orders if o.get('orderId') == trade_info['tp_order_id']]
            if existing_tp:
                return True  # TP order exists and is valid
            else:
                # TP order was filled or canceled, remove the ID to allow recreation
                logger.info(f"TP order {trade_info['tp_order_id']} no longer exists for {symbol}, will recreate if needed")
                del trade_info['tp_order_id']
        except Exception as e:
            logger.warning(f"Error checking TP order status for {symbol}: {e}")
    
    try:
        # Check if position exists
        positions = client.futures_position_information(symbol=symbol)
        position_to_use = None
        
        for position in positions:
            position_amt = float(position.get('positionAmt', 0))
            if abs(position_amt) > 0:
                position_to_use = position
                break
        
        if not position_to_use:
            return False  # No position exists yet
        
        # Position exists, check if TP already exists
        has_orders, open_orders = check_existing_orders(symbol)
        existing_tp = [o for o in open_orders if o.get('type') == 'TAKE_PROFIT_MARKET']
        
        if existing_tp:
            # TP already exists, store the order ID and clean up stored price
            trade_info['tp_order_id'] = existing_tp[0].get('orderId')
            logger.info(f"✅ TP order already exists for {symbol}: {trade_info['tp_order_id']}")
            if 'tp_price' in trade_info:
                del trade_info['tp_price']
            return True
        
        # Create TP order
        tp_price = trade_info['tp_price']
        tp_side = trade_info.get('tp_side', 'SELL')
        position_amt = float(position_to_use.get('positionAmt', 0))
        
        # Get symbol info
        exchange_info = client.futures_exchange_info()
        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
        if not symbol_info:
            logger.error(f"Symbol info not found for {symbol}")
            send_slack_alert(
                error_type="Symbol Info Not Found",
                message=f"Symbol {symbol} not found in Binance exchange info",
                symbol=symbol,
                severity='ERROR'
            )
            return False
            
        price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
        tick_size = float(price_filter['tickSize']) if price_filter else 0.01
        tp_price = format_price_precision(tp_price, tick_size)
        
        # Detect position mode
        try:
            is_hedge_mode = get_position_mode(symbol)
        except:
            is_hedge_mode = False
        
        position_side = position_to_use.get('positionSide', 'BOTH')
        
        # Use stored TP quantity (total of primary + DCA) or position amount as fallback
        tp_quantity = trade_info.get('tp_quantity', abs(position_amt))
        
        # Format quantity precision
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            step_size = float(lot_size_filter['stepSize'])
            tp_quantity = format_quantity_precision(tp_quantity, step_size)
        
        working_type = trade_info.get('tp_working_type', 'MARK_PRICE')
        
        logger.info(f"🔄 Creating TP order for {symbol}: price={tp_price}, qty={tp_quantity}, side={tp_side}, workingType={working_type}, positionSide={position_side}, hedgeMode={is_hedge_mode}")
        
        # Try using closePosition first (recommended by Binance for conditional orders)
        # If that fails, fall back to using quantity
        tp_params_close = {
            'symbol': symbol,
            'side': tp_side,
            'type': 'TAKE_PROFIT_MARKET',
            'timeInForce': 'GTC',
            'closePosition': True,  # Close entire position (recommended for conditional orders)
            'stopPrice': tp_price,
            'workingType': working_type,  # Use mark price for trigger (like Binance UI)
        }
        
        tp_params_quantity = {
            'symbol': symbol,
            'side': tp_side,
            'type': 'TAKE_PROFIT_MARKET',
            'timeInForce': 'GTC',
            'quantity': tp_quantity,
            'stopPrice': tp_price,
            'workingType': working_type,  # Use mark price for trigger (like Binance UI)
        }
        
        # Add positionSide only if in hedge mode
        if is_hedge_mode and position_side != 'BOTH':
            tp_params_close['positionSide'] = position_side
            tp_params_quantity['positionSide'] = position_side
        
        # Strategy: Try multiple approaches in order of preference
        # 1. closePosition=True (recommended by Binance for conditional orders)
        # 2. quantity with reduceOnly=True
        # 3. quantity without reduceOnly
        
        # Try 1: closePosition=True (no reduceOnly needed when using closePosition)
        try:
            tp_order = client.futures_create_order(**tp_params_close)
            trade_info['tp_order_id'] = tp_order.get('orderId')
            logger.info(f"✅ TP order created successfully (using closePosition): Order ID {tp_order.get('orderId')} @ {tp_price} for {symbol}")
            
            # Remove stored TP details since it's now created (but keep tp_order_id)
            if 'tp_price' in trade_info:
                del trade_info['tp_price']
            if 'tp_quantity' in trade_info:
                del trade_info['tp_quantity']
            if 'tp_working_type' in trade_info:
                del trade_info['tp_working_type']
            return True
        except BinanceAPIException as e:
            # If closePosition fails, try with quantity
            if e.code == -4120 or 'order type not supported' in str(e).lower() or 'closePosition' in str(e).lower():
                logger.info(f"closePosition approach failed for {symbol}, trying with quantity: {e.message}")
                # Try 2: quantity with reduceOnly=True
                try:
                    tp_params_qty_reduce = tp_params_quantity.copy()
                    tp_params_qty_reduce['reduceOnly'] = True
                    tp_order = client.futures_create_order(**tp_params_qty_reduce)
                    trade_info['tp_order_id'] = tp_order.get('orderId')
                    logger.info(f"✅ TP order created successfully (using quantity with reduceOnly): Order ID {tp_order.get('orderId')} @ {tp_price} for {symbol} (qty: {tp_quantity})")
                    
                    # Remove stored TP details since it's now created (but keep tp_order_id)
                    if 'tp_price' in trade_info:
                        del trade_info['tp_price']
                    if 'tp_quantity' in trade_info:
                        del trade_info['tp_quantity']
                    if 'tp_working_type' in trade_info:
                        del trade_info['tp_working_type']
                    return True
                except BinanceAPIException as e2:
                    # Try 3: quantity without reduceOnly
                    if e2.code == -1106 or 'reduceonly' in str(e2).lower():
                        logger.info(f"reduceOnly not accepted, trying without it: {e2.message}")
                        try:
                            tp_order = client.futures_create_order(**tp_params_quantity)
                            trade_info['tp_order_id'] = tp_order.get('orderId')
                            logger.info(f"✅ TP order created successfully (using quantity without reduceOnly): Order ID {tp_order.get('orderId')} @ {tp_price} for {symbol} (qty: {tp_quantity})")
                            
                            # Remove stored TP details since it's now created (but keep tp_order_id)
                            if 'tp_price' in trade_info:
                                del trade_info['tp_price']
                            if 'tp_quantity' in trade_info:
                                del trade_info['tp_quantity']
                            if 'tp_working_type' in trade_info:
                                del trade_info['tp_working_type']
                            return True
                        except BinanceAPIException as e3:
                            # All approaches failed
                            if e3.code == -4120 or 'order type not supported' in str(e3).lower():
                                logger.warning(f"⚠️ TAKE_PROFIT_MARKET orders not supported for {symbol} (Code: {e3.code}). All approaches failed. Cleaning up stored TP details.")
                                send_slack_alert(
                                    error_type="Take Profit Order Type Not Supported",
                                    message=f"TAKE_PROFIT_MARKET orders are not supported for {symbol} after trying all approaches. You may need to set TP manually in Binance UI.",
                                    details={'Error_Code': e3.code, 'TP_Price': tp_price, 'TP_Quantity': tp_quantity, 'Symbol': symbol, 'Attempts': 'closePosition, quantity+reduceOnly, quantity'},
                                    symbol=symbol,
                                    severity='WARNING'
                                )
                                # Clean up stored TP details to prevent repeated retries
                                if 'tp_price' in trade_info:
                                    del trade_info['tp_price']
                                if 'tp_quantity' in trade_info:
                                    del trade_info['tp_quantity']
                                if 'tp_working_type' in trade_info:
                                    del trade_info['tp_working_type']
                                return False
                            else:
                                raise e3
                    elif e2.code == -4120 or 'order type not supported' in str(e2).lower():
                        # Try without reduceOnly
                        try:
                            tp_order = client.futures_create_order(**tp_params_quantity)
                            trade_info['tp_order_id'] = tp_order.get('orderId')
                            logger.info(f"✅ TP order created successfully (using quantity without reduceOnly): Order ID {tp_order.get('orderId')} @ {tp_price} for {symbol} (qty: {tp_quantity})")
                            
                            # Remove stored TP details since it's now created (but keep tp_order_id)
                            if 'tp_price' in trade_info:
                                del trade_info['tp_price']
                            if 'tp_quantity' in trade_info:
                                del trade_info['tp_quantity']
                            if 'tp_working_type' in trade_info:
                                del trade_info['tp_working_type']
                            return True
                        except BinanceAPIException as e3:
                            if e3.code == -4120 or 'order type not supported' in str(e3).lower():
                                logger.warning(f"⚠️ TAKE_PROFIT_MARKET orders not supported for {symbol} (Code: {e3.code}). All approaches failed. Cleaning up stored TP details.")
                                send_slack_alert(
                                    error_type="Take Profit Order Type Not Supported",
                                    message=f"TAKE_PROFIT_MARKET orders are not supported for {symbol} after trying all approaches. You may need to set TP manually in Binance UI.",
                                    details={'Error_Code': e3.code, 'TP_Price': tp_price, 'TP_Quantity': tp_quantity, 'Symbol': symbol, 'Attempts': 'closePosition, quantity+reduceOnly, quantity'},
                                    symbol=symbol,
                                    severity='WARNING'
                                )
                                # Clean up stored TP details to prevent repeated retries
                                if 'tp_price' in trade_info:
                                    del trade_info['tp_price']
                                if 'tp_quantity' in trade_info:
                                    del trade_info['tp_quantity']
                                if 'tp_working_type' in trade_info:
                                    del trade_info['tp_working_type']
                                return False
                            else:
                                raise e3
                    else:
                        raise e2
            else:
                # Other error from closePosition - raise to be caught by outer handler
                raise
        
    except BinanceAPIException as e:
        # Check if order type is not supported (error -4120)
        if e.code == -4120 or 'order type not supported' in str(e).lower():
            logger.warning(f"⚠️ TAKE_PROFIT_MARKET orders not supported for {symbol} (Code: {e.code}). This symbol may not support conditional orders. Cleaning up stored TP details.")
            send_slack_alert(
                error_type="Take Profit Order Type Not Supported",
                message=f"TAKE_PROFIT_MARKET orders are not supported for {symbol}. You may need to set TP manually in Binance UI.",
                details={'Error_Code': e.code, 'TP_Price': tp_price, 'TP_Quantity': tp_quantity, 'Symbol': symbol},
                symbol=symbol,
                severity='WARNING'
            )
            # Clean up stored TP details to prevent repeated retries
            if 'tp_price' in trade_info:
                del trade_info['tp_price']
            if 'tp_quantity' in trade_info:
                del trade_info['tp_quantity']
            if 'tp_working_type' in trade_info:
                del trade_info['tp_working_type']
            return False
        else:
            logger.error(f"❌ Binance API error creating TP for {symbol}: {e.message} (Code: {e.code})")
            send_slack_alert(
                error_type="Take Profit Order Creation Failed",
                message=f"{e.message} (Code: {e.code})",
                details={'Error_Code': e.code, 'TP_Price': tp_price, 'TP_Quantity': tp_quantity},
                symbol=symbol,
                severity='ERROR'
            )
            return False
    except Exception as e:
        logger.error(f"❌ Error creating TP for {symbol}: {e}", exc_info=True)
        send_slack_alert(
            error_type="Take Profit Order Creation Error",
            message=str(e),
            details={'TP_Price': tp_price, 'TP_Quantity': tp_quantity},
            symbol=symbol,
            severity='ERROR'
        )
        return False


def create_single_tp_order(symbol, tp_price, tp_quantity, tp_side, trade_info, tp_number=1):
    """Helper function to create a single TP order (used for both TP1 and TP2)
    
    Args:
        symbol: Trading symbol
        tp_price: Take profit price
        tp_quantity: Quantity to close at this TP
        tp_side: 'SELL' for LONG, 'BUY' for SHORT
        trade_info: Trade info dict
        tp_number: 1 for TP1, 2 for TP2 (for logging)
    
    Returns:
        order_id if successful, None if failed
    """
    try:
        # Get symbol info
        exchange_info = client.futures_exchange_info()
        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
        if not symbol_info:
            logger.error(f"Symbol info not found for {symbol}")
            return None
            
        price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
        tick_size = float(price_filter['tickSize']) if price_filter else 0.01
        tp_price = format_price_precision(tp_price, tick_size)
        
        # Format quantity precision
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            step_size = float(lot_size_filter['stepSize'])
            tp_quantity = format_quantity_precision(tp_quantity, step_size)
        
        # Get position info
        positions = client.futures_position_information(symbol=symbol)
        position_to_use = None
        for position in positions:
            position_amt = float(position.get('positionAmt', 0))
            if abs(position_amt) > 0:
                position_to_use = position
                break
        
        if not position_to_use:
            return None  # No position exists
        
        # Detect position mode
        try:
            is_hedge_mode = get_position_mode(symbol)
        except:
            is_hedge_mode = False
        
        position_side = position_to_use.get('positionSide', 'BOTH')
        working_type = trade_info.get('tp_working_type', 'MARK_PRICE')
        
        logger.info(f"🔄 Creating TP{tp_number} order for {symbol}: price={tp_price}, qty={tp_quantity}, side={tp_side}, workingType={working_type}")
        
        # Try using quantity with reduceOnly=True (required for partial closes)
        tp_params = {
            'symbol': symbol,
            'side': tp_side,
            'type': 'TAKE_PROFIT_MARKET',
            'timeInForce': 'GTC',
            'quantity': tp_quantity,
            'stopPrice': tp_price,
            'workingType': working_type,
            'reduceOnly': True,  # Required for partial position closes
        }
        
        # Add positionSide only if in hedge mode
        if is_hedge_mode and position_side != 'BOTH':
            tp_params['positionSide'] = position_side
        
        # Try to create the order
        try:
            tp_order = client.futures_create_order(**tp_params)
            order_id = tp_order.get('orderId')
            logger.info(f"✅ TP{tp_number} order created successfully: Order ID {order_id} @ {tp_price} for {symbol} (qty: {tp_quantity})")
            return order_id
        except BinanceAPIException as e:
            if e.code == -1106 or 'reduceonly' in str(e).lower():
                # Try without reduceOnly
                try:
                    tp_params_no_reduce = tp_params.copy()
                    del tp_params_no_reduce['reduceOnly']
                    tp_order = client.futures_create_order(**tp_params_no_reduce)
                    order_id = tp_order.get('orderId')
                    logger.info(f"✅ TP{tp_number} order created successfully (without reduceOnly): Order ID {order_id} @ {tp_price} for {symbol} (qty: {tp_quantity})")
                    return order_id
                except Exception as e2:
                    logger.error(f"❌ Failed to create TP{tp_number} order for {symbol}: {e2}")
                    return None
            else:
                logger.error(f"❌ Failed to create TP{tp_number} order for {symbol}: {e.message} (Code: {e.code})")
                return None
    except Exception as e:
        logger.error(f"❌ Error creating TP{tp_number} order for {symbol}: {e}", exc_info=True)
        return None


def create_tp1_tp2_if_needed(symbol, trade_info):
    """Create TP orders if position exists and TPs are stored but not created yet
    High Confidence (>=90%): Single TP (100% of position at main TP)
    Lower Confidence (<90%): TP1 + TP2 (70% at TP1, 30% at TP2)
    """
    # Check if using single TP mode
    use_single_tp = trade_info.get('use_single_tp', False)
    
    if use_single_tp:
        # Single TP mode: Only TP2 exists (main TP)
        if 'tp2_price' not in trade_info or not trade_info.get('tp2_price'):
            return False  # TP not configured
    else:
        # TP1 + TP2 mode: Both TPs exist
        if 'tp1_price' not in trade_info or 'tp2_price' not in trade_info:
            return False  # TPs not configured
    
    # Check existing TP orders
    tp1_exists = False
    tp2_exists = False
    
    if not use_single_tp:
        # TP1 + TP2 mode: Check both
        if 'tp1_order_id' in trade_info:
            try:
                open_orders = client.futures_get_open_orders(symbol=symbol)
                existing_tp1 = [o for o in open_orders if o.get('orderId') == trade_info['tp1_order_id']]
                if existing_tp1:
                    tp1_exists = True
                else:
                    logger.info(f"TP1 order {trade_info['tp1_order_id']} no longer exists for {symbol}")
                    del trade_info['tp1_order_id']
            except Exception as e:
                logger.warning(f"Error checking TP1 order status for {symbol}: {e}")
    
    # Check TP2 (always exists, either as main TP or as TP2)
    if 'tp2_order_id' in trade_info:
        try:
            open_orders = client.futures_get_open_orders(symbol=symbol)
            existing_tp2 = [o for o in open_orders if o.get('orderId') == trade_info['tp2_order_id']]
            if existing_tp2:
                tp2_exists = True
            else:
                logger.info(f"TP2 order {trade_info['tp2_order_id']} no longer exists for {symbol}")
                del trade_info['tp2_order_id']
        except Exception as e:
            logger.warning(f"Error checking TP2 order status for {symbol}: {e}")
    
    # If all required TPs exist, we're done
    if use_single_tp:
        if tp2_exists:
            return True
    else:
        if tp1_exists and tp2_exists:
            return True
    
    # Check if position exists
    try:
        positions = client.futures_position_information(symbol=symbol)
        position_to_use = None
        for position in positions:
            position_amt = float(position.get('positionAmt', 0))
            if abs(position_amt) > 0:
                position_to_use = position
                break
        
        if not position_to_use:
            return False  # No position exists yet
        
        # Get TP details
        tp_side = trade_info.get('tp_side', 'SELL')
        
        if use_single_tp:
            # Single TP mode: Only create TP2 (main TP)
            tp2_price = trade_info['tp2_price']
            tp2_quantity = trade_info.get('tp2_quantity', 0)
            
            if not tp2_exists and tp2_price and tp2_quantity > 0:
                tp2_order_id = create_single_tp_order(symbol, tp2_price, tp2_quantity, tp_side, trade_info, tp_number=2)
                if tp2_order_id:
                    trade_info['tp2_order_id'] = tp2_order_id
                    logger.info(f"✅ Main TP order created and stored for {symbol} (Single TP mode)")
                    # Clean up stored price
                    if 'tp2_price' in trade_info:
                        del trade_info['tp2_price']
                    if 'tp2_quantity' in trade_info:
                        del trade_info['tp2_quantity']
                    return True
        else:
            # TP1 + TP2 mode: Create both
            tp1_price = trade_info.get('tp1_price')
            tp2_price = trade_info['tp2_price']
            tp1_quantity = trade_info.get('tp1_quantity', 0)
            tp2_quantity = trade_info.get('tp2_quantity', 0)
            
            # Create TP1 if it doesn't exist
            if not tp1_exists and tp1_price and tp1_quantity > 0:
                tp1_order_id = create_single_tp_order(symbol, tp1_price, tp1_quantity, tp_side, trade_info, tp_number=1)
                if tp1_order_id:
                    trade_info['tp1_order_id'] = tp1_order_id
                    logger.info(f"✅ TP1 order created and stored for {symbol}")
                else:
                    logger.warning(f"⚠️ Failed to create TP1 order for {symbol}")
            
            # Create TP2 if it doesn't exist
            if not tp2_exists and tp2_price and tp2_quantity > 0:
                tp2_order_id = create_single_tp_order(symbol, tp2_price, tp2_quantity, tp_side, trade_info, tp_number=2)
                if tp2_order_id:
                    trade_info['tp2_order_id'] = tp2_order_id
                    logger.info(f"✅ TP2 order created and stored for {symbol}")
                else:
                    logger.warning(f"⚠️ Failed to create TP2 order for {symbol}")
            
            # If both orders created successfully, clean up stored prices (but keep order IDs)
            if 'tp1_order_id' in trade_info and 'tp2_order_id' in trade_info:
                if 'tp1_price' in trade_info:
                    del trade_info['tp1_price']
                if 'tp2_price' in trade_info:
                    del trade_info['tp2_price']
                if 'tp1_quantity' in trade_info:
                    del trade_info['tp1_quantity']
                if 'tp2_quantity' in trade_info:
                    del trade_info['tp2_quantity']
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"❌ Error creating TP1/TP2 orders for {symbol}: {e}", exc_info=True)
        return False


def update_trailing_stop_loss(symbol, trade_info, position):
    """Update stop loss to breakeven/profit after TP1 is filled (trailing stop loss)
    
    Args:
        symbol: Trading symbol
        trade_info: Trade info dict
        position: Position data from Binance
    
    Returns:
        bool: True if SL was updated, False otherwise
    """
    if not ENABLE_TRAILING_STOP_LOSS:
        return False
    
    # Only apply trailing SL if TP1 was filled (for TP1+TP2 mode)
    if trade_info.get('use_single_tp', False):
        return False  # Single TP mode doesn't need trailing SL
    
    # Check if TP1 order was filled
    tp1_order_id = trade_info.get('tp1_order_id')
    if not tp1_order_id:
        return False  # TP1 not created yet
    
    # Check if TP1 order still exists (if not, it was filled)
    try:
        open_orders = client.futures_get_open_orders(symbol=symbol)
        tp1_still_exists = any(o.get('orderId') == tp1_order_id for o in open_orders)
        
        if tp1_still_exists:
            return False  # TP1 not filled yet, no trailing SL needed
        
        # TP1 was filled - check if we already moved SL
        if trade_info.get('sl_moved_to_breakeven', False):
            return False  # Already moved SL, no need to update again
        
    except Exception as e:
        logger.warning(f"Error checking TP1 status for {symbol}: {e}")
        return False
    
    # TP1 was filled - calculate new stop loss (breakeven + small profit)
    try:
        # Get position details
        position_amt = float(position.get('positionAmt', 0))
        entry_price = float(position.get('entryPrice', 0))
        mark_price = float(position.get('markPrice', 0))
        
        if abs(position_amt) == 0 or entry_price <= 0:
            return False  # Invalid position data
        
        # Get original stop loss and entry prices
        original_sl = safe_float(trade_info.get('original_stop_loss'), default=0)
        original_entry1 = safe_float(trade_info.get('original_entry1'), default=entry_price)
        original_entry2 = safe_float(trade_info.get('original_entry2'), default=entry_price)
        
        # Calculate average entry (for breakeven calculation)
        if original_entry2 and original_entry2 > 0 and original_entry2 != original_entry1:
            avg_entry = (original_entry1 + original_entry2) / 2
        else:
            avg_entry = original_entry1
        
        # Determine position side
        is_long = position_amt > 0
        
        # Calculate new stop loss: breakeven + small profit (0.5% default)
        breakeven_profit_percent = TRAILING_SL_BREAKEVEN_PERCENT / 100.0
        
        if is_long:
            new_sl_price = avg_entry * (1 + breakeven_profit_percent)
            # Ensure new SL is above current mark price (safety check)
            if new_sl_price > mark_price:
                new_sl_price = mark_price * 0.999  # Slightly below current price
        else:  # SHORT
            new_sl_price = avg_entry * (1 - breakeven_profit_percent)
            # Ensure new SL is below current mark price (safety check)
            if new_sl_price < mark_price:
                new_sl_price = mark_price * 1.001  # Slightly above current price
        
        # Get symbol info for precision
        exchange_info = client.futures_exchange_info()
        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
        if not symbol_info:
            return False
        
        price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
        tick_size = float(price_filter['tickSize']) if price_filter else 0.01
        new_sl_price = format_price_precision(new_sl_price, tick_size)
        
        # Get current stop loss order
        current_sl_id = trade_info.get('sl_order_id')
        current_sl_price = safe_float(trade_info.get('current_sl_price'), default=0)
        
        # Only update if new SL is better than current SL
        if is_long:
            sl_needs_update = new_sl_price > current_sl_price
        else:  # SHORT
            sl_needs_update = new_sl_price < current_sl_price or current_sl_price == 0
        
        if not sl_needs_update:
            return False  # Current SL is already better
        
        # Cancel old stop loss if exists
        if current_sl_id:
            try:
                client.futures_cancel_order(symbol=symbol, orderId=current_sl_id)
                logger.info(f"🔄 Cancelled old stop loss order {current_sl_id} for {symbol}")
            except Exception as e:
                logger.warning(f"Error cancelling old SL for {symbol}: {e}")
        
        # Create new stop loss at breakeven + profit
        sl_side = 'SELL' if is_long else 'BUY'
        position_side = position.get('positionSide', 'BOTH')
        
        try:
            is_hedge_mode = get_position_mode(symbol)
        except:
            is_hedge_mode = False
        
        sl_params = {
            'symbol': symbol,
            'side': sl_side,
            'type': 'STOP_MARKET',
            'timeInForce': 'GTC',
            'stopPrice': new_sl_price,
            'closePosition': True,  # Close entire remaining position
            'workingType': 'MARK_PRICE',
        }
        
        if is_hedge_mode and position_side != 'BOTH':
            sl_params['positionSide'] = position_side
        
        new_sl_order = client.futures_create_order(**sl_params)
        new_sl_id = new_sl_order.get('orderId')
        
        # Update trade info
        trade_info['sl_order_id'] = new_sl_id
        trade_info['current_sl_price'] = new_sl_price
        trade_info['sl_moved_to_breakeven'] = True
        trade_info['tp1_filled'] = True  # Mark TP1 as filled
        
        logger.info(f"✅ Trailing stop loss updated for {symbol}: Moved SL to breakeven+profit @ ${new_sl_price:,.8f} "
                   f"(Original SL: ${original_sl:,.8f}, Entry: ${avg_entry:,.8f})")
        
        # Send Slack notification
        send_slack_alert(
            error_type="Trailing Stop Loss Updated",
            message=f"TP1 filled - Stop loss moved to breakeven+{TRAILING_SL_BREAKEVEN_PERCENT}% profit",
            details={
                'Original_SL': f"${original_sl:,.8f}",
                'New_SL': f"${new_sl_price:,.8f}",
                'Entry_Price': f"${avg_entry:,.8f}",
                'TP1_Filled': True
            },
            symbol=symbol,
            severity='INFO'
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating trailing stop loss for {symbol}: {e}", exc_info=True)
        return False


# Background thread to create TP orders when positions exist
# This mimics Binance UI behavior: TP is "set" when creating limit order, but placed when order fills
# Optimized to minimize API calls and stay within Binance rate limits
def create_missing_tp_orders():
    """Background function to automatically create TP orders for positions with stored TP details
    NOTE: This background thread does NOT call AI validation - it only creates TP orders for existing positions
    Optimized to reduce API calls:
    - Only checks symbols with stored TP details (skips symbols without TP config)
    - Uses longer intervals to stay well within Binance rate limits
    - Caches results to avoid redundant checks"""
    check_count = 0
    symbols_without_tp_logged = set()  # Track symbols we've already logged warnings for
    
    while True:
        try:
            # Optimized intervals to reduce API calls while still being effective:
            # - First 4 checks (4 minutes): Every 1 minute to catch newly filled orders quickly
            # - Next 10 checks (20 minutes): Every 2 minutes for active monitoring
            # - After that: Every 5 minutes for ongoing checks
            if check_count < 4:
                sleep_time = 60  # Check every 1 minute for first 4 minutes
            elif check_count < 14:
                sleep_time = 120  # Check every 2 minutes for next 20 minutes
            else:
                sleep_time = 300  # Then check every 5 minutes (reduces API calls significantly)
            
            time.sleep(sleep_time)
            check_count += 1
            
            try:
                # Only check symbols that have stored TP details (optimization: skip symbols without TP config)
                # Check for symbols with TP configured (either single TP or TP1+TP2)
                symbols_to_check = []
                for s, info in active_trades.items():
                    if info.get('use_single_tp', False):
                        # Single TP mode: only need tp2_price
                        if 'tp2_price' in info and info.get('tp2_price'):
                            symbols_to_check.append(s)
                    else:
                        # TP1+TP2 mode: need both
                        if 'tp1_price' in info and 'tp2_price' in info:
                            symbols_to_check.append(s)
                
                if not symbols_to_check:
                    # No symbols with stored TP details - skip API calls entirely
                    continue
                
                # Get ALL open positions from Binance (single API call)
                # This single call is used for BOTH TP creation AND trailing stop loss checks
                all_positions = client.futures_position_information()
                positions_dict = {p['symbol']: p for p in all_positions if abs(float(p.get('positionAmt', 0))) > 0}
                
                # Check trailing stop loss for ALL positions (uses same position data - no extra API call!)
                # This runs BEFORE TP creation check to use the same position data
                if ENABLE_TRAILING_STOP_LOSS and positions_dict:
                    for symbol, position in positions_dict.items():
                        if symbol in active_trades:
                            try:
                                trade_info = active_trades[symbol]
                                # Only check if we have TP1 configured (TP1+TP2 mode)
                                if not trade_info.get('use_single_tp', False):
                                    update_trailing_stop_loss(symbol, trade_info, position)
                            except Exception as e:
                                logger.debug(f"Error checking trailing SL for {symbol}: {e}")
                
                if not positions_dict:
                    # No open positions - but check if there are pending orders before cleaning up TP
                    # If orders are pending, keep TP details and wait for order to fill
                    for symbol in list(symbols_to_check):
                        use_single_tp = active_trades.get(symbol, {}).get('use_single_tp', False)
                        has_tp = False
                        if use_single_tp:
                            has_tp = 'tp2_price' in active_trades.get(symbol, {}) and active_trades[symbol].get('tp2_price')
                        else:
                            has_tp = ('tp1_price' in active_trades.get(symbol, {}) or 'tp2_price' in active_trades.get(symbol, {}))
                        
                        if symbol in active_trades and has_tp:
                            # Check if there are pending orders for this symbol
                            try:
                                has_orders, open_orders = check_existing_orders(symbol, log_result=False)
                                if has_orders:
                                    # There are pending orders - keep TP details and wait for order to fill
                                    logger.debug(f"⏳ Background thread: Keeping TP1/TP2 for {symbol} (pending orders, waiting for fill)")
                                    continue  # Don't clean up - order might fill soon
                                else:
                                    # No orders and no position - clean up TP details
                                    logger.info(f"🧹 Background thread: Cleaning up stored TP1/TP2 for {symbol} (no position and no pending orders)")
                                    if 'tp1_price' in active_trades[symbol]:
                                        del active_trades[symbol]['tp1_price']
                                    if 'tp2_price' in active_trades[symbol]:
                                        del active_trades[symbol]['tp2_price']
                                    if 'tp1_quantity' in active_trades[symbol]:
                                        del active_trades[symbol]['tp1_quantity']
                                    if 'tp2_quantity' in active_trades[symbol]:
                                        del active_trades[symbol]['tp2_quantity']
                                    if 'tp_working_type' in active_trades[symbol]:
                                        del active_trades[symbol]['tp_working_type']
                            except Exception as e:
                                logger.debug(f"Error checking orders for {symbol}: {e}")
                                # On error, keep TP details (safer to keep than delete)
                                continue
                    continue
                
                # Only log if we're actually checking symbols (reduce log spam)
                if len(symbols_to_check) > 0:
                    logger.debug(f"Background thread: Checking {len(symbols_to_check)} symbol(s) with stored TP details")
                
                # Only check symbols that have stored TP details AND have open positions
                symbols_checked = 0
                for symbol in symbols_to_check:
                    if symbol not in positions_dict:
                        # Symbol has stored TP but no position - check if orders are pending
                        # Keep TP details if orders exist (waiting for fill), only clean up if no orders
                        try:
                            has_orders, open_orders = check_existing_orders(symbol, log_result=False)
                            
                            # Also check if we have tracked order IDs for this symbol
                            has_tracked_orders = False
                            if symbol in active_trades:
                                # Check if we have any order IDs tracked (Order 1, Order 2, or Order 3)
                                if ('primary_order_id' in active_trades[symbol] or 
                                    'dca_order_id' in active_trades[symbol] or 
                                    'optimized_entry1_order_id' in active_trades[symbol]):
                                    # Verify these specific orders still exist
                                    tracked_order_ids = []
                                    if 'primary_order_id' in active_trades[symbol]:
                                        tracked_order_ids.append(active_trades[symbol]['primary_order_id'])
                                    if 'dca_order_id' in active_trades[symbol]:
                                        tracked_order_ids.append(active_trades[symbol]['dca_order_id'])
                                    if 'optimized_entry1_order_id' in active_trades[symbol]:
                                        tracked_order_ids.append(active_trades[symbol]['optimized_entry1_order_id'])
                                    
                                    # Check if any tracked orders still exist
                                    if tracked_order_ids:
                                        existing_order_ids = [o.get('orderId') for o in open_orders]
                                        has_tracked_orders = any(oid in existing_order_ids for oid in tracked_order_ids)
                            
                            if has_orders or has_tracked_orders:
                                # There are pending orders - keep TP details and wait for order to fill
                                logger.debug(f"⏳ Background thread: Keeping TP for {symbol} (pending orders, waiting for fill)")
                                continue  # Don't clean up - order might fill soon
                            else:
                                # No orders and no position - clean up TP details
                                logger.info(f"🧹 Background thread: Cleaning up stored TP1/TP2 for {symbol} (no position and no pending orders)")
                                if symbol in active_trades:
                                    if 'tp1_price' in active_trades[symbol]:
                                        del active_trades[symbol]['tp1_price']
                                    if 'tp2_price' in active_trades[symbol]:
                                        del active_trades[symbol]['tp2_price']
                                    if 'tp1_quantity' in active_trades[symbol]:
                                        del active_trades[symbol]['tp1_quantity']
                                    if 'tp2_quantity' in active_trades[symbol]:
                                        del active_trades[symbol]['tp2_quantity']
                                    if 'tp_working_type' in active_trades[symbol]:
                                        del active_trades[symbol]['tp_working_type']
                        except Exception as e:
                            logger.debug(f"Error checking orders for {symbol}: {e}")
                            # On error, keep TP details (safer to keep than delete)
                            continue
                        continue
                    
                    symbols_checked += 1
                    position = positions_dict[symbol]
                    position_amt = float(position.get('positionAmt', 0))
                    
                    try:
                        # Check if TP orders already exist (don't log every check to reduce spam)
                        has_orders, open_orders = check_existing_orders(symbol, log_result=False)
                        existing_tps = [o for o in open_orders if o.get('type') == 'TAKE_PROFIT_MARKET']
                        
                        use_single_tp = active_trades.get(symbol, {}).get('use_single_tp', False)
                        
                        if use_single_tp:
                            # Single TP mode: only need 1 TP order
                            if len(existing_tps) >= 1:
                                if symbol in active_trades:
                                    active_trades[symbol]['tp2_order_id'] = existing_tps[0].get('orderId')
                                    # Clean up stored TP details
                                    if 'tp2_price' in active_trades[symbol]:
                                        del active_trades[symbol]['tp2_price']
                                    if 'tp2_quantity' in active_trades[symbol]:
                                        del active_trades[symbol]['tp2_quantity']
                                    if 'tp_working_type' in active_trades[symbol]:
                                        del active_trades[symbol]['tp_working_type']
                                continue  # TP exists, skip
                        else:
                            # TP1+TP2 mode: need 2 TP orders
                            if len(existing_tps) >= 2:
                                if symbol in active_trades:
                                    if len(existing_tps) >= 1:
                                        active_trades[symbol]['tp1_order_id'] = existing_tps[0].get('orderId')
                                    if len(existing_tps) >= 2:
                                        active_trades[symbol]['tp2_order_id'] = existing_tps[1].get('orderId')
                                    # Clean up stored TP details
                                    if 'tp1_price' in active_trades[symbol]:
                                        del active_trades[symbol]['tp1_price']
                                    if 'tp2_price' in active_trades[symbol]:
                                        del active_trades[symbol]['tp2_price']
                                    if 'tp1_quantity' in active_trades[symbol]:
                                        del active_trades[symbol]['tp1_quantity']
                                    if 'tp2_quantity' in active_trades[symbol]:
                                        del active_trades[symbol]['tp2_quantity']
                                    if 'tp_working_type' in active_trades[symbol]:
                                        del active_trades[symbol]['tp_working_type']
                                continue  # Both TPs exist, skip
                        
                        # No TP orders exist - we have stored TP details, create TP1 and TP2 orders
                        trade_info = active_trades[symbol]
                        logger.info(f"🔄 Background thread: Position exists for {symbol} with stored TP details - creating TP1 and TP2 orders")
                        success = create_tp1_tp2_if_needed(symbol, trade_info)
                        if success:
                            logger.info(f"✅ Background thread: TP1 and TP2 orders created successfully for {symbol}")
                        else:
                            logger.warning(f"⚠️ Background thread: Failed to create TP1/TP2 for {symbol} (check logs for details)")
                        
                    
                    except Exception as e:
                        logger.error(f"Error processing position {symbol}: {e}", exc_info=True)
                        send_slack_alert(
                            error_type="Background TP Check Error",
                            message=str(e),
                            details={'Position_Amount': position_amt if 'position_amt' in locals() else 'Unknown'},
                            symbol=symbol,
                            severity='WARNING'
                        )
                
                # Check trailing stop loss for ALL positions (not just those with TP details)
                # This runs after TP creation check, uses same position data (no extra API call!)
                if ENABLE_TRAILING_STOP_LOSS and positions_dict:
                    for symbol, position in positions_dict.items():
                        if symbol in active_trades:
                            try:
                                trade_info = active_trades[symbol]
                                # Only check if we have TP1 configured (TP1+TP2 mode)
                                if not trade_info.get('use_single_tp', False):
                                    update_trailing_stop_loss(symbol, trade_info, position)
                            except Exception as e:
                                logger.debug(f"Error checking trailing SL for {symbol}: {e}")
                
                # Also check for positions without stored TP details - calculate TP from position
                for symbol, position in positions_dict.items():
                    if symbol not in symbols_to_check:
                        # Position exists but no stored TP details - calculate TP from current position
                        # Check if there are pending orders (entry might not be filled yet)
                        try:
                            has_orders, open_orders = check_existing_orders(symbol, log_result=False)
                            existing_tp = [o for o in open_orders if o.get('type') == 'TAKE_PROFIT_MARKET']
                            
                            if existing_tp:
                                # TP already exists, mark as processed to avoid repeated checks
                                if symbol not in symbols_without_tp_logged:
                                    symbols_without_tp_logged.add(symbol)
                                continue
                            
                            if not has_orders:
                                # No pending orders and no TP - calculate TP from position
                                position_amt = float(position.get('positionAmt', 0))
                                entry_price = float(position.get('entryPrice', 0))
                                position_side = position.get('positionSide', 'BOTH')
                                
                                if entry_price > 0 and abs(position_amt) > 0:
                                    # Calculate TP: 2.1% profit target (adjustable)
                                    DEFAULT_TP_PERCENT = 0.021  # 2.1% profit
                                    
                                    if position_amt > 0:  # LONG position
                                        tp_price = entry_price * (1 + DEFAULT_TP_PERCENT)
                                        tp_side = 'SELL'
                                    else:  # SHORT position
                                        tp_price = entry_price * (1 - DEFAULT_TP_PERCENT)
                                        tp_side = 'BUY'
                                    
                                    # Get symbol info for precision
                                    try:
                                        exchange_info = client.futures_exchange_info()
                                        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
                                        
                                        if symbol_info:
                                            # Format price precision
                                            price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
                                            if price_filter:
                                                tick_size = float(price_filter['tickSize'])
                                                tp_price = format_price_precision(tp_price, tick_size)
                                            
                                            # Format quantity precision
                                            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                                            if lot_size_filter:
                                                step_size = float(lot_size_filter['stepSize'])
                                                tp_quantity = format_quantity_precision(abs(position_amt), step_size)
                                            else:
                                                tp_quantity = abs(position_amt)
                                            
                                            # Detect position mode
                                            is_hedge_mode = get_position_mode(symbol)
                                            
                                            # Create temporary trade_info for TP creation
                                            # Note: position_side will be retrieved from the position object by create_tp_if_needed
                                            temp_trade_info = {
                                                'tp_price': tp_price,
                                                'tp_quantity': tp_quantity,
                                                'tp_side': tp_side,
                                                'tp_working_type': 'MARK_PRICE'
                                            }
                                            
                                            logger.info(f"🔄 Background thread: Creating TP for {symbol} from position (calculated: {tp_price}, qty: {tp_quantity})")
                                            success = create_tp_if_needed(symbol, temp_trade_info)
                                            
                                            if success:
                                                logger.info(f"✅ Background thread: TP order created successfully for {symbol} (calculated from position)")
                                                # Only mark as processed if TP creation succeeded
                                                if symbol not in symbols_without_tp_logged:
                                                    symbols_without_tp_logged.add(symbol)
                                            else:
                                                logger.warning(f"⚠️ Background thread: Failed to create calculated TP for {symbol} (will retry on next cycle)")
                                                # Don't add to symbols_without_tp_logged - allow retry on next cycle
                                        else:
                                            logger.debug(f"Symbol info not found for {symbol}")
                                    except Exception as e:
                                        logger.debug(f"Error getting symbol info or creating TP for {symbol}: {e}")
                                        # Don't add to symbols_without_tp_logged - allow retry on next cycle
                                else:
                                    logger.debug(f"Skipping TP calculation for {symbol}: invalid entry_price or position_amt")
                        except Exception as e:
                            logger.debug(f"Error calculating TP for {symbol}: {e}")
                            pass
                            
            except Exception as e:
                logger.error(f"Error getting positions in background thread: {e}", exc_info=True)
                send_slack_alert(
                    error_type="Background Thread Error",
                    message=f"Error getting positions: {str(e)}",
                    severity='ERROR'
                )
                
        except Exception as e:
            logger.error(f"Error in TP creation background thread: {e}", exc_info=True)
            send_slack_alert(
                error_type="TP Creation Background Thread Error",
                message=str(e),
                severity='ERROR'
            )

# Start background thread for TP creation (only if client is initialized)
if client:
    tp_thread = threading.Thread(target=create_missing_tp_orders, daemon=True)
    tp_thread.start()
    logger.info("Background TP creation thread started")
else:
    logger.warning("Binance client not initialized - TP creation thread not started")  # seconds - prevent duplicate EXIT processing


def close_position_at_market(symbol, signal_side, is_hedge_mode=False):
    """Close position at market price"""
    try:
        # Get ALL positions first, then filter by symbol (more reliable than filtering in API call)
        all_positions = client.futures_position_information()
        position_to_close = None
        
        for position in all_positions:
            position_symbol = position.get('symbol', '')
            position_amt = float(position.get('positionAmt', 0))
            
            # Match symbol (case-insensitive) and check if position exists
            if position_symbol.upper() == symbol.upper() and abs(position_amt) > 0:
                position_side = position.get('positionSide', 'BOTH')
                # Check if position side matches (or if it's BOTH mode)
                if position_side == 'BOTH' or position_side == signal_side.upper() or signal_side.upper() == 'BOTH':
                    position_to_close = position
                    break
        
        if not position_to_close:
            logger.info(f"No open position found for {symbol} to close")
            return {'success': False, 'error': 'No position to close'}
        
        position_amt = float(position_to_close.get('positionAmt', 0))
        if abs(position_amt) == 0:
            logger.info(f"Position amount is zero for {symbol}")
            return {'success': False, 'error': 'Position amount is zero'}
        
        # Determine close side (opposite of position)
        # If position is positive (LONG), we need to SELL
        # If position is negative (SHORT), we need to BUY
        close_side = 'SELL' if position_amt > 0 else 'BUY'
        close_quantity = abs(position_amt)
        
        # Get symbol info for quantity precision
        exchange_info = client.futures_exchange_info()
        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
        if symbol_info:
            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            if lot_size_filter:
                step_size = float(lot_size_filter['stepSize'])
                close_quantity = format_quantity_precision(close_quantity, step_size)
        
        # Create market order to close position
        # Try with reduceOnly first, then without if it fails (some accounts/modes don't accept it)
        close_params = {
            'symbol': symbol,
            'side': close_side,
            'type': 'MARKET',
            'quantity': close_quantity,
        }
        
        # Only include positionSide if in Hedge mode
        if is_hedge_mode:
            position_side_str = 'LONG' if position_amt > 0 else 'SHORT'
            close_params['positionSide'] = position_side_str
        
        # Try with reduceOnly first (for safety)
        close_params_with_reduce = close_params.copy()
        close_params_with_reduce['reduceOnly'] = True
        
        logger.info(f"Closing position at market for {symbol}: {close_params_with_reduce}")
        try:
            result = client.futures_create_order(**close_params_with_reduce)
            logger.info(f"Position closed successfully: {result}")
        except BinanceAPIException as e:
            # If reduceOnly is not accepted, try without it
            if e.code == -1106 or 'reduceonly' in str(e).lower() or 'not required' in str(e).lower():
                logger.warning(f"reduceOnly not accepted for {symbol}, retrying without it: {e}")
                try:
                    result = client.futures_create_order(**close_params)
                    logger.info(f"Position closed successfully (without reduceOnly): {result}")
                except Exception as e2:
                    logger.error(f"Failed to close position even without reduceOnly: {e2}")
                    raise e2  # Re-raise the retry error
            else:
                raise  # Re-raise if it's a different error
        
        return {
            'success': True,
            'order_id': result.get('orderId'),
            'symbol': symbol,
            'quantity': close_quantity,
            'side': close_side
        }
        
    except BinanceAPIException as e:
        logger.error(f"Binance API error closing position: {e}")
        send_slack_alert(
            error_type="Position Close Failed",
            message=f"{e.message} (Code: {e.code})",
            details={'Error_Code': e.code, 'Close_Side': close_side, 'Quantity': close_quantity},
            symbol=symbol,
            severity='ERROR'
        )
        return {'success': False, 'error': f'Binance API error: {e.message}'}
    except Exception as e:
        logger.error(f"Error closing position: {e}", exc_info=True)
        send_slack_alert(
            error_type="Position Close Error",
            message=str(e),
            details={'Close_Side': close_side, 'Quantity': close_quantity},
            symbol=symbol,
            severity='ERROR'
        )
        return {'success': False, 'error': str(e)}


def create_limit_order(signal_data):
    """Create a Binance Futures limit order with configurable entry size and leverage"""
    try:
        # Extract signal data
        token = signal_data.get('token')
        event = signal_data.get('event')
        signal_side = signal_data.get('signal_side')
        symbol = format_symbol(signal_data.get('symbol', ''))
        timeframe = signal_data.get('timeframe', 'Unknown')  # Extract timeframe early for rejection notifications
        
        # Safely parse prices (handles None, "null" string, and invalid values)
        entry_price = safe_float(signal_data.get('entry_price'), default=None)
        stop_loss = safe_float(signal_data.get('stop_loss'), default=None)
        take_profit = safe_float(signal_data.get('take_profit'), default=None)
        second_entry_price = safe_float(signal_data.get('second_entry_price'), default=None)
        
        # Extract indicators if available (for ATR-based Entry 2 calculation)
        indicators = signal_data.get('indicators', {})
        
        reduce_only = signal_data.get('reduce_only', False)
        order_subtype = signal_data.get('order_subtype', 'primary_entry')
        
        # Verify token
        if not verify_webhook_token(token):
            logger.warning(f"Invalid webhook token received")
            return {'success': False, 'error': 'Invalid token'}
        
        # Validate required fields for ENTRY events
        if event == 'ENTRY':
            if entry_price is None or entry_price <= 0:
                logger.warning(f"Invalid or missing entry_price in webhook payload. Discarding request. entry_price={signal_data.get('entry_price')}")
                return {'success': False, 'error': 'Invalid or missing entry_price (NA/null)'}
            
            # AI Signal Validation (ONLY for NEW ENTRY signals - NOT for order tracking or TP creation)
            logger.info(f"🔍 [AI VALIDATION] Processing NEW ENTRY signal for {symbol} - AI validation will run")
            validation_result = validate_signal_with_ai(signal_data)
            
            # Extract validation results
            is_valid = validation_result.get('is_valid', True)
            confidence_score = validation_result.get('confidence_score', 100.0)
            confidence_threshold = AI_VALIDATION_MIN_CONFIDENCE  # Default: 55%
            reasoning = validation_result.get('reasoning', '')
            entry1_is_bad, entry2_is_good_from_parsing = parse_entry_analysis_from_reasoning(reasoning)
            has_high_volatility, price_change_pct = check_recent_price_volatility(symbol, days=7)
            
            # Check if Entry 1 failed validation
            # Entry 1 fails ONLY if: is_valid=False (explicitly rejected by AI)
            # NOTE: If is_valid=True, Entry 1 is approved and will proceed (even if confidence is 45-49%, approval logic handles it)
            # NOTE: Parsing result (entry1_is_bad) is only used as additional context, not to determine failure
            # If Entry 1 is approved with good confidence, ignore parsing result (parsing can have false positives)
            entry1_failed = not is_valid  # Only fail if AI explicitly rejects (is_valid=False)
            # Only add parsing result if Entry 1 is already failing
            if entry1_failed and entry1_is_bad:
                logger.info(f"   Entry 1 failed AND parsing detected Entry 1 as bad - will check Entry 2")
            elif entry1_failed:
                logger.info(f"   Entry 1 failed (is_valid={is_valid}, confidence={confidence_score:.1f}%) - will check Entry 2")
            elif entry1_is_bad and is_valid and confidence_score >= 50.0:
                # Entry 1 is approved but parsing says it's bad - ignore parsing (false positive)
                logger.info(f"   Entry 1 APPROVED (is_valid={is_valid}, confidence={confidence_score:.1f}%) - ignoring parsing result (Entry 1 is good)")
                entry1_is_bad = False  # Override parsing result since Entry 1 is actually approved
            
            # Check if we have Entry 2 price (original)
            entry2_price_original = second_entry_price if second_entry_price and second_entry_price > 0 else None
            
            # Get optimized Entry 2 price if available (from AI optimization)
            entry2_price_optimized = None
            if 'optimized_prices' in validation_result:
                opt_prices = validation_result['optimized_prices']
                entry2_price_optimized = safe_float(opt_prices.get('second_entry_price'), default=None)
            
            # ENTRY 2 VALIDATION LOGIC (CRITICAL - CHECK BEFORE REJECTION):
            # If Entry 1 failed, ALWAYS check Entry 2 as standalone trade (both original and optimized)
            # Only reject completely if BOTH Entry 2 options fail
            entry2_standalone_valid = False
            entry2_standalone_result = None
            entry2_price_to_use = None
            
            if entry1_failed and (entry2_price_original is not None or entry2_price_optimized is not None):
                logger.info(f"🔍 Entry 1 failed validation (is_valid={is_valid}, confidence={confidence_score:.1f}%, parsed_bad={entry1_is_bad})")
                logger.info(f"   Checking Entry 2 as standalone trade to avoid missing profitable trades")
                
                # Try Entry 2 with ORIGINAL price first
                if entry2_price_original is not None:
                    logger.info(f"   📍 Testing Entry 2 with ORIGINAL price: ${entry2_price_original:,.8f}")
                    entry2_result_original = validate_entry2_standalone_with_ai(
                        signal_data=signal_data,
                        entry2_price=entry2_price_original,
                        original_validation_result=validation_result
                    )
                    
                    entry2_valid_original = entry2_result_original.get('is_valid', False)
                    entry2_confidence_original = entry2_result_original.get('confidence_score', 0.0)
                    
                    if entry2_valid_original and entry2_confidence_original >= 50.0:
                        logger.info(f"✅ AI APPROVED Entry 2 with ORIGINAL price: Confidence={entry2_confidence_original:.1f}%")
                        entry2_standalone_valid = True
                        entry2_standalone_result = entry2_result_original
                        entry2_price_to_use = entry2_price_original
                    else:
                        logger.warning(f"🚫 AI REJECTED Entry 2 with ORIGINAL price: Confidence={entry2_confidence_original:.1f}%")
                
                # If original Entry 2 failed, try OPTIMIZED Entry 2 price
                if not entry2_standalone_valid and entry2_price_optimized is not None:
                    logger.info(f"   📍 Testing Entry 2 with OPTIMIZED price: ${entry2_price_optimized:,.8f}")
                    entry2_result_optimized = validate_entry2_standalone_with_ai(
                        signal_data=signal_data,
                        entry2_price=entry2_price_optimized,
                        original_validation_result=validation_result
                    )
                    
                    entry2_valid_optimized = entry2_result_optimized.get('is_valid', False)
                    entry2_confidence_optimized = entry2_result_optimized.get('confidence_score', 0.0)
                    
                    if entry2_valid_optimized and entry2_confidence_optimized >= 50.0:
                        logger.info(f"✅ AI APPROVED Entry 2 with OPTIMIZED price: Confidence={entry2_confidence_optimized:.1f}%")
                        entry2_standalone_valid = True
                        entry2_standalone_result = entry2_result_optimized
                        entry2_price_to_use = entry2_price_optimized
                    else:
                        logger.warning(f"🚫 AI REJECTED Entry 2 with OPTIMIZED price: Confidence={entry2_confidence_optimized:.1f}%")
                
                # If both Entry 2 options failed, log it but continue to check if we should still reject
                if not entry2_standalone_valid:
                    logger.warning(f"🚫 Both Entry 2 options (original and optimized) were REJECTED by AI")
            
            # Special case: Use Entry 2 only if Entry 1 failed AND Entry 2 passed validation
            # NOTE: Volatility check is removed - if AI approves Entry 2, we trust it (don't want to miss profitable trades)
            should_use_entry2_only = (
                entry1_failed and
                entry2_standalone_valid and
                entry2_standalone_result is not None and
                entry2_price_to_use is not None and
                entry2_standalone_result.get('confidence_score', 0.0) >= 50.0
                # Volatility check removed: (has_high_volatility or price_change_pct > 5.0)
                # If AI approves Entry 2 with >=50% confidence, we trust it regardless of volatility
            )
            
            if should_use_entry2_only:
                entry2_confidence = entry2_standalone_result.get('confidence_score', 60.0)
                logger.info(f"🎯 SPECIAL CASE DETECTED: Entry 1 rejected but Entry 2 APPROVED by AI as standalone trade for {symbol}")
                logger.info(f"   Entry 1 Analysis: Rejected (is_valid={is_valid}, confidence={confidence_score:.1f}%)")
                logger.info(f"   Entry 2 Standalone Validation: ✅ APPROVED by AI")
                logger.info(f"   Entry 2 Price: ${entry2_price_to_use:,.8f} ({'OPTIMIZED' if entry2_price_to_use == entry2_price_optimized else 'ORIGINAL'})")
                logger.info(f"   Entry 2 Confidence: {entry2_confidence:.1f}%")
                logger.info(f"   Recent Volatility: {price_change_pct:.2f}% over 7 days (High: {has_high_volatility})")
                logger.info(f"   Decision: Skipping Entry 1, creating Entry 2 only order with $20 and custom TP (4-5%)")
                
                # Set flags for Entry 2 only trade
                signal_data['_special_entry2_only'] = True
                signal_data['_entry2_only_price'] = entry2_price_to_use
                signal_data['_entry2_standalone_result'] = entry2_standalone_result
                
                # Override validation to allow this special case (use Entry 2's validation result)
                validation_result['is_valid'] = True
                validation_result['confidence_score'] = entry2_confidence
                validation_result['risk_level'] = entry2_standalone_result.get('risk_level', 'MEDIUM')
                validation_result['special_case'] = 'ENTRY2_ONLY'
                validation_result['special_case_reason'] = f'Entry 1 rejected but Entry 2 APPROVED by AI as standalone trade (Confidence: {entry2_confidence:.1f}%, Price: ${entry2_price_to_use:,.8f}). Recent volatility: {price_change_pct:.2f}%'
                validation_result['entry2_standalone_reasoning'] = entry2_standalone_result.get('reasoning', '')
            
            # APPROVAL LOGIC (More lenient - matches AI prompt instructions):
            # 1. If AI explicitly approves (is_valid=True) and confidence >= 50%: APPROVE
            # 2. If confidence >= 55% (threshold): APPROVE
            # 3. If confidence 50-54% and AI says is_valid=True: APPROVE (AI prompt says "APPROVE with caution")
            # 4. If confidence 45-49% and is_valid=True and R/R >= 1.0: APPROVE (AI prompt allows this)
            # 5. Only reject if confidence < 45% OR (confidence < 50% AND is_valid=False)
            # BUT: If Entry 2 passed validation, we already handled it above, so don't reject here
            
            should_approve = False
            if is_valid and confidence_score >= 50.0:
                # AI explicitly approved and confidence is 50%+: APPROVE
                should_approve = True
                logger.info(f"✅ AI Validation APPROVED signal for {symbol}: Confidence={confidence_score:.1f}% (AI explicitly approved with is_valid=True)")
            elif confidence_score >= confidence_threshold:
                # Confidence meets threshold: APPROVE
                should_approve = True
                logger.info(f"✅ AI Validation APPROVED signal for {symbol}: Confidence={confidence_score:.1f}% (meets threshold {confidence_threshold}%)")
            elif is_valid and confidence_score >= 45.0:
                # AI approved with 45-49% confidence: APPROVE (AI prompt allows this if R/R >= 1.0)
                should_approve = True
                logger.info(f"✅ AI Validation APPROVED signal for {symbol}: Confidence={confidence_score:.1f}% (AI approved, within acceptable range 45-49%)")
            
            # Only reject if Entry 1 failed AND Entry 2 also failed (both options rejected)
            if not should_approve and not signal_data.get('_special_entry2_only', False):
                # Check if Entry 2 validation was attempted but failed
                if entry1_failed and (entry2_price_original is not None or entry2_price_optimized is not None):
                    if not entry2_standalone_valid:
                        # Both Entry 1 and Entry 2 failed - complete rejection
                        rejection_reason = f"Signal REJECTED: Entry 1 failed (is_valid={is_valid}, confidence={confidence_score:.1f}%) AND Entry 2 standalone validation also failed (both original and optimized prices rejected)"
                        logger.warning(f"🚫 COMPLETE REJECTION for {symbol}: {rejection_reason}")
                        logger.info(f"   Entry 1 Reasoning: {validation_result.get('reasoning', 'No reasoning provided')}")
                        logger.info(f"   Entry 2 was tested but also rejected by AI")
                        logger.info(f"   This is a FALSE SIGNAL - both Entry 1 and Entry 2 failed validation")
                    else:
                        # Entry 2 passed but conditions not met (shouldn't happen, but safety check)
                        rejection_reason = f"Entry 1 failed but Entry 2 validation conditions not fully met"
                        logger.warning(f"⚠️  Edge case: Entry 2 passed but conditions not met")
                else:
                    # Entry 1 failed and no Entry 2 available
                    rejection_reason = f"Confidence score {confidence_score:.1f}% is below acceptable threshold (AI: is_valid={is_valid}, threshold: {confidence_threshold}%)"
                    logger.warning(f"🚫 AI Validation REJECTED signal for {symbol}: {rejection_reason}")
                logger.info(f"   Reasoning: {validation_result.get('reasoning', 'No reasoning provided')}")
                logger.info(f"   Risk Level: {validation_result.get('risk_level', 'UNKNOWN')}")
                
                # Send rejection notification to Slack exception channel
                full_reason = f"{rejection_reason}\n\n{validation_result.get('reasoning', 'No detailed reasoning provided')}"
                send_signal_rejection_notification(
                    symbol=symbol,
                    signal_side=signal_side,
                    timeframe=timeframe,
                    entry_price=entry_price,
                    rejection_reason=full_reason,
                    confidence_score=confidence_score,
                    risk_level=validation_result.get('risk_level'),
                    validation_result=validation_result
                )
                
                return {
                    'success': False,
                    'error': f'Signal rejected: Entry 1 failed and Entry 2 also failed validation',
                    'validation_result': validation_result
                }
            
            # Log successful validation
            logger.info(f"✅ AI Validation APPROVED signal for {symbol}: Confidence={confidence_score:.1f}%, "
                       f"Risk={validation_result.get('risk_level', 'UNKNOWN')}, "
                       f"Reasoning={validation_result.get('reasoning', 'No reasoning')}")
            
            # Apply optimized prices if available
            if 'optimized_prices' in validation_result:
                opt_prices = validation_result['optimized_prices']
                # Update prices with optimized values
                if opt_prices.get('entry_price') and opt_prices['entry_price'] != entry_price:
                    entry_price = opt_prices['entry_price']
                    logger.info(f"🔄 [PRICE UPDATE] Using AI-optimized entry price: ${entry_price:,.8f}")
                if opt_prices.get('stop_loss') and opt_prices['stop_loss'] != stop_loss:
                    stop_loss = opt_prices['stop_loss']
                    logger.info(f"🔄 [PRICE UPDATE] Using AI-optimized stop loss: ${stop_loss:,.8f}")
                if opt_prices.get('take_profit') and opt_prices['take_profit'] != take_profit:
                    take_profit = opt_prices['take_profit']
                    logger.info(f"🔄 [PRICE UPDATE] Using AI-optimized take profit: ${take_profit:,.8f}")
                
                # Calculate optimized Entry 2 (DCA) if Entry 1 was optimized
                # Keep Entry 2 close to original Entry 2, maintaining relative spacing from Entry 1
                if opt_prices.get('entry_price') and opt_prices['entry_price'] != safe_float(signal_data.get('entry_price'), default=entry_price):
                    original_entry = safe_float(signal_data.get('entry_price'), default=entry_price)
                    original_entry2 = safe_float(signal_data.get('second_entry_price'), default=None)
                    
                    if original_entry2 and original_entry:
                        # Calculate the ORIGINAL spacing between Entry 1 and Entry 2
                        if signal_side == 'LONG':
                            original_spacing = original_entry - original_entry2  # Entry 2 is below Entry 1
                        else:  # SHORT
                            original_spacing = original_entry2 - original_entry  # Entry 2 is above Entry 1
                        
                        # Calculate original spacing as percentage
                        original_spacing_pct = (original_spacing / original_entry) * 100 if original_entry > 0 else 0
                        
                        # Apply the SAME spacing percentage to the optimized Entry 1
                        # This keeps Entry 2 close to its original position relative to Entry 1
                        if signal_side == 'LONG':
                            optimized_entry2 = opt_prices['entry_price'] * (1 - original_spacing_pct / 100)
                        else:  # SHORT
                            optimized_entry2 = opt_prices['entry_price'] * (1 + original_spacing_pct / 100)
                        
                        opt_prices['second_entry_price'] = optimized_entry2
                        logger.info(f"🔄 [PRICE UPDATE] Calculated optimized Entry 2: ${optimized_entry2:,.8f} (maintaining original {original_spacing_pct:.2f}% spacing from optimized Entry 1, original Entry 2 was ${original_entry2:,.8f})")
        
        # Handle EXIT events - close position at market price and cancel all orders for symbol
        if event == 'EXIT':
            logger.info(f"Processing EXIT event for {symbol} (AI validation skipped for EXIT events)")
            
            # Check for duplicate EXIT processing
            current_time = time.time()
            if symbol in recent_exits:
                if current_time - recent_exits[symbol] < EXIT_COOLDOWN:
                    logger.warning(f"EXIT event for {symbol} already processed recently. Discarding duplicate EXIT alert.")
                    return {'success': False, 'error': 'EXIT already processed recently'}
            
            # Detect position mode
            is_hedge_mode = get_position_mode(symbol)
            
            # Cancel ALL orders for this symbol first (even if entry didn't fill)
            logger.info(f"Canceling all orders for {symbol}")
            canceled_count = cancel_all_limit_orders(symbol)
            logger.info(f"Canceled {canceled_count} limit orders for {symbol}")
            
            # Also cancel any TP/SL orders
            try:
                open_orders = client.futures_get_open_orders(symbol=symbol)
                for order in open_orders:
                    if order.get('type') in ['TAKE_PROFIT_MARKET', 'STOP_MARKET']:
                        try:
                            client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
                            logger.info(f"Canceled {order.get('type')} order {order['orderId']} for {symbol}")
                        except Exception as e:
                            logger.warning(f"Failed to cancel {order.get('type')} order: {e}")
            except Exception as e:
                logger.warning(f"Error canceling TP/SL orders: {e}")
            
            # Initialize variables for exit notification
            exit_price = None
            entry_prices = None
            total_pnl = 0.0
            total_pnl_percent = 0.0
            
            # Check for ANY open position (don't filter by signal_side - close all positions)
            # Get ALL positions first, then filter by symbol (more reliable than filtering in API call)
            try:
                all_positions = client.futures_position_information()
                positions_to_close = []
                
                for position in all_positions:
                    position_symbol = position.get('symbol', '')
                    position_amt = float(position.get('positionAmt', 0))
                    
                    # Match symbol (case-insensitive)
                    if position_symbol.upper() == symbol.upper() and abs(position_amt) > 0:
                        positions_to_close.append(position)
                        logger.info(f"Found open position for {symbol}: {position_amt} @ {position.get('entryPrice')} (side: {position.get('positionSide', 'BOTH')})")
                
                # Get current market price for exit notification (always try to get it)
                try:
                    ticker = client.futures_symbol_ticker(symbol=symbol)
                    exit_price = float(ticker.get('price', 0))
                except Exception as e:
                    logger.warning(f"Could not get current price for {symbol}: {e}")
                    exit_price = None
                
                # Get entry prices and trade info from active_trades for notification
                if symbol in active_trades:
                    trade_info = active_trades[symbol]
                    entry_prices = {
                        'entry1': trade_info.get('original_entry1'),
                        'entry2': trade_info.get('original_entry2'),
                        'optimized_entry1': None  # We don't store this separately, but could calculate if needed
                    }
                
                if positions_to_close:
                    # Get symbol info for quantity precision (once, outside loop)
                    exchange_info = client.futures_exchange_info()
                    symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
                    step_size = None
                    if symbol_info:
                        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                        if lot_size_filter:
                            step_size = float(lot_size_filter['stepSize'])
                    
                    # Close ALL positions for this symbol
                    logger.info(f"Closing {len(positions_to_close)} position(s) at market price for {symbol}")
                    total_pnl = 0.0
                    total_pnl_percent = 0.0
                    
                    for position in positions_to_close:
                        position_amt = float(position.get('positionAmt', 0))
                        position_side_binance = position.get('positionSide', 'BOTH')
                        
                        # Determine close side (opposite of position)
                        close_side = 'SELL' if position_amt > 0 else 'BUY'
                        close_quantity = abs(position_amt)
                        
                        # Format quantity precision if step_size available
                        if step_size:
                            close_quantity = format_quantity_precision(close_quantity, step_size)
                        
                        # Create market order to close position
                        # Try with reduceOnly first, then without if it fails (some accounts/modes don't accept it)
                        close_params = {
                            'symbol': symbol,
                            'side': close_side,
                            'type': 'MARKET',
                            'quantity': close_quantity,
                        }
                        
                        # Only include positionSide if in Hedge mode
                        if is_hedge_mode and position_side_binance != 'BOTH':
                            close_params['positionSide'] = position_side_binance
                        
                        # Try with reduceOnly first (for safety)
                        close_params_with_reduce = close_params.copy()
                        close_params_with_reduce['reduceOnly'] = True
                        
                        logger.info(f"Closing position at market for {symbol}: {close_params_with_reduce}")
                        try:
                            result = client.futures_create_order(**close_params_with_reduce)
                            logger.info(f"✅ Position closed successfully: {result}")
                            
                            # Calculate P&L if we have entry price and exit price
                            position_entry_price = float(position.get('entryPrice', 0))
                            if position_entry_price > 0 and exit_price and exit_price > 0:
                                # Calculate P&L based on position side
                                if position_amt > 0:  # LONG position
                                    pnl_percent = ((exit_price - position_entry_price) / position_entry_price) * 100
                                else:  # SHORT position
                                    pnl_percent = ((position_entry_price - exit_price) / position_entry_price) * 100
                                
                                # Calculate P&L in USD (approximate based on position size)
                                position_value = abs(position_amt) * position_entry_price
                                pnl_usd = (pnl_percent / 100) * (position_value / LEVERAGE)  # Divide by leverage to get margin used
                                total_pnl += pnl_usd
                                total_pnl_percent = pnl_percent  # Use last position's percentage
                        except BinanceAPIException as e:
                            # If reduceOnly is not accepted, try without it
                            if e.code == -1106 or 'reduceonly' in str(e).lower():
                                logger.warning(f"reduceOnly not accepted for {symbol}, retrying without it: {e}")
                                try:
                                    result = client.futures_create_order(**close_params)
                                    logger.info(f"✅ Position closed successfully (without reduceOnly): {result}")
                                except Exception as e2:
                                    logger.error(f"❌ Failed to close position (retry without reduceOnly): {e2}", exc_info=True)
                            else:
                                logger.error(f"❌ Failed to close position: {e}", exc_info=True)
                                send_slack_alert(
                                    error_type="Position Close Failed (EXIT)",
                                    message=str(e),
                                    details={'Position_Amount': position_amt, 'Close_Side': close_side},
                                    symbol=symbol,
                                    severity='ERROR'
                                )
                        except Exception as e:
                            logger.error(f"❌ Failed to close position: {e}", exc_info=True)
                            send_slack_alert(
                                error_type="Position Close Error (EXIT)",
                                message=str(e),
                                details={'Position_Amount': position_amt},
                                symbol=symbol,
                                severity='ERROR'
                            )
                else:
                    logger.info(f"No open position found for {symbol} - position may already be closed or entry never filled")
            except Exception as e:
                logger.error(f"Error checking/closing positions for {symbol}: {e}", exc_info=True)
                send_slack_alert(
                    error_type="EXIT Position Close Error",
                    message=str(e),
                    symbol=symbol,
                    severity='ERROR'
                )
            
            # Send exit notification to Slack before cleaning up tracking
            try:
                timeframe = signal_data.get('timeframe', 'Unknown')
                # Determine exit reason
                reason = signal_data.get('exit_reason', 'Manual Exit')
                if not reason or reason == '':
                    reason = 'Manual Exit'
                
                # Send notification with P&L if available
                pnl = total_pnl if total_pnl != 0 else None
                pnl_percent = total_pnl_percent if total_pnl_percent != 0 else None
                
                send_exit_notification(
                    symbol=symbol,
                    signal_side=signal_side,
                    timeframe=timeframe,
                    exit_price=exit_price,
                    entry_prices=entry_prices,
                    pnl=pnl,
                    pnl_percent=pnl_percent,
                    reason=reason
                )
            except Exception as e:
                logger.debug(f"Failed to send exit notification: {e}")
            
            # Clean up tracking
            if symbol in active_trades:
                logger.info(f"Cleaning up closed trade tracking for {symbol}")
                del active_trades[symbol]
            
            # Track EXIT processing
            recent_exits[symbol] = current_time
            
            # Clean up old exit tracking
            if len(recent_exits) > 1000:
                sorted_exits = sorted(recent_exits.items(), key=lambda x: x[1])
                for key, _ in sorted_exits[:-1000]:
                    del recent_exits[key]
            
            # AI Analysis: Find new trading opportunities after exit
            try:
                logger.info(f"🤖 [POST-EXIT AI ANALYSIS] Analyzing {symbol} for new trading opportunities...")
                timeframe = signal_data.get('timeframe', '1H')
                
                # Get current market price first
                try:
                    ticker = client.futures_symbol_ticker(symbol=symbol)
                    current_price = float(ticker.get('price', 0))
                except Exception as e:
                    logger.warning(f"Could not get current price for {symbol}: {e}")
                    current_price = None
                
                opportunity = analyze_symbol_for_opportunities(symbol, timeframe, current_price=current_price)
                
                if opportunity.get('opportunity_found') and opportunity.get('confidence_score', 0) >= 90:
                    opp_side = opportunity.get('signal_side')
                    opp_entry = opportunity.get('entry_price')
                    opp_sl = opportunity.get('stop_loss')
                    opp_tp = opportunity.get('take_profit')
                    opp_confidence = opportunity.get('confidence_score', 0)
                    opp_reasoning = opportunity.get('reasoning', '')
                    
                    # Validate entry price is close to current price (within 2% for immediate execution)
                    if current_price and current_price > 0 and opp_entry:
                        entry_distance_pct = abs((opp_entry - current_price) / current_price) * 100
                        MAX_ENTRY_DISTANCE_PCT = 2.0  # Maximum 2% away from current price
                        
                        if entry_distance_pct > MAX_ENTRY_DISTANCE_PCT:
                            logger.warning(f"⚠️ [POST-EXIT AI ANALYSIS] Entry price ${opp_entry:,.8f} is {entry_distance_pct:.2f}% away from current price ${current_price:,.8f} (max: {MAX_ENTRY_DISTANCE_PCT}%)")
                            logger.warning(f"   Adjusting entry to current price and recalculating SL/TP")
                            
                            # Adjust entry to current price and recalculate SL/TP based on percentages
                            old_entry = opp_entry
                            opp_entry = current_price
                            
                            # Recalculate SL and TP based on original percentages
                            if opp_sl and old_entry > 0:
                                # Calculate original SL percentage
                                if opp_side == 'LONG':
                                    sl_percent = ((old_entry - opp_sl) / old_entry) * 100
                                    opp_sl = opp_entry * (1 - sl_percent / 100)
                                else:  # SHORT
                                    sl_percent = ((opp_sl - old_entry) / old_entry) * 100
                                    opp_sl = opp_entry * (1 + sl_percent / 100)
                            
                            # Recalculate TP to 3.5% from new entry (ensuring it's in 2-5% range)
                            target_tp_percent = 3.5
                            if opp_side == 'LONG':
                                opp_tp = opp_entry * (1 + target_tp_percent / 100)
                            else:  # SHORT
                                opp_tp = opp_entry * (1 - target_tp_percent / 100)
                            
                            logger.info(f"🔄 [POST-EXIT AI ANALYSIS] Adjusted entry from ${old_entry:,.8f} to ${opp_entry:,.8f} (current price)")
                            logger.info(f"   New SL: ${opp_sl:,.8f}, New TP: ${opp_tp:,.8f}")
                        else:
                            logger.info(f"✅ [POST-EXIT AI ANALYSIS] Entry price ${opp_entry:,.8f} is {entry_distance_pct:.2f}% from current price ${current_price:,.8f} (acceptable)")
                    
                    # Ensure TP is in 2-5% range (more achievable)
                    # If AI didn't provide TP, calculate it as 3.5% from entry
                    if opp_entry and not opp_tp:
                        # Calculate TP as 3.5% from entry (middle of 2-5% range)
                        target_tp_percent = 3.5
                        if opp_side == 'LONG':
                            opp_tp = opp_entry * (1 + target_tp_percent / 100)
                        else:  # SHORT
                            opp_tp = opp_entry * (1 - target_tp_percent / 100)
                        logger.info(f"📊 [POST-EXIT AI ANALYSIS] Calculated TP as {target_tp_percent}% from entry (AI didn't provide TP)")
                    elif opp_entry and opp_tp:
                        # Calculate current TP percentage
                        if opp_side == 'LONG':
                            tp_percent = ((opp_tp - opp_entry) / opp_entry) * 100
                        else:  # SHORT
                            tp_percent = ((opp_entry - opp_tp) / opp_entry) * 100
                        
                        # If TP is outside 2-5% range, adjust it
                        if tp_percent < 2.0 or tp_percent > 5.0:
                            # Use 3.5% as default (middle of 2-5% range)
                            target_tp_percent = 3.5
                            if opp_side == 'LONG':
                                opp_tp = opp_entry * (1 + target_tp_percent / 100)
                            else:  # SHORT
                                opp_tp = opp_entry * (1 - target_tp_percent / 100)
                            logger.info(f"🔄 [POST-EXIT AI ANALYSIS] Adjusted TP to {target_tp_percent}% (was {tp_percent:.2f}%) for better achievability")
                        else:
                            logger.info(f"✅ [POST-EXIT AI ANALYSIS] TP is {tp_percent:.2f}% (within 2-5% range)")
                    
                    logger.info(f"✅ [POST-EXIT AI ANALYSIS] Found {opp_side} opportunity for {symbol}")
                    logger.info(f"   Entry: ${opp_entry:,.8f}, SL: ${opp_sl:,.8f}, TP: ${opp_tp:,.8f}")
                    logger.info(f"   Confidence: {opp_confidence:.1f}%")
                    logger.info(f"   Reasoning: {opp_reasoning[:200]}...")
                    
                    # Create new signal data for the opportunity
                    if opp_entry and opp_sl and opp_tp:
                        new_signal_data = {
                            'token': signal_data.get('token'),  # Use same token
                            'event': 'ENTRY',
                            'signal_side': opp_side,
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'entry_price': opp_entry,
                            'stop_loss': opp_sl,
                            'take_profit': opp_tp,
                            'second_entry_price': None,  # Single entry only
                            '_post_exit_ai_trade': True,  # Flag to indicate this is from post-exit AI
                            '_ai_confidence': opp_confidence,
                            '_ai_reasoning': opp_reasoning,
                            '_entry_size_usd': 10.0  # Use $10 entry size as requested
                        }
                        
                        # Create the new trade with $10 entry size
                        logger.info(f"💰 [POST-EXIT AI TRADE] Creating new {opp_side} trade for {symbol} with $10 entry size")
                        try:
                            # Call create_limit_order with the new signal data
                            # This will use the $10 entry size from _entry_size_usd
                            trade_result = create_limit_order(new_signal_data)
                            
                            if trade_result.get('success'):
                                logger.info(f"✅ [POST-EXIT AI TRADE] Successfully created {opp_side} trade for {symbol}")
                                send_slack_alert(
                                    error_type="Post-Exit AI Trade Created",
                                    message=f"Created {opp_side} trade for {symbol} after exit (AI confidence: {opp_confidence:.1f}%)",
                                    details={
                                        'Symbol': symbol,
                                        'Side': opp_side,
                                        'Entry': f"${opp_entry:,.8f}",
                                        'SL': f"${opp_sl:,.8f}",
                                        'TP': f"${opp_tp:,.8f}",
                                        'Confidence': f"{opp_confidence:.1f}%",
                                        'Entry_Size': "$10"
                                    },
                                    symbol=symbol,
                                    severity='INFO'
                                )
                            else:
                                logger.warning(f"⚠️ [POST-EXIT AI TRADE] Failed to create trade: {trade_result.get('error', 'Unknown error')}")
                        except Exception as e:
                            logger.error(f"❌ [POST-EXIT AI TRADE] Error creating trade for {symbol}: {e}", exc_info=True)
                    else:
                        logger.warning(f"⚠️ [POST-EXIT AI ANALYSIS] Opportunity found but missing required prices (entry: {opp_entry}, sl: {opp_sl}, tp: {opp_tp})")
                else:
                    logger.info(f"❌ [POST-EXIT AI ANALYSIS] No high-confidence opportunity found for {symbol} (confidence: {opportunity.get('confidence_score', 0):.1f}%)")
            except Exception as e:
                logger.error(f"❌ [POST-EXIT AI ANALYSIS] Error analyzing {symbol} for opportunities: {e}", exc_info=True)
                # Don't fail the exit if AI analysis fails
            
            return {'success': True, 'message': 'EXIT event processed'}
        
        # Only process ENTRY events
        if event != 'ENTRY':
            logger.info(f"Ignoring event: {event}")
            return {'success': False, 'error': f'Event {event} not processed'}
        
        # Get order side
        side = get_order_side(signal_side)
        position_side = get_position_side(signal_side)
        
        # Detect position mode (One-Way vs Hedge)
        is_hedge_mode = get_position_mode(symbol)
        
        # Determine which entry this is
        is_primary_entry = (order_subtype == 'primary_entry' or order_subtype not in ['dca_fill', 'second_entry'])
        is_dca_entry = (order_subtype == 'dca_fill' or order_subtype == 'second_entry')
        
        # Check for existing position and orders BEFORE processing
        has_orders, open_orders = check_existing_orders(symbol)
        has_position, position_info = check_existing_position(symbol, signal_side)
        
        # IMPORTANT: Prevent duplicate alerts when position is already open and both orders are filled
        # This handles the case where TradingView sends duplicate DCA alerts when price returns to DCA level
        if has_position:
            # Simple check: If position exists and there are no pending LIMIT orders (only TP/SL orders may exist),
            # then both entry orders are already filled - this is a duplicate alert
            pending_limit_orders = [o for o in open_orders if o.get('type') == 'LIMIT']
            
            if len(pending_limit_orders) == 0:
                # No pending limit orders = both entry orders are filled
                logger.warning(f"⚠️ Duplicate alert ignored: Position already open for {symbol} and no pending limit orders found. Both entry orders are already filled. This is likely a duplicate DCA alert from TradingView.")
                return {
                    'success': False, 
                    'error': 'Duplicate alert ignored - position already open with both orders filled',
                    'message': f'Position exists for {symbol} and both primary/DCA orders are already filled (no pending limit orders). Ignoring duplicate alert.'
                }
            
            # Additional check: Verify by checking order status in active_trades
            # This is a secondary verification method
            if symbol in active_trades:
                trade_info = active_trades[symbol]
                primary_order_id = trade_info.get('primary_order_id')
                dca_order_id = trade_info.get('dca_order_id')
                optimized_entry1_order_id = trade_info.get('optimized_entry1_order_id')
                
                # Check if all orders exist and are filled (at least Order 1 and Order 3, Order 2 is optional)
                if primary_order_id and dca_order_id:
                    try:
                        primary_order = client.futures_get_order(symbol=symbol, orderId=primary_order_id)
                        dca_order = client.futures_get_order(symbol=symbol, orderId=dca_order_id)
                        
                        # Check Order 2 if it exists
                        order2_filled = True  # Default to True if Order 2 doesn't exist
                        if optimized_entry1_order_id:
                            try:
                                order2 = client.futures_get_order(symbol=symbol, orderId=optimized_entry1_order_id)
                                order2_filled = order2.get('status') == 'FILLED'
                            except:
                                order2_filled = True  # If we can't check, assume it's fine
                        
                        if (primary_order.get('status') == 'FILLED' and 
                            dca_order.get('status') == 'FILLED' and
                            order2_filled):
                            logger.warning(f"⚠️ Duplicate alert ignored: All orders confirmed as FILLED on Binance for {symbol}. Ignoring duplicate alert.")
                            return {
                                'success': False, 
                                'error': 'Duplicate alert ignored - all orders already filled',
                                'message': f'All orders (Order 1, Order 2, Order 3) are already FILLED for {symbol}. Ignoring duplicate alert.'
                            }
                    except Exception as e:
                        logger.debug(f"Could not verify order status for duplicate check: {e}")
                        # Continue processing if we can't verify (fallback to pending orders check above)
        
        # IMPORTANT: When new trade alert comes for any symbol, close old orders for that symbol first
        # (Only if position doesn't exist or orders aren't filled)
        logger.info(f"Checking for existing orders/positions for {symbol} - will close old ones if found")
        
        # If position exists but orders aren't both filled, close it (new trade replaces old trade)
        if has_position:
            logger.info(f"Existing position found for {symbol}. Closing it before creating new trade.")
            close_result = close_position_at_market(symbol, signal_side, is_hedge_mode)
            if close_result.get('success'):
                logger.info(f"Old position closed successfully")
            else:
                logger.warning(f"Failed to close old position: {close_result.get('error')}")
        
        # Cancel all existing orders for this symbol (old trade orders)
        if has_orders:
            logger.info(f"Found {len(open_orders)} existing orders for {symbol}. Canceling all orders.")
            canceled_count = cancel_all_limit_orders(symbol)
            # Also cancel TP/SL orders
            for order in open_orders:
                if order.get('type') in ['TAKE_PROFIT_MARKET', 'STOP_MARKET']:
                    try:
                        client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
                        logger.info(f"Canceled {order.get('type')} order {order['orderId']}")
                    except Exception as e:
                        logger.warning(f"Failed to cancel {order.get('type')} order: {e}")
            logger.info(f"Canceled {canceled_count} limit orders for {symbol}")
            
            # Wait a moment for Binance to process cancellations
            time.sleep(0.5)
            
            # Re-check orders to ensure they're canceled (prevent false duplicate detection)
            has_orders_after_cancel, _ = check_existing_orders(symbol)
            if has_orders_after_cancel:
                logger.warning(f"Some orders still exist after cancellation. Waiting longer...")
                time.sleep(1.0)
                # Try canceling again
                cancel_all_limit_orders(symbol)
        
        # Clean up old tracking (already done above, but ensure it's clean)
        if symbol in active_trades:
            logger.info(f"Cleaning up old trade tracking for {symbol}")
            del active_trades[symbol]
        
        # Store ORIGINAL Entry 1 price BEFORE optimization (for Order 1)
        original_entry1_price = safe_float(signal_data.get('entry_price'), default=entry_price)
        
        # Get optimized Entry 1 price (if AI optimized it)
        # CRITICAL: Only use optimized price if it's BETTER than original:
        # - For LONG: Optimized price must be LOWER than original (better entry)
        # - For SHORT: Optimized price must be HIGHER than original (better entry)
        optimized_entry1_price = None
        if 'optimized_prices' in validation_result and validation_result.get('optimized_prices', {}).get('entry_price'):
            opt_entry1 = validation_result['optimized_prices']['entry_price']
            # Check if optimized price is actually better than original
            if signal_side == 'LONG':
                # For LONG: Optimized price must be LOWER (better entry closer to support)
                if opt_entry1 < original_entry1_price:
                    optimized_entry1_price = opt_entry1
                    logger.info(f"🔄 [PRICE UPDATE] AI optimized Entry 1 for LONG: ${original_entry1_price:,.8f} → ${optimized_entry1_price:,.8f} (better entry - lower)")
                else:
                    logger.info(f"⚠️  [PRICE UPDATE] AI suggested Entry 1 ${opt_entry1:,.8f} is NOT LOWER than original ${original_entry1_price:,.8f} - skipping Order 2 (not better for LONG)")
            else:  # SHORT
                # For SHORT: Optimized price must be HIGHER (better entry closer to resistance)
                if opt_entry1 > original_entry1_price:
                    optimized_entry1_price = opt_entry1
                    logger.info(f"🔄 [PRICE UPDATE] AI optimized Entry 1 for SHORT: ${original_entry1_price:,.8f} → ${optimized_entry1_price:,.8f} (better entry - higher)")
                else:
                    logger.info(f"⚠️  [PRICE UPDATE] AI suggested Entry 1 ${opt_entry1:,.8f} is NOT HIGHER than original ${original_entry1_price:,.8f} - skipping Order 2 (not better for SHORT)")
        
        # Get primary entry price - use optimized entry_price if available, otherwise original
        primary_entry_price = entry_price
        
        # Get DCA entry price (second entry) - prioritize AI-suggested Entry 2, then recalculated Entry 2, then original
        # Priority: 1) AI-suggested Entry 2, 2) Recalculated Entry 2 (if Entry 1 was optimized), 3) Original Entry 2
        if 'optimized_prices' in validation_result and validation_result.get('optimized_prices', {}).get('second_entry_price'):
            dca_entry_price = validation_result['optimized_prices']['second_entry_price']
            # Check if this is AI-suggested (from AI validation) or recalculated (from Entry 1 optimization)
            if 'suggested_second_entry_price' in validation_result.get('price_suggestions', {}):
                logger.info(f"🔄 [PRICE UPDATE] Using AI-suggested Entry 2 (DCA): ${dca_entry_price:,.8f}")
            else:
                logger.info(f"🔄 [PRICE UPDATE] Using recalculated Entry 2 (DCA): ${dca_entry_price:,.8f} (based on Entry 1 optimization)")
        else:
            dca_entry_price = second_entry_price if second_entry_price and second_entry_price > 0 else None
            if dca_entry_price:
                logger.info(f"ℹ️  [PRICE] Using original Entry 2 (DCA): ${dca_entry_price:,.8f}")
        
        # If this is a primary entry, we need both prices to create both orders
        if is_primary_entry and not dca_entry_price:
            logger.warning(f"Primary entry signal received but no second_entry_price provided. Using entry_price for both.")
            dca_entry_price = primary_entry_price  # Fallback to same price if not provided
        
        # If this is a DCA entry, use the second_entry_price from JSON
        if is_dca_entry:
            if dca_entry_price and dca_entry_price > 0:
                entry_price = dca_entry_price
                logger.info(f"Using DCA entry price from second_entry_price: {entry_price}")
            else:
                # Use entry_price as fallback for DCA
                dca_entry_price = entry_price
                logger.info(f"Using entry_price as DCA entry price: {entry_price}")
        
        # Validate entry price (should already be validated above, but double-check)
        if entry_price is None or entry_price <= 0:
            logger.warning(f"Invalid entry price after parsing: {entry_price}. Discarding request.")
            return {'success': False, 'error': 'Invalid entry price (NA/null or <= 0)'}
        
        # Get symbol info for precision
        exchange_info = client.futures_exchange_info()
        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
        
        if not symbol_info:
            return {'success': False, 'error': f'Symbol {symbol} not found'}
        
        # Set leverage
        try:
            client.futures_change_leverage(symbol=symbol, leverage=LEVERAGE)
            logger.info(f"Set leverage to {LEVERAGE}X for {symbol}")
        except Exception as e:
            logger.warning(f"Could not set leverage (may already be set): {e}")
        
        # Get price precision
        price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
        tick_size = float(price_filter['tickSize']) if price_filter else 0.01
        
        # Format prices to tick size precision (removes floating point errors)
        original_entry1_price = format_price_precision(original_entry1_price, tick_size)
        primary_entry_price = format_price_precision(primary_entry_price, tick_size)
        if optimized_entry1_price:
            optimized_entry1_price = format_price_precision(optimized_entry1_price, tick_size)
        if dca_entry_price:
            dca_entry_price = format_price_precision(dca_entry_price, tick_size)
        
        # Get custom entry size from signal_data if present (for post-exit AI trades)
        custom_entry_size = safe_float(signal_data.get('_entry_size_usd'), default=None)
        
        # Calculate quantities for 3 orders:
        # Order 1: Custom size or $10 with original Entry 1
        # Order 2: Custom size/2 or $5 with optimized Entry 1 (if exists)
        # Order 3: Custom size or $10 with Entry 2 (original or optimized)
        entry1_size = custom_entry_size if custom_entry_size else 10.0
        entry2_size = (custom_entry_size / 2.0) if custom_entry_size else 5.0
        entry3_size = custom_entry_size if custom_entry_size else 10.0
        
        order1_quantity = calculate_quantity(original_entry1_price, symbol_info, entry_size_usd=entry1_size)
        order2_quantity = calculate_quantity(optimized_entry1_price, symbol_info, entry_size_usd=entry2_size) if optimized_entry1_price else None
        order3_quantity = calculate_quantity(dca_entry_price, symbol_info, entry_size_usd=entry3_size) if dca_entry_price else None
        
        # Legacy variables for backward compatibility (used in TP calculations)
        primary_quantity = order1_quantity
        dca_quantity = order3_quantity if order3_quantity else order1_quantity
        
        # Risk Validation: Check if trade risk is within acceptable limits
        # This includes risk from pending orders (orders waiting to fill)
        # Can be disabled via ENABLE_RISK_VALIDATION=false if needed
        if ENABLE_RISK_VALIDATION and stop_loss and stop_loss > 0:
            is_valid_risk, risk_info, risk_error = validate_risk_per_trade(
                symbol, primary_entry_price, stop_loss, primary_quantity, signal_side
            )
            
            if not is_valid_risk:
                logger.error(f"❌ Risk validation FAILED for {symbol}: {risk_error}")
                send_slack_alert(
                    error_type="Risk Validation Failed",
                    message=risk_error,
                    details={
                        'Risk_Percent': f"{risk_info['risk_percent']:.2f}%",
                        'Max_Risk_Percent': f"{MAX_RISK_PERCENT}%",
                        'Current_Trade_Risk': f"${risk_info['current_trade_risk']:.2f}",
                        'Pending_Orders_Risk': f"${risk_info['pending_orders_risk']:.2f}",
                        'Total_Risk': f"${risk_info['total_risk']:.2f}",
                        'Account_Balance': f"${risk_info['account_balance']:.2f}",
                        'Pending_Orders_Count': len(risk_info['pending_orders'])
                    },
                    symbol=symbol,
                    severity='WARNING'
                )
                return {
                    'success': False,
                    'error': f'Risk validation failed: {risk_error}',
                    'risk_info': risk_info
                }
            
            # Log risk info if available
            if risk_info:
                logger.info(f"💰 Risk Validation: {risk_info['risk_percent']:.2f}% of account "
                          f"(${risk_info['total_risk']:.2f} total risk: ${risk_info['current_trade_risk']:.2f} current + "
                          f"${risk_info['pending_orders_risk']:.2f} pending from {len(risk_info['pending_orders'])} orders)")
        elif not ENABLE_RISK_VALIDATION:
            logger.debug(f"Risk validation disabled - skipping risk check for {symbol}")
        
        # Initialize active trades tracking
        if symbol not in active_trades:
            active_trades[symbol] = {
                'primary_filled': False, 
                'dca_filled': False, 
                'optimized_entry1_filled': False,  # Track Order 2 (optimized Entry 1)
                'position_open': False,
                'primary_order_id': None,
                'dca_order_id': None,
                'optimized_entry1_order_id': None,  # Order 2 ID
                'tp_order_id': None,
                'sl_order_id': None,
                'original_stop_loss': None,
                'original_entry1': None,
                'original_entry2': None,
                'current_sl_price': None,
                'sl_moved_to_breakeven': False,
                'tp1_filled': False
            }
        
        # Store original prices for trailing stop loss
        active_trades[symbol]['original_stop_loss'] = stop_loss
        active_trades[symbol]['original_entry1'] = original_entry1_price  # Store original Entry 1
        active_trades[symbol]['original_entry2'] = dca_entry_price if dca_entry_price else None
        
        current_time = time.time()
        order_results = []
        
        # SPECIAL CASE: Post-exit AI trade (single entry with $10 size)
        if signal_data.get('_post_exit_ai_trade', False):
            ai_entry_price = format_price_precision(entry_price, tick_size)
            ai_entry_size = safe_float(signal_data.get('_entry_size_usd'), default=10.0)
            ai_quantity = calculate_quantity(ai_entry_price, symbol_info, entry_size_usd=ai_entry_size)
            
            logger.info(f"🤖 [POST-EXIT AI TRADE] Creating single entry order for {symbol}")
            logger.info(f"   Entry Price: ${ai_entry_price:,.8f}")
            logger.info(f"   Order Size: ${ai_entry_size}")
            logger.info(f"   Side: {signal_side}")
            
            # Create single entry order
            ai_order_params = {
                'symbol': symbol,
                'side': side,
                'type': 'LIMIT',
                'timeInForce': 'GTC',
                'quantity': ai_quantity,
                'price': ai_entry_price,
            }
            if is_hedge_mode:
                ai_order_params['positionSide'] = position_side
            
            logger.info(f"Creating POST-EXIT AI order: {ai_order_params}")
            try:
                ai_order_result = client.futures_create_order(**ai_order_params)
                order_results.append(ai_order_result)
                active_trades[symbol]['primary_order_id'] = ai_order_result.get('orderId')
                active_trades[symbol]['primary_filled'] = False
                active_trades[symbol]['dca_filled'] = False
                active_trades[symbol]['optimized_entry1_filled'] = False
                active_trades[symbol]['position_open'] = True
                active_trades[symbol]['dca_order_id'] = None
                active_trades[symbol]['optimized_entry1_order_id'] = None
                active_trades[symbol]['original_entry1'] = ai_entry_price
                active_trades[symbol]['original_entry2'] = None
                active_trades[symbol]['_post_exit_ai_trade'] = True
                active_trades[symbol]['_ai_confidence'] = safe_float(signal_data.get('_ai_confidence'), default=0.0)
                active_trades[symbol]['_ai_reasoning'] = signal_data.get('_ai_reasoning', '')
                
                # Store TP/SL for TP creation
                if stop_loss:
                    active_trades[symbol]['tp2_price'] = format_price_precision(stop_loss, tick_size)  # Use SL as TP trigger
                if take_profit:
                    active_trades[symbol]['tp1_price'] = format_price_precision(take_profit, tick_size)
                    active_trades[symbol]['use_single_tp'] = True
                    active_trades[symbol]['tp2_price'] = format_price_precision(take_profit, tick_size)
                    active_trades[symbol]['tp_side'] = 'SELL' if signal_side == 'LONG' else 'BUY'
                    active_trades[symbol]['tp_quantity'] = ai_quantity
                    active_trades[symbol]['tp_working_type'] = 'MARK_PRICE'
                
                logger.info(f"✅ POST-EXIT AI order created successfully: Order ID {ai_order_result.get('orderId')} @ ${ai_entry_price:,.8f} (${ai_entry_size} size)")
                
                # Send notification
                ai_confidence = active_trades[symbol]['_ai_confidence']
                send_signal_notification(
                    symbol=symbol,
                    signal_side=signal_side,
                    timeframe=timeframe,
                    confidence_score=ai_confidence,
                    risk_level='LOW' if ai_confidence >= 95 else 'MEDIUM' if ai_confidence >= 90 else 'HIGH',
                    entry1_price=ai_entry_price,
                    entry2_price=None,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    tp1_price=take_profit,
                    use_single_tp=True,
                    optimized_entry1_price=None
                )
                
                return {
                    'success': True,
                    'message': f'POST-EXIT AI trade created: {signal_side} {symbol} @ ${ai_entry_price:,.8f}',
                    'order_id': ai_order_result.get('orderId'),
                    'orders': order_results
                }
            except BinanceAPIException as e:
                logger.error(f"❌ Failed to create POST-EXIT AI order: {e}", exc_info=True)
                send_slack_alert(
                    error_type="POST-EXIT AI Order Creation Failed",
                    message=str(e),
                    details={'Symbol': symbol, 'Side': signal_side, 'Price': ai_entry_price},
                    symbol=symbol,
                    severity='ERROR'
                )
                return {'success': False, 'error': f'Failed to create POST-EXIT AI order: {str(e)}'}
            except Exception as e:
                logger.error(f"❌ Unexpected error creating POST-EXIT AI order: {e}", exc_info=True)
                return {'success': False, 'error': f'Unexpected error: {str(e)}'}
        
        # SPECIAL CASE: Entry 2 only (Entry 1 is bad but Entry 2 is good)
        if signal_data.get('_special_entry2_only', False):
            entry2_only_price = signal_data.get('_entry2_only_price')
            if not entry2_only_price or entry2_only_price <= 0:
                logger.error(f"❌ Special Entry 2 only case but no valid Entry 2 price provided")
                return {'success': False, 'error': 'Special Entry 2 only case but no valid Entry 2 price'}
            
            # Format Entry 2 price
            entry2_only_price = format_price_precision(entry2_only_price, tick_size)
            
            # Calculate quantity for $20 order
            entry2_quantity = calculate_quantity(entry2_only_price, symbol_info, entry_size_usd=20.0)
            
            # Calculate custom TP: 4-5% from entry (use 4.5% as default)
            tp_percentage = 4.5  # 4.5% default, can be adjusted
            if signal_side == 'LONG':
                custom_tp = entry2_only_price * (1 + tp_percentage / 100)
            else:  # SHORT
                custom_tp = entry2_only_price * (1 - tp_percentage / 100)
            custom_tp = format_price_precision(custom_tp, tick_size)
            
            logger.info(f"🎯 SPECIAL CASE: Creating Entry 2 only order for {symbol}")
            logger.info(f"   Entry 2 Price: ${entry2_only_price:,.8f}")
            logger.info(f"   Order Size: $20")
            logger.info(f"   Custom TP: ${custom_tp:,.8f} ({tp_percentage}% from entry)")
            
            # Create Entry 2 only order
            entry2_order_params = {
                'symbol': symbol,
                'side': side,
                'type': 'LIMIT',
                'timeInForce': 'GTC',
                'quantity': entry2_quantity,
                'price': entry2_only_price,
            }
            if is_hedge_mode:
                entry2_order_params['positionSide'] = position_side
            
            logger.info(f"Creating SPECIAL Entry 2 only order: {entry2_order_params}")
            try:
                entry2_order_result = client.futures_create_order(**entry2_order_params)
                order_results.append(entry2_order_result)
                active_trades[symbol]['dca_order_id'] = entry2_order_result.get('orderId')
                active_trades[symbol]['dca_filled'] = False
                active_trades[symbol]['position_open'] = True
                active_trades[symbol]['primary_order_id'] = None  # No Order 1
                active_trades[symbol]['optimized_entry1_order_id'] = None  # No Order 2
                active_trades[symbol]['original_entry1'] = None  # No Entry 1
                active_trades[symbol]['original_entry2'] = entry2_only_price
                active_trades[symbol]['_special_entry2_only'] = True
                active_trades[symbol]['_custom_tp'] = custom_tp
                active_trades[symbol]['_custom_tp_percentage'] = tp_percentage
                # Store TP price for TP creation (use single TP mode for Entry 2 only)
                active_trades[symbol]['use_single_tp'] = True
                active_trades[symbol]['tp2_price'] = custom_tp
                active_trades[symbol]['tp_side'] = 'SELL' if signal_side == 'LONG' else 'BUY'
                active_trades[symbol]['tp_quantity'] = entry2_quantity  # Store quantity for TP
                active_trades[symbol]['tp_working_type'] = 'MARK_PRICE'  # Use mark price for trigger
                logger.info(f"✅ SPECIAL Entry 2 only order created successfully: Order ID {entry2_order_result.get('orderId')} @ ${entry2_only_price:,.8f} (${20.0} size)")
            except BinanceAPIException as e:
                logger.error(f"❌ Failed to create SPECIAL Entry 2 only order: {e.message} (Code: {e.code})")
                send_slack_alert(
                    error_type="Special Entry 2 Only Order Creation Failed",
                    message=f"{e.message} (Code: {e.code})",
                    details={'Error_Code': e.code, 'Entry_Price': entry2_only_price, 'Quantity': entry2_quantity, 'Side': side},
                    symbol=symbol,
                    severity='ERROR'
                )
                return {'success': False, 'error': f'Failed to create Entry 2 only order: {e.message}'}
            except Exception as e:
                logger.error(f"❌ Unexpected error creating SPECIAL Entry 2 only order: {e}")
                send_slack_alert(
                    error_type="Special Entry 2 Only Order Creation Failed",
                    message=str(e),
                    details={'Entry_Price': entry2_only_price, 'Quantity': entry2_quantity},
                    symbol=symbol,
                    severity='ERROR'
                )
                return {'success': False, 'error': f'Unexpected error creating Entry 2 only order: {str(e)}'}
            
            # Send special notification for Entry 2 only case
            try:
                # Use custom notification format for Entry 2 only case
                if SLACK_SIGNAL_WEBHOOK_URL:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
                    environment = 'TESTNET' if BINANCE_TESTNET else 'PRODUCTION'
                    side_emoji = '📈' if signal_side == 'LONG' else '📉'
                    formatted_symbol = symbol.replace('.P', '').upper()
                    formatted_timeframe = timeframe.upper() if timeframe else 'N/A'
                    
                    slack_message = f"""{side_emoji} *SPECIAL ENTRY 2 ONLY SIGNAL - ORDER OPENED*

*Symbol:* `{formatted_symbol}`
*Timeframe:* `{formatted_timeframe}`
*Environment:* {environment}
*Time:* {timestamp}

🎯 *SPECIAL CASE: Entry 1 Rejected, Entry 2 Only*

*Entry Order:*
  • Entry 2 Only: ${entry2_only_price:,.8f} - $20.00 (Entry 1 skipped - not optimal)

*Risk Management:*
  • Stop Loss: {f'${stop_loss:,.8f}' if stop_loss else 'N/A'}
  • Take Profit: ${custom_tp:,.8f} ({tp_percentage}% from entry - Custom TP)

*Reason:*
{validation_result.get('special_case_reason', 'Entry 1 rejected but Entry 2 is optimal')}

*AI Analysis:*
{validation_result.get('reasoning', 'No detailed reasoning')[:600]}"""
                    
                    def send_async():
                        try:
                            payload = {'text': slack_message}
                            response = requests.post(
                                SLACK_SIGNAL_WEBHOOK_URL,
                                json=payload,
                                headers={'Content-Type': 'application/json'},
                                timeout=5
                            )
                            response.raise_for_status()
                            logger.info(f"✅ Special Entry 2 only notification sent to Slack for {symbol}")
                        except Exception as e:
                            logger.debug(f"Failed to send Slack special Entry 2 only notification: {e}")
                    
                    thread = threading.Thread(target=send_async, daemon=True)
                    thread.start()
            except Exception as e:
                logger.warning(f"Failed to send special Entry 2 only notification: {e}")
            
            return {
                'success': True,
                'message': 'Special Entry 2 only order created successfully',
                'order_id': entry2_order_result.get('orderId'),
                'entry_price': entry2_only_price,
                'custom_tp': custom_tp,
                'special_case': 'ENTRY2_ONLY'
            }
        
        # If this is a primary entry, create 3 entry orders:
        # Order 1: $10 with original Entry 1 price
        # Order 2: $5 with optimized Entry 1 price (if AI optimized, otherwise skip)
        # Order 3: $10 with Entry 2 price (original or optimized)
        if is_primary_entry:
            # ORDER 1: $10 with original Entry 1 price
            order1_params = {
                'symbol': symbol,
                'side': side,
                'type': 'LIMIT',
                'timeInForce': 'GTC',
                'quantity': order1_quantity,
                'price': original_entry1_price,
            }
            if is_hedge_mode:
                order1_params['positionSide'] = position_side
            
            logger.info(f"Creating ORDER 1 (Original Entry 1, $10): {order1_params}")
            try:
                order1_result = client.futures_create_order(**order1_params)
                order_results.append(order1_result)
                active_trades[symbol]['primary_order_id'] = order1_result.get('orderId')
                active_trades[symbol]['primary_filled'] = False
                active_trades[symbol]['position_open'] = True
                logger.info(f"✅ ORDER 1 created successfully: Order ID {order1_result.get('orderId')} @ ${original_entry1_price:,.8f} (${10.0} size)")
            except BinanceAPIException as e:
                logger.error(f"❌ Failed to create ORDER 1: {e.message} (Code: {e.code})")
                send_slack_alert(
                    error_type="Order 1 Creation Failed",
                    message=f"{e.message} (Code: {e.code})",
                    details={'Error_Code': e.code, 'Entry_Price': original_entry1_price, 'Quantity': order1_quantity, 'Side': side},
                    symbol=symbol,
                    severity='ERROR'
                )
                return {'success': False, 'error': f'Failed to create Order 1: {e.message}'}
            except Exception as e:
                logger.error(f"❌ Unexpected error creating ORDER 1: {e}")
                send_slack_alert(
                    error_type="Order 1 Creation Error",
                    message=str(e),
                    details={'Entry_Price': original_entry1_price, 'Quantity': order1_quantity, 'Side': side},
                    symbol=symbol,
                    severity='ERROR'
                )
                return {'success': False, 'error': f'Unexpected error: {str(e)}'}
            
            # Track Order 1
            order_key = f"{symbol}_{original_entry1_price}_{side}_ORDER1"
            recent_orders[order_key] = current_time
            
            # ORDER 2: $5 with optimized Entry 1 price (only if AI optimized Entry 1 AND it's actually better)
            # Double-check: Optimized price must be different from original AND better (lower for LONG, higher for SHORT)
            should_create_order2 = False
            if optimized_entry1_price and order2_quantity:
                # Verify optimized price is actually different and better
                price_diff = abs(optimized_entry1_price - original_entry1_price)
                min_price_diff = original_entry1_price * 0.001  # At least 0.1% difference to avoid floating point issues
                
                if signal_side == 'LONG':
                    # For LONG: Optimized must be LOWER (better entry)
                    if optimized_entry1_price < original_entry1_price and price_diff >= min_price_diff:
                        should_create_order2 = True
                    else:
                        logger.warning(f"⚠️  ORDER 2 SKIPPED: Optimized Entry 1 ${optimized_entry1_price:,.8f} is NOT LOWER than original ${original_entry1_price:,.8f} (or difference too small: {price_diff:.8f} < {min_price_diff:.8f})")
                else:  # SHORT
                    # For SHORT: Optimized must be HIGHER (better entry)
                    if optimized_entry1_price > original_entry1_price and price_diff >= min_price_diff:
                        should_create_order2 = True
                    else:
                        logger.warning(f"⚠️  ORDER 2 SKIPPED: Optimized Entry 1 ${optimized_entry1_price:,.8f} is NOT HIGHER than original ${original_entry1_price:,.8f} (or difference too small: {price_diff:.8f} < {min_price_diff:.8f})")
            
            if should_create_order2:
                order2_params = {
                    'symbol': symbol,
                    'side': side,
                    'type': 'LIMIT',
                    'timeInForce': 'GTC',
                    'quantity': order2_quantity,
                    'price': optimized_entry1_price,
                }
                if is_hedge_mode:
                    order2_params['positionSide'] = position_side
                
                logger.info(f"Creating ORDER 2 (Optimized Entry 1, $5): {order2_params}")
                try:
                    order2_result = client.futures_create_order(**order2_params)
                    order_results.append(order2_result)
                    active_trades[symbol]['optimized_entry1_order_id'] = order2_result.get('orderId')
                    active_trades[symbol]['optimized_entry1_filled'] = False
                    logger.info(f"✅ ORDER 2 created successfully: Order ID {order2_result.get('orderId')} @ ${optimized_entry1_price:,.8f} (${5.0} size)")
                    
                    # Track Order 2
                    order_key = f"{symbol}_{optimized_entry1_price}_{side}_ORDER2"
                    recent_orders[order_key] = current_time
                except BinanceAPIException as e:
                    logger.warning(f"⚠️ Failed to create ORDER 2: {e.message} (Code: {e.code}) - continuing with Order 1 and Order 3")
                    send_slack_alert(
                        error_type="Order 2 Creation Failed",
                        message=f"{e.message} (Code: {e.code})",
                        details={'Error_Code': e.code, 'Entry_Price': optimized_entry1_price, 'Quantity': order2_quantity, 'Side': side},
                        symbol=symbol,
                        severity='WARNING'
                    )
                    # Continue - Order 2 is optional
                except Exception as e:
                    logger.warning(f"⚠️ Unexpected error creating ORDER 2: {e} - continuing with Order 1 and Order 3")
                    send_slack_alert(
                        error_type="Order 2 Creation Error",
                        message=str(e),
                        details={'Entry_Price': optimized_entry1_price, 'Quantity': order2_quantity, 'Side': side},
                        symbol=symbol,
                        severity='WARNING'
                    )
                    # Continue - Order 2 is optional
            else:
                if optimized_entry1_price:
                    logger.info(f"ℹ️  ORDER 2 skipped: Optimized Entry 1 price is not better than original (or difference too small)")
                else:
                    logger.info(f"ℹ️  ORDER 2 skipped: Entry 1 was not optimized by AI (using original Entry 1 only)")
            
            # ORDER 3: $10 with Entry 2 price (original or optimized)
            if dca_entry_price and order3_quantity:
                order3_params = {
                    'symbol': symbol,
                    'side': side,
                    'type': 'LIMIT',
                    'timeInForce': 'GTC',
                    'quantity': order3_quantity,
                    'price': dca_entry_price,
                }
                if is_hedge_mode:
                    order3_params['positionSide'] = position_side
                
                logger.info(f"Creating ORDER 3 (Entry 2, $10): {order3_params}")
                try:
                    order3_result = client.futures_create_order(**order3_params)
                    order_results.append(order3_result)
                    active_trades[symbol]['dca_order_id'] = order3_result.get('orderId')
                    active_trades[symbol]['dca_filled'] = False
                    logger.info(f"✅ ORDER 3 created successfully: Order ID {order3_result.get('orderId')} @ ${dca_entry_price:,.8f} (${10.0} size)")
                    
                    # Track Order 3
                    order_key = f"{symbol}_{dca_entry_price}_{side}_ORDER3"
                    recent_orders[order_key] = current_time
                except BinanceAPIException as e:
                    logger.warning(f"⚠️ Failed to create ORDER 3: {e.message} (Code: {e.code}) - continuing with Order 1")
                    send_slack_alert(
                        error_type="Order 3 Creation Failed",
                        message=f"{e.message} (Code: {e.code})",
                        details={'Error_Code': e.code, 'DCA_Price': dca_entry_price, 'Quantity': order3_quantity, 'Side': side},
                        symbol=symbol,
                        severity='WARNING'
                    )
                    # Continue - Order 3 is optional if Entry 2 not provided
                except Exception as e:
                    logger.warning(f"⚠️ Unexpected error creating ORDER 3: {e} - continuing with Order 1")
                    send_slack_alert(
                        error_type="Order 3 Creation Error",
                        message=str(e),
                        details={'DCA_Price': dca_entry_price, 'Quantity': order3_quantity, 'Side': side},
                        symbol=symbol,
                        severity='WARNING'
                    )
            else:
                logger.info(f"ℹ️  ORDER 3 skipped: Entry 2 not provided in signal")
            
            # Use Order 1 result for response (primary order)
            primary_order_result = order1_result
            
            # Smart TP Strategy: Based on AI Confidence Score
            # High Confidence (>=90%): Use single TP (main TP from signal) - trust the signal completely
            # Lower Confidence (<90%): Use TP1 + TP2 strategy - secure profits early
            
            confidence_score = validation_result.get('confidence_score', 100.0) if validation_result else 100.0
            use_single_tp = confidence_score >= TP_HIGH_CONFIDENCE_THRESHOLD
            
            # Calculate entry price for TP calculation
            # TP1: Always use original Entry 1 price only (4% from Entry 1)
            # TP2: Use weighted average of all filled orders
            entry_price_for_tp1 = original_entry1_price  # Always original Entry 1 for TP1
            
            # Calculate weighted average entry price for TP2 (accounting for all 3 orders)
            # Order 1: $10 at original Entry 1
            # Order 2: $5 at optimized Entry 1 (if exists)
            # Order 3: $10 at Entry 2 (if exists)
            total_usd = 10.0  # Order 1
            weighted_sum = original_entry1_price * 10.0
            
            if optimized_entry1_price and order2_quantity:
                total_usd += 5.0
                weighted_sum += optimized_entry1_price * 5.0
            
            if dca_entry_price and order3_quantity:
                total_usd += 10.0
                weighted_sum += dca_entry_price * 10.0
            
            if total_usd > 10.0:
                entry_price_for_tp2 = weighted_sum / total_usd
                logger.info(f"📊 Weighted average entry for TP2: ${entry_price_for_tp2:,.8f} (based on ${total_usd} total USD across orders)")
            else:
                # Only Order 1 exists
                entry_price_for_tp2 = original_entry1_price
                logger.info(f"📊 Only Order 1 exists - using original Entry 1 for TP2: ${entry_price_for_tp2:,.8f}")
            
            tp_side = 'SELL' if side == 'BUY' else 'BUY'
            # Calculate total quantity from all orders
            total_qty = order1_quantity
            if order2_quantity:
                total_qty += order2_quantity
            if order3_quantity:
                total_qty += order3_quantity
            logger.info(f"📊 Total position size: {total_qty} (Order 1: {order1_quantity}, Order 2: {order2_quantity if order2_quantity else 0}, Order 3: {order3_quantity if order3_quantity else 0})")
            
            if use_single_tp:
                # HIGH CONFIDENCE: Use single TP (main TP from signal)
                if take_profit and take_profit > 0:
                    main_tp_price = format_price_precision(take_profit, tick_size)
                else:
                    # If no TP provided, calculate a conservative TP
                    default_tp_percent = 0.05  # 5% profit for high confidence
                    if side == 'BUY':  # LONG position
                        main_tp_price = entry_price_for_tp2 * (1 + default_tp_percent)
                    else:  # SHORT position
                        main_tp_price = entry_price_for_tp2 * (1 - default_tp_percent)
                    main_tp_price = format_price_precision(main_tp_price, tick_size)
                    logger.info(f"📊 High confidence signal ({confidence_score:.1f}%) - TP not provided, calculating with {default_tp_percent*100}% profit: {main_tp_price}")
                
                # Store as single TP (100% of position)
                active_trades[symbol]['tp1_price'] = None  # No TP1
                active_trades[symbol]['tp2_price'] = main_tp_price  # Use TP2 as main TP
                active_trades[symbol]['tp_side'] = tp_side
                active_trades[symbol]['tp1_quantity'] = 0  # No TP1
                active_trades[symbol]['tp2_quantity'] = total_qty  # 100% at main TP
                active_trades[symbol]['tp_working_type'] = 'MARK_PRICE'
                active_trades[symbol]['use_single_tp'] = True  # Flag for single TP mode
                logger.info(f"📝 HIGH CONFIDENCE ({confidence_score:.1f}%) - Single TP configured for {symbol}:")
                logger.info(f"   → Main TP: @ ${main_tp_price:,.8f} (closes 100% = {total_qty} of position)")
                logger.info(f"   → Strategy: Trusting signal completely - using single TP")
            else:
                    # LOWER CONFIDENCE: Use TP1 + TP2 strategy (secure profits early)
                    # Calculate TP1: 3-4% profit from Entry 1 ONLY (not average)
                    tp1_percent = TP1_PERCENT / 100.0
                    if side == 'BUY':  # LONG position
                        tp1_price = entry_price_for_tp1 * (1 + tp1_percent)  # Use Entry 1 only
                    else:  # SHORT position
                        tp1_price = entry_price_for_tp1 * (1 - tp1_percent)  # Use Entry 1 only
                    tp1_price = format_price_precision(tp1_price, tick_size)
                    logger.info(f"📊 TP1 calculated: {TP1_PERCENT}% from Entry 1 (${original_entry1_price:,.8f}) = ${tp1_price:,.8f}")
                    
                    # TP2: Use the TP from webhook (or AI-optimized TP)
                    if take_profit and take_profit > 0:
                        tp2_price = format_price_precision(take_profit, tick_size)
                    else:
                        # If no TP provided, calculate a default TP2 (higher than TP1)
                        default_tp2_percent = 0.055  # 5.5% profit
                        if side == 'BUY':  # LONG position
                            tp2_price = entry_price_for_tp2 * (1 + default_tp2_percent)
                        else:  # SHORT position
                            tp2_price = entry_price_for_tp2 * (1 - default_tp2_percent)
                        tp2_price = format_price_precision(tp2_price, tick_size)
                        logger.info(f"📊 TP2 not provided in webhook - calculating with {default_tp2_percent*100}% profit: {tp2_price}")
                    
                    # Calculate TP1 and TP2 quantities based on split percentages
                    tp1_quantity = total_qty * (TP1_SPLIT / 100.0)  # 70% of position
                    tp2_quantity = total_qty * (TP2_SPLIT / 100.0)  # 30% of position
                    
                    # Store TP1 and TP2 details
                    active_trades[symbol]['tp1_price'] = tp1_price
                    active_trades[symbol]['tp2_price'] = tp2_price
                    active_trades[symbol]['tp_side'] = tp_side
                    active_trades[symbol]['tp1_quantity'] = tp1_quantity
                    active_trades[symbol]['tp2_quantity'] = tp2_quantity
                    active_trades[symbol]['tp_working_type'] = 'MARK_PRICE'
                    active_trades[symbol]['use_single_tp'] = False  # Flag for TP1+TP2 mode
                    logger.info(f"📝 LOWER CONFIDENCE ({confidence_score:.1f}%) - TP1 + TP2 configured for {symbol}:")
                    logger.info(f"   → TP1: {TP1_PERCENT}% profit @ ${tp1_price:,.8f} (closes {TP1_SPLIT}% = {tp1_quantity} of position)")
                    logger.info(f"   → TP2: @ ${tp2_price:,.8f} (closes {TP2_SPLIT}% = {tp2_quantity} of position)")
                    logger.info(f"   → Strategy: Securing profits early with TP1, letting TP2 run")
            
            logger.info(f"   → TPs will be created automatically when position opens (background thread checks every 1min/2min)")
            
            # Send signal notification to Slack
            try:
                timeframe = signal_data.get('timeframe', 'Unknown')
                confidence_score = validation_result.get('confidence_score', 100.0) if validation_result else 100.0
                risk_level = validation_result.get('risk_level', 'MEDIUM') if validation_result else 'MEDIUM'
                # Get TP prices for notification
                if use_single_tp:
                    main_tp = active_trades[symbol].get('tp2_price')
                    tp1_for_notif = None
                else:
                    main_tp = active_trades[symbol].get('tp2_price')
                    tp1_for_notif = active_trades[symbol].get('tp1_price')
                
                send_signal_notification(
                    symbol=symbol,
                    signal_side=signal_side,
                    timeframe=timeframe,
                    confidence_score=confidence_score,
                    risk_level=risk_level,
                    entry1_price=original_entry1_price,  # Order 1: Original Entry 1
                    entry2_price=dca_entry_price,  # Order 3: Entry 2
                    stop_loss=stop_loss,
                    take_profit=main_tp,  # Main TP (TP2 in dual mode, or single TP in high confidence)
                    tp1_price=tp1_for_notif,  # TP1 (only in dual mode)
                    use_single_tp=use_single_tp,  # Flag for notification formatting
                    validation_result=validation_result,
                    optimized_entry1_price=optimized_entry1_price  # Order 2: Optimized Entry 1 (if exists)
                )
            except Exception as e:
                logger.debug(f"Failed to send signal notification: {e}")
            
            # Use primary order result for response
            order_result = primary_order_result
            entry_price = primary_entry_price
            quantity = primary_quantity
            entry_type = "PRIMARY"
            
            # Check if position exists and create TP1 and TP2 immediately (in case entry filled quickly)
            # Also schedule a delayed retry in case Binance hasn't updated position yet
            if symbol in active_trades and 'tp1_price' in active_trades[symbol] and 'tp2_price' in active_trades[symbol]:
                # Immediate check
                create_tp1_tp2_if_needed(symbol, active_trades[symbol])
                # Delayed retry (5 seconds) in case position updates are delayed
                delayed_tp_creation(symbol, delay_seconds=5)
                # Another delayed retry (15 seconds) for slower fills
                delayed_tp_creation(symbol, delay_seconds=15)
            
        else:
            # This is a DCA entry - create only DCA order
            # Ensure dca_entry_price is set (should be set above, but use entry_price as fallback)
            if not dca_entry_price:
                dca_entry_price = entry_price
            
            dca_order_params = {
                'symbol': symbol,
                'side': side,
                'type': 'LIMIT',
                'timeInForce': 'GTC',
                'quantity': dca_quantity,
                'price': dca_entry_price,
            }
            if is_hedge_mode:
                dca_order_params['positionSide'] = position_side
            
            logger.info(f"Creating DCA entry limit order: {dca_order_params}")
            try:
                order_result = client.futures_create_order(**dca_order_params)
                order_results.append(order_result)
                active_trades[symbol]['dca_order_id'] = order_result.get('orderId')
                active_trades[symbol]['dca_filled'] = False
                logger.info(f"✅ DCA entry order created successfully: Order ID {order_result.get('orderId')}")
            except BinanceAPIException as e:
                logger.error(f"❌ Failed to create DCA entry order: {e.message} (Code: {e.code})")
                send_slack_alert(
                    error_type="DCA Entry Order Creation Failed",
                    message=f"{e.message} (Code: {e.code})",
                    details={'Error_Code': e.code, 'DCA_Price': dca_price, 'Quantity': dca_qty, 'Side': side},
                    symbol=symbol,
                    severity='ERROR'
                )
                return {'success': False, 'error': f'Failed to create DCA order: {e.message}'}
            except Exception as e:
                logger.error(f"❌ Unexpected error creating DCA entry order: {e}")
                send_slack_alert(
                    error_type="DCA Entry Order Creation Error",
                    message=str(e),
                    details={'DCA_Price': dca_price, 'Quantity': dca_qty, 'Side': side},
                    symbol=symbol,
                    severity='ERROR'
                )
                return {'success': False, 'error': f'Unexpected error: {str(e)}'}
            
            # Track order
            order_key = f"{symbol}_{dca_entry_price}_{side}_DCA"
            recent_orders[order_key] = current_time
            
            entry_price = dca_entry_price
            quantity = dca_quantity
            entry_type = "DCA"
            
            # When DCA entry alert comes, check if position exists and create TP1/TP2 immediately
            # This handles the case where an entry filled and TradingView sent DCA fill alert
            if symbol in active_trades and 'tp1_price' in active_trades[symbol] and 'tp2_price' in active_trades[symbol]:
                logger.info(f"DCA entry alert received - checking for position to create TP1/TP2 immediately")
                create_tp1_tp2_if_needed(symbol, active_trades[symbol])
        
        # Cleanup closed positions periodically
        if len(active_trades) > 100:
            cleanup_closed_positions()
        
        # Clean old orders from tracking (keep last 1000)
        if len(recent_orders) > 1000:
            # Remove oldest entries
            sorted_orders = sorted(recent_orders.items(), key=lambda x: x[1])
            for key, _ in sorted_orders[:-1000]:
                del recent_orders[key]
        
        logger.info(f"{entry_type} order(s) created successfully: {order_results}")
        
        # After creating orders, check if position exists and create TP1/TP2 immediately
        # This handles cases where entry filled between webhook calls
        if symbol in active_trades and 'tp1_price' in active_trades[symbol] and 'tp2_price' in active_trades[symbol]:
            logger.info(f"Checking for position to create TP1/TP2 orders immediately")
            create_tp1_tp2_if_needed(symbol, active_trades[symbol])
        
        # For DCA entry: Create TP order if not exists (using same TP price from primary entry)
        if is_dca_entry and take_profit and take_profit > 0:
            # Check for existing TP orders
            has_orders, open_orders = check_existing_orders(symbol)
            existing_tp = [o for o in open_orders if o.get('type') == 'TAKE_PROFIT_MARKET']
            
            if not existing_tp:
                # Get current position quantity to update TP
                try:
                    positions = client.futures_position_information(symbol=symbol)
                    current_position_qty = 0
                    for pos in positions:
                        pos_amt = float(pos.get('positionAmt', 0))
                        if abs(pos_amt) > 0:
                            pos_side = pos.get('positionSide', 'BOTH')
                            if pos_side == 'BOTH' or pos_side == position_side.upper():
                                current_position_qty = abs(pos_amt)
                                break
                    
                    if current_position_qty > 0:
                        # Update TP with current position quantity
                        tp_side = 'SELL' if side == 'BUY' else 'BUY'
                        tp_price = format_price_precision(take_profit, tick_size)
                        tp_params = {
                            'symbol': symbol,
                            'side': tp_side,
                            'type': 'TAKE_PROFIT_MARKET',
                            'timeInForce': 'GTC',
                            'quantity': current_position_qty,  # Current position quantity
                            'stopPrice': tp_price
                            # Note: reduceOnly is not required for TAKE_PROFIT_MARKET orders
                        }
                        if is_hedge_mode:
                            tp_params['positionSide'] = position_side
                        tp_order = client.futures_create_order(**tp_params)
                        if symbol in active_trades:
                            active_trades[symbol]['tp_order_id'] = tp_order.get('orderId')
                        logger.info(f"Take profit order created for DCA entry: {tp_order}")
                except BinanceAPIException as e:
                    # Check if order type is not supported (error -4120)
                    if e.code == -4120 or 'order type not supported' in str(e).lower():
                        logger.warning(f"⚠️ TAKE_PROFIT_MARKET orders not supported for {symbol} (Code: {e.code}). This symbol may not support conditional orders. You may need to set TP manually in Binance UI.")
                        send_slack_alert(
                            error_type="DCA Take Profit Order Type Not Supported",
                            message=f"TAKE_PROFIT_MARKET orders are not supported for {symbol}. You may need to set TP manually in Binance UI.",
                            details={'Error_Code': e.code, 'TP_Price': tp_price, 'Symbol': symbol, 'DCA_Entry_Price': dca_entry_price if 'dca_entry_price' in locals() else 'Unknown'},
                            symbol=symbol,
                            severity='WARNING'
                        )
                    else:
                        logger.error(f"Failed to create take profit order for DCA: {e}")
                        send_slack_alert(
                            error_type="DCA Take Profit Creation Failed",
                            message=str(e),
                            details={'Error_Code': e.code, 'DCA_Entry_Price': dca_entry_price if 'dca_entry_price' in locals() else 'Unknown'},
                            symbol=symbol,
                            severity='ERROR'
                        )
                except Exception as e:
                    logger.error(f"Failed to create take profit order for DCA: {e}")
                    send_slack_alert(
                        error_type="DCA Take Profit Creation Failed",
                        message=str(e),
                        details={'DCA_Entry_Price': dca_entry_price if 'dca_entry_price' in locals() else 'Unknown'},
                        symbol=symbol,
                        severity='ERROR'
                    )
            else:
                logger.info(f"Take profit order already exists for {symbol}, skipping creation")
        
        return {
            'success': True,
            'order_id': order_result.get('orderId'),
            'symbol': symbol,
            'side': side,
            'price': entry_price,
            'quantity': quantity,
            'entry_type': entry_type,
            'leverage': LEVERAGE,
            'position_value_usd': ENTRY_SIZE_USD * LEVERAGE
        }
        
    except BinanceAPIException as e:
        logger.error(f"Binance API error: {e}")
        # symbol is always defined by this point (extracted at function start)
        send_slack_alert(
            error_type="Binance API Error",
            message=f"{e.message} (Code: {e.code})",
            details={'Error_Code': e.code, 'Event': event if 'event' in locals() else 'UNKNOWN'},
            symbol=symbol if 'symbol' in locals() else None,
            severity='ERROR'
        )
        return {'success': False, 'error': f'Binance API error: {e.message}'}
    except BinanceOrderException as e:
        logger.error(f"Binance order error: {e}")
        send_slack_alert(
            error_type="Binance Order Error",
            message=f"{e.message} (Code: {e.code})",
            details={'Error_Code': e.code, 'Event': event if 'event' in locals() else 'UNKNOWN'},
            symbol=symbol if 'symbol' in locals() else None,
            severity='ERROR'
        )
        return {'success': False, 'error': f'Binance order error: {e.message}'}
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        send_slack_alert(
            error_type="Unexpected Error",
            message=str(e),
            details={'Event': event if 'event' in locals() else 'UNKNOWN'},
            symbol=symbol if 'symbol' in locals() else None,
            severity='ERROR'
        )
        return {'success': False, 'error': str(e)}
