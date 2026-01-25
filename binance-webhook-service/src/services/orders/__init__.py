# Order management module - functions are in order_manager.py
from .order_manager import (
    create_limit_order, create_tp_if_needed, create_single_tp_order,
    create_tp1_tp2_if_needed, update_trailing_stop_loss,
    close_position_at_market, calculate_quantity,
    delayed_tp_creation, create_missing_tp_orders
)
