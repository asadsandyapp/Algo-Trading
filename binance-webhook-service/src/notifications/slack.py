"""
Notification module for Binance Webhook Service
Handles Slack notifications for errors, signals, and exits
"""
import requests
import threading
from datetime import datetime
try:
    # Try relative import first (when imported as package)
    from ..config import (
        SLACK_WEBHOOK_URL, SLACK_SIGNAL_WEBHOOK_URL, BINANCE_TESTNET,
        AI_VALIDATION_MIN_CONFIDENCE, TP1_PERCENT
    )
    from ..core import logger
except ImportError:
    # Fall back to absolute import (when src/ is in Python path)
    from config import (
        SLACK_WEBHOOK_URL, SLACK_SIGNAL_WEBHOOK_URL, BINANCE_TESTNET,
        AI_VALIDATION_MIN_CONFIDENCE, TP1_PERCENT
    )
    from core import logger


def send_slack_alert(error_type, message, details=None, symbol=None, severity='ERROR'):
    """
    Send a beautiful error notification to Slack webhook
    
    Args:
        error_type: Type of error (e.g., 'Binance API Error', 'Order Creation Failed')
        message: Main error message
        details: Additional details dict (optional)
        symbol: Trading symbol if applicable (optional)
        severity: ERROR, WARNING, or CRITICAL
    """
    if not SLACK_WEBHOOK_URL:
        return  # Skip if webhook URL not configured
    
    try:
        # Determine emoji based on severity
        emoji_map = {
            'ERROR': '🚨',
            'WARNING': '⚠️',
            'CRITICAL': '🔥'
        }
        emoji = emoji_map.get(severity, '🚨')
        
        # Build the message
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        environment = 'TESTNET' if BINANCE_TESTNET else 'PRODUCTION'
        
        # Format the message with beautiful structure
        slack_message = f"""{emoji} *{severity}: {error_type}*

*App:* Binance Trading Bot
*Environment:* {environment}
*Module:* Webhook Service
*Time:* {timestamp}"""
        
        if symbol:
            slack_message += f"\n*Symbol:* {symbol}"
        
        slack_message += f"\n*Message:* {message}"
        
        # Add additional details if provided
        if details:
            slack_message += "\n*Details:*"
            for key, value in details.items():
                if value is not None:
                    slack_message += f"\n  • *{key}:* {value}"
        
        # Send to Slack (non-blocking in a thread)
        def send_async():
            try:
                payload = {'text': slack_message}
                response = requests.post(
                    SLACK_WEBHOOK_URL,
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=5
                )
                response.raise_for_status()
            except Exception as e:
                # Don't log Slack errors to avoid infinite loops
                logger.debug(f"Failed to send Slack notification: {e}")
        
        # Send in background thread to avoid blocking
        thread = threading.Thread(target=send_async, daemon=True)
        thread.start()
        
    except Exception as e:
        # Silently fail - don't break the service if Slack is down
        logger.debug(f"Error preparing Slack notification: {e}")


def send_signal_rejection_notification(symbol, signal_side, timeframe, entry_price, 
                                       rejection_reason, confidence_score=None, risk_level=None, 
                                       validation_result=None):
    """
    Send a rejection notification to Slack exception channel when signal is rejected
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        signal_side: 'LONG' or 'SHORT'
        timeframe: Trading timeframe (e.g., '1H', '4H')
        entry_price: Entry price from signal
        rejection_reason: Reason for rejection
        confidence_score: AI confidence score if available
        risk_level: Risk level if available
        validation_result: Full validation result dict (optional)
    """
    if not SLACK_WEBHOOK_URL:
        return  # Skip if webhook URL not configured
    
    try:
        # Determine side emoji
        side_emoji = '📈' if signal_side == 'LONG' else '📉'
        
        # Build the message
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        environment = 'TESTNET' if BINANCE_TESTNET else 'PRODUCTION'
        
        # Format symbol consistently (remove .P suffix if present, uppercase)
        formatted_symbol = symbol.replace('.P', '').upper()
        # Format timeframe consistently (uppercase, ensure proper format)
        formatted_timeframe = timeframe.upper() if timeframe else 'N/A'
        
        # Format entry price consistently
        entry_str = f'${entry_price:,.8f}' if entry_price else 'N/A'
        
        # Build rejection message
        slack_message = f"""🚫 *SIGNAL REJECTED - AI VALIDATION FAILED*

*Symbol:* `{formatted_symbol}`
*Timeframe:* `{formatted_timeframe}`
*Side:* {side_emoji} {signal_side}
*Environment:* {environment}
*Time:* {timestamp}

*Signal Details:*
  • Entry Price: {entry_str}"""
        
        if confidence_score is not None:
            slack_message += f"\n  • Confidence Score: {confidence_score:.1f}% (Threshold: {AI_VALIDATION_MIN_CONFIDENCE}%)"
        
        if risk_level:
            risk_emoji_map = {
                'LOW': '🟢',
                'MEDIUM': '🟡',
                'HIGH': '🔴'
            }
            risk_emoji = risk_emoji_map.get(risk_level, '⚪')
            slack_message += f"\n  • Risk Level: {risk_emoji} {risk_level}"
        
        slack_message += f"\n\n*Rejection Reason:*\n{rejection_reason}"
        
        # Add AI reasoning if available (truncated)
        if validation_result and validation_result.get('reasoning'):
            reasoning = validation_result.get('reasoning', '')
            # Convert to single line
            reasoning = ' '.join(reasoning.split())
            # Truncate if too long
            if len(reasoning) > 500:
                reasoning = reasoning[:497] + "..."
            slack_message += f"\n\n*AI Analysis:* {reasoning}"
        
        # Send to Slack (non-blocking in a thread)
        def send_async():
            try:
                payload = {'text': slack_message}
                response = requests.post(
                    SLACK_WEBHOOK_URL,
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=5
                )
                response.raise_for_status()
                logger.info(f"✅ Rejection notification sent to Slack for {symbol}")
            except Exception as e:
                logger.debug(f"Failed to send Slack rejection notification: {e}")
        
        # Send in background thread to avoid blocking
        thread = threading.Thread(target=send_async, daemon=True)
        thread.start()
        
    except Exception as e:
        logger.debug(f"Error preparing Slack rejection notification: {e}")


def send_signal_notification(symbol, signal_side, timeframe, confidence_score, risk_level, 
                             entry1_price, entry2_price, stop_loss, take_profit, 
                             tp1_price=None, use_single_tp=False, validation_result=None,
                             optimized_entry1_price=None):
    """
    Send a beautiful signal notification to Slack signal channel after order is opened
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        signal_side: 'LONG' or 'SHORT'
        timeframe: Trading timeframe (e.g., '1H', '4H')
        confidence_score: AI confidence score (0-100)
        risk_level: Risk level from AI validation ('LOW', 'MEDIUM', 'HIGH')
        entry1_price: Original Entry 1 price (Order 1: $10)
        entry2_price: Entry 2 price (Order 3: $10, optional)
        stop_loss: Stop loss price
        take_profit: Take profit price
        tp1_price: TP1 price (optional)
        use_single_tp: Whether using single TP strategy
        validation_result: Full validation result dict (optional)
        optimized_entry1_price: Optimized Entry 1 price (Order 2: $5, optional)
    """
    if not SLACK_SIGNAL_WEBHOOK_URL:
        return  # Skip if webhook URL not configured
    
    try:
        # Determine signal strength emoji and text based on confidence score
        if confidence_score >= 80:
            strength_emoji = '🔥'
            strength_text = 'VERY STRONG'
        elif confidence_score >= 65:
            strength_emoji = '✅'
            strength_text = 'STRONG'
        elif confidence_score >= 50:
            strength_emoji = '⚡'
            strength_text = 'MODERATE'
        else:
            strength_emoji = '⚠️'
            strength_text = 'WEAK'
        
        # Determine side emoji
        side_emoji = '📈' if signal_side == 'LONG' else '📉'
        
        # Determine risk level emoji
        risk_emoji_map = {
            'LOW': '🟢',
            'MEDIUM': '🟡',
            'HIGH': '🔴'
        }
        risk_emoji = risk_emoji_map.get(risk_level, '⚪')
        
        # Build the message
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        environment = 'TESTNET' if BINANCE_TESTNET else 'PRODUCTION'
        
        # Format symbol consistently (remove .P suffix if present, uppercase)
        formatted_symbol = symbol.replace('.P', '').upper()
        # Format timeframe consistently (uppercase, ensure proper format)
        formatted_timeframe = timeframe.upper() if timeframe else 'N/A'
        
        # Format prices consistently (all with same format: $X,XXX.XXXXXXXX)
        entry1_str = f'${entry1_price:,.8f}' if entry1_price else 'N/A'
        optimized_entry1_str = f'${optimized_entry1_price:,.8f}' if optimized_entry1_price else None
        entry2_str = f'${entry2_price:,.8f}' if entry2_price else 'N/A'
        stop_loss_str = f'${stop_loss:,.8f}' if stop_loss else 'N/A'
        tp1_str = f'${tp1_price:,.8f}' if tp1_price else 'N/A'
        tp2_str = f'${take_profit:,.8f}' if take_profit else 'N/A'
        
        # Calculate total investment
        total_investment = 10.0  # Order 1
        if optimized_entry1_price:
            total_investment += 5.0  # Order 2
        if entry2_price:
            total_investment += 10.0  # Order 3
        
        # Format the message with beautiful structure (consistent formatting)
        slack_message = f"""{side_emoji} *NEW {signal_side} SIGNAL - ORDERS OPENED*

*Symbol:* `{formatted_symbol}`
*Timeframe:* `{formatted_timeframe}`
*Environment:* {environment}
*Time:* {timestamp}

*Signal Strength:* {strength_emoji} {strength_text} ({confidence_score:.1f}%)
*Risk Level:* {risk_emoji} {risk_level}

*Entry Orders:*
  • Order 1: {entry1_str} - $10.00 (Original Entry 1)"""
        
        # Add Order 2 if optimized Entry 1 exists
        if optimized_entry1_str:
            slack_message += f"\n  • Order 2: {optimized_entry1_str} - $5.00 (AI Optimized Entry 1)"
        
        # Add Order 3 if Entry 2 exists
        if entry2_price:
            slack_message += f"\n  • Order 3: {entry2_str} - $10.00 (Entry 2)"
        
        slack_message += f"""

*Risk Management:*
  • Stop Loss: {stop_loss_str}"""
        
        # Add TP information based on strategy
        if use_single_tp:
            slack_message += f"""
  • Take Profit (100%): {tp2_str} - Main TP (High Confidence: {confidence_score:.1f}%)"""
        else:
            slack_message += f"""
  • Take Profit 1 (70%): {tp1_str} - {TP1_PERCENT}% profit
  • Take Profit 2 (30%): {tp2_str} - Original/AI TP (Lower Confidence: {confidence_score:.1f}%)"""
        
        slack_message += "\n\n"
        
        # Add AI reasoning if available (single line only)
        if validation_result and validation_result.get('reasoning'):
            reasoning = validation_result.get('reasoning', '')
            # Convert to single line by replacing newlines with spaces
            reasoning = ' '.join(reasoning.split())
            # Show more complete reasoning (increased from 200 to 600 chars for better context)
            if len(reasoning) > 600:
                reasoning = reasoning[:597] + "..."
            slack_message += f"*AI Analysis:* {reasoning}\n"
        
        # Send to Slack (non-blocking in a thread)
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
                logger.info(f"✅ Signal notification sent to Slack for {symbol}")
            except Exception as e:
                # Don't log Slack errors to avoid infinite loops
                logger.debug(f"Failed to send Slack signal notification: {e}")
        
        # Send in background thread to avoid blocking
        thread = threading.Thread(target=send_async, daemon=True)
        thread.start()
        
    except Exception as e:
        # Silently fail - don't break the service if Slack is down
        logger.debug(f"Error preparing Slack signal notification: {e}")


def send_exit_notification(symbol, signal_side, timeframe, exit_price, entry_prices=None, 
                           pnl=None, pnl_percent=None, reason=None):
    """
    Send exit notification to Slack signal channel when position is closed
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        signal_side: 'LONG' or 'SHORT'
        timeframe: Trading timeframe (e.g., '1H', '4H')
        exit_price: Price at which position was closed
        entry_prices: Dict with 'entry1', 'entry2', 'optimized_entry1' prices (optional)
        pnl: Profit/Loss in USD (optional)
        pnl_percent: Profit/Loss percentage (optional)
        reason: Reason for exit (e.g., 'TP Hit', 'SL Hit', 'Manual Exit') (optional)
    """
    if not SLACK_SIGNAL_WEBHOOK_URL:
        return  # Skip if webhook URL not configured
    
    try:
        # Determine side emoji
        side_emoji = '📈' if signal_side == 'LONG' else '📉'
        
        # Determine P&L emoji and text
        if pnl is not None:
            pnl_emoji = '💰' if pnl >= 0 else '📉'
            pnl_text = f"+${abs(pnl):.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
            if pnl_percent is not None:
                pnl_text += f" ({'+' if pnl_percent >= 0 else ''}{pnl_percent:.2f}%)"
        else:
            pnl_emoji = '📊'
            pnl_text = 'N/A'
        
        # Build the message
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        environment = 'TESTNET' if BINANCE_TESTNET else 'PRODUCTION'
        
        # Format symbol consistently (remove .P suffix if present, uppercase)
        formatted_symbol = symbol.replace('.P', '').upper()
        # Format timeframe consistently (uppercase, ensure proper format)
        formatted_timeframe = timeframe.upper() if timeframe else 'N/A'
        
        # Format prices consistently (all with same format: $X,XXX.XXXXXXXX)
        exit_str = f'${exit_price:,.8f}' if exit_price else 'N/A'
        
        # Format the message with beautiful structure
        slack_message = f"""{side_emoji} *{signal_side} POSITION CLOSED - EXIT*

*Symbol:* `{formatted_symbol}`
*Timeframe:* `{formatted_timeframe}`
*Environment:* {environment}
*Time:* {timestamp}

*Exit Details:*
  • Exit Price: {exit_str}"""
        
        # Add reason if provided
        if reason:
            slack_message += f"\n  • Reason: {reason}"
        
        # Add entry prices if available
        has_entry_info = False
        if entry_prices:
            # Check if we have at least one entry price
            if entry_prices.get('entry1') or entry_prices.get('entry2') or entry_prices.get('optimized_entry1') or entry_prices.get('average_entry'):
                has_entry_info = True
                slack_message += "\n\n*Entry Prices:*"
                if entry_prices.get('entry1'):
                    entry1_str = f'${entry_prices["entry1"]:,.8f}'
                    slack_message += f"\n  • Order 1: {entry1_str} - $10.00 (Original Entry 1)"
                
                if entry_prices.get('optimized_entry1'):
                    opt_entry1_str = f'${entry_prices["optimized_entry1"]:,.8f}'
                    slack_message += f"\n  • Order 2: {opt_entry1_str} - $10.00 (AI Optimized Entry 1)"
                
                if entry_prices.get('entry2'):
                    entry2_str = f'${entry_prices["entry2"]:,.8f}'
                    slack_message += f"\n  • Order 3: {entry2_str} - $10.00 (Entry 2 / DCA)"
                
                # Show average entry price if available (useful when multiple entries filled)
                if entry_prices.get('average_entry'):
                    avg_entry_str = f'${entry_prices["average_entry"]:,.8f}'
                    slack_message += f"\n  • Average Entry: {avg_entry_str} (Weighted average of filled orders)"
        
        # Add Stop Loss and Take Profit if available
        if entry_prices:
            has_sl_tp = False
            if entry_prices.get('stop_loss'):
                if not has_entry_info:
                    slack_message += "\n"
                sl_str = f'${entry_prices["stop_loss"]:,.8f}'
                slack_message += f"\n*Stop Loss:* {sl_str}"
                has_sl_tp = True
            
            if entry_prices.get('take_profit'):
                tp_str = f'${entry_prices["take_profit"]:,.8f}'
                slack_message += f"\n*Take Profit:* {tp_str}"
                has_sl_tp = True
        
        # Add P&L if available
        if pnl is not None:
            slack_message += f"\n\n*Profit/Loss:* {pnl_emoji} {pnl_text}"
        
        slack_message += "\n"
        
        # Send to Slack (non-blocking in a thread)
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
                logger.info(f"✅ Exit notification sent to Slack for {symbol}")
            except Exception as e:
                # Don't log Slack errors to avoid infinite loops
                logger.debug(f"Failed to send Slack exit notification: {e}")
        
        # Send in background thread to avoid blocking
        thread = threading.Thread(target=send_async, daemon=True)
        thread.start()
        
    except Exception as e:
        # Silently fail - don't break the service if Slack is down
        logger.debug(f"Error preparing Slack exit notification: {e}")

