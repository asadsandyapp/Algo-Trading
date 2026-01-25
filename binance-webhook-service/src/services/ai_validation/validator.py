"""
AI Validation service for Binance Webhook Service
Handles Gemini API integration and signal validation
"""
import re
import time
import math
import threading
from typing import Dict, Tuple

# Import dependencies
try:
    # Try relative import first (when imported as package)
    from ...core import client, logger, gemini_client, gemini_model_name
    from ...config import (
        ENABLE_AI_VALIDATION, AI_VALIDATION_MIN_CONFIDENCE, ENABLE_AI_PRICE_SUGGESTIONS,
        GEMINI_MODEL_NAMES, TP1_PERCENT, TP1_SPLIT, TP2_SPLIT
    )
    from ...models.state import validation_cache, VALIDATION_CACHE_TTL
    from ...utils.helpers import format_symbol, safe_float
except ImportError:
    # Fall back to absolute import (when src/ is in Python path)
    from core import client, logger, gemini_client, gemini_model_name
    from config import (
        ENABLE_AI_VALIDATION, AI_VALIDATION_MIN_CONFIDENCE, ENABLE_AI_PRICE_SUGGESTIONS,
        GEMINI_MODEL_NAMES, TP1_PERCENT, TP1_SPLIT, TP2_SPLIT
    )
    from models.state import validation_cache, VALIDATION_CACHE_TTL
    from utils.helpers import format_symbol, safe_float

def validate_entry2_standalone_with_ai(signal_data, entry2_price, original_validation_result):
    """Explicitly ask AI to validate Entry 2 as a standalone trade when Entry 1 is rejected
    
    Args:
        signal_data: Original signal data
        entry2_price: Entry 2 price to validate
        original_validation_result: Original AI validation result (Entry 1 was rejected)
    
    Returns:
        dict: Validation result with 'is_valid', 'confidence_score', 'reasoning', 'risk_level'
    """
    if not entry2_price or entry2_price <= 0:
        return {
            'is_valid': False,
            'confidence_score': 0.0,
            'reasoning': 'Entry 2 price is missing or invalid',
            'risk_level': 'HIGH'
        }
    
    symbol = format_symbol(signal_data.get('symbol', ''))
    signal_side = signal_data.get('signal_side', 'LONG')
    timeframe = signal_data.get('timeframe', '1H')
    stop_loss = safe_float(signal_data.get('stop_loss'), default=None)
    take_profit = safe_float(signal_data.get('take_profit'), default=None)
    
    logger.info(f"🔍 [ENTRY 2 STANDALONE VALIDATION] Asking AI to validate Entry 2 as standalone trade for {symbol}")
    
    # Create a modified signal data with Entry 2 as the primary entry
    entry2_signal_data = signal_data.copy()
    entry2_signal_data['entry_price'] = entry2_price
    # Clear second_entry_price since we're validating Entry 2 as standalone (no DCA)
    entry2_signal_data['second_entry_price'] = None
    entry2_signal_data['_entry2_standalone_validation'] = True
    entry2_signal_data['_original_entry1_rejected'] = True
    entry2_signal_data['_original_rejection_reason'] = original_validation_result.get('reasoning', 'Entry 1 rejected')
    
    # Call AI validation with a special prompt for Entry 2 standalone
    try:
        validation_result = validate_signal_with_ai(entry2_signal_data)
        
        # Add a flag to indicate this is Entry 2 standalone validation
        validation_result['_entry2_standalone'] = True
        validation_result['_entry2_price'] = entry2_price
        
        logger.info(f"✅ Entry 2 standalone validation result for {symbol}: Valid={validation_result.get('is_valid')}, Confidence={validation_result.get('confidence_score', 0):.1f}%")
        return validation_result
    except Exception as e:
        logger.error(f"❌ Error validating Entry 2 standalone for {symbol}: {e}", exc_info=True)
        return {
            'is_valid': False,
            'confidence_score': 0.0,
            'reasoning': f'Error validating Entry 2 standalone: {str(e)}',
            'risk_level': 'HIGH'
        }


def parse_entry_analysis_from_reasoning(reasoning):
    """Parse AI reasoning to detect if Entry 1 is bad but Entry 2 is good
    
    Args:
        reasoning: AI reasoning text
    
    Returns:
        tuple: (entry1_is_bad: bool, entry2_is_good: bool)
    """
    if not reasoning:
        return False, False
    
    reasoning_lower = reasoning.lower()
    
    # FIRST: Check for POSITIVE keywords about Entry 1 (if found, Entry 1 is NOT bad)
    entry1_good_keywords = [
        'entry 1.*optimal',
        'entry 1.*is optimal',
        'entry 1.*perfect',
        'entry 1.*excellent',
        'entry 1.*well-positioned',
        'entry 1.*at.*support',  # For LONG
        'entry 1.*at.*resistance',  # For SHORT
        'entry 1.*aligns.*perfectly',
        'entry 1.*high-probability',
    ]
    
    entry1_is_good = False
    for keyword in entry1_good_keywords:
        if re.search(keyword, reasoning_lower):
            entry1_is_good = True
            break
    
    # Check Entry 1 analysis - only mark as bad if NO positive keywords found
    entry1_bad_keywords = [
        'entry 1.*not optimal',
        'entry 1.*not at.*optimal',
        'entry 1.*not.*good',
        'entry 1.*poor',
        'entry 1.*bad',
        'entry 1.*sub-optimal',
        'entry 1.*not.*institutional',
        'entry 1.*not.*well-positioned',
        'entry 1.*should.*be.*replaced',
        'entry 1.*needs.*optimization',
    ]
    
    entry1_is_bad = False
    # Only mark as bad if there are negative keywords AND no positive keywords
    if not entry1_is_good:
        for keyword in entry1_bad_keywords:
            if re.search(keyword, reasoning_lower):
                entry1_is_bad = True
                break
    
    # Check Entry 2 analysis
    entry2_good_keywords = [
        'entry 2.*correct',
        'entry 2.*optimal',
        'entry 2.*good',
        'entry 2.*well-positioned',
        'entry 2.*at.*resistance',  # For SHORT
        'entry 2.*at.*support',      # For LONG
        'entry 2.*above entry 1.*correct',  # For SHORT
        'entry 2.*below entry 1.*correct',  # For LONG
    ]
    
    entry2_is_good = False
    for keyword in entry2_good_keywords:
        if re.search(keyword, reasoning_lower):
            entry2_is_good = True
            break
    
    # Also check for explicit statements
    if 'entry 2 analysis' in reasoning_lower:
        entry2_section = reasoning_lower.split('entry 2 analysis')[1] if 'entry 2 analysis' in reasoning_lower else ''
        if entry2_section:
            # Check if Entry 2 section says it's good/optimal/correct
            if any(word in entry2_section[:200] for word in ['correct', 'optimal', 'good', 'well-positioned', 'better']):
                entry2_is_good = True
    
    logger.info(f"🔍 Entry analysis parsing: Entry 1 bad={entry1_is_bad}, Entry 1 good={entry1_is_good}, Entry 2 good={entry2_is_good}")
    return entry1_is_bad, entry2_is_good


def validate_signal_with_ai(signal_data):
    """
    Validate trading signal using AI (Google Gemini API)
    
    Args:
        signal_data: Dictionary containing signal information
        
    Returns:
        dict: {
            'is_valid': bool,
            'confidence_score': float (0-100),
            'reasoning': str,
            'risk_level': str (LOW/MEDIUM/HIGH),
            'error': str (if validation failed)
        }
    """
    # Check if AI validation is enabled
    if not ENABLE_AI_VALIDATION:
        logger.debug("AI validation is disabled, skipping validation")
        return {
            'is_valid': True,
            'confidence_score': 100.0,
            'reasoning': 'AI validation disabled',
            'risk_level': 'UNKNOWN'
        }
    
    # Check if Gemini client is available
    if not gemini_client:
        logger.warning("Gemini client not available, proceeding without AI validation (fail-open)")
        return {
            'is_valid': True,
            'confidence_score': 100.0,
            'reasoning': 'AI validation unavailable, proceeding',
            'risk_level': 'UNKNOWN'
        }
    
    # Extract signal details
    symbol = format_symbol(signal_data.get('symbol', ''))
    signal_side = signal_data.get('signal_side', '').upper()
    entry_price = safe_float(signal_data.get('entry_price'), default=None)
    second_entry_price = safe_float(signal_data.get('second_entry_price'), default=None)
    stop_loss = safe_float(signal_data.get('stop_loss'), default=None)
    take_profit = safe_float(signal_data.get('take_profit'), default=None)
    timeframe = signal_data.get('timeframe', 'Unknown')
    quality_score = safe_float(signal_data.get('quality_score'), default=None)  # Script's quality score (6-17)
    
    # Extract indicator values from TradingView script (if provided)
    indicators = signal_data.get('indicators', {})
    
    # Check cache first
    cache_key = f"{symbol}_{signal_side}_{entry_price}_{stop_loss}_{take_profit}"
    current_time = time.time()
    if cache_key in validation_cache:
        cached_result, cache_time = validation_cache[cache_key]
        if current_time - cache_time < VALIDATION_CACHE_TTL:
            logger.debug(f"Using cached validation result for {symbol}")
            return cached_result
    
    # Validate required fields
    if not entry_price or entry_price <= 0:
        logger.warning(f"Cannot validate signal: invalid entry_price")
        return {
            'is_valid': True,  # Fail-open: proceed if we can't validate
            'confidence_score': 50.0,
            'reasoning': 'Invalid entry price, proceeding without validation',
            'risk_level': 'MEDIUM'
        }
    
    # Calculate risk/reward ratio
    risk_reward_ratio = None
    if stop_loss and take_profit:
        if signal_side == 'LONG':
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
        else:  # SHORT
            risk = abs(stop_loss - entry_price)
            reward = abs(entry_price - take_profit)
        
        if risk > 0:
            risk_reward_ratio = reward / risk
    
    # Fetch real-time market data for technical analysis
    market_data = {}
    try:
        if client:
            # Get current market price
            ticker = client.futures_symbol_ticker(symbol=symbol)
            current_price = float(ticker.get('price', 0))
            market_data['current_price'] = current_price
            
            # Calculate price distance from entry
            if current_price > 0:
                price_distance_pct = abs((entry_price - current_price) / current_price) * 100
                market_data['price_distance_pct'] = price_distance_pct
                market_data['entry_vs_current'] = 'ABOVE' if entry_price > current_price else 'BELOW'
            
            # Get recent candles for technical analysis (last 50 candles)
            # Map timeframe to Binance interval
            timeframe_map = {
                '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
                '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'
            }
            interval = timeframe_map.get(timeframe.lower(), '1h')  # Default to 1h
            
            try:
                klines = client.futures_klines(symbol=symbol, interval=interval, limit=100)  # Increased to 100 for better analysis
                if klines:
                    # Extract OHLCV data
                    opens = [float(k[1]) for k in klines]
                    highs = [float(k[2]) for k in klines]
                    lows = [float(k[3]) for k in klines]
                    closes = [float(k[4]) for k in klines]
                    volumes = [float(k[5]) for k in klines]
                    
                    # Store raw candle data for AI analysis
                    market_data['candle_count'] = len(klines)
                    market_data['recent_candles'] = {
                        'last_10_closes': closes[-10:] if len(closes) >= 10 else closes,
                        'last_10_highs': highs[-10:] if len(highs) >= 10 else highs,
                        'last_10_lows': lows[-10:] if len(lows) >= 10 else lows,
                        'last_10_volumes': volumes[-10:] if len(volumes) >= 10 else volumes
                    }
                    
                    # Fetch lower timeframe data for tighter TP/SL analysis (15m and 1h)
                    # This helps find the nearest realistic reversal points
                    lower_tf_data = {}
                    try:
                        # Get 15m data for near-term support/resistance
                        klines_15m = client.futures_klines(symbol=symbol, interval='15m', limit=50)
                        if klines_15m:
                            highs_15m = [float(k[2]) for k in klines_15m]
                            lows_15m = [float(k[3]) for k in klines_15m]
                            closes_15m = [float(k[4]) for k in klines_15m]
                            lower_tf_data['15m'] = {
                                'recent_high': max(highs_15m[-20:]) if len(highs_15m) >= 20 else max(highs_15m) if highs_15m else None,
                                'recent_low': min(lows_15m[-20:]) if len(lows_15m) >= 20 else min(lows_15m) if lows_15m else None,
                                'resistance_levels': sorted(set(highs_15m[-30:]), reverse=True)[:3] if len(highs_15m) >= 30 else [],
                                'support_levels': sorted(set(lows_15m[-30:]))[:3] if len(lows_15m) >= 30 else []
                            }
                    except Exception as e:
                        logger.debug(f"Could not fetch 15m data for {symbol}: {e}")
                    
                    try:
                        # Get 1h data for short-term support/resistance
                        klines_1h = client.futures_klines(symbol=symbol, interval='1h', limit=50)
                        if klines_1h:
                            highs_1h = [float(k[2]) for k in klines_1h]
                            lows_1h = [float(k[3]) for k in klines_1h]
                            closes_1h = [float(k[4]) for k in klines_1h]
                            lower_tf_data['1h'] = {
                                'recent_high': max(highs_1h[-20:]) if len(highs_1h) >= 20 else max(highs_1h) if highs_1h else None,
                                'recent_low': min(lows_1h[-20:]) if len(lows_1h) >= 20 else min(lows_1h) if lows_1h else None,
                                'resistance_levels': sorted(set(highs_1h[-30:]), reverse=True)[:3] if len(highs_1h) >= 30 else [],
                                'support_levels': sorted(set(lows_1h[-30:]))[:3] if len(lows_1h) >= 30 else []
                            }
                    except Exception as e:
                        logger.debug(f"Could not fetch 1h data for {symbol}: {e}")
                    
                    if lower_tf_data:
                        market_data['lower_timeframe_levels'] = lower_tf_data
                    
                    # Calculate technical indicators for AI's own analysis
                    # Trend analysis (multiple timeframes)
                    if len(closes) >= 5:
                        short_trend = ((closes[-1] - closes[-5]) / closes[-5]) * 100  # Last 5 candles
                        market_data['short_term_trend_pct'] = short_trend
                        market_data['short_term_direction'] = 'UP' if short_trend > 0.3 else 'DOWN' if short_trend < -0.3 else 'SIDEWAYS'
                    
                    if len(closes) >= 20:
                        medium_trend = ((closes[-1] - closes[-20]) / closes[-20]) * 100  # Last 20 candles
                        market_data['medium_term_trend_pct'] = medium_trend
                        market_data['medium_term_direction'] = 'UP' if medium_trend > 0.5 else 'DOWN' if medium_trend < -0.5 else 'SIDEWAYS'
                    
                    if len(closes) >= 50:
                        long_trend = ((closes[-1] - closes[-50]) / closes[-50]) * 100  # Last 50 candles
                        market_data['long_term_trend_pct'] = long_trend
                        market_data['long_term_direction'] = 'UP' if long_trend > 1 else 'DOWN' if long_trend < -1 else 'SIDEWAYS'
                    
                    # Moving averages (for AI's own trend analysis)
                    if len(closes) >= 20:
                        market_data['sma_20'] = sum(closes[-20:]) / 20
                    if len(closes) >= 50:
                        market_data['sma_50'] = sum(closes[-50:]) / 50
                    
                    # Support/Resistance levels (multiple levels)
                    if len(highs) >= 20:
                        recent_highs = highs[-20:]
                        recent_lows = lows[-20:]
                        market_data['resistance_level'] = max(recent_highs)
                        market_data['support_level'] = min(recent_lows)
                        market_data['recent_high'] = max(recent_highs)
                        market_data['recent_low'] = min(recent_lows)
                    
                    # Volume analysis (comprehensive)
                    if len(volumes) >= 20:
                        avg_volume = sum(volumes[-20:]) / 20
                    current_volume = volumes[-1] if volumes else 0
                    market_data['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1.0
                    market_data['volume_status'] = 'HIGH' if market_data['volume_ratio'] > 1.5 else 'NORMAL' if market_data['volume_ratio'] > 0.5 else 'LOW'
                    # Volume trend
                    recent_vol_avg = sum(volumes[-5:]) / 5
                    older_vol_avg = sum(volumes[-20:-5]) / 15 if len(volumes) >= 20 else avg_volume
                    market_data['volume_trend'] = 'INCREASING' if recent_vol_avg > older_vol_avg * 1.2 else 'DECREASING' if recent_vol_avg < older_vol_avg * 0.8 else 'STABLE'
                    
                    # Price position relative to recent range
                    if market_data.get('recent_high', 0) > market_data.get('recent_low', 0):
                        price_position = ((current_price - market_data['recent_low']) / 
                                         (market_data['recent_high'] - market_data['recent_low'])) * 100
                        market_data['price_position_in_range'] = price_position
                        if price_position > 80:
                            market_data['price_level'] = 'NEAR_RESISTANCE'
                        elif price_position < 20:
                            market_data['price_level'] = 'NEAR_SUPPORT'
                        else:
                            market_data['price_level'] = 'MID_RANGE'
                    
                    # Volatility (ATR-like calculation)
                    if len(highs) >= 14 and len(lows) >= 14:
                        true_ranges = [highs[i] - lows[i] for i in range(max(0, len(highs)-14), len(highs))]
                        avg_true_range = sum(true_ranges) / len(true_ranges) if true_ranges else 0
                        volatility_pct = (avg_true_range / current_price) * 100 if current_price > 0 else 0
                        market_data['volatility_pct'] = volatility_pct
                        market_data['volatility_status'] = 'HIGH' if volatility_pct > 2 else 'MODERATE' if volatility_pct > 1 else 'LOW'
                    
                    # Price momentum (rate of change)
                    if len(closes) >= 10:
                        momentum = ((closes[-1] - closes[-10]) / closes[-10]) * 100
                        market_data['momentum_pct'] = momentum
                        market_data['momentum_direction'] = 'BULLISH' if momentum > 0.5 else 'BEARISH' if momentum < -0.5 else 'NEUTRAL'
                    
                    # Price action patterns (simplified)
                    if len(closes) >= 3:
                        # Check for higher highs/lower lows pattern
                        if len(highs) >= 3 and len(lows) >= 3:
                            recent_highs_3 = highs[-3:]
                            recent_lows_3 = lows[-3:]
                            if recent_highs_3[-1] > recent_highs_3[0] and recent_lows_3[-1] > recent_lows_3[0]:
                                market_data['price_pattern'] = 'HIGHER_HIGHS_HIGHER_LOWS'  # Bullish
                            elif recent_highs_3[-1] < recent_highs_3[0] and recent_lows_3[-1] < recent_lows_3[0]:
                                market_data['price_pattern'] = 'LOWER_HIGHS_LOWER_LOWS'  # Bearish
                            else:
                                market_data['price_pattern'] = 'MIXED'
                    
                    # Legacy fields for backward compatibility
                    recent_closes = closes[-20:] if len(closes) >= 20 else closes
                    if len(recent_closes) >= 2:
                        price_change = ((recent_closes[-1] - recent_closes[0]) / recent_closes[0]) * 100
                        market_data['recent_trend_pct'] = price_change
                        market_data['trend_direction'] = 'UP' if price_change > 0.5 else 'DOWN' if price_change < -0.5 else 'SIDEWAYS'
                    
            except Exception as e:
                logger.debug(f"Could not fetch klines for {symbol}: {e}")
                market_data['klines_error'] = str(e)
            
            # Fetch funding rate data (critical for signal validation)
            try:
                # Get current funding rate
                funding_info = client.futures_funding_rate(symbol=symbol, limit=1)
                if funding_info and len(funding_info) > 0:
                    current_funding = float(funding_info[0].get('fundingRate', 0))
                    market_data['funding_rate'] = current_funding
                    market_data['funding_rate_pct'] = current_funding * 100  # Convert to percentage
                    
                    # Get historical funding rates (last 24 periods = 3 days to analyze frequency)
                    historical_funding = client.futures_funding_rate(symbol=symbol, limit=24)
                    if historical_funding and len(historical_funding) > 0:
                        funding_rates = [float(f.get('fundingRate', 0)) for f in historical_funding]
                        funding_timestamps = [int(f.get('fundingTime', 0)) for f in historical_funding]
                        avg_funding = sum(funding_rates) / len(funding_rates) if funding_rates else 0
                        max_funding = max(funding_rates) if funding_rates else 0
                        min_funding = min(funding_rates) if funding_rates else 0
                        
                        market_data['funding_rate_24h_avg'] = avg_funding
                        market_data['funding_rate_24h_avg_pct'] = avg_funding * 100
                        market_data['funding_rate_24h_max'] = max_funding
                        market_data['funding_rate_24h_max_pct'] = max_funding * 100
                        market_data['funding_rate_24h_min'] = min_funding
                        market_data['funding_rate_24h_min_pct'] = min_funding * 100
                        
                        # Analyze funding rate frequency and consistency
                        # Check if funding is consistently high (accumulating) vs occasional
                        # Count how many periods have significant funding (same direction as current)
                        if current_funding > 0:
                            # For positive funding, count how many periods are positive
                            significant_periods = sum(1 for fr in funding_rates if fr > 0.0001)  # >0.01% per 8h
                            consistent_positive = significant_periods >= len(funding_rates) * 0.6  # 60%+ periods positive
                        elif current_funding < 0:
                            # For negative funding, count how many periods are negative
                            significant_periods = sum(1 for fr in funding_rates if fr < -0.0001)  # <-0.01% per 8h
                            consistent_negative = significant_periods >= len(funding_rates) * 0.6  # 60%+ periods negative
                        else:
                            consistent_positive = False
                            consistent_negative = False
                        
                        # Calculate funding consistency (how often it's in the same direction)
                        if current_funding > 0:
                            same_direction_count = sum(1 for fr in funding_rates if fr > 0)
                            funding_consistency = same_direction_count / len(funding_rates) if funding_rates else 0
                            market_data['funding_consistency'] = funding_consistency
                            market_data['funding_is_accumulating'] = funding_consistency >= 0.6  # 60%+ in same direction
                        elif current_funding < 0:
                            same_direction_count = sum(1 for fr in funding_rates if fr < 0)
                            funding_consistency = same_direction_count / len(funding_rates) if funding_rates else 0
                            market_data['funding_consistency'] = funding_consistency
                            market_data['funding_is_accumulating'] = funding_consistency >= 0.6  # 60%+ in same direction
                        else:
                            market_data['funding_consistency'] = 0
                            market_data['funding_is_accumulating'] = False
                        
                        # Calculate estimated daily funding cost (if position held for 24h)
                        # Funding happens every 8 hours, so 3 times per day
                        # If funding is consistent, multiply by 3 for daily cost estimate
                        if market_data.get('funding_is_accumulating', False):
                            # If accumulating, estimate higher cost (use current funding)
                            estimated_daily_funding = abs(current_funding) * 3  # 3 funding periods per day
                        else:
                            # If occasional, use average (less frequent, lower cost)
                            estimated_daily_funding = abs(avg_funding) * 3
                        
                        market_data['estimated_daily_funding'] = estimated_daily_funding
                        market_data['estimated_daily_funding_pct'] = estimated_daily_funding * 100  # Convert to percentage
                        
                        # Determine funding rate status with frequency consideration
                        # Extreme thresholds: >0.1% (0.001) = very positive, <-0.1% = very negative
                        if current_funding > 0.001:  # >0.1% per 8h = very positive
                            market_data['funding_rate_status'] = 'EXTREMELY_POSITIVE'
                            market_data['funding_rate_risk'] = 'HIGH'  # Market heavily longed, reversal risk
                        elif current_funding > 0.0005:  # >0.05% per 8h = positive
                            market_data['funding_rate_status'] = 'POSITIVE'
                            market_data['funding_rate_risk'] = 'MODERATE'
                        elif current_funding < -0.001:  # <-0.1% per 8h = very negative
                            market_data['funding_rate_status'] = 'EXTREMELY_NEGATIVE'
                            market_data['funding_rate_risk'] = 'HIGH'  # Market heavily shorted, reversal risk
                        elif current_funding < -0.0005:  # <-0.05% per 8h = negative
                            market_data['funding_rate_status'] = 'NEGATIVE'
                            market_data['funding_rate_risk'] = 'MODERATE'
                        else:  # Between -0.05% and 0.05%
                            market_data['funding_rate_status'] = 'NEUTRAL'
                            market_data['funding_rate_risk'] = 'LOW'
                        
                        # Funding rate trend (increasing/decreasing)
                        if len(funding_rates) >= 2:
                            recent_avg = sum(funding_rates[:3]) / min(3, len(funding_rates))
                            older_avg = sum(funding_rates[3:]) / max(1, len(funding_rates) - 3)
                            if recent_avg > older_avg * 1.2:
                                market_data['funding_rate_trend'] = 'INCREASING'
                            elif recent_avg < older_avg * 0.8:
                                market_data['funding_rate_trend'] = 'DECREASING'
                            else:
                                market_data['funding_rate_trend'] = 'STABLE'
                    
                    logger.debug(f"📊 Funding rate for {symbol}: {current_funding*100:.4f}% (Status: {market_data.get('funding_rate_status', 'UNKNOWN')})")
            except Exception as e:
                logger.debug(f"Could not fetch funding rate for {symbol}: {e}")
                market_data['funding_rate_error'] = str(e)
    except Exception as e:
        logger.debug(f"Could not fetch market data for {symbol}: {e}")
        market_data['error'] = str(e)
    
    # Build indicator values section for AI prompt
    indicator_info = ""
    if indicators:
        rsi_val = safe_float(indicators.get('rsi'), default=None)
        macd_line = safe_float(indicators.get('macd_line'), default=None)
        macd_signal = safe_float(indicators.get('macd_signal'), default=None)
        macd_hist = safe_float(indicators.get('macd_histogram'), default=None)
        stoch_k = safe_float(indicators.get('stoch_k'), default=None)
        stoch_d = safe_float(indicators.get('stoch_d'), default=None)
        ema200_val = safe_float(indicators.get('ema200'), default=None)
        atr_val = safe_float(indicators.get('atr'), default=None)
        bb_upper = safe_float(indicators.get('bb_upper'), default=None)
        bb_basis = safe_float(indicators.get('bb_basis'), default=None)
        bb_lower = safe_float(indicators.get('bb_lower'), default=None)
        smv_norm = safe_float(indicators.get('smv_normalized'), default=None)
        cum_smv = safe_float(indicators.get('cum_smv'), default=None)
        supertrend_val = safe_float(indicators.get('supertrend'), default=None)
        supertrend_bull = indicators.get('supertrend_bull', False)
        obv_val = safe_float(indicators.get('obv'), default=None)
        rel_vol_pct = safe_float(indicators.get('relative_volume_percentile'), default=None)
        mfi_val = safe_float(indicators.get('mfi'), default=None)
        vol_ratio = safe_float(indicators.get('volume_ratio'), default=None)
        has_bull_div = indicators.get('has_bullish_divergence', False)
        has_bear_div = indicators.get('has_bearish_divergence', False)
        at_bottom = indicators.get('at_bottom', False)
        at_top = indicators.get('at_top', False)
        smart_money_buy = indicators.get('smart_money_buying', False)
        smart_money_sell = indicators.get('smart_money_selling', False)
        price_above_ema200 = indicators.get('price_above_ema200', False)
        price_below_ema200 = indicators.get('price_below_ema200', False)
        
        indicator_info = f"""
TRADINGVIEW INDICATOR VALUES (from your script):
- RSI: {(f'{rsi_val:.2f}' if rsi_val is not None else 'N/A')} (Oversold: <30, Overbought: >85)
- MACD Line: {(f'{macd_line:.4f}' if macd_line is not None else 'N/A')}
- MACD Signal: {(f'{macd_signal:.4f}' if macd_signal is not None else 'N/A')}
- MACD Histogram: {(f'{macd_hist:.4f}' if macd_hist is not None else 'N/A')} (Positive = bullish momentum)
- Stochastic K: {(f'{stoch_k:.2f}' if stoch_k is not None else 'N/A')} (Oversold: <20, Overbought: >80)
- Stochastic D: {(f'{stoch_d:.2f}' if stoch_d is not None else 'N/A')}
- EMA 200: ${(f'{ema200_val:,.8f}' if ema200_val is not None else 'N/A')} (Price {'ABOVE' if price_above_ema200 else 'BELOW'} EMA200 = {'BULLISH' if price_above_ema200 else 'BEARISH'} trend)
- ATR: {(f'{atr_val:.8f}' if atr_val is not None else 'N/A')} (Volatility measure)
- Bollinger Bands: Upper=${(f'{bb_upper:,.8f}' if bb_upper is not None else 'N/A')}, Basis=${(f'{bb_basis:,.8f}' if bb_basis is not None else 'N/A')}, Lower=${(f'{bb_lower:,.8f}' if bb_lower is not None else 'N/A')}
- Smart Money Volume (Normalized): {(f'{smv_norm:.2f}' if smv_norm is not None else 'N/A')} (Positive = buying pressure)
- Cumulative SMV: {(f'{cum_smv:.2f}' if cum_smv is not None else 'N/A')} ({'BUYING' if smart_money_buy else 'SELLING' if smart_money_sell else 'NEUTRAL'} pressure)
- Supertrend: ${(f'{supertrend_val:,.8f}' if supertrend_val is not None else 'N/A')} ({'BULLISH' if supertrend_bull else 'BEARISH'})
- OBV: {(f'{obv_val:.2f}' if obv_val is not None else 'N/A')} (Rising = buying pressure)
- Relative Volume Percentile: {(f'{rel_vol_pct:.1f}' if rel_vol_pct is not None else 'N/A')}% (High: >70%, Low: <30%)
- MFI (Money Flow Index): {(f'{mfi_val:.2f}' if mfi_val is not None else 'N/A')} (Oversold: <20, Overbought: >80)
- Volume Ratio: {(f'{vol_ratio:.2f}' if vol_ratio is not None else 'N/A')}x (vs average)
- Bullish Divergence: {'YES ✅' if has_bull_div else 'NO'}
- Bearish Divergence: {'YES ✅' if has_bear_div else 'NO'}
- At Bottom/Top: {'BOTTOM ✅' if at_bottom else 'TOP ✅' if at_top else 'MID-RANGE'}"""
    
    # Build prompt for AI - Enhanced with real market data AND indicator values for technical analysis
    market_info = ""
    if market_data.get('current_price'):
        # Build funding rate warning message (combines magnitude + frequency + timeframe)
        funding_warning = ""
        if market_data.get('funding_rate') is not None:
            current_funding = market_data.get('funding_rate', 0)
            is_accumulating = market_data.get('funding_is_accumulating', False)
            is_1h = timeframe and '1h' in timeframe.lower()
            estimated_daily = market_data.get('estimated_daily_funding_pct', 0)  # Already in percentage
            
            if signal_side == 'SHORT' and current_funding < 0:
                if is_accumulating:
                    if is_1h:
                        funding_warning = f"🚨 CRITICAL RED FLAG: SHORT signal (1H) with NEGATIVE funding (ACCUMULATING) = Market already shorted + HIGH COST ({estimated_daily:.4f}% daily) = STRONG REJECTION SIGNAL"
                    else:
                        funding_warning = f"🚨 MAJOR RED FLAG: SHORT signal with NEGATIVE funding (ACCUMULATING) = Market already shorted + HIGH COST ({estimated_daily:.4f}% daily) = LOWER CONFIDENCE or REJECT"
                else:
                    # Occasional funding - might be acceptable with strong factors
                    if abs(current_funding) < 0.0003:  # <0.03% per 8h
                        funding_warning = f"⚠️ MODERATE CONCERN: SHORT signal with NEGATIVE funding (OCCASIONAL, small) = Can pass with STRONG other factors (8+ indicators, perfect structure)"
                    else:
                        funding_warning = f"⚠️ CONCERN: SHORT signal with NEGATIVE funding (OCCASIONAL) = Lower confidence but can pass with strong factors"
            elif signal_side == 'LONG' and current_funding > 0:
                if is_accumulating:
                    if is_1h:
                        funding_warning = f"🚨 CRITICAL RED FLAG: LONG signal (1H) with POSITIVE funding (ACCUMULATING) = Market already longed + HIGH COST ({estimated_daily:.4f}% daily) = STRONG REJECTION SIGNAL"
                    else:
                        funding_warning = f"🚨 MAJOR RED FLAG: LONG signal with POSITIVE funding (ACCUMULATING) = Market already longed + HIGH COST ({estimated_daily:.4f}% daily) = LOWER CONFIDENCE or REJECT"
                else:
                    # Occasional funding - might be acceptable with strong factors
                    if abs(current_funding) < 0.0003:  # <0.03% per 8h
                        funding_warning = f"⚠️ MODERATE CONCERN: LONG signal with POSITIVE funding (OCCASIONAL, small) = Can pass with STRONG other factors (8+ indicators, perfect structure)"
                    else:
                        funding_warning = f"⚠️ CONCERN: LONG signal with POSITIVE funding (OCCASIONAL) = Lower confidence but can pass with strong factors"
            else:
                funding_warning = "✅ FAVORABLE: Funding rate aligns with signal direction"
        else:
            funding_warning = "⚠️ Funding rate data unavailable"
        
        # Determine funding rate direction
        funding_direction = "N/A"
        if market_data.get('funding_rate') is not None:
            current_funding = market_data.get('funding_rate', 0)
            if current_funding < 0:
                funding_direction = "NEGATIVE"
            elif current_funding > 0:
                funding_direction = "POSITIVE"
            else:
                funding_direction = "NEUTRAL"
        
        market_info = f"""
═══════════════════════════════════════════════════════════════
REAL-TIME MARKET DATA (from Binance API) - FOR YOUR INDEPENDENT ANALYSIS:
═══════════════════════════════════════════════════════════════
CURRENT MARKET CONDITIONS:
- Current Market Price: ${market_data['current_price']:,.8f}
- Entry Price (from signal): ${entry_price:,.8f}
- Entry Price vs Current: Entry is {market_data.get('entry_vs_current', 'N/A')} current price
- Price Distance: {market_data.get('price_distance_pct', 0):.2f}% from current price

TREND ANALYSIS (Multiple Timeframes):
- Short-term Trend (last 5 candles): {market_data.get('short_term_direction', market_data.get('trend_direction', 'N/A'))} ({market_data.get('short_term_trend_pct', market_data.get('recent_trend_pct', 0)):+.2f}%)
- Medium-term Trend (last 20 candles): {market_data.get('medium_term_direction', market_data.get('trend_direction', 'N/A'))} ({market_data.get('medium_term_trend_pct', market_data.get('recent_trend_pct', 0)):+.2f}%)
- Long-term Trend (last 50 candles): {market_data.get('long_term_direction', market_data.get('trend_direction', 'N/A'))} ({market_data.get('long_term_trend_pct', market_data.get('recent_trend_pct', 0)):+.2f}%)
- Overall Trend ({timeframe}): {market_data.get('trend_direction', 'N/A')} ({market_data.get('recent_trend_pct', 0):+.2f}%)

MOVING AVERAGES:
- SMA 20: ${(f"{market_data.get('sma_20'):,.8f}" if market_data.get('sma_20') is not None else 'N/A')}
- SMA 50: ${(f"{market_data.get('sma_50'):,.8f}" if market_data.get('sma_50') is not None else 'N/A')}
- Price vs SMA 20: {'ABOVE' if market_data.get('sma_20') and market_data['current_price'] > market_data['sma_20'] else 'BELOW' if market_data.get('sma_20') else 'N/A'}
- Price vs SMA 50: {'ABOVE' if market_data.get('sma_50') and market_data['current_price'] > market_data['sma_50'] else 'BELOW' if market_data.get('sma_50') else 'N/A'}

SUPPORT & RESISTANCE LEVELS:
- Resistance Level: ${market_data.get('resistance_level', market_data.get('recent_high', 0)):,.8f}
- Support Level: ${market_data.get('support_level', market_data.get('recent_low', 0)):,.8f}
- Price Range (last 20 candles): ${market_data.get('recent_low', 0):,.8f} - ${market_data.get('recent_high', 0):,.8f}
- Current Price Position: {market_data.get('price_level', 'N/A')} ({market_data.get('price_position_in_range', 0):.1f}% of range)

VOLUME ANALYSIS:
- Volume Status: {market_data.get('volume_status', 'N/A')} (current/avg ratio: {market_data.get('volume_ratio', 1):.2f}x)
- Volume Trend: {market_data.get('volume_trend', 'N/A')} (increasing/decreasing/stable)

MOMENTUM & VOLATILITY:
- Price Momentum: {market_data.get('momentum_direction', 'N/A')} ({market_data.get('momentum_pct', 0):+.2f}%)
- Volatility: {market_data.get('volatility_status', 'N/A')} ({market_data.get('volatility_pct', 0):.2f}%)

FUNDING RATE ANALYSIS (CRITICAL FOR SIGNAL VALIDATION - PRIMARY REJECTION FACTOR):
⚠️ CURRENT FUNDING RATE + FREQUENCY + TIMEFRAME = COMBINED DECISION ⚠️
- Current Funding Rate: {(f"{market_data.get('funding_rate_pct', 0):.4f}%" if market_data.get('funding_rate') is not None else 'N/A')} per 8h
- Funding Rate Status: {market_data.get('funding_rate_status', 'N/A')}
- Funding Rate Risk: {market_data.get('funding_rate_risk', 'N/A')}
- Funding Rate Trend: {market_data.get('funding_rate_trend', 'N/A')}
- Funding Consistency: {(f"{market_data.get('funding_consistency', 0)*100:.1f}%" if market_data.get('funding_consistency') is not None else 'N/A')} (how often in same direction)
- Funding Is Accumulating: {'YES - Consistent funding (HIGH COST RISK)' if market_data.get('funding_is_accumulating', False) else 'NO - Occasional funding (lower cost)' if market_data.get('funding_is_accumulating') is not None else 'N/A'}
- Estimated Daily Funding Cost: {(f"{market_data.get('estimated_daily_funding_pct', 0):.4f}%" if market_data.get('estimated_daily_funding_pct') is not None else 'N/A')} (if position held 24h)
- 24h Average Funding: {(f"{market_data.get('funding_rate_24h_avg_pct', 0):.4f}%" if market_data.get('funding_rate_24h_avg') is not None else 'N/A')}
- 24h Max Funding: {(f"{market_data.get('funding_rate_24h_max_pct', 0):.4f}%" if market_data.get('funding_rate_24h_max') is not None else 'N/A')}
- 24h Min Funding: {(f"{market_data.get('funding_rate_24h_min_pct', 0):.4f}%" if market_data.get('funding_rate_24h_min') is not None else 'N/A')}
- Signal Timeframe: {timeframe} {'⚠️ 1H TIMEFRAME - FUNDING VALIDATION IS EVEN MORE CRITICAL' if timeframe and '1h' in timeframe.lower() else ''}
- ⚠️ CRITICAL VALIDATION: For {signal_side} signal, CURRENT funding rate is {funding_direction}
- ⚠️ {funding_warning}

PRICE ACTION PATTERNS:
- Pattern: {market_data.get('price_pattern', 'N/A')} (Higher Highs/Higher Lows = Bullish, Lower Highs/Lower Lows = Bearish)

═══════════════════════════════════════════════════════════════"""
    
    # Check if this is Entry 2 standalone validation
    is_entry2_standalone = signal_data.get('_entry2_standalone_validation', False)
    original_rejection_reason = signal_data.get('_original_rejection_reason', '')
    
    # Add special header for Entry 2 standalone validation
    entry2_standalone_header = ""
    if is_entry2_standalone:
        entry2_standalone_header = f"""
═══════════════════════════════════════════════════════════════
⚠️ CRITICAL: ENTRY 2 STANDALONE VALIDATION ⚠️
═══════════════════════════════════════════════════════════════

ENTRY 1 WAS REJECTED - You are now validating Entry 2 as a STANDALONE TRADE.

Original Entry 1 Rejection Reason:
{original_rejection_reason}

CURRENT TASK: Evaluate Entry 2 (${entry_price:,.8f}) as a STANDALONE trade opportunity.
- Entry 1 is NOT being used - this is Entry 2 ONLY
- You must evaluate if Entry 2 alone is worth trading with $20 position size
- Entry 2 will use a custom TP of 4-5% from entry (not the original TP)
- This is a RARE case - only approve if Entry 2 is STRONG enough to trade alone

CRITICAL QUESTIONS TO ANSWER:
1. Is Entry 2 at an optimal institutional liquidity zone?
2. Does Entry 2 have strong technical support (support/resistance, indicators)?
3. Is Entry 2's Risk/Reward acceptable for a standalone trade?
4. Would you trade Entry 2 alone if Entry 1 didn't exist?

DECISION CRITERIA:
- APPROVE Entry 2 standalone if: Strong technical setup, good R/R, optimal entry zone
- REJECT Entry 2 standalone if: Weak setup, not at optimal zone, or Entry 1 rejection reasons also apply to Entry 2
- Note: R/R ratio is NOT a rejection factor - focus on indicators and structure

═══════════════════════════════════════════════════════════════
"""

    prompt = f"""{entry2_standalone_header}You are an INSTITUTIONAL CRYPTO TRADER and WHALE with 20+ years of elite professional trading experience.
You have achieved consistent 200% monthly returns (2x per month) through:
- Deep understanding of crypto market structure, order flow, and institutional/whale behavior
- Mastery of multi-timeframe analysis (HTF → LTF) and market structure (BOS, CHoCH)
- Ability to identify liquidity pools, stop-hunt zones, order blocks, and fair value gaps
- Understanding of volume & open interest behavior, funding rates, and market microstructure
- Experience trading with institutional capital - you think like a whale, not retail
- Mastery of risk management with minimum RR ≥ 1:3 for all trades
- Ability to identify high-probability institutional liquidity zones for entries
- Institutional-level understanding of market manipulation, smart money flow, and retail vs. professional behavior

YOUR TRADING PHILOSOPHY (INSTITUTIONAL/WHALE PERSPECTIVE):
- You think like a whale/institution, not retail - you see what institutions see
- You understand market structure, liquidity, and order flow at the deepest level
- You know that entries must be at institutional liquidity zones, not arbitrary price levels
- You understand that markets move based on liquidity grabs, stop hunts, and order blocks
- You combine multi-timeframe analysis with market structure for superior edge
- You trust your institutional-level analysis but verify with multiple confirmations
- You've seen every market scenario - bull markets, bear markets, crashes, pumps, liquidations
- You only take trades with minimum RR ≥ 1:3 - anything less is retail behavior

YOUR ANALYSIS APPROACH (INSTITUTIONAL METHODOLOGY):
1. FIRST: Multi-timeframe analysis (HTF → LTF) - understand the dominant trend and market structure
2. SECOND: Market structure analysis (BOS, CHoCH) - identify break of structure and change of character
3. THIRD: Liquidity analysis - identify liquidity pools, stop-hunt zones, and institutional entry zones
4. FOURTH: Order flow analysis - identify order blocks, fair value gaps, and optimal entry points
5. FIFTH: Volume & OI analysis - understand institutional accumulation/distribution and funding rates
6. SIXTH: Validate signal against YOUR institutional analysis - does it align with whale behavior?
7. SEVENTH: Optimize entries ONLY at high-probability institutional liquidity zones
8. EIGHTH: Combine YOUR institutional analysis with TradingView indicators - both must align

IMPORTANT CONTEXT:
- This signal comes from a TradingView indicator that already filters signals (65% win rate, quality score ≥ 7)
- Your job is to PERFORM TECHNICAL ANALYSIS and OPTIMIZE PRICES, not just validate
- You have REAL-TIME market data with support/resistance levels - USE IT for technical analysis
- You MUST analyze Entry 1, Entry 2, Stop Loss, and Take Profit prices using the market data provided
- Check if prices are at optimal support/resistance levels from market_data
- If prices are not optimal, suggest better levels based on technical analysis
- Focus on TradingView indicators (PRIMARY - 70%) and technical market analysis (SECONDARY - 30%)
- Risk/Reward ratio is NOT a primary rejection factor - focus on indicators and market structure instead
- Only reject if MULTIPLE red flags: weak indicators AND strong counter-trend AND unfavorable market structure

MANDATORY ANALYSIS REQUIREMENTS:
1. You MUST analyze Entry 1 price: Is it at support/resistance from market_data? If not, suggest optimal level
2. You MUST analyze Entry 2 price: Is it at support/resistance? If not, suggest optimal level  
3. You MUST analyze Stop Loss: Check lower timeframe support/resistance from market_data['lower_timeframe_levels']
4. You MUST analyze Take Profit: Check lower timeframe support/resistance levels
5. You MUST calculate actual Risk/Reward ratio and validate it
6. Your reasoning MUST show this analysis - do NOT just count indicators

═══════════════════════════════════════════════════════════════
STEP 1: INSTITUTIONAL MULTI-TIMEFRAME & MARKET STRUCTURE ANALYSIS
═══════════════════════════════════════════════════════════════

Before looking at the signal, analyze the market using YOUR INSTITUTIONAL/WHALE METHODOLOGY.
Think like the whale/institution you are - see liquidity, order flow, and market structure.

1. MULTI-TIMEFRAME ANALYSIS (HTF → LTF - Institutional Approach):
   - HIGHER TIMEFRAME (HTF) - 4H/1D: What is the DOMINANT TREND? (This is your primary filter)
   - CURRENT TIMEFRAME ({timeframe}): What is the INTERMEDIATE TREND? (Does it align with HTF?)
   - LOWER TIMEFRAME (LTF) - 15m/5m: What is the SHORT-TERM MOMENTUM? (For entry timing)
   - TREND ALIGNMENT: Are all timeframes aligned? (HTF + CTF + LTF = HIGH PROBABILITY)
   - TREND CONFLICT: If timeframes conflict, is this a REVERSAL or CORRECTION? (Evaluate carefully)
   - Moving Averages: Are they aligned across timeframes? (Bullish alignment = price > SMA20 > SMA50)
   - Price vs MAs: Is price respecting or rejecting key levels? (Institutional behavior)

2. MARKET STRUCTURE ANALYSIS (BOS, CHoCH - Institutional Methodology):
   - BREAK OF STRUCTURE (BOS): Has there been a BOS? (Bullish BOS = higher high breaks previous high)
   - CHANGE OF CHARACTER (CHoCH): Has there been a CHoCH? (Bullish CHoCH = higher low after lower low)
   - MARKET STRUCTURE: Is structure BULLISH, BEARISH, or NEUTRAL? (This determines bias)
   - STRUCTURE BREAKS: Are we in a structure break or consolidation? (Structure breaks = high probability)
   - SWING POINTS: Identify key swing highs/lows (these are institutional levels)
   - STRUCTURE ALIGNMENT: Does the signal align with market structure? (Must align for approval)

3. LIQUIDITY ANALYSIS (Institutional/Whale Perspective):
   - LIQUIDITY POOLS: Where are the liquidity zones? (Above resistance for longs, below support for shorts)
   - STOP-HUNT ZONES: Where are retail stops likely placed? (Institutions hunt these)
   - LIQUIDITY GRABS: Has price grabbed liquidity? (Liquidity grab = potential reversal)
   - INSTITUTIONAL ENTRY ZONES: Where would institutions enter? (At liquidity zones, not random prices)
   - ORDER BLOCKS: Are there order blocks (institutional entry zones) near the signal?
   - FAIR VALUE GAPS (FVG): Are there FVGs that need to be filled? (These are entry zones)

4. SUPPORT & RESISTANCE (Institutional Level Identification):
   - REAL SUPPORT: Where will INSTITUTIONAL buyers step in? (Not just recent lows)
   - REAL RESISTANCE: Where will INSTITUTIONAL sellers step in? (Not just recent highs)
   - ORDER BLOCKS: Identify bullish/bearish order blocks (institutional entry zones)
   - LIQUIDITY LEVELS: Identify liquidity pools above/below key levels
   - PSYCHOLOGICAL LEVELS: Round numbers, previous swing points (institutions use these)
   - IS PRICE AT INSTITUTIONAL ZONE? (Order block, liquidity pool, FVG = high probability entry)

5. VOLUME & OPEN INTEREST ANALYSIS (Institutional Order Flow):
   - VOLUME PROFILE: Is volume INCREASING on moves in trend direction? (Institutional accumulation)
   - VOLUME DIVERGENCE: Is volume CONFIRMING or DIVERGING from price? (Divergence = warning)
   - OPEN INTEREST (OI): Is OI increasing or decreasing? (Increasing OI = institutional interest)
   - SMART MONEY: Are institutions buying or selling? (Use Smart Money indicators)
   - ACCUMULATION/DISTRIBUTION: Are institutions accumulating or distributing? (This determines direction)

6. FUNDING RATE ANALYSIS (CRITICAL FOR SIGNAL VALIDATION - PRIMARY REJECTION FACTOR):
   ⚠️ THIS IS A PRIMARY CONCERN - FUNDING RATE CAN REJECT SIGNALS ⚠️
   
   - CURRENT FUNDING RATE: Check the CURRENT funding rate from market_data (this is the most important)
   - FUNDING RATE STATUS: Is it EXTREMELY_POSITIVE, POSITIVE, NEUTRAL, NEGATIVE, or EXTREMELY_NEGATIVE?
   - FUNDING FREQUENCY: Check if funding is ACCUMULATING (consistent, every period) or OCCASIONAL (infrequent)
   - FUNDING IS ACCUMULATING: If funding_is_accumulating = True, funding is consistent (high cost risk)
   - ESTIMATED DAILY COST: Check estimated_daily_funding_pct (if high, you'll pay a lot)
   
   CRITICAL VALIDATION RULES (COMBINE TIMEFRAME + FUNDING MAGNITUDE + FREQUENCY):
   
   * For SHORT signals:
     - If CURRENT funding rate is NEGATIVE (shorts paying longs):
       → Check if funding_is_accumulating (consistent negative funding)
       → If ACCUMULATING (consistent): Market is ALREADY heavily shorted = HIGH SQUEEZE RISK + HIGH COST
         → This is a MAJOR RED FLAG - LOWER CONFIDENCE SIGNIFICANTLY or REJECT
         → For 1H timeframe: EVEN MORE CRITICAL - STRONG REJECTION SIGNAL
         → Example: SHORT signal (1H) with -0.05% funding (accumulating) = REJECT or CONFIDENCE -25%
       → If OCCASIONAL (small, infrequent): Might be acceptable with other STRONG factors
         → Check estimated_daily_funding_pct - if <0.1% daily, might pass with strong indicators
         → Example: SHORT signal with -0.02% funding (occasional, not accumulating) = Lower confidence -10% but can pass
     - If CURRENT funding rate is EXTREMELY_NEGATIVE (<-0.1%): 
       → Market is EXTREMELY shorted = VERY HIGH SQUEEZE RISK
       → This is a CRITICAL RED FLAG - STRONGLY CONSIDER REJECTION (regardless of frequency)
       → Example: SHORT signal with -0.15% funding = REJECT (unless other factors are extremely strong)
     - If CURRENT funding rate is POSITIVE or EXTREMELY_POSITIVE: 
       → FAVORABLE for SHORT (longs paying shorts) = Market not yet shorted = GOOD
   
   * For LONG signals:
     - If CURRENT funding rate is POSITIVE (longs paying shorts):
       → Check if funding_is_accumulating (consistent positive funding)
       → If ACCUMULATING (consistent): Market is ALREADY heavily longed = HIGH REVERSAL RISK + HIGH COST
         → This is a MAJOR RED FLAG - LOWER CONFIDENCE SIGNIFICANTLY or REJECT
         → For 1H timeframe: EVEN MORE CRITICAL - STRONG REJECTION SIGNAL
         → Example: LONG signal (1H) with +0.05% funding (accumulating) = REJECT or CONFIDENCE -25%
       → If OCCASIONAL (small, infrequent): Might be acceptable with other STRONG factors
         → Check estimated_daily_funding_pct - if <0.1% daily, might pass with strong indicators
         → Example: LONG signal with +0.02% funding (occasional, not accumulating) = Lower confidence -10% but can pass
     - If CURRENT funding rate is EXTREMELY_POSITIVE (>0.1%): 
       → Market is EXTREMELY longed = VERY HIGH REVERSAL RISK
       → This is a CRITICAL RED FLAG - STRONGLY CONSIDER REJECTION (regardless of frequency)
       → Example: LONG signal with +0.15% funding = REJECT (unless other factors are extremely strong)
     - If CURRENT funding rate is NEGATIVE or EXTREMELY_NEGATIVE: 
       → FAVORABLE for LONG (shorts paying longs) = Market not yet longed = GOOD
   
   TIMEFRAME + FUNDING COMBINATION (CRITICAL):
   - For 1H timeframe signals: Funding rate validation is EVEN MORE CRITICAL
   - 1H + Contradictory funding + Accumulating = STRONG REJECTION SIGNAL
   - 1H + Contradictory funding + Occasional (small) = Can pass with STRONG other factors (8+ indicators, perfect structure)
   - For longer timeframes (4H, 1D): Still important but slightly less critical
   - 4H/1D + Contradictory funding + Accumulating = Moderate concern, lower confidence
   - 4H/1D + Contradictory funding + Occasional = Minor concern, small confidence reduction
   
   CONFIDENCE ADJUSTMENT (COMBINED FACTORS):
   - SHORT signal (1H) + NEGATIVE funding + ACCUMULATING: CONFIDENCE -25% to -35% (or REJECT)
   - LONG signal (1H) + POSITIVE funding + ACCUMULATING: CONFIDENCE -25% to -35% (or REJECT)
   - SHORT signal (1H) + NEGATIVE funding + OCCASIONAL (small): CONFIDENCE -10% to -15% (can pass with strong factors)
   - LONG signal (1H) + POSITIVE funding + OCCASIONAL (small): CONFIDENCE -10% to -15% (can pass with strong factors)
   - SHORT signal (4H/1D) + NEGATIVE funding + ACCUMULATING: CONFIDENCE -15% to -20%
   - LONG signal (4H/1D) + POSITIVE funding + ACCUMULATING: CONFIDENCE -15% to -20%
   - EXTREME funding (very negative/positive): CONFIDENCE -30% to -40% (or REJECT)
   
   DECISION LOGIC (PRIORITY ORDER):
   1. If funding is EXTREME (>0.1% or <-0.1%) AND contradicts signal: STRONG REJECTION (unless all factors extremely strong)
   2. If funding contradicts signal + ACCUMULATING + 1H timeframe: STRONG REJECTION or CONFIDENCE -25%
   3. If funding contradicts signal + ACCUMULATING + 4H/1D timeframe: Lower confidence -15% to -20%
   4. If funding contradicts signal + OCCASIONAL (small) + 1H timeframe: Lower confidence -10% to -15% (can pass with strong factors)
   5. If funding contradicts signal + OCCASIONAL (small) + 4H/1D timeframe: Lower confidence -5% to -10% (can pass)
   6. If funding aligns with signal direction: FAVORABLE (no penalty, may even boost confidence)
   
   MAIN CONCERN: Don't want to pay too much in funding costs
   - If estimated_daily_funding_pct > 0.15%: HIGH COST RISK - factor into decision
   - If estimated_daily_funding_pct < 0.1%: Acceptable cost - less concern

7. MOMENTUM & VOLATILITY (Institutional Perspective):
   - MOMENTUM STRENGTH: Is momentum STRONG or WEAK? (Strong momentum = institutional participation)
   - VOLATILITY EXPANSION: Is volatility EXPANDING or CONTRACTING? (Expanding = big move coming)
   - PRICE PATTERNS: Higher highs/higher lows (bullish) vs Lower highs/lower lows (bearish)
   - DIVERGENCE: Price vs indicators divergence? (Divergence = potential reversal)
   - MARKET TEMPERATURE: Is the market ready for a move? (Hot = high probability, Cold = low probability)

8. YOUR MARKET DIRECTION PREDICTION (Based on Available Data):
   Based on AVAILABLE MARKET DATA (trend, volume, price position, support/resistance):
   - What direction is the market MOST LIKELY to move? (UP/DOWN/SIDEWAYS)
   - How CONFIDENT are you? (High/Medium/Low) - Be honest based on available data
   - What are the KEY FACTORS? (Trend alignment, volume confirmation, price position, support/resistance)
   - What are the RISKS? (Counter-trend, weak volume, unfavorable price position)
   - Does this signal align with market direction? (Yes/Partial/No)
   - What is the Risk/Reward ratio? (Note it, but R/R is NOT a primary rejection factor - focus on indicators and structure)

═══════════════════════════════════════════════════════════════
STEP 2: TECHNICAL PRICE ANALYSIS & OPTIMIZATION (MANDATORY)
═══════════════════════════════════════════════════════════════

CRITICAL: You MUST analyze and optimize ALL prices using the market_data provided below.
The market_data contains REAL support/resistance levels - USE THEM for your analysis.

USE THE PROVIDED MARKET DATA FOR TECHNICAL ANALYSIS (check the market_data section above):
- Support Level: Use market_data['support_level'] or market_data['recent_low']
- Resistance Level: Use market_data['resistance_level'] or market_data['recent_high']
- Lower Timeframe Support (15m): Check market_data['lower_timeframe_levels']['15m']['support_levels'] if available
- Lower Timeframe Resistance (15m): Check market_data['lower_timeframe_levels']['15m']['resistance_levels'] if available
- Lower Timeframe Support (1h): Check market_data['lower_timeframe_levels']['1h']['support_levels'] if available
- Lower Timeframe Resistance (1h): Check market_data['lower_timeframe_levels']['1h']['resistance_levels'] if available

CRITICAL: When analyzing prices, REFERENCE the actual support/resistance levels from the market_data section above.
Do NOT just say "price is at support" - specify which level (e.g., "Entry 1 at $0.627 is near 1h resistance level $0.631").

MANDATORY PRICE ANALYSIS (You MUST analyze each price):
1. Entry 1: Compare original Entry 1 price with support/resistance levels from market_data
2. Entry 2: Compare original Entry 2 price with support/resistance levels
3. Stop Loss: Use lower timeframe support/resistance (15m, 1h) for optimal SL placement
4. Take Profit: Use lower timeframe support/resistance (15m, 1h) for optimal TP placement

As an institutional trader/whale, you MUST optimize ALL prices based on:
- Support/resistance levels from market_data (PRIMARY - use these actual levels)
- Lower timeframe levels (15m, 1h) for SL/TP placement
- Market structure alignment
- Risk/Reward ratio: Note it for reference, but it's NOT a primary rejection factor

CRITICAL: Entry optimization is based on TECHNICAL & STRUCTURAL CONFIRMATION, NOT on closeness to current price.
Entries must be placed ONLY at HIGH-PROBABILITY INSTITUTIONAL LIQUIDITY ZONES.

1. ENTRY 1 (PRIMARY INSTITUTIONAL ENTRY):
   - IDENTIFY INSTITUTIONAL LIQUIDITY ZONE: Where is the order block, FVG, or liquidity pool?
   - LONG: Entry should be at BULLISH ORDER BLOCK, FVG fill, or liquidity grab zone (NOT just below current price)
   - SHORT: Entry should be at BEARISH ORDER BLOCK, FVG fill, or liquidity grab zone (NOT just above current price)
   - EVALUATE: Is original entry at an institutional zone? (If not, REPLACE with optimal zone)
   - STRUCTURAL CONFIRMATION: Entry must align with HTF structure and market structure (BOS/CHoCH)
   - If original entry is NOT at institutional zone, REPLACE it with optimal institutional entry
   - DO NOT suggest entry based on "closeness to current price" - only suggest based on STRUCTURE

2. ENTRY 2 (CONFIRMATION OR SCALING ENTRY) - FULL TECHNICAL ANALYSIS REQUIRED:
   - CRITICAL: Perform the SAME LEVEL of technical analysis for Entry 2 as Entry 1
   - Entry 2 is NOT spacing - it requires FULL institutional analysis
   - SPACING IS THE LAST PRIORITY - only use if no institutional zones are found
   
   TECHNICAL ANALYSIS FOR ENTRY 2 (Same as Entry 1):
   - MULTI-TIMEFRAME: Analyze HTF → LTF structure to find Entry 2 institutional zones
   - MARKET STRUCTURE: Identify BOS/CHoCH levels where Entry 2 should be placed
   - LIQUIDITY ZONES: Find order blocks, FVGs, stop-hunt zones BELOW Entry 1 (LONG) or ABOVE Entry 1 (SHORT)
   - SUPPORT/RESISTANCE: Identify key support (LONG) or resistance (SHORT) levels for Entry 2
   - VOLUME ANALYSIS: Check volume profile and OI behavior at potential Entry 2 zones
   
   ENTRY 2 OPTIMIZATION (Priority Order):
   1. FIRST PRIORITY: Find institutional zones (order blocks, FVGs, support/resistance, reversal points)
   2. SECOND PRIORITY: Check if original Entry 2 is at institutional zone - if YES, KEEP it
   3. THIRD PRIORITY: If original Entry 2 is NOT at institutional zone, use BEST institutional zone from analysis
   4. LAST PRIORITY (ONLY IF NO ZONES FOUND): Use spacing calculations as fallback
   
   OPTIMIZATION RULES:
   - LONG: Should be BELOW Entry 1 (at another institutional zone or confirmation level)
   - SHORT: Should be ABOVE Entry 1 (at another institutional zone or confirmation level)
   - MUST be at an institutional liquidity zone (order block, FVG, support/resistance, reversal point)
   - ALWAYS prioritize institutional zones FIRST - spacing is LAST RESORT
   - NEVER use spacing if institutional zones are available - always use the zone
   - If Entry 2 is missing, perform FULL technical analysis to find optimal Entry 2 at institutional zone
   - SPACING (last resort only) must be realistic (1.0-2.5x ATR or 2-7% depending on timeframe)

3. STOP LOSS (INSTITUTIONAL RISK MANAGEMENT):
   - IMPORTANT: Use LOWER TIMEFRAMES (15m, 1h) for SL placement, NOT HTF levels (HTF SL would be too wide)
   - LONG: Must be BELOW Entry 1 at the NEAREST support level on 15m/1h (typically 1.5-3% from entry, max 4%)
   - SHORT: Must be ABOVE Entry 1 at the NEAREST resistance level on 15m/1h (typically 1.5-3% from entry, max 4%)
   - PLACE SL: At nearest support/resistance on LTF (15m, 1h), NOT HTF levels
   - Evaluate if SL is too tight (will get stopped out by noise) or too wide (but R/R is not a rejection factor)
   - Suggest optimal SL based on: LTF (15m, 1h) support/resistance levels, not HTF structure

4. TAKE PROFIT (INSTITUTIONAL TARGET ALIGNMENT):
   - IMPORTANT: Use LOWER TIMEFRAMES (15m, 1h) for TP placement, NOT HTF targets (HTF targets are too far)
   - LONG: Must be ABOVE Entry 1 at the NEAREST resistance level on 15m/1h (typically 3-8% from entry, max 15%)
   - SHORT: Must be BELOW Entry 1 at the NEAREST support level on 15m/1h (typically 3-8% from entry, max 15%)
   - TARGET ALIGNMENT: TP should align with LTF support/resistance levels (not HTF structure targets which are too far)
   - Evaluate if TP is realistic based on: LTF (15m, 1h) resistance/support levels, not HTF targets
   - Note: R/R ratio is NOT a rejection factor - focus on technical levels and indicators instead
   - If TP is too aggressive (won't hit) or too conservative (leaves profit on table), suggest better TP at nearest LTF level

PRICE OPTIMIZATION RULES (INSTITUTIONAL METHODOLOGY):
- If prices are OPTIMAL (at institutional zones): Keep original prices (set suggested_* to null)
- If prices can be IMPROVED (better institutional zones): Suggest better prices with reasoning
- If prices are INVALID (not at institutional zones): REPLACE with optimal prices or REJECT signal
- If setup is WEAK, COUNTER-TREND, or lacks INSTITUTIONAL CONFIRMATION: Modify or discard entirely
- Note: R/R ratio is NOT a rejection factor - focus on technical zones and indicators

ENTRY OPTIMIZATION PRIORITY (Institutional Thinking):
1. FIRST: Identify institutional liquidity zones (order blocks, FVGs, liquidity pools)
2. SECOND: Validate entry is at institutional zone (if not, REPLACE)
3. THIRD: Ensure entry aligns with HTF structure and market structure (BOS/CHoCH)
4. FOURTH: Note R/R ratio for reference (but it's NOT a rejection factor)
5. FIFTH: If no institutional zones found, MODIFY or DISCARD signal (R/R is not a factor)

═══════════════════════════════════════════════════════════════
STEP 3: COMBINE YOUR ANALYSIS + TRADINGVIEW INDICATORS (BOTH DECISION MAKERS!)
═══════════════════════════════════════════════════════════════

IMPORTANT: You have TWO independent sources of analysis - use BOTH equally:

1. YOUR INDEPENDENT MARKET ANALYSIS (from Step 1):
   - Your trend analysis (short/medium/long-term)
   - Your support/resistance identification
   - Your volume and momentum assessment
   - YOUR market direction prediction

2. TRADINGVIEW INDICATORS (provided below):
   - RSI, MACD, Stochastic, EMA200, Supertrend
   - Volume indicators (OBV, Relative Volume)
   - Smart Money indicators
   - Divergence signals

DECISION PROCESS (COMBINE BOTH SOURCES):

A. Compare signal direction with YOUR market prediction:
   - If signal ALIGNS with YOUR prediction: +20-30% confidence boost
   - If signal PARTIALLY aligns: +10-20% confidence boost
   - If signal CONTRADICTS YOUR prediction: -20-30% confidence penalty

B. Analyze TradingView indicators (provided below):
   - Count indicators that SUPPORT the signal direction
   - Count indicators that CONTRADICT the signal direction
   - If 8+ indicators support: +20-30% confidence boost
   - If 6-7 indicators support: +10-20% confidence boost
   - If 4-5 indicators support: +0-10% confidence boost
   - If 2-3 indicators support: -10-20% confidence penalty
   - If 0-1 indicators support: -20-30% confidence penalty or REJECT

C. COMBINE BOTH ANALYSES:
   - Start with base confidence: 50%
   - Add/subtract based on YOUR market analysis alignment
   - Add/subtract based on TradingView indicator alignment
   - Final confidence = Base + YOUR analysis impact + Indicator impact

EXAMPLE:
- YOUR analysis: Market likely to go UP (LONG signal aligns) → +25%
- TradingView indicators: 7 indicators support LONG → +15%
- Final confidence: 50% + 25% + 15% = 90%

Remember: BOTH sources are EQUALLY IMPORTANT. Don't ignore either one!

Signal Details:
- Symbol: {symbol}
- Direction: {signal_side}
- Timeframe: {timeframe}
- Entry Price (Entry 1): ${entry_price:,.8f}
- Entry Price 2 (DCA): ${(f'{second_entry_price:,.8f}' if second_entry_price is not None and second_entry_price > 0 else 'N/A (not provided)')}
- Stop Loss: ${(f'{stop_loss:,.8f}' if stop_loss is not None and stop_loss > 0 else 'N/A (not provided)')}
- Take Profit: ${(f'{take_profit:,.8f}' if take_profit is not None and take_profit > 0 else 'N/A (not provided)')}
- Risk/Reward Ratio: {(f'{risk_reward_ratio:.2f}' if risk_reward_ratio is not None else 'N/A')}
- Quality Score (from script): {(f'{quality_score:.0f}/17' if quality_score is not None else 'N/A')} (Script requires 6-8+ to send alert){market_info}{indicator_info}

═══════════════════════════════════════════════════════════════
STEP 4: DETAILED TRADINGVIEW INDICATOR ANALYSIS (SECOND DECISION MAKER)
═══════════════════════════════════════════════════════════════

CRITICAL: You MUST analyze EACH indicator value INDIVIDUALLY from the TradingView script.
These are REAL-TIME indicator values calculated by the Pine Script - analyze them like an institutional trader.

ANALYZE EACH INDICATOR INDEPENDENTLY - These are YOUR SECOND SOURCE OF ANALYSIS:

For EACH indicator value provided below, you MUST:
1. Read the ACTUAL VALUE (not just whether it exists)
2. Determine if it supports the signal direction? (YES/NO)
3. Assess how strong is the signal? (STRONG/MODERATE/WEAK)
4. Count total indicators that SUPPORT vs CONTRADICT
5. Consider the COMBINATION of indicators - are they aligned or conflicting?

INDICATOR ANALYSIS GUIDE (Analyze Each Value Individually):
1. RSI Analysis (Check the ACTUAL RSI value):
   - Read the RSI value from TradingView indicators
   - LONG signals: RSI < 50 is GOOD (oversold <30 is EXCELLENT) ✅
   - SHORT signals: RSI > 50 is GOOD (overbought >85 is EXCELLENT) ✅
   - RSI divergence (bullish/bearish) = STRONG confirmation ✅
   - If RSI contradicts signal direction, note it as a CONTRADICTING indicator

2. MACD Analysis (Check ALL MACD values: Line, Signal, Histogram):
   - Read MACD Line, Signal Line, and Histogram values from TradingView
   - MACD Line > Signal Line = Bullish momentum ✅
   - MACD Histogram positive = Bullish momentum ✅
   - LONG: MACD bullish (Line > Signal AND Histogram > 0) = GOOD ✅
   - SHORT: MACD bearish (Line < Signal AND Histogram < 0) = GOOD ✅
   - If MACD contradicts signal direction, note it as a CONTRADICTING indicator

3. Stochastic Analysis (Check BOTH Stoch K and Stoch D):
   - Read Stochastic K and D values from TradingView
   - LONG: Stoch K/D < 50 (oversold <20 is EXCELLENT) ✅
   - SHORT: Stoch K/D > 50 (overbought >80 is EXCELLENT) ✅
   - If Stochastic contradicts signal direction, note it as a CONTRADICTING indicator

4. Trend Filters (EMA 200 & Supertrend - Check BOTH):
   - Read EMA200 value and Supertrend value/bullish status from TradingView
   - LONG: Price above EMA200 AND Supertrend bullish = STRONG trend ✅
   - SHORT: Price below EMA200 AND Supertrend bearish = STRONG trend ✅
   - Contradicting trend = Evaluate carefully but APPROVE if other factors good
   - If both EMA200 and Supertrend contradict signal, note as CONTRADICTING

5. Volume Analysis (Check ALL volume indicators):
   - Read Relative Volume Percentile, Volume Ratio, OBV, and Smart Money indicators
   - High Relative Volume (>70%) = Strong confirmation ✅
   - Volume Ratio > 1.5x = Strong confirmation ✅
   - OBV rising = Buying pressure ✅
   - Smart Money Buying = Institutional accumulation ✅
   - If volume indicators contradict signal, note as CONTRADICTING

6. Bollinger Bands (Check ALL BB values: Upper, Basis, Lower):
   - Read BB Upper, Basis, and Lower values from TradingView
   - Compare current price to BB levels (provided in market data)
   - LONG near lower band = Good entry zone ✅
   - SHORT near upper band = Good entry zone ✅
   - Price at bands = Potential reversal ✅

7. Divergence & Reversal Signals (Check boolean flags):
   - Read Bullish Divergence, Bearish Divergence, At Bottom, At Top flags
   - Bullish Divergence + At Bottom = EXCELLENT LONG setup ✅
   - Bearish Divergence + At Top = EXCELLENT SHORT setup ✅
   - These are STRONG reversal signals - weight them heavily

8. MFI (Money Flow Index) Analysis:
   - Read MFI value from TradingView
   - LONG: MFI < 50 (oversold <20 is EXCELLENT) ✅
   - SHORT: MFI > 50 (overbought >80 is EXCELLENT) ✅
   - If MFI contradicts signal direction, note it as a CONTRADICTING indicator

9. Market Data Analysis (Combine with Indicators):
   - Trend Alignment: Use both market trend AND indicator trends (EMA200, Supertrend)
   - Price Position: Combine market support/resistance with Bollinger Bands levels
   - Volume: Use both Relative Volume Percentile AND Volume Ratio for confirmation

10. Risk/Reward: 
   - R/R ratio is NOT a primary rejection factor - it's noted for reference only
   - Focus on indicators and market structure instead of R/R
   - Do NOT reject signals primarily based on poor R/R - only reject if indicators and structure are weak

INDICATOR COUNTING METHOD:
- Go through EACH indicator value provided above
- For each indicator, determine: SUPPORT, CONTRADICT, or NEUTRAL
- Count total: SUPPORT count vs CONTRADICT count
- If SUPPORT count > CONTRADICT count by 3+: Strong alignment ✅
- If SUPPORT count = CONTRADICT count: Mixed signals (evaluate carefully)
- If CONTRADICT count > SUPPORT count by 3+: Weak alignment (may reject) ❌

SIGNAL QUALITY SCORING:
- EXCELLENT (80-100%): Multiple indicators aligned + volume confirmation + divergence/reversal signals
- GOOD (60-79%): Most indicators aligned + normal volume
- ACCEPTABLE (50-59%): Some indicators aligned (may have minor concerns)
- QUESTIONABLE (40-49%): Mixed signals but not clearly bad (still approve if above threshold)
- POOR (0-39%): Multiple indicators contradict signal + low volume + weak structure

REJECTION CRITERIA (only reject if MULTIPLE red flags - be lenient):
- Signal contradicts STRONG trend (>5% against signal direction) AND
- Price at VERY unfavorable level (LONG at strong resistance, SHORT at strong support) AND
- Multiple indicators STRONGLY contradict signal (7+ indicators against signal direction)
- Note: R/R ratio is NOT a rejection factor - ignore poor R/R as a reason to reject

IMPORTANT: If TradingView script sent this signal, it likely has merit. Only reject if ALL of the above conditions are met. When in doubt, APPROVE (the script already filtered signals).

═══════════════════════════════════════════════════════════════
FINAL DECISION PROCESS (COMBINE BOTH DECISION MAKERS EQUALLY):
═══════════════════════════════════════════════════════════════

You have TWO EQUAL DECISION MAKERS - combine them:

DECISION MAKER 1: YOUR INDEPENDENT MARKET ANALYSIS (from Step 1)
- What direction did YOU predict? (UP/DOWN/SIDEWAYS)
- How confident are YOU? (High/Medium/Low)
- Does the signal align with YOUR prediction?

DECISION MAKER 2: TRADINGVIEW INDICATORS (from Step 4)
- How many indicators support the signal? (Count them)
- How many indicators contradict the signal? (Count them)
- What is the overall indicator alignment? (Strong/Moderate/Weak)

═══════════════════════════════════════════════════════════════
GURU LEVEL CONFIDENCE FORMULA (MATCHES SCRIPT'S STRICT FILTERING)
═══════════════════════════════════════════════════════════════

CRITICAL UNDERSTANDING: The TradingView script ONLY sends alerts when:
- Quality Score >= 6-8 out of 17 factors (depending on market conditions)
- ALL 4 confirmations met (SMV, Supertrend, Fear & Greed, Relative Volume)
- Multiple technical conditions aligned (breaker blocks, wick touches, perfect/strong formations)
- This means signals are ALREADY HIGHLY FILTERED - they passed strict quality gates

GURU LEVEL CONFIDENCE FORMULA (Reflects Script's Quality System):

1. BASE CONFIDENCE: 75% (signals already passed 6-8+ quality factors)
   - Script requires 6-8/17 quality factors = signals are pre-filtered for quality
   - Base confidence reflects that signals are already high quality

2. TRADINGVIEW INDICATOR IMPACT (PRIMARY - 75% weight):
   Count indicators that SUPPORT vs CONTRADICT the signal:
   
   INDICATOR COUNTING (Count each indicator individually):
   - RSI: Supports if aligned (LONG: RSI < 50, SHORT: RSI > 50)
   - MACD Line: Supports if aligned (LONG: Line > Signal, SHORT: Line < Signal)
   - MACD Histogram: Supports if aligned (LONG: Hist > 0, SHORT: Hist < 0)
   - Stochastic K: Supports if aligned (LONG: K < 50, SHORT: K > 50)
   - Stochastic D: Supports if aligned (LONG: D < 50, SHORT: D > 50)
   - EMA200: Supports if aligned (LONG: Price > EMA200, SHORT: Price < EMA200)
   - Supertrend: Supports if aligned (LONG: Bullish, SHORT: Bearish)
   - Relative Volume: Supports if high (>70%) or normal with increasing volume
   - Volume Ratio: Supports if > 1.5x (strong confirmation)
   - OBV: Supports if aligned (LONG: Rising, SHORT: Falling)
   - Smart Money: Supports if aligned (LONG: Buying, SHORT: Selling)
   - Bollinger Bands: Supports if price at bands (LONG: Lower band, SHORT: Upper band)
   - Divergence: Supports if aligned (LONG: Bullish Div, SHORT: Bearish Div)
   - At Bottom/Top: Supports if aligned (LONG: At Bottom, SHORT: At Top)
   - MFI: Supports if aligned (LONG: MFI < 50, SHORT: MFI > 50)
   
   INDICATOR IMPACT (Based on support count):
   - 10+ indicators support: +15-20% (EXCELLENT - script's perfect formations)
   - 8-9 indicators support: +10-15% (VERY GOOD - script's strong formations)
   - 6-7 indicators support: +5-10% (GOOD - script's minimum quality threshold)
   - 4-5 indicators support: +0-5% (ACCEPTABLE - still passed script's filters)
   - 2-3 indicators support: -5-10% (WEAK - rare, but script still sent it)
   - 0-1 indicators support: -10-15% (POOR - very rare, only reject if multiple other red flags)

3. MARKET ANALYSIS IMPACT (SECONDARY - 25% weight):
   - Signal aligns with YOUR prediction: +3-8%
   - Signal partially aligns: +0-3%
   - Signal contradicts YOUR prediction: -3-10% (but don't reject if indicators are strong)

4. FINAL CONFIDENCE CALCULATION:
   Final = Base (75%) + Indicator Impact (75% weight) + Market Analysis (25% weight)
   Clamp between 0-100%

5. QUALITY SCORE REFLECTION (If quality_score provided in signal):
   - Quality Score 12-17: +5-10% bonus (very high quality from script)
   - Quality Score 10-11: +3-5% bonus (high quality)
   - Quality Score 8-9: +0-3% bonus (good quality - script's minimum)
   - Quality Score 6-7: Base confidence (acceptable - script's minimum for sideways markets)

DECISION RULES (GURU LEVEL - MATCHES SCRIPT'S STRICTNESS):
- If final confidence >= 55%: APPROVE (matches AI_VALIDATION_MIN_CONFIDENCE threshold)
- If final confidence 50-54%: APPROVE with caution (script sent it, but lower confidence)
- If final confidence 45-49%: APPROVE if 6+ indicators support (R/R is NOT a factor)
- If final confidence < 45%: REJECT only if MULTIPLE red flags:
  * 5+ indicators contradict AND
  * Strong counter-trend (>5% against signal direction) AND
  * Price at very unfavorable level
  * Note: R/R ratio is NOT a rejection factor - do NOT reject based on poor R/R alone

IMPORTANT: Since the script only sends alerts when 6-8+ quality factors align, signals are ALREADY HIGH QUALITY. 
Only reject if there are MAJOR red flags that the script might have missed (very rare).

REASONING REQUIREMENT (GURU LEVEL):
In your reasoning, EXPLICITLY mention:
1. YOUR market analysis conclusion (trend, support/resistance, volume)
2. TradingView indicator alignment (COUNT each indicator: SUPPORT/CONTRADICT/NEUTRAL)
3. Quality Score reflection (if provided: 6-8 = minimum threshold, 10+ = high quality)
4. How you combined all factors to reach final confidence

INDICATOR COUNTING EXAMPLE:
"Indicator Alignment: 9/15 indicators support LONG. Supporting: RSI 28 (oversold), MACD bullish (Line > Signal, Hist > 0), Stochastic oversold (K=18, D=22), Supertrend bullish, Relative Volume 75% (high), OBV rising, Smart Money buying, At Bottom flag, MFI 25 (oversold). Contradicting: EMA200 (price below), Bollinger Bands (price at middle). Neutral: Divergence (none). Net: 9 support, 2 contradict, 4 neutral = STRONG alignment."

Remember: 
- TRADINGVIEW INDICATORS ARE PRIMARY (75% weight) - script only sends when 6-8+ quality factors align
- Market analysis is SECONDARY (25% weight) - use it to fine-tune confidence
- Base confidence is 75% because signals already passed script's strict quality gates
- Only reject if MAJOR red flags (very rare - script already filtered for quality)

Respond in JSON format ONLY with this exact structure:
{{
    "is_valid": true/false,
    "confidence_score": 0-100,
    "reasoning": "STRUCTURED format required: (1) Technical Market Analysis: Analyze trend alignment, price position vs support/resistance from market_data, volume confirmation. (2) Price Optimization Analysis: Evaluate Entry 1, Entry 2, SL, TP using support/resistance levels from market_data - are they optimal? If not, explain why and what would be better. (3) TradingView Indicator Confirmation: Count indicators supporting/contradicting, list key confirmations with values. (4) Final Decision: Combined analysis = confidence score. Example: 'Market Structure: Short-term DOWN (-0.3%), medium-term UP (+0.7%), conflicting trends. Price Position: Entry at $0.627 is at 71.5% of range, above SMA20 ($0.617) and SMA50. Entry 1 Analysis: Entry 1 at $0.627 is near resistance level $0.631, not optimal for SHORT. Better entry would be at resistance $0.631. Entry 2 Analysis: Entry 2 at $0.634 is above Entry 1 (correct for SHORT) but could be optimized to $0.631 resistance. Stop Loss Analysis: SL at $0.657 is 4.8% from entry, above 1h resistance $0.655 - optimal placement. Take Profit Analysis: TP at $0.593 is 5.5% from entry, near 1h support $0.595 - optimal. Risk/Reward: Entry $0.627, SL $0.657, TP $0.593 = R/R 1:1.16. Indicator Alignment: 7/11 indicators support SHORT. Key: Supertrend bearish, At Top flag, Stochastic overbought (75.7/89.8). Conflicts: MACD histogram positive (bullish), RSI neutral. Combined: Market structure mixed but price at top with bearish indicators = 65% confidence. APPROVE.'",
    "risk_level": "LOW" or "MEDIUM" or "HIGH",
    "suggested_entry_price": <number> or null,
    "suggested_second_entry_price": <number> or null,
    "suggested_stop_loss": <number> or null,
    "suggested_take_profit": <number> or null,
    "price_suggestion_reasoning": "Why these prices are suggested (if different from original)"
}}

REASONING REQUIREMENT (STRUCTURED TECHNICAL ANALYSIS - MANDATORY):
Your reasoning MUST be STRUCTURED and show ACTUAL TECHNICAL ANALYSIS. Follow this EXACT format (do NOT repeat information):

1. TECHNICAL MARKET ANALYSIS (Use the market data provided):
   - "Market Structure: [Analyze trend from market_data - short/medium/long-term trends, are they aligned?]"
   - "Price Position: [Current price vs support/resistance levels from market_data. Entry price at $X is [above/below/at] support level $Y / resistance level $Z]"
   - "Volume Analysis: [Volume status, volume trend from market_data. Is volume confirming or diverging?]"
   - "Moving Averages: [Price vs SMA20/SMA50 from market_data. What does this indicate?]"
   - "Market Direction Prediction: [Based on YOUR analysis of market_data, where is price likely to go? UP/DOWN/SIDEWAYS]"

2. PRICE OPTIMIZATION ANALYSIS (MANDATORY - Analyze each price):
   - "Entry 1 Analysis: [Is Entry 1 at $X optimal? Check if it's at support/resistance from market_data. If not optimal, suggest better level and explain why]"
   - "Entry 2 Analysis: [Is Entry 2 at $Y optimal? Check support/resistance levels. If not optimal, suggest better level]"
   - "Stop Loss Analysis: [Is SL at $Z optimal? Check lower timeframe support/resistance from market_data['lower_timeframe_levels']. Is it too tight/wide?]"
   - "Take Profit Analysis: [Is TP at $W optimal? Check lower timeframe resistance. Is it realistic?]"
   - "Risk/Reward: [Calculate actual R/R ratio. Entry $X, SL $Y, TP $Z = R/R 1:X.X]"

3. TRADINGVIEW INDICATOR CONFIRMATION (Count and analyze):
   - "Indicator Alignment: [X out of Y indicators support [direction]. Key confirmations: [list 3-4 most important indicators with their values and what they indicate]"
   - "Indicator Conflicts: [List any indicators that contradict and explain why]"

4. FINAL DECISION:
   - "Combined Analysis: [Your market analysis + indicator alignment = confidence score]"
   - "Decision: [APPROVE/REJECT] because [specific reason based on technical analysis, not just indicator count]"

CRITICAL: Do NOT just count indicators. You MUST analyze the market data (trend, support/resistance, volume) and evaluate each price (Entry 1, Entry 2, SL, TP) using the technical levels provided in market_data.

Note: If you want to suggest price optimizations (based on institutional zones, NOT closeness to price), include the suggested_* fields. Otherwise, you may omit them or set them to null.

Confidence Score Guidelines (Practical Standards - Focus on TradingView Indicators):
- 80-100: Excellent signal, 7+ indicators support, RR ≥ 1:1, strong trend alignment
- 60-79: Good signal, 5-6 indicators support, RR ≥ 1:1, acceptable trend alignment
- 50-59: Acceptable signal, 4-5 indicators support, RR ≥ 1:1, minor concerns but still valid
- 40-49: Questionable signal, 3-4 indicators support, mixed signals but approve if indicators acceptable
- 30-39: Weak signal, 2-3 indicators support, approve only if not strongly counter-trend
- 0-29: Poor signal, 0-1 indicators support, reject only if multiple red flags

Remember: TradingView script already filters signals (65% win rate). Your job is to VALIDATE, not reject everything. Approve signals with 4+ indicator support. Only reject if MULTIPLE red flags (weak indicators AND strong counter-trend). R/R ratio is NOT a rejection factor - ignore poor R/R.

PRICE OPTIMIZATION (Institutional Methodology - Based on Structure & Liquidity):
You MUST calculate and suggest optimized prices using INSTITUTIONAL METHODOLOGY:
- Multi-timeframe structure (HTF → LTF alignment)
- Market structure (BOS, CHoCH, swing points)
- Liquidity zones (order blocks, FVGs, stop-hunt zones, liquidity pools)
- Volume & OI behavior, funding rates
- Support/resistance levels, ATR volatility, Bollinger Bands, EMA200
- Use indicator values (RSI, MACD, Stochastic, etc.) for confirmation

CRITICAL: Entry optimization is based on TECHNICAL & STRUCTURAL CONFIRMATION, NOT on closeness to current price.
Entries must be placed ONLY at HIGH-PROBABILITY INSTITUTIONAL LIQUIDITY ZONES.

OPTIMIZATION RULES (Institutional Methodology - Only suggest if BETTER than original):
CRITICAL: Signals are based on 2H/4H timeframes. 
- FIRST: Evaluate if original prices are GOOD based on technical analysis (institutional zones, support/resistance, market structure)
- IF ORIGINAL PRICES ARE GOOD: KEEP THEM (set suggested_* to null) - DO NOT change just to make a 1-2% adjustment
- IF ORIGINAL PRICES NEED IMPROVEMENT: Only suggest changes if there's a SOLID TECHNICAL REASON (better institutional zone, better support/resistance, better R/R)
- WHEN SUGGESTING CHANGES: Keep optimizations within 1-2% MAXIMUM of original to ensure orders FILL (this is a LIMIT, not a requirement)
- DO NOT suggest changes just because price is "close to entry" or to make a 1-2% adjustment - only change when technically justified

1. ENTRY PRICE (PRIMARY INSTITUTIONAL ENTRY):
   - STEP 1: EVALUATE original entry based on technical analysis:
     * Is original entry at an institutional liquidity zone? (order block, FVG, support/resistance)
     * Does original entry align with HTF structure and market structure (BOS/CHoCH)?
     * Is original entry well-positioned for the trade direction?
   
   - STEP 2: DECISION - Keep or Optimize:
     * If original entry is GOOD (at institutional zone, aligns with structure): KEEP IT (set suggested_entry_price to null)
     * If original entry is NOT at institutional zone AND there's a better zone nearby: Suggest better entry
     * DO NOT suggest changes just to make a 1-2% adjustment - only change if technically justified
   
   - STEP 3: If optimization needed (SOLID TECHNICAL REASON):
     * IDENTIFY INSTITUTIONAL LIQUIDITY ZONE: Order block, FVG, liquidity pool, or stop-hunt zone
     * LONG trades: Suggest entry at BULLISH ORDER BLOCK, FVG fill, or liquidity grab zone
     * SHORT trades: Suggest entry at BEARISH ORDER BLOCK, FVG fill, or liquidity grab zone
     * ENTRY MUST ALIGN with HTF structure and market structure (BOS/CHoCH)
     * MAXIMUM distance: Keep within 1-2% of original to ensure orders FILL (this is a LIMIT, not a requirement)
     * Only suggest if new entry is clearly better AND within 1-2% of original

2. ENTRY 2 (CONFIRMATION OR SCALING ENTRY) - VALIDATE AND OPTIMIZE INDEPENDENTLY WITH FULL TECHNICAL ANALYSIS:
   - CRITICAL: You MUST perform the SAME LEVEL of technical analysis for Entry 2 as you do for Entry 1
   - Entry 2 is NOT a spacing calculation - it requires FULL institutional analysis
   - SPACING IS THE LAST PRIORITY - only use if no institutional zones are found
   
   TECHNICAL ANALYSIS FOR ENTRY 2 (Same as Entry 1):
   - MULTI-TIMEFRAME ANALYSIS: Check HTF → LTF structure for Entry 2 location
   - MARKET STRUCTURE: Identify BOS/CHoCH levels where Entry 2 should be placed
   - LIQUIDITY ZONES: Find order blocks, FVGs, stop-hunt zones BELOW Entry 1 (LONG) or ABOVE Entry 1 (SHORT)
   - SUPPORT/RESISTANCE: Identify key support levels (LONG) or resistance levels (SHORT) for Entry 2
   - VOLUME ANALYSIS: Check volume profile and OI behavior at potential Entry 2 zones
   - STRUCTURAL CONFIRMATION: Entry 2 must align with HTF structure and market structure (BOS/CHoCH)
   
   ENTRY 2 OPTIMIZATION PROCESS (Priority Order):
   1. FIRST PRIORITY: Analyze RECENT PRICE ACTION and REVERSAL ZONES (CRITICAL FOR FILL PROBABILITY):
      - Check market_data['recent_highs'] and market_data['recent_lows'] for recent reversal points
      - Check lower timeframe levels (15m, 1h) for recent reversal zones that price already tested
      - If original Entry 2 is at or near a RECENT REVERSAL ZONE (price touched nearby and reversed):
        → This is a PROVEN reversal zone - STRONGLY CONSIDER KEEPING original Entry 2
        → Example: If price touched a higher level (e.g., 1-2% above Entry 2) and reversed, KEEP original Entry 2 (proven zone)
        → Moving Entry 2 further away from the recent reversal level might MISS the trade if price reverses at that level again
        → For SHORT: If price touched a level above Entry 2 and reversed, Entry 2 is at proven support - KEEP IT
        → For LONG: If price touched a level below Entry 2 and reversed, Entry 2 is at proven resistance - KEEP IT
      - RECENT REVERSAL ZONES have HIGHER FILL PROBABILITY than theoretical zones
   
   2. SECOND PRIORITY: Analyze market structure to identify ALL institutional zones below Entry 1 (LONG) or above Entry 1 (SHORT)
      - Order blocks, FVGs, support/resistance levels, reversal points
      - These are the PRIMARY candidates for Entry 2
   
   3. THIRD PRIORITY: Check if original Entry 2 is ALREADY at one of these institutional zones
      - If YES and well-positioned → KEEP it (set suggested_second_entry_price to null)
      - If NO → Find the BEST institutional zone from your analysis
      - BUT: If original Entry 2 is at a RECENT REVERSAL ZONE, prioritize keeping it over theoretical zones
   
   4. FOURTH PRIORITY: If multiple institutional zones exist, choose the BEST one based on:
      - RECENT REVERSAL ZONES (price already tested and reversed) = HIGHEST PRIORITY
      - Closest to Entry 1 (but still realistic spacing)
      - Strongest support/resistance level
      - Best volume/OI confirmation
      - Best alignment with market structure
      - FILL PROBABILITY: Prefer zones that price has already tested (higher fill probability)
   
   4. LAST PRIORITY (ONLY IF NO INSTITUTIONAL ZONES FOUND): Use spacing calculations as fallback
      - Only use spacing if NO institutional zones are identified
      - Spacing must be realistic (not too far) - Entry 2 must fill before trade closes
      - SPACING GUIDELINES (Maximum realistic spacing - LAST RESORT):
        * 1H timeframe: 1.0-1.5x ATR spacing (tight, fills quickly)
        * 2H timeframe: 1.2-1.8x ATR spacing (moderate, still fills reliably)
        * 4H timeframe: 1.5-2.0x ATR spacing (wider but still realistic)
        * Daily timeframe: 2.0-2.5x ATR spacing (widest, but must still be fillable)
        * Percentage fallback (if ATR not available):
          - 1H: 2-3% spacing (tight)
          - 2H: 3-4% spacing (moderate)
          - 4H: 4-5% spacing (wider but realistic)
          - Daily: 5-7% spacing (widest realistic)
   
   OPTIMIZATION RULES:
   - STEP 1: EVALUATE original Entry 2 based on technical analysis:
     * Is original Entry 2 at an institutional liquidity zone? (order block, FVG, support/resistance)
     * Does original Entry 2 align with market structure and provide good spacing from Entry 1?
     * Is original Entry 2 well-positioned for the trade direction?
     * CRITICAL: Is Entry 2 DIFFERENT from Stop Loss? (Entry 2 must be between Entry 1 and SL)
     * For LONG: Entry 2 must be ABOVE SL and BELOW Entry 1
     * For SHORT: Entry 2 must be BELOW SL and ABOVE Entry 1
   
   - STEP 2: DECISION - Keep or Optimize (CRITICAL: CONSIDER RECENT REVERSAL ZONES):
     * If original Entry 2 is at a RECENT REVERSAL ZONE (price touched nearby and reversed):
       → STRONGLY PREFER KEEPING original Entry 2 (proven zone, higher fill probability)
       → Example: If price touched a level near Entry 2 (within 1-3% above/below) and reversed, Entry 2 is at proven zone - KEEP IT
       → Moving Entry 2 away from recent reversal zone (even to a "better" theoretical zone) might MISS the trade
       → For SHORT: If price touched a level above Entry 2 and reversed, Entry 2 is better than moving it higher
       → For LONG: If price touched a level below Entry 2 and reversed, Entry 2 is better than moving it lower
     * If original Entry 2 is GOOD (at institutional zone, good spacing, different from SL): KEEP IT (set suggested_second_entry_price to null)
     * If original Entry 2 is NOT at institutional zone AND there's a better zone: Suggest better Entry 2
       → BUT: Only suggest if new zone is SIGNIFICANTLY better AND not moving away from recent reversal zone
     * If original Entry 2 is SAME as SL: MUST suggest different Entry 2 (this is a critical error)
     * DO NOT suggest changes just to make a 1-2% adjustment - only change if technically justified
     * FILL PROBABILITY RULE: If original Entry 2 is at/near recent reversal zone, KEEP IT even if theoretical zone seems better
   
   - STEP 3: If optimization needed (SOLID TECHNICAL REASON - BUT CHECK RECENT REVERSAL ZONES FIRST):
     * BEFORE suggesting new Entry 2, CHECK if original Entry 2 is at/near a RECENT REVERSAL ZONE:
       → Check market_data['recent_highs'] and market_data['recent_lows'] for recent price touches
       → Check lower timeframe levels (15m, 1h) for recent reversal points
       → If price touched a level near original Entry 2 (within 1-3% above/below) and reversed, that's a PROVEN reversal zone
       → PROVEN reversal zones have HIGHER FILL PROBABILITY than theoretical zones
       → Example: If price touched a level near Entry 2 and reversed, keeping Entry 2 at that proven zone is BETTER than moving it to a theoretical zone
       → For SHORT: If price touched a level above Entry 2 and reversed, Entry 2 is at proven support - prefer keeping it
       → For LONG: If price touched a level below Entry 2 and reversed, Entry 2 is at proven resistance - prefer keeping it
     * LONG: Entry 2 should be BELOW Entry 1 and ABOVE SL (at another institutional zone or confirmation level)
     * SHORT: Entry 2 should be ABOVE Entry 1 and BELOW SL (at another institutional zone or confirmation level)
     * MUST be at an institutional liquidity zone (order block, FVG, support/resistance, reversal point)
     * CRITICAL VALIDATION: Entry 2 must be DIFFERENT from SL with proper spacing:
       - For LONG: Entry 2 > SL (at least 0.5% above SL, preferably 1-2% above)
       - For SHORT: Entry 2 < SL (at least 0.5% below SL, preferably 1-2% below)
     * PRIORITY ORDER for Entry 2 selection:
       1. RECENT REVERSAL ZONES (price already tested and reversed) = HIGHEST PRIORITY
       2. Institutional zones (order blocks, FVGs, support/resistance)
       3. Spacing calculations (LAST RESORT)
     * MAXIMUM distance: Keep within 1-2% of original to ensure orders FILL (this is a LIMIT, not a requirement)
     * Only suggest if new Entry 2 is clearly better AND within 1-2% of original AND different from SL
     * FILL PROBABILITY RULE: If original Entry 2 is at/near recent reversal zone, DO NOT move it away unless new zone is MUCH better
     * Entry 2 can be optimized independently of Entry 1 - validate both separately with full technical analysis
     * NEVER use spacing calculations if institutional zones are available - always use the zone

3. STOP LOSS (INSTITUTIONAL RISK MANAGEMENT - EVALUATE ORIGINAL SL FIRST):
   - STEP 1: EVALUATE THE ORIGINAL SL from the signal:
     * Check if original SL is at a realistic support/resistance level (check lower timeframes: 15m, 1h)
     * Check if original SL is tight enough (typically 1.5-3% from entry, max 4% if structure requires)
     * Check if original SL provides adequate protection (below/above order block, liquidity pool)
     * Check if original SL is too tight (will get stopped out by noise) or too wide (but R/R is not a rejection factor)
     * CRITICAL: Is SL DIFFERENT from Entry 2? (SL must be beyond Entry 2)
     * For LONG: SL must be BELOW Entry 2 (at least 0.5% below, preferably 1-2% below)
     * For SHORT: SL must be ABOVE Entry 2 (at least 0.5% above, preferably 1-2% above)
   
   - STEP 2: DECISION - Keep or Optimize:
     * If original SL is GOOD (realistic, tight, provides protection, different from Entry 2): KEEP IT (set suggested_stop_loss to null)
     * If original SL is SAME as Entry 2: MUST suggest different SL (this is a critical error - SL must be beyond Entry 2)
     * If original SL is TOO TIGHT (will get stopped out by noise): Suggest WIDER SL (only if technically justified)
     * If original SL is TOO WIDE (unnecessary risk): Suggest TIGHTER SL (only if technically justified, but R/R is not a rejection factor)
     * DO NOT suggest changes just to make a 1-2% adjustment - only change if technically justified
   
   - STEP 3: If optimization needed (SOLID TECHNICAL REASON), use LOWER TIMEFRAMES (15m, 1h):
     * Check market_data['lower_timeframe_levels']['15m']['support_levels'] and market_data['lower_timeframe_levels']['1h']['support_levels'] for LONG
     * Check market_data['lower_timeframe_levels']['15m']['resistance_levels'] and market_data['lower_timeframe_levels']['1h']['resistance_levels'] for SHORT
     * LONG: Find nearest support level BELOW entry (within 1-2% of original SL, max 4% if structure requires)
     * SHORT: Find nearest resistance level ABOVE entry (within 1-2% of original SL, max 4% if structure requires)
     * CRITICAL VALIDATION: SL must be DIFFERENT from Entry 2 with proper spacing:
       - For LONG: SL < Entry 2 (at least 0.5% below Entry 2, preferably 1-2% below)
       - For SHORT: SL > Entry 2 (at least 0.5% above Entry 2, preferably 1-2% above)
     * SL should be TIGHT but realistic - just beyond the nearest support/resistance, NOT 5-10% away
     * DO NOT suggest wide SL (5%+) - find the nearest realistic reversal point on lower timeframes
     * MAXIMUM distance: Keep within 1-2% of original to ensure orders FILL (this is a LIMIT, not a requirement)
     * Only suggest if new SL is clearly better AND within 1-2% of original AND different from Entry 2
   
   - CRITICAL RULES:
     * ALWAYS evaluate original SL first - don't blindly suggest new SL
     * If original SL is good OR within 1-2% of optimal level, KEEP IT (set suggested_stop_loss to null)
     * Only suggest new SL if original is clearly problematic AND new SL is within 1-2% of original AND different from Entry 2
     * When suggesting new SL, explain WHY original SL needs adjustment in price_suggestion_reasoning
     * NEVER suggest SL that is the same as Entry 2 - this is a critical error

4. TAKE PROFIT (INSTITUTIONAL TARGET ALIGNMENT - EVALUATE ORIGINAL TP FIRST):
   - STEP 1: EVALUATE THE ORIGINAL TP from the signal:
     * Check if original TP is at a realistic support/resistance level (check lower timeframes: 15m, 1h)
     * Check if original TP is achievable (not too far - typically 3-15% from entry is realistic)
     * Check if original TP gives RR ≥ 1:3 (minimum requirement)
     * Check if original TP aligns with market structure and liquidity objectives
   
   - STEP 2: DECISION - Keep or Optimize:
     * If original TP is GOOD (realistic, achievable, RR ≥ 1:3, at support/resistance): KEEP IT (set suggested_take_profit to null)
     * If original TP is TOO AGGRESSIVE (won't hit, too far, unrealistic): Suggest LOWER TP (only if technically justified)
     * If original TP is TOO CONSERVATIVE (leaves profit on table, can go further): Suggest HIGHER TP (only if technically justified)
     * If original TP gives RR < 1:3: Find nearest level that achieves RR ≥ 1:3 (only if technically justified)
     * DO NOT suggest changes just to make a 1-2% adjustment - only change if technically justified
   
   - STEP 3: If optimization needed (SOLID TECHNICAL REASON):
     * MAXIMUM distance: Keep within 1-2% of original to ensure orders FILL (this is a LIMIT, not a requirement)
     * Only suggest if new TP is clearly better AND within 1-2% of original
   
   - STEP 3: If optimization needed, use LOWER TIMEFRAMES (15m, 1h):
     * Check market_data['lower_timeframe_levels']['15m']['resistance_levels'] and market_data['lower_timeframe_levels']['1h']['resistance_levels'] for LONG
     * Check market_data['lower_timeframe_levels']['15m']['support_levels'] and market_data['lower_timeframe_levels']['1h']['support_levels'] for SHORT
     * LONG: Find nearest resistance level ABOVE entry (within 1-2% of original TP, typically 3-8% from entry, max 15% if structure requires)
     * SHORT: Find nearest support level BELOW entry (within 1-2% of original TP, typically 3-8% from entry, max 15% if structure requires)
     * TP should be REALISTIC and ACHIEVABLE - not 20-30% away
     * DO NOT suggest very wide TP (15%+) unless absolutely necessary for RR ≥ 1:3
   
   - CRITICAL RULES:
     * ALWAYS evaluate original TP first - don't blindly suggest new TP
     * If original TP is good OR within 1-2% of optimal level, KEEP IT (set suggested_take_profit to null)
     * Only suggest new TP if original is clearly problematic AND new TP is within 1-2% of original
     * When suggesting new TP, explain WHY original TP needs adjustment in price_suggestion_reasoning

CALCULATION METHOD (Institutional Approach):
- IDENTIFY INSTITUTIONAL ZONES: Order blocks, FVGs, liquidity pools, stop-hunt zones
- VALIDATE STRUCTURE: HTF → LTF alignment, BOS/CHoCH, swing points (HTF for trend direction, LTF for precise levels)
- CALCULATE ENTRY: At institutional liquidity zone (order block, FVG, or liquidity pool) - can use HTF for direction
- CALCULATE SL: At NEAREST support/resistance on LTF (15m, 1h) - NOT HTF (HTF SL would be too wide)
- CALCULATE TP: At NEAREST resistance/support on LTF (15m, 1h) - NOT HTF targets (HTF targets are too far, 20-30% away)
- VALIDATE RR: Must be ≥ 1:3 (if not, modify or discard)
- CRITICAL: HTF is for TREND DIRECTION and ENTRY optimization, LTF (15m, 1h) is for SL/TP placement (tighter, more achievable)

FINAL VALIDATION (CRITICAL - CHECK BEFORE RETURNING):
Before returning your response, VALIDATE that:
1. Entry 2 is DIFFERENT from Stop Loss:
   - For LONG: Entry 2 > SL (Entry 2 must be at least 0.5% above SL)
   - For SHORT: Entry 2 < SL (Entry 2 must be at least 0.5% below SL)
   - If Entry 2 == SL, this is a CRITICAL ERROR - adjust one of them
2. All prices are properly spaced:
   - Entry 1, Entry 2, SL, TP must all be different prices
   - Entry 2 must be between Entry 1 and SL
   - SL must be beyond Entry 2 (further from Entry 1)
3. If you find Entry 2 == SL, you MUST fix it:
   - For LONG: Move SL lower or Entry 2 higher (whichever is more technically justified)
   - For SHORT: Move SL higher or Entry 2 lower (whichever is more technically justified)

If original prices are already optimal (at institutional zones, RR ≥ 1:3, Entry 2 ≠ SL), you may omit suggestion fields (they will use original).
If you suggest prices, they will be APPLIED if they improve the trade (better institutional entry, tighter SL, higher TP, better RR).
If setup is weak, counter-trend, or lacks institutional confirmation, MODIFY or DISCARD entirely."""
    
    try:
        # Call Gemini API with timeout
        logger.info(f"🤖 [AI VALIDATION] Starting validation for NEW ENTRY signal: {symbol} ({signal_side}) @ ${entry_price:,.8f}")
        logger.info(f"🤖 [AI VALIDATION] This validation ONLY runs for new ENTRY signals, NOT for order tracking or TP creation")
        
        # Check if gemini_client is available
        if not gemini_client:
            logger.warning("⚠️ Gemini client is None - AI validation will be skipped (fail-open)")
            return {
                'is_valid': True,
                'confidence_score': 100.0,
                'reasoning': 'Gemini client not initialized, proceeding (fail-open)',
                'risk_level': 'MEDIUM'
            }
        
        logger.info(f"📡 Using Gemini model: {gemini_model_name}")
        logger.debug(f"📤 AI PROMPT (full):\n{prompt}")  # Only log at DEBUG level to reduce log size
        start_time = time.time()
        
        # Use threading to implement timeout
        result_container = {'response': None, 'error': None}
        
        def call_api():
            global gemini_client, gemini_model_name
            try:
                # Try current model first
                logger.info(f"📡 Calling Gemini API with model: {gemini_model_name}")
                response = gemini_client.generate_content(prompt)
                result_container['response'] = response.text
                logger.info(f"✅ Gemini API call successful, received response (length: {len(response.text)} chars)")
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"⚠️ Gemini API call failed with model {gemini_model_name}: {error_msg}")
                
                # Check if we should try alternative models
                should_try_alternatives = False
                # Check if model requires billing (limit: 0 means not available in free tier)
                is_paid_model = 'limit: 0' in error_msg.lower() or '2.5-pro' in gemini_model_name.lower()
                
                if 'not found' in error_msg.lower() or 'not supported' in error_msg.lower():
                    logger.warning(f"🔄 Model {gemini_model_name} not available, trying alternative FREE TIER models...")
                    should_try_alternatives = True
                elif is_paid_model:
                    logger.warning(f"🔄 Model {gemini_model_name} requires billing (not free tier), trying alternative FREE TIER models...")
                    should_try_alternatives = True
                elif '429' in error_msg or 'quota' in error_msg.lower() or 'rate limit' in error_msg.lower():
                    logger.warning(f"🔄 Model {gemini_model_name} quota exceeded, trying alternative FREE TIER models with different limits...")
                    should_try_alternatives = True
                
                # Try alternative models if applicable
                if should_try_alternatives:
                    # Try other FREE TIER models in order of preference (skip current model and paid models)
                    paid_models_to_skip = ['2.5-pro', '2.0-pro', 'ultra']
                    for alt_model_name in GEMINI_MODEL_NAMES:
                        if alt_model_name == gemini_model_name:
                            continue
                        # Skip paid models
                        if any(paid in alt_model_name.lower() for paid in paid_models_to_skip):
                            logger.debug(f"⏭️ Skipping paid model: {alt_model_name}")
                            continue
                        try:
                            logger.info(f"🔄 Trying alternative FREE TIER model: {alt_model_name}")
                            alt_client = genai.GenerativeModel(alt_model_name)
                            response = alt_client.generate_content(prompt)
                            result_container['response'] = response.text
                            logger.info(f"✅ Successfully used alternative FREE TIER model: {alt_model_name}")
                            # Update global client for future use
                            gemini_client = alt_client
                            gemini_model_name = alt_model_name
                            return
                        except Exception as alt_e:
                            alt_error = str(alt_e)
                            # Check if this is a paid model error (limit: 0)
                            if 'limit: 0' in alt_error.lower():
                                logger.debug(f"⏭️ Model {alt_model_name} requires billing, skipping...")
                                continue
                            # Check if quota error - but only skip if it's the SAME model family (they share quota)
                            # Different model families have SEPARATE quotas (e.g., gemini-2.5-flash vs gemini-2.5-flash-lite)
                            if '429' in alt_error or 'quota' in alt_error.lower():
                                # Check if it's the same model family (they share quota pool)
                                current_model_family = gemini_model_name.split('-')[1] if '-' in gemini_model_name else ''
                                alt_model_family = alt_model_name.split('-')[1] if '-' in alt_model_name else ''
                                # If same family (e.g., both 2.5-flash), skip (same quota pool)
                                # If different family (e.g., 2.5-flash vs 2.5-flash-lite), try it (different quota)
                                if current_model_family == alt_model_family and 'lite' not in alt_model_name.lower() and '3-flash' not in alt_model_name.lower():
                                    logger.debug(f"⚠️ Model {alt_model_name} shares quota pool with {gemini_model_name}, skipping...")
                                else:
                                    logger.debug(f"⚠️ Model {alt_model_name} has quota issues but different quota pool, will try anyway...")
                                    # Don't skip - different quota pools might work
                            else:
                                logger.debug(f"❌ Alternative model {alt_model_name} also failed: {alt_error}")
                            continue
                
                # If no alternative worked, return original error
                result_container['error'] = error_msg
                logger.error(f"❌ All Gemini models failed. Last error: {error_msg}")
        
        api_thread = threading.Thread(target=call_api, daemon=True)
        api_thread.start()
        
        # Wait for response or error (with safety timeout to prevent infinite hangs)
        # Using 60s safety timeout - should be enough even for slow free tier
        api_thread.join(timeout=60)
        
        # Wait until we get a response or error (or safety timeout)
        while api_thread.is_alive() and not result_container['response'] and not result_container['error']:
            time.sleep(0.1)  # Small sleep to avoid busy-waiting
            # Check if we've exceeded safety timeout (60s)
            elapsed = time.time() - start_time
            if elapsed > 60:
                logger.warning(f"⏱️ AI validation safety timeout for {symbol} (60s), proceeding without validation (fail-open)")
            return {
                'is_valid': True,
                    'confidence_score': 100.0,  # High score to pass threshold - fail-open design
                    'reasoning': 'AI validation safety timeout, proceeding (fail-open)',
                'risk_level': 'MEDIUM'
            }
        
        # Check for error first - handle gracefully with fail-open design
        if result_container['error']:
            error_msg = result_container['error']
            logger.warning(f"⚠️ AI validation error for {symbol}: {error_msg}")
            
            # Check if it's a quota/rate limit error
            if '429' in error_msg or 'quota' in error_msg.lower() or 'rate limit' in error_msg.lower():
                logger.warning(f"⚠️ Gemini API quota/rate limit exceeded for {symbol}. Proceeding without validation (fail-open)")
                return {
                    'is_valid': True,
                    'confidence_score': 100.0,  # High score to pass threshold - fail-open design
                    'reasoning': f'AI validation unavailable (quota exceeded), proceeding (fail-open)',
                    'risk_level': 'MEDIUM'
                }
            
            # For other errors, also fail-open (don't block trading)
            logger.warning(f"⚠️ AI validation error for {symbol}, proceeding without validation (fail-open)")
            return {
                'is_valid': True,
                'confidence_score': 100.0,  # High score to pass threshold - fail-open design
                'reasoning': f'AI validation error: {error_msg[:200]}... Proceeding without validation (fail-open)',
                'risk_level': 'MEDIUM'
            }
        
        # Check for response
        if result_container['response']:
            logger.info(f"✅ Received AI response after {time.time() - start_time:.2f}s")
        else:
            # No response and no error - should not happen, but fail-open
            logger.warning(f"⚠️ No response or error from AI validation for {symbol}, proceeding (fail-open)")
            return {
                'is_valid': True,
                'confidence_score': 100.0,
                'reasoning': 'No response from AI, proceeding (fail-open)',
                'risk_level': 'MEDIUM'
            }
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ AI validation API call completed in {elapsed_time:.2f}s")
        
        # Parse response
        response_text = result_container['response'].strip()
        
        # Log full response for debugging
        logger.debug(f"📥 AI RESPONSE (full):\n{response_text}")  # Only log at DEBUG level to reduce log size
        
        # Try to extract JSON from response (AI might wrap it in markdown or text)
        import re
        # More flexible regex to capture full JSON including nested objects and optional fields
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"is_valid"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        else:
            # Fallback: try to find JSON block with braces
            json_match = re.search(r'\{.*"is_valid".*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
        
        # Parse JSON response
        logger.info(f"📝 Parsing AI response (length: {len(response_text)} chars)")
        try:
            validation_result = json.loads(response_text)
            logger.info(f"✅ Successfully parsed AI response JSON")
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ Failed to parse AI response as JSON: {e}")
            # Try to extract values manually if JSON parsing fails
            logger.warning(f"Failed to parse AI response as JSON, attempting manual extraction")
            is_valid = 'true' in response_text.lower() or '"is_valid": true' in response_text.lower()
            confidence_match = re.search(r'"confidence_score":\s*(\d+(?:\.\d+)?)', response_text)
            confidence_score = float(confidence_match.group(1)) if confidence_match else 50.0
            
            reasoning_match = re.search(r'"reasoning":\s*"([^"]+)"', response_text)
            reasoning = reasoning_match.group(1) if reasoning_match else "AI validation completed"
            
            risk_match = re.search(r'"risk_level":\s*"([^"]+)"', response_text)
            risk_level = risk_match.group(1) if risk_match else "MEDIUM"
            
            # Extract optional price suggestions
            suggested_entry_match = re.search(r'"suggested_entry_price":\s*([\d.]+|null)', response_text)
            suggested_entry = float(suggested_entry_match.group(1)) if suggested_entry_match and suggested_entry_match.group(1) != 'null' else None
            
            suggested_entry2_match = re.search(r'"suggested_second_entry_price":\s*([\d.]+|null)', response_text)
            suggested_entry2 = float(suggested_entry2_match.group(1)) if suggested_entry2_match and suggested_entry2_match.group(1) != 'null' else None
            
            suggested_sl_match = re.search(r'"suggested_stop_loss":\s*([\d.]+|null)', response_text)
            suggested_sl = float(suggested_sl_match.group(1)) if suggested_sl_match and suggested_sl_match.group(1) != 'null' else None
            
            suggested_tp_match = re.search(r'"suggested_take_profit":\s*([\d.]+|null)', response_text)
            suggested_tp = float(suggested_tp_match.group(1)) if suggested_tp_match and suggested_tp_match.group(1) != 'null' else None
            
            price_reasoning_match = re.search(r'"price_suggestion_reasoning":\s*"([^"]+)"', response_text)
            price_reasoning = price_reasoning_match.group(1) if price_reasoning_match else ""
            
            validation_result = {
                'is_valid': is_valid,
                'confidence_score': confidence_score,
                'reasoning': reasoning,
                'risk_level': risk_level
            }
            
            # Add price suggestions if found
            if suggested_entry or suggested_entry2 or suggested_sl or suggested_tp:
                if suggested_entry:
                    validation_result['suggested_entry_price'] = suggested_entry
                if suggested_entry2:
                    validation_result['suggested_second_entry_price'] = suggested_entry2
                if suggested_sl:
                    validation_result['suggested_stop_loss'] = suggested_sl
                if suggested_tp:
                    validation_result['suggested_take_profit'] = suggested_tp
                if price_reasoning:
                    validation_result['price_suggestion_reasoning'] = price_reasoning
        
        # Validate response structure
        if 'is_valid' not in validation_result:
            validation_result['is_valid'] = True  # Fail-open
        
        if 'confidence_score' not in validation_result:
            validation_result['confidence_score'] = 50.0
        
        if 'reasoning' not in validation_result:
            validation_result['reasoning'] = 'AI validation completed'
        
        if 'risk_level' not in validation_result:
            validation_result['risk_level'] = 'MEDIUM'
        
        # Ensure confidence_score is within valid range
        validation_result['confidence_score'] = max(0, min(100, float(validation_result['confidence_score'])))
        
        # Log validation result
        logger.info(f"📊 AI Validation Result:")
        logger.info(f"   ✅ Valid: {validation_result.get('is_valid', True)}")
        logger.info(f"   📈 Confidence: {validation_result['confidence_score']:.1f}%")
        logger.info(f"   ⚠️  Risk Level: {validation_result.get('risk_level', 'MEDIUM')}")
        # Log full reasoning (split into multiple lines if very long for readability)
        reasoning = validation_result.get('reasoning', 'N/A')
        if len(reasoning) > 500:
            # Split long reasoning into multiple log lines
            logger.info(f"   💭 Reasoning (full):")
            # Split by sentences or chunks of 500 chars
            chunks = [reasoning[i:i+500] for i in range(0, len(reasoning), 500)]
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"      [{i}/{len(chunks)}] {chunk}")
        else:
            logger.info(f"   💭 Reasoning: {reasoning}")
        
        # Extract price suggestions and apply smart optimization
        original_entry2 = safe_float(signal_data.get('second_entry_price'), default=None)
        optimized_prices = {
            'entry_price': entry_price,  # Default to original
            'stop_loss': stop_loss,      # Default to original
            'take_profit': take_profit,  # Default to original
            'second_entry_price': original_entry2,   # Default to original, will be optimized if AI suggests or Entry 1 is optimized
            'applied_optimizations': []
        }
        
        if ENABLE_AI_PRICE_SUGGESTIONS:
            suggested_entry = safe_float(validation_result.get('suggested_entry_price'), default=None)
            suggested_entry2 = safe_float(validation_result.get('suggested_second_entry_price'), default=None)
            suggested_sl = safe_float(validation_result.get('suggested_stop_loss'), default=None)
            suggested_tp = safe_float(validation_result.get('suggested_take_profit'), default=None)
            price_reasoning = validation_result.get('price_suggestion_reasoning', '')
            original_entry2 = safe_float(signal_data.get('second_entry_price'), default=None)
            
            # Smart optimization logic: Only apply if AI suggestion is BETTER
            if suggested_entry:
                if signal_side == 'LONG':
                    # For LONG: Lower entry is better (closer to support)
                    if suggested_entry < entry_price:
                        optimized_prices['entry_price'] = suggested_entry
                        optimized_prices['applied_optimizations'].append(f"Entry optimized: ${entry_price:,.8f} → ${suggested_entry:,.8f} (better entry for LONG)")
                        logger.info(f"✅ [AI OPTIMIZATION] Entry optimized for LONG: ${entry_price:,.8f} → ${suggested_entry:,.8f} (better entry)")
                    else:
                        logger.info(f"⚠️  [AI OPTIMIZATION] Entry suggestion ${suggested_entry:,.8f} is HIGHER than original ${entry_price:,.8f} - keeping original (better for LONG)")
                else:  # SHORT
                    # For SHORT: Higher entry is better (closer to resistance)
                    if suggested_entry > entry_price:
                        optimized_prices['entry_price'] = suggested_entry
                        optimized_prices['applied_optimizations'].append(f"Entry optimized: ${entry_price:,.8f} → ${suggested_entry:,.8f} (better entry for SHORT)")
                        logger.info(f"✅ [AI OPTIMIZATION] Entry optimized for SHORT: ${entry_price:,.8f} → ${suggested_entry:,.8f} (better entry)")
                    else:
                        logger.info(f"⚠️  [AI OPTIMIZATION] Entry suggestion ${suggested_entry:,.8f} is LOWER than original ${entry_price:,.8f} - keeping original (better for SHORT)")
            
            # Entry 2 optimization (independent of Entry 1)
            if suggested_entry2 and original_entry2:
                # Validate Entry 2 is in correct direction relative to Entry 1
                current_entry1 = optimized_prices.get('entry_price', entry_price)
                
                if signal_side == 'LONG':
                    # For LONG: Entry 2 should be BELOW Entry 1
                    if suggested_entry2 < current_entry1:
                        # Validate it's better than original Entry 2 (lower is better for LONG)
                        if suggested_entry2 < original_entry2:
                            optimized_prices['second_entry_price'] = suggested_entry2
                            optimized_prices['applied_optimizations'].append(f"Entry 2 optimized: ${original_entry2:,.8f} → ${suggested_entry2:,.8f} (better entry for LONG)")
                            logger.info(f"✅ [AI OPTIMIZATION] Entry 2 optimized for LONG: ${original_entry2:,.8f} → ${suggested_entry2:,.8f} (better entry)")
                        elif suggested_entry2 == original_entry2:
                            # Same as original - keep it
                            optimized_prices['second_entry_price'] = suggested_entry2
                            logger.info(f"ℹ️  [AI OPTIMIZATION] Entry 2 suggestion ${suggested_entry2:,.8f} matches original - keeping it")
                        else:
                            logger.info(f"⚠️  [AI OPTIMIZATION] Entry 2 suggestion ${suggested_entry2:,.8f} is HIGHER than original ${original_entry2:,.8f} - keeping original (better for LONG)")
                    else:
                        logger.info(f"⚠️  [AI OPTIMIZATION] Entry 2 suggestion ${suggested_entry2:,.8f} is ABOVE Entry 1 ${current_entry1:,.8f} - invalid for LONG, keeping original")
                else:  # SHORT
                    # For SHORT: Entry 2 should be ABOVE Entry 1
                    if suggested_entry2 > current_entry1:
                        # Validate it's better than original Entry 2 (higher is better for SHORT)
                        if suggested_entry2 > original_entry2:
                            optimized_prices['second_entry_price'] = suggested_entry2
                            optimized_prices['applied_optimizations'].append(f"Entry 2 optimized: ${original_entry2:,.8f} → ${suggested_entry2:,.8f} (better entry for SHORT)")
                            logger.info(f"✅ [AI OPTIMIZATION] Entry 2 optimized for SHORT: ${original_entry2:,.8f} → ${suggested_entry2:,.8f} (better entry)")
                        elif suggested_entry2 == original_entry2:
                            # Same as original - keep it
                            optimized_prices['second_entry_price'] = suggested_entry2
                            logger.info(f"ℹ️  [AI OPTIMIZATION] Entry 2 suggestion ${suggested_entry2:,.8f} matches original - keeping it")
                        else:
                            logger.info(f"⚠️  [AI OPTIMIZATION] Entry 2 suggestion ${suggested_entry2:,.8f} is LOWER than original ${original_entry2:,.8f} - keeping original (better for SHORT)")
                    else:
                        logger.info(f"⚠️  [AI OPTIMIZATION] Entry 2 suggestion ${suggested_entry2:,.8f} is BELOW Entry 1 ${current_entry1:,.8f} - invalid for SHORT, keeping original")
            elif suggested_entry2 and not original_entry2:
                # AI suggests Entry 2 but original signal didn't have one - use AI suggestion if valid
                current_entry1 = optimized_prices.get('entry_price', entry_price)
                if signal_side == 'LONG' and suggested_entry2 < current_entry1:
                    optimized_prices['second_entry_price'] = suggested_entry2
                    optimized_prices['applied_optimizations'].append(f"Entry 2 added: ${suggested_entry2:,.8f} (AI suggested, original was missing)")
                    logger.info(f"✅ [AI OPTIMIZATION] Entry 2 added for LONG: ${suggested_entry2:,.8f} (original was missing)")
                elif signal_side == 'SHORT' and suggested_entry2 > current_entry1:
                    optimized_prices['second_entry_price'] = suggested_entry2
                    optimized_prices['applied_optimizations'].append(f"Entry 2 added: ${suggested_entry2:,.8f} (AI suggested, original was missing)")
                    logger.info(f"✅ [AI OPTIMIZATION] Entry 2 added for SHORT: ${suggested_entry2:,.8f} (original was missing)")
            
            if suggested_sl and stop_loss:
                if signal_side == 'LONG':
                    # For LONG: Tighter SL (higher) is better, but must stay below entry
                    if suggested_sl > stop_loss and suggested_sl < optimized_prices['entry_price']:
                        optimized_prices['stop_loss'] = suggested_sl
                        optimized_prices['applied_optimizations'].append(f"SL optimized: ${stop_loss:,.8f} → ${suggested_sl:,.8f} (tighter risk)")
                        logger.info(f"✅ [AI OPTIMIZATION] SL optimized for LONG: ${stop_loss:,.8f} → ${suggested_sl:,.8f} (tighter)")
                    elif suggested_sl < stop_loss:
                        # Even tighter SL - apply if safe
                        optimized_prices['stop_loss'] = suggested_sl
                        optimized_prices['applied_optimizations'].append(f"SL optimized: ${stop_loss:,.8f} → ${suggested_sl:,.8f} (tighter risk)")
                        logger.info(f"✅ [AI OPTIMIZATION] SL optimized for LONG: ${stop_loss:,.8f} → ${suggested_sl:,.8f} (tighter)")
                    else:
                        logger.info(f"⚠️  [AI OPTIMIZATION] SL suggestion ${suggested_sl:,.8f} rejected (wider than original or invalid)")
                else:  # SHORT
                    # For SHORT: Tighter SL (lower) is better, but must stay above entry
                    if suggested_sl < stop_loss and suggested_sl > optimized_prices['entry_price']:
                        optimized_prices['stop_loss'] = suggested_sl
                        optimized_prices['applied_optimizations'].append(f"SL optimized: ${stop_loss:,.8f} → ${suggested_sl:,.8f} (tighter risk)")
                        logger.info(f"✅ [AI OPTIMIZATION] SL optimized for SHORT: ${stop_loss:,.8f} → ${suggested_sl:,.8f} (tighter)")
                    elif suggested_sl > stop_loss:
                        # Even tighter SL - apply if safe
                        optimized_prices['stop_loss'] = suggested_sl
                        optimized_prices['applied_optimizations'].append(f"SL optimized: ${stop_loss:,.8f} → ${suggested_sl:,.8f} (tighter risk)")
                        logger.info(f"✅ [AI OPTIMIZATION] SL optimized for SHORT: ${stop_loss:,.8f} → ${suggested_sl:,.8f} (tighter)")
                    else:
                        logger.info(f"⚠️  [AI OPTIMIZATION] SL suggestion ${suggested_sl:,.8f} rejected (wider than original or invalid)")
            
            if suggested_tp and take_profit:
                # Smart TP optimization: Use AI's analysis to determine BEST TP
                # AI analyzes resistance/support levels, reversal risk, and realistic targets
                # Trust AI's analysis - it considers if original TP might miss by 0.1-0.5% due to reversal
                tp_diff_pct = abs((suggested_tp - take_profit) / take_profit * 100) if take_profit else 0
                
                if signal_side == 'LONG':
                    # For LONG: AI analyzes resistance levels to find BEST TP
                    # Apply AI TP if:
                    # 1. Higher (more profit) OR
                    # 2. Better positioned at resistance (even if slightly lower, within 2% - avoids reversal risk)
                    if suggested_tp > take_profit:
                        # Higher TP = more profit, apply it
                        optimized_prices['take_profit'] = suggested_tp
                        optimized_prices['applied_optimizations'].append(f"TP optimized: ${take_profit:,.8f} → ${suggested_tp:,.8f} (higher profit, +{tp_diff_pct:.2f}%)")
                        logger.info(f"✅ [AI OPTIMIZATION] TP optimized for LONG: ${take_profit:,.8f} → ${suggested_tp:,.8f} (higher profit, +{tp_diff_pct:.2f}%)")
                    elif suggested_tp >= take_profit * 0.98:  # Within 2% of original (AI found better resistance level, avoids reversal)
                        # AI TP is slightly lower but better positioned at resistance - use it (avoids missing TP by 0.1-0.5%)
                        optimized_prices['take_profit'] = suggested_tp
                        optimized_prices['applied_optimizations'].append(f"TP optimized: ${take_profit:,.8f} → ${suggested_tp:,.8f} (better resistance level, avoids reversal, -{tp_diff_pct:.2f}%)")
                        logger.info(f"✅ [AI OPTIMIZATION] TP optimized for LONG: ${take_profit:,.8f} → ${suggested_tp:,.8f} (better positioned at resistance, avoids reversal risk, -{tp_diff_pct:.2f}%)")
                    else:
                        # AI TP is significantly lower (>2%) - keep original (it's better)
                        logger.info(f"⚠️  [AI OPTIMIZATION] TP suggestion ${suggested_tp:,.8f} is {tp_diff_pct:.2f}% LOWER than original ${take_profit:,.8f} - keeping original (better profit)")
                else:  # SHORT
                    # For SHORT: AI analyzes support levels to find BEST TP
                    # Apply AI TP if:
                    # 1. Lower (more profit) OR
                    # 2. Better positioned at support (even if slightly higher, within 2% - avoids reversal risk)
                    if suggested_tp < take_profit:
                        # Lower TP = more profit, apply it
                        optimized_prices['take_profit'] = suggested_tp
                        optimized_prices['applied_optimizations'].append(f"TP optimized: ${take_profit:,.8f} → ${suggested_tp:,.8f} (higher profit, -{tp_diff_pct:.2f}%)")
                        logger.info(f"✅ [AI OPTIMIZATION] TP optimized for SHORT: ${take_profit:,.8f} → ${suggested_tp:,.8f} (higher profit, -{tp_diff_pct:.2f}%)")
                    elif suggested_tp <= take_profit * 1.02:  # Within 2% of original (AI found better support level, avoids reversal)
                        # AI TP is slightly higher but better positioned at support - use it (avoids missing TP by 0.1-0.5%)
                        optimized_prices['take_profit'] = suggested_tp
                        optimized_prices['applied_optimizations'].append(f"TP optimized: ${take_profit:,.8f} → ${suggested_tp:,.8f} (better support level, avoids reversal, +{tp_diff_pct:.2f}%)")
                        logger.info(f"✅ [AI OPTIMIZATION] TP optimized for SHORT: ${take_profit:,.8f} → ${suggested_tp:,.8f} (better positioned at support, avoids reversal risk, +{tp_diff_pct:.2f}%)")
                    else:
                        # AI TP is significantly higher (>2%) - keep original (it's better)
                        logger.info(f"⚠️  [AI OPTIMIZATION] TP suggestion ${suggested_tp:,.8f} is {tp_diff_pct:.2f}% HIGHER than original ${take_profit:,.8f} - keeping original (better profit)")
            
            # Store suggestions for logging (even if not applied)
            if suggested_entry or suggested_entry2 or suggested_sl or suggested_tp:
                validation_result['price_suggestions'] = {
                    'entry_price': suggested_entry,
                    'second_entry_price': suggested_entry2,
                    'stop_loss': suggested_sl,
                    'take_profit': suggested_tp,
                    'reasoning': price_reasoning,
                    'original_entry': entry_price,
                    'original_second_entry': original_entry2,
                    'original_stop_loss': stop_loss,
                    'original_take_profit': take_profit,
                    'optimized_entry': optimized_prices['entry_price'],
                    'optimized_stop_loss': optimized_prices['stop_loss'],
                    'optimized_take_profit': optimized_prices['take_profit'],
                    'applied_optimizations': optimized_prices['applied_optimizations']
                }
                
                # Log comparison with applied optimizations
                logger.info(f"💡 [AI PRICE OPTIMIZATION] Analysis for {symbol}:")
                logger.info(f"   ┌─────────────────────────────────────────────────────────────────────────────┐")
                logger.info(f"   │ PRICE COMPARISON: Original (TradingView) vs AI Suggested vs Applied       │")
                logger.info(f"   ├─────────────────────────────────────────────────────────────────────────────┤")
                if suggested_entry:
                    diff_pct = ((suggested_entry - entry_price) / entry_price * 100) if entry_price else 0
                    applied = optimized_prices['entry_price']
                    applied_diff = ((applied - entry_price) / entry_price * 100) if entry_price else 0
                    status = "✅ APPLIED" if applied != entry_price else "❌ REJECTED (worse)"
                    logger.info(f"   │ 📍 Entry:      Original=${entry_price:,.8f}  →  AI=${suggested_entry:,.8f} ({diff_pct:+.2f}%)  →  Applied=${applied:,.8f} ({applied_diff:+.2f}%) {status} │")
                else:
                    logger.info(f"   │ 📍 Entry:      Original=${entry_price:,.8f}  →  No AI suggestion (keeping original)                    │")
                if suggested_entry2:
                    if original_entry2:
                        diff_pct = ((suggested_entry2 - original_entry2) / original_entry2 * 100) if original_entry2 else 0
                        applied = optimized_prices.get('second_entry_price', original_entry2)
                        applied_diff = ((applied - original_entry2) / original_entry2 * 100) if original_entry2 else 0
                        # Status: APPLIED if changed, KEPT if matches original (not rejected)
                        if applied != original_entry2:
                            status = "✅ APPLIED"
                        elif abs(diff_pct) < 0.01:  # Matches original (within 0.01%)
                            status = "✅ KEPT (matches original)"
                        else:
                            status = "❌ REJECTED (worse)"
                        logger.info(f"   │ 📍 Entry 2:    Original=${original_entry2:,.8f}  →  AI=${suggested_entry2:,.8f} ({diff_pct:+.2f}%)  →  Applied=${applied:,.8f} ({applied_diff:+.2f}%) {status} │")
                    else:
                        applied = optimized_prices.get('second_entry_price')
                        status = "✅ APPLIED" if applied else "❌ REJECTED"
                        logger.info(f"   │ 📍 Entry 2:    Original=N/A (not provided)  →  AI=${suggested_entry2:,.8f}  →  Applied=${applied:,.8f if applied else 'N/A'} {status} │")
                elif original_entry2:
                    logger.info(f"   │ 📍 Entry 2:    Original=${original_entry2:,.8f}  →  No AI suggestion (keeping original)                    │")
                if suggested_sl:
                    diff_pct = ((suggested_sl - stop_loss) / stop_loss * 100) if stop_loss else 0
                    applied = optimized_prices['stop_loss']
                    applied_diff = ((applied - stop_loss) / stop_loss * 100) if stop_loss else 0
                    status = "✅ APPLIED" if applied != stop_loss else "❌ REJECTED"
                    logger.info(f"   │ 🛑 Stop Loss:   Original=${stop_loss:,.8f}  →  AI=${suggested_sl:,.8f} ({diff_pct:+.2f}%)  →  Applied=${applied:,.8f} ({applied_diff:+.2f}%) {status} │")
                else:
                    sl_display = f"${stop_loss:,.8f}" if stop_loss else "N/A"
                    logger.info(f"   │ 🛑 Stop Loss:   Original={sl_display:<15}  →  No AI suggestion (keeping original)                    │")
                if suggested_tp:
                    diff_pct = ((suggested_tp - take_profit) / take_profit * 100) if take_profit else 0
                    applied = optimized_prices['take_profit']
                    applied_diff = ((applied - take_profit) / take_profit * 100) if take_profit else 0
                    status = "✅ APPLIED" if applied != take_profit else "❌ REJECTED (worse)"
                    logger.info(f"   │ 🎯 Take Profit: Original=${take_profit:,.8f}  →  AI=${suggested_tp:,.8f} ({diff_pct:+.2f}%)  →  Applied=${applied:,.8f} ({applied_diff:+.2f}%) {status} │")
                else:
                    tp_display = f"${take_profit:,.8f}" if take_profit else "N/A"
                    logger.info(f"   │ 🎯 Take Profit: Original={tp_display:<15}  →  No AI suggestion (keeping original)                    │")
                logger.info(f"   ├─────────────────────────────────────────────────────────────────────────────┤")
                if price_reasoning:
                    reasoning_lines = [price_reasoning[i:i+70] for i in range(0, len(price_reasoning), 70)]
                    for line in reasoning_lines:
                        logger.info(f"   │ 💭 AI Reasoning: {line:<70} │")
                if optimized_prices['applied_optimizations']:
                    logger.info(f"   │ ✅ APPLIED OPTIMIZATIONS: {len(optimized_prices['applied_optimizations'])} price(s) optimized          │")
                    for opt in optimized_prices['applied_optimizations']:
                        logger.info(f"   │    • {opt:<68} │")
                else:
                    logger.info(f"   │ ⚠️  No optimizations applied (AI suggestions were worse than original)      │")
                logger.info(f"   └─────────────────────────────────────────────────────────────────────────────┘")
        
        # Store optimized prices in validation result for use in order creation
        validation_result['optimized_prices'] = optimized_prices
        
        # Cache the result
        validation_cache[cache_key] = (validation_result, current_time)
        
        # Clean up old cache entries (keep last 100)
        if len(validation_cache) > 100:
            sorted_cache = sorted(validation_cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_cache[:-100]:
                del validation_cache[key]
        
        logger.info(f"✅ AI Validation Result for {symbol}: Valid={validation_result['is_valid']}, "
                   f"Confidence={validation_result['confidence_score']:.1f}%, "
                   f"Risk={validation_result['risk_level']}, "
                   f"Reasoning={validation_result['reasoning']}")
        
        return validation_result
        
    except Exception as e:
        logger.warning(f"⚠️ AI validation error for {symbol}: {e}. Proceeding without validation (fail-open)")
        # Return high confidence score to ensure signal passes threshold (fail-open design)
        # This ensures API errors don't block legitimate trading signals
        return {
            'is_valid': True,  # Fail-open: proceed if validation fails
            'confidence_score': 100.0,  # High score to pass threshold - fail-open design
            'reasoning': f'AI validation error: {str(e)}, proceeding (fail-open)',
            'risk_level': 'MEDIUM',
            'error': str(e)
        }
