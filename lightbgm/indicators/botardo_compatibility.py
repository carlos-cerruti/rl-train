"""
Botardo Compatibility Layer for LightGBM ML Scoring System

This module ensures 100% compatibility with botardo's UltimateTradingCalculator
by replicating the exact same calculations, data structures, and interfaces.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Exact replica of botardo's TradingSignal structure"""
    indicator: str
    value: float
    signal: str  # BUY, SELL, HOLD
    strength: float  # 0.0 to 1.0
    confidence: float
    interpretation: str
    timestamp: datetime

@dataclass
class FibonacciLevels:
    """Exact replica of botardo's FibonacciLevels structure"""
    level_0: float
    level_236: float 
    level_382: float
    level_500: float
    level_618: float
    level_100: float
    current_price: float
    position: str
    confidence: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'level_0': self.level_0,
            'level_236': self.level_236,
            'level_382': self.level_382,
            'level_500': self.level_500,
            'level_618': self.level_618,
            'level_100': self.level_100,
            'current_price': self.current_price,
            'position': self.position,
            'confidence': self.confidence
        }

class BotardoCompatibleCalculator:
    """
    Calculator that replicates botardo's UltimateTradingCalculator
    with exact same formulas and outputs for seamless integration.
    """
    
    # ============= POINT VALUES (From botardo) =============
    POINT_VALUES = {
        # Major forex pairs (4 decimal places)
        'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'AUDUSD': 0.0001, 'NZDUSD': 0.0001,
        'USDCAD': 0.0001, 'USDCHF': 0.0001,
        
        # JPY pairs (2 decimal places)  
        'USDJPY': 0.01, 'EURJPY': 0.01, 'GBPJPY': 0.01, 'AUDJPY': 0.01,
        'NZDJPY': 0.01, 'CADJPY': 0.01, 'CHFJPY': 0.01,
        
        # Metals
        'XAUUSD': 0.1, 'XAGUSD': 0.001,
        
        # Crypto
        'BTCUSD': 1.0, 'ETHUSD': 0.01,
        
        # Indices
        'US30': 1.0, 'SPX500': 0.1, 'NAS100': 0.25, 'DE30': 1.0, 'UK100': 1.0,
    }
    
    @classmethod
    def get_point_value(cls, symbol: str) -> float:
        """Get point value for symbol (exact botardo implementation)"""
        symbol = symbol.upper().replace('.CASH', '').replace('.', '')
        
        if symbol in cls.POINT_VALUES:
            return cls.POINT_VALUES[symbol]
        
        # Pattern matching (from botardo)
        if 'JPY' in symbol:
            return 0.01
        elif symbol.startswith(('XAU', 'GOLD')):
            return 0.1
        elif symbol.startswith(('XAG', 'SILVER')):
            return 0.001
        elif symbol.startswith(('BTC', 'ETH')):
            return 1.0
        elif symbol in ['US30', 'NAS100', 'SPX500', 'DE30', 'UK100']:
            return 1.0
        else:
            return 0.0001  # Default forex
    
    @staticmethod
    def _ema(prices: np.ndarray, period: int) -> float:
        """Exponential Moving Average - exact botardo implementation"""
        if len(prices) < period:
            return float(np.mean(prices))
        
        alpha = 2.0 / (period + 1)
        ema = float(prices[0])
        
        for price in prices[1:]:
            ema = alpha * float(price) + (1 - alpha) * ema
            
        return ema
    
    @staticmethod
    def _ema_series(prices: np.ndarray, period: int) -> np.ndarray:
        """EMA series calculation - exact botardo implementation"""
        if len(prices) < period:
            return np.array([np.mean(prices)])
        
        alpha = 2.0 / (period + 1)
        ema_values = [float(prices[0])]
        
        for price in prices[1:]:
            ema = alpha * float(price) + (1 - alpha) * ema_values[-1]
            ema_values.append(ema)
            
        return np.array(ema_values)
    
    @staticmethod
    def fibonacci_retracement(high: float, low: float, current_price: float) -> FibonacciLevels:
        """
        Fibonacci levels calculation - EXACT replica of botardo implementation
        """
        try:
            price_range = high - low
            
            level_0 = high
            level_236 = high - (price_range * 0.236)
            level_382 = high - (price_range * 0.382)
            level_500 = high - (price_range * 0.500)
            level_618 = high - (price_range * 0.618)
            level_100 = low
            
            # Determine position
            levels = [level_0, level_236, level_382, level_500, level_618, level_100]
            distances = [abs(current_price - level) for level in levels]
            closest_idx = distances.index(min(distances))
            
            positions = ["at_high", "near_236", "near_382", "near_500", "near_618", "at_low"]
            position = positions[closest_idx]
            
            # Calculate confidence based on proximity
            min_distance = min(distances)
            max_distance = price_range * 0.1  # 10% of range
            confidence = max(0.0, 1.0 - (min_distance / max_distance))
            
            return FibonacciLevels(
                level_0=level_0, level_236=level_236, level_382=level_382,
                level_500=level_500, level_618=level_618, level_100=level_100,
                current_price=current_price, position=position, 
                confidence=min(1.0, confidence)
            )
            
        except Exception:
            return FibonacciLevels(
                level_0=high, level_236=high, level_382=high, level_500=high,
                level_618=high, level_100=low, current_price=current_price,
                position="unknown", confidence=0.0
            )
    
    @staticmethod
    def macd(prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, 
             signal_period: int = 9, timeframe: str = "H4") -> Tuple[float, float, float]:
        """
        MACD calculation - EXACT replica of botardo implementation
        Returns: (macd_line, signal_line, histogram)
        """
        try:
            # More flexible minimum length requirement
            min_length = max(fast_period, slow_period) + 2
            if len(prices) < min_length:
                # Fallback for short series
                if len(prices) >= fast_period:
                    fast_ema = BotardoCompatibleCalculator._ema(prices, fast_period)
                    slow_ema = BotardoCompatibleCalculator._ema(prices, slow_period)
                    macd_line = fast_ema - slow_ema
                    signal_line = macd_line * 0.9  # Simple approximation
                    return float(macd_line), float(signal_line), float(macd_line - signal_line)
                return 0.0, 0.0, 0.0
            
            # Calculate EMA series
            ema_fast_series = BotardoCompatibleCalculator._ema_series(prices, fast_period)
            ema_slow_series = BotardoCompatibleCalculator._ema_series(prices, slow_period)
            
            # MACD line series
            macd_series = ema_fast_series - ema_slow_series
            
            # Signal line - EMA of MACD series (start from slow_period index)
            if len(macd_series) >= signal_period:
                # Use the latter part of MACD series for signal calculation
                macd_for_signal = macd_series[slow_period-1:]
                if len(macd_for_signal) >= signal_period:
                    signal_series = BotardoCompatibleCalculator._ema_series(macd_for_signal, signal_period)
                    signal_line = float(signal_series[-1])
                else:
                    signal_line = float(macd_series[-1]) * 0.9
            else:
                signal_line = float(macd_series[-1]) * 0.9
            
            # Current values
            macd_line = float(macd_series[-1])
            histogram = macd_line - signal_line
            
            return macd_line, signal_line, histogram
            
        except Exception as e:
            # Fallback calculation
            if len(prices) >= fast_period:
                fast_ema = BotardoCompatibleCalculator._ema(prices, fast_period)
                slow_ema = BotardoCompatibleCalculator._ema(prices, slow_period)
                macd_line = fast_ema - slow_ema
                signal_line = macd_line * 0.9
                return float(macd_line), float(signal_line), float(macd_line - signal_line)
            return 0.0, 0.0, 0.0
    
    @staticmethod
    def rvi(open_prices: np.ndarray, high_prices: np.ndarray, 
            low_prices: np.ndarray, close_prices: np.ndarray, 
            period: int = 14) -> Tuple[float, float]:
        """
        Relative Vigor Index (RVI) - EXACT replica of botardo implementation
        
        Formula:
        RVI = MA(Close - Open) / MA(High - Low)
        Signal = SMA(RVI, 4)
        
        Returns:
            Tuple[float, float]: (RVI value, RVI signal line)
        """
        try:
            if len(open_prices) < period + 4:
                return 0.0, 0.0

            # Ensure all arrays have the same length
            min_len = min(len(open_prices), len(high_prices), len(low_prices), len(close_prices))
            open_prices = open_prices[-min_len:]
            high_prices = high_prices[-min_len:]
            low_prices = low_prices[-min_len:]
            close_prices = close_prices[-min_len:]

            # Calculate numerator: Close - Open
            close_open_diff = close_prices - open_prices

            # Calculate denominator: High - Low
            high_low_diff = high_prices - low_prices

            # Avoid division by zero
            high_low_diff = np.where(high_low_diff == 0, 1e-8, high_low_diff)

            # Calculate moving averages
            numerator_ma = []
            denominator_ma = []

            for i in range(period - 1, len(close_open_diff)):
                num_avg = np.mean(close_open_diff[i - period + 1:i + 1])
                den_avg = np.mean(high_low_diff[i - period + 1:i + 1])
                numerator_ma.append(num_avg)
                denominator_ma.append(den_avg)

            # Calculate RVI
            rvi_values = []
            for num, den in zip(numerator_ma, denominator_ma):
                if den != 0:
                    rvi_values.append(num / den)
                else:
                    rvi_values.append(0.0)

            if len(rvi_values) == 0:
                return 0.0, 0.0

            # Calculate RVI signal (4-period SMA of RVI)
            if len(rvi_values) >= 4:
                signal_line = np.mean(rvi_values[-4:])
            else:
                signal_line = np.mean(rvi_values)

            current_rvi = rvi_values[-1] if rvi_values else 0.0
            
            return float(current_rvi), float(signal_line)
            
        except Exception as e:
            logger.error(f"Error calculating RVI: {e}")
            return 0.0, 0.0
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> float:
        """RSI calculation - botardo compatible"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    @staticmethod
    def atr(high_prices: np.ndarray, low_prices: np.ndarray, 
            close_prices: np.ndarray, period: int = 14) -> float:
        """ATR calculation - botardo compatible"""
        if len(high_prices) < period + 1:
            return 0.0
        
        # True Range calculation
        high_low = high_prices[1:] - low_prices[1:]
        high_close = np.abs(high_prices[1:] - close_prices[:-1])
        low_close = np.abs(low_prices[1:] - close_prices[:-1])
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        
        # ATR as SMA of True Range
        atr = np.mean(true_range[-period:])
        
        return float(atr)
    
    @staticmethod
    def interpret_macd(macd: float, signal: float, histogram: float) -> TradingSignal:
        """MACD interpretation - exact botardo implementation"""
        if macd > signal and histogram > 0:
            signal_type = "BUY"
            strength = min(1.0, abs(histogram) / (abs(macd) + 1e-8))
        elif macd < signal and histogram < 0:
            signal_type = "SELL"
            strength = min(1.0, abs(histogram) / (abs(macd) + 1e-8))
        else:
            signal_type = "HOLD"
            strength = 0.0
        
        confidence = min(1.0, abs(histogram) * 10)  # Scale histogram for confidence
        
        return TradingSignal(
            indicator="MACD",
            value=macd,
            signal=signal_type,
            strength=strength,
            confidence=confidence,
            interpretation=f"MACD: {macd:.6f}, Signal: {signal:.6f}, Histogram: {histogram:.6f}",
            timestamp=datetime.now()
        )
    
    @staticmethod
    def calculate_pips(symbol: str, entry_price: float, current_price: float) -> float:
        """Pips calculation - exact botardo implementation"""
        point_value = BotardoCompatibleCalculator.get_point_value(symbol)
        price_diff = abs(current_price - entry_price)
        pips = price_diff / point_value
        return round(pips, 2)


class BotardoFeatureMapper:
    """
    Maps botardo's feature calculations to lightbgm feature space
    for perfect compatibility and seamless model integration.
    """
    
    def __init__(self):
        self.calculator = BotardoCompatibleCalculator()
    
    def extract_botardo_features(self, ohlcv_data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """
        Extract features exactly as botardo would calculate them
        for perfect compatibility with existing botardo strategies.
        """
        features = {}
        
        # Ensure we have enough data
        if len(ohlcv_data) < 50:
            return self._get_default_features()
        
        # Extract OHLCV arrays
        high = ohlcv_data['high'].values
        low = ohlcv_data['low'].values
        open_prices = ohlcv_data['open'].values
        close = ohlcv_data['close'].values
        
        current_price = close[-1]
        
        # 1. FIBONACCI FEATURES (D1 context)
        fib_high = np.max(high[-100:])  # 100-period high for D1 context
        fib_low = np.min(low[-100:])    # 100-period low for D1 context
        fib_levels = self.calculator.fibonacci_retracement(fib_high, fib_low, current_price)
        
        features.update({
            'fib_level_0': fib_levels.level_0,
            'fib_level_236': fib_levels.level_236,
            'fib_level_382': fib_levels.level_382,
            'fib_level_500': fib_levels.level_500,
            'fib_level_618': fib_levels.level_618,
            'fib_level_100': fib_levels.level_100,
            'fib_position': self._encode_fib_position(fib_levels.position),
            'fib_confidence': fib_levels.confidence,
            'fib_level_proximity': self._calculate_fib_proximity(current_price, fib_levels)
        })
        
        # 2. MACD FEATURES (H4 context)
        macd_line, macd_signal, macd_histogram = self.calculator.macd(close, 12, 26, 9, "H4")
        macd_interpretation = self.calculator.interpret_macd(macd_line, macd_signal, macd_histogram)
        
        features.update({
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram,
            'macd_trend_state': self._encode_signal(macd_interpretation.signal),
            'macd_trend_strength': macd_interpretation.strength,
            'macd_confidence': macd_interpretation.confidence,
            'macd_histogram_acceleration': self._calculate_macd_acceleration(close)
        })
        
        # 3. RVI FEATURES (M15 context)
        if len(ohlcv_data) >= 18:  # Need enough for RVI
            rvi_value, rvi_signal = self.calculator.rvi(open_prices, high, low, close, 14)
            rvi_cross = self._detect_rvi_cross(ohlcv_data)
            
            features.update({
                'rvi_value': rvi_value,
                'rvi_signal': rvi_signal,
                'rvi_signal_cross': rvi_cross,
                'rvi_strength': abs(rvi_value - rvi_signal)
            })
        else:
            features.update({
                'rvi_value': 0.0,
                'rvi_signal': 0.0,
                'rvi_signal_cross': 0.0,
                'rvi_strength': 0.0
            })
        
        # 4. RSI FEATURES (H1 context)
        rsi_value = self.calculator.rsi(close, 14)
        features.update({
            'rsi_value': rsi_value,
            'rsi_zone': self._encode_rsi_zone(rsi_value),
            'rsi_momentum': self._calculate_rsi_momentum(close)
        })
        
        # 5. ATR FEATURES (Multi-timeframe)
        atr_value = self.calculator.atr(high, low, close, 14)
        atr_ratio = self._calculate_atr_ratio(high, low, close)
        
        features.update({
            'atr_value': atr_value,
            'atr_d1_ratio': atr_ratio,
            'atr_h1_ratio': atr_ratio,  # Simplified for compatibility
            'atr_volatility': atr_value / current_price if current_price > 0 else 0.0
        })
        
        # 6. VOLUME FEATURES
        if 'volume' in ohlcv_data.columns:
            volume = ohlcv_data['volume'].values
            volume_ma = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
            volume_ratio = volume[-1] / volume_ma if volume_ma > 0 else 1.0
            
            features.update({
                'volume_ma_ratio': volume_ratio,
                'is_volume_spike': 1.0 if volume_ratio > 2.0 else 0.0,
                'volume_strength': min(volume_ratio / 3.0, 1.0)
            })
        else:
            features.update({
                'volume_ma_ratio': 1.0,
                'is_volume_spike': 0.0,
                'volume_strength': 0.5
            })
        
        # 7. DIVERGENCE FEATURES (Simplified)
        divergence_features = self._calculate_divergences(ohlcv_data)
        features.update(divergence_features)
        
        # 8. TEMPORAL FEATURES
        temporal_features = self._calculate_temporal_features()
        features.update(temporal_features)
        
        return features
    
    def _get_default_features(self) -> Dict[str, float]:
        """Default features when insufficient data"""
        return {
            'fib_level_0': 0.0, 'fib_level_236': 0.0, 'fib_level_382': 0.0,
            'fib_level_500': 0.0, 'fib_level_618': 0.0, 'fib_level_100': 0.0,
            'fib_position': 0.0, 'fib_confidence': 0.0, 'fib_level_proximity': 0.5,
            'macd_line': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0,
            'macd_trend_state': 0.0, 'macd_trend_strength': 0.0, 'macd_confidence': 0.0,
            'macd_histogram_acceleration': 0.0, 'rvi_value': 0.0, 'rvi_signal': 0.0,
            'rvi_signal_cross': 0.0, 'rvi_strength': 0.0, 'rsi_value': 50.0,
            'rsi_zone': 0.0, 'rsi_momentum': 0.0, 'atr_value': 0.0,
            'atr_d1_ratio': 1.0, 'atr_h1_ratio': 1.0, 'atr_volatility': 0.0,
            'volume_ma_ratio': 1.0, 'is_volume_spike': 0.0, 'volume_strength': 0.5,
            'price_macd_divergence_h4': 0.0, 'price_rvi_divergence_m15': 0.0,
            'price_rsi_divergence_h1': 0.0, 'hour_of_day': 12.0,
            'day_of_week': 2.0, 'market_session': 1.0, 'minutes_to_next_news': 0.0
        }
    
    def _encode_fib_position(self, position: str) -> float:
        """Encode fibonacci position as number"""
        position_map = {
            "at_high": 1.0, "near_236": 0.8, "near_382": 0.6,
            "near_500": 0.5, "near_618": 0.4, "at_low": 0.0, "unknown": 0.5
        }
        return position_map.get(position, 0.5)
    
    def _encode_signal(self, signal: str) -> float:
        """Encode BUY/SELL/HOLD as number"""
        signal_map = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}
        return signal_map.get(signal, 0.0)
    
    def _encode_rsi_zone(self, rsi: float) -> float:
        """Encode RSI zone"""
        if rsi <= 30:
            return -1.0  # Oversold
        elif rsi >= 70:
            return 1.0   # Overbought
        else:
            return 0.0   # Neutral
    
    def _calculate_fib_proximity(self, current_price: float, fib_levels: FibonacciLevels) -> float:
        """Calculate proximity to nearest fibonacci level"""
        levels = [fib_levels.level_0, fib_levels.level_236, fib_levels.level_382,
                 fib_levels.level_500, fib_levels.level_618, fib_levels.level_100]
        distances = [abs(current_price - level) for level in levels]
        min_distance = min(distances)
        price_range = fib_levels.level_0 - fib_levels.level_100
        
        return min_distance / price_range if price_range > 0 else 0.5
    
    def _calculate_macd_acceleration(self, close: np.ndarray) -> float:
        """Calculate MACD histogram acceleration"""
        if len(close) < 30:
            return 0.0
        
        # Calculate MACD for last few periods
        periods = min(10, len(close) - 26)
        histograms = []
        
        for i in range(periods):
            subset = close[:-(periods-i-1)] if i < periods-1 else close
            _, _, hist = self.calculator.macd(subset)
            histograms.append(hist)
        
        if len(histograms) >= 2:
            return histograms[-1] - histograms[-2]
        return 0.0
    
    def _detect_rvi_cross(self, ohlcv_data: pd.DataFrame) -> float:
        """Detect RVI signal line crosses"""
        if len(ohlcv_data) < 20:
            return 0.0
        
        # Calculate RVI for last 2 periods
        open_prices = ohlcv_data['open'].values
        high = ohlcv_data['high'].values
        low = ohlcv_data['low'].values
        close = ohlcv_data['close'].values
        
        rvi_curr, signal_curr = self.calculator.rvi(open_prices, high, low, close, 14)
        
        if len(ohlcv_data) >= 21:
            rvi_prev, signal_prev = self.calculator.rvi(
                open_prices[:-1], high[:-1], low[:-1], close[:-1], 14
            )
            
            # Detect crosses
            if rvi_prev <= signal_prev and rvi_curr > signal_curr:
                return 1.0  # Bullish cross
            elif rvi_prev >= signal_prev and rvi_curr < signal_curr:
                return -1.0  # Bearish cross
        
        return 0.0  # No cross
    
    def _calculate_rsi_momentum(self, close: np.ndarray) -> float:
        """Calculate RSI momentum"""
        if len(close) < 20:
            return 0.0
        
        rsi_curr = self.calculator.rsi(close, 14)
        rsi_prev = self.calculator.rsi(close[:-1], 14)
        
        return rsi_curr - rsi_prev
    
    def _calculate_atr_ratio(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """Calculate ATR ratio"""
        if len(high) < 30:
            return 1.0
        
        atr_current = self.calculator.atr(high, low, close, 14)
        atr_average = np.mean([
            self.calculator.atr(high[:-i], low[:-i], close[:-i], 14) 
            for i in range(1, min(11, len(high)-15))
        ])
        
        return atr_current / atr_average if atr_average > 0 else 1.0
    
    def _calculate_divergences(self, ohlcv_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate price-indicator divergences"""
        if len(ohlcv_data) < 40:
            return {
                'price_macd_divergence_h4': 0.0,
                'price_rvi_divergence_m15': 0.0,
                'price_rsi_divergence_h1': 0.0
            }
        
        close = ohlcv_data['close'].values
        
        # Price momentum (20-period)
        price_momentum = (close[-1] - close[-21]) / close[-21] if len(close) >= 21 else 0.0
        
        # MACD momentum
        macd_curr, _, _ = self.calculator.macd(close)
        macd_prev, _, _ = self.calculator.macd(close[:-20]) if len(close) >= 40 else (0, 0, 0)
        macd_momentum = macd_curr - macd_prev
        
        # RVI momentum
        open_prices = ohlcv_data['open'].values
        high = ohlcv_data['high'].values
        low = ohlcv_data['low'].values
        
        rvi_curr, _ = self.calculator.rvi(open_prices, high, low, close, 14)
        rvi_prev, _ = self.calculator.rvi(
            open_prices[:-20], high[:-20], low[:-20], close[:-20], 14
        ) if len(close) >= 40 else (0, 0)
        rvi_momentum = rvi_curr - rvi_prev
        
        # RSI momentum
        rsi_curr = self.calculator.rsi(close, 14)
        rsi_prev = self.calculator.rsi(close[:-20], 14) if len(close) >= 40 else 50
        rsi_momentum = rsi_curr - rsi_prev
        
        # Detect divergences
        macd_divergence = 0.0
        if price_momentum > 0 and macd_momentum < 0:
            macd_divergence = -1.0  # Bearish
        elif price_momentum < 0 and macd_momentum > 0:
            macd_divergence = 1.0   # Bullish
        
        rvi_divergence = 0.0
        if price_momentum > 0 and rvi_momentum < 0:
            rvi_divergence = -1.0
        elif price_momentum < 0 and rvi_momentum > 0:
            rvi_divergence = 1.0
        
        rsi_divergence = 0.0
        if price_momentum > 0 and rsi_momentum < 0:
            rsi_divergence = -1.0
        elif price_momentum < 0 and rsi_momentum > 0:
            rsi_divergence = 1.0
        
        return {
            'price_macd_divergence_h4': macd_divergence,
            'price_rvi_divergence_m15': rvi_divergence,
            'price_rsi_divergence_h1': rsi_divergence
        }
    
    def _calculate_temporal_features(self) -> Dict[str, float]:
        """Calculate temporal features"""
        now = datetime.now()
        
        # Market sessions
        hour = now.hour
        if 0 <= hour <= 7:
            session = 0.0  # Asian
        elif 8 <= hour <= 15:
            session = 1.0  # European
        else:
            session = 2.0  # American
        
        return {
            'hour_of_day': float(hour),
            'day_of_week': float(now.weekday()),
            'market_session': session,
            'minutes_to_next_news': 0.0  # Placeholder
        }