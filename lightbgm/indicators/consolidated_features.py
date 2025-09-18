"""
Consolidated Feature Generation for LightGBM ML Scoring System

This module integrates all components:
1. Botardo's exact calculations for perfect compatibility
2. MQ5 profitable strategy features (testing1.txt)
3. All base indicators: RSI, MACD, RVI, ADX, ATR, Volume, Fibonacci
4. Optimizable parameters for LightGBM training
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime

from .botardo_compatibility import (
    BotardoCompatibleCalculator, 
    BotardoFeatureMapper,
    TradingSignal, 
    FibonacciLevels
)
from .mq5_strategy_replicator import (
    MQ5StrategyReplicator,
    MQ5StrategyConfig,
    MQ5SetupSignal
)

logger = logging.getLogger(__name__)

@dataclass
class ConsolidatedFeatureConfig:
    """Master configuration for all indicators and strategies"""
    
    # FIBONACCI parameters (from MQ5 profitable)
    fib_lookback_period: int = 4  # MQ5 uses only 4 bars
    fib_levels: List[float] = None
    
    # MACD parameters (from MQ5 profitable: 50/26/20)
    macd_fast_period: int = 50    # Unusual: Fast > Slow
    macd_slow_period: int = 26
    macd_signal_period: int = 20
    
    # RSI parameters (from MQ5 profitable: period=2, levels=20/70)
    rsi_period: int = 2           # Very sensitive
    rsi_oversold: float = 20.0
    rsi_overbought: float = 70.0
    
    # RVI parameters (Base indicator)
    rvi_period: int = 14
    rvi_signal_period: int = 4
    
    # ADX parameters (Base indicator) 
    adx_period: int = 14
    adx_threshold: float = 25.0
    
    # ATR parameters (Base indicator - multi-timeframe)
    atr_period: int = 14
    atr_d1_multiplier: float = 3.0
    atr_h4_multiplier: float = 2.0
    atr_h1_multiplier: float = 1.5
    
    # Volume parameters (Base indicator)
    volume_ma_period: int = 20
    volume_spike_threshold: float = 2.0
    
    # Divergence parameters
    divergence_lookback: int = 20
    
    # Risk management (from MQ5)
    stop_loss_points: int = 15000
    take_profit_points: int = 20000
    
    def __post_init__(self):
        if self.fib_levels is None:
            self.fib_levels = [23.6, 38.2, 50.0, 61.8, 78.6]


class ConsolidatedFeatureEngineer:
    """
    Master feature engineer that consolidates all systems:
    - Botardo compatibility (exact calculations)
    - MQ5 profitable strategy replication
    - All base indicators with optimizable parameters
    """
    
    def __init__(self, config: ConsolidatedFeatureConfig):
        self.config = config
        self.calculator = BotardoCompatibleCalculator()
        
        logger.info("🚀 Consolidated Feature Engineer initialized")
        logger.info(f"📊 RSI: Period={config.rsi_period}, Levels={config.rsi_oversold}-{config.rsi_overbought}")
        logger.info(f"📈 MACD: {config.macd_fast_period}/{config.macd_slow_period}/{config.macd_signal_period}")
        logger.info(f"🌀 Fibonacci: {config.fib_lookback_period} bars")
        logger.info(f"⚡ RVI: Period={config.rvi_period}")
        logger.info(f"📊 ADX: Period={config.adx_period}")
        logger.info(f"📈 ATR: Multi-TF D1={config.atr_d1_multiplier}x")
    
    def generate_consolidated_features(self, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """
        Generate complete feature set integrating all systems.
        Returns features compatible with both botardo and lightbgm.
        """
        if len(data) < 100:
            return self._get_default_features()
        
        # Extract OHLCV
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values
        volume = data.get('volume', pd.Series([1000] * len(data))).values
        
        current_price = close_prices[-1]
        features = {}
        
        # ============= BASE INDICATORS (CORE) =============
        
        # 1. RSI (Base - from MQ5 profitable: period=2, very sensitive)
        rsi_features = self._calculate_rsi_features(close_prices)
        features.update(rsi_features)
        
        # 2. MACD (Base - from MQ5 profitable: 50/26/20)
        macd_features = self._calculate_macd_features(close_prices)
        features.update(macd_features)
        
        # 3. RVI (Base - exact botardo implementation)
        rvi_features = self._calculate_rvi_features(open_prices, high_prices, low_prices, close_prices)
        features.update(rvi_features)
        
        # 4. ADX (Base - trend strength from botardo)
        adx_features = self._calculate_adx_features(high_prices, low_prices, close_prices)
        features.update(adx_features)
        
        # 5. ATR (Base - multi-timeframe volatility)
        atr_features = self._calculate_atr_features(high_prices, low_prices, close_prices)
        features.update(atr_features)
        
        # 6. Volume (Base - strength and regime)
        volume_features = self._calculate_volume_features(volume, close_prices)
        features.update(volume_features)
        
        # 7. Fibonacci (from MQ5 profitable: 4 bars only)
        fib_features = self._calculate_fibonacci_features(high_prices, low_prices, current_price)
        features.update(fib_features)
        
        # ============= ADVANCED FEATURES =============
        
        # 8. Divergences (multi-indicator)
        divergence_features = self._calculate_divergence_features(data)
        features.update(divergence_features)
        
        # 9. MQ5 Strategy Features (profitable setup signals)
        mq5_features = self._calculate_mq5_strategy_features(data, symbol)
        features.update(mq5_features)
        
        # 10. Market Context (temporal and regime)
        context_features = self._calculate_context_features(data)
        features.update(context_features)
        
        # 11. Risk Management Features
        risk_features = self._calculate_risk_features(current_price, symbol)
        features.update(risk_features)
        
        return features
    
    def _calculate_rsi_features(self, close_prices: np.ndarray) -> Dict[str, float]:
        """RSI features with MQ5 profitable settings (period=2)"""
        rsi_value = self.calculator.rsi(close_prices, self.config.rsi_period)
        
        return {
            'rsi_value': rsi_value,
            'rsi_oversold': 1.0 if rsi_value <= self.config.rsi_oversold else 0.0,
            'rsi_overbought': 1.0 if rsi_value >= self.config.rsi_overbought else 0.0,
            'rsi_zone': self._encode_rsi_zone(rsi_value),
            'rsi_momentum': self._calculate_rsi_momentum(close_prices),
            'rsi_strength': abs(rsi_value - 50) / 50,  # Distance from neutral
            'rsi_extremes': 1.0 if rsi_value <= 10 or rsi_value >= 90 else 0.0
        }
    
    def _calculate_macd_features(self, close_prices: np.ndarray) -> Dict[str, float]:
        """MACD features with MQ5 profitable settings (50/26/20)"""
        macd_line, signal_line, histogram = self.calculator.macd(
            close_prices, 
            self.config.macd_fast_period,
            self.config.macd_slow_period, 
            self.config.macd_signal_period
        )
        
        # Signal interpretation
        signal_type = "HOLD"
        if macd_line > signal_line and histogram > 0:
            signal_type = "BUY"
        elif macd_line < signal_line and histogram < 0:
            signal_type = "SELL"
        
        return {
            'macd_line': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram,
            'macd_trend_state': self._encode_signal(signal_type),
            'macd_trend_strength': min(1.0, abs(histogram) / (abs(macd_line) + 1e-8)),
            'macd_histogram_acceleration': self._calculate_macd_acceleration(close_prices),
            'macd_bullish': 1.0 if signal_type == "BUY" else 0.0,
            'macd_bearish': 1.0 if signal_type == "SELL" else 0.0,
            'macd_cross_up': self._detect_macd_cross(close_prices, direction="up"),
            'macd_cross_down': self._detect_macd_cross(close_prices, direction="down")
        }
    
    def _calculate_rvi_features(self, open_prices: np.ndarray, high_prices: np.ndarray,
                               low_prices: np.ndarray, close_prices: np.ndarray) -> Dict[str, float]:
        """RVI features - exact botardo implementation"""
        rvi_value, rvi_signal = self.calculator.rvi(
            open_prices, high_prices, low_prices, close_prices, self.config.rvi_period
        )
        
        # Signal crosses
        cross_signal = self._detect_rvi_cross(open_prices, high_prices, low_prices, close_prices)
        
        return {
            'rvi_value': rvi_value,
            'rvi_signal': rvi_signal,
            'rvi_signal_cross': cross_signal,
            'rvi_strength': abs(rvi_value - rvi_signal),
            'rvi_bullish': 1.0 if rvi_value > rvi_signal else 0.0,
            'rvi_bearish': 1.0 if rvi_value < rvi_signal else 0.0,
            'rvi_momentum': self._calculate_rvi_momentum(open_prices, high_prices, low_prices, close_prices)
        }
    
    def _calculate_adx_features(self, high_prices: np.ndarray, low_prices: np.ndarray,
                               close_prices: np.ndarray) -> Dict[str, float]:
        """ADX features for trend strength"""
        adx_value = self._calculate_adx(high_prices, low_prices, close_prices)
        
        return {
            'adx_value': adx_value,
            'adx_strong_trend': 1.0 if adx_value > self.config.adx_threshold else 0.0,
            'adx_trend_strength': min(1.0, adx_value / 50),  # Normalize to 0-1
            'adx_trending': 1.0 if adx_value > self.config.adx_threshold else 0.0,
            'adx_ranging': 1.0 if adx_value < self.config.adx_threshold else 0.0,
            'adx_momentum': self._calculate_adx_momentum(high_prices, low_prices, close_prices)
        }
    
    def _calculate_atr_features(self, high_prices: np.ndarray, low_prices: np.ndarray,
                               close_prices: np.ndarray) -> Dict[str, float]:
        """ATR features - multi-timeframe volatility"""
        atr_value = self.calculator.atr(high_prices, low_prices, close_prices, self.config.atr_period)
        current_price = close_prices[-1]
        
        # ATR ratios for different timeframes
        atr_ratio = self._calculate_atr_ratio(high_prices, low_prices, close_prices)
        
        return {
            'atr_value': atr_value,
            'atr_d1_ratio': atr_ratio,  # Structural volatility
            'atr_h4_ratio': atr_ratio * 0.7,  # Tactical volatility
            'atr_h1_ratio': atr_ratio * 0.5,  # Intraday volatility
            'atr_volatility': atr_value / current_price if current_price > 0 else 0.0,
            'atr_normalized': min(1.0, atr_value / (current_price * 0.02)),  # 2% normalization
            'atr_regime': self._classify_atr_regime(atr_ratio)
        }
    
    def _calculate_volume_features(self, volume: np.ndarray, close_prices: np.ndarray) -> Dict[str, float]:
        """Volume features - strength and regime analysis"""
        if len(volume) < self.config.volume_ma_period:
            return {
                'volume_ma_ratio': 1.0,
                'is_volume_spike': 0.0,
                'volume_strength': 0.5,
                'volume_regime': 2.0,  # Normal
                'volume_trend': 0.0
            }
        
        volume_ma = np.mean(volume[-self.config.volume_ma_period:])
        volume_ratio = volume[-1] / volume_ma if volume_ma > 0 else 1.0
        
        # Volume regime classification
        if volume_ratio < 0.5:
            regime = 0.0  # Dead
        elif volume_ratio < 0.7:
            regime = 1.0  # Quiet
        elif volume_ratio < 1.5:
            regime = 2.0  # Normal
        elif volume_ratio < 2.5:
            regime = 3.0  # Active
        else:
            regime = 4.0  # Breakout
        
        return {
            'volume_ma_ratio': volume_ratio,
            'is_volume_spike': 1.0 if volume_ratio > self.config.volume_spike_threshold else 0.0,
            'volume_strength': min(volume_ratio / 3.0, 1.0),
            'volume_regime': regime,
            'volume_trend': self._calculate_volume_trend(volume)
        }
    
    def _calculate_fibonacci_features(self, high_prices: np.ndarray, low_prices: np.ndarray,
                                    current_price: float) -> Dict[str, float]:
        """Fibonacci features with MQ5 settings (4 bars only)"""
        # Use only last N bars (MQ5 setting)
        lookback = min(self.config.fib_lookback_period, len(high_prices))
        
        if lookback < 2:
            return self._get_default_fib_features()
        
        recent_high = np.max(high_prices[-lookback:])
        recent_low = np.min(low_prices[-lookback:])
        
        fib_levels = self.calculator.fibonacci_retracement(recent_high, recent_low, current_price)
        
        features = {
            'fib_level_0': fib_levels.level_0,
            'fib_level_236': fib_levels.level_236,
            'fib_level_382': fib_levels.level_382,
            'fib_level_500': fib_levels.level_500,
            'fib_level_618': fib_levels.level_618,
            'fib_level_100': fib_levels.level_100,
            'fib_position': self._encode_fib_position(fib_levels.position),
            'fib_confidence': fib_levels.confidence,
            'fib_level_proximity': self._calculate_fib_proximity(current_price, fib_levels),
            'fib_range_ratio': (recent_high - recent_low) / recent_high if recent_high > 0 else 0.0
        }
        
        return features
    
    def _calculate_divergence_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate price-indicator divergences"""
        if len(data) < 40:
            return {
                'price_macd_divergence_h4': 0.0,
                'price_rvi_divergence_m15': 0.0,
                'price_rsi_divergence_h1': 0.0,
                'divergence_strength': 0.0
            }
        
        close = data['close'].values
        open_prices = data['open'].values
        high = data['high'].values
        low = data['low'].values
        
        # Price momentum
        price_momentum = (close[-1] - close[-21]) / close[-21] if len(close) >= 21 else 0.0
        
        # Indicator momentums
        macd_momentum = self._calculate_indicator_momentum(close, 'macd')
        rvi_momentum = self._calculate_indicator_momentum_rvi(open_prices, high, low, close)
        rsi_momentum = self._calculate_indicator_momentum(close, 'rsi')
        
        # Detect divergences
        macd_div = self._detect_divergence(price_momentum, macd_momentum)
        rvi_div = self._detect_divergence(price_momentum, rvi_momentum)
        rsi_div = self._detect_divergence(price_momentum, rsi_momentum)
        
        return {
            'price_macd_divergence_h4': macd_div,
            'price_rvi_divergence_m15': rvi_div,
            'price_rsi_divergence_h1': rsi_div,
            'divergence_strength': abs(macd_div) + abs(rvi_div) + abs(rsi_div)
        }
    
    def _calculate_mq5_strategy_features(self, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Generate MQ5 profitable strategy features"""
        try:
            # Create MQ5 replicator with current config
            mq5_config = MQ5StrategyConfig(
                rsi_period=self.config.rsi_period,
                rsi_up_level=self.config.rsi_overbought,
                rsi_down_level=self.config.rsi_oversold,
                fibo_num_bars=self.config.fib_lookback_period,
                macd_fast_period=self.config.macd_fast_period,
                macd_slow_period=self.config.macd_slow_period,
                macd_signal_period=self.config.macd_signal_period
            )
            
            mq5_replicator = MQ5StrategyReplicator(mq5_config)
            setup_signal = mq5_replicator.analyze_setup(data, symbol)
            
            return {
                'mq5_signal_type': self._encode_signal(setup_signal.signal_type),
                'mq5_strength': setup_signal.strength,
                'mq5_confidence': setup_signal.confidence,
                'mq5_is_valid_setup': 1.0 if setup_signal.is_valid_setup else 0.0,
                'mq5_risk_reward': setup_signal.risk_reward,
                'mq5_expected_sl': setup_signal.expected_sl,
                'mq5_expected_tp': setup_signal.expected_tp
            }
            
        except Exception as e:
            logger.warning(f"Error calculating MQ5 features: {e}")
            return {
                'mq5_signal_type': 0.0,
                'mq5_strength': 0.0,
                'mq5_confidence': 0.0,
                'mq5_is_valid_setup': 0.0,
                'mq5_risk_reward': 0.0,
                'mq5_expected_sl': 0.0,
                'mq5_expected_tp': 0.0
            }
    
    def _calculate_context_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Market context and temporal features"""
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
            'minutes_to_next_news': 0.0,  # Placeholder
            'bars_count': float(len(data)),
            'data_quality': 1.0 if len(data) >= 100 else len(data) / 100
        }
    
    def _calculate_risk_features(self, current_price: float, symbol: str) -> Dict[str, float]:
        """Risk management features from MQ5 strategy"""
        point_value = self.calculator.get_point_value(symbol)
        
        # Calculate SL/TP distances in price terms
        sl_distance = self.config.stop_loss_points * point_value
        tp_distance = self.config.take_profit_points * point_value
        
        return {
            'risk_stop_loss_distance': sl_distance,
            'risk_take_profit_distance': tp_distance,
            'risk_reward_ratio': tp_distance / sl_distance if sl_distance > 0 else 0.0,
            'risk_points_sl': float(self.config.stop_loss_points),
            'risk_points_tp': float(self.config.take_profit_points),
            'risk_point_value': point_value
        }
    
    # ============= HELPER METHODS =============
    
    def _encode_signal(self, signal: str) -> float:
        """Encode signal type as number"""
        signal_map = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}
        return signal_map.get(signal, 0.0)
    
    def _encode_rsi_zone(self, rsi: float) -> float:
        """Encode RSI zone"""
        if rsi <= self.config.rsi_oversold:
            return -1.0  # Oversold
        elif rsi >= self.config.rsi_overbought:
            return 1.0   # Overbought
        else:
            return 0.0   # Neutral
    
    def _encode_fib_position(self, position: str) -> float:
        """Encode fibonacci position"""
        position_map = {
            "at_high": 1.0, "near_236": 0.8, "near_382": 0.6,
            "near_500": 0.5, "near_618": 0.4, "at_low": 0.0, "unknown": 0.5
        }
        return position_map.get(position, 0.5)
    
    def _calculate_fib_proximity(self, current_price: float, fib_levels: FibonacciLevels) -> float:
        """Calculate proximity to nearest fibonacci level"""
        levels = [fib_levels.level_0, fib_levels.level_236, fib_levels.level_382,
                 fib_levels.level_500, fib_levels.level_618, fib_levels.level_100]
        distances = [abs(current_price - level) for level in levels]
        min_distance = min(distances)
        price_range = fib_levels.level_0 - fib_levels.level_100
        
        return min_distance / price_range if price_range > 0 else 0.5
    
    def _calculate_rsi_momentum(self, close_prices: np.ndarray) -> float:
        """Calculate RSI momentum"""
        if len(close_prices) < self.config.rsi_period + 5:
            return 0.0
        
        rsi_curr = self.calculator.rsi(close_prices, self.config.rsi_period)
        rsi_prev = self.calculator.rsi(close_prices[:-1], self.config.rsi_period)
        
        return rsi_curr - rsi_prev
    
    def _calculate_macd_acceleration(self, close_prices: np.ndarray) -> float:
        """Calculate MACD histogram acceleration"""
        if len(close_prices) < 30:
            return 0.0
        
        _, _, hist_curr = self.calculator.macd(
            close_prices, self.config.macd_fast_period, 
            self.config.macd_slow_period, self.config.macd_signal_period
        )
        
        _, _, hist_prev = self.calculator.macd(
            close_prices[:-1], self.config.macd_fast_period,
            self.config.macd_slow_period, self.config.macd_signal_period
        )
        
        return hist_curr - hist_prev
    
    def _detect_macd_cross(self, close_prices: np.ndarray, direction: str) -> float:
        """Detect MACD line crosses"""
        if len(close_prices) < 30:
            return 0.0
        
        macd_curr, signal_curr, _ = self.calculator.macd(
            close_prices, self.config.macd_fast_period,
            self.config.macd_slow_period, self.config.macd_signal_period
        )
        
        macd_prev, signal_prev, _ = self.calculator.macd(
            close_prices[:-1], self.config.macd_fast_period,
            self.config.macd_slow_period, self.config.macd_signal_period
        )
        
        if direction == "up":
            return 1.0 if macd_prev <= signal_prev and macd_curr > signal_curr else 0.0
        else:  # down
            return 1.0 if macd_prev >= signal_prev and macd_curr < signal_curr else 0.0
    
    def _detect_rvi_cross(self, open_prices: np.ndarray, high_prices: np.ndarray,
                         low_prices: np.ndarray, close_prices: np.ndarray) -> float:
        """Detect RVI signal crosses"""
        if len(close_prices) < 20:
            return 0.0
        
        rvi_curr, signal_curr = self.calculator.rvi(
            open_prices, high_prices, low_prices, close_prices, self.config.rvi_period
        )
        
        rvi_prev, signal_prev = self.calculator.rvi(
            open_prices[:-1], high_prices[:-1], low_prices[:-1], close_prices[:-1], 
            self.config.rvi_period
        )
        
        if rvi_prev <= signal_prev and rvi_curr > signal_curr:
            return 1.0  # Bullish cross
        elif rvi_prev >= signal_prev and rvi_curr < signal_curr:
            return -1.0  # Bearish cross
        else:
            return 0.0  # No cross
    
    def _calculate_rvi_momentum(self, open_prices: np.ndarray, high_prices: np.ndarray,
                               low_prices: np.ndarray, close_prices: np.ndarray) -> float:
        """Calculate RVI momentum"""
        if len(close_prices) < 20:
            return 0.0
        
        rvi_curr, _ = self.calculator.rvi(
            open_prices, high_prices, low_prices, close_prices, self.config.rvi_period
        )
        
        rvi_prev, _ = self.calculator.rvi(
            open_prices[:-5], high_prices[:-5], low_prices[:-5], close_prices[:-5], 
            self.config.rvi_period
        )
        
        return rvi_curr - rvi_prev
    
    def _calculate_adx(self, high_prices: np.ndarray, low_prices: np.ndarray,
                      close_prices: np.ndarray) -> float:
        """Calculate ADX (Average Directional Index)"""
        if len(high_prices) < self.config.adx_period + 1:
            return 25.0  # Default neutral value
        
        # Simplified ADX calculation
        period = self.config.adx_period
        
        # True Range
        tr_list = []
        for i in range(1, len(high_prices)):
            tr = max(
                high_prices[i] - low_prices[i],
                abs(high_prices[i] - close_prices[i-1]),
                abs(low_prices[i] - close_prices[i-1])
            )
            tr_list.append(tr)
        
        # Directional Movement
        dm_plus = []
        dm_minus = []
        
        for i in range(1, len(high_prices)):
            up_move = high_prices[i] - high_prices[i-1]
            down_move = low_prices[i-1] - low_prices[i]
            
            if up_move > down_move and up_move > 0:
                dm_plus.append(up_move)
            else:
                dm_plus.append(0)
                
            if down_move > up_move and down_move > 0:
                dm_minus.append(down_move)
            else:
                dm_minus.append(0)
        
        # Average TR and DM
        if len(tr_list) >= period and len(dm_plus) >= period:
            atr = np.mean(tr_list[-period:])
            adm_plus = np.mean(dm_plus[-period:])
            adm_minus = np.mean(dm_minus[-period:])
            
            di_plus = (adm_plus / atr) * 100 if atr > 0 else 0
            di_minus = (adm_minus / atr) * 100 if atr > 0 else 0
            
            dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100 if (di_plus + di_minus) > 0 else 0
            return dx
        
        return 25.0  # Default
    
    def _calculate_adx_momentum(self, high_prices: np.ndarray, low_prices: np.ndarray,
                               close_prices: np.ndarray) -> float:
        """Calculate ADX momentum"""
        if len(high_prices) < self.config.adx_period + 10:
            return 0.0
        
        adx_curr = self._calculate_adx(high_prices, low_prices, close_prices)
        adx_prev = self._calculate_adx(high_prices[:-5], low_prices[:-5], close_prices[:-5])
        
        return adx_curr - adx_prev
    
    def _calculate_atr_ratio(self, high_prices: np.ndarray, low_prices: np.ndarray,
                            close_prices: np.ndarray) -> float:
        """Calculate ATR ratio (current vs average)"""
        if len(high_prices) < 30:
            return 1.0
        
        atr_current = self.calculator.atr(high_prices, low_prices, close_prices, self.config.atr_period)
        
        # Calculate historical average
        atr_values = []
        for i in range(5, min(15, len(high_prices) - self.config.atr_period)):
            atr_hist = self.calculator.atr(
                high_prices[:-i], low_prices[:-i], close_prices[:-i], self.config.atr_period
            )
            atr_values.append(atr_hist)
        
        atr_average = np.mean(atr_values) if atr_values else atr_current
        
        return atr_current / atr_average if atr_average > 0 else 1.0
    
    def _classify_atr_regime(self, atr_ratio: float) -> float:
        """Classify ATR volatility regime"""
        if atr_ratio < 0.5:
            return 0.0  # Very Low
        elif atr_ratio < 0.8:
            return 1.0  # Low
        elif atr_ratio < 1.2:
            return 2.0  # Normal
        elif atr_ratio < 1.8:
            return 3.0  # High
        else:
            return 4.0  # Very High
    
    def _calculate_volume_trend(self, volume: np.ndarray) -> float:
        """Calculate volume trend"""
        if len(volume) < 10:
            return 0.0
        
        # Linear regression slope of volume
        x = np.arange(len(volume[-10:]))
        y = volume[-10:]
        
        if len(x) > 1:
            slope = np.corrcoef(x, y)[0, 1] if np.std(y) > 0 else 0.0
            return slope
        
        return 0.0
    
    def _calculate_indicator_momentum(self, close_prices: np.ndarray, indicator: str) -> float:
        """Calculate momentum for various indicators"""
        if len(close_prices) < 40:
            return 0.0
        
        if indicator == 'macd':
            macd_curr, _, _ = self.calculator.macd(
                close_prices, self.config.macd_fast_period,
                self.config.macd_slow_period, self.config.macd_signal_period
            )
            macd_prev, _, _ = self.calculator.macd(
                close_prices[:-20], self.config.macd_fast_period,
                self.config.macd_slow_period, self.config.macd_signal_period
            )
            return macd_curr - macd_prev
            
        elif indicator == 'rsi':
            rsi_curr = self.calculator.rsi(close_prices, self.config.rsi_period)
            rsi_prev = self.calculator.rsi(close_prices[:-20], self.config.rsi_period)
            return rsi_curr - rsi_prev
        
        return 0.0
    
    def _calculate_indicator_momentum_rvi(self, open_prices: np.ndarray, high_prices: np.ndarray,
                                         low_prices: np.ndarray, close_prices: np.ndarray) -> float:
        """Calculate RVI momentum"""
        if len(close_prices) < 40:
            return 0.0
        
        rvi_curr, _ = self.calculator.rvi(
            open_prices, high_prices, low_prices, close_prices, self.config.rvi_period
        )
        
        rvi_prev, _ = self.calculator.rvi(
            open_prices[:-20], high_prices[:-20], low_prices[:-20], close_prices[:-20], 
            self.config.rvi_period
        )
        
        return rvi_curr - rvi_prev
    
    def _detect_divergence(self, price_momentum: float, indicator_momentum: float) -> float:
        """Detect divergence between price and indicator"""
        threshold = 0.001  # Minimum momentum threshold
        
        if abs(price_momentum) < threshold or abs(indicator_momentum) < threshold:
            return 0.0
        
        if price_momentum > 0 and indicator_momentum < 0:
            return -1.0  # Bearish divergence
        elif price_momentum < 0 and indicator_momentum > 0:
            return 1.0   # Bullish divergence
        else:
            return 0.0   # No divergence
    
    def _get_default_features(self) -> Dict[str, float]:
        """Default features when insufficient data"""
        return {
            # RSI features
            'rsi_value': 50.0, 'rsi_oversold': 0.0, 'rsi_overbought': 0.0,
            'rsi_zone': 0.0, 'rsi_momentum': 0.0, 'rsi_strength': 0.0, 'rsi_extremes': 0.0,
            
            # MACD features
            'macd_line': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0,
            'macd_trend_state': 0.0, 'macd_trend_strength': 0.0, 'macd_histogram_acceleration': 0.0,
            'macd_bullish': 0.0, 'macd_bearish': 0.0, 'macd_cross_up': 0.0, 'macd_cross_down': 0.0,
            
            # RVI features
            'rvi_value': 0.0, 'rvi_signal': 0.0, 'rvi_signal_cross': 0.0,
            'rvi_strength': 0.0, 'rvi_bullish': 0.0, 'rvi_bearish': 0.0, 'rvi_momentum': 0.0,
            
            # ADX features
            'adx_value': 25.0, 'adx_strong_trend': 0.0, 'adx_trend_strength': 0.5,
            'adx_trending': 0.0, 'adx_ranging': 1.0, 'adx_momentum': 0.0,
            
            # ATR features
            'atr_value': 0.0, 'atr_d1_ratio': 1.0, 'atr_h4_ratio': 1.0, 'atr_h1_ratio': 1.0,
            'atr_volatility': 0.0, 'atr_normalized': 0.5, 'atr_regime': 2.0,
            
            # Volume features
            'volume_ma_ratio': 1.0, 'is_volume_spike': 0.0, 'volume_strength': 0.5,
            'volume_regime': 2.0, 'volume_trend': 0.0,
            
            # Fibonacci features
            'fib_level_0': 0.0, 'fib_level_236': 0.0, 'fib_level_382': 0.0,
            'fib_level_500': 0.0, 'fib_level_618': 0.0, 'fib_level_100': 0.0,
            'fib_position': 0.5, 'fib_confidence': 0.0, 'fib_level_proximity': 0.5,
            'fib_range_ratio': 0.0,
            
            # Divergence features
            'price_macd_divergence_h4': 0.0, 'price_rvi_divergence_m15': 0.0,
            'price_rsi_divergence_h1': 0.0, 'divergence_strength': 0.0,
            
            # MQ5 strategy features
            'mq5_signal_type': 0.0, 'mq5_strength': 0.0, 'mq5_confidence': 0.0,
            'mq5_is_valid_setup': 0.0, 'mq5_risk_reward': 0.0, 'mq5_expected_sl': 0.0,
            'mq5_expected_tp': 0.0,
            
            # Context features
            'hour_of_day': 12.0, 'day_of_week': 2.0, 'market_session': 1.0,
            'minutes_to_next_news': 0.0, 'bars_count': 0.0, 'data_quality': 0.0,
            
            # Risk features
            'risk_stop_loss_distance': 0.0, 'risk_take_profit_distance': 0.0,
            'risk_reward_ratio': 1.33, 'risk_points_sl': 15000.0, 'risk_points_tp': 20000.0,
            'risk_point_value': 1.0
        }
    
    def _get_default_fib_features(self) -> Dict[str, float]:
        """Default Fibonacci features"""
        return {
            'fib_level_0': 0.0, 'fib_level_236': 0.0, 'fib_level_382': 0.0,
            'fib_level_500': 0.0, 'fib_level_618': 0.0, 'fib_level_100': 0.0,
            'fib_position': 0.5, 'fib_confidence': 0.0, 'fib_level_proximity': 0.5,
            'fib_range_ratio': 0.0
        }