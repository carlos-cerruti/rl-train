"""
MQ5 Strategy Replicator for LightGBM ML Training

This module replicates the exact profitable MQ5 strategy from testing1.txt
and optimizes the thresholds for maximum performance using historical data.

Strategy Configuration (Profitable Setup):
- RSI: Use=true, Period=2, Upper=70.0, Lower=20.0, TimeFrame=30min
- Fibonacci: Use=true, TimeFrame=16387(?), NumBars=4
- MACD: Use=true, Fast=50, Slow=26, Signal=20, TimeFrame=16388(?)
- SL=15000, TP=20000 (points)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime

from .botardo_compatibility import BotardoCompatibleCalculator, TradingSignal, FibonacciLevels

logger = logging.getLogger(__name__)

@dataclass
class MQ5StrategyConfig:
    """Configuration matching the profitable MQ5 strategy"""
    
    # RSI Settings (ENABLED)
    rsi_use: bool = True
    rsi_period: int = 2  # Very short period for quick signals
    rsi_up_level: float = 70.0
    rsi_down_level: float = 20.0
    rsi_timeframe: int = 30  # 30-minute bars
    
    # Fibonacci Settings (ENABLED)
    fibo_use: bool = True
    fibo_timeframe: int = 16387  # Need to decode this MQ5 timeframe
    fibo_num_bars: int = 4  # Only 4 bars for Fibo calculation
    fibo_trading_range: int = 0  # No range restriction
    
    # MACD Settings (ENABLED)
    macd_use: bool = True
    macd_fast_period: int = 50  # Unusual: Fast > Slow!
    macd_slow_period: int = 26
    macd_signal_period: int = 20
    macd_timeframe: int = 16388
    
    # Risk Management
    stop_loss: int = 15000  # Points
    take_profit: int = 20000  # Points
    risk_reward_ratio: float = 1.33  # 20000/15000
    
    # Position Management
    max_buy_positions: int = 2
    max_sell_positions: int = 2
    fixed_lot: float = 0.1
    
    # Trading Hours (24/7)
    trade_24_7: bool = True

@dataclass
class MQ5SetupSignal:
    """Complete setup signal matching MQ5 strategy"""
    timestamp: datetime
    signal_type: str  # BUY, SELL, NONE
    strength: float  # 0.0 to 1.0
    confidence: float
    
    # Individual component signals
    rsi_signal: TradingSignal
    fibo_signal: Dict[str, Any]
    macd_signal: TradingSignal
    
    # Setup validation
    is_valid_setup: bool
    expected_sl: float
    expected_tp: float
    risk_reward: float
    
    # Feature vector for ML
    feature_vector: Dict[str, float]

class MQ5StrategyReplicator:
    """
    Replicates the exact profitable MQ5 strategy for ML training data generation.
    
    This class generates the same signals that the profitable MQ5 EA would generate,
    allowing us to train LightGBM on historically successful setups.
    """
    
    def __init__(self, config: Optional[MQ5StrategyConfig] = None):
        self.config = config or MQ5StrategyConfig()
        self.calculator = BotardoCompatibleCalculator()
        
        # Decode MQ5 timeframes (best guess based on common values)
        self.timeframe_map = {
            16387: 'H1',    # 1 hour
            16388: 'H4',    # 4 hours  
            16386: 'M30',   # 30 minutes
            30: 'M30'       # 30 minutes
        }
        
        logger.info("MQ5 Strategy Replicator initialized with profitable config")
        logger.info(f"RSI: Period={self.config.rsi_period}, Levels={self.config.rsi_down_level}-{self.config.rsi_up_level}")
        logger.info(f"MACD: {self.config.macd_fast_period}/{self.config.macd_slow_period}/{self.config.macd_signal_period}")
        logger.info(f"Fibonacci: NumBars={self.config.fibo_num_bars}")
        logger.info(f"Risk Management: SL={self.config.stop_loss}, TP={self.config.take_profit}")
    
    def analyze_setup(self, ohlcv_data: pd.DataFrame, symbol: str) -> MQ5SetupSignal:
        """
        Analyze market data and generate setup signal exactly as MQ5 EA would.
        
        Returns complete signal with all components that can be used for:
        1. Live trading decisions
        2. ML training data generation
        3. Backtesting validation
        """
        try:
            if len(ohlcv_data) < 100:  # Need sufficient data
                return self._create_no_signal()
            
            current_price = ohlcv_data['close'].iloc[-1]
            timestamp = ohlcv_data.index[-1] if hasattr(ohlcv_data.index[-1], 'to_pydatetime') else datetime.now()
            
            # 1. RSI Analysis (Period=2, very sensitive)
            rsi_signal = self._analyze_rsi(ohlcv_data)
            
            # 2. Fibonacci Analysis (NumBars=4, tight levels)
            fibo_signal = self._analyze_fibonacci(ohlcv_data)
            
            # 3. MACD Analysis (Unusual 50/26/20 config)
            macd_signal = self._analyze_macd(ohlcv_data)
            
            # 4. Combine signals (AND logic - all must agree)
            combined_signal, strength, confidence = self._combine_signals(
                rsi_signal, fibo_signal, macd_signal
            )
            
            # 5. Calculate SL/TP levels
            sl_price, tp_price, risk_reward = self._calculate_sl_tp(
                current_price, combined_signal, symbol
            )
            
            # 6. Validate setup (risk management)
            is_valid = self._validate_setup(
                combined_signal, strength, confidence, risk_reward
            )
            
            # 7. Generate feature vector for ML
            feature_vector = self._generate_feature_vector(
                ohlcv_data, rsi_signal, fibo_signal, macd_signal, symbol
            )
            
            return MQ5SetupSignal(
                timestamp=timestamp,
                signal_type=combined_signal,
                strength=strength,
                confidence=confidence,
                rsi_signal=rsi_signal,
                fibo_signal=fibo_signal,
                macd_signal=macd_signal,
                is_valid_setup=is_valid,
                expected_sl=sl_price,
                expected_tp=tp_price,
                risk_reward=risk_reward,
                feature_vector=feature_vector
            )
            
        except Exception as e:
            logger.error(f"Error analyzing MQ5 setup: {e}")
            return self._create_no_signal()
    
    def _analyze_rsi(self, data: pd.DataFrame) -> TradingSignal:
        """RSI Analysis with Period=2 (very short, very sensitive)"""
        close_prices = data['close'].values
        
        # Calculate RSI with period=2 (extremely sensitive)
        rsi_value = self.calculator.rsi(close_prices, self.config.rsi_period)
        
        # Apply MQ5 thresholds
        if rsi_value <= self.config.rsi_down_level:
            signal_type = "BUY"  # Oversold, expect bounce
            strength = (self.config.rsi_down_level - rsi_value) / self.config.rsi_down_level
        elif rsi_value >= self.config.rsi_up_level:
            signal_type = "SELL"  # Overbought, expect pullback
            strength = (rsi_value - self.config.rsi_up_level) / (100 - self.config.rsi_up_level)
        else:
            signal_type = "HOLD"
            strength = 0.0
        
        # Confidence based on distance from threshold
        if signal_type != "HOLD":
            distance_from_threshold = abs(rsi_value - 50) / 50  # Distance from neutral
            confidence = min(1.0, distance_from_threshold)
        else:
            confidence = 0.0
        
        return TradingSignal(
            indicator="RSI",
            value=rsi_value,
            signal=signal_type,
            strength=min(1.0, strength),
            confidence=confidence,
            interpretation=f"RSI({self.config.rsi_period})={rsi_value:.2f}, Levels:{self.config.rsi_down_level}-{self.config.rsi_up_level}",
            timestamp=datetime.now()
        )
    
    def _analyze_fibonacci(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fibonacci Analysis with NumBars=4 (very tight, recent levels)"""
        high_prices = data['high'].values
        low_prices = data['low'].values
        current_price = data['close'].iloc[-1]
        
        # Use only last 4 bars for Fibonacci (MQ5 setting)
        if len(high_prices) >= self.config.fibo_num_bars:
            recent_high = np.max(high_prices[-self.config.fibo_num_bars:])
            recent_low = np.min(low_prices[-self.config.fibo_num_bars:])
        else:
            recent_high = np.max(high_prices)
            recent_low = np.min(low_prices)
        
        # Calculate Fibonacci levels
        fib_levels = self.calculator.fibonacci_retracement(recent_high, recent_low, current_price)
        
        # Determine signal based on Fibonacci position
        signal_type = "HOLD"
        strength = 0.0
        
        # Check proximity to key Fibonacci levels
        key_levels = {
            'level_236': fib_levels.level_236,
            'level_382': fib_levels.level_382,
            'level_500': fib_levels.level_500,
            'level_618': fib_levels.level_618
        }
        
        # Find closest level and determine signal
        min_distance = float('inf')
        closest_level = None
        
        for level_name, level_price in key_levels.items():
            distance = abs(current_price - level_price) / (recent_high - recent_low)
            if distance < min_distance:
                min_distance = distance
                closest_level = level_name
        
        # Signal based on Fibonacci position and trend
        proximity_threshold = 0.02  # 2% of range
        
        if min_distance <= proximity_threshold:
            # Near a Fibonacci level - potential reversal
            if current_price < fib_levels.level_500:
                signal_type = "BUY"  # In lower half, expect bounce
            else:
                signal_type = "SELL"  # In upper half, expect pullback
            
            strength = 1.0 - (min_distance / proximity_threshold)
        
        return {
            'signal_type': signal_type,
            'strength': strength,
            'confidence': fib_levels.confidence,
            'position': fib_levels.position,
            'closest_level': closest_level,
            'levels': fib_levels.to_dict(),
            'num_bars_used': self.config.fibo_num_bars,
            'recent_high': recent_high,
            'recent_low': recent_low
        }
    
    def _analyze_macd(self, data: pd.DataFrame) -> TradingSignal:
        """MACD Analysis with unusual 50/26/20 configuration"""
        close_prices = data['close'].values
        
        # Calculate MACD with MQ5 settings (Fast=50 > Slow=26 is unusual!)
        macd_line, signal_line, histogram = self.calculator.macd(
            close_prices, 
            self.config.macd_fast_period,  # 50
            self.config.macd_slow_period,  # 26
            self.config.macd_signal_period  # 20
        )
        
        # Analyze signal
        signal_type = "HOLD"
        strength = 0.0
        
        # MACD signals
        if macd_line > signal_line and histogram > 0:
            signal_type = "BUY"
            strength = min(1.0, abs(histogram) / (abs(macd_line) + 1e-8))
        elif macd_line < signal_line and histogram < 0:
            signal_type = "SELL"
            strength = min(1.0, abs(histogram) / (abs(macd_line) + 1e-8))
        
        # Confidence based on histogram strength
        confidence = min(1.0, abs(histogram) * 100)  # Scale for confidence
        
        return TradingSignal(
            indicator="MACD",
            value=macd_line,
            signal=signal_type,
            strength=strength,
            confidence=confidence,
            interpretation=f"MACD({self.config.macd_fast_period}/{self.config.macd_slow_period}/{self.config.macd_signal_period}): Line={macd_line:.6f}, Signal={signal_line:.6f}, Hist={histogram:.6f}",
            timestamp=datetime.now()
        )
    
    def _combine_signals(self, rsi_signal: TradingSignal, fibo_signal: Dict, 
                        macd_signal: TradingSignal) -> Tuple[str, float, float]:
        """Combine all signals using AND logic (all must agree)"""
        
        # Extract individual signals
        signals = [rsi_signal.signal, fibo_signal['signal_type'], macd_signal.signal]
        strengths = [rsi_signal.strength, fibo_signal['strength'], macd_signal.strength]
        confidences = [rsi_signal.confidence, fibo_signal['confidence'], macd_signal.confidence]
        
        # All signals must agree for valid setup
        if all(s == "BUY" for s in signals):
            combined_signal = "BUY"
        elif all(s == "SELL" for s in signals):
            combined_signal = "SELL"
        else:
            combined_signal = "HOLD"
        
        # Combined strength and confidence
        if combined_signal != "HOLD":
            combined_strength = np.mean(strengths)
            combined_confidence = np.mean(confidences)
        else:
            combined_strength = 0.0
            combined_confidence = 0.0
        
        return combined_signal, combined_strength, combined_confidence
    
    def _calculate_sl_tp(self, current_price: float, signal: str, symbol: str) -> Tuple[float, float, float]:
        """Calculate SL/TP based on MQ5 point values"""
        if signal == "HOLD":
            return 0.0, 0.0, 0.0
        
        # Get point value for symbol
        point_value = self.calculator.get_point_value(symbol)
        
        # Convert MQ5 points to price
        sl_distance = self.config.stop_loss * point_value
        tp_distance = self.config.take_profit * point_value
        
        if signal == "BUY":
            sl_price = current_price - sl_distance
            tp_price = current_price + tp_distance
        else:  # SELL
            sl_price = current_price + sl_distance
            tp_price = current_price - tp_distance
        
        # Calculate actual risk/reward ratio
        risk = abs(current_price - sl_price)
        reward = abs(tp_price - current_price)
        risk_reward = reward / risk if risk > 0 else 0.0
        
        return sl_price, tp_price, risk_reward
    
    def _validate_setup(self, signal: str, strength: float, confidence: float, 
                       risk_reward: float) -> bool:
        """Validate setup based on quality thresholds"""
        if signal == "HOLD":
            return False
        
        # Quality thresholds
        min_strength = 0.3      # Minimum signal strength
        min_confidence = 0.2    # Minimum confidence
        min_risk_reward = 1.0   # Minimum R:R ratio
        
        return (strength >= min_strength and 
                confidence >= min_confidence and 
                risk_reward >= min_risk_reward)
    
    def _generate_feature_vector(self, data: pd.DataFrame, rsi_signal: TradingSignal,
                               fibo_signal: Dict, macd_signal: TradingSignal,
                               symbol: str) -> Dict[str, float]:
        """Generate feature vector for ML training"""
        features = {}
        
        # RSI features
        features.update({
            'rsi_value': rsi_signal.value,
            'rsi_signal': self._encode_signal(rsi_signal.signal),
            'rsi_strength': rsi_signal.strength,
            'rsi_confidence': rsi_signal.confidence,
            'rsi_oversold': 1.0 if rsi_signal.value <= self.config.rsi_down_level else 0.0,
            'rsi_overbought': 1.0 if rsi_signal.value >= self.config.rsi_up_level else 0.0
        })
        
        # Fibonacci features
        features.update({
            'fibo_signal': self._encode_signal(fibo_signal['signal_type']),
            'fibo_strength': fibo_signal['strength'],
            'fibo_confidence': fibo_signal['confidence'],
            'fibo_position': self._encode_fib_position(fibo_signal['position']),
            'fibo_range_ratio': (fibo_signal['recent_high'] - fibo_signal['recent_low']) / fibo_signal['recent_high']
        })
        
        # Add individual Fibonacci levels
        if 'levels' in fibo_signal:
            for level_name, level_value in fibo_signal['levels'].items():
                if level_name != 'current_price':
                    features[f'fib_{level_name}'] = level_value
        
        # MACD features
        features.update({
            'macd_line': macd_signal.value,
            'macd_signal': self._encode_signal(macd_signal.signal),
            'macd_strength': macd_signal.strength,
            'macd_confidence': macd_signal.confidence
        })
        
        # Combined features
        combined_strength = np.mean([rsi_signal.strength, fibo_signal['strength'], macd_signal.strength])
        combined_confidence = np.mean([rsi_signal.confidence, fibo_signal['confidence'], macd_signal.confidence])
        
        features.update({
            'combined_strength': combined_strength,
            'combined_confidence': combined_confidence,
            'signals_agreement': self._calculate_agreement([rsi_signal.signal, fibo_signal['signal_type'], macd_signal.signal])
        })
        
        # Market context
        features.update({
            'current_price': data['close'].iloc[-1],
            'volatility': np.std(data['close'].tail(20)) / data['close'].iloc[-1],
            'hour_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday()
        })
        
        return features
    
    def _encode_signal(self, signal: str) -> float:
        """Encode signal type as number"""
        signal_map = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}
        return signal_map.get(signal, 0.0)
    
    def _encode_fib_position(self, position: str) -> float:
        """Encode fibonacci position"""
        position_map = {
            "at_high": 1.0, "near_236": 0.8, "near_382": 0.6,
            "near_500": 0.5, "near_618": 0.4, "at_low": 0.0, "unknown": 0.5
        }
        return position_map.get(position, 0.5)
    
    def _calculate_agreement(self, signals: List[str]) -> float:
        """Calculate agreement level between signals"""
        if all(s == signals[0] for s in signals):
            return 1.0  # Perfect agreement
        elif all(s != "HOLD" for s in signals):
            return 0.0  # All active but disagree
        else:
            return 0.5  # Partial agreement
    
    def _create_no_signal(self) -> MQ5SetupSignal:
        """Create a no-signal result"""
        return MQ5SetupSignal(
            timestamp=datetime.now(),
            signal_type="HOLD",
            strength=0.0,
            confidence=0.0,
            rsi_signal=TradingSignal("RSI", 50.0, "HOLD", 0.0, 0.0, "No signal", datetime.now()),
            fibo_signal={'signal_type': 'HOLD', 'strength': 0.0, 'confidence': 0.0, 'position': 'unknown'},
            macd_signal=TradingSignal("MACD", 0.0, "HOLD", 0.0, 0.0, "No signal", datetime.now()),
            is_valid_setup=False,
            expected_sl=0.0,
            expected_tp=0.0,
            risk_reward=0.0,
            feature_vector={}
        )


class MQ5ThresholdOptimizer:
    """
    Optimizes the thresholds of the MQ5 strategy for maximum profitability.
    
    This class takes the base profitable MQ5 configuration and fine-tunes
    the key parameters (RSI levels, Fibonacci sensitivity, MACD periods)
    using historical data to find the optimal settings.
    """
    
    def __init__(self, base_config: Optional[MQ5StrategyConfig] = None):
        self.base_config = base_config or MQ5StrategyConfig()
        self.replicator = MQ5StrategyReplicator(self.base_config)
        
    def optimize_thresholds(self, historical_data: pd.DataFrame, 
                          n_trials: int = 200) -> Dict[str, Any]:
        """
        Optimize strategy thresholds using Optuna.
        
        Parameters to optimize:
        - RSI levels (currently 20/70)
        - RSI period (currently 2)
        - Fibonacci sensitivity 
        - MACD periods (currently 50/26/20)
        - SL/TP ratios
        """
        import optuna
        
        def objective(trial):
            # Suggest optimized parameters
            optimized_config = MQ5StrategyConfig(
                # RSI optimization
                rsi_period=trial.suggest_int('rsi_period', 2, 8),
                rsi_up_level=trial.suggest_float('rsi_up_level', 65.0, 85.0),
                rsi_down_level=trial.suggest_float('rsi_down_level', 15.0, 35.0),
                
                # Fibonacci optimization
                fibo_num_bars=trial.suggest_int('fibo_num_bars', 3, 10),
                
                # MACD optimization (keep unusual fast>slow if profitable)
                macd_fast_period=trial.suggest_int('macd_fast_period', 40, 60),
                macd_slow_period=trial.suggest_int('macd_slow_period', 20, 30),
                macd_signal_period=trial.suggest_int('macd_signal_period', 15, 25),
                
                # Risk management optimization
                stop_loss=trial.suggest_int('stop_loss', 10000, 20000),
                take_profit=trial.suggest_int('take_profit', 15000, 30000)
            )
            
            # Test configuration on historical data
            optimizer_replicator = MQ5StrategyReplicator(optimized_config)
            return self._evaluate_config(optimizer_replicator, historical_data)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study
        }
    
    def _evaluate_config(self, replicator: MQ5StrategyReplicator, 
                        data: pd.DataFrame) -> float:
        """Evaluate configuration performance on historical data"""
        signals = []
        
        # Generate signals for sliding windows
        window_size = 100
        for i in range(window_size, len(data), 10):  # Every 10 bars
            window_data = data.iloc[i-window_size:i]
            signal = replicator.analyze_setup(window_data, "US30")
            if signal.is_valid_setup:
                signals.append(signal)
        
        if len(signals) < 5:  # Need minimum signals
            return 0.0
        
        # Calculate performance metrics
        total_trades = len(signals)
        profitable_trades = sum(1 for s in signals if s.risk_reward > 1.0)
        win_rate = profitable_trades / total_trades
        
        # Average signal quality
        avg_strength = np.mean([s.strength for s in signals])
        avg_confidence = np.mean([s.confidence for s in signals])
        avg_risk_reward = np.mean([s.risk_reward for s in signals])
        
        # Composite score
        score = (win_rate * 0.4 + 
                avg_strength * 0.3 + 
                avg_confidence * 0.2 + 
                min(avg_risk_reward / 2.0, 1.0) * 0.1)
        
        return score