"""
Consolidated Dynamic Feature Engineering Pipeline for LightGBM ML Scoring System

This module integrates:
1. Botardo's UltimateTradingCalculator compatibility
2. MQ5 profitable strategy replication (testing1.txt)
3. Core indicators: RSI, MACD, RVI, ADX, Fibonacci, Volume, ATR
4. Dynamic parameter optimization for LightGBM
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import warnings
import typer
from loguru import logger
from tqdm import tqdm

from lightbgm.config import PROCESSED_DATA_DIR
from lightbgm.indicators.botardo_compatibility import (
    BotardoCompatibleCalculator, 
    BotardoFeatureMapper,
    TradingSignal, 
    FibonacciLevels
)
from lightbgm.indicators.mq5_strategy_replicator import (
    MQ5StrategyReplicator,
    MQ5StrategyConfig,
    MQ5SetupSignal
)

warnings.filterwarnings('ignore')

app = typer.Typer()


@dataclass
class ConsolidatedFeatureConfig:
    """
    Consolidated configuration integrating:
    1. Botardo's proven calculations
    2. MQ5 profitable strategy settings
    3. Core base indicators (RSI, MACD, RVI, ADX, ATR, Volume, Fibonacci)
    """
    
    # FIBONACCI parameters (from MQ5 profitable strategy)
    fib_lookback_period: int = 4  # MQ5 uses only 4 bars!
    fib_levels: List[float] = None
    fib_timeframe: str = "H1"
    
    # MACD parameters (from MQ5 profitable: 50/26/20)
    macd_fast_period: int = 50    # Unusual: Fast > Slow (from MQ5)
    macd_slow_period: int = 26
    macd_signal_period: int = 20  # Longer signal period
    macd_timeframe: str = "H4"
    
    # RSI parameters (from MQ5 profitable: period=2, levels=20/70)
    rsi_period: int = 2           # Very sensitive (from MQ5)
    rsi_oversold: float = 20.0    # Lower threshold (from MQ5)
    rsi_overbought: float = 70.0  # From MQ5
    rsi_timeframe: str = "M30"
    
    # RVI parameters (Base indicator from botardo)
    rvi_period: int = 14
    rvi_signal_period: int = 4    # Signal SMA period
    rvi_timeframe: str = "M15"
    
    # ADX parameters (Base indicator from botardo)
    adx_period: int = 14
    adx_threshold: float = 25.0   # Trend strength threshold
    adx_timeframe: str = "H1"
    
    # ATR parameters (Base indicator - multi-timeframe)
    atr_period: int = 14
    atr_d1_multiplier: float = 3.0  # Catastrophic stop
    atr_h4_multiplier: float = 2.0  # Working stop
    atr_h1_multiplier: float = 1.5  # Trailing stop
    
    # Volume parameters (Base indicator)
    volume_ma_period: int = 20
    volume_spike_threshold: float = 2.0
    volume_timeframe: str = "H1"
    
    # Divergence detection parameters
    divergence_lookback: int = 20
    divergence_min_bars: int = 5
    
    # Risk management (from MQ5 profitable)
    stop_loss_points: int = 15000   # From MQ5
    take_profit_points: int = 20000 # From MQ5
    risk_reward_target: float = 1.33
    
    def __post_init__(self):
        if self.fib_levels is None:
            self.fib_levels = [23.6, 38.2, 50.0, 61.8, 78.6]


class ConsolidatedFeatureEngineer:
    """
    Consolidated Feature Engineering Pipeline integrating:
    1. Botardo's UltimateTradingCalculator (exact compatibility)
    2. MQ5 profitable strategy replication (testing1.txt)
    3. Core base indicators: RSI, MACD, RVI, ADX, ATR, Volume, Fibonacci
    4. Dynamic optimization for LightGBM training
    
    This ensures lightbgm can enhance botardo with perfect compatibility.
    """
    
    def __init__(self, config: ConsolidatedFeatureConfig):
        self.config = config
        self.botardo_calculator = BotardoCompatibleCalculator()
        self.botardo_mapper = BotardoFeatureMapper()
        self.mq5_replicator = MQ5StrategyReplicator(self._config_to_mq5_config())
        
        logger.info("🚀 Consolidated Feature Engineer initialized")
        logger.info(f"📊 RSI: Period={config.rsi_period}, Levels={config.rsi_oversold}-{config.rsi_overbought}")
        logger.info(f"📈 MACD: {config.macd_fast_period}/{config.macd_slow_period}/{config.macd_signal_period}")
        logger.info(f"🌀 Fibonacci: {config.fib_lookback_period} bars lookback")
        logger.info(f"⚡ RVI: Period={config.rvi_period}, Signal={config.rvi_signal_period}")
        logger.info(f"📊 ADX: Period={config.adx_period}, Threshold={config.adx_threshold}")
        logger.info(f"📈 ATR: Multi-TF with multipliers D1={config.atr_d1_multiplier}, H4={config.atr_h4_multiplier}")
    
    def _config_to_mq5_config(self) -> MQ5StrategyConfig:
        """Convert consolidated config to MQ5 config"""
        return MQ5StrategyConfig(
            rsi_period=self.config.rsi_period,
            rsi_up_level=self.config.rsi_overbought,
            rsi_down_level=self.config.rsi_oversold,
            fibo_num_bars=self.config.fib_lookback_period,
            macd_fast_period=self.config.macd_fast_period,
            macd_slow_period=self.config.macd_slow_period,
            macd_signal_period=self.config.macd_signal_period,
            stop_loss=self.config.stop_loss_points,
            take_profit=self.config.take_profit_points
        )
        
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features for the dataset.
        
        Args:
            data: DataFrame with OHLCV data and metadata
            
        Returns:
            DataFrame with engineered features
        """
        features_list = []
        
        # Group by symbol and timeframe for feature generation
        for (symbol, timeframe), group_data in data.groupby(['symbol', 'timeframe']):
            logger.info(f"Generating features for {symbol} - {timeframe}")
            
            if len(group_data) < self.config.fib_lookback_period:
                continue
                
            group_features = self._generate_features_for_group(group_data, symbol, timeframe)
            features_list.append(group_features)
        
        if features_list:
            return pd.concat(features_list, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _generate_features_for_group(self, data: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        """Generate features for a specific symbol-timeframe group."""
        data = data.copy().sort_values('timestamp')
        
        # Initialize feature columns
        features = data[['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        # Generate features based on timeframe
        if timeframe == "D1":
            features = self._add_fibonacci_features(features)
            
        elif timeframe == "H4":
            features = self._add_macd_features(features)
            features = self._add_atr_features(features, 'D1')
            
        elif timeframe == "H1":
            features = self._add_rsi_features(features)
            features = self._add_volume_features(features)
            features = self._add_atr_features(features, 'H1')
            
        elif timeframe == "M15":
            features = self._add_rvi_features(features)
        
        # Add temporal features
        features = self._add_temporal_features(features)
        
        # Add divergence features
        features = self._add_divergence_features(features)
        
        return features
    
    def _add_fibonacci_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add Fibonacci retracement features (D1 timeframe)."""
        data = data.copy()
        
        # Calculate rolling high and low for Fibonacci levels
        high_period = data['high'].rolling(window=self.config.fib_lookback_period).max()
        low_period = data['low'].rolling(window=self.config.fib_lookback_period).min()
        
        # Calculate Fibonacci levels
        fib_range = high_period - low_period
        
        # Find closest Fibonacci level
        closest_levels = []
        closest_distances = []
        
        for i, close_price in enumerate(data['close']):
            if pd.isna(fib_range.iloc[i]) or fib_range.iloc[i] == 0:
                closest_levels.append(np.nan)
                closest_distances.append(np.nan)
                continue
                
            distances = []
            for level in self.config.fib_levels:
                fib_price = low_period.iloc[i] + (fib_range.iloc[i] * level / 100)
                distance = abs(close_price - fib_price) / fib_range.iloc[i]
                distances.append((distance, level))
            
            # Find minimum distance
            min_distance, closest_level = min(distances, key=lambda x: x[0])
            closest_levels.append(closest_level)
            closest_distances.append(min_distance)
        
        data['fib_closest_level'] = closest_levels
        data['fib_level_proximity'] = closest_distances
        
        return data
    
    def _add_macd_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add MACD features (H4 timeframe)."""
        data = data.copy()
        
        # Calculate MACD
        macd_indicator = MACD(
            close=data['close'],
            window_slow=self.config.macd_slow_period,
            window_fast=self.config.macd_fast_period,
            window_sign=self.config.macd_signal_period
        )
        
        macd_line = macd_indicator.macd()
        macd_signal = macd_indicator.macd_signal()
        macd_histogram = macd_indicator.macd_diff()
        
        # Trend state
        data['macd_trend_state'] = np.where(
            macd_line > macd_signal, 1,  # BULLISH
            np.where(macd_line < macd_signal, -1, 0)  # BEARISH, NEUTRAL
        )
        
        # Trend strength (normalized MACD histogram)
        data['macd_trend_strength'] = np.abs(macd_histogram) / (data['close'].rolling(20).std() + 1e-8)
        
        # Histogram acceleration
        data['macd_histogram_acceleration'] = macd_histogram.diff()
        
        return data
    
    def _add_rvi_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add RVI-like features using Stochastic oscillator (M15 timeframe)."""
        data = data.copy()
        
        # Use Stochastic oscillator as RVI proxy
        stoch = StochasticOscillator(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            window=self.config.rvi_period,
            smooth_window=self.config.rvi_signal_period
        )
        
        rvi_value = stoch.stoch()
        rvi_signal = stoch.stoch_signal()
        
        data['rvi_value'] = rvi_value
        
        # Signal crosses
        data['rvi_signal_cross'] = np.where(
            (rvi_value > rvi_signal) & (rvi_value.shift(1) <= rvi_signal.shift(1)), 1,  # BULLISH_CROSS
            np.where(
                (rvi_value < rvi_signal) & (rvi_value.shift(1) >= rvi_signal.shift(1)), -1,  # BEARISH_CROSS
                0  # NONE
            )
        )
        
        return data
    
    def _add_rsi_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add RSI features (H1 timeframe)."""
        data = data.copy()
        
        # Calculate RSI
        rsi_indicator = RSIIndicator(close=data['close'], window=self.config.rsi_period)
        rsi_value = rsi_indicator.rsi()
        
        data['rsi_value'] = rsi_value
        
        # RSI zones
        data['rsi_zone'] = np.where(
            rsi_value <= self.config.rsi_oversold, -1,  # OVERSOLD
            np.where(rsi_value >= self.config.rsi_overbought, 1, 0)  # OVERBOUGHT, NEUTRAL
        )
        
        return data
    
    def _add_atr_features(self, data: pd.DataFrame, timeframe_label: str) -> pd.DataFrame:
        """Add ATR features for volatility measurement."""
        data = data.copy()
        
        # Calculate ATR
        atr_indicator = AverageTrueRange(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            window=self.config.atr_period
        )
        
        atr_value = atr_indicator.average_true_range()
        
        # ATR ratio (current ATR vs rolling mean)
        atr_mean = atr_value.rolling(window=50).mean()
        atr_ratio = atr_value / (atr_mean + 1e-8)
        
        data[f'atr_{timeframe_label.lower()}_ratio'] = atr_ratio
        
        return data
    
    def _add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume features (H1 timeframe)."""
        data = data.copy()
        
        # Volume moving average
        volume_ma = data['volume'].rolling(window=self.config.volume_ma_period).mean()
        data['volume_ma_ratio'] = data['volume'] / (volume_ma + 1e-8)
        
        # Volume spikes
        data['is_volume_spike'] = (
            data['volume_ma_ratio'] > self.config.volume_spike_threshold
        ).astype(int)
        
        return data
    
    def _add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add temporal context features."""
        data = data.copy()
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        data['hour_of_day'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        
        # Market sessions (simplified)
        data['market_session'] = np.where(
            data['hour_of_day'].between(0, 7), 0,  # ASIAN
            np.where(data['hour_of_day'].between(8, 15), 1,  # EUROPEAN
                     2)  # AMERICAN
        )
        
        # Placeholder for news timing (would require external news calendar)
        data['minutes_to_next_news'] = np.nan
        
        return data
    
    def _add_divergence_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add divergence detection features."""
        data = data.copy()
        
        # Simplified divergence detection
        price_momentum = data['close'].pct_change(self.config.divergence_lookback)
        
        # For MACD divergence (if MACD data exists)
        if 'macd_trend_strength' in data.columns:
            macd_momentum = data['macd_trend_strength'].pct_change(self.config.divergence_lookback)
            data['price_macd_divergence_h4'] = np.where(
                (price_momentum > 0) & (macd_momentum < 0), -1,  # Bearish divergence
                np.where((price_momentum < 0) & (macd_momentum > 0), 1, 0)  # Bullish divergence
            )
        else:
            data['price_macd_divergence_h4'] = 0
        
        # For RVI divergence (if RVI data exists)
        if 'rvi_value' in data.columns:
            rvi_momentum = data['rvi_value'].pct_change(self.config.divergence_lookback)
            data['price_rvi_divergence_m15'] = np.where(
                (price_momentum > 0) & (rvi_momentum < 0), -1,
                np.where((price_momentum < 0) & (rvi_momentum > 0), 1, 0)
            )
        else:
            data['price_rvi_divergence_m15'] = 0
        
        # For RSI divergence (if RSI data exists)
        if 'rsi_value' in data.columns:
            rsi_momentum = data['rsi_value'].pct_change(self.config.divergence_lookback)
            data['price_rsi_divergence_h1'] = np.where(
                (price_momentum > 0) & (rsi_momentum < 0), -1,
                np.where((price_momentum < 0) & (rsi_momentum > 0), 1, 0)
            )
        else:
            data['price_rsi_divergence_h1'] = 0
        
        return data


def create_optimizable_features(trial, data: pd.DataFrame) -> pd.DataFrame:
    """
    Create features with Optuna-optimized parameters for LightGBM training.
    
    Args:
        trial: Optuna trial object for parameter optimization
        data: Input OHLCV data
        
    Returns:
        DataFrame with optimized features
    """
    # Suggest optimal parameters
    config = FeatureConfig(
        # Fibonacci parameters
        fib_lookback_period=trial.suggest_int('fib_lookback_period', 50, 200),
        
        # MACD parameters
        macd_fast_period=trial.suggest_int('macd_fast_period', 8, 16),
        macd_slow_period=trial.suggest_int('macd_slow_period', 20, 35),
        macd_signal_period=trial.suggest_int('macd_signal_period', 6, 12),
        
        # RVI parameters
        rvi_period=trial.suggest_int('rvi_period', 10, 20),
        rvi_signal_period=trial.suggest_int('rvi_signal_period', 2, 5),
        
        # RSI parameters
        rsi_period=trial.suggest_int('rsi_period', 10, 20),
        rsi_oversold=trial.suggest_float('rsi_oversold', 20, 35),
        rsi_overbought=trial.suggest_float('rsi_overbought', 65, 80),
        
        # ATR parameters
        atr_period=trial.suggest_int('atr_period', 10, 20),
        
        # Volume parameters
        volume_ma_period=trial.suggest_int('volume_ma_period', 10, 30),
        volume_spike_threshold=trial.suggest_float('volume_spike_threshold', 1.5, 3.0),
        
        # Divergence parameters
        divergence_lookback=trial.suggest_int('divergence_lookback', 10, 30),
        divergence_min_bars=trial.suggest_int('divergence_min_bars', 3, 8)
    )
    
    # Generate features with optimized config
    engineer = DynamicFeatureEngineer(config)
    return engineer.generate_features(data)


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "historical_dataset.parquet",
    output_path: Path = PROCESSED_DATA_DIR / "features.parquet",
):
    """Generate features from historical dataset for LightGBM training."""
    logger.info("Starting feature engineering for LightGBM ML scoring system...")
    
    # Load historical data
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    logger.info(f"Loading data from {input_path}")
    data = pd.read_parquet(input_path)
    
    # Generate features with default config
    config = FeatureConfig()
    engineer = DynamicFeatureEngineer(config)
    
    logger.info("Generating features...")
    features = engineer.generate_features(data)
    
    # Save features
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path)
    
    logger.success(f"Features saved to {output_path}")
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Feature columns: {list(features.columns)}")


if __name__ == "__main__":
    app()
