"""Real-time setup scoring system for trading ML system."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import pickle
import json
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import lightgbm as lgb
from loguru import logger

from lightbgm.features import FeatureConfig, DynamicFeatureEngineer


@dataclass
class ScoringResult:
    """Container for setup scoring results."""
    score: float
    confidence: str  # 'HIGH', 'MEDIUM', 'LOW'
    features: Dict[str, float]
    timestamp: datetime
    symbol: str
    timeframe: str
    signal_strength: float
    risk_assessment: str


class SetupScorer:
    """
    Real-time setup scoring system using trained LightGBM model.
    
    This class loads the optimized model and feature configuration
    to score trading setups in real-time.
    """
    
    def __init__(self, model_path: str, config_path: str, params_path: str):
        """
        Initialize the scorer with trained artifacts.
        
        Args:
            model_path: Path to the trained LightGBM model
            config_path: Path to the feature configuration
            params_path: Path to the best parameters
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.params_path = Path(params_path)
        
        # Load artifacts
        self._load_artifacts()
        
        # Initialize feature engineer
        self.feature_engineer = DynamicFeatureEngineer(self.feature_config)
        
        logger.info("SetupScorer initialized successfully")
    
    def _load_artifacts(self) -> None:
        """Load trained model and configuration artifacts."""
        try:
            # Load trained model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load feature configuration
            with open(self.config_path, 'r') as f:
                config_dict = json.load(f)
                self.feature_config = FeatureConfig(**config_dict)
            
            # Load best parameters
            with open(self.params_path, 'r') as f:
                self.best_params = json.load(f)
            
            logger.success("All artifacts loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            raise
    
    def score_setup(self, 
                   market_data: pd.DataFrame, 
                   symbol: str, 
                   timeframe: str) -> ScoringResult:
        """
        Score a trading setup in real-time.
        
        Args:
            market_data: Recent OHLCV data for the symbol/timeframe
            symbol: Trading symbol (e.g., 'US30', 'US100', 'US500')
            timeframe: Timeframe ('D1', 'H4', 'H1', 'M15')
            
        Returns:
            ScoringResult with probability and metadata
        """
        try:
            # Prepare data for feature engineering
            data_with_metadata = market_data.copy()
            data_with_metadata['symbol'] = symbol
            data_with_metadata['timeframe'] = timeframe
            data_with_metadata['timestamp'] = data_with_metadata.index
            data_with_metadata = data_with_metadata.reset_index(drop=True)
            
            # Generate features
            features_df = self.feature_engineer.generate_features(data_with_metadata)
            
            if features_df.empty:
                logger.warning("No features generated for scoring")
                return self._create_empty_result(symbol, timeframe)
            
            # Get the latest feature row
            latest_features = features_df.iloc[-1:].copy()
            
            # Prepare features for prediction
            feature_cols = self._get_feature_columns(latest_features)
            X = latest_features[feature_cols].fillna(0)
            
            # Make prediction
            probability = self.model.predict(X, num_iteration=self.model.best_iteration)[0]
            
            # Determine confidence level
            confidence = self._determine_confidence(probability)
            
            # Calculate signal strength
            signal_strength = self._calculate_signal_strength(latest_features.iloc[0])
            
            # Assess risk
            risk_assessment = self._assess_risk(latest_features.iloc[0], probability)
            
            # Create result
            result = ScoringResult(
                score=float(probability),
                confidence=confidence,
                features=latest_features.iloc[0][feature_cols].to_dict(),
                timestamp=datetime.now(),
                symbol=symbol,
                timeframe=timeframe,
                signal_strength=signal_strength,
                risk_assessment=risk_assessment
            )
            
            logger.info(f"Setup scored: {symbol}-{timeframe} = {probability:.4f} ({confidence})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error scoring setup: {e}")
            return self._create_empty_result(symbol, timeframe)
    
    def score_multi_timeframe(self, 
                            market_data_dict: Dict[str, pd.DataFrame], 
                            symbol: str) -> Dict[str, ScoringResult]:
        """
        Score setups across multiple timeframes for confluence analysis.
        
        Args:
            market_data_dict: Dictionary of timeframe -> OHLCV data
            symbol: Trading symbol
            
        Returns:
            Dictionary of timeframe -> ScoringResult
        """
        results = {}
        
        for timeframe, data in market_data_dict.items():
            try:
                result = self.score_setup(data, symbol, timeframe)
                results[timeframe] = result
                
            except Exception as e:
                logger.error(f"Error scoring {symbol}-{timeframe}: {e}")
                results[timeframe] = self._create_empty_result(symbol, timeframe)
        
        return results
    
    def find_confluence_setups(self, 
                              multi_tf_results: Dict[str, ScoringResult],
                              min_confluence_score: float = 0.7) -> Optional[Dict[str, Any]]:
        """
        Analyze multi-timeframe results for confluence signals.
        
        Args:
            multi_tf_results: Results from multiple timeframes
            min_confluence_score: Minimum average score for confluence
            
        Returns:
            Confluence analysis or None if no confluence found
        """
        # Extract scores and confidences
        scores = [r.score for r in multi_tf_results.values() if r.score > 0]
        confidences = [r.confidence for r in multi_tf_results.values() if r.score > 0]
        
        if len(scores) < 2:  # Need at least 2 timeframes
            return None
        
        # Calculate confluence metrics
        avg_score = np.mean(scores)
        min_score = np.min(scores)
        score_std = np.std(scores)
        
        # Count high confidence signals
        high_confidence_count = sum(1 for c in confidences if c == 'HIGH')
        
        # Determine if confluence exists
        has_confluence = (
            avg_score >= min_confluence_score and
            min_score >= 0.6 and
            score_std <= 0.2 and
            high_confidence_count >= 1
        )
        
        if has_confluence:
            return {
                'has_confluence': True,
                'average_score': avg_score,
                'minimum_score': min_score,
                'score_consistency': 1 - score_std,  # Higher is better
                'high_confidence_signals': high_confidence_count,
                'participating_timeframes': list(multi_tf_results.keys()),
                'overall_confidence': 'HIGH' if avg_score >= 0.8 else 'MEDIUM'
            }
        
        return None
    
    def _get_feature_columns(self, features_df: pd.DataFrame) -> List[str]:
        """Get the feature columns used by the model."""
        # Exclude metadata columns
        exclude_cols = [
            'symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'adj_close', 'future_return', 'setup_success'
        ]
        
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        return feature_cols
    
    def _determine_confidence(self, probability: float) -> str:
        """Determine confidence level based on probability."""
        if probability >= 0.8:
            return 'HIGH'
        elif probability >= 0.6:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _calculate_signal_strength(self, features: pd.Series) -> float:
        """Calculate overall signal strength based on key features."""
        strength_factors = []
        
        # MACD strength
        if 'macd_trend_strength' in features:
            strength_factors.append(min(features['macd_trend_strength'], 1.0))
        
        # Fibonacci proximity (closer = stronger)
        if 'fib_level_proximity' in features and not pd.isna(features['fib_level_proximity']):
            fib_strength = 1 - min(features['fib_level_proximity'], 1.0)
            strength_factors.append(fib_strength)
        
        # Volume confirmation
        if 'volume_ma_ratio' in features:
            volume_strength = min(features['volume_ma_ratio'] / 2.0, 1.0)
            strength_factors.append(volume_strength)
        
        # RSI position
        if 'rsi_value' in features:
            rsi_strength = self._rsi_strength(features['rsi_value'])
            strength_factors.append(rsi_strength)
        
        # Return average if we have factors, otherwise return 0.5
        return np.mean(strength_factors) if strength_factors else 0.5
    
    def _rsi_strength(self, rsi_value: float) -> float:
        """Calculate RSI-based signal strength."""
        if pd.isna(rsi_value):
            return 0.5
        
        # Stronger signals at oversold/overbought levels
        if rsi_value <= 30:
            return 1.0  # Strong oversold
        elif rsi_value >= 70:
            return 1.0  # Strong overbought
        elif 30 < rsi_value <= 40:
            return 0.8  # Moderate oversold
        elif 60 <= rsi_value < 70:
            return 0.8  # Moderate overbought
        else:
            return 0.3  # Neutral zone
    
    def _assess_risk(self, features: pd.Series, probability: float) -> str:
        """Assess setup risk based on features and probability."""
        risk_factors = []
        
        # Volatility risk (ATR)
        if 'atr_d1_ratio' in features and not pd.isna(features['atr_d1_ratio']):
            if features['atr_d1_ratio'] > 1.5:
                risk_factors.append('HIGH_VOLATILITY')
        
        # Volume risk
        if 'volume_ma_ratio' in features:
            if features['volume_ma_ratio'] < 0.5:
                risk_factors.append('LOW_VOLUME')
        
        # Probability risk
        if probability < 0.6:
            risk_factors.append('LOW_PROBABILITY')
        
        # Market session risk (simplified)
        if 'market_session' in features:
            if features['market_session'] == 0:  # Asian session - typically lower volume
                risk_factors.append('LOW_ACTIVITY_SESSION')
        
        # Determine overall risk
        if len(risk_factors) >= 3:
            return 'HIGH'
        elif len(risk_factors) >= 1:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _create_empty_result(self, symbol: str, timeframe: str) -> ScoringResult:
        """Create empty result for error cases."""
        return ScoringResult(
            score=0.0,
            confidence='LOW',
            features={},
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=timeframe,
            signal_strength=0.0,
            risk_assessment='HIGH'
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_type': 'LightGBM',
            'feature_config': self.feature_config.__dict__,
            'best_params': self.best_params,
            'num_features': self.model.num_feature(),
            'num_trees': self.model.num_trees(),
            'objective': self.model.params.get('objective', 'unknown')
        }


class RealtimeFeatureCalculator:
    """
    Helper class for calculating features in real-time with minimal data.
    
    This is optimized for live trading where you may only have recent bars.
    """
    
    def __init__(self, feature_config: FeatureConfig):
        self.config = feature_config
        self.engineer = DynamicFeatureEngineer(feature_config)
    
    def calculate_minimal_features(self, 
                                 recent_data: pd.DataFrame, 
                                 symbol: str, 
                                 timeframe: str) -> Optional[pd.Series]:
        """
        Calculate features from minimal recent data.
        
        Args:
            recent_data: Recent OHLCV data (minimum required bars)
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            Feature series or None if insufficient data
        """
        # Ensure we have enough data
        min_required = max(
            self.config.fib_lookback_period,
            self.config.macd_slow_period + self.config.macd_signal_period,
            50  # For various rolling calculations
        )
        
        if len(recent_data) < min_required:
            logger.warning(f"Insufficient data: {len(recent_data)} < {min_required}")
            return None
        
        # Prepare data
        data_with_metadata = recent_data.copy()
        data_with_metadata['symbol'] = symbol
        data_with_metadata['timeframe'] = timeframe
        data_with_metadata['timestamp'] = data_with_metadata.index
        data_with_metadata = data_with_metadata.reset_index(drop=True)
        
        # Generate features
        features_df = self.engineer.generate_features(data_with_metadata)
        
        if features_df.empty:
            return None
        
        # Return latest feature row
        return features_df.iloc[-1]


def create_scorer_from_artifacts(artifacts_dir: str) -> SetupScorer:
    """
    Create a SetupScorer from optimization artifacts directory.
    
    Args:
        artifacts_dir: Directory containing optimization artifacts
        
    Returns:
        Initialized SetupScorer
    """
    artifacts_path = Path(artifacts_dir)
    
    model_path = artifacts_path / "best_model.pkl"
    config_path = artifacts_path / "feature_config.json"
    params_path = artifacts_path / "best_params.json"
    
    return SetupScorer(
        model_path=str(model_path),
        config_path=str(config_path),
        params_path=str(params_path)
    )


if __name__ == "__main__":
    # Example usage
    artifacts_dir = "models/optimization_results"
    
    try:
        scorer = create_scorer_from_artifacts(artifacts_dir)
        print("SetupScorer created successfully!")
        print(f"Model info: {scorer.get_model_info()}")
        
    except Exception as e:
        print(f"Error creating scorer: {e}")
        print("Make sure to run optimization first to generate artifacts.")