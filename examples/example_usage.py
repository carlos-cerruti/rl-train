"""
Example usage of the LightGBM ML Scoring System for Trading Indices.

This script demonstrates how to use the complete pipeline from data generation
to real-time setup scoring.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the lightbgm package to path
sys.path.append(str(Path(__file__).parent.parent))

from lightbgm.data_sources.historical_generator import HistoricalDataGenerator, DataConfig
from lightbgm.features import DynamicFeatureEngineer, FeatureConfig
from lightbgm.optimization.optuna_optimizer import LightGBMOptunaOptimizer
from lightbgm.scoring.setup_scorer import SetupScorer, create_scorer_from_artifacts
from lightbgm.modeling.train import *

from loguru import logger


def example_1_quick_test():
    """Example 1: Quick test with sample data"""
    logger.info("=== Example 1: Quick Test with Sample Data ===")
    
    # Generate sample market data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='4H')
    
    # Sample OHLCV data for US30
    np.random.seed(42)
    base_price = 35000
    sample_data = pd.DataFrame({
        'open': base_price + np.cumsum(np.random.normal(0, 50, len(dates))),
        'high': base_price + np.cumsum(np.random.normal(25, 50, len(dates))),
        'low': base_price + np.cumsum(np.random.normal(-25, 50, len(dates))),
        'close': base_price + np.cumsum(np.random.normal(0, 50, len(dates))),
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Ensure OHLC logic
    for i in range(len(sample_data)):
        sample_data.iloc[i]['high'] = max(sample_data.iloc[i]['open'], sample_data.iloc[i]['close']) + abs(np.random.normal(0, 20))
        sample_data.iloc[i]['low'] = min(sample_data.iloc[i]['open'], sample_data.iloc[i]['close']) - abs(np.random.normal(0, 20))
    
    # Test feature engineering
    logger.info("Testing feature engineering...")
    
    # Prepare data for feature engineering
    data_with_metadata = sample_data.copy()
    data_with_metadata['symbol'] = 'US30'
    data_with_metadata['timeframe'] = 'H4'
    data_with_metadata['timestamp'] = data_with_metadata.index
    data_with_metadata = data_with_metadata.reset_index(drop=True)
    
    # Generate features
    config = FeatureConfig()
    engineer = DynamicFeatureEngineer(config)
    features = engineer.generate_features(data_with_metadata)
    
    logger.success(f"Features generated: {features.shape}")
    logger.info(f"Feature columns: {list(features.columns)}")
    
    # Show sample of features
    if not features.empty:
        logger.info("Sample features (last 5 rows):")
        print(features.tail().to_string())


def example_2_full_pipeline_small():
    """Example 2: Run full pipeline with small dataset for testing"""
    logger.info("=== Example 2: Full Pipeline with Small Dataset ===")
    
    try:
        # Step 1: Generate small dataset
        logger.info("Step 1: Generating small historical dataset...")
        
        config = DataConfig(
            symbols=["US30", "US100"],  # Just 2 symbols for testing
            timeframes=["H4", "H1"],    # Just 2 timeframes
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 6, 1),  # 5 months of data
            data_source="yfinance"
        )
        
        generator = HistoricalDataGenerator(config)
        
        # Create output directory
        output_dir = Path("examples/test_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset = generator.generate_dataset(str(output_dir))
        logger.success(f"Dataset generated: {dataset.shape}")
        
        # Step 2: Quick optimization (few trials for demo)
        logger.info("Step 2: Running quick optimization...")
        
        optimizer = LightGBMOptunaOptimizer(n_trials=10)  # Only 10 trials for demo
        results = optimizer.optimize(dataset)
        
        # Save artifacts
        artifacts_dir = Path("examples/test_artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        optimizer.save_artifacts(str(artifacts_dir))
        
        logger.success(f"Optimization completed with score: {results['best_score']:.4f}")
        
        # Step 3: Test scorer
        logger.info("Step 3: Testing scorer...")
        
        scorer = create_scorer_from_artifacts(str(artifacts_dir))
        
        # Create test data
        test_dates = pd.date_range(start='2023-06-01', end='2023-06-02', freq='4H')
        test_data = pd.DataFrame({
            'open': 35000 + np.random.normal(0, 100, len(test_dates)),
            'high': 35100 + np.random.normal(0, 100, len(test_dates)),
            'low': 34900 + np.random.normal(0, 100, len(test_dates)),
            'close': 35000 + np.random.normal(0, 100, len(test_dates)),
            'volume': np.random.randint(1000, 5000, len(test_dates))
        }, index=test_dates)
        
        # Score setup
        result = scorer.score_setup(test_data, "US30", "H4")
        
        logger.success("Scoring completed!")
        logger.info(f"Setup Score: {result.score:.4f}")
        logger.info(f"Confidence: {result.confidence}")
        logger.info(f"Signal Strength: {result.signal_strength:.4f}")
        logger.info(f"Risk Assessment: {result.risk_assessment}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.info("This is expected for demo purposes - some dependencies might not be installed")


def example_3_multi_timeframe_confluence():
    """Example 3: Multi-timeframe confluence analysis"""
    logger.info("=== Example 3: Multi-Timeframe Confluence Analysis ===")
    
    # This example shows how to analyze confluence across timeframes
    # Note: This requires trained artifacts
    
    artifacts_dir = Path("examples/test_artifacts")
    
    if not artifacts_dir.exists():
        logger.warning("Artifacts not found. Run example_2 first or use real artifacts.")
        return
    
    try:
        scorer = create_scorer_from_artifacts(str(artifacts_dir))
        
        # Create sample data for different timeframes
        base_time = datetime.now()
        
        timeframes_data = {
            'D1': pd.DataFrame({
                'open': [35000, 35100, 35200],
                'high': [35200, 35300, 35400],
                'low': [34800, 34900, 35000],
                'close': [35100, 35200, 35300],
                'volume': [10000, 12000, 11000]
            }, index=pd.date_range(base_time - timedelta(days=3), periods=3, freq='D')),
            
            'H4': pd.DataFrame({
                'open': [35200, 35250, 35300],
                'high': [35350, 35400, 35450],
                'low': [35150, 35200, 35250],
                'close': [35250, 35300, 35350],
                'volume': [5000, 5500, 6000]
            }, index=pd.date_range(base_time - timedelta(hours=12), periods=3, freq='4H')),
            
            'H1': pd.DataFrame({
                'open': [35300, 35320, 35340],
                'high': [35380, 35400, 35420],
                'low': [35280, 35300, 35320],
                'close': [35320, 35340, 35360],
                'volume': [2000, 2200, 2100]
            }, index=pd.date_range(base_time - timedelta(hours=3), periods=3, freq='H'))
        }
        
        # Score all timeframes
        multi_tf_results = scorer.score_multi_timeframe(timeframes_data, "US30")
        
        logger.info("Multi-timeframe scoring results:")
        for tf, result in multi_tf_results.items():
            logger.info(f"  {tf}: Score={result.score:.4f}, Confidence={result.confidence}")
        
        # Check for confluence
        confluence = scorer.find_confluence_setups(multi_tf_results)
        
        if confluence:
            logger.success("Confluence detected!")
            logger.info(f"  Average Score: {confluence['average_score']:.4f}")
            logger.info(f"  Overall Confidence: {confluence['overall_confidence']}")
            logger.info(f"  Participating TFs: {confluence['participating_timeframes']}")
        else:
            logger.info("No confluence detected in this example")
            
    except Exception as e:
        logger.error(f"Confluence analysis failed: {e}")


def example_4_real_time_simulation():
    """Example 4: Real-time scoring simulation"""
    logger.info("=== Example 4: Real-Time Scoring Simulation ===")
    
    artifacts_dir = Path("examples/test_artifacts")
    
    if not artifacts_dir.exists():
        logger.warning("Artifacts not found. Run example_2 first.")
        return
    
    try:
        scorer = create_scorer_from_artifacts(str(artifacts_dir))
        
        # Simulate real-time data updates
        logger.info("Simulating real-time data stream...")
        
        base_price = 35000
        current_time = datetime.now()
        
        # Simulate 10 price updates
        for i in range(10):
            # Generate new "real-time" data
            price_change = np.random.normal(0, 50)
            new_price = base_price + price_change
            
            # Create recent data (last 100 bars)
            recent_data = pd.DataFrame({
                'open': np.random.normal(new_price, 20, 100),
                'high': np.random.normal(new_price + 30, 25, 100),
                'low': np.random.normal(new_price - 30, 25, 100),
                'close': np.random.normal(new_price, 20, 100),
                'volume': np.random.randint(1000, 5000, 100)
            }, index=pd.date_range(current_time - timedelta(hours=400), periods=100, freq='4H'))
            
            # Score the setup
            result = scorer.score_setup(recent_data, "US30", "H4")
            
            logger.info(f"Update {i+1}: Price={new_price:.0f}, Score={result.score:.4f}, Confidence={result.confidence}")
            
            # Update for next iteration
            base_price = new_price
            current_time += timedelta(hours=4)
            
    except Exception as e:
        logger.error(f"Real-time simulation failed: {e}")


def example_5_feature_importance_analysis():
    """Example 5: Analyze feature importance from trained model"""
    logger.info("=== Example 5: Feature Importance Analysis ===")
    
    artifacts_dir = Path("examples/test_artifacts")
    importance_file = artifacts_dir / "feature_importance.csv"
    
    if not importance_file.exists():
        logger.warning("Feature importance file not found. Run optimization first.")
        return
    
    try:
        # Load feature importance
        importance_df = pd.read_csv(importance_file)
        
        logger.info("Top 10 Most Important Features:")
        top_features = importance_df.head(10)
        
        for i, row in top_features.iterrows():
            logger.info(f"  {i+1:2d}. {row['feature']:25s} = {row['importance']:8.2f}")
        
        # Analyze feature groups
        feature_groups = {
            'Fibonacci': ['fib_'],
            'MACD': ['macd_'],
            'RSI': ['rsi_'],
            'RVI': ['rvi_'],
            'Volume': ['volume_'],
            'ATR': ['atr_'],
            'Temporal': ['hour_', 'day_', 'market_session'],
            'Divergence': ['divergence']
        }
        
        logger.info("\nFeature Group Importance:")
        for group_name, prefixes in feature_groups.items():
            group_importance = 0
            count = 0
            
            for _, row in importance_df.iterrows():
                if any(prefix in row['feature'] for prefix in prefixes):
                    group_importance += row['importance']
                    count += 1
            
            if count > 0:
                avg_importance = group_importance / count
                logger.info(f"  {group_name:15s}: Avg={avg_importance:8.2f}, Total={group_importance:8.2f}, Count={count}")
        
    except Exception as e:
        logger.error(f"Feature importance analysis failed: {e}")


def main():
    """Run all examples"""
    logger.info("🚀 LightGBM ML Scoring System - Examples")
    logger.info("=" * 50)
    
    # Run examples
    try:
        example_1_quick_test()
        print("\n")
        
        # Note: Example 2 requires internet connection and may take time
        logger.info("⚠️  Example 2 requires internet connection and may take several minutes...")
        response = input("Run full pipeline example? (y/N): ")
        if response.lower() == 'y':
            example_2_full_pipeline_small()
            print("\n")
            
            example_3_multi_timeframe_confluence()
            print("\n")
            
            example_4_real_time_simulation()
            print("\n")
            
            example_5_feature_importance_analysis()
        
        logger.success("All examples completed!")
        
    except KeyboardInterrupt:
        logger.info("Examples interrupted by user")
    except Exception as e:
        logger.error(f"Examples failed: {e}")


if __name__ == "__main__":
    main()