"""Training pipeline for LightGBM ML scoring system with Optuna optimization."""

from pathlib import Path
import pandas as pd
import typer
from loguru import logger
from typing import Optional

from lightbgm.config import MODELS_DIR, PROCESSED_DATA_DIR
from lightbgm.optimization.optuna_optimizer import LightGBMOptunaOptimizer, run_optimization
from lightbgm.data_sources.historical_generator import HistoricalDataGenerator, DataConfig, create_sample_dataset
from lightbgm.features import DynamicFeatureEngineer, FeatureConfig
from lightbgm.scoring.setup_scorer import create_scorer_from_artifacts

app = typer.Typer()


@app.command()
def generate_data(
    output_dir: Path = PROCESSED_DATA_DIR,
    symbols: str = "US30,US100,US500",
    timeframes: str = "D1,H4,H1,M15",
    start_year: int = 2019,
    end_year: int = 2024
):
    """Generate historical dataset for training."""
    logger.info("Generating historical dataset for LightGBM training...")
    
    from datetime import datetime
    
    # Parse input parameters
    symbol_list = symbols.split(',')
    timeframe_list = timeframes.split(',')
    
    # Create data configuration
    config = DataConfig(
        symbols=symbol_list,
        timeframes=timeframe_list,
        start_date=datetime(start_year, 1, 1),
        end_date=datetime(end_year, 1, 1),
        data_source="yfinance"
    )
    
    # Generate dataset
    generator = HistoricalDataGenerator(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = generator.generate_dataset(str(output_dir))
    
    logger.success(f"Historical dataset generated: {dataset.shape}")


@app.command()
def generate_features(
    input_path: Path = PROCESSED_DATA_DIR / "historical_dataset.parquet",
    output_path: Path = PROCESSED_DATA_DIR / "features.parquet"
):
    """Generate features from historical dataset."""
    logger.info("Generating features for LightGBM training...")
    
    if not input_path.exists():
        logger.error(f"Input dataset not found: {input_path}")
        logger.info("Run 'generate-data' command first to create the dataset")
        return
    
    # Load data
    data = pd.read_parquet(input_path)
    logger.info(f"Loaded dataset: {data.shape}")
    
    # Generate features
    config = FeatureConfig()
    engineer = DynamicFeatureEngineer(config)
    features = engineer.generate_features(data)
    
    # Save features
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path)
    
    logger.success(f"Features generated and saved: {features.shape}")


@app.command()
def optimize(
    data_path: Path = PROCESSED_DATA_DIR / "historical_dataset.parquet",
    output_dir: Path = MODELS_DIR / "optimization_results",
    n_trials: int = 500,
    cv_folds: int = 5
):
    """Run Optuna optimization for LightGBM model and features."""
    logger.info(f"Starting LightGBM optimization with {n_trials} trials...")
    
    if not data_path.exists():
        logger.error(f"Dataset not found: {data_path}")
        logger.info("Run 'generate-data' command first")
        return
    
    # Run optimization
    try:
        optimizer = LightGBMOptunaOptimizer(
            n_trials=n_trials,
            cv_folds=cv_folds
        )
        
        # Load data
        data = pd.read_parquet(data_path)
        logger.info(f"Loaded data: {data.shape}")
        
        # Run optimization
        results = optimizer.optimize(data)
        
        # Save artifacts
        output_dir.mkdir(parents=True, exist_ok=True)
        optimizer.save_artifacts(str(output_dir))
        
        logger.success(f"Optimization completed!")
        logger.info(f"Best score: {results['best_score']:.4f}")
        logger.info(f"Best parameters saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise


@app.command()
def train_final(
    data_path: Path = PROCESSED_DATA_DIR / "historical_dataset.parquet",
    artifacts_dir: Path = MODELS_DIR / "optimization_results",
    output_dir: Path = MODELS_DIR / "final_model"
):
    """Train final model with best parameters from optimization."""
    logger.info("Training final LightGBM model with optimized parameters...")
    
    if not artifacts_dir.exists():
        logger.error(f"Optimization artifacts not found: {artifacts_dir}")
        logger.info("Run 'optimize' command first")
        return
    
    try:
        # Create scorer (this will load the optimized model)
        scorer = create_scorer_from_artifacts(str(artifacts_dir))
        
        # Copy artifacts to final model directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        for artifact_file in artifacts_dir.glob("*"):
            shutil.copy2(artifact_file, output_dir / artifact_file.name)
        
        logger.success(f"Final model ready at: {output_dir}")
        logger.info(f"Model info: {scorer.get_model_info()}")
        
    except Exception as e:
        logger.error(f"Final training failed: {e}")
        raise


@app.command()
def test_scorer(
    artifacts_dir: Path = MODELS_DIR / "optimization_results",
    test_symbol: str = "US30",
    test_timeframe: str = "H4"
):
    """Test the trained scorer with sample data."""
    logger.info("Testing the trained LightGBM scorer...")
    
    if not artifacts_dir.exists():
        logger.error(f"Artifacts not found: {artifacts_dir}")
        logger.info("Run 'optimize' command first")
        return
    
    try:
        # Create scorer
        scorer = create_scorer_from_artifacts(str(artifacts_dir))
        
        # Create sample data for testing
        import numpy as np
        from datetime import datetime, timedelta
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=100),
            end=datetime.now(),
            freq='4H'
        )
        
        # Generate sample OHLCV data
        np.random.seed(42)
        base_price = 35000  # US30 approximate price
        
        sample_data = pd.DataFrame({
            'open': base_price + np.random.normal(0, 100, len(dates)),
            'high': base_price + np.random.normal(50, 100, len(dates)),
            'low': base_price + np.random.normal(-50, 100, len(dates)),
            'close': base_price + np.random.normal(0, 100, len(dates)),
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        # Ensure OHLC logic
        sample_data['high'] = sample_data[['open', 'close']].max(axis=1) + np.abs(np.random.normal(0, 20, len(dates)))
        sample_data['low'] = sample_data[['open', 'close']].min(axis=1) - np.abs(np.random.normal(0, 20, len(dates)))
        
        # Test scoring
        result = scorer.score_setup(sample_data, test_symbol, test_timeframe)
        
        logger.success("Scorer test completed!")
        logger.info(f"Test result:")
        logger.info(f"  Symbol: {result.symbol}")
        logger.info(f"  Timeframe: {result.timeframe}")
        logger.info(f"  Score: {result.score:.4f}")
        logger.info(f"  Confidence: {result.confidence}")
        logger.info(f"  Signal Strength: {result.signal_strength:.4f}")
        logger.info(f"  Risk Assessment: {result.risk_assessment}")
        
    except Exception as e:
        logger.error(f"Scorer test failed: {e}")
        raise


@app.command()
def full_pipeline(
    n_trials: int = 100,
    start_year: int = 2020,
    end_year: int = 2024
):
    """Run the complete training pipeline from data generation to final model."""
    logger.info("Running complete LightGBM ML scoring pipeline...")
    
    try:
        # Step 1: Generate data
        logger.info("Step 1: Generating historical data...")
        generate_data(
            start_year=start_year,
            end_year=end_year
        )
        
        # Step 2: Optimize model
        logger.info("Step 2: Running optimization...")
        optimize(n_trials=n_trials)
        
        # Step 3: Prepare final model
        logger.info("Step 3: Preparing final model...")
        train_final()
        
        # Step 4: Test scorer
        logger.info("Step 4: Testing scorer...")
        test_scorer()
        
        logger.success("Complete pipeline finished successfully!")
        logger.info("Your LightGBM ML scoring system is ready to use!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.parquet",
    model_path: Path = MODELS_DIR / "optimization_results",
    n_trials: int = 100
):
    """Main training command - runs optimization if needed."""
    if not features_path.exists():
        logger.info("Features not found, running data generation...")
        generate_data()
        generate_features()
    
    if not model_path.exists():
        logger.info("Running optimization...")
        optimize(n_trials=n_trials)
    
    logger.success("LightGBM ML scoring system training complete!")


if __name__ == "__main__":
    app()
