"""Optuna optimization pipeline for LightGBM ML scoring system."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import pickle
import json
from pathlib import Path
from loguru import logger
import warnings

from lightbgm.features import create_optimizable_features, FeatureConfig, DynamicFeatureEngineer

warnings.filterwarnings('ignore')


class LightGBMOptunaOptimizer:
    """
    Integrated optimization pipeline that optimizes both feature engineering
    parameters and LightGBM hyperparameters simultaneously using Optuna.
    """
    
    def __init__(self, 
                 study_name: str = "lightgbm_trading_scorer",
                 direction: str = "maximize",
                 n_trials: int = 500,
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize the optimizer.
        
        Args:
            study_name: Name for the Optuna study
            direction: Optimization direction ('maximize' or 'minimize')
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.study_name = study_name
        self.direction = direction
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Initialize study
        sampler = TPESampler(seed=random_state)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=30)
        
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner
        )
        
        self.best_model = None
        self.best_features_config = None
        self.feature_importance = None
        
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        This function optimizes both feature engineering parameters
        and LightGBM hyperparameters simultaneously.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Score to optimize (AUC-ROC)
        """
        try:
            # 1. Generate features with optimized parameters
            features_df = create_optimizable_features(trial, self.raw_data)
            
            if features_df.empty or len(features_df) < 100:
                return 0.0
            
            # 2. Prepare data for modeling
            X, y = self._prepare_modeling_data(features_df)
            
            if len(X) < 50:  # Minimum samples for cross-validation
                return 0.0
            
            # 3. Suggest LightGBM hyperparameters
            lgb_params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'device': 'gpu',  # Use GPU for faster training
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'verbosity': -1,
                'seed': self.random_state,
                
                # Optimized hyperparameters
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 15),
                'max_bin': trial.suggest_int('max_bin', 200, 300),
            }
            
            # 4. Perform time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Create LightGBM datasets
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                # Train model
                model = lgb.train(
                    lgb_params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=1000,
                    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
                )
                
                # Predict and score
                y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
                auc_score = roc_auc_score(y_val, y_pred_proba)
                cv_scores.append(auc_score)
                
                # Report intermediate result for pruning
                trial.report(auc_score, fold)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Return mean cross-validation score
            mean_cv_score = np.mean(cv_scores)
            
            logger.info(f"Trial {trial.number}: CV AUC = {mean_cv_score:.4f}")
            
            return mean_cv_score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            return 0.0
    
    def _prepare_modeling_data(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for modeling.
        
        This function creates trading setup labels based on future price movements.
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            Tuple of (features, target)
        """
        # Sort by timestamp
        features_df = features_df.sort_values('timestamp')
        
        # Create target variable (setup success)
        # Look ahead N periods to determine if setup was successful
        lookahaid_periods = 10  # Customize based on trading timeframe
        
        # Calculate future returns
        features_df['future_return'] = features_df.groupby(['symbol', 'timeframe'])['close'].pct_change(lookahaid_periods).shift(-lookahaid_periods)
        
        # Define successful setup (positive return above threshold)
        return_threshold = 0.01  # 1% minimum return for success
        features_df['setup_success'] = (features_df['future_return'] > return_threshold).astype(int)
        
        # Remove rows with NaN target
        features_df = features_df.dropna(subset=['setup_success', 'future_return'])
        
        # Select feature columns (exclude metadata and target)
        feature_cols = [col for col in features_df.columns 
                       if col not in ['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume',
                                     'adj_close', 'future_return', 'setup_success']]
        
        X = features_df[feature_cols]
        y = features_df['setup_success']
        
        # Handle any remaining NaN values
        X = X.fillna(0)
        
        return X, y
    
    def optimize(self, raw_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the optimization process.
        
        Args:
            raw_data: Raw OHLCV data for optimization
            
        Returns:
            Dictionary with optimization results
        """
        self.raw_data = raw_data
        
        logger.info(f"Starting optimization with {self.n_trials} trials...")
        logger.info(f"Data shape: {raw_data.shape}")
        
        # Run optimization
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        # Get best parameters
        best_params = self.study.best_params
        
        logger.success(f"Optimization completed!")
        logger.info(f"Best score: {self.study.best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Train final model with best parameters
        self._train_final_model(best_params)
        
        return {
            'best_score': self.study.best_value,
            'best_params': best_params,
            'n_trials': len(self.study.trials),
            'study': self.study
        }
    
    def _train_final_model(self, best_params: Dict[str, Any]) -> None:
        """Train final model with best parameters on full dataset."""
        logger.info("Training final model with best parameters...")
        
        # Separate feature and model parameters
        feature_params = {k: v for k, v in best_params.items() 
                         if k.startswith(('fib_', 'macd_', 'rvi_', 'rsi_', 'atr_', 'volume_', 'divergence_'))}
        
        lgb_params = {k: v for k, v in best_params.items() 
                     if k not in feature_params}
        
        # Add fixed LightGBM parameters
        lgb_params.update({
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'verbosity': -1,
            'seed': self.random_state,
        })
        
        # Create final features config
        self.best_features_config = FeatureConfig(**feature_params)
        
        # Generate final features
        engineer = DynamicFeatureEngineer(self.best_features_config)
        features_df = engineer.generate_features(self.raw_data)
        
        # Prepare final modeling data
        X, y = self._prepare_modeling_data(features_df)
        
        # Train final model
        train_data = lgb.Dataset(X, label=y)
        
        self.best_model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=1000,
            callbacks=[lgb.log_evaluation(100)]
        )
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.best_model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        logger.success("Final model training completed!")
    
    def save_artifacts(self, output_dir: str) -> None:
        """Save optimization artifacts."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save best model
        model_path = output_path / "best_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Save best parameters
        params_path = output_path / "best_params.json"
        with open(params_path, 'w') as f:
            json.dump(self.study.best_params, f, indent=2)
        
        # Save feature config
        config_path = output_path / "feature_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.best_features_config.__dict__, f, indent=2)
        
        # Save feature importance
        importance_path = output_path / "feature_importance.csv"
        self.feature_importance.to_csv(importance_path, index=False)
        
        # Save study
        study_path = output_path / "optuna_study.pkl"
        with open(study_path, 'wb') as f:
            pickle.dump(self.study, f)
        
        logger.success(f"Artifacts saved to {output_path}")


def run_optimization(data_path: str, output_dir: str, n_trials: int = 500) -> Dict[str, Any]:
    """
    Run the complete optimization pipeline.
    
    Args:
        data_path: Path to the historical dataset
        output_dir: Directory to save optimization artifacts
        n_trials: Number of optimization trials
        
    Returns:
        Optimization results
    """
    # Load data
    logger.info(f"Loading data from {data_path}")
    data = pd.read_parquet(data_path)
    
    # Initialize optimizer
    optimizer = LightGBMOptunaOptimizer(n_trials=n_trials)
    
    # Run optimization
    results = optimizer.optimize(data)
    
    # Save artifacts
    optimizer.save_artifacts(output_dir)
    
    return results


if __name__ == "__main__":
    # Example usage
    data_path = "data/processed/historical_dataset.parquet"
    output_dir = "models/optimization_results"
    
    results = run_optimization(data_path, output_dir, n_trials=100)
    
    print(f"Optimization completed with best score: {results['best_score']:.4f}")