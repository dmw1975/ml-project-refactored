"""Training pipeline for ML models."""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.pipelines.base import BasePipeline
from src.utils.io import check_all_existing_models, prompt_consolidated_retrain
from src.config import settings


class TrainingPipeline(BasePipeline):
    """Pipeline for training ML models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the training pipeline.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.trained_models = {}
        
    def run(
        self, 
        models: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        force_retrain: bool = False,
        use_one_hot: bool = False,
        **kwargs
    ):
        """
        Run the training pipeline.
        
        Args:
            models: List of model types to train (None = all)
            datasets: List of datasets to use (None = all)
            force_retrain: Force retraining even if models exist
            use_one_hot: Use one-hot encoding for tree models
            **kwargs: Additional arguments for specific models
        """
        self.start_timing()
        
        # Default to all models if none specified
        if models is None:
            models = ['linear_regression', 'elasticnet', 'xgboost', 'lightgbm', 'catboost']
            
        # Default to all datasets if none specified
        if datasets is None or 'all' in datasets:
            datasets = ['all']
            
        # Check for existing models
        should_retrain = force_retrain
        if not force_retrain:
            existing_models = check_all_existing_models(datasets=datasets)
            if existing_models:
                should_retrain = prompt_consolidated_retrain(existing_models)
        
        # Train each model type
        if 'linear_regression' in models and should_retrain:
            self._train_linear_regression()
            
        if 'elasticnet' in models and should_retrain:
            self._train_elasticnet(datasets, **kwargs)
            
        if 'xgboost' in models and should_retrain:
            self._train_xgboost(datasets, use_one_hot, **kwargs)
            
        if 'lightgbm' in models and should_retrain:
            self._train_lightgbm(datasets, use_one_hot, **kwargs)
            
        if 'catboost' in models and should_retrain:
            self._train_catboost(datasets, use_one_hot, **kwargs)
            
        self.report_timing()
        return self.trained_models
    
    def _train_linear_regression(self):
        """Train linear regression models."""
        print("\nTraining linear regression models...")
        with self.time_step("Linear Regression Training"):
            from src.models.linear_regression import train_all_models
            linear_models = train_all_models()
            self.trained_models['linear_regression'] = linear_models
            
    def _train_elasticnet(self, datasets: List[str], **kwargs):
        """Train ElasticNet models."""
        print("\nTraining ElasticNet models...")
        with self.time_step("ElasticNet Training"):
            from src.models.elastic_net import train_elasticnet_models
            
            # Extract ElasticNet-specific parameters
            use_optuna = not kwargs.get('elasticnet_grid', False)
            n_trials = kwargs.get('optimize_elasticnet', 100)
            
            if use_optuna:
                print(f"  🎯 Using Optuna optimization with {n_trials} trials")
            else:
                print("  📊 Using grid search optimization (legacy)")
                
            elastic_models = train_elasticnet_models(
                datasets=datasets,
                use_optuna=use_optuna,
                n_trials=n_trials
            )
            self.trained_models['elasticnet'] = elastic_models
            
    def _train_xgboost(self, datasets: List[str], use_one_hot: bool, **kwargs):
        """Train XGBoost models."""
        print("\nTraining XGBoost models...")
        with self.time_step("XGBoost Training"):
            if use_one_hot:
                print("  🔢 Using one-hot encoded XGBoost implementation (legacy mode)")
                from src.models.xgboost_categorical import train_xgboost_models
                n_trials = kwargs.get('optimize_xgboost', settings.XGBOOST_PARAMS.get('n_trials', 50))
                xgboost_models = train_xgboost_models(
                    datasets=datasets, 
                    n_trials=n_trials,
                    force_retune=kwargs.get('force_retune', False)
                )
            else:
                print("  🌳 Using native categorical XGBoost implementation (default)")
                from src.models.xgboost_categorical import train_xgboost_categorical_models
                xgboost_models = train_xgboost_categorical_models(datasets=datasets)
            self.trained_models['xgboost'] = xgboost_models
            
    def _train_lightgbm(self, datasets: List[str], use_one_hot: bool, **kwargs):
        """Train LightGBM models."""
        print("\nTraining LightGBM models...")
        with self.time_step("LightGBM Training"):
            if use_one_hot:
                print("  🔢 Using one-hot encoded LightGBM implementation (legacy mode)")
                from src.models.lightgbm_categorical import train_lightgbm_models
                n_trials = kwargs.get('optimize_lightgbm', settings.LIGHTGBM_PARAMS.get('n_trials', 50))
                lightgbm_models = train_lightgbm_models(datasets=datasets, n_trials=n_trials)
            else:
                print("  🌳 Using native categorical LightGBM implementation (default)")
                from src.models.lightgbm_categorical import train_lightgbm_categorical_models
                lightgbm_models = train_lightgbm_categorical_models(datasets=datasets)
            self.trained_models['lightgbm'] = lightgbm_models
            
    def _train_catboost(self, datasets: List[str], use_one_hot: bool, **kwargs):
        """Train CatBoost models."""
        print("\nTraining CatBoost models...")
        with self.time_step("CatBoost Training"):
            if use_one_hot:
                print("  🔢 Using one-hot encoded CatBoost implementation (legacy mode)")
                from src.models.catboost_categorical import train_catboost_models
                n_trials = kwargs.get('optimize_catboost', settings.CATBOOST_PARAMS.get('n_trials', 50))
                catboost_models = train_catboost_models(
                    datasets=datasets,
                    n_trials=n_trials,
                    force_retune=kwargs.get('force_retune', False)
                )
            else:
                print("  🌳 Using native categorical CatBoost implementation (default)")
                from src.models.catboost_categorical import train_catboost_categorical_models
                catboost_models = train_catboost_categorical_models(datasets=datasets)
            self.trained_models['catboost'] = catboost_models