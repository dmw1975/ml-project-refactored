#!/usr/bin/env python3
"""
Fixed XGBoost feature removal analysis for main.py integration.
This version works properly with all visualizations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedXGBoostAnalyzer:
    """Fixed XGBoost feature removal analyzer with working visualizations."""
    
    def __init__(self, excluded_feature: str = 'top_3_shareholder_percentage',
                 base_output_dir: Optional[Path] = None):
        """Initialize the analyzer."""
        self.excluded_feature = excluded_feature
        self.base_output_dir = base_output_dir or Path("outputs/feature_removal_experiment")
        self.models_data = {}
        
    def run_analysis(self):
        """Run the complete feature removal analysis."""
        logger.info("Starting XGBoost feature removal analysis...")
        
        # Create output directories
        self._create_output_directories()
        
        # Load and prepare data
        X_train_with, X_test_with, X_train_without, X_test_without, y_train, y_test = self._prepare_data()
        
        # Train models
        self._train_models(X_train_with, X_test_with, X_train_without, X_test_without, y_train, y_test)
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Generate metrics comparison
        self.generate_metrics_comparison()
        
        # Create summary report
        self._create_summary_report()
        
        logger.info(f"\nAnalysis complete! Results saved to {self.base_output_dir}")
        
    def _create_output_directories(self):
        """Create necessary output directories."""
        dirs = [
            self.base_output_dir / "models",
            self.base_output_dir / "visualizations" / "residuals",
            self.base_output_dir / "visualizations" / "shap",
            self.base_output_dir / "visualizations" / "comparisons",
            self.base_output_dir / "visualizations" / "metrics",
            self.base_output_dir / "metrics",
            self.base_output_dir / "logs"
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def _prepare_data(self):
        """Load and prepare data for analysis."""
        logger.info("Loading data...")
        
        # Load tree models dataset
        tree_data = pd.read_csv('data/processed/tree_models_dataset.csv')
        scores_df = pd.read_csv('data/raw/score.csv')
        
        # Set index and align
        if 'issuer_name' in tree_data.columns:
            tree_data = tree_data.set_index('issuer_name')
        scores_df = scores_df.set_index('issuer_name')
        
        common_indices = tree_data.index.intersection(scores_df.index)
        X = tree_data.loc[common_indices].copy()
        y = scores_df.loc[common_indices, 'esg_score'].copy()
        
        # Convert categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            X[col] = X[col].astype('category')
        
        # Load train/test split
        with open('data/processed/unified/train_test_split.pkl', 'rb') as f:
            split_data = pickle.load(f)
            train_idx = split_data['train_indices']
            test_idx = split_data['test_indices']
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Create datasets with and without feature
        X_train_with = X_train.copy()
        X_test_with = X_test.copy()
        X_train_without = X_train.drop(columns=[self.excluded_feature])
        X_test_without = X_test.drop(columns=[self.excluded_feature])
        
        logger.info(f"Data prepared. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        return X_train_with, X_test_with, X_train_without, X_test_without, y_train, y_test
        
    def _train_models(self, X_train_with, X_test_with, X_train_without, X_test_without, y_train, y_test):
        """Train XGBoost models with and without the feature."""
        logger.info("\nTraining models...")
        
        # Basic XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'enable_categorical': True,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': 0
        }
        
        # Train model WITH feature
        logger.info("Training model WITH feature...")
        model_with = xgb.XGBRegressor(**params)
        model_with.fit(X_train_with, y_train)
        y_pred_with = model_with.predict(X_test_with)
        
        # Calculate metrics
        rmse_with = np.sqrt(mean_squared_error(y_test, y_pred_with))
        r2_with = r2_score(y_test, y_pred_with)
        logger.info(f"  RMSE: {rmse_with:.4f}, R²: {r2_with:.4f}")
        
        # Save model data
        self.models_data['XGBoost_with_feature'] = {
            'model': model_with,
            'model_name': 'XGBoost_with_all_features',
            'X_train': X_train_with,
            'X_test': X_test_with,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred_with,
            'rmse': rmse_with,
            'r2': r2_with,
            'feature_names': list(X_train_with.columns)
        }
        
        # Train model WITHOUT feature
        logger.info("Training model WITHOUT feature...")
        model_without = xgb.XGBRegressor(**params)
        model_without.fit(X_train_without, y_train)
        y_pred_without = model_without.predict(X_test_without)
        
        # Calculate metrics
        rmse_without = np.sqrt(mean_squared_error(y_test, y_pred_without))
        r2_without = r2_score(y_test, y_pred_without)
        logger.info(f"  RMSE: {rmse_without:.4f}, R²: {r2_without:.4f}")
        
        # Save model data
        self.models_data['XGBoost_without_feature'] = {
            'model': model_without,
            'model_name': f'XGBoost_without_{self.excluded_feature}',
            'X_train': X_train_without,
            'X_test': X_test_without,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred_without,
            'rmse': rmse_without,
            'r2': r2_without,
            'feature_names': list(X_train_without.columns)
        }
        
        # Save models
        for name, data in self.models_data.items():
            with open(self.base_output_dir / "models" / f"{name}.pkl", 'wb') as f:
                pickle.dump(data, f)
                
        logger.info("Models trained and saved")
        
    def generate_visualizations(self):
        """Generate all visualizations."""
        logger.info("\nGenerating visualizations...")
        
        # Import visualization functions
        from src.visualization.viz_factory import create_residual_plot, create_model_comparison_plot, create_metrics_table
        from src.visualization.plots.shap_plots import create_all_shap_visualizations
        from src.visualization.core.interfaces import VisualizationConfig
        
        # 1. Generate residual plots
        logger.info("Creating residual plots...")
        for model_name, model_data in self.models_data.items():
            try:
                config = VisualizationConfig(
                    output_dir=self.base_output_dir / "visualizations" / "residuals"
                )
                create_residual_plot(model_data, config)
                logger.info(f"  ✓ Created residual plot for {model_name}")
            except Exception as e:
                logger.error(f"  ✗ Failed residual plot for {model_name}: {e}")
        
        # 2. Generate SHAP visualizations
        logger.info("Creating SHAP visualizations...")
        try:
            shap_paths = create_all_shap_visualizations(
                models=self.models_data,
                output_dir=self.base_output_dir / "visualizations" / "shap"
            )
            logger.info("  ✓ Created SHAP visualizations")
        except Exception as e:
            logger.error(f"  ✗ Failed SHAP visualizations: {e}")
        
        # 3. Generate model comparison plot
        logger.info("Creating model comparison plot...")
        try:
            config = VisualizationConfig(
                output_dir=self.base_output_dir / "visualizations" / "comparisons"
            )
            create_model_comparison_plot(
                models=list(self.models_data.values()),
                config=config
            )
            logger.info("  ✓ Created model comparison plot")
        except Exception as e:
            logger.error(f"  ✗ Failed model comparison: {e}")
        
        # 4. Generate metrics table
        logger.info("Creating metrics table...")
        try:
            config = VisualizationConfig(
                output_dir=self.base_output_dir / "visualizations" / "metrics"
            )
            create_metrics_table(
                models=list(self.models_data.values()),
                config=config
            )
            logger.info("  ✓ Created metrics table")
        except Exception as e:
            logger.error(f"  ✗ Failed metrics table: {e}")
            
    def generate_metrics_comparison(self):
        """Generate metrics comparison CSV."""
        logger.info("\nGenerating metrics comparison...")
        
        with_data = self.models_data['XGBoost_with_feature']
        without_data = self.models_data['XGBoost_without_feature']
        
        results = {
            'excluded_feature': self.excluded_feature,
            'rmse_with_feature': with_data['rmse'],
            'rmse_without_feature': without_data['rmse'],
            'rmse_change': without_data['rmse'] - with_data['rmse'],
            'rmse_change_pct': (without_data['rmse'] - with_data['rmse']) / with_data['rmse'] * 100,
            'r2_with_feature': with_data['r2'],
            'r2_without_feature': without_data['r2'],
            'r2_change': without_data['r2'] - with_data['r2']
        }
        
        # Save to CSV
        df = pd.DataFrame([results])
        df.to_csv(self.base_output_dir / "metrics" / "feature_removal_comparison.csv", index=False)
        
        # Print summary
        print("\n" + "="*60)
        print("FEATURE REMOVAL ANALYSIS SUMMARY")
        print("="*60)
        print(f"Removed feature: {self.excluded_feature}")
        print(f"\nModel Performance Comparison:")
        print(f"RMSE change: {results['rmse_change']:+.4f} ({results['rmse_change_pct']:+.2f}%)")
        print(f"R² change: {results['r2_change']:+.4f}")
        print("="*60)
        
    def _create_summary_report(self):
        """Create a summary report."""
        with_data = self.models_data['XGBoost_with_feature']
        without_data = self.models_data['XGBoost_without_feature']
        
        report = f"""# XGBoost Feature Removal Analysis Report

## Analysis Overview
- **Excluded Feature**: {self.excluded_feature}
- **Models Trained**: 2 (with and without feature)
- **Output Directory**: {self.base_output_dir}

## Results Summary

### Model Performance Impact
- RMSE changed by **{(without_data['rmse'] - with_data['rmse']) / with_data['rmse'] * 100:.2f}%** when feature removed
- R² changed by **{without_data['r2'] - with_data['r2']:.4f}** points

### Detailed Metrics
| Metric | With Feature | Without Feature | Change |
|--------|-------------|-----------------|---------|
| RMSE   | {with_data['rmse']:.4f} | {without_data['rmse']:.4f} | {without_data['rmse'] - with_data['rmse']:+.4f} |
| R²     | {with_data['r2']:.4f} | {without_data['r2']:.4f} | {without_data['r2'] - with_data['r2']:+.4f} |

### Visualizations Generated
1. **Residual Plots**: `visualizations/residuals/`
2. **SHAP Analysis**: `visualizations/shap/`
3. **Model Comparison**: `visualizations/comparisons/`
4. **Metrics Table**: `visualizations/metrics/`

### Key Files
- Model files: `models/`
- Metrics comparison: `metrics/feature_removal_comparison.csv`

---
Analysis completed: {datetime.now()}
"""
        
        with open(self.base_output_dir / "ANALYSIS_REPORT.md", 'w') as f:
            f.write(report)
            
        logger.info(f"Created summary report: {self.base_output_dir / 'ANALYSIS_REPORT.md'}")


# For use by main.py
IsolatedXGBoostAnalyzer = FixedXGBoostAnalyzer