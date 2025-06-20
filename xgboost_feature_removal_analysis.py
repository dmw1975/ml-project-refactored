#!/usr/bin/env python3
"""
Isolated XGBoost Feature Removal Analysis
========================================

This script performs an isolated analysis of XGBoost model performance
with and without the 'top_3_shareholder_percentage' feature.

The analysis is completely isolated from the main pipeline and outputs
results to a separate directory structure.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
import shutil
import sys
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import necessary components
from src.data.data_categorical import load_tree_models_data, get_categorical_features
from src.models.xgboost_categorical import train_xgboost_categorical
from src.config import settings
from src.utils.io import ensure_dir
from src.visualization.core.interfaces import VisualizationConfig
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics."""
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred)
    }


class IsolatedXGBoostAnalyzer:
    """
    Performs isolated XGBoost feature removal analysis
    """
    
    def __init__(self, base_output_dir: str = "outputs/feature_removal_experiment"):
        """Initialize the analyzer with isolated output directory."""
        self.base_output_dir = Path(base_output_dir)
        self.datasets = ["Base_Random_categorical", "Yeo_Random_categorical"]
        self.excluded_feature = "top_3_shareholder_percentage"
        self.models_data = {}
        self.results = {}
        
        # Create isolated directory structure
        self._setup_directories()
        
    def _setup_directories(self):
        """Create isolated directory structure for experiment."""
        directories = [
            self.base_output_dir / "models",
            self.base_output_dir / "visualizations" / "residuals",
            self.base_output_dir / "visualizations" / "performance",
            self.base_output_dir / "visualizations" / "shap",
            self.base_output_dir / "visualizations" / "comparisons",
            self.base_output_dir / "metrics",
            self.base_output_dir / "logs"
        ]
        
        for directory in directories:
            ensure_dir(directory)
            
        logger.info(f"Created isolated experiment directories under {self.base_output_dir}")
    
    def load_and_prepare_data(self) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """Load data and prepare versions with and without the target feature."""
        logger.info("Loading tree models data...")
        
        # Load original data
        X, y = load_tree_models_data()
        
        # Create a copy for feature removal
        X_removed = X.copy()
        
        # Check if feature exists before removing
        if self.excluded_feature in X_removed.columns:
            X_removed = X_removed.drop(columns=[self.excluded_feature])
            logger.info(f"Removed feature '{self.excluded_feature}' from dataset")
        else:
            logger.warning(f"Feature '{self.excluded_feature}' not found in dataset")
        
        # Store both versions
        data_versions = {
            "original": (X, y),
            "feature_removed": (X_removed, y)
        }
        
        logger.info(f"Original features: {X.shape[1]}")
        logger.info(f"After removal: {X_removed.shape[1]}")
        
        return data_versions
    
    def prepare_dataset_features(self, X, y, dataset_name):
        """Prepare features based on dataset type."""
        from src.models.xgboost_categorical import (
            get_base_and_yeo_features_categorical,
            add_random_feature_categorical
        )
        
        # Get categorical columns
        categorical_cols = get_categorical_features()
        
        # Get all columns
        all_features = X.copy()
        
        # Determine which columns to use based on dataset name
        if "Base" in dataset_name:
            # Use base (non-transformed) quantitative features
            quantitative_cols = [col for col in all_features.columns 
                               if col not in categorical_cols and not col.startswith('yeo_joh_')]
            features_to_use = all_features[quantitative_cols + categorical_cols].copy()
        elif "Yeo" in dataset_name:
            # Use Yeo-transformed quantitative features
            yeo_cols = [col for col in all_features.columns 
                       if col.startswith('yeo_joh_')]
            features_to_use = all_features[yeo_cols + categorical_cols].copy()
        else:
            features_to_use = all_features.copy()
        
        # Add random feature if needed
        if "Random" in dataset_name:
            features_to_use = add_random_feature_categorical(features_to_use)
        
        return features_to_use
    
    def train_models(self, data_versions: Dict) -> Dict:
        """Train XGBoost models with and without the feature."""
        models = {}
        categorical_columns = get_categorical_features()
        
        for version_name, (X_all, y) in data_versions.items():
            logger.info(f"\nTraining models for {version_name} version...")
            
            # For each dataset type
            for dataset in self.datasets:
                logger.info(f"Training on {dataset}...")
                
                # Prepare features based on dataset type
                X = self.prepare_dataset_features(X_all, y, dataset)
                logger.info(f"Using {X.shape[1]} features for {dataset}")
                
                # Train using the enhanced categorical function
                # This returns a dict with both 'basic' and 'optuna' models
                result = train_xgboost_categorical(
                    X, y, dataset, categorical_columns,
                    test_size=0.2, random_state=42
                )
                
                # Extract both models from result
                for model_type in ['basic', 'optuna']:
                    if model_type in result:
                        model_name = f"XGBoost_{dataset}_{model_type}_{version_name}"
                        model_data = result[model_type]
                        
                        # Add model name to data
                        model_data['model_name'] = model_name
                        model_data['dataset'] = dataset
                        model_data['version'] = version_name
                        
                        models[model_name] = model_data
                        
                        # Save model
                        model_path = self.base_output_dir / "models" / f"{model_name}.pkl"
                        with open(model_path, 'wb') as f:
                            pickle.dump(model_data, f)
                        
                        logger.info(f"Saved {model_type} model for {dataset}")
                
                logger.info(f"Completed training for {dataset}")
        
        self.models_data = models
        return models
    
    def generate_visualizations(self):
        """Generate all visualizations using existing infrastructure."""
        logger.info("\nGenerating visualizations...")
        
        # Import visualization functions
        from src.visualization.viz_factory import (
            create_residual_plot, create_all_residual_plots,
            create_model_comparison_plot, create_metrics_table
        )
        from src.visualization.plots.shap_plots import create_all_shap_visualizations
        # Note: create_performance_plots doesn't exist in current codebase
        
        # Custom visualization config for isolated output
        viz_config = VisualizationConfig(
            output_dir=self.base_output_dir / "visualizations"
        )
        
        # 1. Generate residual plots
        logger.info("Creating residual plots...")
        try:
            # Try individual plots since we have specific models
            residual_dir = self.base_output_dir / "visualizations" / "residuals"
            residual_dir.mkdir(parents=True, exist_ok=True)
            
            for model_name, model_data in self.models_data.items():
                try:
                    # Update config for residuals
                    residual_config = VisualizationConfig(
                        output_dir=residual_dir
                    )
                    path = create_residual_plot(
                        model_data,
                        config=residual_config
                    )
                    logger.info(f"Created residual plot for {model_name}")
                except Exception as e2:
                    logger.error(f"Failed to create residual plot for {model_name}: {e2}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            logger.error(f"Failed to create residual plots: {e}")
            import traceback
            traceback.print_exc()
        
        # 2. Generate performance metrics plots
        # Note: create_performance_plots function doesn't exist in current codebase
        # This section is commented out until the function is implemented
        logger.info("Skipping performance metrics plots (function not available)...")
        # try:
        #     # Create performance plots
        #     create_performance_plots(
        #         models=list(self.models_data.values()),
        #         output_dir=self.base_output_dir / "visualizations" / "performance"
        #     )
        #     logger.info("Created performance metrics plots")
        # except Exception as e:
        #     logger.error(f"Failed to create performance metrics: {e}")
        
        # 3. Generate SHAP visualizations
        logger.info("Creating SHAP visualizations...")
        try:
            # Create all SHAP visualizations
            shap_paths = create_all_shap_visualizations(
                models=self.models_data,
                output_dir=self.base_output_dir / "visualizations" / "shap"
            )
            logger.info(f"Created SHAP visualizations")
        except Exception as e:
            logger.error(f"Failed to create SHAP visualizations: {e}")
            import traceback
            traceback.print_exc()
        
        # 4. Generate model comparison plot
        logger.info("Creating model comparison plot...")
        try:
            # Update config for comparison plot
            comparison_config = VisualizationConfig(
                output_dir=self.base_output_dir / "visualizations" / "comparisons"
            )
            comparison_path = create_model_comparison_plot(
                models=list(self.models_data.values()),
                config=comparison_config
            )
            logger.info("Created model comparison plot")
        except Exception as e:
            logger.error(f"Failed to create model comparison: {e}")
            import traceback
            traceback.print_exc()
        
        # 5. Generate metrics table
        logger.info("Creating metrics table...")
        try:
            # Update config for metrics table
            metrics_config = VisualizationConfig(
                output_dir=self.base_output_dir / "visualizations" / "metrics"
            )
            table_path = create_metrics_table(
                models=list(self.models_data.values()),
                config=metrics_config
            )
            logger.info("Created metrics table")
        except Exception as e:
            logger.error(f"Failed to create metrics table: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_metrics_comparison(self):
        """Generate metrics comparison between original and feature-removed models."""
        logger.info("\nGenerating metrics comparison...")
        
        metrics_comparison = []
        
        for dataset in self.datasets:
            for model_type in ["base", "optuna"]:
                # Get corresponding models
                original_name = f"XGBoost_{dataset}_{model_type}_original"
                removed_name = f"XGBoost_{dataset}_{model_type}_feature_removed"
                
                if original_name in self.models_data and removed_name in self.models_data:
                    original_metrics = self.models_data[original_name].get('metrics', {})
                    removed_metrics = self.models_data[removed_name].get('metrics', {})
                    
                    comparison = {
                        'dataset': dataset,
                        'model_type': model_type,
                        'original_rmse': original_metrics.get('RMSE', None),
                        'removed_rmse': removed_metrics.get('RMSE', None),
                        'rmse_change': None,
                        'original_r2': original_metrics.get('R2', None),
                        'removed_r2': removed_metrics.get('R2', None),
                        'r2_change': None
                    }
                    
                    # Calculate changes
                    if comparison['original_rmse'] and comparison['removed_rmse']:
                        comparison['rmse_change'] = (
                            comparison['removed_rmse'] - comparison['original_rmse']
                        )
                    
                    if comparison['original_r2'] and comparison['removed_r2']:
                        comparison['r2_change'] = (
                            comparison['removed_r2'] - comparison['original_r2']
                        )
                    
                    metrics_comparison.append(comparison)
        
        # Save comparison
        comparison_df = pd.DataFrame(metrics_comparison)
        comparison_path = self.base_output_dir / "metrics" / "feature_removal_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        logger.info(f"Saved metrics comparison to {comparison_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("FEATURE REMOVAL ANALYSIS SUMMARY")
        print("="*60)
        print(f"Removed feature: {self.excluded_feature}")
        print("\nModel Performance Comparison:")
        print(comparison_df.to_string(index=False))
        print("="*60)
        
        return comparison_df
    
    def run_analysis(self):
        """Execute the complete feature removal analysis."""
        logger.info("Starting XGBoost feature removal analysis...")
        
        try:
            # Load and prepare data
            data_versions = self.load_and_prepare_data()
            
            # Train models
            self.train_models(data_versions)
            
            # Generate visualizations
            self.generate_visualizations()
            
            # Generate metrics comparison
            self.generate_metrics_comparison()
            
            logger.info(f"\nAnalysis complete! Results saved to {self.base_output_dir}")
            
            # Create summary report
            self._create_summary_report()
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def _create_summary_report(self):
        """Create a summary report of the analysis."""
        report_path = self.base_output_dir / "ANALYSIS_REPORT.md"
        
        report_content = f"""# XGBoost Feature Removal Analysis Report

## Analysis Overview
- **Excluded Feature**: {self.excluded_feature}
- **Datasets Analyzed**: {', '.join(self.datasets)}
- **Models Trained**: {len(self.models_data)}
- **Output Directory**: {self.base_output_dir}

## Results Summary

### Model Performance Impact
See `metrics/feature_removal_comparison.csv` for detailed metrics.

### Visualizations Generated
1. **Residual Plots**: `visualizations/residuals/`
2. **Performance Metrics**: `visualizations/performance/`
3. **SHAP Analysis**: `visualizations/shap/`
4. **Model Comparisons**: `visualizations/comparisons/`

## Key Findings
The analysis compares XGBoost model performance with and without the '{self.excluded_feature}' feature
across different datasets and optimization strategies.

## Files Generated
- Model files: `models/`
- Metrics comparison: `metrics/feature_removal_comparison.csv`
- Visualizations: `visualizations/`

---
Analysis completed: {pd.Timestamp.now()}
"""
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Created summary report: {report_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run isolated XGBoost feature removal analysis"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="outputs/feature_removal_experiment",
        help='Output directory for analysis results'
    )
    parser.add_argument(
        '--feature',
        type=str,
        default="top_3_shareholder_percentage",
        help='Feature to exclude from analysis'
    )
    
    args = parser.parse_args()
    
    # Create and run analyzer
    analyzer = IsolatedXGBoostAnalyzer(base_output_dir=args.output_dir)
    if args.feature != "top_3_shareholder_percentage":
        analyzer.excluded_feature = args.feature
    
    analyzer.run_analysis()


if __name__ == "__main__":
    main()