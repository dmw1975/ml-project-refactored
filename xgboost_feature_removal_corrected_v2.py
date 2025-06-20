#!/usr/bin/env python3
"""
Corrected XGBoost feature removal analysis.
This version:
- Only trains the models we need (without features)
- Uses existing Optuna-optimized parameters
- Removes both raw and Yeo-Johnson versions of the feature
- Generates visualizations consistent with main pipeline
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import pickle
import json
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
import matplotlib.pyplot as plt

# Import configuration and visualization components
from src.config import settings
from src.config.hyperparameters import XGBOOST_PARAMS, get_optuna_params
from src.visualization.comprehensive import create_comprehensive_visualizations
from src.visualization.core.interfaces import VisualizationConfig
from src.visualization.core.style import setup_visualization_style
from src.visualization.core.registry import get_adapter_for_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CorrectedXGBoostFeatureRemoval:
    """Corrected XGBoost feature removal analyzer."""
    
    def __init__(self, 
                 excluded_features: List[str] = None,
                 base_output_dir: Optional[Path] = None):
        """
        Initialize the analyzer.
        
        Args:
            excluded_features: List of features to exclude. Default removes both versions
                             of top_3_shareholder_percentage
            base_output_dir: Output directory for results
        """
        if excluded_features is None:
            # Remove BOTH raw and transformed versions
            self.excluded_features = [
                'top_3_shareholder_percentage',
                'yeo_joh_top_3_shareholder_percentage'
            ]
        else:
            self.excluded_features = excluded_features
            
        self.base_output_dir = base_output_dir or Path("outputs/feature_removal_experiment_corrected")
        self.models_data = {}
        
        # Only test on the Optuna-optimized models
        self.target_models = {
            'XGBoost_Base_Random_categorical_optuna': 'Base_Random',
            'XGBoost_Yeo_Random_categorical_optuna': 'Yeo_Random'
        }
        
        # Setup visualization style
        self.style = setup_visualization_style('whitegrid')
        
    def run_analysis(self):
        """Run the complete feature removal analysis."""
        logger.info("Starting corrected XGBoost feature removal analysis...")
        logger.info(f"Features to remove: {self.excluded_features}")
        logger.info(f"Target models: {list(self.target_models.keys())}")
        
        # Create output directories
        self._create_output_directories()
        
        # Load existing XGBoost models
        existing_models = self._load_existing_models()
        
        # Process each target model
        all_results = []
        for model_name, dataset_key in self.target_models.items():
            if model_name not in existing_models:
                logger.warning(f"Model {model_name} not found in existing models. Skipping...")
                continue
                
            logger.info(f"\nProcessing {model_name}...")
            
            # Get existing model data and parameters
            existing_model_data = existing_models[model_name]
            
            # Load and prepare data for this specific dataset
            X_train_without, X_test_without, y_train, y_test = self._prepare_data(dataset_key)
            
            # Train model without features using existing parameters
            results = self._train_without_features(
                X_train_without, X_test_without, y_train, y_test,
                model_name, existing_model_data
            )
            all_results.append(results)
        
        # Generate comprehensive comparison
        self._generate_comprehensive_comparison(all_results)
        
        # Generate visualizations with main pipeline standards
        self._generate_visualizations()
        
        # Create detailed report
        self._create_detailed_report(all_results)
        
        logger.info(f"\nAnalysis complete! Results saved to {self.base_output_dir}")
        
    def _create_output_directories(self):
        """Create necessary output directories."""
        dirs = [
            self.base_output_dir / "models",
            self.base_output_dir / "visualizations" / "residuals",
            self.base_output_dir / "visualizations" / "shap",
            self.base_output_dir / "visualizations" / "comparisons",
            self.base_output_dir / "visualizations" / "performance",
            self.base_output_dir / "visualizations" / "metrics",
            self.base_output_dir / "metrics",
            self.base_output_dir / "logs"
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def _load_existing_models(self) -> Dict:
        """Load existing XGBoost models."""
        try:
            with open('outputs/models/xgboost_models.pkl', 'rb') as f:
                models = pickle.load(f)
                logger.info(f"Loaded {len(models)} existing XGBoost models")
                return models
        except Exception as e:
            logger.error(f"Could not load existing models: {e}")
            raise
            
    def _prepare_data(self, dataset_key: str) -> Tuple:
        """Load and prepare data for a specific dataset."""
        logger.info(f"Loading data for {dataset_key}...")
        
        # Load the unified tree models dataset
        tree_data = pd.read_csv('data/processed/unified/tree_models_unified.csv')
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
        
        # Load the specific train/test split for this dataset
        split_file = f'data/pkl/{dataset_key}_train_test_split.pkl'
        if not Path(split_file).exists():
            # Fall back to unified split
            split_file = 'data/processed/unified/train_test_split.pkl'
            logger.info(f"Using unified train/test split")
            
        with open(split_file, 'rb') as f:
            split_data = pickle.load(f)
            train_idx = split_data['train_indices']
            test_idx = split_data['test_indices']
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Remove specified features
        cols_to_drop = [col for col in self.excluded_features if col in X_train.columns]
        logger.info(f"Removing features: {cols_to_drop}")
        
        X_train_without = X_train.drop(columns=cols_to_drop)
        X_test_without = X_test.drop(columns=cols_to_drop)
        
        logger.info(f"Data prepared. Train shape: {X_train.shape} -> {X_train_without.shape}")
        
        return X_train_without, X_test_without, y_train, y_test
        
    def _train_without_features(self, X_train, X_test, y_train, y_test,
                               model_name: str, existing_model_data: Dict) -> Dict:
        """Train model without features using existing optimized parameters."""
        
        # Extract existing model parameters
        if 'best_params' in existing_model_data:
            params = existing_model_data['best_params'].copy()
            logger.info(f"Using existing Optuna-optimized parameters")
        elif 'model' in existing_model_data:
            try:
                params = existing_model_data['model'].get_params()
                logger.info(f"Using parameters from existing model")
            except:
                logger.warning(f"Could not extract parameters, using defaults")
                params = XGBOOST_PARAMS['basic'].copy()
        else:
            params = XGBOOST_PARAMS['basic'].copy()
            
        # Ensure required parameters are set
        params.update({
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'enable_categorical': True,
            'verbosity': 0
        })
        
        # Get existing model metrics for comparison
        # First check direct keys (these should be present)
        existing_metrics = {
            'RMSE': existing_model_data.get('RMSE'),
            'MAE': existing_model_data.get('MAE'),
            'R2': existing_model_data.get('R2'),
            'MSE': existing_model_data.get('MSE')
        }
        
        # If not found directly, check in metrics dict
        if not existing_metrics['RMSE'] and 'metrics' in existing_model_data:
            metrics_dict = existing_model_data['metrics']
            if 'test_rmse' in metrics_dict:
                existing_metrics = {
                    'RMSE': metrics_dict.get('test_rmse'),
                    'MAE': metrics_dict.get('test_mae'),
                    'R2': metrics_dict.get('test_r2'),
                    'MSE': metrics_dict.get('test_rmse', 0) ** 2 if metrics_dict.get('test_rmse') else None
                }
        
        # Train model WITHOUT features
        logger.info(f"Training {model_name} WITHOUT features...")
        model_without = xgb.XGBRegressor(**params)
        model_without.fit(X_train, y_train)
        
        # Predictions and metrics
        y_pred = model_without.predict(X_test)
        rmse_without = np.sqrt(mean_squared_error(y_test, y_pred))
        mae_without = mean_absolute_error(y_test, y_pred)
        r2_without = r2_score(y_test, y_pred)
        mse_without = mean_squared_error(y_test, y_pred)
        
        # Log comparison
        logger.info(f"  Original model - RMSE: {existing_metrics.get('RMSE', 'N/A')}, "
                   f"R²: {existing_metrics.get('R2', 'N/A')}")
        logger.info(f"  WITHOUT features - RMSE: {rmse_without:.4f}, R²: {r2_without:.4f}")
        
        # Calculate changes
        rmse_with = existing_metrics.get('RMSE', rmse_without)
        r2_with = existing_metrics.get('R2', r2_without)
        
        # Save model data for visualization
        model_name_without = f"{model_name}_without_features"
        
        self.models_data[model_name_without] = {
            'model': model_without,
            'model_name': model_name_without,
            'model_type': 'xgboost',
            'dataset': model_name.split('_')[1:3],  # Extract dataset name
            'has_features': False,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'metrics': {
                'RMSE': rmse_without,
                'MAE': mae_without,
                'R2': r2_without,
                'MSE': mse_without
            },
            'params': params,
            'feature_names': list(X_train.columns)
        }
        
        # Save model to disk
        with open(self.base_output_dir / 'models' / f'{model_name_without}.pkl', 'wb') as f:
            pickle.dump(self.models_data[model_name_without], f)
        
        return {
            'model_name': model_name,
            'rmse_with': rmse_with,
            'rmse_without': rmse_without,
            'mae_with': existing_metrics.get('MAE', mae_without),
            'mae_without': mae_without,
            'r2_with': r2_with,
            'r2_without': r2_without,
            'mse_with': existing_metrics.get('MSE', mse_without),
            'mse_without': mse_without,
            'rmse_change': rmse_without - rmse_with,
            'rmse_change_pct': ((rmse_without - rmse_with) / rmse_with) * 100 if rmse_with > 0 else 0,
            'r2_change': r2_without - r2_with,
            'mae_change': mae_without - existing_metrics.get('MAE', mae_without),
            'mse_change': mse_without - existing_metrics.get('MSE', mse_without)
        }
        
    def _generate_comprehensive_comparison(self, results: List[Dict]):
        """Generate comprehensive comparison CSV."""
        df = pd.DataFrame(results)
        
        # Add additional analysis columns
        df['excluded_features'] = ', '.join(self.excluded_features)
        df['timestamp'] = datetime.now()
        
        # Save to CSV
        output_path = self.base_output_dir / 'metrics' / 'feature_removal_comparison.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"Saved comparison to {output_path}")
        
    def _generate_visualizations(self):
        """Generate visualizations using main pipeline standards."""
        logger.info("\nGenerating visualizations with main pipeline standards...")
        
        # Create visualization config matching main pipeline
        viz_config = VisualizationConfig(
            output_dir=self.base_output_dir / 'visualizations',
            format='png',
            dpi=300,
            save=True,
            show=False,
            style='whitegrid',
            colors=self.style['colors'],
            figsize=(12, 8),
            title_fontsize=14,
            label_fontsize=12,
            annotation_fontsize=10
        )
        
        # Apply matplotlib defaults
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
        
        # Generate comprehensive visualizations for each model
        for model_name, model_data in self.models_data.items():
            logger.info(f"Creating visualizations for {model_name}...")
            
            try:
                # Use the comprehensive visualization function
                # Note: create_comprehensive_visualizations doesn't accept config parameter
                # It will use default settings from the models
                create_comprehensive_visualizations(
                    models={model_name: model_data},
                    visualization_dir=viz_config.output_dir
                )
            except Exception as e:
                logger.warning(f"Could not generate some visualizations for {model_name}: {e}")
                
        # Generate comparison plots
        self._generate_comparison_plots(viz_config)
                
    def _generate_comparison_plots(self, viz_config):
        """Generate comparison plots between with and without features."""
        logger.info("Generating comparison plots...")
        
        # Load existing model metrics for comparison
        try:
            with open('outputs/models/xgboost_models.pkl', 'rb') as f:
                existing_models = pickle.load(f)
                
            # Create comparison plot
            from src.visualization.plots.metrics import plot_feature_removal_comparison
            
            for model_name in self.target_models.keys():
                if model_name in existing_models:
                    existing_metrics = existing_models[model_name].get('metrics', {})
                    model_name_without = f"{model_name}_without_features"
                    
                    if model_name_without in self.models_data:
                        without_metrics = self.models_data[model_name_without]['metrics']
                        
                        # Generate comparison plot
                        fig = plot_feature_removal_comparison(
                            best_model_metrics=existing_metrics,
                            feature_removal_metrics=without_metrics,
                            config=viz_config
                        )
                        
                        # Save with specific name
                        output_path = self.base_output_dir / 'visualizations' / 'comparisons' / f'{model_name}_comparison.png'
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        fig.savefig(output_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        
                        logger.info(f"Saved comparison plot for {model_name}")
                        
        except Exception as e:
            logger.warning(f"Could not generate comparison plots: {e}")
            
    def _create_detailed_report(self, results: List[Dict]):
        """Create a detailed markdown report."""
        report = ["# Corrected XGBoost Feature Removal Analysis Report\n"]
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report.append("## Analysis Overview\n")
        report.append(f"- **Excluded Features**: {', '.join(self.excluded_features)}\n")
        report.append(f"- **Models Tested**: {', '.join(self.target_models.keys())}\n")
        report.append(f"- **Approach**: Compare existing Optuna-optimized models with feature-removed versions\n")
        report.append(f"- **Total Models Trained**: {len(results)} (without features only)\n")
        
        report.append("\n## Results Summary\n")
        
        for result in results:
            report.append(f"\n### {result['model_name']}\n")
            report.append("| Metric | With Features | Without Features | Change | Change % |\n")
            report.append("|--------|--------------|------------------|--------|----------|\n")
            report.append(f"| RMSE | {result['rmse_with']:.4f} | {result['rmse_without']:.4f} | "
                         f"{result['rmse_change']:.4f} | {result['rmse_change_pct']:.2f}% |\n")
            report.append(f"| MAE | {result['mae_with']:.4f} | {result['mae_without']:.4f} | "
                         f"{result['mae_change']:.4f} | "
                         f"{(result['mae_change']/result['mae_with']*100) if result['mae_with'] > 0 else 0:.2f}% |\n")
            report.append(f"| R² | {result['r2_with']:.4f} | {result['r2_without']:.4f} | "
                         f"{result['r2_change']:.4f} | - |\n")
            report.append(f"| MSE | {result['mse_with']:.4f} | {result['mse_without']:.4f} | "
                         f"{result['mse_change']:.4f} | "
                         f"{(result['mse_change']/result['mse_with']*100) if result['mse_with'] > 0 else 0:.2f}% |\n")
        
        report.append("\n## Key Findings\n")
        
        # Calculate average impacts
        df = pd.DataFrame(results)
        avg_rmse_change = df['rmse_change_pct'].mean()
        avg_r2_change = df['r2_change'].mean()
        
        report.append(f"- Average RMSE increase: {avg_rmse_change:.2f}%\n")
        report.append(f"- Average R² decrease: {-avg_r2_change:.4f}\n")
        
        if len(results) > 1:
            max_impact_idx = df['rmse_change_pct'].idxmax()
            min_impact_idx = df['rmse_change_pct'].idxmin()
            report.append(f"- Most impacted model: {df.loc[max_impact_idx, 'model_name']}\n")
            report.append(f"- Least impacted model: {df.loc[min_impact_idx, 'model_name']}\n")
        
        report.append("\n## Methodology\n")
        report.append("- Used existing Optuna-optimized model parameters\n")
        report.append("- Used the same train/test splits as original models\n")
        report.append("- Removed both raw and Yeo-Johnson transformed versions of features\n")
        report.append("- Compared against existing model metrics (no retraining)\n")
        report.append("- All visualizations use main pipeline styling standards\n")
        
        report.append("\n## Visualization Standards Applied\n")
        report.append("- Style: whitegrid (consistent with main pipeline)\n")
        report.append("- DPI: 300 (publication quality)\n")
        report.append("- Figure size: 12x8 inches (standard)\n")
        report.append("- Color palette: Main pipeline default colors\n")
        report.append("- Font sizes: Title=14pt, Labels=12pt, Annotations=10pt\n")
        
        # Save report
        report_path = self.base_output_dir / 'ANALYSIS_REPORT.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Saved detailed report to {report_path}")


def main():
    """Run the corrected analysis."""
    analyzer = CorrectedXGBoostFeatureRemoval()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()