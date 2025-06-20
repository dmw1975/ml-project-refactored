#!/usr/bin/env python3
"""
Proper XGBoost feature removal analysis with Optuna optimization.
This version:
- Uses categorical dataset (not one-hot encoded)
- Removes only top_3_shareholder_percentage feature
- Runs NEW Optuna optimization on the feature-removed dataset
- Compares properly optimized models
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
import optuna
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
from datetime import datetime
import matplotlib.pyplot as plt
import shap

# Import configuration and optimization settings
from src.config import settings
from src.config.hyperparameters import XGBOOST_PARAMS

# Import visualization modules
from src.visualization.plots.shap_plots import (
    SHAPVisualizer, create_shap_visualizations, 
    identify_categorical_features, create_model_comparison_shap_plot
)
from src.visualization.plots.residuals import ResidualPlot, plot_residuals
from src.visualization.core.registry import get_adapter_for_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProperXGBoostFeatureRemoval:
    """Proper XGBoost feature removal analyzer with Optuna optimization."""
    
    def __init__(self, 
                 excluded_features: Optional[List[str]] = None,
                 n_trials: int = 100,
                 base_output_dir: Optional[Path] = None):
        """
        Initialize the analyzer.
        
        Args:
            excluded_features: List of features to exclude
            n_trials: Number of Optuna trials
            base_output_dir: Output directory for results
        """
        # Default features to exclude
        if excluded_features is None:
            self.excluded_features = [
                'top_1_shareholder_percentage',
                'top_2_shareholder_percentage', 
                'top_3_shareholder_percentage',
                'random_feature'
            ]
        else:
            self.excluded_features = excluded_features
        self.n_trials = n_trials
        # Use absolute path from script location
        project_root = Path(__file__).parent
        self.base_output_dir = base_output_dir or (project_root / "outputs" / "feature_removal")
        self.results = {}
        
    def run_analysis(self):
        """Run the complete feature removal analysis with proper optimization."""
        logger.info("Starting PROPER XGBoost feature removal analysis...")
        logger.info(f"Features to remove: {', '.join(self.excluded_features)}")
        logger.info(f"Number of features to remove: {len(self.excluded_features)}")
        logger.info(f"Optuna trials: {self.n_trials}")
        
        # Create output directories
        self._create_output_directories()
        
        # Step 1: Load categorical datasets (both with and without the feature)
        datasets = self._load_datasets()
        
        # Step 2: Run analysis for each dataset type
        for dataset_name in ['Base_Random', 'Yeo_Random']:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {dataset_name} dataset...")
            logger.info(f"{'='*60}")
            
            # Get data with and without the feature
            data_with = datasets[f'{dataset_name}_with']
            data_without = datasets[f'{dataset_name}_without']
            
            # Step 3: Load existing optimized model WITH feature
            logger.info(f"\nLoading existing optimized model for {dataset_name} WITH feature...")
            results_with = self._load_existing_optimized_model(dataset_name)
            
            if results_with is None:
                logger.warning(f"Could not load existing model, will optimize {dataset_name} WITH feature...")
                results_with = self._optimize_and_train(
                    data_with['X_train'], data_with['X_test'],
                    data_with['y_train'], data_with['y_test'],
                    f"{dataset_name}_with_feature"
                )
            else:
                # Add the test data to results for visualization
                results_with['X_test'] = data_with['X_test']
                results_with['y_test'] = data_with['y_test']
                results_with['X_train'] = data_with['X_train']
                results_with['y_train'] = data_with['y_train']
                if results_with['model'] is not None:
                    results_with['y_pred'] = results_with['model'].predict(data_with['X_test'])
                else:
                    # If model not available, skip predictions
                    results_with['y_pred'] = None
                    logger.warning(f"Model object not available for {dataset_name}, skipping predictions")
                results_with['n_features'] = data_with['X_train'].shape[1]
                results_with['feature_names'] = list(data_with['X_train'].columns)
                results_with['model_type'] = 'xgboost'
                results_with['categorical_features'] = data_with['X_train'].select_dtypes(include=['category']).columns.tolist()
                results_with['data_info'] = {
                    'n_train': len(data_with['X_train']),
                    'n_test': len(data_with['X_test'])
                }
                results_with['cv_folds'] = 5  # Standard CV folds
            
            # Step 4: Run NEW Optuna optimization ONLY for WITHOUT feature
            logger.info(f"\nOptimizing {dataset_name} WITHOUT feature...")
            results_without = self._optimize_and_train(
                data_without['X_train'], data_without['X_test'],
                data_without['y_train'], data_without['y_test'],
                f"{dataset_name}_without_feature"
            )
            
            # Store results for comparison
            self.results[dataset_name] = {
                'with_feature': results_with,
                'without_feature': results_without,
                'comparison': self._calculate_comparison(results_with, results_without)
            }
            
        # Step 4: Generate comprehensive report and visualizations
        self._generate_report()
        self._generate_visualizations()
        
        # Step 5: Generate SHAP visualizations for all models
        logger.info("\nGenerating SHAP visualizations...")
        self._generate_shap_visualizations()
        
        # Step 6: Generate residual plots for all models
        logger.info("\nGenerating residual plots...")
        self._generate_residual_plots()
        
        logger.info(f"\nAnalysis complete! Results saved to {self.base_output_dir}")
        
    def _create_output_directories(self):
        """Create necessary output directories."""
        dirs = [
            self.base_output_dir / "models",
            self.base_output_dir / "optuna_studies",
            self.base_output_dir / "visualization",
            self.base_output_dir / "visualization" / "shap",
            self.base_output_dir / "visualization" / "residual",
            self.base_output_dir / "reports"
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def _load_datasets(self) -> Dict[str, Dict]:
        """Load categorical datasets with and without the feature."""
        # Import data loading functions from data_categorical module
        from src.data.data_categorical import load_tree_models_data, get_categorical_features
        from src.data.data_categorical import get_base_and_yeo_features_categorical, add_random_feature_categorical
        
        # Load the base tree models data
        features, target = load_tree_models_data()
        categorical_columns = get_categorical_features()
        
        # Get base and yeo feature sets
        base_features_df, yeo_features_df = get_base_and_yeo_features_categorical()
        
        results = {}
        
        # Process Base_Random dataset
        if 'Base_Random' in ['Base_Random', 'Yeo_Random']:
            # Add random feature to base features
            base_random_data = add_random_feature_categorical(base_features_df)
            results.update(self._create_dataset_splits('Base_Random', base_random_data, target, categorical_columns))
        
        # Process Yeo_Random dataset  
        if 'Yeo_Random' in ['Base_Random', 'Yeo_Random']:
            # Add random feature to yeo features
            yeo_random_data = add_random_feature_categorical(yeo_features_df)
            results.update(self._create_dataset_splits('Yeo_Random', yeo_random_data, target, categorical_columns))
                
        return results
    
    def _create_dataset_splits(self, dataset_name: str, X: pd.DataFrame, y: pd.Series, categorical_columns: List[str]) -> Dict:
        """Create train/test splits with and without the feature."""
        try:
            # Ensure X and y have the same index alignment
            common_indices = X.index.intersection(y.index)
            X = X.loc[common_indices].copy()
            y = y.loc[common_indices].copy()
            
            # No feature exclusion here - we want to analyze the impact of removing top_3_shareholder_percentage
            # The script will create WITH and WITHOUT versions later
            
            # Handle NaN values - following xgboost_categorical.py approach
            if X.isnull().any().any() or y.isnull().any():
                logger.info(f"Warning: Found NaN values. Dropping rows with NaN...")
                # Create a mask for rows without NaN
                mask = ~(X.isnull().any(axis=1) | y.isnull())
                X = X[mask].copy()
                y = y[mask].copy()
                logger.info(f"After dropping NaN: {X.shape[0]} samples remaining")
            
            logger.info(f"{dataset_name} data shape: {X.shape}")
            logger.info(f"Categorical columns: {categorical_columns}")
            logger.info(f"All features: {list(X.columns)[:10]}...")  # Show first 10 features
            # Log which features we'll be removing
            for feature in self.excluded_features:
                if dataset_name == 'Yeo_Random' and feature != 'random_feature' and not feature.startswith('yeo_joh_'):
                    yeo_feature = f'yeo_joh_{feature}'
                    logger.info(f"Feature '{yeo_feature}' exists: {yeo_feature in X.columns}")
                else:
                    logger.info(f"Feature '{feature}' exists: {feature in X.columns}")
            
            # Create train/test split (using stratification on sector)
            from sklearn.model_selection import train_test_split
            
            # For stratification, find a categorical column without NaN values
            stratify_col = None
            for col in categorical_columns:
                if col in X.columns and X[col].notna().all():
                    stratify_col = X[col]
                    logger.info(f"Using {col} for stratification")
                    break
            
            if stratify_col is None:
                # Use target bins for stratification
                stratify_col = pd.qcut(y, q=min(10, len(y)//10), labels=False, duplicates='drop')
                logger.info("Using target quantiles for stratification")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, 
                stratify=stratify_col
            )
            
            # Create with/without versions
            results = {}
            
            # With feature
            results[f'{dataset_name}_with'] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
            
            # Without features - handle both regular and yeo versions
            features_to_remove = []
            for feature in self.excluded_features:
                if dataset_name == 'Yeo_Random' and feature != 'random_feature' and not feature.startswith('yeo_joh_'):
                    # For Yeo dataset, look for yeo version of the feature
                    yeo_feature = f'yeo_joh_{feature}'
                    if yeo_feature in X_train.columns:
                        features_to_remove.append(yeo_feature)
                    else:
                        logger.warning(f"Feature '{yeo_feature}' not found in {dataset_name}")
                else:
                    if feature in X_train.columns:
                        features_to_remove.append(feature)
                    else:
                        logger.warning(f"Feature '{feature}' not found in {dataset_name}")
            
            if features_to_remove:
                X_train_without = X_train.drop(columns=features_to_remove)
                X_test_without = X_test.drop(columns=features_to_remove)
                logger.info(f"Removed {len(features_to_remove)} features for comparison: {', '.join(features_to_remove)}")
            else:
                X_train_without = X_train
                X_test_without = X_test
                logger.warning(f"No features removed from {dataset_name}")
                
            results[f'{dataset_name}_without'] = {
                'X_train': X_train_without,
                'X_test': X_test_without,
                'y_train': y_train,
                'y_test': y_test
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to create dataset splits: {e}")
            raise
    
    def _optimize_and_train(self, X_train, X_test, y_train, y_test, model_name: str) -> Dict:
        """Run Optuna optimization and train the final model."""
        
        # Create Optuna objective function
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
                'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'tree_method': 'hist',
                'enable_categorical': True,
                'verbosity': 0,
                'random_state': 42
            }
            
            # Perform cross-validation manually to avoid sklearn compatibility issues
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in cv.split(X_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model = xgb.XGBRegressor(**params)
                model.fit(X_fold_train, y_fold_train)
                
                y_pred = model.predict(X_fold_val)
                mse = mean_squared_error(y_fold_val, y_pred)
                scores.append(mse)
            
            return np.mean(scores)  # Return MSE (lower is better)
        
        # Run Optuna optimization
        logger.info(f"Starting Optuna optimization for {model_name}...")
        study = optuna.create_study(
            direction='minimize',
            study_name=model_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best MSE: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        # Train final model with best parameters
        best_params = study.best_params.copy()
        best_params.update({
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'enable_categorical': True,
            'verbosity': 0,
            'random_state': 42
        })
        
        final_model = xgb.XGBRegressor(**best_params)
        final_model.fit(X_train, y_train)
        
        # Make predictions and calculate metrics
        y_pred = final_model.predict(X_test)
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred)
        }
        
        logger.info(f"Final model performance - RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R2']:.4f}")
        
        # Save everything
        results = {
            'model': final_model,
            'study': study,
            'best_params': best_params,
            'metrics': metrics,
            'model_name': model_name,
            'n_features': X_train.shape[1],
            'feature_names': list(X_train.columns),
            'y_test': y_test,
            'y_pred': y_pred,
            'X_test': X_test,  # Needed for SHAP analysis
            'X_train': X_train,  # Needed for residual analysis
            'y_train': y_train,  # Needed for residual analysis
            'model_type': 'xgboost',  # For SHAP adapter
            'categorical_features': X_train.select_dtypes(include=['category']).columns.tolist(),
            'data_info': {
                'n_train': len(X_train),
                'n_test': len(X_test)
            },
            'cv_folds': 5  # Number of CV folds used in optimization
        }
        
        # Save model and study
        model_path = self.base_output_dir / 'models' / f'{model_name}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(results, f)
            
        study_path = self.base_output_dir / 'optuna_studies' / f'{model_name}_study.pkl'
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
        
        return results
    
    def _calculate_comparison(self, results_with: Dict, results_without: Dict) -> Dict:
        """Calculate comparison metrics between models with and without the feature."""
        comparison = {}
        
        for metric in ['RMSE', 'MAE', 'R2', 'MSE']:
            with_val = results_with['metrics'][metric]
            without_val = results_without['metrics'][metric]
            
            change = without_val - with_val
            pct_change = (change / with_val * 100) if with_val != 0 else 0
            
            comparison[metric] = {
                'with_feature': with_val,
                'without_feature': without_val,
                'change': change,
                'pct_change': pct_change
            }
        
        # Feature importance comparison
        comparison['feature_importance'] = {
            'n_features_with': results_with['n_features'],
            'n_features_without': results_without['n_features'],
            'removed_features': self.excluded_features,
            'n_removed': len(self.excluded_features)
        }
        
        return comparison
    
    def _save_feature_removal_metrics(self, results_dict: Dict[str, Any], output_path: Path) -> Path:
        """
        Save feature removal analysis results in the same format as model_metrics_comparison.csv
        
        Args:
            results_dict: Dictionary containing analysis results
            output_path: Path to save the CSV file
            
        Returns:
            Path to saved CSV file
        """
        metrics_data = []
        
        # Process each dataset result
        for dataset_name, results in results_dict.items():
            # Create entries for both "with feature" and "without feature" models
            for model_type in ['with_feature', 'without_feature']:
                model_data = results[model_type]
                metrics = model_data['metrics']
                
                # Determine model name
                if model_type == 'with_feature':
                    model_name = f"XGBoost_{dataset_name}_categorical_optuna_original"
                else:
                    model_name = f"XGBoost_{dataset_name}_categorical_optuna_removed_{len(self.excluded_features)}_features"
                
                # Get training/testing samples from the data
                data_info = model_data.get('data_info', {})
                n_train = data_info.get('n_train', 1761)  # Default values from the CSV
                n_test = data_info.get('n_test', 441)
                
                # Get feature counts
                n_features = model_data.get('n_features', 33 if 'Random' in dataset_name else 32)
                if model_type == 'without_feature':
                    n_features -= len(self.excluded_features)
                
                # For tree models, categorical features are always 7
                n_categorical = 7
                n_numerical = n_features - n_categorical
                
                # Get CV information
                cv_folds = model_data.get('cv_folds', 5)
                
                # Create metric row
                metric_row = {
                    'Model Name': model_name,
                    'RMSE': metrics['RMSE'],
                    'MAE': metrics['MAE'],
                    'R2': metrics['R2'],
                    'MSE': metrics['MSE'],
                    'Number of CV Folds': cv_folds,
                    'Number of Testing Samples': n_test,
                    'Number of Training Samples': n_train,
                    'Number of Quantitative Features': n_numerical,
                    'Number of Qualitative Features': n_categorical,
                    # Additional columns for feature removal analysis
                    'Features Removed': ', '.join(self.excluded_features) if model_type == 'without_feature' else 'None',
                    'Number of Features Removed': len(self.excluded_features) if model_type == 'without_feature' else 0,
                    'Model Type': model_type.replace('_', ' ').title(),
                    'Dataset': dataset_name
                }
                
                metrics_data.append(metric_row)
        
        # Create DataFrame
        metrics_df = pd.DataFrame(metrics_data)
        
        # Sort by dataset and model type
        metrics_df = metrics_df.sort_values(['Dataset', 'Model Type'])
        
        # Save to CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(output_path, index=False)
        
        logger.info(f"Saved feature removal metrics to: {output_path}")
        return output_path
    
    def _generate_report(self):
        """Generate comprehensive analysis report."""
        # Load baseline models for comparison
        baseline_models = self._load_best_baseline_models()
        
        report = ["# Proper XGBoost Feature Removal Analysis Report\n"]
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report.append("## Analysis Overview\n")
        report.append(f"- **Excluded Features**: {', '.join(self.excluded_features)}\n")
        report.append(f"- **Number of Features Removed**: {len(self.excluded_features)}\n")
        report.append(f"- **Optuna Trials**: {self.n_trials} trials for feature-removed models\n")
        report.append(f"- **Approach**: Using existing optimized models WITH features, new optimization for WITHOUT features\n")
        report.append("- **Dataset Type**: Categorical (native XGBoost categorical handling)\n\n")
        
        report.append("## Results Summary\n")
        
        for dataset_name, results in self.results.items():
            report.append(f"\n### {dataset_name} Dataset\n")
            
            comparison = results['comparison']
            
            # Metrics table with baseline comparison
            if baseline_models:
                # Find corresponding baseline
                baseline_key = 'Base_baseline' if 'Base' in dataset_name else 'Yeo_baseline'
                baseline = baseline_models.get(baseline_key, {})
                
                report.append("| Metric | With Feature | Without Feature | Best Baseline | Change | Change % |\n")
                report.append("|--------|--------------|-----------------|---------------|--------|----------|\n")
                
                for metric in ['RMSE', 'MAE', 'R2', 'MSE']:
                    comp = comparison[metric]
                    baseline_val = baseline.get(metric, 'N/A')
                    baseline_str = f"{baseline_val:.4f}" if isinstance(baseline_val, (int, float)) else baseline_val
                    
                    report.append(f"| {metric} | {comp['with_feature']:.4f} | "
                                f"{comp['without_feature']:.4f} | "
                                f"{baseline_str} | "
                                f"{comp['change']:+.4f} | "
                                f"{comp['pct_change']:+.2f}% |\n")
                    
                if baseline:
                    report.append(f"\n*Best baseline model: {baseline['model_name']}*\n")
            else:
                # Original table without baseline
                report.append("| Metric | With Feature | Without Feature | Change | Change % |\n")
                report.append("|--------|--------------|-----------------|--------|----------|\n")
                
                for metric in ['RMSE', 'MAE', 'R2', 'MSE']:
                    comp = comparison[metric]
                    report.append(f"| {metric} | {comp['with_feature']:.4f} | "
                                f"{comp['without_feature']:.4f} | "
                                f"{comp['change']:+.4f} | "
                                f"{comp['pct_change']:+.2f}% |\n")
            
            # Model source information
            with_source = results['with_feature'].get('source', 'optimized')
            if with_source == 'existing_optimized':
                report.append(f"\n*Note: WITH feature model loaded from existing optimized model*\n")
            elif with_source == 'baseline_csv':
                report.append(f"\n*Note: WITH feature metrics loaded from baseline_comparison.csv (model object not available)*\n")
            
            # Hyperparameter comparison
            report.append(f"\n#### Optimized Hyperparameters - {dataset_name}\n")
            report.append("| Parameter | With Feature | Without Feature |\n")
            report.append("|-----------|--------------|------------------|\n")
            
            params_with = results['with_feature']['best_params']
            params_without = results['without_feature']['best_params']
            
            for param in sorted(set(params_with.keys()) | set(params_without.keys())):
                if param not in ['objective', 'eval_metric', 'tree_method', 'enable_categorical', 'verbosity', 'random_state']:
                    val_with = params_with.get(param, 'N/A')
                    val_without = params_without.get(param, 'N/A')
                    
                    if isinstance(val_with, float):
                        val_with = f"{val_with:.6f}"
                    if isinstance(val_without, float):
                        val_without = f"{val_without:.6f}"
                        
                    report.append(f"| {param} | {val_with} | {val_without} |\n")
        
        report.append("\n## Key Findings\n")
        
        # Calculate average impact
        avg_rmse_change = np.mean([self.results[ds]['comparison']['RMSE']['pct_change'] 
                                   for ds in self.results])
        avg_r2_change = np.mean([self.results[ds]['comparison']['R2']['change'] 
                                 for ds in self.results])
        
        report.append(f"- Average RMSE change: {avg_rmse_change:+.2f}%\n")
        report.append(f"- Average R² change: {avg_r2_change:+.4f}\n")
        
        if avg_rmse_change > 0:
            report.append(f"- **Conclusion**: Removing these {len(self.excluded_features)} features degrades model performance\n")
        else:
            report.append(f"- **Conclusion**: Removing these {len(self.excluded_features)} features improves model performance\n")
        
        report.append("\n## Methodology\n")
        report.append("1. Used categorical datasets with native XGBoost categorical handling\n")
        report.append("2. Loaded existing optimized models WITH features from repository\n")
        report.append("3. Removed feature from training and test sets\n")
        report.append("4. Ran NEW Optuna optimization ONLY for datasets WITHOUT features\n")
        report.append("5. Compared existing optimized models (with) against newly optimized models (without)\n")
        report.append("6. Each optimization used 5-fold cross-validation\n")
        
        # Save report
        report_path = self.base_output_dir / 'reports' / 'ANALYSIS_REPORT.md'
        with open(report_path, 'w') as f:
            f.write(''.join(report))
        
        logger.info(f"Report saved to {report_path}")
        
        # Save CSV metrics with the same format as model_metrics_comparison.csv
        csv_path = self.base_output_dir / "visualization" / "xgboost_feature_removal_metrics_analysis.csv"
        self._save_feature_removal_metrics(self.results, csv_path)
    
    def _load_existing_optimized_model(self, dataset_name: str) -> Optional[Dict]:
        """Load existing optimized XGBoost model from the repository."""
        try:
            # Map dataset names to model names in the repository
            model_mapping = {
                'Base_Random': 'XGBoost_Base_Random_categorical_optuna',
                'Yeo_Random': 'XGBoost_Yeo_Random_categorical_optuna'
            }
            
            model_name = model_mapping.get(dataset_name)
            if not model_name:
                logger.warning(f"No model mapping found for {dataset_name}")
                return None
            
            # Try different possible locations for the model
            possible_paths = [
                Path('outputs/models/xgboost_models.pkl'),  # This is the main location
                Path(f'outputs/models/{model_name}.pkl'),
                Path(f'outputs/models/tree_models/{model_name}.pkl')
            ]
            
            for model_path in possible_paths:
                if model_path.exists():
                    logger.info(f"Loading existing model from {model_path}")
                    
                    with open(model_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # If it's a collection of models, extract the specific one
                    if isinstance(data, dict) and model_name in data:
                        model_data = data[model_name]
                    else:
                        model_data = data
                    
                    # Ensure we have the required fields
                    if 'model' in model_data and 'metrics' in model_data:
                        # Extract metrics - check different possible locations
                        metrics = model_data.get('metrics', {})
                        
                        # Try to get metrics from different possible keys
                        rmse = metrics.get('RMSE') or metrics.get('test_rmse') or metrics.get('rmse')
                        mae = metrics.get('MAE') or metrics.get('test_mae') or metrics.get('mae')
                        r2 = metrics.get('R2') or metrics.get('test_r2') or metrics.get('r2')
                        mse = metrics.get('MSE') or metrics.get('test_mse') or (rmse**2 if rmse else None)
                        
                        # Get hyperparameters
                        best_params = model_data.get('best_params') or model_data.get('params') or {}
                        
                        result = {
                            'model': model_data['model'],
                            'metrics': {
                                'RMSE': rmse,
                                'MAE': mae,
                                'R2': r2,
                                'MSE': mse
                            },
                            'best_params': best_params,
                            'model_name': model_name,
                            'source': 'existing_optimized'
                        }
                        
                        logger.info(f"Successfully loaded {model_name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
                        return result
                    
            # If not found in files, try loading from baseline comparison
            baseline_df = pd.read_csv('outputs/metrics/baseline_comparison.csv')
            model_row = baseline_df[baseline_df['Model'] == model_name]
            
            if not model_row.empty:
                row = model_row.iloc[0]
                logger.info(f"Loading metrics from baseline_comparison.csv for {model_name}")
                
                return {
                    'model': None,  # Model object not available from CSV
                    'metrics': {
                        'RMSE': row['RMSE'],
                        'MAE': row['MAE'],
                        'R2': row['R²'],
                        'MSE': row['RMSE'] ** 2
                    },
                    'model_name': model_name,
                    'source': 'baseline_csv',
                    'best_params': {}  # Parameters not available from CSV
                }
                
            logger.warning(f"Could not find existing optimized model for {dataset_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading existing model for {dataset_name}: {e}")
            return None
    
    def _load_best_baseline_models(self):
        """Load the best XGBoost models from baseline_comparison.csv."""
        try:
            baseline_df = pd.read_csv('outputs/metrics/baseline_comparison.csv')
            
            # Filter for XGBoost models with optuna optimization (excluding Random)
            xgb_models = baseline_df[
                (baseline_df['Model'].str.contains('XGBoost')) & 
                (baseline_df['Model'].str.contains('optuna')) &
                (~baseline_df['Model'].str.contains('Random'))
            ]
            
            # Get the best Base and Yeo models separately
            baseline_results = {}
            
            # Find best Base model
            base_models = xgb_models[xgb_models['Model'].str.contains('Base')]
            if not base_models.empty:
                best_base = base_models.nsmallest(1, 'RMSE').iloc[0]
                baseline_results['Base_baseline'] = {
                    'model_name': best_base['Model'],
                    'RMSE': best_base['RMSE'],
                    'MAE': best_base['MAE'],
                    'R2': best_base['R²'],
                    'MSE': best_base['RMSE'] ** 2
                }
            
            # Find best Yeo model
            yeo_models = xgb_models[xgb_models['Model'].str.contains('Yeo')]
            if not yeo_models.empty:
                best_yeo = yeo_models.nsmallest(1, 'RMSE').iloc[0]
                baseline_results['Yeo_baseline'] = {
                    'model_name': best_yeo['Model'],
                    'RMSE': best_yeo['RMSE'],
                    'MAE': best_yeo['MAE'],
                    'R2': best_yeo['R²'],
                    'MSE': best_yeo['RMSE'] ** 2
                }
            
            logger.info(f"Loaded baseline models: {list(baseline_results.keys())}")
            return baseline_results
            
        except Exception as e:
            logger.warning(f"Could not load baseline comparison: {e}")
            return {}
    
    def _generate_visualizations(self):
        """Generate comparison visualizations."""
        # Load best baseline models for comparison
        baseline_models = self._load_best_baseline_models()
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        metrics = ['RMSE', 'MAE', 'R2', 'MSE']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Prepare data for plotting
            datasets = list(self.results.keys())
            with_values = [self.results[ds]['comparison'][metric]['with_feature'] for ds in datasets]
            without_values = [self.results[ds]['comparison'][metric]['without_feature'] for ds in datasets]
            
            # Add baseline values if available
            baseline_values = []
            for ds in datasets:
                if 'Base_Random' in ds and 'Base_baseline' in baseline_models:
                    baseline_values.append(baseline_models['Base_baseline'][metric])
                elif 'Yeo_Random' in ds and 'Yeo_baseline' in baseline_models:
                    baseline_values.append(baseline_models['Yeo_baseline'][metric])
                else:
                    baseline_values.append(None)
            
            x = np.arange(len(datasets))
            width = 0.25 if baseline_models else 0.35
            
            # Create bars
            if baseline_models and any(v is not None for v in baseline_values):
                bars1 = ax.bar(x - width, with_values, width, label='With Feature', alpha=0.8)
                bars2 = ax.bar(x, without_values, width, label='Without Feature', alpha=0.8)
                # Only plot baseline bars where values exist
                bars3_list = []
                for i, val in enumerate(baseline_values):
                    if val is not None:
                        bar = ax.bar(x[i] + width, val, width, alpha=0.8, color='#9b59b6')
                        bars3_list.append(bar)
                # Add label only once
                if bars3_list:
                    bars3_list[0].set_label('Best Baseline')
            else:
                bars1 = ax.bar(x - width/2, with_values, width, label='With Feature', alpha=0.8)
                bars2 = ax.bar(x + width/2, without_values, width, label='Without Feature', alpha=0.8)
            
            # Color bars based on performance
            for i, (with_val, without_val) in enumerate(zip(with_values, without_values)):
                if metric in ['RMSE', 'MAE', 'MSE']:  # Lower is better
                    color1 = '#2ecc71' if with_val <= without_val else '#e74c3c'
                    color2 = '#e74c3c' if with_val <= without_val else '#2ecc71'
                else:  # R2 - Higher is better
                    color1 = '#2ecc71' if with_val >= without_val else '#e74c3c'
                    color2 = '#e74c3c' if with_val >= without_val else '#2ecc71'
                
                bars1[i].set_color(color1)
                bars2[i].set_color(color2)
            
            # Add value labels
            all_bars = [bars1, bars2]
                
            for bars in all_bars:
                for bar in bars:
                    height = bar.get_height()
                    if height is not None:  # Skip None values
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.4f}',
                               ha='center', va='bottom', fontsize=8)
            
            # Add baseline value labels if available
            if baseline_models and any(v is not None for v in baseline_values):
                for i, val in enumerate(baseline_values):
                    if val is not None:
                        ax.text(x[i] + width, val, f'{val:.4f}',
                               ha='center', va='bottom', fontsize=8)
            
            # Customize plot
            ax.set_xlabel('Dataset')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(datasets)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        title = f'Feature Removal Impact: Removing {len(self.excluded_features)} Features'
        if baseline_models:
            title += '\n(Compared with Best Baseline Models from Repository)'
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.base_output_dir / 'visualization' / 'feature_removal_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {plot_path}")
    
    def _generate_shap_visualizations(self):
        """Generate all SHAP visualizations for each model."""
        shap_dir = self.base_output_dir / 'visualization' / 'shap'
        
        # Collect only models without feature for SHAP analysis
        all_models = {}
        
        for dataset_name, results in self.results.items():
            # Add only without-feature model
            model_data_without = results['without_feature'].copy()
            model_key_without = f"{dataset_name}_without_feature"
            all_models[model_key_without] = model_data_without
        
        # Generate SHAP visualizations for each model
        for model_name, model_data in all_models.items():
            logger.info(f"Creating SHAP visualizations for {model_name}...")
            
            try:
                # Create model-specific output directory
                model_shap_dir = shap_dir / model_name
                model_shap_dir.mkdir(parents=True, exist_ok=True)
                
                # Get model and data
                model = model_data.get('model')
                X_test = model_data.get('X_test')
                
                # Skip if model not available
                if model is None:
                    logger.warning(f"Model object not available for {model_name}, skipping SHAP analysis")
                    continue
                
                # Sample data for SHAP analysis
                n_samples = min(100, len(X_test))
                sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
                X_sample = X_test.iloc[sample_indices]
                
                # Create SHAP explainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
                # Create visualizer
                visualizer = SHAPVisualizer(model_data)
                
                # 1. Summary plot (bar)
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
                plt.title(f'SHAP Feature Importance - {model_name}', fontsize=14, pad=20)
                plt.xlabel('Mean |SHAP Value| (Average Impact on Model Output)', fontsize=12)
                plt.tight_layout()
                plt.savefig(model_shap_dir / f"{model_name}_shap_summary_bar.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                # 2. Summary plot (dot)
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, show=False)
                plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14, pad=20)
                plt.tight_layout()
                plt.savefig(model_shap_dir / f"{model_name}_shap_summary_dot.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                # 3. Waterfall plot for first instance
                waterfall_path = model_shap_dir / f"{model_name}_shap_waterfall.png"
                visualizer.create_shap_waterfall_plot(shap_values, X_sample, 0, waterfall_path)
                
                # 4. Force plot for single instance (matplotlib doesn't support multiple)
                plt.figure(figsize=(20, 3))
                shap.force_plot(explainer.expected_value, shap_values[0], X_sample.iloc[0], 
                               matplotlib=True, show=False)
                plt.title(f'SHAP Force Plot - {model_name} (First instance)', fontsize=14)
                plt.savefig(model_shap_dir / f"{model_name}_shap_force.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                # Also create force plots for a few more individual instances
                for i in range(min(3, len(X_sample))):
                    plt.figure(figsize=(20, 3))
                    shap.force_plot(explainer.expected_value, shap_values[i], X_sample.iloc[i],
                                   matplotlib=True, show=False)
                    plt.title(f'SHAP Force Plot - {model_name} (Instance {i+1})', fontsize=14)
                    plt.savefig(model_shap_dir / f"{model_name}_shap_force_instance_{i+1}.png", dpi=300, bbox_inches='tight')
                    plt.close()
                
                # 5. Dependence plots for top features
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                top_features_idx = np.argsort(mean_abs_shap)[::-1][:5]
                
                for idx in top_features_idx:
                    feature_name = X_sample.columns[idx]
                    dependence_path = model_shap_dir / f"{model_name}_shap_dependence_{feature_name}.png"
                    visualizer.create_shap_dependence_plot(shap_values, X_sample, feature_name, dependence_path)
                
                # 6. Categorical feature plots
                categorical_features = model_data.get('categorical_features', [])
                for cat_feature in categorical_features[:3]:  # Top 3 categorical features
                    if cat_feature in X_sample.columns:
                        cat_path = model_shap_dir / f"{model_name}_shap_categorical_{cat_feature}.png"
                        visualizer.create_categorical_shap_plot(shap_values, X_sample, cat_feature, cat_path)
                
                # 7. Interaction plot for top 2 features
                if len(top_features_idx) >= 2:
                    plt.figure(figsize=(10, 8))
                    shap.dependence_plot(
                        top_features_idx[0], shap_values, X_sample,
                        interaction_index=top_features_idx[1], show=False
                    )
                    plt.title(f'SHAP Interaction Plot - {model_name}', fontsize=14)
                    plt.tight_layout()
                    plt.savefig(model_shap_dir / f"{model_name}_shap_interaction.png", dpi=300, bbox_inches='tight')
                    plt.close()
                
                logger.info(f"SHAP visualizations created for {model_name}")
                
            except Exception as e:
                logger.error(f"Error creating SHAP visualizations for {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Create custom comparison heatmap for feature removal analysis
        try:
            logger.info("Creating SHAP comparison heatmap...")
            self._create_feature_removal_shap_comparison(all_models, shap_dir)
        except Exception as e:
            logger.error(f"Error creating SHAP comparison heatmap: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_feature_removal_shap_comparison(self, all_models: Dict[str, Any], shap_dir: Path):
        """Create separate SHAP comparison heatmaps for Base_Random and Yeo_Random datasets."""
        logger.info("Creating separate SHAP feature importance comparisons for Base_Random and Yeo_Random...")
        
        # Separate models by dataset type
        base_models = {name: data for name, data in all_models.items() if 'Base_Random' in name}
        yeo_models = {name: data for name, data in all_models.items() if 'Yeo_Random' in name}
        
        # Create separate plots
        if base_models:
            self._create_dataset_specific_removal_plot(base_models, shap_dir, 'Base_Random')
        
        if yeo_models:
            self._create_dataset_specific_removal_plot(yeo_models, shap_dir, 'Yeo_Random')
    
    def _create_dataset_specific_removal_plot(self, models: Dict[str, Any], shap_dir: Path, dataset_type: str):
        """Create a SHAP comparison heatmap for a specific dataset type."""
        logger.info(f"Creating SHAP comparison for {dataset_type} models...")
        
        # Collect SHAP importance for each model
        shap_importance_data = {}
        all_features = set()
        
        for model_name, model_data in models.items():
            model = model_data.get('model')
            X_test = model_data.get('X_test')
            
            if model is None or X_test is None:
                logger.warning(f"Skipping {model_name} - model or test data not available")
                continue
                
            try:
                # Sample data for SHAP
                n_samples = min(100, len(X_test))
                sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
                X_sample = X_test.iloc[sample_indices]
                
                # Calculate SHAP values
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
                # Calculate mean absolute SHAP values
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                
                # Store results
                feature_importance = {}
                for i, feature in enumerate(X_sample.columns):
                    feature_importance[feature] = mean_abs_shap[i]
                    all_features.add(feature)
                
                # Use descriptive names for the plot
                display_name = 'With Feature' if 'with_feature' in model_name else 'Without Feature'
                shap_importance_data[display_name] = feature_importance
                logger.info(f"Calculated SHAP importance for {model_name}")
                
            except Exception as e:
                logger.error(f"Error calculating SHAP for {model_name}: {e}")
                continue
        
        if len(shap_importance_data) < 2:
            logger.warning(f"Not enough {dataset_type} models with SHAP values for comparison")
            return
        
        
        # Create DataFrame for easier manipulation
        importance_df = pd.DataFrame(shap_importance_data)
        
        # Select top 15 features based on maximum importance across models
        max_importance = importance_df.max(axis=1)
        top_features = max_importance.nlargest(15).index.tolist()
        importance_df_top = importance_df.loc[top_features]
        
        # Normalize values (0-1 scale for each model) - matching main pipeline
        normalized_df = importance_df_top.copy()
        for col in normalized_df.columns:
            max_val = normalized_df[col].max()
            if max_val > 0:
                normalized_df[col] = normalized_df[col] / max_val
        
        # Create heatmap with same formatting as main pipeline
        plt.figure(figsize=(10, 8))
        
        # Use viridis colormap to match main pipeline
        cmap = plt.cm.viridis
        
        # Create heatmap with annotations
        ax = plt.gca()
        im = ax.imshow(normalized_df.values, cmap=cmap, aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(len(normalized_df.columns)))
        ax.set_yticks(np.arange(len(normalized_df.index)))
        ax.set_xticklabels(normalized_df.columns)
        ax.set_yticklabels(normalized_df.index)
        
        # Rotate the tick labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized |SHAP Value| (0-1)', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(normalized_df.index)):
            for j in range(len(normalized_df.columns)):
                ax.text(j, i, f'{normalized_df.iloc[i, j]:.3f}',
                        ha="center", va="center", 
                        color="white" if normalized_df.iloc[i, j] > 0.5 else "black",
                        fontsize=8)
        
        # Add grid
        ax.set_xticks(np.arange(len(normalized_df.columns) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(normalized_df.index) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)
        
        plt.xlabel('Model Configuration', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Feature Importance Comparison - {dataset_type}\nRemoved: {len(self.excluded_features)} features (top_1, top_2, top_3, random)\n(Normalized 0-1 Scale per Model)', 
                  fontsize=14)
        plt.tight_layout()
        
        # Save plot with dataset-specific filename
        comparison_path = shap_dir / f'feature_removal_shap_comparison_{dataset_type.lower()}.png'
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP comparison heatmap saved to {comparison_path}")
    
    def _generate_residual_plots(self):
        """Generate residual analysis plots for each model."""
        residuals_dir = self.base_output_dir / 'visualization' / 'residual'
        
        for dataset_name, results in self.results.items():
            # Create residual plots only for without-feature model
            model_data_without = results['without_feature'].copy()
            model_name_without = f"{dataset_name}_without_feature"
            
            logger.info(f"Creating residual plots for {model_name_without}...")
            try:
                # Create configuration for residual plot
                config = {
                    'output_dir': residuals_dir,
                    'save': True,
                    'show': False,
                    'figsize': (16, 12),
                    'dpi': 300,
                    'grid': True,
                    'title_fontsize': 14,
                    'label_fontsize': 12,
                    'annotation_fontsize': 10
                }
                
                # Create residual plot
                residual_plot = ResidualPlot(model_data_without, config)
                residual_plot.plot()
                
                logger.info(f"Residual plots created for {model_name_without}")
                
            except Exception as e:
                logger.error(f"Error creating residual plots for {model_name_without}: {e}")
                import traceback
                traceback.print_exc()


def main():
    """Run the proper feature removal analysis."""
    analyzer = ProperXGBoostFeatureRemoval(
        # Remove all 4 features: top_1, top_2, top_3, and random
        excluded_features=[
            'top_1_shareholder_percentage',
            'top_2_shareholder_percentage',
            'top_3_shareholder_percentage',
            'random_feature'
        ],
        n_trials=100  # Reduced for testing - increase to 100+ for production
    )
    analyzer.run_analysis()


if __name__ == "__main__":
    main()