#!/usr/bin/env python3
"""
Enhanced XGBoost feature removal analysis with iterative tracking.
This version:
- Tracks feature removal iterations
- Saves detailed metrics for each iteration
- Creates comprehensive CSV output matching model_metrics_comparison.csv format
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
from datetime import datetime

# Import configuration
from src.config import settings

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_removal_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IterativeXGBoostFeatureRemoval:
    """Enhanced XGBoost feature removal analyzer with iterative tracking."""
    
    def __init__(self, 
                 n_trials: int = 100,
                 base_output_dir: Optional[Path] = None):
        """
        Initialize the analyzer.
        
        Args:
            n_trials: Number of Optuna trials for optimization
            base_output_dir: Output directory for results
        """
        self.n_trials = n_trials
        # Use absolute path from script location
        project_root = Path(__file__).parent
        self.base_output_dir = base_output_dir or (project_root / "outputs" / "feature_removal")
        self.iteration_results = []  # Track results for each iteration
        
    def run_iterative_analysis(self, 
                              X_train: pd.DataFrame, 
                              X_test: pd.DataFrame,
                              y_train: pd.Series,
                              y_test: pd.Series,
                              feature_importance_df: pd.DataFrame,
                              model_name: str = "XGBoost_Optuna",
                              n_iterations: int = 10):
        """
        Run iterative feature removal analysis.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target
            feature_importance_df: DataFrame with feature importance scores
            model_name: Base model name
            n_iterations: Number of features to remove iteratively
        """
        logger.info(f"Starting iterative feature removal analysis for {model_name}")
        logger.info(f"Initial features: {X_train.shape[1]}")
        logger.info(f"Planned iterations: {n_iterations}")
        
        # Sort features by importance (ascending - least important first)
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=True)
        features_to_remove = feature_importance_df['feature'].tolist()[:n_iterations]
        
        # Track cumulative removed features
        removed_features = []
        current_X_train = X_train.copy()
        current_X_test = X_test.copy()
        
        # Iteration 0: Baseline with all features
        logger.info("\nIteration 0: Baseline model with all features")
        baseline_results = self._train_and_evaluate(
            current_X_train, current_X_test, y_train, y_test,
            iteration=0, removed_features=[], model_name=model_name
        )
        self.iteration_results.append(baseline_results)
        
        # Iterative feature removal
        for iteration in range(1, n_iterations + 1):
            if iteration <= len(features_to_remove):
                # Remove next feature
                feature_to_remove = features_to_remove[iteration - 1]
                removed_features.append(feature_to_remove)
                
                logger.info(f"\nIteration {iteration}: Removing '{feature_to_remove}'")
                logger.info(f"Total features removed: {len(removed_features)}")
                
                # Remove feature from datasets
                if feature_to_remove in current_X_train.columns:
                    current_X_train = current_X_train.drop(columns=[feature_to_remove])
                    current_X_test = current_X_test.drop(columns=[feature_to_remove])
                    logger.info(f"Remaining features: {current_X_train.shape[1]}")
                else:
                    logger.warning(f"Feature '{feature_to_remove}' not found in dataset")
                
                # Train and evaluate model
                iteration_results = self._train_and_evaluate(
                    current_X_train, current_X_test, y_train, y_test,
                    iteration=iteration, 
                    removed_features=removed_features.copy(),
                    model_name=model_name,
                    feature_removed_this_iter=feature_to_remove,
                    feature_importance=feature_importance_df[
                        feature_importance_df['feature'] == feature_to_remove
                    ]['importance'].values[0] if feature_to_remove in feature_importance_df['feature'].values else 0
                )
                self.iteration_results.append(iteration_results)
        
        # Save comprehensive results
        self._save_iterative_metrics()
        self._generate_summary_report()
        
    def _train_and_evaluate(self, 
                           X_train: pd.DataFrame, 
                           X_test: pd.DataFrame,
                           y_train: pd.Series,
                           y_test: pd.Series,
                           iteration: int,
                           removed_features: List[str],
                           model_name: str,
                           feature_removed_this_iter: str = None,
                           feature_importance: float = None) -> Dict[str, Any]:
        """
        Train and evaluate XGBoost model with Optuna optimization.
        
        Returns dictionary with all metrics and metadata.
        """
        logger.debug(f"Training model for iteration {iteration}")
        
        # Define objective function for Optuna
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
            
            # Cross-validation
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            
            for train_idx, val_idx in cv.split(X_train):
                X_fold_train = X_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_train = y_train.iloc[train_idx]
                y_fold_val = y_train.iloc[val_idx]
                
                model = xgb.XGBRegressor(**params)
                model.fit(X_fold_train, y_fold_train)
                
                y_pred = model.predict(X_fold_val)
                mse = mean_squared_error(y_fold_val, y_pred)
                cv_scores.append(mse)
            
            return np.mean(cv_scores)
        
        # Run Optuna optimization
        logger.debug(f"Starting Optuna optimization with {self.n_trials} trials")
        study = optuna.create_study(
            direction='minimize',
            study_name=f"{model_name}_iter_{iteration}",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
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
        
        # Calculate metrics
        y_pred = final_model.predict(X_test)
        
        # Calculate CV scores for the final model manually
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in cv.split(X_train):
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            cv_model = xgb.XGBRegressor(**best_params)
            cv_model.fit(X_fold_train, y_fold_train)
            
            y_pred_cv = cv_model.predict(X_fold_val)
            mse_cv = mean_squared_error(y_fold_val, y_pred_cv)
            cv_scores.append(np.sqrt(mse_cv))
        
        cv_rmse_scores = np.array(cv_scores)
        
        # Compile results
        results = {
            'iteration': iteration,
            'model_name': f"{model_name}_iter_{iteration}",
            'metrics': {
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred),
                'R2': r2_score(y_test, y_pred),
                'MSE': mean_squared_error(y_test, y_pred)
            },
            'cv_metrics': {
                'cv_rmse_mean': np.mean(cv_rmse_scores),
                'cv_rmse_std': np.std(cv_rmse_scores),
                'cv_scores': cv_rmse_scores.tolist()
            },
            'data_info': {
                'n_train': len(X_train),
                'n_test': len(X_test),
                'n_features': X_train.shape[1],
                'n_categorical': len([col for col in X_train.columns if X_train[col].dtype == 'category']),
                'n_numerical': X_train.shape[1] - len([col for col in X_train.columns if X_train[col].dtype == 'category'])
            },
            'feature_removal': {
                'features_removed': removed_features,
                'n_features_removed': len(removed_features),
                'feature_removed_this_iter': feature_removed_this_iter,
                'feature_importance_score': feature_importance
            },
            'optimization': {
                'best_params': best_params,
                'best_value': study.best_value,
                'n_trials': len(study.trials)
            },
            'cv_folds': 5
        }
        
        logger.info(f"Iteration {iteration} - RMSE: {results['metrics']['RMSE']:.4f}, "
                   f"R²: {results['metrics']['R2']:.4f}")
        
        return results
    
    def _save_iterative_metrics(self):
        """
        Save iterative feature removal metrics in the same format as model_metrics_comparison.csv
        with additional columns for feature removal tracking.
        """
        logger.info("Saving iterative metrics to CSV")
        
        metrics_data = []
        
        for result in self.iteration_results:
            metrics = result['metrics']
            data_info = result['data_info']
            feature_removal = result['feature_removal']
            cv_metrics = result['cv_metrics']
            
            metric_row = {
                # Standard columns from model_metrics_comparison.csv
                'Model Name': result['model_name'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R2': metrics['R2'],
                'MSE': metrics['MSE'],
                'Number of CV Folds': result['cv_folds'],
                'Number of Testing Samples': data_info['n_test'],
                'Number of Training Samples': data_info['n_train'],
                'Number of Quantitative Features': data_info['n_numerical'],
                'Number of Qualitative Features': data_info['n_categorical'],
                
                # Additional columns for feature removal analysis
                'Iteration': result['iteration'],
                'Number of Features Removed This Iteration': 1 if result['iteration'] > 0 else 0,
                'Total Features Removed': feature_removal['n_features_removed'],
                'Features Removed': ', '.join(feature_removal['features_removed']) if feature_removal['features_removed'] else 'None',
                'Feature Removed This Iteration': feature_removal.get('feature_removed_this_iter', 'None'),
                'Feature Importance Score': feature_removal.get('feature_importance_score', 0),
                
                # CV metrics
                'CV RMSE Mean': cv_metrics['cv_rmse_mean'],
                'CV RMSE Std': cv_metrics['cv_rmse_std'],
                
                # Total features
                'Total Features': data_info['n_features']
            }
            
            metrics_data.append(metric_row)
        
        # Create DataFrame
        metrics_df = pd.DataFrame(metrics_data)
        
        # Sort by iteration
        metrics_df = metrics_df.sort_values('Iteration')
        
        # Create output directory
        output_dir = self.base_output_dir / "visualization"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV without timestamp
        csv_path = output_dir / "xgboost_feature_removal_metrics_analysis.csv"
        metrics_df.to_csv(csv_path, index=False)
        
        logger.info(f"Saved metrics to: {csv_path}")
        
        return metrics_df
    
    def _generate_summary_report(self):
        """Generate a summary report of the iterative analysis."""
        logger.info("Generating summary report")
        
        report_lines = []
        report_lines.append("# XGBoost Iterative Feature Removal Analysis Report")
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"\n## Analysis Summary")
        report_lines.append(f"- Total iterations: {len(self.iteration_results)}")
        report_lines.append(f"- Features removed: {len(self.iteration_results) - 1}")
        report_lines.append(f"- Optimization trials per iteration: {self.n_trials}")
        
        # Performance degradation summary
        baseline_rmse = self.iteration_results[0]['metrics']['RMSE']
        final_rmse = self.iteration_results[-1]['metrics']['RMSE']
        rmse_change = ((final_rmse - baseline_rmse) / baseline_rmse) * 100
        
        baseline_r2 = self.iteration_results[0]['metrics']['R2']
        final_r2 = self.iteration_results[-1]['metrics']['R2']
        r2_change = final_r2 - baseline_r2
        
        report_lines.append(f"\n## Performance Impact")
        report_lines.append(f"- Baseline RMSE: {baseline_rmse:.4f}")
        report_lines.append(f"- Final RMSE: {final_rmse:.4f}")
        report_lines.append(f"- RMSE Change: {rmse_change:+.2f}%")
        report_lines.append(f"- Baseline R²: {baseline_r2:.4f}")
        report_lines.append(f"- Final R²: {final_r2:.4f}")
        report_lines.append(f"- R² Change: {r2_change:+.4f}")
        
        # Feature removal details
        report_lines.append(f"\n## Feature Removal Details")
        report_lines.append("| Iteration | Feature Removed | Importance Score | RMSE | R² |")
        report_lines.append("|-----------|----------------|------------------|------|-----|")
        
        for result in self.iteration_results:
            iter_num = result['iteration']
            feature = result['feature_removal'].get('feature_removed_this_iter', 'None')
            importance = result['feature_removal'].get('feature_importance_score', 0)
            rmse = result['metrics']['RMSE']
            r2 = result['metrics']['R2']
            
            report_lines.append(f"| {iter_num} | {feature} | {importance:.6f} | {rmse:.4f} | {r2:.4f} |")
        
        # Save report
        report_dir = self.base_output_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = report_dir / "iterative_feature_removal_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Saved report to: {report_path}")


# Example usage function
def run_iterative_feature_removal_analysis():
    """Example function to run the iterative feature removal analysis."""
    
    # Load your data here
    # This is a placeholder - replace with actual data loading
    from src.data.data_categorical import load_tree_models_data, get_base_and_yeo_features_categorical
    
    logger.info("Loading data for feature removal analysis")
    
    # Load features and target
    features, target = load_tree_models_data()
    base_features_df, yeo_features_df = get_base_and_yeo_features_categorical()
    
    # Add random feature for testing
    from src.data.data_categorical import add_random_feature_categorical
    base_random_df = add_random_feature_categorical(base_features_df)
    
    # Create train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        base_random_df, target, test_size=0.2, random_state=42
    )
    
    # Load or calculate feature importance
    # This is a placeholder - replace with actual feature importance
    feature_importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': np.random.rand(len(X_train.columns))  # Replace with actual importance
    })
    
    # Run analysis
    analyzer = IterativeXGBoostFeatureRemoval(n_trials=50)  # Reduced for example
    analyzer.run_iterative_analysis(
        X_train, X_test, y_train, y_test,
        feature_importance_df,
        model_name="XGBoost_Base_Random_Optuna",
        n_iterations=10
    )


if __name__ == "__main__":
    run_iterative_feature_removal_analysis()