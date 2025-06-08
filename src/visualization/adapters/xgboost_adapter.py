"""Adapter for XGBoost models."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any, Union

from src.visualization.core.interfaces import ModelData
from src.visualization.utils.data_prep import prepare_prediction_data, prepare_residuals

# Check if optuna is available
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

class XGBoostAdapter(ModelData):
    """Adapter for XGBoost models."""
    
    def __init__(self, model_data: Dict[str, Any]):
        """
        Initialize XGBoost adapter.
        
        Args:
            model_data (dict): XGBoost model data dictionary
        """
        self.model_data = model_data
        self.model_name = model_data.get('model_name', 'Unknown')
        self.model = model_data.get('model', None)

    def get_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get test set predictions and actual values."""
        y_test = self.model_data.get('y_test')
        y_pred = self.model_data.get('y_pred')
        
        if y_test is None or y_pred is None:
            raise ValueError(f"Missing y_test or y_pred in model data for {self.model_name}")
        
        # Standardize and return
        return prepare_prediction_data(y_test, y_pred)
    
    def get_residuals(self) -> np.ndarray:
        """Get model residuals."""
        # Calculate from predictions
        y_test, y_pred = self.get_predictions()
        return y_test - y_pred
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance data."""
        if self.model is None:
            return pd.DataFrame(columns=['Feature', 'Importance', 'Std'])
        
        # Check if precomputed feature importance exists
        if 'feature_importance' in self.model_data:
            importance_df = self.model_data['feature_importance'].copy()
            
            # Ensure correct format
            if not isinstance(importance_df, pd.DataFrame):
                raise ValueError(f"Feature importance is not a DataFrame for {self.model_name}")
            
            # Handle different column naming conventions
            if 'feature' in importance_df.columns and 'Feature' not in importance_df.columns:
                importance_df = importance_df.rename(columns={'feature': 'Feature'})
            if 'importance' in importance_df.columns and 'Importance' not in importance_df.columns:
                importance_df = importance_df.rename(columns={'importance': 'Importance'})
            
            # Ensure required columns
            if 'Feature' not in importance_df.columns or 'Importance' not in importance_df.columns:
                raise ValueError(f"Feature importance DataFrame missing required columns for {self.model_name}. Found: {list(importance_df.columns)}")
            
            # Add Std column if missing
            if 'Std' not in importance_df.columns:
                importance_df['Std'] = 0.0
                
            return importance_df
        
        # Extract feature importance from model
        importance = self.model.feature_importances_
        std = np.zeros_like(importance)
        
        # Get feature names
        if hasattr(self.model, 'feature_names_in_'):
            feature_names = self.model.feature_names_in_
        elif 'feature_names' in self.model_data:
            feature_names = self.model_data['feature_names']
        else:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        # Create DataFrame
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance,
            'Std': std
        })
        
        # Sort by importance
        df = df.sort_values('Importance', ascending=False)
        
        return df
    
    def get_metrics(self) -> Dict[str, float]:
        """Get model performance metrics."""
        metrics = {}
        
        # First check if metrics are stored in a nested 'metrics' dictionary
        if 'metrics' in self.model_data and isinstance(self.model_data['metrics'], dict):
            # Extract test metrics from nested structure
            nested_metrics = self.model_data['metrics']
            if 'test_rmse' in nested_metrics:
                metrics['RMSE'] = nested_metrics['test_rmse']
            if 'test_mae' in nested_metrics:
                metrics['MAE'] = nested_metrics['test_mae']
            if 'test_r2' in nested_metrics:
                metrics['R2'] = nested_metrics['test_r2']
            if 'test_mse' in nested_metrics:
                metrics['MSE'] = nested_metrics['test_mse']
        
        # If not found in nested structure, check top-level
        if not metrics:
            for metric in ['RMSE', 'MAE', 'MSE', 'R2']:
                if metric in self.model_data:
                    metrics[metric] = self.model_data[metric]
        
        # Calculate any missing metrics
        if 'RMSE' not in metrics and 'MSE' in metrics:
            metrics['RMSE'] = np.sqrt(metrics['MSE'])
            
        if 'MSE' not in metrics and 'RMSE' in metrics:
            metrics['MSE'] = metrics['RMSE'] ** 2
            
        if 'R2' not in metrics or 'MAE' not in metrics or 'RMSE' not in metrics:
            y_test, y_pred = self.get_predictions()
            
            if 'R2' not in metrics:
                from sklearn.metrics import r2_score
                metrics['R2'] = r2_score(y_test, y_pred)
                
            if 'MAE' not in metrics:
                from sklearn.metrics import mean_absolute_error
                metrics['MAE'] = mean_absolute_error(y_test, y_pred)
                
            if 'RMSE' not in metrics:
                from sklearn.metrics import mean_squared_error
                metrics['RMSE'] = np.sqrt(mean_squared_error(y_test, y_pred))
                
            if 'MSE' not in metrics:
                metrics['MSE'] = metrics['RMSE'] ** 2
        
        return metrics
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        if self.model is None:
            return {}
            
        # Check if best_params exists (for Optuna)
        if 'best_params' in self.model_data:
            return self.model_data['best_params']
            
        # Extract hyperparameters from model
        params = self.model.get_params()
        return params
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        metadata = {
            'model_name': self.model_name,
            'model_type': self.model_data.get('model_type', 'XGBoost'),
        }
        
        # Extract dataset and variant from model name
        # e.g., "XGBoost_Base_categorical_basic" -> dataset="Base", variant="categorical_basic"
        if self.model_name and self.model_name != 'Unknown':
            parts = self.model_name.split('_')
            if len(parts) >= 2:
                # Handle different naming patterns
                if parts[0].lower() == 'xgboost' and len(parts) >= 3:
                    # Extract dataset (Base, Yeo, Base_Random, Yeo_Random)
                    if len(parts) >= 4 and parts[2].lower() == 'random':
                        dataset = f"{parts[1]}_{parts[2]}"
                        variant_start = 3
                    else:
                        dataset = parts[1]
                        variant_start = 2
                    
                    # Extract variant (categorical_basic, categorical_optuna, etc.)
                    variant_parts = parts[variant_start:]
                    variant = '_'.join(variant_parts) if variant_parts else 'basic'
                    
                    # Create descriptive model name
                    dataset_display = dataset.replace('_', ' ')
                    variant_display = variant.replace('_', ' ').title()
                    
                    # Override the generic model_name with a descriptive one
                    metadata['model_name'] = f"XGBoost_{dataset}_{variant}"
        
        # Add additional metadata if available
        for key in ['dataset', 'n_features', 'n_companies', 'n_companies_train', 'n_companies_test']:
            if key in self.model_data:
                metadata[key] = self.model_data[key]
        
        return metadata
        
    def get_study(self) -> Optional[Any]:
        """
        Get the Optuna study object for optimization visualizations.
        
        Returns:
            Optuna study object if available, None otherwise
        """
        if not OPTUNA_AVAILABLE:
            return None
            
        # Check if study is available in model data
        if 'study' in self.model_data:
            return self.model_data['study']
            
        return None

    def get_model_type(self) -> str:
        """Get model type."""
        return "XGBoost"
    
    def get_dataset_name(self) -> str:
        """Get dataset name from model name."""
        if hasattr(self, 'model_name') and self.model_name:
            # Extract dataset from model name (e.g., "XGBoost_Base_categorical_optuna" -> "Base")
            parts = self.model_name.split('_')
            if len(parts) >= 2:
                # Handle cases like Base, Yeo, Base_Random, Yeo_Random
                if len(parts) >= 3 and parts[2] == 'Random':
                    return f"{parts[1]}_{parts[2]}"
                else:
                    return parts[1]
        return "Unknown"
    
    def get_raw_model_data(self) -> dict:
        """Get raw model data dictionary."""
        return self.model_data