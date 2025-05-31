"""Adapter for ElasticNet models."""

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

class ElasticNetAdapter(ModelData):
    """Adapter for ElasticNet models."""
    
    def __init__(self, model_data: Dict[str, Any]):
        """
        Initialize ElasticNet adapter.
        
        Args:
            model_data (dict): ElasticNet model data dictionary
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
            importance_df = self.model_data['feature_importance']
            
            # Ensure correct format
            if not isinstance(importance_df, pd.DataFrame):
                raise ValueError(f"Feature importance is not a DataFrame for {self.model_name}")
            
            # Ensure required columns
            if 'Feature' not in importance_df.columns or 'Importance' not in importance_df.columns:
                raise ValueError(f"Feature importance DataFrame missing required columns for {self.model_name}")
            
            # Add Std column if missing
            if 'Std' not in importance_df.columns:
                importance_df['Std'] = 0.0
                
            return importance_df
        
        # Extract feature importance from model coefficients
        # For ElasticNet, we use the absolute value of coefficients as importance
        importance = np.abs(self.model.coef_)
        std = np.zeros_like(importance)
        
        # Get feature names
        if hasattr(self.model, 'feature_names_in_'):
            feature_names = self.model.feature_names_in_
        elif 'feature_names' in self.model_data:
            feature_names = self.model_data['feature_names']
        else:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        # Pre-scaling for better visibility in cross-model comparisons
        # Apply a base scaling to make ElasticNet values more comparable with tree-based models
        # This is applied here so that individual ElasticNet plots are still readable,
        # while cross-model comparisons have more consistent scale
        base_scale = 100  # Apply a base scaling in the adapter itself
        importance = importance * base_scale
        print(f"ElasticNetAdapter: Applied base scaling of {base_scale}x to feature importance values")
        print(f"  Min: {np.min(importance)}, Max: {np.max(importance)}, Mean: {np.mean(importance)}")
        
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
        
        # Copy standard metrics from model data
        for metric in ['RMSE', 'MAE', 'MSE', 'R2']:
            if metric in self.model_data:
                metrics[metric] = self.model_data[metric]
        
        # Calculate any missing metrics
        if 'RMSE' not in metrics and 'MSE' in metrics:
            metrics['RMSE'] = np.sqrt(metrics['MSE'])
            
        if 'MSE' not in metrics and 'RMSE' in metrics:
            metrics['MSE'] = metrics['RMSE'] ** 2
            
        if 'R2' not in metrics or 'MAE' not in metrics:
            y_test, y_pred = self.get_predictions()
            
            if 'R2' not in metrics:
                from sklearn.metrics import r2_score
                metrics['R2'] = r2_score(y_test, y_pred)
                
            if 'MAE' not in metrics:
                from sklearn.metrics import mean_absolute_error
                metrics['MAE'] = mean_absolute_error(y_test, y_pred)
        
        return metrics
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        if self.model is None:
            return {}
            
        # Check if best_params exists (for CV)
        if 'best_params' in self.model_data:
            return self.model_data['best_params']
            
        # Extract hyperparameters from model
        params = {
            'alpha': getattr(self.model, 'alpha', None),
            'l1_ratio': getattr(self.model, 'l1_ratio', None),
        }
        return {k: v for k, v in params.items() if v is not None}
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        metadata = {
            'model_name': self.model_name,
            'model_type': 'elasticnet',
        }
        
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
        
        # ElasticNet models might store grid search results instead of Optuna study
        if 'cv_results' in self.model_data:
            # Check if we can convert grid search results to an Optuna-like interface
            try:
                # We could implement conversion from grid search to Optuna study here
                # For now, we'll return None to indicate no Optuna study available
                print(f"Grid search results found for {self.model_name}, but conversion to Optuna study not implemented yet")
                return None
            except Exception as e:
                print(f"Error converting grid search results to Optuna study: {e}")
                return None
            
        return None
        
    def get_raw_model_data(self) -> Dict[str, Any]:
        """
        Get the raw model data dictionary.
        
        This is useful for accessing any model data that doesn't have a specific getter method.
        
        Returns:
            The raw model data dictionary
        """

    def get_model_type(self) -> str:
        """Get model type."""
        return "ElasticNet"
    
    def get_dataset_name(self) -> str:
        """Get dataset name from model name."""
        if hasattr(self, 'model_name') and self.model_name:
            # Extract dataset from model name
            parts = self.model_name.split('_')
            
            # Special handling for ElasticNet models with "LR" prefix
            # e.g., "ElasticNet_LR_Base_basic" -> "Base"
            if len(parts) >= 3 and parts[0] == 'ElasticNet' and parts[1] == 'LR':
                # Skip the "LR" part
                if len(parts) >= 4 and parts[3] == 'Random':
                    return f"{parts[2]}_{parts[3]}"
                else:
                    return parts[2]
            # Standard case for other models
            elif len(parts) >= 2:
                # Handle cases like Base, Yeo, Base_Random, Yeo_Random
                if len(parts) >= 3 and parts[2] == 'Random':
                    return f"{parts[1]}_{parts[2]}"
                else:
                    return parts[1]
        return "Unknown"
    
    def get_raw_model_data(self) -> dict:
        """Get raw model data dictionary."""
        return self.model_data