"""Adapter for CatBoost models."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any, Union

from visualization_new.core.interfaces import ModelData
from visualization_new.utils.data_prep import prepare_prediction_data, prepare_residuals

# Check if optuna is available
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

class CatBoostAdapter(ModelData):
    """Adapter for CatBoost models."""
    
    def __init__(self, model_data: Dict[str, Any]):
        """
        Initialize CatBoost adapter.
        
        Args:
            model_data (dict): CatBoost model data dictionary
        """
        self.model_data = model_data
        self.model_name = model_data.get('model_name', 'Unknown')
        self.model = model_data.get('model', None)

    def get_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get test set predictions and actual values."""
        y_test = self.model_data.get('y_test')
        y_pred = self.model_data.get('y_pred') or self.model_data.get('y_test_pred')
        
        if y_test is None or y_pred is None:
            raise ValueError(f"Missing y_test or y_pred/y_test_pred in model data for {self.model_name}")
        
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
        
        # Extract feature importance from model
        try:
            importance = self.model.get_feature_importance()
            std = np.zeros_like(importance)
        except:
            # Fallback: try to use feature_importances_ attribute
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                std = np.zeros_like(importance)
            else:
                return pd.DataFrame(columns=['Feature', 'Importance', 'Std'])
        
        # Get feature names
        if hasattr(self.model, 'feature_names_'):
            feature_names = self.model.feature_names_
        elif 'feature_names' in self.model_data:
            feature_names = self.model_data['feature_names']
        else:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        # Create DataFrame
        df = pd.DataFrame({
            'Feature': feature_names[:len(importance)],  # Ensure lengths match
            'Importance': importance,
            'Std': std
        })
        
        # Sort by importance
        df = df.sort_values('Importance', ascending=False)
        
        return df
    
    def get_metrics(self) -> Dict[str, float]:
        """Get model performance metrics."""
        metrics = {}
        
        # Check for CatBoost-specific metric keys
        catboost_mapping = {
            'test_mse': 'MSE',
            'test_mae': 'MAE',
            'test_r2': 'R2',
            'test_rmse': 'RMSE'
        }
        
        # First try CatBoost-specific keys
        for cb_key, std_key in catboost_mapping.items():
            if cb_key in self.model_data:
                metrics[std_key] = float(self.model_data[cb_key])
        
        # Then check standard metrics from model data
        for metric in ['RMSE', 'MAE', 'MSE', 'R2']:
            if metric in self.model_data and metric not in metrics:
                metrics[metric] = self.model_data[metric]
        
        # Calculate any missing metrics
        if 'RMSE' not in metrics and 'MSE' in metrics:
            metrics['RMSE'] = np.sqrt(metrics['MSE'])
            
        if 'MSE' not in metrics and 'RMSE' in metrics:
            metrics['MSE'] = metrics['RMSE'] ** 2
            
        # If still missing key metrics, calculate from predictions
        if 'R2' not in metrics or 'MAE' not in metrics or 'RMSE' not in metrics:
            try:
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
            except Exception as e:
                print(f"Warning: Could not calculate metrics for {self.model_name}: {e}")
        
        return metrics
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        if self.model is None:
            return {}
            
        # Check if best_params exists (for Optuna)
        if 'best_params' in self.model_data:
            return self.model_data['best_params']
            
        # Extract hyperparameters from model
        try:
            # Get parameters from model
            params = self.model.get_params()
        except:
            # Fallback to empty dict
            params = {}
            
        return params
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        metadata = {
            'model_name': self.model_name,
            'model_type': self.model_data.get('model_type', 'CatBoost'),
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
            
        return None
        
    def get_raw_model_data(self) -> Dict[str, Any]:
        """
        Get the raw model data dictionary.
        
        This is useful for accessing any model data that doesn't have a specific getter method.
        
        Returns:
            The raw model data dictionary
        """
        return self.model_data