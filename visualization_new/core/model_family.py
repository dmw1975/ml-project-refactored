"""Model family definitions and utilities."""

from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple

class ModelFamilyType(Enum):
    """Enumeration of model family types."""
    LINEAR = "Linear"
    ELASTICNET = "ElasticNet"  # Separate type for ElasticNet
    XGBOOST = "XGBoost"
    LIGHTGBM = "LightGBM"
    CATBOOST = "CatBoost"
    UNKNOWN = "Unknown"
    
    @classmethod
    def from_model_name(cls, model_name: str) -> "ModelFamilyType":
        """
        Determine model family from model name.
        
        Args:
            model_name: Model name
            
        Returns:
            ModelFamilyType: Model family type
        """
        if model_name.startswith('ElasticNet_'):
            return cls.ELASTICNET  # ElasticNet has its own type now
        elif model_name.startswith('LR_'):
            return cls.LINEAR
        elif 'XGB' in model_name:
            return cls.XGBOOST
        elif 'LightGBM' in model_name:
            return cls.LIGHTGBM
        elif 'CatBoost' in model_name:
            return cls.CATBOOST
        else:
            return cls.UNKNOWN
    
    @property
    def color(self) -> str:
        """
        Get color for model family.
        
        Returns:
            str: Color hexcode
        """
        colors = {
            self.LINEAR: '#9b59b6',    # Purple
            self.ELASTICNET: '#f39c12', # Orange
            self.XGBOOST: '#3498db',   # Blue
            self.LIGHTGBM: '#2ecc71',  # Green
            self.CATBOOST: '#e74c3c',  # Red
            self.UNKNOWN: '#95a5a6'    # Gray
        }
        return colors.get(self, '#95a5a6')
    
    @property
    def short_name(self) -> str:
        """
        Get short name for model family.
        
        Returns:
            str: Short name
        """
        short_names = {
            self.LINEAR: 'LIN',
            self.ELASTICNET: 'ENET',
            self.XGBOOST: 'XGB',
            self.LIGHTGBM: 'LGBM',
            self.CATBOOST: 'CB',
            self.UNKNOWN: 'UNK'
        }
        return short_names.get(self, 'UNK')


class ModelFamily:
    """
    Model family information.
    
    Attributes:
        family_type: Type of model family
        name: Full name of model family
        color: Color for visualization
        short_name: Short name for labels
        basic_variant: Name of basic (untuned) variant
        tuned_variant: Name of tuned variant
    """
    
    def __init__(
        self, 
        family_type: ModelFamilyType,
        basic_variant: str = "Basic",
        tuned_variant: str = "Tuned"
    ):
        """
        Initialize model family.
        
        Args:
            family_type: Type of model family
            basic_variant: Name of basic variant
            tuned_variant: Name of tuned variant
        """
        self.family_type = family_type
        self.name = family_type.value
        self.color = family_type.color
        self.short_name = family_type.short_name
        self.basic_variant = basic_variant
        self.tuned_variant = tuned_variant
    
    @classmethod
    def from_model_name(cls, model_name: str) -> "ModelFamily":
        """
        Create model family from model name.
        
        Args:
            model_name: Model name
            
        Returns:
            ModelFamily: Model family object
        """
        family_type = ModelFamilyType.from_model_name(model_name)
        
        # Determine basic and tuned variant names
        if family_type == ModelFamilyType.LINEAR:
            basic_variant = "Linear Regression"
            tuned_variant = "Linear Regression"  # Linear models are not tuned
        elif family_type == ModelFamilyType.ELASTICNET:
            basic_variant = "ElasticNet"  # Not used since all ElasticNet models are "tuned"
            tuned_variant = "ElasticNet"
        elif family_type == ModelFamilyType.XGBOOST:
            basic_variant = "XGBoost Basic"
            tuned_variant = "XGBoost Optuna"
        elif family_type == ModelFamilyType.LIGHTGBM:
            basic_variant = "LightGBM Basic"
            tuned_variant = "LightGBM Optuna"
        elif family_type == ModelFamilyType.CATBOOST:
            basic_variant = "CatBoost Basic"
            tuned_variant = "CatBoost Optuna"
        else:
            basic_variant = "Basic"
            tuned_variant = "Tuned"
        
        return cls(family_type, basic_variant, tuned_variant)
    
    def is_tuned_variant(self, model_name: str) -> bool:
        """
        Check if model name is tuned variant.
        
        Args:
            model_name: Model name
            
        Returns:
            bool: True if tuned, False if basic
        """
        if self.family_type == ModelFamilyType.LINEAR:
            return False  # Linear models are not tuned
        elif self.family_type == ModelFamilyType.ELASTICNET:
            # For now consider all ElasticNet models as "tuned" since they use CV
            return True 
        elif self.family_type == ModelFamilyType.XGBOOST:
            return 'optuna' in model_name
        elif self.family_type == ModelFamilyType.LIGHTGBM:
            return 'optuna' in model_name
        elif self.family_type == ModelFamilyType.CATBOOST:
            return 'optuna' in model_name
        else:
            return False
    
    def get_variant_name(self, is_tuned: bool) -> str:
        """
        Get variant name based on tuning status.
        
        Args:
            is_tuned: Whether model is tuned
            
        Returns:
            str: Variant name
        """
        return self.tuned_variant if is_tuned else self.basic_variant
    
    def get_dataset_from_model_name(self, model_name: str) -> str:
        """
        Extract dataset name from model name.
        
        Args:
            model_name: Model name
            
        Returns:
            str: Dataset name
        """
        # Debug info to help diagnose issues
        original_model_name = model_name
        print(f"Extracting dataset from model: {model_name}, Family type: {self.family_type}")
        
        if self.family_type == ModelFamilyType.LINEAR:
            # Extract from LR_Base, LR_Yeo, etc.
            parts = model_name.split('_')
            if len(parts) >= 2:
                dataset = '_'.join(parts[1:])
                print(f"  Linear model: Extracted dataset '{dataset}' from {model_name}")
                return dataset
        elif self.family_type == ModelFamilyType.ELASTICNET:
            # Extract from ElasticNet_LR_Base, ElasticNet_LR_Yeo, etc.
            parts = model_name.split('_')
            print(f"  ElasticNet model: Parsed parts: {parts}")
            
            if len(parts) >= 3:
                # For ElasticNet_LR_Base, return "Base"
                dataset = '_'.join(parts[2:])
                print(f"  ElasticNet model: Extracted dataset '{dataset}' from {model_name}")
                return dataset
        else:
            # For other models, extract dataset after family name
            # E.g., XGB_Base, LightGBM_Yeo, etc.
            if self.family_type == ModelFamilyType.XGBOOST:
                prefix = 'XGB_'
            elif self.family_type == ModelFamilyType.LIGHTGBM:
                prefix = 'LightGBM_'
            elif self.family_type == ModelFamilyType.CATBOOST:
                prefix = 'CatBoost_'
            else:
                print(f"  Unknown model type: {self.family_type}")
                return 'Unknown'
                
            if model_name.startswith(prefix):
                parts = model_name[len(prefix):].split('_')
                # Remove 'basic' or 'optuna' suffix
                if parts and (parts[-1] == 'basic' or parts[-1] == 'optuna'):
                    dataset = '_'.join(parts[:-1])
                else:
                    dataset = '_'.join(parts)
                    
                print(f"  Tree model: Extracted dataset '{dataset}' from {model_name}")
                return dataset
        
        # If we couldn't extract the dataset, print more detailed debug info and return Unknown
        print(f"⚠️ Warning: Could not extract dataset from {original_model_name} with family type {self.family_type}")
        print(f"  Model name parts: {original_model_name.split('_')}")
        return 'Unknown'


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get model information from model name.
    
    Args:
        model_name: Model name
        
    Returns:
        Dict[str, Any]: Dictionary with model information
    """
    family = ModelFamily.from_model_name(model_name)
    
    return {
        'model_name': model_name,
        'family': family.name,
        'family_type': family.family_type,
        'color': family.color,
        'short_name': family.short_name,
        'is_tuned': family.is_tuned_variant(model_name),
        'variant': family.get_variant_name(family.is_tuned_variant(model_name)),
        'dataset': family.get_dataset_from_model_name(model_name)
    }