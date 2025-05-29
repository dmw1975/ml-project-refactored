"""Registry for model adapters."""

from typing import Dict, Any, Type, Optional, List, Union
import importlib
import re

from visualization_new.core.interfaces import ModelData
from visualization_new.core.model_family import ModelFamilyType, ModelFamily, get_model_info

# Registry of model types to adapter classes
_ADAPTER_REGISTRY = {}

def register_adapter(model_type: str, adapter_class: Type[ModelData]) -> None:
    """
    Register adapter class for a model type.
    
    Args:
        model_type: Model type identifier
        adapter_class: Adapter class for the model type
    """
    _ADAPTER_REGISTRY[model_type.lower()] = adapter_class
    print(f"Registered adapter {adapter_class.__name__} for model type {model_type}")

def get_adapter_for_model(model_data: Dict[str, Any]) -> ModelData:
    """
    Get adapter for model data.
    
    Args:
        model_data: Model data dictionary or adapter instance
        
    Returns:
        ModelData: Adapter instance for the model data
        
    Raises:
        ValueError: If no adapter is found for the model type
    """
    # If it's already an adapter, return it
    if hasattr(model_data, 'get_model_type'):
        return model_data
        
    # Try to determine model type
    model_type = None
    
    # Check model_type in model data
    if 'model_type' in model_data:
        model_type = model_data['model_type']
    
    # Check model_name in model data
    elif 'model_name' in model_data:
        name = model_data['model_name'].lower()
        
        # Check for known model types in name
        if 'xgb' in name or 'xgboost' in name:
            model_type = 'xgboost'
        elif 'lightgbm' in name or 'lgbm' in name:
            model_type = 'lightgbm'
        elif 'catboost' in name:
            model_type = 'catboost'
        elif 'elasticnet' in name:
            model_type = 'elasticnet'
        elif 'lr_' in name:
            model_type = 'linearregression'
            
        # Print model type for debugging
        print(f"Detected model type: {model_type} from name: {name}")
    
    # Check model class
    elif 'model' in model_data:
        model = model_data['model']
        class_name = model.__class__.__name__.lower()
        
        if 'xgb' in class_name:
            model_type = 'xgboost'
        elif 'lightgbm' in class_name or 'lgbm' in class_name:
            model_type = 'lightgbm'
        elif 'catboost' in class_name:
            model_type = 'catboost'
        elif 'elasticnet' in class_name:
            model_type = 'elasticnet'
        elif 'linearregression' in class_name:
            model_type = 'linear_regression'
    
    if model_type is None:
        raise ValueError(f"Could not determine model type from model data")
    
    # Convert to lowercase and extract base model type if needed
    model_type = model_type.lower()
    
    # If model_type contains spaces or underscores, extract the first part
    # (e.g., "LightGBM Optuna" -> "lightgbm")
    if ' ' in model_type:
        model_type = model_type.split()[0]
    
    # Check if adapter exists for model type
    if model_type in _ADAPTER_REGISTRY:
        adapter_class = _ADAPTER_REGISTRY[model_type]
        return adapter_class(model_data)
    
    # Try to dynamically import adapter
    try:
        module_name = f"visualization_new.adapters.{model_type}_adapter"
        module = importlib.import_module(module_name)
        
        # Find adapter class in module
        for attr_name in dir(module):
            if attr_name.lower().endswith('adapter') and attr_name.lower().startswith(model_type):
                adapter_class = getattr(module, attr_name)
                
                # Register adapter for future use
                register_adapter(model_type, adapter_class)
                
                return adapter_class(model_data)
    except (ImportError, AttributeError):
        pass
    
    raise ValueError(f"No adapter found for model type {model_type}")

def load_model(model_name: str) -> ModelData:
    """
    Load model by name and create adapter.
    
    Args:
        model_name: Model name
        
    Returns:
        ModelData: Adapter instance for the model
        
    Raises:
        ValueError: If model cannot be loaded
    """
    # Import utils from main project
    from pathlib import Path
    import sys
    
    # Add project root to path if needed
    project_root = Path(__file__).parent.parent.parent.absolute()
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
        
    # Import io and settings
    from utils import io
    from config import settings
    
    # Try to determine model type from name
    model_type = None
    if 'xgb' in model_name.lower() or 'xgboost' in model_name.lower():
        model_type = 'xgboost'
    elif 'lightgbm' in model_name.lower() or 'lgbm' in model_name.lower():
        model_type = 'lightgbm'
    elif 'catboost' in model_name.lower():
        model_type = 'catboost'
    elif 'elasticnet' in model_name.lower():
        model_type = 'elasticnet'
    elif model_name.lower().startswith('lr_'):
        model_type = 'linear_regression'
    
    if model_type is None:
        raise ValueError(f"Could not determine model type from name: {model_name}")
    
    # Try to load model from file
    try:
        # Construct filename based on model type
        filename_map = {
            'xgboost': 'xgboost_models.pkl',
            'lightgbm': 'lightgbm_models.pkl',
            'catboost': 'catboost_models.pkl',
            'elasticnet': 'elasticnet_models.pkl',
            'linear_regression': 'linear_regression_models.pkl'
        }
        
        filename = filename_map.get(model_type)
        if filename:
            # Load models
            models = io.load_model(filename, settings.MODEL_DIR)
            
            # Find model with matching name
            for name, model_data in models.items():
                if name == model_name:
                    return get_adapter_for_model(model_data)
            
            # If not found, try case-insensitive match
            for name, model_data in models.items():
                if name.lower() == model_name.lower():
                    return get_adapter_for_model(model_data)
    except Exception as e:
        raise ValueError(f"Error loading model {model_name}: {e}")
    
    raise ValueError(f"Model {model_name} not found")

# Auto-register adapters
def _auto_register_adapters():
    """Auto-register all available adapters."""
    import pkgutil
    import importlib
    import visualization_new.adapters
    
    # Import all adapter modules
    for _, name, _ in pkgutil.iter_modules(visualization_new.adapters.__path__, 'visualization_new.adapters.'):
        try:
            module = importlib.import_module(name)
            
            # Find adapter classes in module
            for attr_name in dir(module):
                if attr_name.endswith('Adapter'):
                    adapter_class = getattr(module, attr_name)
                    
                    # Extract model type from adapter name
                    match = re.match(r'(.+)Adapter', attr_name)
                    if match:
                        model_type = match.group(1).lower()
                        register_adapter(model_type, adapter_class)
        except ImportError:
            pass

# Register adapters when module is imported
_auto_register_adapters()