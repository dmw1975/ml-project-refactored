"""Debug script for LightGBM feature visualization."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from visualization_new.utils.io import load_all_models
from visualization_new.core.registry import get_adapter_for_model, register_adapter
from visualization_new.plots.features import plot_feature_importance
from visualization_new.adapters.lightgbm_adapter import LightGBMAdapter

def debug_adapter():
    """Debug LightGBM adapter registration and model loading."""
    print("Loading all models...")
    models = load_all_models()
    
    # Print available adapters
    print("\nRegistered adapters:")
    from visualization_new.core.registry import _ADAPTER_REGISTRY
    for model_type, adapter_class in _ADAPTER_REGISTRY.items():
        print(f"  - {model_type}: {adapter_class.__name__}")
    
    # Print LightGBM model keys
    lightgbm_models = [k for k in models.keys() if 'LightGBM' in k]
    print(f"\nFound {len(lightgbm_models)} LightGBM models: {lightgbm_models}")
    
    # Try to get adapter for LightGBM model
    model_name = 'LightGBM_Base_optuna'
    print(f"\nTrying to get adapter for {model_name}...")
    
    model_data = models[model_name]
    print(f"Model data keys: {model_data.keys()}")
    print(f"Model name: {model_data.get('model_name')}")
    
    # Try to determine model type
    if 'model_type' in model_data:
        model_type = model_data['model_type']
        print(f"Model type from model_data: {model_type}")
    else:
        name = model_data['model_name'].lower()
        if 'lightgbm' in name:
            model_type = 'lightgbm'
            print(f"Detected model type from name: {model_type}")
    
    # Force register adapter if needed
    if 'lightgbm' not in _ADAPTER_REGISTRY:
        print("LightGBM adapter not in registry! Registering manually...")
        register_adapter('lightgbm', LightGBMAdapter)
    
    # Try to get adapter with explicit model type
    model_data['model_type'] = 'lightgbm'
    try:
        adapter = get_adapter_for_model(model_data)
        print(f"Successfully created adapter: {adapter.__class__.__name__}")
        
        # Try to get feature importance
        importance_df = adapter.get_feature_importance()
        print(f"Feature importance shape: {importance_df.shape}")
        print("Top 5 features:")
        print(importance_df.head(5))
        
        # Try to create plot
        fig = plot_feature_importance(adapter)
        print("Successfully created feature importance plot!")
        
    except Exception as e:
        print(f"Error creating adapter: {e}")

if __name__ == "__main__":
    debug_adapter()