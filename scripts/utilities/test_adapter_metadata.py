"""Test adapter metadata retrieval to debug model names."""

import pickle
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.config.settings import MODEL_DIR
from src.visualization.core.registry import get_adapter_for_model


def test_adapter_metadata():
    """Test metadata retrieval through adapters."""
    
    # Test LightGBM models
    print("Testing LightGBM adapter...")
    print("="*60)
    
    lightgbm_path = MODEL_DIR / 'lightgbm_models.pkl'
    with open(lightgbm_path, 'rb') as f:
        lightgbm_models = pickle.load(f)
    
    # Test first model
    model_key = 'LightGBM_Base_categorical_basic'
    model_data = lightgbm_models[model_key]
    
    print(f"\nModel key: {model_key}")
    print(f"Model data has model_name: {'model_name' in model_data}")
    print(f"Model name in data: {model_data.get('model_name', 'NOT FOUND')}")
    
    # Create adapter
    adapter = get_adapter_for_model(model_data)
    print(f"\nAdapter type: {type(adapter).__name__}")
    
    # Get metadata
    metadata = adapter.get_metadata()
    print(f"Metadata: {metadata}")
    print(f"Model name from metadata: {metadata.get('model_name', 'NOT FOUND')}")
    
    # Test XGBoost models
    print("\n\nTesting XGBoost adapter...")
    print("="*60)
    
    xgboost_path = MODEL_DIR / 'xgboost_models.pkl'
    with open(xgboost_path, 'rb') as f:
        xgboost_models = pickle.load(f)
    
    # Test first model
    model_key = 'XGBoost_Base_categorical_basic'
    model_data = xgboost_models[model_key]
    
    print(f"\nModel key: {model_key}")
    print(f"Model data has model_name: {'model_name' in model_data}")
    print(f"Model name in data: {model_data.get('model_name', 'NOT FOUND')}")
    
    # Create adapter
    adapter = get_adapter_for_model(model_data)
    print(f"\nAdapter type: {type(adapter).__name__}")
    
    # Get metadata
    metadata = adapter.get_metadata()
    print(f"Metadata: {metadata}")
    print(f"Model name from metadata: {metadata.get('model_name', 'NOT FOUND')}")
    
    # Test ElasticNet models
    print("\n\nTesting ElasticNet adapter...")
    print("="*60)
    
    elasticnet_path = MODEL_DIR / 'elasticnet_models.pkl'
    with open(elasticnet_path, 'rb') as f:
        elasticnet_models = pickle.load(f)
    
    # Test first model
    model_key = 'ElasticNet_LR_Base_optuna'
    model_data = elasticnet_models[model_key]
    
    print(f"\nModel key: {model_key}")
    print(f"Model data has model_name: {'model_name' in model_data}")
    print(f"Model name in data: {model_data.get('model_name', 'NOT FOUND')}")
    
    # Create adapter
    adapter = get_adapter_for_model(model_data)
    print(f"\nAdapter type: {type(adapter).__name__}")
    
    # Get metadata
    metadata = adapter.get_metadata()
    print(f"Metadata: {metadata}")
    print(f"Model name from metadata: {metadata.get('model_name', 'NOT FOUND')}")


if __name__ == "__main__":
    test_adapter_metadata()