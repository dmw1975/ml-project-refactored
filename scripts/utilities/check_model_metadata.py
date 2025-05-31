"""Check model metadata in pickle files to debug model name issues."""

import pickle
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.config.settings import MODEL_DIR


def check_model_metadata():
    """Check metadata for all saved models."""
    
    # Model files to check
    model_files = [
        'lightgbm_models.pkl',
        'xgboost_models.pkl',
        'catboost_models.pkl',
        'linear_regression_models.pkl',
        'elasticnet_models.pkl'
    ]
    
    print("Checking model metadata in pickle files...\n")
    
    for model_file in model_files:
        file_path = MODEL_DIR / model_file
        
        if not file_path.exists():
            print(f"Warning: {model_file} not found")
            continue
            
        print(f"\n{'='*60}")
        print(f"Checking: {model_file}")
        print(f"{'='*60}")
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            if isinstance(data, dict):
                print(f"Found {len(data)} models in file")
                
                for key, model_data in data.items():
                    print(f"\nModel key: {key}")
                    
                    # Check structure
                    if isinstance(model_data, dict):
                        print(f"  Model data type: dict with keys: {list(model_data.keys())}")
                        
                        # Check for model_name in different places
                        if 'model_name' in model_data:
                            print(f"  model_name (top level): {model_data['model_name']}")
                        
                        if 'metadata' in model_data:
                            metadata = model_data['metadata']
                            print(f"  metadata keys: {list(metadata.keys()) if isinstance(metadata, dict) else 'Not a dict'}")
                            if isinstance(metadata, dict) and 'model_name' in metadata:
                                print(f"  model_name (in metadata): {metadata['model_name']}")
                        
                        # Check for model object
                        if 'model' in model_data:
                            model_obj = model_data['model']
                            print(f"  model object type: {type(model_obj).__name__}")
                            
                            # Check if model has model_name attribute
                            if hasattr(model_obj, 'model_name'):
                                print(f"  model.model_name: {model_obj.model_name}")
                    else:
                        print(f"  Model data type: {type(model_data).__name__}")
                        
            else:
                print(f"File contains: {type(data).__name__}")
                
        except Exception as e:
            print(f"Error loading {model_file}: {e}")


if __name__ == "__main__":
    check_model_metadata()