#!/usr/bin/env python
"""
Script to debug the structure of model files.
This will print out the keys and structure of the model data.
"""
import sys
from pathlib import Path
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from config import settings
from utils.io import load_model

def print_model_structure(model_file, max_depth=3):
    """Print the structure of a model file."""
    try:
        print(f"\n{'='*80}")
        print(f"Examining model file: {model_file}")
        print(f"{'='*80}")
        
        # Load the model data
        model_data = load_model(Path(model_file).name, settings.MODEL_DIR)
        
        # Check if it's a dictionary of models
        if isinstance(model_data, dict):
            print(f"Found a dictionary of {len(model_data)} models")
            
            # Get a sample model
            sample_name = next(iter(model_data))
            sample = model_data[sample_name]
            
            print(f"\nSample model: {sample_name}")
            print(f"  Top-level keys: {list(sample.keys())}")
            
            # Print structure recursively
            def print_structure(obj, prefix='', depth=0):
                if depth > max_depth:
                    return
                
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        print(f"{prefix}{key}:")
                        print_structure(value, prefix + '  ', depth + 1)
                elif isinstance(obj, list):
                    print(f"{prefix}List with {len(obj)} elements")
                    if obj and depth < max_depth:
                        print_structure(obj[0], prefix + '  ', depth + 1)
                else:
                    type_str = type(obj).__name__
                    print(f"{prefix}Type: {type_str}")
                    
                    # For dataframes, show columns
                    if type_str == 'DataFrame':
                        print(f"{prefix}  Shape: {obj.shape}")
                        print(f"{prefix}  Columns: {list(obj.columns)}")
            
            # Print structure of the sample model
            print("\nDetailed structure:")
            print_structure(sample)
            
            # Examine all models for test indices or similar fields
            indices_fields = ['test_indices', 'test_index', 'indices', 'test_idx', 'X_test', 'y_test', 'train_test_split', 'split']
            for model_name, model in model_data.items():
                for field in indices_fields:
                    if field in model:
                        field_type = type(model[field]).__name__
                        print(f"\nModel {model_name} has field '{field}' of type {field_type}")
                        if field_type == 'DataFrame':
                            print(f"  Shape: {model[field].shape}")
                            print(f"  Columns: {list(model[field].columns)}")
                        elif field_type == 'ndarray':
                            print(f"  Shape: {model[field].shape}")
                            print(f"  Sample: {model[field][:5] if len(model[field]) > 5 else model[field]}")
                        elif field_type == 'dict':
                            print(f"  Keys: {list(model[field].keys())}")
            
        else:
            print(f"Model data is not a dictionary, it's a {type(model_data)}")
            
    except Exception as e:
        print(f"Error examining {model_file}: {e}")

def main():
    """Main entry point."""
    # Check XGBoost models
    print_model_structure("xgboost_models.pkl")
    
    # Check LightGBM models
    print_model_structure("lightgbm_models.pkl")
    
    # Check ElasticNet models
    print_model_structure("elasticnet_models.pkl")

if __name__ == "__main__":
    main()