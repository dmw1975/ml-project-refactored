"""Script to check feature names in models and pickle files."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from utils import io
from config import settings

def check_feature_names():
    """Check feature names in original_feature_names.pkl."""
    try:
        names = io.load_model('original_feature_names.pkl', settings.DATA_DIR / 'pkl')
        print(f'Loaded {len(names)} feature names from pickle file')
        print(f'First few names: {names[:5]}')
        
        # Check for generic names
        generic_count = sum(1 for name in names if isinstance(name, str) and name.startswith('feature_') and name[8:].isdigit())
        if generic_count > 0:
            print(f"Warning: Found {generic_count} generic feature names in pickle file")
        else:
            print("All feature names are descriptive (non-generic)")
            
        return names
    except Exception as e:
        print(f'Error loading original feature names: {e}')
        return None

def check_feature_importance_files():
    """Check feature importance files."""
    # Check if LightGBM feature importance files exist
    feature_imp_dir = settings.FEATURE_IMPORTANCE_DIR
    if not feature_imp_dir.exists():
        print(f"Feature importance directory does not exist: {feature_imp_dir}")
        return
        
    # List all LightGBM feature importance files
    lgbm_files = list(feature_imp_dir.glob("LightGBM*_importance.csv"))
    print(f"Found {len(lgbm_files)} LightGBM feature importance files")
    
    # Check content of first file if any exist
    if lgbm_files:
        import pandas as pd
        try:
            df = pd.read_csv(lgbm_files[0])
            print(f"Columns in {lgbm_files[0].name}: {df.columns.tolist()}")
            if 'Feature' in df.columns:
                print(f"Sample feature names: {df['Feature'].head(5).tolist()}")
                
                # Check for generic names
                generic_count = sum(1 for name in df['Feature'] if isinstance(name, str) and str(name).startswith('feature_') and str(name)[8:].isdigit())
                if generic_count > 0:
                    print(f"Warning: Found {generic_count} generic feature names in importance file")
                else:
                    print("All feature names in importance file are descriptive (non-generic)")
        except Exception as e:
            print(f"Error reading importance file: {e}")

def check_model_feature_names():
    """Check feature names in model objects."""
    # Load LightGBM models
    try:
        lightgbm_models = io.load_model("lightgbm_models.pkl", settings.MODEL_DIR)
        print(f"Loaded {len(lightgbm_models)} LightGBM models")
        
        # Check first model
        model_name = next(iter(lightgbm_models))
        model_data = lightgbm_models[model_name]
        
        # Check for feature information
        keys_to_check = ['feature_names', 'original_feature_names', 'feature_name_mapping', 'X_test_clean']
        for key in keys_to_check:
            if key in model_data:
                if key == 'X_test_clean' and hasattr(model_data[key], 'columns'):
                    cols = model_data[key].columns.tolist()
                    print(f"X_test_clean columns (first 5): {cols[:5]}")
                    
                    # Check for generic names
                    sample_cols = model_data[key].columns[:10]
                    if all(col.startswith('feature_') and col[8:].isdigit() for col in sample_cols):
                        print("Warning: X_test has generic feature names")
                    else:
                        print("X_test has descriptive column names")
                else:
                    value = model_data[key]
                    if isinstance(value, list) or hasattr(value, '__iter__'):
                        try:
                            print(f"{key} (first 5): {list(value)[:5]}")
                        except:
                            print(f"{key} exists but couldn't show sample")
                    else:
                        print(f"{key} exists but is not iterable")
            else:
                print(f"{key} not found in model data")
                
        # Check model object feature name methods
        model = model_data.get('model')
        if model is not None:
            if hasattr(model, 'feature_name'):
                try:
                    feature_names = model.feature_name()
                    print(f"From model.feature_name() (first 5): {feature_names[:5]}")
                except Exception as e:
                    print(f"Error calling model.feature_name(): {e}")
            else:
                print("Model does not have feature_name() method")
            
            if hasattr(model, 'feature_names_'):
                print(f"From model.feature_names_ (first 5): {model.feature_names_[:5]}")
            else:
                print("Model does not have feature_names_ attribute")
    except Exception as e:
        print(f"Error checking model feature names: {e}")

if __name__ == "__main__":
    print("Checking feature names in pickle file...")
    pickle_names = check_feature_names()
    
    print("\nChecking feature importance files...")
    check_feature_importance_files()
    
    print("\nChecking feature names in model objects...")
    check_model_feature_names()