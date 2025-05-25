"""
Check if ElasticNet models have Optuna study objects.

This script loads the elasticnet_models.pkl file and checks
if the models have study objects, which would be required
for generating contour plots.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings
from utils import io

def check_elasticnet_models():
    """Check if elasticnet models have study objects."""
    print("Loading elasticnet models...")
    
    # Path to models directory
    models_dir = settings.MODEL_DIR
    
    # Try to load elasticnet models
    try:
        elasticnet_models = io.load_model("elasticnet_models.pkl", models_dir)
        print(f"Loaded ElasticNet models: {len(elasticnet_models) if elasticnet_models else 0}")
        
        # Check if models have study objects
        models_with_study = 0
        if elasticnet_models:
            print("\nChecking for study objects in ElasticNet models:")
            for model_name, model_data in elasticnet_models.items():
                has_study = 'study' in model_data and model_data['study'] is not None
                print(f"  {model_name}: {'Has study object' if has_study else 'No study object'}")
                if has_study:
                    models_with_study += 1
            
            print(f"\nTotal ElasticNet models with study objects: {models_with_study} / {len(elasticnet_models)}")
            
            # Print all keys in the first model data
            first_model = next(iter(elasticnet_models.values()))
            print("\nKeys in ElasticNet model data:")
            for key in first_model.keys():
                print(f"  - {key}")
        else:
            print("No ElasticNet models found.")
            
    except Exception as e:
        print(f"Error loading ElasticNet models: {e}")

if __name__ == "__main__":
    check_elasticnet_models()