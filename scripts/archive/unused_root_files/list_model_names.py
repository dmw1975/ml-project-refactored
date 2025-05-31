#!/usr/bin/env python3
"""List all available model names."""

from src.visualization.utils.io import load_all_models

def list_models():
    """List all available models."""
    all_models = load_all_models()
    
    for model_type, models_dict in all_models.items():
        print(f"\n{model_type}:")
        # Check if models_dict is actually a dict of models
        if isinstance(models_dict, dict) and len(models_dict) > 0:
            first_key = next(iter(models_dict))
            first_value = models_dict[first_key]
            
            # If the value is a dict with model data, extract model names
            if isinstance(first_value, dict) and 'model_name' in first_value:
                # This is a dict of model data
                for key, model_data in models_dict.items():
                    if 'model_name' in model_data:
                        print(f"  - {model_data['model_name']}")
            else:
                # This looks like it might be a single model's data
                if 'model_name' in models_dict:
                    print(f"  - {models_dict['model_name']}")
                else:
                    # Just list the keys
                    for key in sorted(models_dict.keys()):
                        print(f"  - Key: {key}")

if __name__ == "__main__":
    list_models()