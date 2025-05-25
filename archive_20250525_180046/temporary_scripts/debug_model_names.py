#!/usr/bin/env python3
"""Debug model names to see why they show as Unknown."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import required modules
from utils.io import load_all_models
from visualization_new.core.registry import get_adapter_for_model

def main():
    """Debug model names."""
    print("Loading all models...")
    all_models = load_all_models()
    
    print(f"\nFound {len(all_models)} models\n")
    
    # Check each model
    for model_name, model_data in all_models.items():
        print(f"\nModel key: {model_name}")
        print(f"  model_data type: {type(model_data)}")
        
        # Check what's in model_data
        if isinstance(model_data, dict):
            print(f"  Keys in model_data: {list(model_data.keys())}")
            print(f"  model_name in dict: {model_data.get('model_name', 'NOT FOUND')}")
            print(f"  model_type in dict: {model_data.get('model_type', 'NOT FOUND')}")
        
        # Create adapter and check metadata
        try:
            adapter = get_adapter_for_model(model_data)
            metadata = adapter.get_metadata()
            print(f"  Adapter metadata:")
            print(f"    model_name: {metadata.get('model_name', 'NOT FOUND')}")
            print(f"    model_type: {metadata.get('model_type', 'NOT FOUND')}")
        except Exception as e:
            print(f"  Error creating adapter: {e}")

if __name__ == "__main__":
    main()