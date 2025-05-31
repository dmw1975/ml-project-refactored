#!/usr/bin/env python3
"""Debug load_all_models structure."""

from visualization_new.utils.io import load_all_models

def debug_structure():
    """Debug the structure returned by load_all_models."""
    all_models = load_all_models()
    
    print(f"Type of all_models: {type(all_models)}")
    print(f"Keys in all_models: {list(all_models.keys())}")
    
    # Check first key
    if all_models:
        first_key = list(all_models.keys())[0]
        print(f"\nFirst key: {first_key}")
        first_value = all_models[first_key]
        print(f"Type of first value: {type(first_value)}")
        
        if isinstance(first_value, dict):
            print(f"Keys in first value: {list(first_value.keys())[:5]}")  # Show first 5

if __name__ == "__main__":
    debug_structure()