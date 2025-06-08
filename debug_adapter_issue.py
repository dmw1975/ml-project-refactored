#!/usr/bin/env python3
"""Debug adapter issue for dataset comparison."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.io import load_all_models
from src.visualization.core.registry import get_adapter_for_model

# Load models
all_models = load_all_models()

# Check first few models
for i, (model_name, model_data) in enumerate(all_models.items()):
    if i >= 5:  # Check first 5
        break
    
    print(f"\nModel: {model_name}")
    print(f"  Keys: {list(model_data.keys())[:5]}...")  # First 5 keys
    
    # Check what get_adapter_for_model expects
    has_model_name = 'model_name' in model_data
    has_model = 'model' in model_data
    
    print(f"  Has 'model_name': {has_model_name}")
    print(f"  Has 'model': {has_model}")
    
    if has_model:
        print(f"  Model class: {model_data['model'].__class__.__name__}")
    
    # Try to get adapter
    try:
        adapter = get_adapter_for_model(model_data)
        print(f"  ✓ Adapter: {adapter.__class__.__name__}")
    except Exception as e:
        print(f"  ✗ Error: {e}")