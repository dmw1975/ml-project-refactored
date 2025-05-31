#!/usr/bin/env python3
"""Check the contents of train_test_split.pkl file."""

import pickle
from pathlib import Path
import pandas as pd

# Path to the pickle file
pkl_path = Path("/mnt/d/ml_project_refactored/data/processed/unified/train_test_split.pkl")

if pkl_path.exists():
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print("Contents of train_test_split.pkl:")
    print(f"Type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        
        # Check each key
        for key, value in data.items():
            print(f"\n{key}:")
            print(f"  Type: {type(value)}")
            if hasattr(value, 'shape'):
                print(f"  Shape: {value.shape}")
            if hasattr(value, 'columns'):
                print(f"  Columns: {list(value.columns)[:5]}...")  # Show first 5 columns
                
                # Check for gics_sector
                if 'gics_sector' in value.columns:
                    print("\n  Sector distribution:")
                    sector_dist = value['gics_sector'].value_counts(normalize=True).sort_index()
                    for sector, pct in sector_dist.items():
                        print(f"    {sector}: {pct:.4f}")
else:
    print(f"File not found: {pkl_path}")