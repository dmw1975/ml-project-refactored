#!/usr/bin/env python3
"""Fix CatBoost feature importance column names."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils import io
from config import settings

def main():
    """Fix CatBoost feature importance column names."""
    # Load CatBoost models
    models = io.load_model('catboost_models.pkl', settings.MODEL_DIR)
    
    print("Fixing CatBoost feature importance column names...")
    
    updated = False
    for model_name, model_data in models.items():
        if 'feature_importance' in model_data:
            fi = model_data['feature_importance']
            
            # Check if columns need renaming
            if 'feature' in fi.columns and 'Feature' not in fi.columns:
                # Rename columns to match expected format
                fi.rename(columns={'feature': 'Feature', 'importance': 'Importance'}, inplace=True)
                
                # Add Std column if missing
                if 'Std' not in fi.columns:
                    fi['Std'] = 0.0
                
                print(f"  Fixed {model_name}")
                updated = True
    
    if updated:
        # Save the updated models
        io.save_model(models, 'catboost_models.pkl', settings.MODEL_DIR)
        print("\nSaved updated CatBoost models.")
    else:
        print("\nNo updates needed.")

if __name__ == "__main__":
    main()