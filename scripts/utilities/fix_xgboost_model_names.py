#!/usr/bin/env python3
"""Fix XGBoost model names to remove duplication."""

import pickle
import sys
from pathlib import Path
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def fix_xgboost_model_names():
    """Fix duplicated XGBoost_ prefix in model names."""
    
    # Load XGBoost models
    xgboost_path = project_root / 'outputs' / 'models' / 'xgboost_models.pkl'
    
    print(f"Loading XGBoost models from: {xgboost_path}")
    
    try:
        # Create backup first
        backup_path = xgboost_path.with_suffix('.pkl.bak')
        shutil.copy2(xgboost_path, backup_path)
        print(f"Created backup at: {backup_path}")
        
        with open(xgboost_path, 'rb') as f:
            xgboost_models = pickle.load(f)
        
        print(f"\nFound {len(xgboost_models)} XGBoost models")
        
        # Fix model names
        models_fixed = False
        
        for i, (key, model_data) in enumerate(xgboost_models.items()):
            print(f"\n{i+1}. Key: {key}")
            
            if isinstance(model_data, dict) and 'model_name' in model_data:
                old_name = model_data['model_name']
                print(f"   Current model_name: {old_name}")
                
                # Fix duplicated XGBoost_ prefix
                if old_name.startswith("XGBoost_XGBoost_"):
                    new_name = old_name.replace("XGBoost_XGBoost_", "XGBoost_", 1)
                    model_data['model_name'] = new_name
                    models_fixed = True
                    print(f"   Fixed to: {new_name}")
                elif not old_name.startswith("XGBoost_") and key.startswith("XGBoost_"):
                    # Use the key as model name if it already has proper prefix
                    model_data['model_name'] = key
                    models_fixed = True
                    print(f"   Fixed to: {key}")
                else:
                    print(f"   No fix needed")
        
        # Save fixed models
        if models_fixed:
            print("\nSaving fixed XGBoost models...")
            with open(xgboost_path, 'wb') as f:
                pickle.dump(xgboost_models, f)
            print("Models saved successfully!")
            
            # Clean up old residual plots with wrong names
            residuals_dir = project_root / 'outputs' / 'visualization' / 'residuals' / 'xgboost'
            if residuals_dir.exists():
                print("\nCleaning up old residual plots...")
                for plot_file in residuals_dir.glob("XGBoost_XGBoost_*.png"):
                    print(f"   Removing: {plot_file.name}")
                    plot_file.unlink()
        else:
            print("\nAll model names are correct, no fixes needed.")
            
        return xgboost_models
        
    except Exception as e:
        print(f"Error fixing XGBoost models: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    fix_xgboost_model_names()