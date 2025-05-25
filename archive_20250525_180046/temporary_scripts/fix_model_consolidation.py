#!/usr/bin/env python3
"""Fix the model consolidation to properly save model dictionaries."""

import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import settings


def fix_consolidation():
    """Re-consolidate models properly from backup."""
    model_dir = settings.MODEL_DIR
    backup_dir = model_dir / 'backup_20250524_140301'
    
    # First, fix linear regression models
    lr_file = model_dir / 'linear_regression_models.pkl'
    if lr_file.exists():
        print("Fixing Linear Regression models...")
        with open(lr_file, 'rb') as f:
            models = pickle.load(f)
        
        for name, model_data in models.items():
            if isinstance(model_data, dict):
                model_data['model_type'] = 'linearregression'
        
        with open(lr_file, 'wb') as f:
            pickle.dump(models, f)
        print("✓ Fixed Linear Regression models")
    
    # Fix ElasticNet models
    en_file = model_dir / 'elasticnet_models.pkl'
    if en_file.exists():
        print("Fixing ElasticNet models...")
        with open(en_file, 'rb') as f:
            models = pickle.load(f)
        
        for name, model_data in models.items():
            if isinstance(model_data, dict):
                model_data['model_type'] = 'elasticnet'
        
        with open(en_file, 'wb') as f:
            pickle.dump(models, f)
        print("✓ Fixed ElasticNet models")
    
    # Now we need to properly re-consolidate XGBoost and LightGBM from backup
    # Check if we have the proper backup files
    if backup_dir.exists():
        print(f"\nUsing backup from {backup_dir}")
        
        # Re-consolidate XGBoost
        xgb_backup = backup_dir / 'xgboost_models.pkl'
        if xgb_backup.exists():
            print("Re-consolidating XGBoost models from backup...")
            with open(xgb_backup, 'rb') as f:
                xgb_models = pickle.load(f)
            
            # Save to main directory
            with open(model_dir / 'xgboost_models.pkl', 'wb') as f:
                pickle.dump(xgb_models, f)
            print(f"✓ Restored {len(xgb_models)} XGBoost models from backup")
        
        # Re-consolidate LightGBM 
        lgb_backup = backup_dir / 'lightgbm_models.pkl'
        if lgb_backup.exists():
            print("Re-consolidating LightGBM models from backup...")
            with open(lgb_backup, 'rb') as f:
                lgb_models = pickle.load(f)
            
            # Save to main directory
            with open(model_dir / 'lightgbm_models.pkl', 'wb') as f:
                pickle.dump(lgb_models, f)
            print(f"✓ Restored {len(lgb_models)} LightGBM models from backup")
    else:
        print("\n⚠️  No backup found. Attempting to reconstruct from enhanced model outputs...")
        
        # Try to load from enhanced model runs
        from utils.io import load_model
        
        # Reconstruct XGBoost models
        print("\nReconstructing XGBoost models...")
        xgb_models = {}
        
        # Load old non-categorical models if they exist
        try:
            xgb_backup = load_model('xgboost_models.pkl', backup_dir)
            for name, data in xgb_backup.items():
                if isinstance(data, dict) and 'model' in data:
                    xgb_models[name] = data
        except:
            pass
        
        # Try to load the categorical models from their training outputs
        # This requires running the training again or finding the output files
        print("⚠️  Cannot reconstruct categorical models without re-training")
        print("    Please run: python main.py --train-xgboost")
        
    print("\n✅ Model consolidation fixes completed!")
    print("\nNext steps:")
    print("1. If models are still missing, re-run training:")
    print("   python main.py --train-xgboost --train-lightgbm --train-catboost")
    print("2. Then run the visualization test again")


if __name__ == "__main__":
    fix_consolidation()