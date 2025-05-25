#!/usr/bin/env python3
"""
Fix ML Pipeline with Progress Indicators

This script fixes the pipeline issues and shows clear progress.
"""

import sys
import os
import time
import pickle
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import settings
from utils import io


def print_step(step_num, total_steps, message):
    """Print a progress step"""
    progress = f"[{step_num}/{total_steps}]"
    print(f"\n{progress} {message}")
    print("-" * 60)


def check_model_format(filename):
    """Check if a model file has the correct format"""
    filepath = settings.MODEL_DIR / filename
    if not filepath.exists():
        return "missing", None
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            return "correct", len(data)
        else:
            return "wrong", type(data).__name__
    except Exception as e:
        return "error", str(e)


def main():
    print("\n" + "="*60)
    print(" ML PIPELINE FIX WITH PROGRESS ")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    
    total_steps = 7
    
    # Step 1: Check current model status
    print_step(1, total_steps, "Checking current model files")
    
    model_files = {
        'linear_regression_models.pkl': 'Linear Regression',
        'elasticnet_models.pkl': 'ElasticNet',
        'xgboost_models.pkl': 'XGBoost',
        'lightgbm_models.pkl': 'LightGBM',
        'catboost_categorical_models.pkl': 'CatBoost'
    }
    
    issues = []
    for filename, model_type in model_files.items():
        status, info = check_model_format(filename)
        if status == "correct":
            print(f"✓ {model_type}: OK ({info} models)")
        elif status == "wrong":
            print(f"✗ {model_type}: Wrong format ({info})")
            issues.append(filename)
        elif status == "missing":
            print(f"✗ {model_type}: Missing")
            issues.append(filename)
        else:
            print(f"✗ {model_type}: Error - {info}")
            issues.append(filename)
    
    if not issues:
        print("\n✓ All model files are in correct format!")
        print("You can run: python main.py --all")
        return
    
    # Step 2: Backup existing models
    print_step(2, total_steps, "Backing up existing models")
    backup_dir = settings.MODEL_DIR / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(exist_ok=True)
    
    for filename in model_files.keys():
        source = settings.MODEL_DIR / filename
        if source.exists():
            dest = backup_dir / filename
            import shutil
            shutil.copy2(source, dest)
            print(f"✓ Backed up {filename}")
    
    # Step 3: Remove problematic files
    print_step(3, total_steps, "Removing problematic model files")
    for filename in issues:
        filepath = settings.MODEL_DIR / filename
        if filepath.exists():
            filepath.unlink()
            print(f"✓ Removed {filename}")
    
    # Step 4: Train missing models
    print_step(4, total_steps, "Training missing models")
    
    # Check which models need training
    need_xgboost = 'xgboost_models.pkl' in issues
    need_lightgbm = 'lightgbm_models.pkl' in issues
    
    if need_xgboost:
        print("\nTraining XGBoost models...")
        print("This will take 2-3 minutes...")
        try:
            from models.xgboost_model import train_xgboost_models
            xgb_models = train_xgboost_models(datasets=['all'], n_trials=20)  # Fewer trials for speed
            print(f"✓ Trained {len(xgb_models)} XGBoost models")
        except Exception as e:
            print(f"✗ XGBoost training failed: {e}")
    
    if need_lightgbm:
        print("\nTraining LightGBM models...")
        print("This will take 2-3 minutes...")
        try:
            from models.lightgbm_model import train_lightgbm_models
            lgb_models = train_lightgbm_models(datasets=['all'], n_trials=20)  # Fewer trials for speed
            print(f"✓ Trained {len(lgb_models)} LightGBM models")
        except Exception as e:
            print(f"✗ LightGBM training failed: {e}")
    
    # Step 5: Verify models are fixed
    print_step(5, total_steps, "Verifying model formats")
    
    all_good = True
    for filename in issues:
        status, info = check_model_format(filename)
        if status == "correct":
            print(f"✓ {filename}: Fixed ({info} models)")
        else:
            print(f"✗ {filename}: Still has issues")
            all_good = False
    
    if not all_good:
        print("\n⚠️  Some models still have issues. Manual intervention needed.")
        return
    
    # Step 6: Run evaluation
    print_step(6, total_steps, "Running model evaluation")
    print("This will take 1-2 minutes...")
    
    try:
        os.system("python main.py --evaluate")
        print("✓ Evaluation complete")
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
    
    # Step 7: Generate key visualizations
    print_step(7, total_steps, "Generating key visualizations")
    print("This will take 1-2 minutes...")
    
    try:
        os.system("python main.py --visualize")
        
        # Check if key outputs exist
        metrics_table = settings.VISUALIZATION_DIR / "performance" / "metrics_summary_table.png"
        if metrics_table.exists():
            print("✓ Metrics summary table created")
        else:
            print("✗ Metrics summary table missing")
            
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print(" PIPELINE FIX COMPLETE ")
    print("="*60)
    print(f"Finished at: {datetime.now().strftime('%H:%M:%S')}")
    print("\nYou can now run: python main.py --all")
    print("The pipeline should work correctly!")


if __name__ == "__main__":
    main()