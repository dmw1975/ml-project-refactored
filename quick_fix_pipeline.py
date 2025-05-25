#!/usr/bin/env python3
"""
Quick Pipeline Fix - Just fix the essential models
"""

import sys
import os
from pathlib import Path
import pickle

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import settings

print("\n=== QUICK PIPELINE FIX ===")
print("This will fix just XGBoost and LightGBM models\n")

# Check current status
print("1. Checking current model files...")
model_dir = settings.MODEL_DIR

xgb_file = model_dir / "xgboost_models.pkl"
lgb_file = model_dir / "lightgbm_models.pkl"

need_xgb = False
need_lgb = False

# Check XGBoost
if xgb_file.exists():
    with open(xgb_file, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        print("✓ XGBoost models: OK")
    else:
        print(f"✗ XGBoost models: Wrong format ({type(data).__name__})")
        need_xgb = True
else:
    print("✗ XGBoost models: Missing")
    need_xgb = True

# Check LightGBM  
if lgb_file.exists():
    with open(lgb_file, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        print("✓ LightGBM models: OK")
    else:
        print(f"✗ LightGBM models: Wrong format ({type(data).__name__})")
        need_lgb = True
else:
    print("✗ LightGBM models: Missing")
    need_lgb = True

if not need_xgb and not need_lgb:
    print("\n✓ All models are OK! You can run: python main.py --all")
    sys.exit(0)

# Remove problematic files
print("\n2. Removing problematic files...")
if need_xgb and xgb_file.exists():
    xgb_file.unlink()
    print("✓ Removed xgboost_models.pkl")
    
if need_lgb and lgb_file.exists():
    lgb_file.unlink()
    print("✓ Removed lightgbm_models.pkl")

# Train only what's needed
print("\n3. Training models (this will take a few minutes)...")
print("You'll see training progress below:\n")

if need_xgb:
    print("Training XGBoost...")
    os.system("python main.py --train-xgboost --optimize-xgboost 10")
    
if need_lgb:
    print("\nTraining LightGBM...")
    os.system("python main.py --train-lightgbm --optimize-lightgbm 10")

print("\n4. Verifying fix...")
# Quick verification
all_good = True

if need_xgb and xgb_file.exists():
    with open(xgb_file, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        print(f"✓ XGBoost: Fixed ({len(data)} models)")
    else:
        print("✗ XGBoost: Still has issues")
        all_good = False
        
if need_lgb and lgb_file.exists():
    with open(lgb_file, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        print(f"✓ LightGBM: Fixed ({len(data)} models)")
    else:
        print("✗ LightGBM: Still has issues")
        all_good = False

if all_good:
    print("\n✓ PIPELINE FIXED!")
    print("\nYou can now run:")
    print("  python main.py --evaluate")
    print("  python main.py --visualize")
    print("  python main.py --all")
else:
    print("\n✗ Some issues remain. Please check the output above.")