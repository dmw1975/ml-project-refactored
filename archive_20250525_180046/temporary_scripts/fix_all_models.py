#!/usr/bin/env python3
"""
Comprehensive fix for all model issues in the pipeline
"""

import sys
import os
import pickle
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import settings
from utils import io


def fix_all_model_issues():
    """Fix all model format and consistency issues"""
    
    print("\n" + "="*70)
    print(" COMPREHENSIVE MODEL FIX ")
    print("="*70)
    
    # 1. Check what we have
    print("\n1. Checking current model files...")
    model_dir = settings.MODEL_DIR
    
    issues = []
    
    # Expected files
    expected_files = {
        'linear_regression_models.pkl': 'Linear Regression',
        'elasticnet_models.pkl': 'ElasticNet', 
        'xgboost_models.pkl': 'XGBoost',
        'lightgbm_models.pkl': 'LightGBM',
        'catboost_models.pkl': 'CatBoost'
    }
    
    for filename, model_type in expected_files.items():
        filepath = model_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                if isinstance(data, dict):
                    # Check if the models have required fields
                    missing_fields = []
                    for model_name, model_data in data.items():
                        if 'y_test' not in model_data or 'y_pred' not in model_data:
                            missing_fields.append(model_name)
                    
                    if missing_fields:
                        print(f"✗ {model_type}: Missing y_test/y_pred for {len(missing_fields)} models")
                        issues.append((filename, 'missing_fields', missing_fields))
                    else:
                        print(f"✓ {model_type}: OK ({len(data)} models)")
                else:
                    print(f"✗ {model_type}: Wrong format ({type(data).__name__})")
                    issues.append((filename, 'wrong_format', None))
            except Exception as e:
                print(f"✗ {model_type}: Error reading - {e}")
                issues.append((filename, 'error', None))
        else:
            print(f"✗ {model_type}: Missing")
            issues.append((filename, 'missing', None))
    
    # 2. Fix XGBoost models
    print("\n2. Fixing XGBoost models...")
    xgb_individual_files = [
        'xgboost_base_categorical.pkl',
        'xgboost_yeo_categorical.pkl',
        'xgboost_base_random_categorical.pkl',
        'xgboost_yeo_random_categorical.pkl'
    ]
    
    xgb_models = {}
    for file in xgb_individual_files:
        filepath = model_dir / file
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Extract model name from filename
                model_name = file.replace('.pkl', '').replace('xgboost', 'XGBoost').replace('_', ' ').title()
                model_name = model_name.replace(' Categorical', '_categorical')
                
                xgb_models[model_name] = model_data
                print(f"  ✓ Loaded {file}")
            except Exception as e:
                print(f"  ✗ Error loading {file}: {e}")
    
    if xgb_models:
        # Save as combined file
        io.save_model(xgb_models, "xgboost_models.pkl", settings.MODEL_DIR)
        print(f"  ✓ Created xgboost_models.pkl with {len(xgb_models)} models")
    
    # 3. Fix LightGBM models
    print("\n3. Fixing LightGBM models...")
    lgb_individual_files = [
        'lightgbm_base_categorical.pkl',
        'lightgbm_yeo_categorical.pkl', 
        'lightgbm_base_random_categorical.pkl',
        'lightgbm_yeo_random_categorical.pkl'
    ]
    
    lgb_models = {}
    for file in lgb_individual_files:
        filepath = model_dir / file
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Extract model name from filename
                model_name = file.replace('.pkl', '').replace('lightgbm', 'LightGBM').replace('_', ' ').title()
                model_name = model_name.replace(' Categorical', '_categorical')
                
                lgb_models[model_name] = model_data
                print(f"  ✓ Loaded {file}")
            except Exception as e:
                print(f"  ✗ Error loading {file}: {e}")
    
    if lgb_models:
        # Save as combined file
        io.save_model(lgb_models, "lightgbm_models.pkl", settings.MODEL_DIR)
        print(f"  ✓ Created lightgbm_models.pkl with {len(lgb_models)} models")
    
    # 4. Fix CatBoost models (retrain if needed)
    print("\n4. Checking CatBoost models...")
    catboost_needs_retrain = False
    
    catboost_file = model_dir / "catboost_models.pkl"
    if catboost_file.exists():
        with open(catboost_file, 'rb') as f:
            catboost_models = pickle.load(f)
        
        for model_name, model_data in catboost_models.items():
            if 'y_test' not in model_data or 'y_pred' not in model_data:
                catboost_needs_retrain = True
                break
    else:
        catboost_needs_retrain = True
    
    if catboost_needs_retrain:
        print("  ✗ CatBoost models need retraining (missing y_test/y_pred)")
        print("  → Run: python main.py --train-catboost")
    else:
        print("  ✓ CatBoost models OK")
    
    # 5. Summary and recommendations
    print("\n" + "="*70)
    print(" FIX SUMMARY ")
    print("="*70)
    
    # Check final status
    all_good = True
    for filename in expected_files.keys():
        filepath = model_dir / filename
        if not filepath.exists():
            print(f"✗ {filename} still missing")
            all_good = False
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                print(f"✓ {filename} exists with {len(data)} models")
            else:
                print(f"✗ {filename} has wrong format")
                all_good = False
    
    if catboost_needs_retrain:
        print("\nNext steps:")
        print("1. Retrain CatBoost models: python main.py --train-catboost")
        print("2. Run evaluation: python main.py --evaluate")
        print("3. Generate visualizations: python main.py --visualize")
    elif all_good:
        print("\n✓ All models are now in correct format!")
        print("\nYou can now run:")
        print("  python main.py --evaluate --visualize")
        print("  python main.py --all")
    else:
        print("\n⚠️  Some issues remain. You may need to retrain some models.")


if __name__ == "__main__":
    fix_all_model_issues()