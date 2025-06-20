#!/usr/bin/env python3
"""
Comprehensive validation of the Base features fix.
This script verifies that Base models now correctly include all features.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.data.data_categorical import load_tree_models_data, get_categorical_features
from src.models.xgboost_categorical import get_base_and_yeo_features_categorical


def validate_base_features():
    """Validate that Base features now include all necessary features."""
    print("=" * 70)
    print("VALIDATING BASE FEATURES FIX")
    print("=" * 70)
    
    # 1. Check the raw fixed dataset
    print("\n1. Checking raw fixed dataset...")
    fixed_data_path = Path("data/raw/combined_df_for_tree_models_FIXED.csv")
    if fixed_data_path.exists():
        df_fixed = pd.read_csv(fixed_data_path)
        print(f"✅ Fixed dataset exists: {df_fixed.shape}")
        
        # Count feature types
        raw_numerical = [c for c in df_fixed.columns if not c.startswith('yeo_joh_') 
                        and c not in ['issuer_name', 'gics_sector', 'gics_sub_ind', 
                                    'issuer_cntry_domicile_name', 'cntry_of_risk',
                                    'top_1_shareholder_location', 'top_2_shareholder_location',
                                    'top_3_shareholder_location']]
        yeo_numerical = [c for c in df_fixed.columns if c.startswith('yeo_joh_')]
        
        print(f"   - Raw numerical features: {len(raw_numerical)}")
        print(f"   - Yeo-transformed features: {len(yeo_numerical)}")
        print(f"   - Categorical features: 7")
    else:
        print(f"❌ Fixed dataset not found at {fixed_data_path}")
        return False
    
    # 2. Check processed tree models dataset
    print("\n2. Checking processed tree models dataset...")
    processed_path = Path("data/processed/tree_models_dataset.csv")
    if processed_path.exists():
        df_processed = pd.read_csv(processed_path)
        print(f"✅ Processed dataset exists: {df_processed.shape}")
    else:
        print(f"❌ Processed dataset not found at {processed_path}")
        return False
    
    # 3. Test get_base_and_yeo_features_categorical function
    print("\n3. Testing get_base_and_yeo_features_categorical function...")
    try:
        base_features, yeo_features = get_base_and_yeo_features_categorical()
        print(f"✅ Function executed successfully")
        print(f"   - Base features shape: {base_features.shape}")
        print(f"   - Yeo features shape: {yeo_features.shape}")
        
        if base_features.shape[1] == 33 and yeo_features.shape[1] == 33:
            print(f"✅ Both datasets have 33 features as expected!")
        else:
            print(f"❌ Feature counts don't match expected 33")
            return False
            
    except Exception as e:
        print(f"❌ Error in get_base_and_yeo_features_categorical: {e}")
        return False
    
    # 4. Verify specific features are present
    print("\n4. Verifying specific features...")
    expected_raw_features = ['market_cap_usd', 'hist_pe', 'hist_roe', 'return_usd', 'vola']
    expected_yeo_features = ['yeo_joh_' + f for f in expected_raw_features]
    categorical_features = get_categorical_features()
    
    print("   Checking Base dataset:")
    for feat in expected_raw_features[:3]:
        if feat in base_features.columns:
            print(f"   ✅ {feat} present")
        else:
            print(f"   ❌ {feat} missing")
            
    print("\n   Checking Yeo dataset:")
    for feat in expected_yeo_features[:3]:
        if feat in yeo_features.columns:
            print(f"   ✅ {feat} present")
        else:
            print(f"   ❌ {feat} missing")
    
    print("\n   Checking categorical features in both:")
    for cat in categorical_features[:3]:
        base_has = cat in base_features.columns
        yeo_has = cat in yeo_features.columns
        if base_has and yeo_has:
            print(f"   ✅ {cat} present in both")
        else:
            print(f"   ❌ {cat} missing in {'Base' if not base_has else 'Yeo'}")
    
    # 5. Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    success = True
    if base_features.shape[1] == 33 and yeo_features.shape[1] == 33:
        print("✅ SUCCESS: Base features fix is working correctly!")
        print(f"   - Base: {base_features.shape[1]} features (26 raw numerical + 7 categorical)")
        print(f"   - Yeo: {yeo_features.shape[1]} features (26 transformed numerical + 7 categorical)")
    else:
        print("❌ FAILURE: Feature counts don't match expected values")
        success = False
    
    # Show feature examples
    print("\nExample features from each dataset:")
    print("\nBase features (first 5 non-categorical):")
    non_cat_base = [c for c in base_features.columns if c not in categorical_features][:5]
    for i, feat in enumerate(non_cat_base, 1):
        print(f"   {i}. {feat}")
        
    print("\nYeo features (first 5 non-categorical):")
    non_cat_yeo = [c for c in yeo_features.columns if c not in categorical_features][:5]
    for i, feat in enumerate(non_cat_yeo, 1):
        print(f"   {i}. {feat}")
    
    return success


def check_existing_models():
    """Check if existing models need to be retrained."""
    print("\n" + "=" * 70)
    print("CHECKING EXISTING MODELS")
    print("=" * 70)
    
    model_files = [
        "outputs/models/xgboost_models.pkl",
        "outputs/models/lightgbm_models.pkl",
        "outputs/models/catboost_models.pkl"
    ]
    
    needs_retraining = []
    
    for model_file in model_files:
        model_path = Path(model_file)
        if model_path.exists():
            print(f"\nChecking {model_path.name}...")
            try:
                with open(model_path, 'rb') as f:
                    models = pickle.load(f)
                
                # Check Base models
                base_models = [k for k in models.keys() if 'Base' in k and 'Random' not in k]
                for model_key in base_models:
                    model_data = models[model_key]
                    if 'X_train' in model_data:
                        n_features = model_data['X_train'].shape[1]
                        print(f"   {model_key}: {n_features} features")
                        if n_features < 30:  # Should have ~33 features
                            print(f"   ⚠️  Model has only {n_features} features - needs retraining!")
                            needs_retraining.append(model_key)
                            
            except Exception as e:
                print(f"   Error loading {model_path.name}: {e}")
    
    if needs_retraining:
        print(f"\n⚠️  The following models need retraining with the fixed dataset:")
        for model in needs_retraining:
            print(f"   - {model}")
        print("\nRun 'python main.py --train' to retrain all models with the corrected features.")
    else:
        print("\n✅ All existing models appear to have the correct number of features.")
    
    return needs_retraining


if __name__ == "__main__":
    # Run validation
    success = validate_base_features()
    
    # Check existing models
    needs_retraining = check_existing_models()
    
    # Final recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    if success:
        print("✅ The Base features fix has been successfully applied!")
        print("\nNext steps:")
        print("1. Run 'python main.py --train' to retrain all models with the corrected features")
        print("2. Run 'python main.py --evaluate' to evaluate the retrained models")
        print("3. Run 'python main.py --visualize' to regenerate all visualizations")
        print("\nAlternatively, run 'python main.py --all' to do all steps at once.")
    else:
        print("❌ The Base features fix has issues that need to be resolved.")
        print("Please check the error messages above.")