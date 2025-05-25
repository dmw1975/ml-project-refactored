#!/usr/bin/env python3
"""
Test that CV fold scores are being stored in the enhanced tree model implementations.
"""

import sys
from pathlib import Path
import pickle

# Import the enhanced implementations
from enhanced_xgboost_categorical import train_enhanced_xgboost_categorical
from enhanced_lightgbm_categorical import train_enhanced_lightgbm_categorical
from enhanced_catboost_categorical import train_enhanced_catboost_categorical
from data_categorical import load_tree_models_data, get_categorical_features

def test_cv_fold_scores():
    """Test that all tree models store CV fold scores."""
    
    # Load data
    print("Loading data...")
    X, y = load_tree_models_data()
    categorical_columns = get_categorical_features()
    
    # Use a small subset for testing
    X_subset = X.iloc[:500]
    y_subset = y.iloc[:500]
    
    # Test each model type
    models_to_test = [
        ("XGBoost", train_enhanced_xgboost_categorical),
        ("LightGBM", train_enhanced_lightgbm_categorical),
        ("CatBoost", train_enhanced_catboost_categorical)
    ]
    
    for model_name, train_func in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing {model_name} CV fold scores...")
        print('='*60)
        
        try:
            # Train models (this will train both basic and optuna versions)
            results = train_func(X_subset, y_subset, "Test", categorical_columns)
            
            # Check if CV fold scores are stored in the optuna model
            optuna_key = f"{model_name}_Test_categorical_optuna"
            if optuna_key in results:
                optuna_result = results[optuna_key]
                
                # Check for CV scores
                if 'cv_scores' in optuna_result:
                    cv_scores = optuna_result['cv_scores']
                    cv_mean = optuna_result.get('cv_mean', 'Not found')
                    cv_std = optuna_result.get('cv_std', 'Not found')
                    
                    print(f"✅ {model_name} stores CV fold scores!")
                    print(f"   CV Scores: {cv_scores}")
                    print(f"   CV Mean: {cv_mean}")
                    print(f"   CV Std: {cv_std}")
                    
                    # Check study for CV info
                    if 'study' in optuna_result:
                        study = optuna_result['study']
                        best_trial = study.best_trial
                        
                        # Check user attributes
                        user_attrs = best_trial.user_attrs
                        print(f"   User attributes in best trial: {list(user_attrs.keys())}")
                else:
                    print(f"❌ {model_name} does NOT store CV fold scores!")
                    print(f"   Available keys: {list(optuna_result.keys())}")
            else:
                print(f"❌ {model_name} optuna model not found in results!")
                print(f"   Available models: {list(results.keys())}")
                
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("Enhanced implementations should store CV fold scores in the optuna model results.")
    print("The scores should be accessible via:")
    print("  - results[model_key]['cv_scores'] - list of fold scores")
    print("  - results[model_key]['cv_mean'] - mean CV score")
    print("  - results[model_key]['cv_std'] - standard deviation of CV scores")
    print("  - study.best_trial.user_attrs - for accessing from the Optuna study")

if __name__ == "__main__":
    test_cv_fold_scores()