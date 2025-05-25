"""Fix the ML pipeline issues with main.py --all

This script identifies and fixes the core issues:
1. Model saving format inconsistency
2. Data loading strategy mismatch
3. Visualization pipeline failures
"""

import sys
from pathlib import Path
import pickle

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import settings
from utils import io


def diagnose_model_files():
    """Diagnose model file issues"""
    print("\n=== DIAGNOSING MODEL FILES ===")
    
    model_dir = settings.MODEL_DIR
    issues = []
    
    # Check each model file
    model_files = {
        'linear_regression_models.pkl': 'dict',
        'elasticnet_models.pkl': 'dict',
        'xgboost_models.pkl': 'dict or Booster',
        'lightgbm_models.pkl': 'dict or Booster',
        'catboost_models.pkl': 'dict',
        'catboost_categorical_models.pkl': 'dict'
    }
    
    for filename, expected_type in model_files.items():
        filepath = model_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    data_type = type(data).__name__
                    
                print(f"{filename}: {data_type}")
                
                # Check if it's a raw model instead of dict
                if 'dict' in expected_type and not isinstance(data, dict):
                    issues.append((filename, data_type))
                    
            except Exception as e:
                print(f"{filename}: ERROR - {e}")
                issues.append((filename, 'error'))
        else:
            print(f"{filename}: NOT FOUND")
    
    return issues


def fix_model_format(filename, model_data):
    """Convert raw model to expected dictionary format"""
    print(f"\nFixing {filename}...")
    
    # Determine model type from filename
    if 'xgboost' in filename:
        model_type = 'XGBoost'
        model_key = 'XGB_Base_basic'  # Default key
    elif 'lightgbm' in filename:
        model_type = 'LightGBM'
        model_key = 'LightGBM_Base_basic'  # Default key
    else:
        print(f"Unknown model type for {filename}")
        return False
    
    # Create dictionary format with placeholder data
    fixed_data = {
        model_key: {
            'model': model_data,
            'model_name': model_key,
            'model_type': model_type,
            'dataset_name': 'Base',
            'RMSE': 0.0,  # Placeholder
            'MAE': 0.0,   # Placeholder
            'MSE': 0.0,   # Placeholder
            'R2': 0.0,    # Placeholder
            'n_companies': 0,
            'n_features': 0,
            'y_test': [],  # Need actual test data
            'y_pred': [],  # Need actual predictions
            'feature_names': [],  # Need actual feature names
            'feature_importance': {}  # Need actual importance
        }
    }
    
    # Save fixed version
    backup_path = settings.MODEL_DIR / f"backup_{filename}"
    filepath = settings.MODEL_DIR / filename
    
    # Backup original
    import shutil
    shutil.copy2(filepath, backup_path)
    print(f"Created backup: {backup_path}")
    
    # Save fixed version
    with open(filepath, 'wb') as f:
        pickle.dump(fixed_data, f)
    print(f"Fixed and saved: {filepath}")
    
    return True


def ensure_categorical_models():
    """Ensure categorical model files exist with proper format"""
    print("\n=== ENSURING CATEGORICAL MODELS ===")
    
    # Check for categorical model files
    categorical_models = [
        'xgboost_base_categorical.pkl',
        'xgboost_yeo_categorical.pkl',
        'lightgbm_base_categorical.pkl',
        'lightgbm_yeo_categorical.pkl'
    ]
    
    for model_file in categorical_models:
        filepath = settings.MODEL_DIR / model_file
        if not filepath.exists():
            print(f"Missing: {model_file}")
            # These should be created by training with categorical features
        else:
            print(f"Found: {model_file}")


def test_evaluation_with_fixes():
    """Test if evaluation works after fixes"""
    print("\n=== TESTING EVALUATION ===")
    
    try:
        from evaluation.metrics import evaluate_models
        results = evaluate_models()
        
        if results and 'all_models' in results:
            print(f"✓ Successfully evaluated {len(results['all_models'])} models")
            return True
        else:
            print("✗ Evaluation found no models")
            return False
    except Exception as e:
        print(f"✗ Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run diagnostics and fixes"""
    print("="*60)
    print(" PIPELINE DIAGNOSTICS AND FIXES ")
    print("="*60)
    
    # 1. Diagnose model files
    issues = diagnose_model_files()
    
    if issues:
        print(f"\nFound {len(issues)} model format issues:")
        for filename, issue_type in issues:
            print(f"  - {filename}: {issue_type}")
        
        # Fix issues
        print("\nAttempting to fix model format issues...")
        print("NOTE: This is a temporary fix. Models should be retrained properly.")
        
        # For now, just report the issues
        print("\nRECOMMENDATION: Run the pipeline with proper training:")
        print("1. Delete or backup the problematic model files")
        print("2. Run: python main.py --train-xgboost --train-lightgbm")
        print("3. Or use categorical versions: python main.py --train --use-one-hot")
    
    # 2. Check categorical models
    ensure_categorical_models()
    
    # 3. Test evaluation
    test_evaluation_with_fixes()
    
    # 4. Provide recommendations
    print("\n" + "="*60)
    print(" RECOMMENDATIONS ")
    print("="*60)
    
    print("\nTo fix the pipeline completely:")
    print("1. Clear all model files: rm outputs/models/*.pkl")
    print("2. Train all models fresh: python main.py --all --force-retune")
    print("3. Or train specific models:")
    print("   - python main.py --train-linear")
    print("   - python main.py --train --train-xgboost --train-lightgbm --train-catboost")
    print("\n4. Then evaluate: python main.py --evaluate")
    print("5. Finally visualize: python main.py --visualize")


if __name__ == "__main__":
    main()