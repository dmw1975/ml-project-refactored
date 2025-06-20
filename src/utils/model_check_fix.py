#!/usr/bin/env python3
"""
Fix the model checking logic to handle missing tree models correctly.

The current issue is that check_all_existing_models returns empty dict for missing
tree models, but returns filled dict for existing linear/elasticnet models, which
causes the pipeline to skip ALL training if ANY models exist.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import settings
import pickle


def check_model_files_individually():
    """
    Check which model files actually exist on disk.
    
    Returns:
        dict: Status of each model type
    """
    model_files = {
        'linear_regression': 'linear_regression_models.pkl',
        'elasticnet': 'elasticnet_models.pkl',
        'xgboost': 'xgboost_models.pkl',
        'lightgbm': 'lightgbm_models.pkl',
        'catboost': 'catboost_models.pkl'
    }
    
    status = {}
    
    for model_type, filename in model_files.items():
        file_path = settings.MODEL_DIR / filename
        if file_path.exists():
            try:
                with open(file_path, 'rb') as f:
                    models = pickle.load(f)
                    model_count = len(models) if isinstance(models, dict) else 0
                    status[model_type] = {
                        'exists': True,
                        'path': str(file_path),
                        'model_count': model_count,
                        'models': list(models.keys()) if isinstance(models, dict) else []
                    }
            except Exception as e:
                status[model_type] = {
                    'exists': True,
                    'path': str(file_path),
                    'error': str(e)
                }
        else:
            status[model_type] = {
                'exists': False,
                'path': str(file_path)
            }
    
    return status


def check_individual_model_training_needed(model_type, datasets=None):
    """
    Check if a specific model type needs training.
    
    Args:
        model_type: Type of model ('xgboost', 'lightgbm', 'catboost', etc.)
        datasets: List of datasets to check
        
    Returns:
        bool: True if training is needed, False otherwise
    """
    model_file_map = {
        'xgboost': 'xgboost_models.pkl',
        'lightgbm': 'lightgbm_models.pkl',
        'catboost': 'catboost_models.pkl',
        'elasticnet': 'elasticnet_models.pkl',
        'linear_regression': 'linear_regression_models.pkl'
    }
    
    if model_type not in model_file_map:
        return True  # Unknown model type, train it
    
    file_path = settings.MODEL_DIR / model_file_map[model_type]
    
    # If file doesn't exist, definitely need training
    if not file_path.exists():
        return True
    
    # Try to load and check models
    try:
        with open(file_path, 'rb') as f:
            models = pickle.load(f)
            
        if not isinstance(models, dict) or not models:
            return True  # Empty or invalid file
            
        # For tree models, check for both basic and optuna variants
        if model_type in ['xgboost', 'lightgbm', 'catboost']:
            expected_variants = []
            dataset_names = datasets if datasets and 'all' not in datasets else ['Base', 'Yeo', 'Base_Random', 'Yeo_Random']
            
            for dataset in dataset_names:
                # Clean dataset name (remove 'LR_' prefix if present)
                clean_dataset = dataset.replace('LR_', '')
                
                # Expected model names
                model_prefix = {
                    'xgboost': 'XGBoost',
                    'lightgbm': 'LightGBM',
                    'catboost': 'CatBoost'
                }[model_type]
                
                # Check for categorical variants
                expected_variants.extend([
                    f"{model_prefix}_{clean_dataset}_categorical_basic",
                    f"{model_prefix}_{clean_dataset}_categorical_optuna"
                ])
            
            # Check if all expected models exist
            missing_models = [m for m in expected_variants if m not in models]
            if missing_models:
                print(f"Missing {model_type} models: {missing_models}")
                return True
                
        return False  # All expected models exist
        
    except Exception as e:
        print(f"Error checking {model_type} models: {e}")
        return True  # Error loading, need to train


def get_tree_model_training_status(datasets=None):
    """
    Get detailed status of tree model training needs.
    
    Args:
        datasets: List of datasets to check
        
    Returns:
        dict: Status for each tree model type
    """
    tree_models = ['xgboost', 'lightgbm', 'catboost']
    status = {}
    
    for model_type in tree_models:
        needs_training = check_individual_model_training_needed(model_type, datasets)
        status[model_type] = {
            'needs_training': needs_training,
            'file_exists': (settings.MODEL_DIR / f"{model_type}_models.pkl").exists()
        }
    
    return status


if __name__ == "__main__":
    print("Checking model files individually...")
    print("=" * 80)
    
    status = check_model_files_individually()
    
    for model_type, info in status.items():
        print(f"\n{model_type.upper()}:")
        print(f"  File exists: {info['exists']}")
        print(f"  Path: {info['path']}")
        if info['exists'] and 'model_count' in info:
            print(f"  Model count: {info['model_count']}")
            if info['models']:
                print(f"  Models: {', '.join(info['models'][:3])}{'...' if len(info['models']) > 3 else ''}")
        elif 'error' in info:
            print(f"  Error: {info['error']}")
    
    print("\n" + "=" * 80)
    print("Checking tree model training needs...")
    print("=" * 80)
    
    tree_status = get_tree_model_training_status()
    
    for model_type, status in tree_status.items():
        print(f"\n{model_type.upper()}:")
        print(f"  Needs training: {status['needs_training']}")
        print(f"  File exists: {status['file_exists']}")