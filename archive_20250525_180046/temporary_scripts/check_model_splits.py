#!/usr/bin/env python3
"""Check if models use different train/test splits."""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.io import load_all_models

def main():
    """Check train/test splits across models."""
    print("Loading all models...")
    all_models = load_all_models()
    
    # Group models by their test set sizes and values
    test_set_info = {}
    train_set_info = {}
    
    for model_name, model_data in all_models.items():
        # Skip models without y_test
        if 'y_test' not in model_data or model_data['y_test'] is None:
            continue
            
        y_test = model_data['y_test']
        y_train = model_data.get('y_train', None)
        
        # Get test set statistics
        if hasattr(y_test, 'values'):
            y_test_values = y_test.values.flatten()
        else:
            y_test_values = np.array(y_test).flatten()
            
        test_size = len(y_test_values)
        test_mean = np.mean(y_test_values)
        test_median = np.median(y_test_values)
        test_std = np.std(y_test_values)
        
        # Get train set statistics if available
        if y_train is not None:
            if hasattr(y_train, 'values'):
                y_train_values = y_train.values.flatten()
            else:
                y_train_values = np.array(y_train).flatten()
                
            train_size = len(y_train_values)
            train_mean = np.mean(y_train_values)
            train_median = np.median(y_train_values)
        else:
            train_size = 0
            train_mean = 0
            train_median = 0
        
        # Group by model type
        if model_name.startswith('LR_'):
            model_type = 'Linear Regression'
        elif model_name.startswith('ElasticNet_'):
            model_type = 'ElasticNet'
        elif model_name.startswith('XGBoost_'):
            model_type = 'XGBoost'
        elif model_name.startswith('LightGBM_'):
            model_type = 'LightGBM'
        elif model_name.startswith('CatBoost_'):
            model_type = 'CatBoost'
        else:
            model_type = 'Unknown'
        
        # Store info
        key = f"{model_type}_{test_size}"
        if key not in test_set_info:
            test_set_info[key] = {
                'models': [],
                'test_size': test_size,
                'test_mean': test_mean,
                'test_median': test_median,
                'test_std': test_std,
                'train_size': train_size,
                'train_mean': train_mean,
                'train_median': train_median,
                'model_type': model_type
            }
        test_set_info[key]['models'].append(model_name)
    
    # Print summary
    print("\n=== Test Set Summary by Model Type ===")
    
    # Group by model type
    by_type = {}
    for key, info in test_set_info.items():
        model_type = info['model_type']
        if model_type not in by_type:
            by_type[model_type] = []
        by_type[model_type].append(info)
    
    for model_type, infos in by_type.items():
        print(f"\n{model_type}:")
        for info in infos:
            print(f"  Test size: {info['test_size']}")
            print(f"  Test mean: {info['test_mean']:.4f}")
            print(f"  Test median: {info['test_median']:.4f}")
            print(f"  Train size: {info['train_size']}")
            print(f"  Train mean: {info['train_mean']:.4f}")
            print(f"  Train median: {info['train_median']:.4f}")
            print(f"  Models: {len(info['models'])}")
            print()
    
    # Check for differences
    print("\n=== Key Observations ===")
    
    # Check if linear models have different test sets than tree models
    linear_test_means = []
    tree_test_means = []
    
    for key, info in test_set_info.items():
        if info['model_type'] in ['Linear Regression', 'ElasticNet']:
            linear_test_means.append(info['test_mean'])
        elif info['model_type'] in ['XGBoost', 'LightGBM']:
            tree_test_means.append(info['test_mean'])
    
    if linear_test_means and tree_test_means:
        linear_mean = np.mean(linear_test_means)
        tree_mean = np.mean(tree_test_means)
        print(f"Linear models - Average test set mean: {linear_mean:.4f}")
        print(f"Tree models - Average test set mean: {tree_mean:.4f}")
        print(f"Difference: {abs(linear_mean - tree_mean):.4f}")
        
        if abs(linear_mean - tree_mean) > 0.01:
            print("\n⚠️  Linear and tree models appear to use DIFFERENT test sets!")
            print("This explains why their baseline values are different.")
        else:
            print("\n✓ Linear and tree models appear to use the SAME test sets.")

if __name__ == "__main__":
    main()