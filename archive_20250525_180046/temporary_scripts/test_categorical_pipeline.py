"""Quick test of categorical pipeline with reduced trials."""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from data_categorical import load_tree_models_data, get_base_and_yeo_features_categorical
from models.catboost_categorical import train_catboost_with_categorical, optimize_catboost_with_categorical

def test_categorical_pipeline():
    """Test categorical pipeline with quick settings."""
    print("=" * 60)
    print("TESTING CATEGORICAL PIPELINE")
    print("=" * 60)
    
    # Load data
    features, target = load_tree_models_data()
    print(f"Loaded data: {features.shape} features, {len(target)} samples")
    
    # Get base features
    base_features, yeo_features = get_base_and_yeo_features_categorical(use_tree_data=True)
    
    # Test basic CatBoost
    print("\n--- Testing Basic CatBoost ---")
    basic_results = train_catboost_with_categorical(
        base_features, target, "Test_Basic_Categorical"
    )
    
    # Test optimized CatBoost with fewer trials
    print("\n--- Testing Optimized CatBoost (5 trials) ---")
    optuna_results = optimize_catboost_with_categorical(
        base_features, target, "Test_Optuna_Categorical", n_trials=5
    )
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    comparison_data = {
        'Model': ['Basic CatBoost', 'Optimized CatBoost'],
        'Train R²': [basic_results['train_r2'], optuna_results['train_r2']],
        'Test R²': [basic_results['test_r2'], optuna_results['test_r2']],
        'Train MSE': [basic_results['train_mse'], optuna_results['train_mse']],
        'Test MSE': [basic_results['test_mse'], optuna_results['test_mse']],
        'Categorical Features': [len(basic_results['categorical_features']), len(optuna_results['categorical_features'])]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.round(4))
    
    print(f"\nCategorical features used: {basic_results['categorical_features']}")
    print(f"Total features: {len(base_features.columns)}")
    print(f"Quantitative features: {len(base_features.columns) - len(basic_results['categorical_features'])}")
    
    print("\n✅ Categorical pipeline test completed successfully!")
    
    return basic_results, optuna_results

if __name__ == "__main__":
    test_categorical_pipeline()