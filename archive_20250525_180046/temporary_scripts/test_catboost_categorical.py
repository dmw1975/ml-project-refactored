"""Test CatBoost with categorical features."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from data_categorical import load_tree_models_data, get_base_and_yeo_features_categorical
from models.catboost_categorical import train_catboost_with_categorical

def test_catboost_categorical():
    """Test CatBoost with categorical features."""
    print("Testing CatBoost with categorical features...")
    
    # Load data
    features, target = load_tree_models_data()
    print(f"Loaded data: {features.shape} features, {len(target)} samples")
    
    # Get base features
    base_features, _ = get_base_and_yeo_features_categorical(use_tree_data=True)
    print(f"Base features: {base_features.shape}")
    
    # Train a simple model
    results = train_catboost_with_categorical(
        base_features.head(100),  # Use subset for quick test
        target.head(100), 
        "Test_Base_Categorical"
    )
    
    print(f"✅ Test successful! Test R²: {results['test_r2']:.4f}")
    print(f"Categorical features used: {results['categorical_features']}")
    
    return results

if __name__ == "__main__":
    test_catboost_categorical()