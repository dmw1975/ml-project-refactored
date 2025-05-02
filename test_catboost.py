"""Test script for the CatBoost model implementation."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings
from models.catboost_model import train_basic_catboost, optimize_catboost_with_optuna, train_catboost_models
from data import load_features_data, load_scores_data, get_base_and_yeo_features
from evaluation.importance import analyze_feature_importance
from visualization.feature_plots import plot_top_features, plot_feature_importance_by_model
from visualization.create_residual_plots import create_thesis_residual_plot

def test_basic_catboost():
    """Test basic CatBoost model training."""
    print("\n=== Testing Basic CatBoost Model ===")
    
    # Load data
    feature_df = load_features_data()
    score_df = load_scores_data()
    
    # Get feature set
    LR_Base, _, _, _ = get_base_and_yeo_features(feature_df)
    
    # Train basic model
    model_results = train_basic_catboost(LR_Base, score_df, "CatBoost_Base_test")
    
    # Validate outputs
    assert 'model_name' in model_results, "Missing model_name in results"
    assert 'model' in model_results, "Missing model in results"
    assert 'RMSE' in model_results, "Missing RMSE in results"
    assert 'R2' in model_results, "Missing R2 in results"
    assert 'y_test' in model_results, "Missing y_test in results"
    assert 'y_pred' in model_results, "Missing y_pred in results"
    
    print("✓ Basic CatBoost model trained successfully!")
    return model_results

def test_optuna_catboost(n_trials=5):
    """Test CatBoost hyperparameter optimization with Optuna."""
    print("\n=== Testing CatBoost Optuna Optimization ===")
    
    # Load data
    feature_df = load_features_data()
    score_df = load_scores_data()
    
    # Get feature set
    LR_Base, _, _, _ = get_base_and_yeo_features(feature_df)
    
    # Optimize model with a reduced number of trials
    model_results = optimize_catboost_with_optuna(LR_Base, score_df, "CatBoost_Base_optuna_test", n_trials=n_trials)
    
    # Validate outputs
    assert 'model_name' in model_results, "Missing model_name in results"
    assert 'model' in model_results, "Missing model in results"
    assert 'RMSE' in model_results, "Missing RMSE in results"
    assert 'R2' in model_results, "Missing R2 in results"
    assert 'best_params' in model_results, "Missing best_params in results"
    assert 'study' in model_results, "Missing study in results"
    assert 'cv_mse' in model_results, "Missing cv_mse in results"
    
    print("✓ CatBoost Optuna optimization ran successfully!")
    print(f"  Best params: {model_results['best_params']}")
    print(f"  RMSE: {model_results['RMSE']:.4f}")
    print(f"  R²: {model_results['R2']:.4f}")
    
    return model_results

def test_train_catboost_models():
    """Test full training pipeline for CatBoost models."""
    print("\n=== Testing Full CatBoost Training Pipeline ===")
    
    # Set specific datasets and reduced trials for testing
    model_results = train_catboost_models(
        datasets=['CatBoost_Base'],  # Only run one dataset for testing
        n_trials=5                  # Reduced trials for testing
    )
    
    # Validate outputs
    assert 'CatBoost_Base_basic' in model_results, "Missing basic model in results"
    assert 'CatBoost_Base_optuna' in model_results, "Missing optuna model in results"
    
    print("✓ CatBoost training pipeline ran successfully!")
    print(f"  Models trained: {list(model_results.keys())}")
    
    # Test feature importance analysis
    print("\n=== Testing Feature Importance Analysis ===")
    importance_results, consolidated = analyze_feature_importance(model_results)
    
    if importance_results:
        print("✓ Feature importance analysis ran successfully!")
        
        # Test visualization
        print("\n=== Testing Feature Importance Visualization ===")
        plot_top_features(importance_results, top_n=10)  # Reduced top_n for testing
        plot_feature_importance_by_model(importance_results, top_n=10)
        
        print("✓ Feature importance visualization ran successfully!")

    return model_results

def run_full_test():
    """Run all tests."""
    try:
        print("\n==================================")
        print("Running CatBoost Implementation Tests")
        print("==================================\n")
        
        # Run all tests
        test_basic_catboost()
        test_optuna_catboost()
        models = test_train_catboost_models()
        
        print("\n==================================")
        print("All CatBoost Tests Passed Successfully!")
        print("==================================")
        
        return True, models
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    # Run all tests
    success, models = run_full_test()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)