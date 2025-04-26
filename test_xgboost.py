"""Test script to verify XGBoost functionality."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings
from models.xgboost_model import train_xgboost_models
from evaluation.metrics import evaluate_models
from evaluation.importance import analyze_feature_importance
from visualization.xgboost_plots import visualize_xgboost_models

def test_xgboost_training():
    """Test XGBoost model training with a small number of trials."""
    print("Testing XGBoost model training...")
    
    # Train with fewer trials for testing
    results = train_xgboost_models(n_trials=5)
    
    if results:
        print(f"Successfully trained {len(results)} XGBoost models")
        for model_name, result in results.items():
            print(f"  {model_name}: R² = {result['R2']:.4f}")
        return True
    else:
        print("Failed to train XGBoost models")
        return False

def test_xgboost_evaluation():
    """Test XGBoost model evaluation."""
    print("\nTesting XGBoost model evaluation...")
    
    # Run evaluation
    eval_results = evaluate_models()
    
    if eval_results:
        # Check if XGBoost models are included
        xgb_models = [model for model in eval_results['all_models'].keys() if 'XGB' in model]
        if xgb_models:
            print(f"Successfully evaluated {len(xgb_models)} XGBoost models")
            return True
        else:
            print("No XGBoost models found in evaluation")
            return False
    else:
        print("Failed to evaluate models")
        return False

def test_xgboost_visualization():
    """Test XGBoost visualization generation."""
    print("\nTesting XGBoost visualization...")
    
    try:
        visualize_xgboost_models()
        print("Successfully generated XGBoost visualizations")
        return True
    except Exception as e:
        print(f"Failed to generate XGBoost visualizations: {e}")
        return False

def test_feature_importance():
    """Test feature importance analysis for XGBoost models."""
    print("\nTesting feature importance analysis...")
    
    importance_results = analyze_feature_importance()
    
    if importance_results and importance_results[0]:
        xgb_importance = {k: v for k, v in importance_results[0].items() if 'XGB' in k}
        if xgb_importance:
            print(f"Successfully analyzed importance for {len(xgb_importance)} XGBoost models")
            return True
        else:
            print("No XGBoost feature importance results found")
            return False
    else:
        print("Failed to analyze feature importance")
        return False

def main():
    """Run all XGBoost tests."""
    print("Starting XGBoost integration tests...")
    print("=" * 50)
    
    # Run tests
    train_success = test_xgboost_training()
    eval_success = test_xgboost_evaluation()
    viz_success = test_xgboost_visualization()
    importance_success = test_feature_importance()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"  Training: {'PASSED' if train_success else 'FAILED'}")
    print(f"  Evaluation: {'PASSED' if eval_success else 'FAILED'}")
    print(f"  Visualization: {'PASSED' if viz_success else 'FAILED'}")
    print(f"  Feature Importance: {'PASSED' if importance_success else 'FAILED'}")
    
    if all([train_success, eval_success, viz_success, importance_success]):
        print("\nAll tests PASSED! XGBoost integration is working correctly.")
    else:
        print("\nSome tests FAILED. Please check the output above for details.")

if __name__ == "__main__":
    main()