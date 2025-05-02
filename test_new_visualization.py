"""Test script for the new visualization architecture."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings
import visualization_new as viz

def test_residual_plots():
    """Test residual plot functionality in new architecture."""
    print("Testing new residual plot functionality...")
    try:
        # Create residual plots for all models
        figures = viz.create_all_residual_plots()
        print(f"Successfully created {len(figures)} residual plots.")
    except Exception as e:
        print(f"Error in residual plots: {e}")
        import traceback
        traceback.print_exc()

def test_feature_importance():
    """Test feature importance visualization in new architecture."""
    print("Testing new feature importance visualization...")
    try:
        # Load models
        from visualization_new.utils.io import load_all_models
        models = load_all_models()
        
        # Display available models
        print(f"Found {len(models)} models:")
        for model_name in models.keys():
            print(f"  - {model_name}")
        
        # Choose a model for feature importance visualization
        if models:
            model_name = list(models.keys())[0]
            model_data = models[model_name]
            
            # Create feature importance plot
            fig = viz.create_feature_importance_plot(model_data)
            print(f"Successfully created feature importance plot for {model_name}.")
        else:
            print("No models found to visualize.")
    except Exception as e:
        print(f"Error in feature importance visualization: {e}")
        import traceback
        traceback.print_exc()

def test_model_comparison():
    """Test model comparison visualization in new architecture."""
    print("Testing new model comparison visualization...")
    try:
        # Load models
        from visualization_new.utils.io import load_all_models
        models = load_all_models()
        
        # Create model comparison plot
        from visualization_new.viz_factory import create_model_comparison_plot
        fig = create_model_comparison_plot(list(models.values()))
        print(f"Successfully created model comparison plot for {len(models)} models.")
    except Exception as e:
        print(f"Error in model comparison visualization: {e}")
        import traceback
        traceback.print_exc()

def test_dashboard():
    """Test dashboard visualization in new architecture."""
    print("Testing new dashboard visualization...")
    try:
        # Create dashboard
        fig = viz.create_comparative_dashboard()
        print("Successfully created dashboard visualization.")
    except Exception as e:
        print(f"Error in dashboard visualization: {e}")
        import traceback
        traceback.print_exc()

def run_all_tests():
    """Run all visualization tests."""
    print("=== Testing New Visualization Architecture ===")
    test_residual_plots()
    test_feature_importance()
    test_model_comparison()
    test_dashboard()
    print("=== Visualization Tests Complete ===")

if __name__ == "__main__":
    run_all_tests()