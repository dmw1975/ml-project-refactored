"""Test script to verify visualization functionality."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings
from utils import io

def main():
    """Run visualization tests."""
    print("Testing visualization functionality...")
    
    # Test model performance visualizations
    from visualization.metrics_plots import plot_model_comparison, plot_residuals, plot_statistical_tests
    
    print("\nGenerating model performance visualizations...")
    try:
        plot_model_comparison()
        plot_residuals()
        plot_statistical_tests()
        print("Model performance visualizations completed successfully.")
    except Exception as e:
        print(f"Error in model performance visualizations: {e}")
    
    # Test feature importance visualizations
    from visualization.feature_plots import plot_top_features, plot_feature_importance_by_model, plot_feature_correlations
    
    print("\nGenerating feature importance visualizations...")
    try:
        top_features = plot_top_features()
        plot_feature_importance_by_model()
        if top_features is not None:
            plot_feature_correlations(top_features.index[:20].tolist())
        print("Feature importance visualizations completed successfully.")
    except Exception as e:
        print(f"Error in feature importance visualizations: {e}")
    
    print("\nAll visualizations have been saved to:")
    print(f"- Model performance: {settings.VISUALIZATION_DIR / 'performance'}")
    print(f"- Feature importance: {settings.VISUALIZATION_DIR / 'features'}")

if __name__ == "__main__":
    main()