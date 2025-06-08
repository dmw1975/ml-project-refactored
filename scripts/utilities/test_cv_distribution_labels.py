#!/usr/bin/env python3
"""Test CV distribution plot labels to see what model names are shown."""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.visualization.utils.io import load_all_models
from src.visualization.plots.cv_distributions import CVDistributionPlot
from src.config.settings import VISUALIZATION_DIR


def test_single_cv_plot():
    """Create a test CV distribution plot for CatBoost to check labels."""
    print("Loading all models...")
    models = load_all_models()
    
    # Get just CatBoost models with CV data
    catboost_cv_models = []
    for key, model_data in models.items():
        if isinstance(model_data, dict) and 'catboost' in key.lower():
            if any(k in model_data for k in ['cv_scores', 'cv_fold_scores', 'cv_mean', 'cv_mse']):
                catboost_cv_models.append(model_data)
                print(f"Found CatBoost model with CV data: {key}")
                print(f"  - model_name in data: {model_data.get('model_name', 'NOT FOUND')}")
    
    if not catboost_cv_models:
        print("No CatBoost models with CV data found!")
        return
    
    print(f"\nCreating test plot with {len(catboost_cv_models)} CatBoost models...")
    
    # Create CV distribution plot
    cv_plot = CVDistributionPlot(catboost_cv_models)
    
    # Extract CV metrics to see what names are being used
    print("\nExtracting CV metrics to check model names:")
    for i, model in enumerate(catboost_cv_models):
        metrics = cv_plot._extract_cv_metrics(model)
        if metrics:
            print(f"\nModel {i}:")
            print(f"  - Extracted model_name: {metrics['model_name']}")
            print(f"  - Model type: {metrics['model_type']}")
            print(f"  - Dataset: {metrics['dataset']}")
    
    # Create the actual plot
    print("\nCreating CatBoost CV distribution plot...")
    fig = cv_plot.plot_cv_rmse_distribution(model_types=['CatBoost'])
    
    if fig:
        # Save to test location
        test_output = Path("test_outputs/cv_distribution_test")
        test_output.mkdir(parents=True, exist_ok=True)
        
        output_path = test_output / "catboost_cv_distribution_test.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nTest plot saved to: {output_path}")
        
        # Also save to main output directory to replace the one with "unknown"
        main_output = VISUALIZATION_DIR / "performance" / "cv_distributions" / "catboost_cv_distribution.png"
        fig.savefig(main_output, dpi=300, bbox_inches='tight')
        print(f"Main plot updated at: {main_output}")
        
        plt.close(fig)
    else:
        print("Failed to create plot!")


def test_all_cv_plots():
    """Recreate all CV distribution plots to ensure proper labels."""
    print("\n\nRecreating all CV distribution plots...")
    
    models = load_all_models()
    
    # Filter models with CV data
    cv_models = []
    for key, model_data in models.items():
        if isinstance(model_data, dict):
            if any(k in model_data for k in ['cv_scores', 'cv_fold_scores', 'cv_mean', 'cv_mse']):
                cv_models.append(model_data)
    
    print(f"\nFound {len(cv_models)} models with CV data")
    
    # Create plots
    cv_config = {
        'save': True,
        'output_dir': VISUALIZATION_DIR / "performance" / "cv_distributions",
        'dpi': 300,
        'format': 'png'
    }
    
    from src.visualization.plots.cv_distributions import plot_cv_distributions
    
    figures = plot_cv_distributions(cv_models, cv_config)
    print(f"\nCreated {len(figures)} CV distribution plots")
    
    # Close all figures
    for fig in figures.values():
        if hasattr(fig, 'close'):
            plt.close(fig)


if __name__ == "__main__":
    test_single_cv_plot()
    test_all_cv_plots()