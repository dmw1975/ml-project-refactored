"""Visualization functions for ElasticNet models (DEPRECATED).

This module is deprecated and will be removed in a future version.
Please use visualization_new package instead.
"""

import warnings

warnings.warn(
    "This module is deprecated. Please use visualization_new.adapters.elasticnet_adapter instead.",
    DeprecationWarning,
    stacklevel=2
)

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from config import settings
from utils import io

def plot_elasticnet_feature_importance():
    """Plot ElasticNet built-in feature importance (coefficient magnitudes) for each model."""
    # Import required modules
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from visualization.style import setup_visualization_style, save_figure
    
    # Set up style
    style = setup_visualization_style()
    
    # Load ElasticNet results
    try:
        elasticnet_models = io.load_model("elasticnet_models.pkl", settings.MODEL_DIR)
    except:
        print("No ElasticNet models found. Please train ElasticNet models first.")
        return None
    
    # Set up output directories
    features_dir = settings.VISUALIZATION_DIR / "features"
    linear_dir = features_dir / "linear"
    io.ensure_dir(features_dir)
    io.ensure_dir(linear_dir)
    
    # Plot feature importance for each model
    for name, model_data in elasticnet_models.items():
        if 'model' not in model_data or model_data['model'] is None:
            print(f"Skipping {name}: No model object found")
            continue
        
        # Get model and feature names
        model = model_data['model']
        if 'feature_names' in model_data:
            feature_names = model_data['feature_names']
        elif hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        else:
            print(f"Skipping {name}: No feature names found")
            continue
        
        # Get feature importance
        try:
            # For ElasticNet, importance is the absolute coefficient value
            if hasattr(model, 'coef_'):
                # Get importance from coefficient magnitudes
                importance = np.abs(model.coef_)
                
                # Create DataFrame for plotting
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                })
                
                # Sort by importance
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                # Limit to top 20 features
                if len(importance_df) > 20:
                    importance_df = importance_df.head(20)
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Plot
                sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax, palette='viridis')
                
                # Title and labels
                ax.set_title(f'ElasticNet Coefficient Magnitudes: {name}', fontsize=14)
                ax.set_xlabel('Coefficient Magnitude')
                ax.set_ylabel('Feature')
                
                plt.tight_layout()
                
                # Create filenames
                elasticnet_filename = f"elasticnet_feature_importance_{name}"
                model_filename = f"{name}_top_features"
                
                # Save directly to the linear directory only
                save_figure(fig, elasticnet_filename, linear_dir)
                save_figure(fig, model_filename, linear_dir)
                
                print(f"Feature importance plot saved for {name}")
                plt.close(fig)
            else:
                print(f"Skipping {name}: No coef_ attribute found")
        except Exception as e:
            print(f"Error creating feature importance plot for {name}: {e}")
    
    return True

def visualize_elasticnet_models():
    """Run all ElasticNet visualizations."""
    print("Generating ElasticNet feature importance visualizations...")
    plot_elasticnet_feature_importance()
    
    # Import existing ElasticNet visualizations from metrics_plots
    from visualization.metrics_plots import plot_elasticnet_cv_distribution, plot_elasticnet_best_params
    
    print("Generating ElasticNet CV distribution plots...")
    plot_elasticnet_cv_distribution()
    
    print("ElasticNet visualizations completed.")
    
if __name__ == "__main__":
    visualize_elasticnet_models()