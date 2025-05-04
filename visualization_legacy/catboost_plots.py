"""Visualization functions for CatBoost models (DEPRECATED).

This module is deprecated and will be removed in a future version.
Please use visualization_new package instead.
"""

import warnings

warnings.warn(
    "This module is deprecated. Please use visualization_new.adapters.catboost_adapter instead.",
    DeprecationWarning,
    stacklevel=2
)

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from config import settings
from visualization.style import setup_visualization_style, save_figure
from utils import io

def plot_catboost_comparison():
    """Compare basic vs. Optuna-optimized CatBoost models."""
    # Import required modules
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from visualization.style import setup_visualization_style, save_figure
    
    # Set up style
    style = setup_visualization_style()
    
    # Load CatBoost results
    try:
        catboost_models = io.load_model("catboost_models.pkl", settings.MODEL_DIR)
    except:
        print("No CatBoost models found. Please train CatBoost models first.")
        return None
    
    # Set up main output directory
    perf_dir = settings.VISUALIZATION_DIR / "performance"
    io.ensure_dir(perf_dir)
    
    # Create catboost directory for model-specific performance plots
    output_dir = perf_dir / "catboost"
    io.ensure_dir(output_dir)
    
    # Extract performance metrics
    performance_data = []
    for name, model_data in catboost_models.items():
        performance_data.append({
            'model_name': name,
            'RMSE': model_data['RMSE'],
            'Dataset': name.split("_")[1] + "_" + name.split("_")[2],
            'Type': 'Basic' if 'basic' in name else 'Optimized'
        })
    
    perf_df = pd.DataFrame(performance_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by dataset and type
    sns.barplot(x='Dataset', y='RMSE', hue='Type', data=perf_df, ax=ax, palette='viridis')
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'bottom',
                    xytext = (0, 5), textcoords = 'offset points')
    
    # Title and labels
    ax.set_title('CatBoost Performance Comparison: Basic vs. Optuna-Optimized', fontsize=14)
    ax.set_ylabel('RMSE (lower is better)')
    ax.set_xlabel('Dataset')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    save_figure(fig, "catboost_performance_comparison", output_dir)
    
    print(f"CatBoost comparison plot saved to {output_dir} directory")
    return fig

def plot_catboost_feature_importance():
    """Plot feature importance for CatBoost models."""
    # Import required modules
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from visualization.style import setup_visualization_style, save_figure
    
    # Set up style
    style = setup_visualization_style()
    
    # Load CatBoost results
    try:
        catboost_models = io.load_model("catboost_models.pkl", settings.MODEL_DIR)
    except:
        print("No CatBoost models found. Please train CatBoost models first.")
        return None
    
    # Set up output directories
    output_dir = settings.VISUALIZATION_DIR / "features"
    io.ensure_dir(output_dir)
    
    # Create catboost directory under features for model-specific plots
    catboost_dir = output_dir / "catboost"
    io.ensure_dir(catboost_dir)
    
    # Plot feature importance for each model
    for name, model_data in catboost_models.items():
        if 'model' not in model_data or model_data['model'] is None:
            print(f"Skipping {name}: No model object found")
            continue
        
        # Get model and feature names
        model = model_data['model']
        if 'feature_names' in model_data:
            feature_names = model_data['feature_names']
        elif hasattr(model, 'feature_names_'):
            feature_names = model.feature_names_
        else:
            print(f"Skipping {name}: No feature names found")
            continue
        
        # Get feature importance
        try:
            importance = model.get_feature_importance()
            
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
            fig, ax = plt.figure(figsize=(12, 8), tight_layout=True), plt.gca()
            
            # Prepare data - reverse order to have most important at the top
            feature_names = importance_df['Feature'].values[::-1]
            importance_values = importance_df['Importance'].values[::-1]
            y_pos = np.arange(len(feature_names))
            
            # Plot using horizontal bar chart with standard blue color (#3498db) for consistency
            bars = ax.barh(y_pos, importance_values, align='center', color='#3498db')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + (width * 0.01), bar.get_y() + bar.get_height()/2., 
                        f'{width:.0f}', ha='left', va='center', fontsize=9)
            
            # Title and labels
            ax.set_title(f'Top Features for {name}', fontsize=14)
            ax.set_xlabel('Importance')
            ax.grid(alpha=0.3)
            
            # Create model-specific filename - only save to catboost subdir
            model_filename = f"{name}_top_features"
            
            # Save only to the catboost subdirectory for better organization
            save_figure(fig, model_filename, catboost_dir)
            
            # Save with legacy naming for backwards compatibility - also to catboost dir
            legacy_filename = f"catboost_feature_importance_{name}"
            save_figure(fig, legacy_filename, catboost_dir)
            
            print(f"Feature importance visualization for {name} saved to features/catboost/ directory.")
            plt.close(fig)
        except Exception as e:
            print(f"Error creating feature importance plot for {name}: {e}")
    
    return True

def visualize_catboost_models():
    """Generate all CatBoost visualizations."""
    print("Generating CatBoost model comparison...")
    plot_catboost_comparison()
    
    print("Generating CatBoost feature importance plots...")
    plot_catboost_feature_importance()
    
    print("CatBoost visualizations complete.")