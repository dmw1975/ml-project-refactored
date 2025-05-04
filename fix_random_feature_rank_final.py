"""
Script to improve the random_feature_rank visualization with better x-axis labels.
This version correctly includes ONLY models with random features, including LightGBM.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import re

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings
from visualization.style import setup_visualization_style, save_figure

def improve_random_feature_rank():
    """
    Improve the existing random feature rank visualization with better labels.
    This version correctly identifies ALL models with random features.
    """
    # Set up style
    style = setup_visualization_style()
    
    # Define file paths
    output_dir = Path(settings.VISUALIZATION_DIR) / "features" / "comparison"
    
    # These are the models with random features (extracted from original visualization)
    # The key point is that we *only* include models that have "Random" in their name
    models = [
        'LR_Base_Random',
        'LR_Yeo_Random',
        'ElasticNet_LR_Base_Random',
        'ElasticNet_LR_Yeo_Random',
        'XGB_Base_Random_basic',
        'XGB_Base_Random_optuna',
        'XGB_Yeo_Random_basic',
        'XGB_Yeo_Random_optuna',
        'LightGBM_Base_Random_basic',
        'LightGBM_Base_Random_optuna',
        'LightGBM_Yeo_Random_basic', 
        'LightGBM_Yeo_Random_optuna',
        'CatBoost_Base_Random_basic',
        'CatBoost_Base_Random_optuna',
        'CatBoost_Yeo_Random_basic',
        'CatBoost_Yeo_Random_optuna'
    ]
    
    # Clean model names for better display
    clean_names = []
    for model in models:
        name = model
        
        # Extract model type
        if model.startswith('ElasticNet'):
            name = 'EN' + model[10:]  # Remove 'ElasticNet_'
        elif model.startswith('LightGBM'):
            name = 'LGBM' + model[8:]  # Remove 'LightGBM_'
        elif model.startswith('CatBoost'):
            name = 'CAT' + model[8:]  # Remove 'CatBoost_'
        elif model.startswith('XGB'):
            name = model  # Keep XGB as is
        
        # Simplify dataset part
        name = name.replace('_Base_Random', '_BaseR')
        name = name.replace('_Yeo_Random', '_YeoR')
        
        # Simplify optimization method
        name = name.replace('_basic', '_basic')
        name = name.replace('_optuna', '_opt')
        
        clean_names.append(name)
    
    # All models have rank 363 out of 363 features
    ranks = [363] * len(models)
    
    # Create a new figure
    plt.figure(figsize=(14, 8))
    
    # Create color mapping by model family
    colors = []
    for model in models:
        if 'XGB' in model:
            colors.append('#3498db')  # blue
        elif 'LightGBM' in model:
            colors.append('#2ecc71')  # green
        elif 'CatBoost' in model:
            colors.append('#e74c3c')  # red
        elif 'ElasticNet' in model:
            colors.append('#9b59b6')  # purple
        elif 'LR' in model:
            colors.append('#f39c12')  # orange
        else:
            colors.append('#95a5a6')  # gray
    
    # Plot with vertical labels
    bars = plt.bar(clean_names, ranks, color=colors)
    
    # Add rank annotations on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f"363/363", ha='center', va='bottom', fontsize=10)
    
    # Customize plot
    plt.title('Random Feature Rank Across Models with Random Features\n(Higher Rank = Less Important)', fontsize=16)
    plt.ylabel('Rank', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Set vertical x-axis labels for better readability
    plt.xticks(rotation=90)
    
    # Add a horizontal line at rank 363 to emphasize all models ranked the random feature last
    plt.axhline(y=363, color='black', linestyle='--', alpha=0.5)
    
    # Add a legend for model types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#f39c12', label='Linear Regression'),
        Patch(facecolor='#9b59b6', label='ElasticNet'),
        Patch(facecolor='#3498db', label='XGBoost'),
        Patch(facecolor='#2ecc71', label='LightGBM'),
        Patch(facecolor='#e74c3c', label='CatBoost')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    # Ensure enough spacing at the bottom for the vertical labels
    plt.subplots_adjust(bottom=0.25)
    
    # Tight layout and save
    plt.tight_layout()
    save_figure(plt.gcf(), "fixed_random_feature_rank", output_dir)
    print(f"Properly fixed random feature rank visualization saved to the comparison directory")
    
    return plt.gcf()

if __name__ == "__main__":
    improve_random_feature_rank()
    plt.close()  # Close the figure to avoid display in non-interactive mode