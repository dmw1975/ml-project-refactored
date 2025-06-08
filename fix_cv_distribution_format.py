#!/usr/bin/env python3
"""Fix CV distribution plots to ensure consistent format across all models."""

import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.visualization.core.style import setup_visualization_style

def create_standardized_cv_plot(model_type, models_data, output_path):
    """Create CV distribution plot in standardized format matching XGBoost/ElasticNet."""
    
    # Prepare data for plotting
    plot_data = []
    
    for name, data in models_data.items():
        if 'cv_scores' in data and data['cv_scores'] is not None:
            cv_scores = data['cv_scores']
            
            # Extract dataset info
            if 'Base_Random' in name:
                dataset = 'Base_Random'
            elif 'Yeo_Random' in name:
                dataset = 'Yeo_Random'
            elif 'Yeo' in name:
                dataset = 'Yeo'
            else:
                dataset = 'Base'
            
            # Add each CV fold score
            for score in cv_scores:
                plot_data.append({
                    'Dataset': dataset,
                    'RMSE': score
                })
    
    if not plot_data:
        print(f"No CV data found for {model_type}")
        return False
    
    df = pd.DataFrame(plot_data)
    
    # Create figure with standardized size
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors matching reference standard
    colors = {
        'Base': 'lightblue',
        'Yeo': 'peachpuff',
        'Base_Random': 'lightgreen',
        'Yeo_Random': 'lightcoral'
    }
    
    # Create box plot
    box = sns.boxplot(
        x='Dataset',
        y='RMSE',
        data=df,
        palette=[colors.get(d, 'gray') for d in df['Dataset'].unique()],
        ax=ax
    )
    
    # Add strip plot for individual fold RMSEs
    strip = sns.stripplot(
        x='Dataset',
        y='RMSE',
        data=df,
        color='gray',
        alpha=0.6,
        jitter=True,
        ax=ax
    )
    
    # Add mean as red dot
    for i, dataset in enumerate(df['Dataset'].unique()):
        dataset_data = df[df['Dataset'] == dataset]['RMSE']
        mean = np.mean(dataset_data.values)
        ax.plot(i, mean, 'o', color='red', markersize=8, zorder=10)
    
    # Standardized title format
    ax.set_title(f'{model_type} CV RMSE Distribution by Dataset', fontsize=14)
    ax.set_ylabel('RMSE (lower is better)', fontsize=12)
    ax.set_xlabel('Dataset', fontsize=12)
    
    # Add standardized legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='RMSE Distribution (Boxplot)',
               markerfacecolor='lightblue', markersize=15),
        Line2D([0], [0], marker='o', color='gray', label='Individual CV Fold RMSE',
               linestyle='None', markersize=8, alpha=0.6),
        Line2D([0], [0], marker='o', color='red', label='Mean RMSE',
               linestyle='None', markersize=8)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Save with consistent settings
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True

def main():
    """Fix CV distribution plots for CatBoost and LightGBM."""
    print("=" * 80)
    print("FIXING CV DISTRIBUTION PLOT FORMATS")
    print("=" * 80)
    
    setup_visualization_style()
    
    models_dir = settings.MODEL_DIR
    output_dir = settings.VISUALIZATION_DIR / 'performance' / 'cv_distributions'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process CatBoost and LightGBM with standardized format
    for model_type in ['CatBoost', 'LightGBM']:
        print(f"\nFixing {model_type} CV distribution plot...")
        
        # Load models
        model_file = models_dir / f"{model_type.lower()}_models.pkl"
        if not model_file.exists():
            print(f"  ✗ Model file not found: {model_file}")
            continue
        
        with open(model_file, 'rb') as f:
            models = pickle.load(f)
        
        # Filter models with CV data
        cv_models = {name: data for name, data in models.items() 
                     if isinstance(data, dict) and 'cv_scores' in data and data['cv_scores'] is not None}
        
        print(f"  Found {len(cv_models)} models with CV data out of {len(models)} total")
        
        if cv_models:
            output_path = output_dir / f"{model_type.lower()}_cv_distribution.png"
            if create_standardized_cv_plot(model_type, cv_models, output_path):
                print(f"  ✓ Fixed: {output_path}")
            else:
                print(f"  ✗ Failed to create standardized plot")
        else:
            print(f"  ✗ No models with CV data found")

if __name__ == "__main__":
    main()