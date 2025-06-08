#!/usr/bin/env python3
"""Generate missing CV distribution plots for CatBoost and LightGBM."""

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

def generate_cv_distribution_plot(model_type, models_data, output_path):
    """Generate CV distribution plot for a specific model type."""
    
    # Prepare data for plotting
    plot_data = []
    
    for name, data in models_data.items():
        if 'cv_scores' in data and data['cv_scores'] is not None:
            cv_scores = data['cv_scores']
            # Extract dataset and optimization info from name
            if 'Base_Random' in name:
                dataset = 'Base_Random'
            elif 'Yeo_Random' in name:
                dataset = 'Yeo_Random'
            elif 'Yeo' in name:
                dataset = 'Yeo'
            else:
                dataset = 'Base'
            
            optimization = 'Optuna' if 'optuna' in name else 'Basic'
            
            for score in cv_scores:
                plot_data.append({
                    'Model': f"{model_type}_{optimization}",
                    'Dataset': dataset,
                    'RMSE': score,
                    'Optimization': optimization
                })
    
    if not plot_data:
        print(f"No CV data found for {model_type}")
        return False
    
    df = pd.DataFrame(plot_data)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Create grouped boxplot
    ax = sns.boxplot(
        x='Dataset',
        y='RMSE',
        hue='Optimization',
        data=df,
        palette=['lightblue', 'lightgreen']
    )
    
    # Add title and labels
    plt.title(f'{model_type.upper()} Cross-Validation RMSE Distribution', fontsize=16, pad=20)
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    
    # Add mean values as text
    for i, dataset in enumerate(df['Dataset'].unique()):
        for j, opt in enumerate(['Basic', 'Optuna']):
            subset = df[(df['Dataset'] == dataset) & (df['Optimization'] == opt)]
            if not subset.empty:
                mean_val = subset['RMSE'].mean()
                position = i + (j - 0.5) * 0.4
                plt.text(position, mean_val, f'{mean_val:.3f}', 
                        ha='center', va='bottom', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True

def main():
    """Generate missing CV distribution plots."""
    print("=" * 80)
    print("GENERATING MISSING CV DISTRIBUTION PLOTS")
    print("=" * 80)
    
    setup_visualization_style()
    
    models_dir = settings.MODEL_DIR
    output_dir = settings.VISUALIZATION_DIR / 'performance' / 'cv_distributions'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process CatBoost and LightGBM
    for model_type in ['catboost', 'lightgbm']:
        print(f"\nProcessing {model_type.upper()}...")
        
        # Load models
        model_file = models_dir / f"{model_type}_models.pkl"
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
            output_path = output_dir / f"{model_type}_cv_distribution.png"
            if generate_cv_distribution_plot(model_type, cv_models, output_path):
                print(f"  ✓ Generated: {output_path}")
            else:
                print(f"  ✗ Failed to generate plot")
        else:
            print(f"  ✗ No models with CV data found")
    
    # Verify results
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    expected_files = [
        'catboost_cv_distribution.png',
        'lightgbm_cv_distribution.png',
        'xgboost_cv_distribution.png',
        'elasticnet_cv_distribution.png'
    ]
    
    print("\nCV Distribution Files:")
    for filename in expected_files:
        filepath = output_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} - MISSING")

if __name__ == "__main__":
    main()