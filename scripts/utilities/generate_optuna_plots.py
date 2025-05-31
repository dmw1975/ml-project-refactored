#!/usr/bin/env python3
"""
Generate missing Optuna optimization plots for XGBoost and CatBoost models.
"""

import pickle
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import optuna
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances, plot_contour

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.config.settings import OUTPUT_DIR

def create_optuna_plots(study, model_name, output_dir):
    """Create Optuna visualization plots for a given study."""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # 1. Optimization History Plot
    try:
        # Create the plot
        ax = plot_optimization_history(study)
        
        # Modify the axis
        ax.set_title(f"{model_name} - Optimization History", fontsize=14, fontweight='bold')
        ax.set_xlabel("Trial", fontsize=12)
        ax.set_ylabel("Objective Value", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Get the figure from the axis and save it
        fig = ax.figure
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{model_name}_optuna_optimization_history.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Created optimization history plot for {model_name}")
    except Exception as e:
        print(f"  Error creating optimization history plot for {model_name}: {e}")
    
    # 2. Parameter Importance Plot
    try:
        # Create the plot
        ax = plot_param_importances(study)
        
        # Modify the axis
        ax.set_title(f"{model_name} - Parameter Importance", fontsize=14, fontweight='bold')
        ax.set_xlabel("Importance", fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Get the figure from the axis and save it
        fig = ax.figure
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{model_name}_optuna_param_importance.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Created parameter importance plot for {model_name}")
    except Exception as e:
        print(f"  Error creating parameter importance plot for {model_name}: {e}")
    
    # 3. Contour Plot (for top 2 most important parameters)
    try:
        # Get the parameter names
        param_names = list(study.best_params.keys())
        if len(param_names) >= 2:
            # Use the first two parameters for the contour plot
            ax = plot_contour(study, params=param_names[:2])
            
            # Modify the axis
            ax.set_title(f"{model_name} - Parameter Contour Plot", fontsize=14, fontweight='bold')
            
            # Get the figure from the axis and save it
            fig = ax.figure
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"{model_name}_contour.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  Created contour plot for {model_name}")
        else:
            print(f"  Skipping contour plot for {model_name} (fewer than 2 parameters)")
    except Exception as e:
        print(f"  Error creating contour plot for {model_name}: {e}")

def create_optuna_improvement_plot(models_data, model_type, output_dir):
    """Create a plot comparing basic vs Optuna model performance."""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect metrics for basic and optuna models
    basic_scores = []
    optuna_scores = []
    dataset_names = []
    
    for key, model_data in models_data.items():
        if model_type not in key:
            continue
            
        # Check for metrics - they can be stored in different ways
        test_r2 = None
        if 'metrics' in model_data:
            if 'test_r2' in model_data['metrics']:
                test_r2 = model_data['metrics']['test_r2']
            elif 'test' in model_data['metrics'] and 'r2' in model_data['metrics']['test']:
                test_r2 = model_data['metrics']['test']['r2']
        elif 'test_r2' in model_data:
            test_r2 = model_data['test_r2']
        elif 'R2' in model_data:
            test_r2 = model_data['R2']
            
        if test_r2 is None:
            continue
        
        # Extract dataset name
        parts = key.split('_')
        dataset = parts[1]  # Base or Yeo
        if 'Random' in key:
            dataset += " Random"
        
        if 'basic' in key:
            basic_scores.append(test_r2)
            if 'optuna' not in key:  # Only add dataset name once
                dataset_names.append(dataset)
        elif 'optuna' in key:
            optuna_scores.append(test_r2)
    
    if not basic_scores or not optuna_scores:
        print(f"  No data found for {model_type} improvement plot")
        return
    
    # Create the improvement plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(dataset_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, basic_scores, width, label='Basic', alpha=0.8)
    bars2 = ax.bar(x + width/2, optuna_scores, width, label='Optuna', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Test R² Score')
    ax.set_title(f'{model_type} Model Performance: Basic vs Optuna Optimization')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Set y-axis limits to better show differences
    min_score = min(min(basic_scores), min(optuna_scores))
    max_score = max(max(basic_scores), max(optuna_scores))
    y_margin = (max_score - min_score) * 0.1
    ax.set_ylim(min_score - y_margin, max_score + y_margin)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_type.lower()}_optuna_improvement.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Created Optuna improvement plot for {model_type}")

def main():
    """Main function to generate all missing Optuna plots."""
    
    # Load model data
    xgb_path = os.path.join(OUTPUT_DIR, 'models', 'xgboost_models.pkl')
    cb_path = os.path.join(OUTPUT_DIR, 'models', 'catboost_models.pkl')
    
    # Process XGBoost models
    print("Generating XGBoost Optuna plots...")
    if os.path.exists(xgb_path):
        with open(xgb_path, 'rb') as f:
            xgb_data = pickle.load(f)
        
        xgb_output_dir = os.path.join(OUTPUT_DIR, 'visualizations', 'performance', 'xgboost')
        
        # Generate individual Optuna plots
        for key, model_data in xgb_data.items():
            if 'study' in model_data and 'optuna' in key:
                print(f"  Processing {key}...")
                create_optuna_plots(model_data['study'], key, xgb_output_dir)
        
        # Generate improvement plot
        create_optuna_improvement_plot(xgb_data, 'XGBoost', xgb_output_dir)
    
    # Process CatBoost models
    print("\nGenerating CatBoost Optuna plots...")
    if os.path.exists(cb_path):
        with open(cb_path, 'rb') as f:
            cb_data = pickle.load(f)
        
        cb_output_dir = os.path.join(OUTPUT_DIR, 'visualizations', 'performance', 'catboost')
        
        # Generate individual Optuna plots
        for key, model_data in cb_data.items():
            if 'study' in model_data and 'optuna' in key:
                print(f"  Processing {key}...")
                create_optuna_plots(model_data['study'], key, cb_output_dir)
        
        # Generate improvement plot
        create_optuna_improvement_plot(cb_data, 'CatBoost', cb_output_dir)
    
    print("\nOptuna plot generation complete!")

if __name__ == "__main__":
    main()