"""
Sector weight distribution visualization for training and testing sets.

This module provides a focused visualization of sector weight distributions
in training and testing sets across different models.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except (ImportError, SyntaxError) as e:
    print(f"Warning: Could not import seaborn: {e}")
    sns = None
from typing import Dict, Any, List, Optional, Union, Tuple

from config import settings
from utils import io
from src.visualization.core.interfaces import VisualizationConfig
from src.visualization.utils.io import ensure_dir

def extract_sector_weights(model_data):
    """
    Extract sector weights from model data.
    
    Args:
        model_data: Dictionary containing model data
        
    Returns:
        Dict containing train and test sector weights or None if not available
    """
    model_name = model_data.get('model_name', 'Unknown')
    
    # We need X_test to extract sector information
    if 'X_test' not in model_data:
        return None
        
    X_test = model_data['X_test']
    
    # Find sector columns (starting with gics_sector_)
    sector_cols = [col for col in X_test.columns if col.startswith('gics_sector_')]
    if not sector_cols:
        return None
        
    # Get sector distribution in test set
    test_sectors = {}
    total_test_count = len(X_test)
    
    for col in sector_cols:
        sector_name = col.replace('gics_sector_', '')
        sector_count = X_test[col].sum()
        sector_percent = (sector_count / total_test_count) * 100
        test_sectors[sector_name] = sector_percent
    
    # Calculate train set distribution
    train_sectors = {}
    n_train = model_data.get('n_companies_train', 0)
    n_test = model_data.get('n_companies_test', 0)
    n_total = model_data.get('n_companies', 0)
    
    if n_train > 0 and n_test > 0 and n_total > 0:
        # We need to estimate train distribution based on test distribution and counts
        for sector_name, test_pct in test_sectors.items():
            # Calculate implied total sector count
            test_sector_count = (test_pct / 100) * n_test
            total_sector_count = (test_sector_count / n_test) * n_total
            train_sector_count = total_sector_count - test_sector_count
            train_sector_pct = (train_sector_count / n_train) * 100
            train_sectors[sector_name] = train_sector_pct
    
    return {
        'model_name': model_name,
        'test_sectors': test_sectors,
        'train_sectors': train_sectors
    }

def plot_sector_weights_distribution(
    model_names=None,
    config=None,
    skip_sector_models=True
):
    """
    Visualize sector weight distribution in train and test sets.

    Args:
        model_names: List of model names to include (None for all)
        config: Visualization configuration
        skip_sector_models: If True, skip models with names starting with "Sector_"

    Returns:
        Created figure
    """
    # Process configuration
    if config is None:
        config = {}
    
    if isinstance(config, dict):
        config = VisualizationConfig(**config)
        
    # Set default output directory
    output_dir = config.get('output_dir')
    if output_dir is None:
        output_dir = settings.VISUALIZATION_DIR / "sectors"
    
    # Ensure directory exists
    ensure_dir(output_dir)
    
    # Load all model data
    model_files = {
        'xgboost': 'xgboost_models.pkl',
        'lightgbm': 'lightgbm_models.pkl',
        'catboost': 'catboost_models.pkl', 
        'elasticnet': 'elasticnet_models.pkl',
        'linear': 'linear_regression_models.pkl',
        'sector': 'sector_models.pkl'
    }
    
    all_models = {}
    
    for model_type, filename in model_files.items():
        try:
            models = io.load_model(filename, settings.MODEL_DIR)
            if models and isinstance(models, dict):
                # Add model_type field for filtering
                for name, model in models.items():
                    model['model_type'] = model_type
                all_models.update(models)
                print(f"Loaded {len(models)} {model_type} models")
        except Exception as e:
            print(f"Could not load {model_type} models: {e}")
    
    # Filter by model names if provided
    if model_names is not None:
        filtered_models = {name: all_models[name] for name in model_names if name in all_models}
    else:
        filtered_models = all_models.copy()

    # Filter out models with names starting with "Sector_" if requested
    if skip_sector_models:
        filtered_models = {name: model for name, model in filtered_models.items()
                          if not name.startswith('Sector_')}
        print(f"Filtered out sector-specific models. Remaining models: {len(filtered_models)}")
    
    # Extract sector weights for each model
    sector_data = []
    
    for name, model in filtered_models.items():
        weights = extract_sector_weights(model)
        if weights:
            sector_data.append(weights)
    
    if not sector_data:
        print("No sector data found in models")
        return None
    
    # Create visualization for each model
    for model_data in sector_data:
        model_name = model_data['model_name']
        
        # Get sector distributions
        train_sectors = model_data['train_sectors']
        test_sectors = model_data['test_sectors']
        
        # Convert to DataFrames for easier plotting
        sectors = sorted(list(set(list(train_sectors.keys()) + list(test_sectors.keys()))))
        
        df_data = []
        for sector in sectors:
            df_data.append({
                'Sector': sector,
                'Train (%)': train_sectors.get(sector, 0),
                'Test (%)': test_sectors.get(sector, 0)
            })
        
        df = pd.DataFrame(df_data)
        
        # Sort by biggest sectors first
        df['Total'] = df['Train (%)'] + df['Test (%)']
        df = df.sort_values('Total', ascending=False).drop('Total', axis=1)
        
        # Prepare for grouped bar chart
        sectors = df['Sector']
        train_pcts = df['Train (%)']
        test_pcts = df['Test (%)']
        
        # Calculate difference for highlighting imbalances
        df['Diff'] = np.abs(df['Train (%)'] - df['Test (%)'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set up bar width and positions
        width = 0.35
        x = np.arange(len(sectors))
        
        # Create bars with custom colors
        train_bars = ax.bar(x - width/2, train_pcts, width, label='Train', color='#3498db', alpha=0.9)
        test_bars = ax.bar(x + width/2, test_pcts, width, label='Test', color='#e74c3c', alpha=0.9)
        
        # Add labels and title
        ax.set_title(f'Sector Weight Distribution: {model_name}', fontsize=16)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(sectors, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=12)
        
        # Add value labels on bars
        for bars in [train_bars, test_bars]:
            for bar in bars:
                height = bar.get_height()
                if height > 0.5:  # Only label bars with significant height
                    ax.annotate(f'{height:.1f}%',
                                xy=(bar.get_x() + bar.get_width()/2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9,
                                bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.8))
        
        # Add grid for readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        # Add a textbox with balance metrics
        avg_diff = df['Diff'].mean()
        max_diff = df['Diff'].max()
        max_diff_sector = df.loc[df['Diff'].idxmax(), 'Sector']
        
        textstr = f"Balance Metrics:\n" \
                 f"Avg. Diff: {avg_diff:.2f}%\n" \
                 f"Max Diff: {max_diff:.2f}% ({max_diff_sector})"
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if config.get('save', True):
            output_file = f"{model_name}_sector_weights.png"
            output_path = os.path.join(output_dir, output_file)
            
            plt.savefig(output_path, dpi=config.get('dpi', 300), format=config.get('format', 'png'))
            print(f"Saved sector weights visualization for {model_name} to {output_path}")
        
        # Show if requested
        if config.get('show', False):
            plt.show()
            
        plt.close(fig)
    
    return True

def plot_all_models_sector_summary(config=None, skip_sector_models=True):
    """
    Create a summary of sector weights across all models.

    Args:
        config: Visualization configuration
        skip_sector_models: If True, skip models with names starting with "Sector_"

    Returns:
        Created figure
    """
    # Process configuration
    if config is None:
        config = {}
        
    if isinstance(config, dict):
        config = VisualizationConfig(**config)
    
    # Set default output directory
    output_dir = config.get('output_dir')
    if output_dir is None:
        output_dir = settings.VISUALIZATION_DIR / "sectors"
    
    # Load all sector models
    try:
        sector_models = io.load_model("sector_models.pkl", settings.MODEL_DIR)
        if not sector_models:
            print("No sector models found")
            return None
    except Exception as e:
        print(f"Error loading sector models: {e}")
        return None
    
    # Extract sector weights
    sector_data = []

    for name, model in sector_models.items():
        # Skip sector-specific models if requested
        if skip_sector_models and name.startswith('Sector_'):
            continue

        weights = extract_sector_weights(model)
        if weights:
            sector_data.append(weights)

    print(f"Processing {len(sector_data)} models for sector summary")
    
    if not sector_data:
        print("No sector data found in models")
        return None
    
    # Create a single figure for all sectors
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Get unique sectors across all models
    all_sectors = set()
    for data in sector_data:
        all_sectors.update(data['train_sectors'].keys())
        all_sectors.update(data['test_sectors'].keys())
    
    all_sectors = sorted(list(all_sectors))
    
    # Prepare data for each sector
    sector_summary = {sector: {'train': [], 'test': []} for sector in all_sectors}
    
    for data in sector_data:
        for sector in all_sectors:
            sector_summary[sector]['train'].append(data['train_sectors'].get(sector, 0))
            sector_summary[sector]['test'].append(data['test_sectors'].get(sector, 0))
    
    # Calculate averages and differences
    summary_data = []
    for sector in all_sectors:
        train_avg = np.mean(sector_summary[sector]['train'])
        test_avg = np.mean(sector_summary[sector]['test'])
        diff = np.abs(train_avg - test_avg)
        
        summary_data.append({
            'Sector': sector,
            'Train Avg (%)': train_avg,
            'Test Avg (%)': test_avg,
            'Difference': diff
        })
    
    # Convert to DataFrame and sort
    df = pd.DataFrame(summary_data)
    df = df.sort_values('Train Avg (%)', ascending=False)
    
    # Plot as grouped bar chart
    sectors = df['Sector']
    train_avgs = df['Train Avg (%)']
    test_avgs = df['Test Avg (%)']
    
    # Set up bar width and positions
    width = 0.35
    x = np.arange(len(sectors))
    
    # Create bars
    train_bars = ax.bar(x - width/2, train_avgs, width, label='Train Avg', color='#3498db', alpha=0.9)
    test_bars = ax.bar(x + width/2, test_avgs, width, label='Test Avg', color='#e74c3c', alpha=0.9)
    
    # Add labels and title
    ax.set_title('Average Sector Weight Distribution Across All Models', fontsize=16)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_xlabel('Sector', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(sectors, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12)
    
    # Add value labels
    for bars in [train_bars, test_bars]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.5:  # Only label bars with significant height
                ax.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
    
    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if config.get('save', True):
        output_file = "all_models_sector_summary.png"
        output_path = os.path.join(output_dir, output_file)
        
        plt.savefig(output_path, dpi=config.get('dpi', 300), format=config.get('format', 'png'))
        print(f"Saved sector weights summary to {output_path}")
    
    # Show if requested
    if config.get('show', False):
        plt.show()
    
    return fig

if __name__ == "__main__":
    # Generate visualizations when run directly
    config = {
        'save': True,
        'show': False,
        'dpi': 300,
        'format': 'png',
        'output_dir': settings.VISUALIZATION_DIR / "sectors"
    }

    # By default, skip models with names starting with "Sector_"
    skip_sector_models = True

    print(f"Generating sector weight visualizations (skip_sector_models={skip_sector_models})")

    # Generate individual model visualizations
    plot_sector_weights_distribution(config=config, skip_sector_models=skip_sector_models)

    # Generate summary visualization
    plot_all_models_sector_summary(config=config, skip_sector_models=skip_sector_models)