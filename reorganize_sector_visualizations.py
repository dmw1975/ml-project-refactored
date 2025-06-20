#!/usr/bin/env python3
"""
Reorganize sector visualizations and create ElasticNet plots.

This script:
1. Moves stratification plot to new location
2. Removes duplicate plots
3. Creates ElasticNet sector visualizations
"""

import sys
import os
from pathlib import Path
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from src.config import settings
from src.visualization.components.formats import save_figure
from src.visualization.utils.io import ensure_dir


def reorganize_stratification_plot():
    """Move stratification plot to new location and remove duplicate."""
    print("\n=== REORGANIZING STRATIFICATION PLOT ===")
    
    # Current location
    current_path = settings.VISUALIZATION_DIR / "sectors" / "sector_train_test_distribution.png"
    
    # New location
    new_dir = settings.VISUALIZATION_DIR / "stratified"
    new_path = new_dir / "sector_train_test_distribution.png"
    
    # Create new directory
    ensure_dir(new_dir)
    
    # Move file if it exists
    if current_path.exists():
        shutil.copy2(str(current_path), str(new_path))  # Use copy2 to preserve metadata
        current_path.unlink()  # Remove original
        print(f"✓ Moved stratification plot from {current_path} to {new_path}")
    else:
        print(f"✗ Stratification plot not found at {current_path}")
    
    # Remove duplicate from lightgbm folder
    duplicate_path = settings.VISUALIZATION_DIR / "sectors" / "lightgbm" / "sector_train_test_distribution.png"
    if duplicate_path.exists():
        duplicate_path.unlink()
        print(f"✓ Removed duplicate stratification plot from {duplicate_path}")
    else:
        print(f"✗ No duplicate found at {duplicate_path}")
    
    return True


def create_elasticnet_folders():
    """Create ElasticNet visualization folders."""
    print("\n=== CREATING ELASTICNET FOLDERS ===")
    
    elasticnet_dir = settings.VISUALIZATION_DIR / "sectors" / "elasticnet"
    ensure_dir(elasticnet_dir)
    print(f"✓ Created ElasticNet folder: {elasticnet_dir}")
    
    return elasticnet_dir


def load_elasticnet_sector_data():
    """Load ElasticNet sector model data."""
    print("\n=== LOADING ELASTICNET SECTOR DATA ===")
    
    # First check if we have ElasticNet sector metrics
    elasticnet_metrics_file = settings.METRICS_DIR / "sector_elasticnet_metrics.csv"
    if elasticnet_metrics_file.exists():
        print(f"✓ Found ElasticNet sector metrics file: {elasticnet_metrics_file}")
        elasticnet_df = pd.read_csv(elasticnet_metrics_file)
        print(f"✓ Loaded {len(elasticnet_df)} ElasticNet sector models")
        print(f"  Datasets: {sorted(elasticnet_df['dataset'].unique())}")
        print(f"  Sectors: {sorted(elasticnet_df['sector'].unique())}")
        return elasticnet_df
    
    # If not, check if we need to run the ElasticNet sector models
    print(f"✗ ElasticNet sector metrics not found at {elasticnet_metrics_file}")
    print("  Running ElasticNet sector model training...")
    
    # Import and run the ElasticNet sector models
    from src.models.sector_elastic_net_models import run_sector_elastic_net_models
    
    # Run the models
    sector_models = run_sector_elastic_net_models()
    
    # Now load the generated metrics
    if elasticnet_metrics_file.exists():
        elasticnet_df = pd.read_csv(elasticnet_metrics_file)
        print(f"\n✓ Successfully generated and loaded {len(elasticnet_df)} ElasticNet sector models")
        return elasticnet_df
    else:
        raise RuntimeError("Failed to generate ElasticNet sector metrics")


def create_sector_metric_tables(elasticnet_df, output_dir):
    """Create sector metric summary tables for each dataset."""
    print("\n=== CREATING SECTOR METRIC TABLES ===")
    
    datasets = elasticnet_df['dataset'].unique()
    
    for dataset in datasets:
        # Filter data for this dataset
        dataset_df = elasticnet_df[elasticnet_df['dataset'] == dataset].copy()
        
        if len(dataset_df) == 0:
            continue
        
        # Prepare table data
        table_data = dataset_df[['sector', 'RMSE', 'MAE', 'MSE', 'R2', 'n_companies']].copy()
        table_data = table_data.rename(columns={'R2': 'R²'})
        
        # Sort by RMSE
        table_data = table_data.sort_values('RMSE')
        
        # Create figure
        fig_height = max(10, len(table_data) * 0.8 + 2)
        fig = plt.figure(figsize=(16, fig_height))
        
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        # Create cell colors with alternating rows
        colors = []
        for i in range(len(table_data)):
            if i % 2 == 0:
                row_colors = ['#F2F2F2'] * len(table_data.columns)
            else:
                row_colors = ['white'] * len(table_data.columns)
            colors.append(row_colors)
        
        # Highlight R² column
        r2_col_idx = list(table_data.columns).index('R²')
        for i, row in enumerate(table_data.values):
            r2_value = row[r2_col_idx]
            if r2_value > 0:
                colors[i][r2_col_idx] = '#C6EFCE'  # Light green
            else:
                colors[i][r2_col_idx] = '#FFC7CE'  # Light red
        
        # Format cell text
        cell_text = []
        for row in table_data.values:
            row_text = []
            for i, val in enumerate(row):
                col_name = table_data.columns[i]
                if col_name == 'sector':
                    row_text.append(str(val))
                elif col_name == 'n_companies':
                    row_text.append(f"{int(val)}")
                else:
                    row_text.append(f"{val:.4f}")
            cell_text.append(row_text)
        
        # Create table
        header_color = '#4472C4'
        table = plt.table(
            cellText=cell_text,
            colLabels=table_data.columns,
            cellColours=colors,
            colColours=[header_color] * len(table_data.columns),
            cellLoc='center',
            loc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2.0)
        
        # Style headers
        for i in range(len(table_data.columns)):
            cell = table[(0, i)]
            cell.set_text_props(weight='bold', color='white')
        
        plt.title(f'ElasticNet Sector Performance Metrics - {dataset} Dataset', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Save figure
        filename = f"elasticnet_sector_metrics_{dataset.lower()}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Created {filename}")


def create_sector_performance_boxplots(elasticnet_df, output_dir):
    """Create sector performance boxplots for each dataset."""
    print("\n=== CREATING SECTOR PERFORMANCE BOXPLOTS ===")
    
    datasets = elasticnet_df['dataset'].unique()
    
    for dataset in datasets:
        # Filter data for this dataset
        dataset_df = elasticnet_df[elasticnet_df['dataset'] == dataset].copy()
        
        if len(dataset_df) == 0:
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Sort sectors by mean RMSE
        sector_order = dataset_df.groupby('sector')['RMSE'].mean().sort_values().index
        
        # Create bar plot (since we have one model per sector)
        bars = ax.bar(range(len(sector_order)), 
                      [dataset_df[dataset_df['sector'] == s]['RMSE'].values[0] for s in sector_order],
                      color='#3498db', alpha=0.7)
        
        # Add value labels
        for i, (bar, sector) in enumerate(zip(bars, sector_order)):
            height = bar.get_height()
            r2_val = dataset_df[dataset_df['sector'] == sector]['R2'].values[0]
            n_companies = dataset_df[dataset_df['sector'] == sector]['n_companies'].values[0]
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f'{height:.4f}\nR²={r2_val:.3f}\n(n={int(n_companies)})', 
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_title(f'ElasticNet RMSE by Sector - {dataset} Dataset', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Sector', fontsize=12)
        ax.set_ylabel('RMSE (lower is better)', fontsize=12)
        ax.set_xticks(range(len(sector_order)))
        ax.set_xticklabels(sector_order, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"elasticnet_sector_boxplot_{dataset.lower()}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Created {filename}")


def create_sector_heatmaps(elasticnet_df, output_dir):
    """Create sector heatmaps for different metrics."""
    print("\n=== CREATING SECTOR HEATMAPS ===")
    
    # We need to create a comparison across datasets
    # Pivot data to create sector x dataset grid for each metric
    
    metrics = ['RMSE', 'R2', 'MAE', 'MSE']
    metric_labels = {'RMSE': 'RMSE', 'R2': 'R²', 'MAE': 'MAE', 'MSE': 'MSE'}
    
    for metric in metrics:
        # Create pivot table
        pivot_df = elasticnet_df.pivot_table(
            index='sector',
            columns='dataset',
            values=metric,
            aggfunc='mean'
        )
        
        # Reorder columns to match expected order
        col_order = ['Base', 'Base_Random', 'Yeo', 'Yeo_Random']
        pivot_df = pivot_df.reindex(columns=[col for col in col_order if col in pivot_df.columns])
        
        # Sort sectors by overall performance
        if metric in ['RMSE', 'MAE', 'MSE']:
            # Lower is better
            sector_order = elasticnet_df.groupby('sector')[metric].mean().sort_values().index
        else:
            # Higher is better for R2
            sector_order = elasticnet_df.groupby('sector')[metric].mean().sort_values(ascending=False).index
        
        pivot_df = pivot_df.reindex(sector_order)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 12))
        
        # Create heatmap
        if metric in ['RMSE', 'MAE', 'MSE']:
            # Lower is better - use reversed colormap
            cmap = 'YlOrRd'
        else:
            # Higher is better for R2
            cmap = 'YlGnBu'
        
        sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap=cmap,
                   linewidths=0.5, ax=ax, cbar_kws={'label': metric_labels[metric]})
        
        ax.set_title(f'ElasticNet {metric_labels[metric]} by Sector and Dataset', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Sector', fontsize=12)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"elasticnet_sector_heatmap_{metric.lower()}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Created {filename}")


def main():
    """Run the reorganization and visualization creation."""
    print("=" * 70)
    print("REORGANIZING SECTOR VISUALIZATIONS AND CREATING ELASTICNET PLOTS")
    print("=" * 70)
    
    try:
        # Task 1: Reorganize stratification plot
        reorganize_stratification_plot()
        
        # Task 2: Create ElasticNet folders
        elasticnet_dir = create_elasticnet_folders()
        
        # Task 3: Load ElasticNet data
        elasticnet_df = load_elasticnet_sector_data()
        
        # Since we don't have dataset information in the current data,
        # we'll create visualizations treating all ElasticNet models as one dataset
        # This is a limitation of the current data structure
        
        print("\nCreating visualizations for ElasticNet sector models.")
        print(f"Datasets found: {sorted(elasticnet_df['dataset'].unique())}")
        
        # Task 4: Create metric tables
        create_sector_metric_tables(elasticnet_df, elasticnet_dir)
        
        # Task 5: Create performance boxplots
        create_sector_performance_boxplots(elasticnet_df, elasticnet_dir)
        
        # Task 6: Create heatmaps
        create_sector_heatmaps(elasticnet_df, elasticnet_dir)
        
        print("\n" + "=" * 70)
        print("✓ VISUALIZATION REORGANIZATION COMPLETE!")
        print("=" * 70)
        
        print(f"\nNew folder structure:")
        print(f"  - Stratified plot: {settings.VISUALIZATION_DIR}/stratified/")
        print(f"  - ElasticNet plots: {elasticnet_dir}/")
        # Count actual plots created
        dataset_count = len(elasticnet_df['dataset'].unique())
        metric_count = 4  # RMSE, R2, MAE, MSE
        total_plots = dataset_count + dataset_count + metric_count  # tables + boxplots + heatmaps
        
        print(f"\nTotal ElasticNet plots created: {total_plots}")
        print(f"  - {dataset_count} metric summary tables")
        print(f"  - {dataset_count} performance boxplots")
        print(f"  - {metric_count} metric heatmaps")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())