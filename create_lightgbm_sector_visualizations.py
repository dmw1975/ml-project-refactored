#!/usr/bin/env python3
"""
Create LightGBM sector visualizations from existing data.

This script creates LightGBM visualizations based on sector_lightgbm_metrics.csv.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from src.config import settings
from src.visualization.utils.io import ensure_dir


def load_lightgbm_sector_data():
    """Load LightGBM sector model data."""
    print("\n=== LOADING LIGHTGBM SECTOR DATA ===")
    
    # Load LightGBM sector metrics
    metrics_file = settings.METRICS_DIR / "sector_lightgbm_metrics.csv"
    if not metrics_file.exists():
        raise FileNotFoundError(f"LightGBM sector metrics not found: {metrics_file}")
    
    df = pd.read_csv(metrics_file)
    
    # Extract dataset type from model name
    def extract_dataset(model_name):
        """Extract dataset type from model name."""
        # Format: Sector_LightGBM_<Sector>_<Dataset>
        parts = model_name.split('_')
        if len(parts) >= 4:
            dataset = '_'.join(parts[3:])
            return dataset
        return 'Unknown'
    
    def extract_sector(model_name):
        """Extract sector from model name."""
        # Format: Sector_LightGBM_<Sector>_<Dataset>
        parts = model_name.split('_')
        if len(parts) >= 3:
            # Handle multi-word sectors
            if parts[2] == 'Consumer' and len(parts) > 3:
                if parts[3] in ['Discretionary', 'Staples']:
                    return f"{parts[2]} {parts[3]}"
            elif parts[2] == 'Information' and len(parts) > 3:
                if parts[3] == 'Technology':
                    return f"{parts[2]} {parts[3]}"
            elif parts[2] == 'Health' and len(parts) > 3:
                if parts[3] == 'Care':
                    return f"{parts[2]} {parts[3]}"
            elif parts[2] == 'Real' and len(parts) > 3:
                if parts[3] == 'Estate':
                    return f"{parts[2]} {parts[3]}"
            return parts[2]
        return 'Unknown'
    
    # Add dataset and sector columns if not present
    if 'dataset' not in df.columns:
        df['dataset'] = df['model_name'].apply(extract_dataset)
    if 'sector' not in df.columns or df['sector'].isna().any():
        df['sector'] = df['model_name'].apply(extract_sector)
    
    print(f"✓ Loaded {len(df)} LightGBM sector models")
    print(f"  Datasets: {sorted(df['dataset'].unique())}")
    print(f"  Sectors: {sorted(df['sector'].unique())}")
    
    return df


def create_sector_metric_tables(lightgbm_df, output_dir):
    """Create sector metric summary tables for each dataset."""
    print("\n=== CREATING SECTOR METRIC TABLES ===")
    
    datasets = lightgbm_df['dataset'].unique()
    
    for dataset in datasets:
        # Filter data for this dataset
        dataset_df = lightgbm_df[lightgbm_df['dataset'] == dataset].copy()
        
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
        
        # Highlight R² column based on values
        r2_col_idx = list(table_data.columns).index('R²')
        for i, row in enumerate(table_data.values):
            r2_value = row[r2_col_idx]
            if r2_value > 0.1:
                colors[i][r2_col_idx] = '#C6EFCE'  # Light green
            elif r2_value > 0:
                colors[i][r2_col_idx] = '#FFEB9C'  # Light yellow
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
        header_color = '#2ecc71'  # Green color for LightGBM
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
        
        plt.title(f'LightGBM Sector Performance Metrics - {dataset} Dataset', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Save figure
        filename = f"lightgbm_sector_metrics_{dataset.lower().replace('_', '_')}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Created {filename}")


def create_sector_performance_boxplots(lightgbm_df, output_dir):
    """Create sector performance boxplots for each dataset."""
    print("\n=== CREATING SECTOR PERFORMANCE BOXPLOTS ===")
    
    datasets = lightgbm_df['dataset'].unique()
    
    for dataset in datasets:
        # Filter data for this dataset
        dataset_df = lightgbm_df[lightgbm_df['dataset'] == dataset].copy()
        
        if len(dataset_df) == 0:
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Sort sectors by mean RMSE
        sector_order = dataset_df.groupby('sector')['RMSE'].mean().sort_values().index
        
        # Create bar plot (since we have one model per sector)
        bars = ax.bar(range(len(sector_order)), 
                      [dataset_df[dataset_df['sector'] == s]['RMSE'].values[0] for s in sector_order],
                      color='#2ecc71', alpha=0.7)  # Green color for LightGBM
        
        # Add value labels
        for i, (bar, sector) in enumerate(zip(bars, sector_order)):
            height = bar.get_height()
            r2_val = dataset_df[dataset_df['sector'] == sector]['R2'].values[0]
            n_companies = dataset_df[dataset_df['sector'] == sector]['n_companies'].values[0]
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f'{height:.4f}\nR²={r2_val:.3f}\n(n={int(n_companies)})', 
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_title(f'LightGBM RMSE by Sector - {dataset} Dataset', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Sector', fontsize=12)
        ax.set_ylabel('RMSE (lower is better)', fontsize=12)
        ax.set_xticks(range(len(sector_order)))
        ax.set_xticklabels(sector_order, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"lightgbm_sector_boxplot_{dataset.lower().replace('_', '_')}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Created {filename}")


def create_sector_heatmaps(lightgbm_df, output_dir):
    """Create sector heatmaps for different metrics."""
    print("\n=== CREATING SECTOR HEATMAPS ===")
    
    metrics = ['RMSE', 'R2', 'MAE', 'MSE']
    metric_labels = {'RMSE': 'RMSE', 'R2': 'R²', 'MAE': 'MAE', 'MSE': 'MSE'}
    
    for metric in metrics:
        # Create pivot table
        pivot_df = lightgbm_df.pivot_table(
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
            sector_order = lightgbm_df.groupby('sector')[metric].mean().sort_values().index
        else:
            # Higher is better for R2
            sector_order = lightgbm_df.groupby('sector')[metric].mean().sort_values(ascending=False).index
        
        pivot_df = pivot_df.reindex(sector_order)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 12))
        
        # Create heatmap with green colormap for LightGBM
        if metric in ['RMSE', 'MAE', 'MSE']:
            # Lower is better - use reversed colormap
            cmap = 'RdYlGn_r'  # Red (bad) to Green (good), reversed
        else:
            # Higher is better for R2
            cmap = 'RdYlGn'  # Red (bad) to Green (good)
        
        sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap=cmap,
                   linewidths=0.5, ax=ax, cbar_kws={'label': metric_labels[metric]})
        
        ax.set_title(f'LightGBM {metric_labels[metric]} by Sector and Dataset', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Sector', fontsize=12)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"lightgbm_sector_heatmap_{metric.lower()}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Created {filename}")


def main():
    """Create LightGBM sector visualizations."""
    print("=" * 70)
    print("CREATING LIGHTGBM SECTOR VISUALIZATIONS")
    print("=" * 70)
    
    try:
        # LightGBM folder
        lightgbm_dir = settings.VISUALIZATION_DIR / "sectors" / "lightgbm"
        ensure_dir(lightgbm_dir)
        print(f"✓ LightGBM folder: {lightgbm_dir}")
        
        # Load data
        lightgbm_df = load_lightgbm_sector_data()
        
        print(f"\nCreating visualizations for {len(lightgbm_df['dataset'].unique())} datasets")
        
        # Create visualizations
        create_sector_metric_tables(lightgbm_df, lightgbm_dir)
        create_sector_performance_boxplots(lightgbm_df, lightgbm_dir)
        create_sector_heatmaps(lightgbm_df, lightgbm_dir)
        
        # Count plots created
        dataset_count = len(lightgbm_df['dataset'].unique())
        metric_count = 4  # RMSE, R2, MAE, MSE
        total_plots = dataset_count + dataset_count + metric_count
        
        print("\n" + "=" * 70)
        print("✓ LIGHTGBM VISUALIZATIONS COMPLETE!")
        print("=" * 70)
        
        print(f"\nTotal LightGBM plots created: {total_plots}")
        print(f"  - {dataset_count} metric summary tables")
        print(f"  - {dataset_count} performance boxplots")
        print(f"  - {metric_count} metric heatmaps")
        print(f"\nAll plots saved to: {lightgbm_dir}")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())