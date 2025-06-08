#!/usr/bin/env python3
"""
Fix sector performance boxplots to include model information in titles.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.config import settings
from src.visualization.plots.sectors import SectorPerformanceComparison
from src.visualization.core.interfaces import VisualizationConfig


def update_plot_sector_boxplots():
    """Update the plot_sector_boxplots method to include model information."""
    
    # First, let's check what model types are in the data
    metrics_file = settings.METRICS_DIR / "sector_models_metrics.csv"
    if metrics_file.exists():
        metrics_df = pd.read_csv(metrics_file)
        model_types = sorted(metrics_df['type'].unique())
        print(f"Model types found in sector data: {model_types}")
    else:
        print(f"Metrics file not found at {metrics_file}")
        return
    
    # Create the updated plot with model information
    print("\nCreating updated sector performance boxplots...")
    
    # Load the visualization class
    sector_viz = SectorPerformanceComparison()
    
    # Update the configuration to include model types in title
    config = VisualizationConfig(
        output_dir=settings.VISUALIZATION_DIR / "sectors",
        save=True,
        show=False,
        format='png',
        dpi=300,
        figsize=(20, 10)
    )
    
    # Create the plot with updated title
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Get unique sectors
    sectors = metrics_df['sector'].unique()
    n_models_per_sector = metrics_df.groupby('sector')['type'].nunique().max()
    
    # RMSE by sector
    ax = axes[0]
    # Create box plot data grouped by sector
    rmse_data = []
    sector_labels = []
    for sector in sorted(sectors):
        sector_data = metrics_df[metrics_df['sector'] == sector]['RMSE'].values
        rmse_data.append(sector_data)
        sector_labels.append(sector)
    
    # Create boxplot
    bp = ax.boxplot(rmse_data, labels=sector_labels, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(sectors)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_title('RMSE Distribution by Sector', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sector', fontsize=12)
    ax.set_ylabel('RMSE (lower is better)', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    
    # Add annotation about model types
    model_types_str = ', '.join(model_types)
    ax.text(0.02, 0.98, f'Models included: {model_types_str}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # R² by sector
    ax = axes[1]
    # Create box plot data grouped by sector
    r2_data = []
    for sector in sorted(sectors):
        sector_data = metrics_df[metrics_df['sector'] == sector]['R2'].values
        r2_data.append(sector_data)
    
    # Create boxplot
    bp = ax.boxplot(r2_data, labels=sector_labels, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_title('R² Distribution by Sector', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sector', fontsize=12)
    ax.set_ylabel('R² (higher is better)', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    
    # Add annotation about model types
    ax.text(0.02, 0.98, f'Models included: {model_types_str}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Add main title
    fig.suptitle(f'Sector Performance Comparison Across {len(model_types)} Model Types\n({model_types_str})', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout(pad=2.0)
    
    # Save the updated plot
    output_path = settings.VISUALIZATION_DIR / "sectors" / "sector_performance_boxplots.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved updated plot to {output_path}")
    plt.close()
    
    # Also create version for lightgbm subdirectory if it exists
    lightgbm_dir = settings.VISUALIZATION_DIR / "sectors" / "lightgbm"
    if lightgbm_dir.exists():
        # For LightGBM directory, create a plot showing only LightGBM models
        print("\nCreating LightGBM-specific sector performance boxplots...")
        
        # Filter for LightGBM only
        lightgbm_df = metrics_df[metrics_df['type'] == 'LightGBM']
        
        if not lightgbm_df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            
            # RMSE by sector (LightGBM only)
            ax = axes[0]
            rmse_data = []
            for sector in sorted(sectors):
                sector_data = lightgbm_df[lightgbm_df['sector'] == sector]['RMSE'].values
                if len(sector_data) > 0:
                    rmse_data.append(sector_data)
            
            # Since we only have one model per sector, show as bar plot instead of box
            sector_rmse = lightgbm_df.groupby('sector')['RMSE'].mean().sort_index()
            bars = ax.bar(range(len(sector_rmse)), sector_rmse.values, color=colors)
            ax.set_xticks(range(len(sector_rmse)))
            ax.set_xticklabels(sector_rmse.index, rotation=45, ha='right')
            
            ax.set_title('LightGBM RMSE by Sector', fontsize=14, fontweight='bold')
            ax.set_xlabel('Sector', fontsize=12)
            ax.set_ylabel('RMSE (lower is better)', fontsize=12)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            # R² by sector (LightGBM only)
            ax = axes[1]
            sector_r2 = lightgbm_df.groupby('sector')['R2'].mean().sort_index()
            bars = ax.bar(range(len(sector_r2)), sector_r2.values, color=colors)
            ax.set_xticks(range(len(sector_r2)))
            ax.set_xticklabels(sector_r2.index, rotation=45, ha='right')
            
            ax.set_title('LightGBM R² by Sector', fontsize=14, fontweight='bold')
            ax.set_xlabel('Sector', fontsize=12)
            ax.set_ylabel('R² (higher is better)', fontsize=12)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Add main title
            fig.suptitle('LightGBM Model Performance by Sector', fontsize=16, fontweight='bold')
            
            plt.tight_layout(pad=2.0)
            
            # Save the LightGBM-specific plot
            output_path = lightgbm_dir / "sector_performance_boxplots.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved LightGBM-specific plot to {output_path}")
            plt.close()


def main():
    """Main function."""
    print("Updating sector performance boxplots with model information...")
    print("="*60)
    
    update_plot_sector_boxplots()
    
    print("\nDone!")


if __name__ == "__main__":
    main()