#!/usr/bin/env python
"""Create baseline comparison plots with all models."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.absolute()))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.config import settings
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    sns = None

def create_baseline_plots():
    """Create baseline comparison plots from CSV."""
    
    print("="*80)
    print("CREATING BASELINE COMPARISON PLOTS")
    print("="*80)
    
    # Load baseline comparison data
    csv_path = settings.METRICS_DIR / "baseline_comparison.csv"
    df = pd.read_csv(csv_path)
    print(f"\nLoaded baseline data for {len(df)} models")
    
    # Check model types
    print("\nModel types in data:")
    for model_type in sorted(df['model_type'].unique()):
        count = len(df[df['model_type'] == model_type])
        print(f"  {model_type}: {count}")
    
    # Output directory
    output_dir = settings.VISUALIZATION_DIR / "statistical_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots for each baseline type
    for baseline_type in ['mean', 'median', 'random']:
        print(f"\n\nCreating {baseline_type} baseline comparison plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(f'Model Performance vs {baseline_type.capitalize()} Baseline (Test Set Evaluation)', fontsize=16)
        
        # Sort by improvement percentage
        improvement_col = f'{baseline_type}_improvement_pct'
        df_sorted = df.sort_values(improvement_col, ascending=True)
        
        # Create color map for model types
        model_types = df_sorted['model_type'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_types)))
        color_map = dict(zip(model_types, colors))
        
        # Get colors for each model
        bar_colors = [color_map[mt] for mt in df_sorted['model_type']]
        
        # Left plot: RMSE comparison
        ax1.set_title(f'Model vs {baseline_type.capitalize()} Baseline Performance (Test Set)', fontsize=14)
        
        # Create positions for bars
        y_pos = np.arange(len(df_sorted))
        bar_width = 0.35
        
        # Plot model RMSE
        bars1 = ax1.barh(y_pos - bar_width/2, df_sorted['model_rmse'], bar_width, 
                         label='Model RMSE', color=bar_colors, alpha=0.8)
        
        # Plot baseline RMSE
        bars2 = ax1.barh(y_pos + bar_width/2, df_sorted[f'{baseline_type}_rmse'], bar_width,
                         label=f'{baseline_type.capitalize()} Baseline RMSE', color='lightcoral', alpha=0.8)
        
        # Add value labels
        for i, (model_val, baseline_val) in enumerate(zip(df_sorted['model_rmse'], df_sorted[f'{baseline_type}_rmse'])):
            ax1.text(model_val + 0.02, i - bar_width/2, f'{model_val:.2f}', 
                    va='center', fontsize=8)
            ax1.text(baseline_val + 0.02, i + bar_width/2, f'{baseline_val:.2f}', 
                    va='center', fontsize=8)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(df_sorted['model_name'], fontsize=10)
        ax1.set_xlabel('RMSE (Root Mean Squared Error)', fontsize=12)
        ax1.legend(loc='lower right')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add significance markers
        ax1.text(0.02, 0.02, 'Indicates statistical significance (p < 0.05)', 
                transform=ax1.transAxes, fontsize=9, style='italic')
        ax1.text(0.02, 0.98, f'All metrics and significance tests are based on test set evaluation ({len(df_sorted)} samples)',
                transform=ax1.transAxes, fontsize=9, verticalalignment='top')
        ax1.text(0.02, 0.94, f'P-values from unpaired t-tests comparing model vs baseline residuals',
                transform=ax1.transAxes, fontsize=9, verticalalignment='top')
        
        # Right plot: Improvement percentage
        ax2.set_title(f'Performance Improvement vs {baseline_type.capitalize()} Baseline', fontsize=14)
        
        # Create horizontal bar chart
        bars3 = ax2.barh(y_pos, df_sorted[improvement_col], color=bar_colors, alpha=0.8)
        
        # Add value labels
        for i, improvement in enumerate(df_sorted[improvement_col]):
            x_pos = improvement + 0.5 if improvement > 0 else improvement - 0.5
            ax2.text(x_pos, i, f'{improvement:.1f}%', 
                    va='center', ha='left' if improvement > 0 else 'right', fontsize=8)
        
        # Add significance indicators (assuming all are significant for now)
        significant_mask = df_sorted[improvement_col] > 5  # Simplified threshold
        
        # Color bars based on significance
        for i, (bar, is_sig) in enumerate(zip(bars3, significant_mask)):
            if not is_sig:
                bar.set_facecolor('orange')
                bar.set_label('Not Significant' if i == 0 else "")
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([])  # No labels on right plot
        ax2.set_xlabel('Improvement (%)', fontsize=12)
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax2.grid(axis='x', alpha=0.3)
        
        # Add legend for significance
        if any(~significant_mask):
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='tab:blue', alpha=0.8, label='Significant (p < 0.05)'),
                Patch(facecolor='orange', alpha=0.8, label='Not Significant')
            ]
            ax2.legend(handles=legend_elements, loc='lower right')
        
        # Add model type legend
        legend_elements = []
        for model_type in sorted(model_types):
            legend_elements.append(Patch(facecolor=color_map[model_type], alpha=0.8, label=model_type))
        
        fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.05), 
                  ncol=len(model_types), title="Model Types", fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f"baseline_comparison_{baseline_type}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved {output_path}")

if __name__ == "__main__":
    create_baseline_plots()