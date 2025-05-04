"""Visualization functions for sector-specific model analysis (DEPRECATED).

This module is deprecated and will be removed in a future version.
Please use visualization_new package instead.
"""

import warnings

warnings.warn(
    "This module is deprecated. Please use visualization_new package instead.",
    DeprecationWarning,
    stacklevel=2
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from config import settings
from visualization.style import setup_visualization_style, save_figure
from visualization.metrics_plots import plot_residuals, plot_model_comparison, plot_statistical_tests_filtered
from visualization.statistical_tests import visualize_statistical_tests as plot_statistical_tests
from visualization.feature_plots import plot_top_features, plot_feature_importance_by_model
from utils import io

def plot_sector_model_comparison():
    """
    Create comparison plots specifically for sector-based models.
    """
    # Set up style
    style = setup_visualization_style()
    
    # Load sector metrics
    metrics_file = settings.METRICS_DIR / "sector_models_metrics.csv"
    if not metrics_file.exists():
        print("No sector model metrics found. Please run sector model evaluation first.")
        return None
        
    metrics_df = pd.read_csv(metrics_file)
    
    # Check if sector column exists
    if 'sector' not in metrics_df.columns:
        print("Warning: 'sector' column not found in metrics file.")
        # Try to extract sector from model_name
        if 'model_name' in metrics_df.columns:
            print("Extracting sector information from model names...")
            # Extract sector from model names (format: "Sector_SectorName_...")
            metrics_df['sector'] = metrics_df['model_name'].apply(
                lambda x: x.split('_')[1] if x.startswith('Sector_') and len(x.split('_')) > 2 else 'Unknown'
            )
        else:
            print("Cannot create sector visualizations: Missing required data.")
            return None
    
    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "sectors"
    io.ensure_dir(output_dir)
    
    # 1. Create bar chart comparing performance across sectors
    # Group by sector and get average metrics
    sector_perf = metrics_df.groupby('sector').agg({
        'RMSE': 'mean',
        'R2': 'mean',
        'n_companies': 'mean'  # All models for a sector have the same count
    }).reset_index()
    
    # Sort by RMSE
    sector_perf = sector_perf.sort_values('RMSE')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar chart
    bars = ax.bar(sector_perf['sector'], sector_perf['RMSE'], 
                 color='#3498db', alpha=0.7)
    
    # Add value labels
    for bar, r2, count in zip(bars, sector_perf['R2'], sector_perf['n_companies']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                f'RMSE: {height:.4f}\nR²: {r2:.4f}\n(n={int(count)})', 
                ha='center', va='bottom', fontsize=10)
    
    ax.set_title('Average Model Performance by Sector', fontsize=14)
    ax.set_ylabel('Mean RMSE (lower is better)')
    ax.set_xlabel('Sector')
    
    # Set rotation for x-axis tick labels (corrected)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    save_figure(fig, "sector_performance_comparison", output_dir)
    
    # 2. Heatmap of model types performance across sectors
    if 'type' not in metrics_df.columns:
        print("Warning: 'type' column not found. Skipping model type heatmap.")
    else:
        try:
            # Pivot to create sector x model_type grid with RMSE values
            pivot_df = metrics_df.pivot_table(
                index='sector', 
                columns='type', 
                values='RMSE',
                aggfunc='mean'
            )
            
            # Sort sectors by overall performance
            sector_order = sector_perf['sector'].tolist()
            pivot_df = pivot_df.reindex(sector_order)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create heatmap
            sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='YlGnBu_r',
                       linewidths=0.5, ax=ax)
            
            ax.set_title('Model Type Performance by Sector (RMSE)', fontsize=14)
            
            plt.tight_layout()
            save_figure(fig, "sector_model_type_heatmap", output_dir)
        except Exception as e:
            print(f"Error creating model type heatmap: {e}")
    
    # 3. Compare overall models vs. sector-specific models
    main_metrics_file = settings.METRICS_DIR / "all_models_comparison.csv"
    if main_metrics_file.exists():
        try:
            main_metrics_df = pd.read_csv(main_metrics_file)
            
            # Filter for relevant model types to compare
            if 'model_type' in main_metrics_df.columns:
                main_models = main_metrics_df[main_metrics_df['model_type'] == 'Linear Regression']
            else:
                main_models = main_metrics_df
            
            # Calculate average metrics for each approach
            main_avg = main_models.mean()
            sector_avg = metrics_df.mean()
            
            # Prepare comparison data
            comparison_data = {
                'Approach': ['Overall Models', 'Sector-Specific Models'],
                'RMSE': [main_avg['RMSE'], sector_avg['RMSE']],
                'MAE': [main_avg['MAE'], sector_avg['MAE']],
                'R2': [main_avg['R2'], sector_avg['R2']]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Create comparison plot
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # RMSE comparison
            ax = axes[0]
            bars = ax.bar(comparison_df['Approach'], comparison_df['RMSE'], 
                         color=['#3498db', '#e74c3c'])
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom')
            
            ax.set_title('RMSE Comparison', fontsize=14)
            ax.set_ylabel('RMSE (lower is better)')
            ax.grid(axis='y', alpha=0.3)
            
            # R2 comparison
            ax = axes[1]
            bars = ax.bar(comparison_df['Approach'], comparison_df['R2'], 
                         color=['#3498db', '#e74c3c'])
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom')
            
            ax.set_title('R² Comparison', fontsize=14)
            ax.set_ylabel('R² (higher is better)')
            ax.grid(axis='y', alpha=0.3)
            
            plt.suptitle('Overall vs. Sector-Specific Model Performance', fontsize=16)
            plt.tight_layout()
            save_figure(fig, "overall_vs_sector_comparison", output_dir)
        except Exception as e:
            print(f"Error creating overall vs. sector comparison: {e}")
    
    # 4. Box plot of model performance metrics by sector
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # RMSE by sector
        ax = axes[0]
        sns.boxplot(x='sector', y='RMSE', data=metrics_df, ax=ax, palette='Blues')
        ax.set_title('RMSE Distribution by Sector', fontsize=14)
        ax.set_xlabel('Sector')
        ax.set_ylabel('RMSE (lower is better)')
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # R2 by sector
        ax = axes[1]
        sns.boxplot(x='sector', y='R2', data=metrics_df, ax=ax, palette='Blues')
        ax.set_title('R² Distribution by Sector', fontsize=14)
        ax.set_xlabel('Sector')
        ax.set_ylabel('R² (higher is better)')
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        save_figure(fig, "sector_performance_boxplots", output_dir)
    except Exception as e:
        print(f"Error creating sector performance boxplots: {e}")
    
    print(f"Sector model comparison plots saved to {output_dir}")
    return fig
"""
def plot_sector_feature_importance():
    
    Create visualizations for sector model feature importance.
    
    # Set up style
    style = setup_visualization_style()
    
    # Check if sector importance data exists
    importance_file = settings.FEATURE_IMPORTANCE_DIR / "sector_feature_importance.pkl"
    if not importance_file.exists():
        print("No sector feature importance data found. Please run sector feature importance analysis first.")
        return None
    
    # Load importance data
    importance_results = io.load_model("sector_feature_importance.pkl", 
                                       settings.FEATURE_IMPORTANCE_DIR)
    
    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "sectors"
    io.ensure_dir(output_dir)
    
    # 1. Sector-specific top features
    # Group models by sector
    sectors = set()
    for model_name in importance_results.keys():
        if "Sector_" in model_name:
            # Extract sector name from model name
            sector = model_name.split("_")[1]
            sectors.add(sector)
    
    # For each sector, create a visualization of top features
    for sector in sectors:
        # Find all models for this sector
        sector_models = {k: v for k, v in importance_results.items() if f"Sector_{sector}_" in k}
        
        if not sector_models:
            continue
        
        # Create consolidated table for this sector
        all_features = set()
        for model_name, importance_df in sector_models.items():
            all_features.update(importance_df['Feature'])
        
        # Create DataFrame
        sector_consolidated = pd.DataFrame(index=list(all_features))
        
        # Add importance values for each model
        for model_name, importance_df in sector_models.items():
            # Use model type as column name (more readable than full name)
            model_type = model_name.replace(f"Sector_{sector}_", "")
            
            # Convert to dictionary for easier lookup
            importance_dict = dict(zip(importance_df['Feature'], importance_df['Importance']))
            
            # Add to consolidated DataFrame
            sector_consolidated[model_type] = sector_consolidated.index.map(lambda x: importance_dict.get(x, 0))
        
        # Add average importance
        sector_consolidated['avg_importance'] = sector_consolidated.mean(axis=1)
        
        # Sort by average importance
        sector_consolidated = sector_consolidated.sort_values('avg_importance', ascending=False)
        
        # Save sector-specific consolidated importance
        sector_consolidated.to_csv(f"{settings.FEATURE_IMPORTANCE_DIR}/sector_{sector}_importance.csv")
        
        # Plot top features for this sector
        fig, ax = plt.subplots(figsize=(12, 10))
        
        top_df = sector_consolidated.head(15)
        
        # Plot horizontal bar chart
        bars = ax.barh(top_df.index[::-1], top_df['avg_importance'][::-1], 
                     color='#3498db', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', va='center')
        
        ax.set_xlabel('Average Importance')
        ax.set_title(f'Top 15 Features for {sector} Sector', fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, f"sector_{sector}_top_features", output_dir)
    
    # 2. Compare top features across sectors
    # Load the consolidated sector importance data for each sector
    sector_top_features = {}
    
    for sector in sectors:
        sector_file = settings.FEATURE_IMPORTANCE_DIR / f"sector_{sector}_importance.csv"
        if sector_file.exists():
            sector_df = pd.read_csv(sector_file, index_col=0)
            # Get top 5 features for this sector
            sector_top_features[sector] = sector_df.head(5).index.tolist()
    
    if sector_top_features:
        # Create a matrix showing overlap of top features
        unique_top_features = set()
        for features in sector_top_features.values():
            unique_top_features.update(features)
        
        # Create DataFrame with 1 if feature is in top 5 for sector, 0 otherwise
        feature_matrix = pd.DataFrame(index=list(unique_top_features), columns=list(sector_top_features.keys()))
        
        for sector, features in sector_top_features.items():
            feature_matrix[sector] = feature_matrix.index.map(lambda x: 1 if x in features else 0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, len(unique_top_features) * 0.5))
        
        sns.heatmap(feature_matrix, cmap='Blues', linewidths=0.5, 
                   linecolor='white', ax=ax, cbar=False)
        
        ax.set_title('Top 5 Features by Sector', fontsize=14)
        ax.set_xlabel('Sector')
        ax.set_ylabel('Feature')
        plt.tight_layout()
        
        save_figure(fig, "sector_top_features_comparison", output_dir)
    
    print(f"Sector feature importance visualizations saved to {output_dir}")
    return importance_results
"""

def visualize_sector_models(run_all=False):
    """
    Run all sector model visualizations.
    
    Parameters:
    -----------
    run_all : bool, default=False
        If True, also run the standard visualization functions from metrics_plots and feature_plots
        but with sector model data.
    """
    print("Generating sector model visualizations...")
    
    # Create sector-specific visualizations
    plot_sector_model_comparison()  # UNCOMMENT THIS LINE
    #plot_sector_feature_importance()
    
    # Add call to the new metrics summary table function
    plot_sector_metrics_summary_table()
    
    '''
    # Optionally run standard visualizations with sector data
    if run_all:
        # Set up sector-specific output directory
        output_dir = settings.VISUALIZATION_DIR / "sectors"
        io.ensure_dir(output_dir)
        
        print("Generating standard model comparison plots for sector models...")
        # Load sector metrics
        metrics_file = settings.METRICS_DIR / "sector_models_metrics.csv"
        if metrics_file.exists():
            metrics_df = pd.read_csv(metrics_file)
            plot_model_comparison(metrics_df)
        
        print("Generating residual plots for sector models...")
        # Load sector residuals
        try:
            residuals = io.load_model("sector_model_residuals.pkl", settings.METRICS_DIR)
            # Call plot_residuals with a path to ensure files are saved to sector directory
            plot_residuals(output_dir=output_dir)
        except:
            print("Could not load sector model residuals.")
        
        print("Generating statistical test plots for sector models...")
        # Load statistical tests
        tests_file = settings.METRICS_DIR / "sector_model_comparison_tests.csv"
        if tests_file.exists():
            tests_df = pd.read_csv(tests_file)
            plot_statistical_tests(tests_df)
        
        print("Generating feature importance plots for sector models...")
        # Load sector importance
        try:
            importance_results = io.load_model("sector_feature_importance.pkl", 
                                             settings.FEATURE_IMPORTANCE_DIR)
            plot_top_features(importance_results)
            plot_feature_importance_by_model(importance_results)
        except:
            print("Could not load sector feature importance data.")
    '''
    
    print("Sector model visualizations completed.")

def plot_sector_metrics_summary_table(metrics_df=None):
    """
    Create a summary table visualization of sector model performance metrics.
    
    Parameters:
    -----------
    metrics_df : pandas.DataFrame, optional
        DataFrame containing sector model metrics. If None, it will be loaded.
    """
    # Set up style
    style = setup_visualization_style()
    
    # Load metrics if not provided
    if metrics_df is None:
        metrics_file = settings.METRICS_DIR / "sector_models_metrics.csv"
        if metrics_file.exists():
            metrics_df = pd.read_csv(metrics_file)
        else:
            print("No sector metrics data found. Please run sector model evaluation first.")
            return None
    
    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "sectors"
    io.ensure_dir(output_dir)
    
    # Select and rename columns for the table
    if 'model_name' in metrics_df.columns:
        metrics_df = metrics_df.rename(columns={'model_name': 'Model'})
    elif 'index' in metrics_df.columns:
        metrics_df = metrics_df.rename(columns={'index': 'Model'})
    
    # Make sure R² is spelled correctly
    if 'R2' in metrics_df.columns and 'R²' not in metrics_df.columns:
        metrics_df = metrics_df.rename(columns={'R2': 'R²'})
    
    # Select only the required columns (excluding sector and type)
    table_columns = ['Model', 'MSE', 'MAE', 'RMSE', 'R²', 'n_companies']
    
    # Filter columns that exist in the DataFrame
    available_columns = [col for col in table_columns if col in metrics_df.columns]
    table_data = metrics_df[available_columns].copy()
    
    # Convert n_companies to integer
    if 'n_companies' in table_data.columns:
        table_data['n_companies'] = table_data['n_companies'].astype(int)
    
    # Create figure - increase height for more rows and width for model names
    plt.figure(figsize=(16, len(table_data) * 0.5 + 1))
    
    # Create a table with no cells, just the data
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    # Create cell colors array - initialize with white
    colors = [['white' for _ in range(len(table_data.columns))] for _ in range(len(table_data))]
    
    # Highlight cells with positive R² values
    if 'R²' in table_data.columns:
        r2_col_idx = list(table_data.columns).index('R²')
        for i, row in enumerate(table_data.values):
            r2_value = row[r2_col_idx]
            if r2_value > 0:
                colors[i][r2_col_idx] = '#d9ead3'  # Light green for positive R²
    
    # Convert values to formatted strings
    cell_text = []
    for row in table_data.values:
        row_text = []
        for i, val in enumerate(row):
            col_name = table_data.columns[i]
            if col_name == 'Model':
                # Just use the string value for model name
                row_text.append(str(val))
            elif col_name == 'n_companies':
                # Format as integer
                row_text.append(f"{int(val)}")
            elif isinstance(val, (int, float, np.number)):
                # Format other numeric columns
                row_text.append(f"{val:.4f}")
            else:
                row_text.append(str(val))
        cell_text.append(row_text)
    
    # Set column widths, making Model column wider
    col_widths = [0.4 if table_data.columns[i] == 'Model' else 0.12 for i in range(len(table_data.columns))]
    
    # Create the table with adjusted column widths
    table = plt.table(
        cellText=cell_text,
        colLabels=table_data.columns,
        cellColours=colors,
        cellLoc='center',
        loc='center',
        colWidths=col_widths
    )
    
    # Adjust font size and spacing
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    plt.title('Sector Models Performance Metrics Summary', fontsize=16, pad=20)
    plt.tight_layout()
    
    # Save the figure
    save_figure(plt.gcf(), "sector_metrics_summary_table", output_dir)
    plt.close()
    
    print(f"Sector metrics summary table saved to {output_dir}")
    return table_data

if __name__ == "__main__":
    # Run all sector visualizations
    visualize_sector_models(run_all=True)

