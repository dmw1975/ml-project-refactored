"""Enhanced visualization of statistical test results (DEPRECATED).

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
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from config import settings
from visualization.style import setup_visualization_style, save_figure
from utils import io

def plot_significance_network(tests_df, output_dir=None, threshold=0.05, save=True):
    """
    Create a network graph showing significant model differences.
    
    Args:
        tests_df: DataFrame of statistical test results
        output_dir: Directory to save visualization
        threshold: Significance threshold
        save: Whether to save the plot
    """
    try:
        import networkx as nx
        
        # Set up style
        style = setup_visualization_style()
        
        # Filter for significant differences
        sig_tests = tests_df[tests_df['significant']]
        
        if sig_tests.empty:
            print("No significant differences found.")
            return None
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add all models as nodes
        all_models = set(tests_df['model_a']).union(set(tests_df['model_b']))
        for model in all_models:
            G.add_node(model)
        
        # Add edges for significant differences
        for _, row in sig_tests.iterrows():
            # Edge goes from better model to worse model
            better_model = row['better_model']
            other_model = row['model_a'] if better_model == row['model_b'] else row['model_b']
            
            # Add edge with p-value as attribute
            G.add_edge(better_model, other_model, p_value=row['p_value'])
        
        # Set up output directory if not provided
        if output_dir is None:
            # Save to performance directory instead of statistical_tests directory
            output_dir = settings.VISUALIZATION_DIR / "performance"
            io.ensure_dir(output_dir)
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, seed=42, k=0.5)
        
        # Draw nodes with different colors by model family
        node_colors = []
        node_shapes = []
        for node in G.nodes():
            # Determine if model is basic or tuned
            is_tuned = False
            if 'ElasticNet' in node or 'optuna' in node:
                is_tuned = True
                
            # Assign color by model family
            if 'XGB' in node:
                node_colors.append('#3498db')  # blue for XGBoost family
            elif 'LightGBM' in node:
                node_colors.append('#2ecc71')  # green for LightGBM family
            elif 'CatBoost' in node:
                node_colors.append('#e74c3c')  # red for CatBoost family
            elif 'ElasticNet' in node or 'LR_' in node:
                node_colors.append('#9b59b6')  # purple for Linear family
            else:
                node_colors.append('#95a5a6')  # gray for unknown
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=node_colors, alpha=0.8)
        
        # Draw edges with varying width based on significance (lower p-value = thicker edge)
        edge_widths = []
        edge_colors = []
        for u, v, data in G.edges(data=True):
            # Thicker edges for more significant differences
            p_value = data['p_value']
            width = 1 + (0.05 - min(p_value, 0.05)) * 50
            edge_widths.append(width)
            
            # Set color based on p-value
            if p_value < 0.001:
                edge_colors.append('#e74c3c')  # red for highly significant
            elif p_value < 0.01:
                edge_colors.append('#f39c12')  # orange for moderately significant
            else:
                edge_colors.append('#95a5a6')  # gray for less significant
        
        # Draw edges with arrows
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, 
                              arrowstyle='->', arrowsize=15, alpha=0.7)
        
        # Draw node labels with smaller font
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        # Add legend for model families
        legend_elements = [
            patches.Patch(facecolor='#3498db', edgecolor='black', label='XGBoost Family'),
            patches.Patch(facecolor='#2ecc71', edgecolor='black', label='LightGBM Family'),
            patches.Patch(facecolor='#e74c3c', edgecolor='black', label='CatBoost Family'),
            patches.Patch(facecolor='#9b59b6', edgecolor='black', label='Linear Family'),
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Add title
        plt.title('Model Superiority Network (Holm-Bonferroni corrected, α=0.05)', fontsize=16)
        
        # Add explanation text
        plt.figtext(0.5, 0.01, 
                   'Arrows point from better to worse models. Edge thickness represents significance level.',
                   ha='center', fontsize=12)
        
        # Adjust margins
        plt.tight_layout()
        
        # Save figure to performance directory
        if save:
            # Save to performance directory instead of statistical_tests directory
            perf_dir = settings.VISUALIZATION_DIR / "performance"
            io.ensure_dir(perf_dir)
            
            filename = "model_significance_network"
            save_figure(plt.gcf(), filename, perf_dir)
            print(f"Saved significance network to {perf_dir}/{filename}.png")
        
        return plt.gcf()
    
    except ImportError:
        print("NetworkX library not found. Cannot create network visualization.")
        return None

def plot_significance_matrix(tests_df, output_dir=None, dataset_filter=None, save=True):
    """
    Create an enhanced matrix visualization of pairwise statistical test results.
    
    Args:
        tests_df: DataFrame of statistical test results
        output_dir: Directory to save visualization
        dataset_filter: Optional filter for specific dataset
        save: Whether to save the plot
    """
    # Set up style
    style = setup_visualization_style()
    
    # Set up output directory if not provided
    if output_dir is None:
        # Save to performance directory instead of statistical_tests directory
        output_dir = settings.VISUALIZATION_DIR / "performance"
        io.ensure_dir(output_dir)
    
    # Get all models
    all_models = sorted(list(set(tests_df['model_a']).union(set(tests_df['model_b']))))
    
    # Filter by dataset if specified
    if dataset_filter:
        all_models = [m for m in all_models if dataset_filter in m]
    
    if len(all_models) < 2:
        print(f"Not enough models to compare for dataset {dataset_filter}")
        return None
    
    # Create matrix of p-values
    n_models = len(all_models)
    p_matrix = np.ones((n_models, n_models))
    sig_matrix = np.zeros((n_models, n_models), dtype=bool)
    better_matrix = np.empty((n_models, n_models), dtype=object)
    
    # Fill matrices
    for _, row in tests_df.iterrows():
        if row['model_a'] in all_models and row['model_b'] in all_models:
            i = all_models.index(row['model_a'])
            j = all_models.index(row['model_b'])
            p_matrix[i, j] = row['p_value']
            sig_matrix[i, j] = row['significant']
            better_matrix[i, j] = row['better_model']
    
    # Create custom colormap for p-values (white to red)
    cmap = LinearSegmentedColormap.from_list('pvalue_cmap', ['white', '#ffcccc', '#ff9999', '#ff6666', '#ff3333', '#ff0000'])
    
    # Set up figure with gridspec to support annotations
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    
    # Create heatmap for p-values
    with np.errstate(divide='ignore'):
        log_p = -np.log10(p_matrix)
    np.fill_diagonal(log_p, 0)  # Set diagonal to 0 (white)
    
    # Plot heatmap
    im = ax.imshow(log_p, cmap=cmap, vmin=0, vmax=5)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_models))
    ax.set_yticks(np.arange(n_models))
    
    # Create shortened model names for display
    display_names = []
    for model in all_models:
        # Parse the dataset part and add 'R' for Random datasets
        dataset_part = model.split('_')[1] if '_' in model else ""
        if 'Random' in dataset_part:
            # Mark datasets containing random variable with "R"
            dataset_label = 'R' if dataset_part else ""
        else:
            dataset_label = dataset_part
            
        if 'XGB' in model and 'optuna' in model:
            name = 'XGB Opt ' + dataset_label
        elif 'XGB' in model:
            name = 'XGB ' + dataset_label
        elif 'LightGBM' in model and 'optuna' in model:
            name = 'LGBM Opt ' + dataset_label
        elif 'LightGBM' in model:
            name = 'LGBM ' + dataset_label
        elif 'CatBoost' in model and 'optuna' in model:
            name = 'CB Opt ' + dataset_label
        elif 'CatBoost' in model:
            name = 'CB ' + dataset_label
        elif 'ElasticNet' in model:
            name = 'EN ' + dataset_label
        elif 'LR_' in model:
            name = 'LR ' + dataset_label
        else:
            name = model
        display_names.append(name)
    
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.set_yticklabels(display_names)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('-log10(p-value)', rotation=270, labelpad=15)
    
    # Add significance level lines to the colorbar
    cbar.ax.axhline(y=-np.log10(0.05), color='black', linestyle='--', linewidth=1)
    cbar.ax.text(1.5, -np.log10(0.05), 'α=0.05', va='center', ha='left', fontsize=8)
    
    cbar.ax.axhline(y=-np.log10(0.01), color='black', linestyle='--', linewidth=1)
    cbar.ax.text(1.5, -np.log10(0.01), 'α=0.01', va='center', ha='left', fontsize=8)
    
    cbar.ax.axhline(y=-np.log10(0.001), color='black', linestyle='--', linewidth=1)
    cbar.ax.text(1.5, -np.log10(0.001), 'α=0.001', va='center', ha='left', fontsize=8)
    
    # Add p-value annotations with significance indicators and better model
    for i in range(n_models):
        for j in range(n_models):
            if i != j:  # Skip diagonal
                p_val = p_matrix[i, j]
                
                # Remove 'p' in the chart to get additional space
                if p_val < 0.001:
                    p_text = "<0.001"
                elif p_val < 0.01:
                    p_text = f"{p_val:.3f}"
                else:
                    p_text = f"{p_val:.2f}"
                
                # Determine text color based on background
                text_color = 'white' if log_p[i, j] > 2.5 else 'black'
                
                # Add arrow to indicate better model
                if sig_matrix[i, j]:
                    better = better_matrix[i, j]
                    arrow = "↑" if better == all_models[i] else "↓"
                    p_text += f" {arrow}"
                    
                    # Add rectangle border for significant differences instead of asterisk
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, 
                                       edgecolor='white', linewidth=2)
                    ax.add_patch(rect)
                
                ax.text(j, i, p_text, ha='center', va='center', color=text_color, fontsize=8)
    
    # Set title based on dataset filter
    if dataset_filter:
        # Use more concise display name for dataset if available
        display_name = display_datasets.get(dataset_filter, dataset_filter) if 'display_datasets' in locals() else dataset_filter
        title = f'Statistical Significance Matrix - {display_name} Dataset'
    else:
        title = 'Statistical Significance Matrix - All Models'
    
    ax.set_title(title, fontsize=16, pad=20)
    
    # Add updated legend for arrows and borders
    fig.text(0.5, 0.01, 
             "White border = Significant difference after Holm-Bonferroni correction (α=0.05)\n"
             "↑ = Row model better than column model, ↓ = Column model better than row model\n"
             "R = Dataset containing random variable", 
             ha='center', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figure
    if save:
        filename = f"enhanced_significance_matrix{'_' + dataset_filter if dataset_filter else ''}"
        save_figure(fig, filename, output_dir)
        print(f"Saved enhanced significance matrix to {output_dir}/{filename}.png")
    
    return fig

def plot_win_loss_summary(tests_df, output_dir=None, save=True):
    """
    Create a win-loss summary visualization for each model.
    
    Args:
        tests_df: DataFrame of statistical test results
        output_dir: Directory to save visualization
        save: Whether to save the plot
    """
    # Set up style
    style = setup_visualization_style()
    
    # Set up output directory if not provided
    if output_dir is None:
        # Save to performance directory instead of statistical_tests directory
        output_dir = settings.VISUALIZATION_DIR / "performance"
        io.ensure_dir(output_dir)
    
    # Get significant test results
    sig_tests = tests_df[tests_df['significant']]
    
    if sig_tests.empty:
        print("No significant differences found.")
        return None
    
    # Count wins and losses for each model
    all_models = set(tests_df['model_a']).union(set(tests_df['model_b']))
    win_loss = {model: {'wins': 0, 'losses': 0} for model in all_models}
    
    for _, row in sig_tests.iterrows():
        better = row['better_model']
        worse = row['model_a'] if better == row['model_b'] else row['model_b']
        
        win_loss[better]['wins'] += 1
        win_loss[worse]['losses'] += 1
    
    # Calculate net score (wins - losses)
    for model in win_loss:
        win_loss[model]['net'] = win_loss[model]['wins'] - win_loss[model]['losses']
    
    # Convert to DataFrame for easier plotting
    win_loss_df = pd.DataFrame.from_dict(win_loss, orient='index')
    win_loss_df.reset_index(inplace=True)
    win_loss_df.rename(columns={'index': 'model'}, inplace=True)
    
    # Extract dataset and model type
    win_loss_df['dataset'] = win_loss_df['model'].apply(lambda x: 
                                                     'Base R' if 'Base_Random' in x else  # Mark with 'R' for random datasets
                                                     'Yeo R' if 'Yeo_Random' in x else
                                                     'Base' if 'Base' in x else
                                                     'Yeo' if 'Yeo' in x else 'Unknown')
    
    win_loss_df['model_type'] = win_loss_df['model'].apply(lambda x: 
                                                      'XGBoost' if 'XGB' in x else
                                                      'LightGBM' if 'LightGBM' in x else
                                                      'CatBoost' if 'CatBoost' in x else
                                                      'Linear Regression' if 'LR_' in x else
                                                      'ElasticNet' if 'ElasticNet' in x else
                                                      'Unknown')
    
    win_loss_df['tuned'] = win_loss_df['model'].apply(lambda x:
                                                True if ('optuna' in x) or ('ElasticNet' in x) else 
                                                False)
    
    win_loss_df['optimizer'] = win_loss_df['model'].apply(lambda x:
                                                     'Optuna' if 'optuna' in x else
                                                     'Basic')
    
    # Sort by net score (descending)
    win_loss_df = win_loss_df.sort_values('net', ascending=False)
    
    # Create a color map for model types
    model_colors = {
        'XGBoost': '#3498db',
        'LightGBM': '#2ecc71',
        'CatBoost': '#e74c3c',
        'ElasticNet': '#f39c12',
        'Linear Regression': '#9b59b6',
        'Unknown': '#95a5a6'
    }
    
    # Map colors to DataFrame
    win_loss_df['color'] = win_loss_df['model_type'].map(model_colors)
    
    # Plot win-loss chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot bars
    bars = ax.bar(win_loss_df['model'], win_loss_df['net'], color=win_loss_df['color'])
    
    # Add win/loss annotations
    for i, (_, row) in enumerate(win_loss_df.iterrows()):
        ax.text(i, row['net'] + np.sign(row['net']) * 0.3, 
                f"W: {row['wins']}, L: {row['losses']}", 
                ha='center', va='center', fontsize=9, 
                color='black' if row['net'] > 0 else 'white' if row['net'] < 0 else 'black',
                fontweight='bold')
    
    # Add model type and dataset as markers on x-axis
    for i, (_, row) in enumerate(win_loss_df.iterrows()):
        optimizer = "Opt" if row['optimizer'] == 'Optuna' else "Basic" if row['optimizer'] == 'Basic' else ""
        dataset = row['dataset']
        
        # Add model type and optimizer
        model_text = f"{row['model_type']} {optimizer}"
        ax.text(i, -max(abs(win_loss_df['net'].max()), abs(win_loss_df['net'].min())) - 1, 
                model_text, ha='center', va='top', rotation=45, fontsize=8)
        
        # Add dataset
        ax.text(i, -max(abs(win_loss_df['net'].max()), abs(win_loss_df['net'].min())) - 2.5, 
                dataset, ha='center', va='top', rotation=45, fontsize=8)
    
    # Hide x-tick labels (we're using custom annotations)
    ax.set_xticklabels([])
    ax.set_xticks([])
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add title and labels
    ax.set_title('Model Performance Win/Loss Summary (Holm-Bonferroni corrected, α=0.05)', fontsize=16)
    ax.set_ylabel('Net Score (Wins - Losses)', fontsize=14)
    
    # Add legend for model types
    legend_elements = [patches.Patch(facecolor=color, edgecolor='black', label=model_type) 
                      for model_type, color in model_colors.items() if model_type != 'Unknown']
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Adjust y-axis limits to accommodate annotations
    y_max = win_loss_df['net'].max() + 2
    y_min = min(win_loss_df['net'].min() - 2, -max(abs(win_loss_df['net'].max()), abs(win_loss_df['net'].min())) - 3)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save figure
    if save:
        # Save to performance directory instead of statistical_tests directory
        perf_dir = settings.VISUALIZATION_DIR / "performance"
        io.ensure_dir(perf_dir)
        
        save_figure(fig, "model_win_loss_summary", perf_dir)
        print(f"Saved win-loss summary to {perf_dir}/model_win_loss_summary.png")
    
    return fig

def visualize_statistical_tests(tests_df=None, output_dir=None):
    """
    Create comprehensive visualizations of statistical test results.
    
    Args:
        tests_df: DataFrame of statistical test results
        output_dir: Directory to save visualizations
    """
    # Load tests if not provided
    if tests_df is None:
        tests_file = settings.METRICS_DIR / "model_comparison_tests.csv"
        if tests_file.exists():
            tests_df = pd.read_csv(tests_file)
        else:
            print("No statistical tests data found. Please run evaluation first.")
            return None
    
    # Set up output directory - use performance directory instead
    if output_dir is None:
        output_dir = settings.VISUALIZATION_DIR / "performance"
        io.ensure_dir(output_dir)
    
    # Count significant differences
    sig_count = tests_df['significant'].sum()
    total_count = len(tests_df)
    
    print(f"Found {sig_count} significant differences out of {total_count} comparisons "
          f"({sig_count/total_count:.1%}) after Holm-Bonferroni correction.")
    
    figures = {}
    
    # 1. Create significance network
    print("Creating model significance network...")
    network_fig = plot_significance_network(tests_df, output_dir)
    if network_fig:
        figures['network'] = network_fig
    
    # 2. Create significance matrices
    # Overall matrix
    print("Creating overall significance matrix...")
    overall_matrix = plot_significance_matrix(tests_df, output_dir)
    figures['overall_matrix'] = overall_matrix
    
    # Dataset-specific matrices
    datasets = ['Base', 'Base_Random', 'Yeo', 'Yeo_Random']
    
    # Rename datasets to be more concise, using 'R' to indicate random variable
    display_datasets = {'Base': 'Base', 'Base_Random': 'Base R', 'Yeo': 'Yeo', 'Yeo_Random': 'Yeo R'}
    for dataset in datasets:
        print(f"Creating significance matrix for {dataset} dataset...")
        dataset_matrix = plot_significance_matrix(tests_df, output_dir, dataset_filter=dataset)
        if dataset_matrix:
            figures[f'{dataset}_matrix'] = dataset_matrix
    
    # 3. Create win-loss summary
    print("Creating win-loss summary...")
    win_loss_fig = plot_win_loss_summary(tests_df, output_dir)
    if win_loss_fig:
        figures['win_loss'] = win_loss_fig
    
    print(f"Statistical test visualizations saved to {output_dir}")
    return figures

if __name__ == "__main__":
    # Run all visualizations
    visualize_statistical_tests()