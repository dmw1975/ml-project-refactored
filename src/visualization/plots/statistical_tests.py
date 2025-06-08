"""Statistical test visualizations for model comparison."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except (ImportError, SyntaxError) as e:
    print(f"Warning: Could not import seaborn: {e}")
    sns = None
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

from src.visualization.core.interfaces import ModelData, VisualizationConfig
from src.visualization.core.base import ModelViz, ComparativeViz
from src.visualization.core.registry import get_adapter_for_model
from src.visualization.components.annotations import add_value_labels
from src.visualization.components.layouts import create_grid_layout, create_comparison_layout
from src.visualization.components.formats import format_figure_for_export, save_figure
from src.visualization.utils.io import load_all_models, ensure_dir


class StatisticalTestsPlot(ComparativeViz):
    """Visualizations for statistical test results comparing model performance."""
    
    def __init__(
        self, 
        tests_df: pd.DataFrame,
        config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
    ):
        """
        Initialize statistical tests visualizations.
        
        Args:
            tests_df: DataFrame with statistical test results
            config: Visualization configuration
        """
        # Call parent constructor with empty models list
        super().__init__([], config)
        
        # Store tests DataFrame
        self.tests_df = tests_df
        
    def plot_significance_network(self, threshold: float = 0.05) -> plt.Figure:
        """
        Create a network graph showing significant model differences.
        
        Args:
            threshold: Significance threshold
            
        Returns:
            plt.Figure: Figure with network visualization
        """
        try:
            import networkx as nx
            
            # Filter for significant differences
            sig_tests = self.tests_df[self.tests_df['significant']]
            
            if sig_tests.empty:
                print("No significant differences found.")
                return None
            
            # Create directed graph
            G = nx.DiGraph()
            
            # Add all models as nodes
            all_models = set(self.tests_df['model_a']).union(set(self.tests_df['model_b']))
            for model in all_models:
                G.add_node(model)
            
            # Add edges for significant differences
            for _, row in sig_tests.iterrows():
                # Edge goes from better model to worse model
                better_model = row['better_model']
                other_model = row['model_a'] if better_model == row['model_b'] else row['model_b']
                
                # Add edge with p-value as attribute
                G.add_edge(better_model, other_model, p_value=row['p_value'])
            
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
            
            return plt.gcf()
        
        except ImportError:
            print("NetworkX library not found. Cannot create network visualization.")
            return None

    def plot_significance_matrix(self, dataset_filter: Optional[str] = None) -> plt.Figure:
        """
        Create an enhanced matrix visualization of pairwise statistical test results.
        
        Args:
            dataset_filter: Optional filter for specific dataset
            
        Returns:
            plt.Figure: Figure with significance matrix
        """
        # Get all models
        all_models = sorted(list(set(self.tests_df['model_a']).union(set(self.tests_df['model_b']))))
        
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
        for _, row in self.tests_df.iterrows():
            if row['model_a'] in all_models and row['model_b'] in all_models:
                i = all_models.index(row['model_a'])
                j = all_models.index(row['model_b'])
                p_matrix[i, j] = row['p_value']
                sig_matrix[i, j] = row['significant']
                better_matrix[i, j] = row['better_model']
        
        # Create custom colormap for p-values (white to red)
        cmap = LinearSegmentedColormap.from_list('pvalue_cmap', ['white', '#ffcccc', '#ff9999', '#ff6666', '#ff3333', '#ff0000'])
        
        # Close any existing figures to avoid conflicts
        plt.close('all')
        
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
        
        # Add cell borders to highlight significant differences
        for i in range(n_models):
            for j in range(n_models):
                if i != j and sig_matrix[i, j]:  # Only add borders for significant cells
                    rect = patches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, 
                        linewidth=1.5, 
                        edgecolor='black', 
                        facecolor='none',
                        alpha=0.7
                    )
                    ax.add_patch(rect)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(n_models))
        ax.set_yticks(np.arange(n_models))
        
        # Create shortened model names for display
        display_names = []
        for model in all_models:
            # Determine if model uses random dataset
            is_random = 'Random' in model
            random_indicator = ' R' if is_random else ''
            
            if 'XGB' in model and 'optuna' in model:
                name = 'XGB Opt ' + model.split('_')[1] + random_indicator
            elif 'XGB' in model:
                name = 'XGB Basic ' + model.split('_')[1] + random_indicator
            elif 'LightGBM' in model and 'optuna' in model:
                name = 'LGBM Opt ' + model.split('_')[1] + random_indicator
            elif 'LightGBM' in model:
                name = 'LGBM Basic ' + model.split('_')[1] + random_indicator
            elif 'CatBoost' in model and 'optuna' in model:
                name = 'CB Opt ' + model.split('_')[1] + random_indicator
            elif 'CatBoost' in model:
                name = 'CB Basic ' + model.split('_')[1] + random_indicator
            elif 'ElasticNet' in model:
                name = 'EN ' + model.split('_')[1] + random_indicator
            elif 'LR_' in model:
                name = 'LR ' + model.split('_')[1] + random_indicator
            else:
                name = model + random_indicator
            display_names.append(name)
        
        ax.set_xticklabels(display_names, rotation=45, ha='right')
        ax.set_yticklabels(display_names)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label('-log10(p-value)', rotation=270, labelpad=15)
        
        # Add significance threshold line (p=0.05, -log10(0.05) ≈ 1.301)
        significance_level = -np.log10(0.05)
        
        # Draw a horizontal line on the main plot to indicate significance threshold
        contour_level = np.ones_like(log_p) * significance_level
        # Only show the contour where values exceed the threshold for clarity
        contour_mask = log_p >= significance_level
        contour_level = np.where(contour_mask, significance_level, np.nan)
        ax.contour(np.arange(n_models), np.arange(n_models), contour_level.T, 
                  levels=[significance_level], colors=['black'], linestyles=['--'], linewidths=[1.0])
        
        # Add label to the colorbar
        cbar_y_pos = (significance_level - 0) / (5 - 0)  # Normalized position (vmin=0, vmax=5)
        cbar.ax.axhline(y=cbar_y_pos, color='black', linestyle='--', linewidth=1.0)
        cbar.ax.text(0.5, cbar_y_pos+0.02, 'p=0.05', ha='center', va='bottom', 
                    transform=cbar.ax.transAxes, fontsize=8, backgroundcolor='white', 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        
        # Add only directional arrows for significant differences
        for i in range(n_models):
            for j in range(n_models):
                if i != j:  # Skip diagonal
                    is_significant = sig_matrix[i, j]
                    
                    # Determine text color based on background
                    text_color = 'white' if log_p[i, j] > 2.5 else 'black'
                    
                    # Add arrow to indicate better model (only for significant differences)
                    if is_significant:
                        better = better_matrix[i, j]
                        # Use larger arrows with unicode symbols
                        arrow = "⬆" if better == all_models[i] else "⬇"
                        
                        # Add the arrow with proper formatting
                        ax.text(j, i, arrow, 
                               ha='center', va='center', 
                               color=text_color, 
                               fontsize=12, 
                               fontweight='bold')
        
        # Set title based on dataset filter, including Holm-Bonferroni mention
        if dataset_filter:
            title = f'Statistical Significance Matrix (Holm-Bonferroni Corrected) - {dataset_filter} Dataset'
        else:
            title = 'Statistical Significance Matrix (Holm-Bonferroni Corrected) - All Models'
        
        ax.set_title(title, fontsize=16, pad=20)
        
        # Add improved legend with Holm-Bonferroni mention
        fig.text(0.5, 0.01, 
                 "Cell color intensity shows p-value strength • Black borders indicate significant differences after Holm-Bonferroni correction\n"
                 "⬆ = Row model better than column model, ⬇ = Column model better than row model", 
                 ha='center', fontsize=10)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        return fig

    def plot_win_loss_summary(self) -> plt.Figure:
        """
        Create a win-loss summary visualization for each model.
        
        Returns:
            plt.Figure: Figure with win-loss summary
        """
        # Get significant test results
        sig_tests = self.tests_df[self.tests_df['significant']]
        
        if sig_tests.empty:
            print("No significant differences found.")
            return None
        
        # Count wins and losses for each model
        all_models = set(self.tests_df['model_a']).union(set(self.tests_df['model_b']))
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
        
        # Extract dataset and model information
        win_loss_df['dataset'] = win_loss_df['model'].apply(lambda x: 
                                                       'Base_Random' if 'Base_Random' in x else
                                                       'Yeo_Random' if 'Yeo_Random' in x else
                                                       'Base' if 'Base' in x else
                                                       'Yeo' if 'Yeo' in x else 'Unknown')
        
        win_loss_df['model_family'] = win_loss_df['model'].apply(lambda x: 
                                                        'XGBoost' if 'XGB' in x else
                                                        'LightGBM' if 'LightGBM' in x else
                                                        'CatBoost' if 'CatBoost' in x else
                                                        'Linear' if ('ElasticNet' in x) or ('LR_' in x) else
                                                        'Unknown')
        
        win_loss_df['tuned'] = win_loss_df['model'].apply(lambda x:
                                                  True if ('optuna' in x) or ('ElasticNet' in x) else 
                                                  False)
        
        # Sort by model family, dataset, and tuned status for better grouping
        win_loss_df = win_loss_df.sort_values(['model_family', 'dataset', 'tuned'])
        
        # Create a color map for model families
        family_colors = {
            'XGBoost': '#3498db',
            'LightGBM': '#2ecc71',
            'CatBoost': '#e74c3c',
            'Linear': '#9b59b6',
            'Unknown': '#95a5a6'
        }
        
        # Close any existing figures to avoid conflicts
        plt.close('all')
        
        # Create a plot with grouped bars
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Setup bar positions
        n_models = len(win_loss_df)
        positions = np.arange(n_models)
        
        # Plot bars with colors based on model family
        bars = ax.bar(
            positions, 
            win_loss_df['net'], 
            color=[family_colors[family] for family in win_loss_df['model_family']],
            alpha=0.7
        )
        
        # Add win/loss annotations
        for i, (_, row) in enumerate(win_loss_df.iterrows()):
            offset = 0.3 if row['net'] >= 0 else -0.8
            ax.text(
                i, row['net'] + offset, 
                f"W: {row['wins']}, L: {row['losses']}", 
                ha='center', 
                va='center', 
                fontsize=9, 
                fontweight='bold',
                color='black'
            )
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Set up x-ticks
        ax.set_xticks(positions)
        
        # Create cleaner model labels
        model_labels = []
        for _, row in win_loss_df.iterrows():
            model_name = row['model']
            family = row['model_family']
            dataset = row['dataset']
            tuned = "Tuned" if row['tuned'] else "Basic"
            
            # Shorter labels
            if family == 'XGBoost':
                family_short = 'XGB'
            elif family == 'LightGBM':
                family_short = 'LGBM'
            elif family == 'CatBoost':
                family_short = 'CB'
            elif family == 'Linear':
                family_short = 'LR' if not row['tuned'] else 'EN'
            else:
                family_short = family
                
            # Shorter dataset name
            dataset_short = dataset.replace('_Random', '/R')
            
            # Use tuned only if not Linear family (since LR/EN already indicates tuning)
            tuned_label = "" if family == 'Linear' else f" {tuned}"
            
            model_labels.append(f"{family_short}{tuned_label}\n{dataset_short}")
        
        ax.set_xticklabels(model_labels, rotation=45, ha='right')
        
        # Add title and labels
        ax.set_title('Model Performance Win/Loss Summary (Holm-Bonferroni corrected, α=0.05)', fontsize=16)
        ax.set_ylabel('Net Score (Wins - Losses)', fontsize=14)
        
        # Add legend for model families
        legend_elements = [patches.Patch(facecolor=color, edgecolor='black', label=family) 
                         for family, color in family_colors.items() if family != 'Unknown']
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Adjust y-axis limits to accommodate annotations
        ylim_margin = 1.0
        y_max = win_loss_df['net'].max() + ylim_margin
        y_min = win_loss_df['net'].min() - ylim_margin
        ax.set_ylim(y_min, y_max)
        
        # Add model family group separation lines
        prev_family = None
        for i, family in enumerate(win_loss_df['model_family']):
            if family != prev_family and i > 0:
                ax.axvline(x=i-0.5, color='gray', linestyle='--', alpha=0.5)
            prev_family = family
        
        # Add explanation text
        plt.figtext(0.5, 0.01, 
                   "Net score = Number of models significantly outperformed minus number of models significantly outperforming this model\n"
                   "Models are grouped by family (Linear, XGBoost, LightGBM, CatBoost) and sorted by dataset",
                   ha='center', fontsize=10)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])
        
        return fig
    
    def plot(self) -> Dict[str, plt.Figure]:
        """
        Create all statistical test visualizations.
        
        Returns:
            Dict[str, plt.Figure]: Dictionary of created figures
        """
        figures = {}
        
        # 2. Create significance matrices
        # Overall matrix
        print("Creating significance matrix for all models...")
        overall_matrix = self.plot_significance_matrix()
        if overall_matrix:
            figures['overall_matrix'] = overall_matrix
            
            # Save figure if requested
            if self.config.get('save', True):
                output_dir = self.config.get('output_dir')
                if output_dir is None:
                    # Import settings
                    from pathlib import Path
                    import sys
                    
                    # Add project root to path if needed
                    project_root = Path(__file__).parent.parent.parent.absolute()
                    if str(project_root) not in sys.path:
                        sys.path.append(str(project_root))
                        
                    # Import settings
                    from src.config import settings
                    
                    # Use default output directory
                    output_dir = settings.VISUALIZATION_DIR / "statistical_tests"
                
                # Ensure directory exists
                ensure_dir(output_dir)
                
                # Save figure
                save_figure(
                    fig=overall_matrix,
                    filename="enhanced_significance_matrix",
                    output_dir=output_dir,
                    dpi=self.config.get('dpi', 300),
                    format=self.config.get('format', 'png')
                )
        
        # Dataset-specific matrices (only Base and Yeo - reduced from previous set)
        datasets = ['Base', 'Yeo']  # Removed 'Base_Random' and 'Yeo_Random' for cleaner output
        for dataset in datasets:
            print(f"Creating significance matrix for {dataset} dataset...")
            dataset_matrix = self.plot_significance_matrix(dataset_filter=dataset)
            if dataset_matrix:
                figures[f'{dataset}_matrix'] = dataset_matrix
                
                # Save figure if requested
                if self.config.get('save', True):
                    output_dir = self.config.get('output_dir')
                    if output_dir is None:
                        # Import settings
                        from pathlib import Path
                        import sys
                        
                        # Add project root to path if needed
                        project_root = Path(__file__).parent.parent.parent.absolute()
                        if str(project_root) not in sys.path:
                            sys.path.append(str(project_root))
                            
                        # Import settings
                        from src.config import settings
                        
                        # Use default output directory
                        output_dir = settings.VISUALIZATION_DIR / "statistical_tests"
                    
                    # Ensure directory exists
                    ensure_dir(output_dir)
                    
                    # Save figure
                    save_figure(
                        fig=dataset_matrix,
                        filename=f"enhanced_significance_matrix_{dataset}",
                        output_dir=output_dir,
                        dpi=self.config.get('dpi', 300),
                        format=self.config.get('format', 'png')
                    )
        
        # We no longer generate the win-loss summary
        # Removed for cleaner output
        
        return figures


def plot_statistical_tests(
    tests_df: pd.DataFrame,
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> Dict[str, plt.Figure]:
    """
    Create statistical test visualizations.
    
    Args:
        tests_df: DataFrame with statistical test results
        config: Visualization configuration
        
    Returns:
        Dict[str, plt.Figure]: Dictionary of created figures
    """
    plot = StatisticalTestsPlot(tests_df, config)
    return plot.plot()


def visualize_statistical_tests(
    tests_file: Optional[Union[str, Path]] = None,
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None,
    include_baseline_tests: bool = True
) -> Dict[str, plt.Figure]:
    """
    Load statistical test results and create visualizations.
    
    Args:
        tests_file: Path to CSV file with test results (if None, use default)
        config: Visualization configuration
        include_baseline_tests: Whether to include baseline significance tests using CV results
        
    Returns:
        Dict[str, plt.Figure]: Dictionary of created figures
    """
    # Load tests from file if not provided
    if tests_file is None:
        # Import settings
        import sys
        import os
        
        # Add project root to path if needed
        project_root = Path(__file__).parent.parent.parent.absolute()
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))
            
        # Import settings
        from src.config import settings
        
        tests_file = settings.METRICS_DIR / "model_comparison_tests.csv"
    
    # Clean up any outdated visualizations
    from src.config import settings
    output_dir = settings.VISUALIZATION_DIR / "statistical_tests"
    if output_dir.exists():
        # Remove deprecated matrices
        for deprecated_file in ["enhanced_significance_matrix_Base_Random.png", "enhanced_significance_matrix_Yeo_Random.png"]:
            deprecated_path = output_dir / deprecated_file
            if deprecated_path.exists():
                try:
                    print(f"Removing deprecated visualization: {deprecated_path}")
                    os.remove(deprecated_path)
                except Exception as e:
                    print(f"Error removing {deprecated_path}: {e}")
    
    # Dictionary to store figures
    figures = {}
    
    # Generate pairwise comparison visualizations
    if Path(tests_file).exists():
        # Load tests
        tests_df = pd.read_csv(tests_file)
        
        # Create visualizations
        print("Generating enhanced significance matrix visualizations...")
        pairwise_figures = plot_statistical_tests(tests_df, config)
        
        # Add to figures dictionary
        figures.update(pairwise_figures)
    else:
        print(f"Statistical tests file not found: {tests_file}")
        print("Pairwise comparison visualizations will be skipped.")
    
    # Add baseline significance tests if requested
    if include_baseline_tests:
        try:
            print("Generating baseline significance tests using CV results...")
            from src.evaluation.baseline_significance import run_baseline_significance_analysis
            from src.utils import io
            
            # Load all models
            all_models = io.load_all_models()
            
            if all_models:
                # Check if XGBoost models are missing
                has_xgboost = any('XGB' in model_name for model_name in all_models)
                
                if not has_xgboost:
                    print("WARNING: No XGBoost models found. Continuing analysis with available models.")
                    print("To train XGBoost models, run: python main.py --train-xgboost")
                
                # Run the analysis with available models
                _, baseline_plots = run_baseline_significance_analysis(all_models)
                
                # Add to figures dictionary
                if baseline_plots:
                    for name, fig in baseline_plots.items():
                        figures[f'baseline_{name}'] = fig
                    print("Baseline significance tests completed.")
                else:
                    print("No baseline significance plots were generated.")
                    print("This may be because cross-validation metrics are not available in the models.")
                    print("Check that the models were trained with cross-validation.")
            else:
                print("No models found. Baseline significance tests will be skipped.")
                print("You may need to train models first with:")
                print("  python main.py --train --train-xgboost --train-lightgbm --train-catboost")
        except Exception as e:
            print(f"Error generating baseline significance tests: {e}")
            import traceback
            traceback.print_exc()
    
    return figures