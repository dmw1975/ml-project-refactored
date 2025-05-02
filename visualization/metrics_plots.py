"""Visualization functions for model metrics and performance (DEPRECATED).

This module is deprecated and will be removed in a future version.
Please use visualization_new package instead.
"""

import warnings

warnings.warn(
    "This module is deprecated. Please use visualization_new.plots.metrics instead.",
    DeprecationWarning,
    stacklevel=2
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from config import settings
from visualization.style import setup_visualization_style, save_figure
from utils import io

def mean_confidence_interval(data, confidence=0.95):
    """Calculate mean and confidence interval."""
    import scipy.stats as st
    a = np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def plot_elasticnet_cv_distribution(cv_results=None):
    """
    Plot distribution of cross-validation RMSE scores for ElasticNet models.
    
    Parameters:
    -----------
    cv_results : list, optional
        List of CV result dictionaries. If None, it will be loaded.
    """
    # Import required modules
    import seaborn as sns
    from matplotlib.lines import Line2D
    
    # Set up style
    style = setup_visualization_style()
    
    # Load CV results if not provided
    if cv_results is None:
        try:
            cv_results = io.load_model("elasticnet_params.pkl", settings.MODEL_DIR)
        except:
            print("No ElasticNet cross-validation results found.")
            return
    
    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "performance"
    io.ensure_dir(output_dir)
    
    # Prepare data for the plot
    rmse_data = []
    for result in cv_results:
        dataset = result['dataset']
        cv_df = result['cv_results']
        
        # Get fold RMSEs for each parameter combination
        for _, row in cv_df.iterrows():
            rmse_data.append({
                'Dataset': dataset,
                'RMSE': row['mean_rmse'],
                'Alpha': row['alpha'],
                'L1_Ratio': row['l1_ratio']
            })
    
    rmse_df = pd.DataFrame(rmse_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Boxplot for RMSE distribution
    box = sns.boxplot(x='Dataset', y='RMSE', data=rmse_df, palette='pastel', ax=ax)
    
    # Stripplot for individual fold RMSEs
    strip = sns.stripplot(x='Dataset', y='RMSE', data=rmse_df, color='gray', alpha=0.6, jitter=True, ax=ax)
    
    # Plot mean and 95% CI as red points with error bars
    for i, dataset in enumerate(rmse_df['Dataset'].unique()):
        rmse_vals = rmse_df[rmse_df['Dataset'] == dataset]['RMSE']
        mean = np.mean(rmse_vals)
        ci_low, ci_high = mean_confidence_interval(rmse_vals)[1:]
        err = ax.errorbar(i, mean, yerr=[[mean - ci_low], [ci_high - mean]],
                          fmt='o', color='red', capsize=5, 
                          label='Mean ± 95% CI' if i == 0 else "")
    
    # Title and axes
    ax.set_title('ElasticNet RMSE Distribution per Dataset', fontsize=14)
    ax.set_ylabel('RMSE (lower is better)')
    ax.set_xlabel('Dataset')
    plt.xticks(rotation=15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='RMSE Distribution (Boxplot)',
               markerfacecolor='lightblue', markersize=15),
        Line2D([0], [0], marker='o', color='gray', label='Individual CV Fold RMSE',
               linestyle='None', markersize=8, alpha=0.6),
        Line2D([0], [0], marker='o', color='red', label='Mean ± 95% CI',
               linestyle='None', markersize=8)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    save_figure(fig, "elasticnet_cv_rmse_distribution", output_dir)
    
    # Plot the best parameters distribution
    plot_elasticnet_best_params(cv_results, output_dir)
    
    print(f"ElasticNet CV distribution plot saved to {output_dir}")
    return fig

def plot_elasticnet_best_params(cv_results, output_dir=None):
    """
    Plot best parameters for ElasticNet models.
    
    Parameters:
    -----------
    cv_results : list
        List of CV result dictionaries.
    output_dir : Path, optional
        Directory to save plot. If None, it will use the default.
    """
    # Import required modules
    import seaborn as sns
    if output_dir is None:
        output_dir = settings.VISUALIZATION_DIR / "performance"
        io.ensure_dir(output_dir)
    
    # Extract best parameters
    best_params = []
    for result in cv_results:
        alpha, l1_ratio = result['best_params']
        best_params.append({
            'Dataset': result['dataset'],
            'Alpha': alpha,
            'L1_Ratio': l1_ratio
        })
    
    best_df = pd.DataFrame(best_params)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Alpha plot
    ax = axes[0]
    sns.barplot(x='Dataset', y='Alpha', data=best_df, ax=ax, palette='Blues')
    ax.set_title('Best Alpha by Dataset')
    ax.set_ylabel('Alpha Value')
    ax.set_xlabel('Dataset')
    ax.tick_params(axis='x', rotation=15)
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'bottom',
                    xytext = (0, 5), textcoords = 'offset points')
    
    # L1 Ratio plot
    ax = axes[1]
    sns.barplot(x='Dataset', y='L1_Ratio', data=best_df, ax=ax, palette='Greens')
    ax.set_title('Best L1 Ratio by Dataset')
    ax.set_ylabel('L1 Ratio')
    ax.set_xlabel('Dataset')
    ax.tick_params(axis='x', rotation=15)
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'bottom',
                    xytext = (0, 5), textcoords = 'offset points')
    
    plt.tight_layout()
    save_figure(fig, "elasticnet_best_parameters", output_dir)
    
    return fig

def plot_model_comparison(metrics_df=None):
    """
    Create high-quality comparison plots of model performance metrics for thesis publication.
    
    Parameters:
    -----------
    metrics_df : pandas.DataFrame, optional
        DataFrame containing model metrics. If None, it will be loaded.
    """
    import matplotlib.ticker as ticker
    from matplotlib import rcParams

    # Setup style
    style = setup_visualization_style()
    
    # Load metrics if not provided
    if metrics_df is None:
        metrics_file = settings.METRICS_DIR / "all_models_comparison.csv"
        if metrics_file.exists():
            metrics_df = pd.read_csv(metrics_file)
        else:
            print("No metrics data found. Please run evaluation first.")
            return
    
    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "performance"
    io.ensure_dir(output_dir)

    # Increase global font size for thesis quality
    rcParams.update({'font.size': 14})

    # Sort models by RMSE ascending
    sorted_df = metrics_df.sort_values('RMSE', ascending=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Set color palette (academic style)
    palette = sns.color_palette("Set2", n_colors=len(sorted_df))

    # Barplot
    bars = ax.bar(sorted_df['model_name'], sorted_df['RMSE'], color=palette)

    # Title and labels
    ax.set_title('Root Mean Squared Error (RMSE) Comparison Across Models', fontsize=18, pad=20)
    ax.set_ylabel('RMSE (lower is better)', fontsize=16)
    ax.set_xlabel('Model', fontsize=16)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Tighten Y axis to focus
    ax.set_ylim(0, sorted_df['RMSE'].max() * 1.2)

    # Format Y axis to 2 decimals
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # Highlight best model (lowest RMSE) with a star
    min_rmse = sorted_df['RMSE'].min()
    min_model = sorted_df.loc[sorted_df['RMSE'].idxmin(), 'model_name']
    for bar, model_name in zip(bars, sorted_df['model_name']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.03,
                f'{height:.4f}', ha='center', va='bottom', fontsize=12)
        if model_name == min_model:
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.15,
                    '*', ha='center', va='bottom', fontsize=20, color='red')

    # Legend manually outside
    ax.legend([bars[0]], ['* denotes best performing model'], loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=1, frameon=False, fontsize=12)

    plt.tight_layout()
    save_figure(fig, "thesis_model_comparison_rmse", output_dir, dpi=300, format='png')

    print(f"High-quality RMSE model comparison plot saved to {output_dir}")
    return fig


def plot_residuals(output_dir=None, best_model_name=None, top_n=4):
    """
    Plot thesis-quality residuals analysis for models.
    
    This function is now a wrapper that calls create_residual_plots.py to ensure consistent residual plots.
    
    Parameters:
    -----------
    output_dir : Path or str, optional
        Directory for saving plots. If None, default directory is used.
    best_model_name : str, optional
        Name of specific model to plot. If None, best model by RMSE is used.
    top_n : int, default=4
        Number of models to include in multi-model plot.
    """
    from visualization.create_residual_plots import create_thesis_residual_plot, load_all_models
    
    print("Generating residual plots using create_residual_plots.py...")
    
    # Set default output directory
    if output_dir is None:
        output_dir = settings.VISUALIZATION_DIR / "residuals"
    io.ensure_dir(output_dir)
    
    try:
        # Load all models
        all_models = load_all_models()
        
        if not all_models:
            print("No models found. Please train models first.")
            return None
        
        # Determine which model to plot
        if best_model_name is None:
            # Find best model by loading metrics
            metrics_file = settings.METRICS_DIR / "all_models_comparison.csv"
            if metrics_file.exists():
                metrics_df = pd.read_csv(metrics_file)
                if not metrics_df.empty:
                    best_model_name = metrics_df.sort_values('RMSE').iloc[0]['model_name']
            
            # If still None, use first model
            if best_model_name is None:
                best_model_name = list(all_models.keys())[0]
        
        # Check if model exists
        if best_model_name not in all_models:
            print(f"Model {best_model_name} not found. Available models:")
            for model in all_models.keys():
                print(f"  - {model}")
            
            # Use first model instead
            best_model_name = list(all_models.keys())[0]
            print(f"Using {best_model_name} instead.")
        
        # Create residual plot for the best model
        print(f"Creating residual plot for {best_model_name}...")
        
        plot_path = create_thesis_residual_plot(best_model_name, all_models[best_model_name], output_dir)
        print(f"Residual plot saved to {plot_path}")
        
        return None
    
    except Exception as e:
        print(f"Error generating residual plots: {e}")
        import traceback
        traceback.print_exc()
        return None



# def plot_statistical_tests(tests_df=None):
#     """
#     Plot visualization of statistical comparison tests between models with Holm-Bonferroni correction.
    
#     Parameters:
#     -----------
#     tests_df : pandas.DataFrame, optional
#         DataFrame of test results. If None, it will be loaded.
#     """
#     # Set up style
#     style = setup_visualization_style()
    
#     # Load tests if not provided
#     if tests_df is None:
#         tests_file = settings.METRICS_DIR / "model_comparison_tests.csv"
#         if tests_file.exists():
#             tests_df = pd.read_csv(tests_file)
#         else:
#             print("No statistical tests data found. Please run evaluation first.")
#             return
    
#     # Set up output directory
#     output_dir = settings.VISUALIZATION_DIR / "performance"
#     io.ensure_dir(output_dir)
    
#     # Create heatmap of p-values
#     all_models = sorted(list(set(tests_df['model_a']).union(set(tests_df['model_b']))))
#     n_models = len(all_models)
    
#     # Create matrix of p-values
#     p_matrix = np.ones((n_models, n_models))
#     # Create matrix for significant comparisons after Holm-Bonferroni
#     sig_matrix = np.zeros((n_models, n_models), dtype=bool)
    
#     for _, row in tests_df.iterrows():
#         i = all_models.index(row['model_a'])
#         j = all_models.index(row['model_b'])
#         p_matrix[i, j] = row['p_value']
#         p_matrix[j, i] = row['p_value']  # Mirror for symmetry
        
#         # Record if comparison is significant after Holm-Bonferroni
#         sig_matrix[i, j] = row['significant']
#         sig_matrix[j, i] = row['significant']  # Mirror for symmetry
    
#     # Mark diagonal as NaN to ignore in heatmap
#     np.fill_diagonal(p_matrix, np.nan)
#     np.fill_diagonal(sig_matrix, False)
    
#     # Create enhanced heatmap figure with better size for thesis
#     fig, ax = plt.subplots(figsize=(12, 10))
    
#     # Use log scale for better visualization
#     with np.errstate(divide='ignore'):
#         log_p = -np.log10(p_matrix)
    
#     # Create masked array for NaN values
#     masked_log_p = np.ma.array(log_p, mask=np.isnan(log_p))
    
#     # Create custom colormap with a clear distinction for significance levels
#     cmap = plt.cm.YlOrRd
    
#     # Create heatmap
#     heatmap = ax.pcolor(masked_log_p, cmap=cmap, vmin=0, vmax=4)
    
#     # Set ticks and labels
#     ax.set_xticks(np.arange(n_models) + 0.5)
#     ax.set_yticks(np.arange(n_models) + 0.5)
#     ax.set_xticklabels(all_models, rotation=45, ha='right', fontsize=10)
#     ax.set_yticklabels(all_models, fontsize=10)
    
#     # Add colorbar
#     cbar = plt.colorbar(heatmap)
#     cbar.set_label('-log10(p-value)', rotation=270, labelpad=20)
    
#     # Add significance level markers to colorbar
#     cbar.ax.plot([0, 1], [1.3, 1.3], 'k-', lw=2)  # p=0.05 line
#     cbar.ax.text(0.5, 1.4, 'p=0.05', ha='center', va='bottom')
    
#     cbar.ax.plot([0, 1], [2, 2], 'k-', lw=2)  # p=0.01 line
#     cbar.ax.text(0.5, 2.1, 'p=0.01', ha='center', va='bottom')
    
#     cbar.ax.plot([0, 1], [3, 3], 'k-', lw=2)  # p=0.001 line
#     cbar.ax.text(0.5, 3.1, 'p=0.001', ha='center', va='bottom')
    
#     # Add p-values and significance indicators to cells
#     for i in range(n_models):
#         for j in range(n_models):
#             if not np.isnan(p_matrix[i, j]):
#                 p_val = safe_float(p_matrix[i, j])
#                 if p_val < 0.001:
#                     p_text = '< 0.001'
#                 else:
#                     p_text = f'{p_val:.3f}'
                
#                 # Add asterisks for significant results after Holm-Bonferroni correction
#                 if sig_matrix[i, j]:
#                     p_text = p_text + '*'  # Add asterisk for significant results
                    
#                 # Determine text color based on background darkness
#                 if log_p[i, j] > 2:  # Darker cells
#                     text_color = 'white'
#                 else:
#                     text_color = 'black'
                
#                 # Draw a box around significant comparisons
#                 if sig_matrix[i, j]:
#                     rect = plt.Rectangle((j, i), 1, 1, fill=False, 
#                                         edgecolor='white', linewidth=2)
#                     ax.add_patch(rect)
                
#                 ax.text(j + 0.5, i + 0.5, p_text, 
#                         ha='center', va='center', color=text_color,
#                         fontweight='bold' if sig_matrix[i, j] else 'normal')
    
#     # Add a note about significance marking
#     fig.text(0.5, 0.01, "* indicates significance after Holm-Bonferroni correction (p < adjusted threshold)",
#             ha='center', fontsize=10, style='italic')
    
#     ax.set_title('Statistical Significance of Model Differences\n(-log10 of p-values from paired t-tests)',
#                  fontsize=14)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to make room for the note
    
#     save_figure(fig, "model_statistical_tests_heatmap", output_dir)
    
#     # Create a second heatmap showing only significant results after correction
#     fig2, ax2 = plt.subplots(figsize=(12, 10))
    
#     # Create a mask for non-significant comparisons
#     sig_mask = ~sig_matrix
#     # Also mask the diagonal
#     np.fill_diagonal(sig_mask, True)
    
#     # Create masked array
#     masked_sig_log_p = np.ma.array(log_p, mask=sig_mask)
    
#     # Create heatmap of only significant results
#     sig_heatmap = ax2.pcolor(masked_sig_log_p, cmap=cmap, vmin=0, vmax=4)
    
#     # Set ticks and labels
#     ax2.set_xticks(np.arange(n_models) + 0.5)
#     ax2.set_yticks(np.arange(n_models) + 0.5)
#     ax2.set_xticklabels(all_models, rotation=45, ha='right', fontsize=10)
#     ax2.set_yticklabels(all_models, fontsize=10)
    
#     # Add colorbar
#     cbar2 = plt.colorbar(sig_heatmap)
#     cbar2.set_label('-log10(p-value)', rotation=270, labelpad=20)
    
#     # Add significance level markers to colorbar
#     cbar2.ax.plot([0, 1], [1.3, 1.3], 'k-', lw=2)  # p=0.05 line
#     cbar2.ax.text(0.5, 1.4, 'p=0.05', ha='center', va='bottom')
    
#     cbar2.ax.plot([0, 1], [2, 2], 'k-', lw=2)  # p=0.01 line
#     cbar2.ax.text(0.5, 2.1, 'p=0.01', ha='center', va='bottom')
    
#     cbar2.ax.plot([0, 1], [3, 3], 'k-', lw=2)  # p=0.001 line
#     cbar2.ax.text(0.5, 3.1, 'p=0.001', ha='center', va='bottom')
    
#     # Add p-values to significant cells only
#     for i in range(n_models):
#         for j in range(n_models):
#             if sig_matrix[i, j]:
#                 p_val = safe_float(p_matrix[i, j])
#                 if p_val < 0.001:
#                     p_text = '< 0.001'
#                 else:
#                     p_text = f'{p_val:.3f}'
                
#                 # Get the better model
#                 better_row = tests_df[(tests_df['model_a'] == all_models[i]) & 
#                                      (tests_df['model_b'] == all_models[j])]
#                 if not better_row.empty:
#                     better_model = better_row.iloc[0]['better_model']
#                     is_i_better = better_model == all_models[i]
#                 else:
#                     better_row = tests_df[(tests_df['model_a'] == all_models[j]) & 
#                                          (tests_df['model_b'] == all_models[i])]
#                     if not better_row.empty:
#                         better_model = better_row.iloc[0]['better_model']
#                         is_i_better = better_model == all_models[i]
#                     else:
#                         is_i_better = False
                
#                 # Add an arrow symbol showing direction of superiority
#                 arrow = '↑' if is_i_better else '↓'
#                 p_text = f"{p_text} {arrow}"
                
#                 # Determine text color based on background darkness
#                 if log_p[i, j] > 2:  # Darker cells
#                     text_color = 'white'
#                 else:
#                     text_color = 'black'
                
#                 ax2.text(j + 0.5, i + 0.5, p_text, 
#                         ha='center', va='center', color=text_color,
#                         fontweight='bold')
    
#     # Add a legend for the arrows
#     fig2.text(0.5, 0.01, "↑ indicates row model is superior to column model\n↓ indicates column model is superior to row model",
#              ha='center', fontsize=10, style='italic')
    
#     ax2.set_title('Significant Model Differences After Holm-Bonferroni Correction',
#                  fontsize=14)
#     plt.tight_layout(rect=[0, 0.05, 1, 0.97])  # Adjust layout to make room for the legend
    
#     save_figure(fig2, "model_significant_differences_heatmap", output_dir)
    
#     # Create superiority network graph if NetworkX is available
#     # (This part remains the same as your existing implementation)
#     # ...rest of the function...
    
#     print(f"Statistical test visualizations saved to {output_dir}")
#     return fig

def plot_statistical_tests_filtered(tests_df=None, allowed_model_types=None):
    """
    Plot filtered statistical comparison heatmaps by dataset, including only serious models.
    
    Parameters:
    -----------
    tests_df : pandas.DataFrame, optional
        DataFrame of test results. If None, it will be loaded.
    allowed_model_types : list, optional
        List of substrings that model names must contain (e.g., ['elasticnet', 'xgb', 'catboost', 'lightgbm']).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib import rcParams

    # Load tests if not provided
    if tests_df is None:
        tests_file = settings.METRICS_DIR / "model_comparison_tests.csv"
        if tests_file.exists():
            tests_df = pd.read_csv(tests_file)
        else:
            print("No statistical tests data found. Please run evaluation first.")
            return

    # Set allowed model types
    if allowed_model_types is None:
        allowed_model_types = ['elasticnet', 'xgb', 'catboost', 'lightgbm']

    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "performance/statistical_tests"
    io.ensure_dir(output_dir)

    # Setup visualization style
    style = setup_visualization_style()
    rcParams.update({'font.size': 14})

    # Filter models based on allowed types (case-insensitive)
    def is_allowed(name):
        name = name.lower()
        return any(allowed_type in name for allowed_type in allowed_model_types)

    models = sorted(list(set(tests_df['model_a']).union(set(tests_df['model_b']))))
    allowed_models = [m for m in models if is_allowed(m)]

    print(f"Filtered models for significance testing:")
    for m in allowed_models:
        print(f"  - {m}")

    # Group models by dataset
    datasets = set()
    for model_name in allowed_models:
        if 'LR_Base' in model_name:
            datasets.add('LR_Base')
        elif 'LR_Yeo' in model_name:
            datasets.add('LR_Yeo')
        elif 'LR_Base_Random' in model_name:
            datasets.add('LR_Base_Random')
        elif 'LR_Yeo_Random' in model_name:
            datasets.add('LR_Yeo_Random')

    # Now create a separate heatmap per dataset
    for dataset in datasets:
        dataset_models = [m for m in allowed_models if dataset in m]

        if len(dataset_models) < 2:
            print(f"Skipping {dataset} — not enough models after filtering.")
            continue

        print(f"\nCreating heatmap for dataset group: {dataset}")

        n_models = len(dataset_models)
        p_matrix = np.ones((n_models, n_models))
        sig_matrix = np.zeros((n_models, n_models), dtype=bool)

        for _, row in tests_df.iterrows():
            if row['model_a'] in dataset_models and row['model_b'] in dataset_models:
                i = dataset_models.index(row['model_a'])
                j = dataset_models.index(row['model_b'])
                p_matrix[i, j] = row['p_value']
                p_matrix[j, i] = row['p_value']
                sig_matrix[i, j] = row['significant']
                sig_matrix[j, i] = row['significant']

        np.fill_diagonal(p_matrix, np.nan)

        fig, ax = plt.subplots(figsize=(12, 10))
        with np.errstate(divide='ignore'):
            log_p = -np.log10(p_matrix)
        masked_log_p = np.ma.array(log_p, mask=np.isnan(log_p))

        cmap = plt.cm.YlOrRd
        heatmap = ax.pcolor(masked_log_p, cmap=cmap, vmin=0, vmax=4)

        ax.set_xticks(np.arange(n_models) + 0.5)
        ax.set_yticks(np.arange(n_models) + 0.5)
        ax.set_xticklabels(dataset_models, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(dataset_models, fontsize=10)

        cbar = plt.colorbar(heatmap)
        cbar.set_label('-log10(p-value)', rotation=270, labelpad=20)

        # Annotate p-values
        for i in range(n_models):
            for j in range(n_models):
                if not np.isnan(p_matrix[i, j]):
                    p_val = p_matrix[i, j]
                    p_text = "< 0.001" if p_val < 0.001 else f"{p_val:.3f}"

                    text_color = 'white' if log_p[i, j] > 2 else 'black'

                    if sig_matrix[i, j]:
                        p_text = p_text + '*'

                    ax.text(j + 0.5, i + 0.5, p_text, ha='center', va='center', color=text_color, fontweight='bold')

        ax.set_title(f'Statistical Significance - {dataset}', fontsize=16, pad=20)
        plt.tight_layout()

        filename = f"model_significant_heatmap_{dataset}"
        save_figure(fig, filename, output_dir, dpi=300, format='png')

        print(f"Saved heatmap: {filename}.png")

        plt.close(fig)

    print("\nAll statistical heatmaps saved successfully.")




def plot_metrics_summary_table(metrics_df=None):
    """
    Create a summary table visualization of model performance metrics.
    
    Parameters:
    -----------
    metrics_df : pandas.DataFrame, optional
        DataFrame containing model metrics. If None, it will be loaded.
    """
    # Set up style
    style = setup_visualization_style()
    
    # Load metrics if not provided
    if metrics_df is None:
        metrics_file = settings.METRICS_DIR / "all_models_comparison.csv"
        if metrics_file.exists():
            metrics_df = pd.read_csv(metrics_file)
        else:
            try:
                # Try loading from evaluation
                from evaluation.metrics import evaluate_models
                eval_results = evaluate_models()
                metrics_df = eval_results['metrics_df']
            except:
                print("No metrics data found. Please run evaluation first.")
                return
    
    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "performance"
    io.ensure_dir(output_dir)
    
    # Select only the necessary columns for the table
    if 'model_name' in metrics_df.columns:
        # Rename model_name to Model if it exists
        metrics_df = metrics_df.rename(columns={'model_name': 'Model'})
    elif 'index' in metrics_df.columns:
        # If index column exists, rename to Model
        metrics_df = metrics_df.rename(columns={'index': 'Model'})
    
    # Select and order columns
    table_columns = ['Model', 'MSE', 'MAE', 'R²', 'RMSE']
    # Make sure R² is spelled correctly
    if 'R2' in metrics_df.columns and 'R²' not in metrics_df.columns:
        metrics_df = metrics_df.rename(columns={'R2': 'R²'})
    
    # Filter columns that exist in the DataFrame
    available_columns = [col for col in table_columns if col in metrics_df.columns]
    table_data = metrics_df[available_columns].copy()
    
    # Create figure
    plt.figure(figsize=(12, len(table_data) * 0.5 + 1))
    
    # Reset index to include model names in the table if not already a column
    if 'Model' not in table_data.columns:
        table_data = table_data.reset_index()
        if 'index' in table_data.columns:
            table_data = table_data.rename(columns={'index': 'Model'})
    
    # Create a table with no cells, just the data
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    # Create cell colors array - initialize with white
    colors = [['white' for _ in range(len(table_data.columns))] for _ in range(len(table_data))]
    
    # Highlight the best value in each metric column
    best_indices = {}
    for col in table_data.columns:
        if col == 'Model':
            continue
        
        # Check if the column contains numerics
        if table_data[col].dtype in [np.float64, np.int64, float, int]:
            # For MSE, MAE, RMSE lower is better
            if col in ['MSE', 'MAE', 'RMSE']:
                best_indices[col] = table_data[col].idxmin()
            # For R², higher is better
            elif col in ['R²']:
                best_indices[col] = table_data[col].idxmax()
    
    # Apply colors to the best values
    for col, idx in best_indices.items():
        col_idx = list(table_data.columns).index(col)
        colors[table_data.index.get_loc(idx)][col_idx] = '#d9ead3'  # Light green
    
    # Convert values to formatted strings
    cell_text = []
    for row in table_data.values:
        row_text = []
        for i, val in enumerate(row):
            if i == 0:  # Model name
                row_text.append(str(val))
            else:  # Numeric value
                if isinstance(val, (int, float, np.number)):
                    row_text.append(f"{val:.4f}")
                else:
                    row_text.append(str(val))
        cell_text.append(row_text)
    
    # Create the table
    table = plt.table(
        cellText=cell_text,
        colLabels=table_data.columns,
        cellColours=colors,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    plt.title('Model Performance Metrics Summary', fontsize=16, pad=20)
    plt.tight_layout()
    
    # Save the figure
    save_figure(plt.gcf(), "metrics_summary_table", output_dir)
    plt.close()
    
    print(f"Metrics summary table saved to {output_dir}")
    return

if __name__ == "__main__":
    # Run all visualizations
    plot_model_comparison()
    plot_residuals()
    plot_statistical_tests_filtered()
    plot_elasticnet_cv_distribution()
    plot_metrics_summary_table()