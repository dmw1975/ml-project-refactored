# """Visualization functions for model metrics and performance."""

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# import sys

# # Add project root to path
# project_root = Path(__file__).parent.parent.absolute()
# sys.path.append(str(project_root))

# from config import settings
# from visualization.style import setup_visualization_style, save_figure
# from utils import io
# from utils.helpers import safe_float  # Import the safe_float function

# def mean_confidence_interval(data, confidence=0.95):
#     """Calculate mean and confidence interval."""
#     import scipy.stats as st
#     a = np.array(data)
#     n = len(a)
#     m, se = np.mean(a), st.sem(a)
#     h = se * st.t.ppf((1 + confidence) / 2., n-1)
#     return m, m-h, m+h

# def plot_elasticnet_cv_distribution(cv_results=None):
#     """
#     Plot distribution of cross-validation RMSE scores for ElasticNet models.
    
#     Parameters:
#     -----------
#     cv_results : list, optional
#         List of CV result dictionaries. If None, it will be loaded.
#     """
#     # Set up style
#     style = setup_visualization_style()
    
#     # Load CV results if not provided
#     if cv_results is None:
#         try:
#             cv_results = io.load_model("elasticnet_params.pkl", settings.MODEL_DIR)
#         except:
#             print("No ElasticNet cross-validation results found.")
#             return
    
#     # Set up output directory
#     output_dir = settings.VISUALIZATION_DIR / "performance"
#     io.ensure_dir(output_dir)
    
#     # Prepare data for the plot
#     rmse_data = []
#     for result in cv_results:
#         dataset = result['dataset']
#         cv_df = result['cv_results']
        
#         # Get fold RMSEs for each parameter combination
#         for _, row in cv_df.iterrows():
#             rmse_data.append({
#                 'Dataset': dataset,
#                 'RMSE': row['mean_rmse'],
#                 'Alpha': row['alpha'],
#                 'L1_Ratio': row['l1_ratio']
#             })
    
#     rmse_df = pd.DataFrame(rmse_data)
    
#     # Create the plot
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     # Boxplot for RMSE distribution
#     box = sns.boxplot(x='Dataset', y='RMSE', data=rmse_df, palette='pastel', ax=ax)
    
#     # Stripplot for individual fold RMSEs
#     strip = sns.stripplot(x='Dataset', y='RMSE', data=rmse_df, color='gray', alpha=0.6, jitter=True, ax=ax)
    
#     # Plot mean and 95% CI as red points with error bars
#     for i, dataset in enumerate(rmse_df['Dataset'].unique()):
#         rmse_vals = rmse_df[rmse_df['Dataset'] == dataset]['RMSE']
#         mean = np.mean(rmse_vals)
#         ci_low, ci_high = mean_confidence_interval(rmse_vals)[1:]
#         err = ax.errorbar(i, mean, yerr=[[mean - ci_low], [ci_high - mean]],
#                           fmt='o', color='red', capsize=5, 
#                           label='Mean ± 95% CI' if i == 0 else "")
    
#     # Title and axes
#     ax.set_title('ElasticNet RMSE Distribution per Dataset', fontsize=14)
#     ax.set_ylabel('RMSE (lower is better)')
#     ax.set_xlabel('Dataset')
#     plt.xticks(rotation=15)
#     ax.grid(axis='y', linestyle='--', alpha=0.5)
    
#     # Custom legend
#     from matplotlib.lines import Line2D
#     legend_elements = [
#         Line2D([0], [0], marker='s', color='w', label='RMSE Distribution (Boxplot)',
#                markerfacecolor='lightblue', markersize=15),
#         Line2D([0], [0], marker='o', color='gray', label='Individual CV Fold RMSE',
#                linestyle='None', markersize=8, alpha=0.6),
#         Line2D([0], [0], marker='o', color='red', label='Mean ± 95% CI',
#                linestyle='None', markersize=8)
#     ]
#     ax.legend(handles=legend_elements, loc='upper right')
    
#     plt.tight_layout()
#     save_figure(fig, "elasticnet_cv_rmse_distribution", output_dir)
    
#     # Plot the best parameters distribution
#     plot_elasticnet_best_params(cv_results, output_dir)
    
#     print(f"ElasticNet CV distribution plot saved to {output_dir}")
#     return fig

# def plot_elasticnet_best_params(cv_results, output_dir=None):
#     """
#     Plot best parameters for ElasticNet models.
    
#     Parameters:
#     -----------
#     cv_results : list
#         List of CV result dictionaries.
#     output_dir : Path, optional
#         Directory to save plot. If None, it will use the default.
#     """
#     if output_dir is None:
#         output_dir = settings.VISUALIZATION_DIR / "performance"
#         io.ensure_dir(output_dir)
    
#     # Extract best parameters
#     best_params = []
#     for result in cv_results:
#         alpha, l1_ratio = result['best_params']
#         best_params.append({
#             'Dataset': result['dataset'],
#             'Alpha': alpha,
#             'L1_Ratio': l1_ratio
#         })
    
#     best_df = pd.DataFrame(best_params)
    
#     # Create figure with two subplots
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
#     # Alpha plot
#     ax = axes[0]
#     sns.barplot(x='Dataset', y='Alpha', data=best_df, ax=ax, palette='Blues')
#     ax.set_title('Best Alpha by Dataset')
#     ax.set_ylabel('Alpha Value')
#     ax.set_xlabel('Dataset')
#     ax.tick_params(axis='x', rotation=15)
    
#     # Add value labels
#     for p in ax.patches:
#         ax.annotate(f'{p.get_height():.4f}', 
#                     (p.get_x() + p.get_width() / 2., p.get_height()),
#                     ha = 'center', va = 'bottom',
#                     xytext = (0, 5), textcoords = 'offset points')
    
#     # L1 Ratio plot
#     ax = axes[1]
#     sns.barplot(x='Dataset', y='L1_Ratio', data=best_df, ax=ax, palette='Greens')
#     ax.set_title('Best L1 Ratio by Dataset')
#     ax.set_ylabel('L1 Ratio')
#     ax.set_xlabel('Dataset')
#     ax.tick_params(axis='x', rotation=15)
    
#     # Add value labels
#     for p in ax.patches:
#         ax.annotate(f'{p.get_height():.2f}', 
#                     (p.get_x() + p.get_width() / 2., p.get_height()),
#                     ha = 'center', va = 'bottom',
#                     xytext = (0, 5), textcoords = 'offset points')
    
#     plt.tight_layout()
#     save_figure(fig, "elasticnet_best_parameters", output_dir)
    
#     return fig

# def plot_model_comparison(metrics_df=None):
#     """
#     Create comprehensive comparison plots of model performance metrics.
    
#     Parameters:
#     -----------
#     metrics_df : pandas.DataFrame, optional
#         DataFrame containing model metrics. If None, it will be loaded.
#     """
#     # Set up style
#     style = setup_visualization_style()
    
#     # Load metrics if not provided
#     if metrics_df is None:
#         metrics_file = settings.METRICS_DIR / "all_models_comparison.csv"
#         if metrics_file.exists():
#             metrics_df = pd.read_csv(metrics_file)
#         else:
#             try:
#                 # Try loading from evaluation
#                 from evaluation.metrics import evaluate_models
#                 eval_results = evaluate_models()
#                 metrics_df = eval_results['metrics_df']
#             except:
#                 print("No metrics data found. Please run evaluation first.")
#                 return
    
#     # Set up output directory
#     output_dir = settings.VISUALIZATION_DIR / "performance"
#     io.ensure_dir(output_dir)
    
#     # 1. Create bar charts of key metrics - INCREASED FIGURE SIZE
#     fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
#     # RMSE (lower is better)
#     ax = axes[0, 0]
#     sorted_df = metrics_df.sort_values('RMSE')
#     bars = ax.bar(sorted_df['model_name'], sorted_df['RMSE'], 
#                   color=[style['colors'].get(model, '#666666') for model in sorted_df['model_name']])
#     ax.set_title('Root Mean Squared Error (RMSE)', fontsize=16, pad=15)
#     ax.set_ylabel('RMSE (lower is better)', fontsize=12)
#     ax.set_xlabel('Model', fontsize=12)
#     ax.tick_params(axis='x', rotation=45, labelsize=10)
#     ax.grid(axis='y', alpha=0.3, linestyle='--')
    
#     # Add value labels with more space
#     for bar, model_name in zip(bars, sorted_df['model_name']):
#         height = bar.get_height()
#         r2 = sorted_df.loc[sorted_df['model_name'] == model_name, 'R2'].values[0]  # Get R² value
#         count = sorted_df.loc[sorted_df['model_name'] == model_name, 'n_companies'].values[0]  # Get count
#         ax.text(bar.get_x() + bar.get_width()/2, height + 0.05,
#             f'RMSE: {safe_float(height):.4f}\nR²: {safe_float(r2):.4f}\n(n={int(count)})', 
#             ha='center', va='bottom', fontsize=9)
    
#     # R² (higher is better)
#     ax = axes[0, 1]
#     sorted_df = metrics_df.sort_values('R2', ascending=False)
#     bars = ax.bar(sorted_df['model_name'], sorted_df['R2'], 
#                   color=[style['colors'].get(model, '#666666') for model in sorted_df['model_name']])
#     ax.set_title('R² Score', fontsize=16, pad=15)
#     ax.set_ylabel('R² (higher is better)', fontsize=12)
#     ax.set_xlabel('Model', fontsize=12)
#     ax.tick_params(axis='x', rotation=45, labelsize=10)
#     ax.grid(axis='y', alpha=0.3, linestyle='--')
    
#     # Add value labels with more space
#     for bar in bars:
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
#                 f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
#     # MAE (lower is better)
#     ax = axes[1, 0]
#     sorted_df = metrics_df.sort_values('MAE')
#     bars = ax.bar(sorted_df['model_name'], sorted_df['MAE'], 
#                   color=[style['colors'].get(model, '#666666') for model in sorted_df['model_name']])
#     ax.set_title('Mean Absolute Error (MAE)', fontsize=16, pad=15)
#     ax.set_ylabel('MAE (lower is better)', fontsize=12)
#     ax.set_xlabel('Model', fontsize=12)
#     ax.tick_params(axis='x', rotation=45, labelsize=10)
#     ax.grid(axis='y', alpha=0.3, linestyle='--')
    
#     # Add value labels with more space
#     for bar in bars:
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2, height + 0.03,
#                 f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
#     # Model Type Comparison
#     ax = axes[1, 1]
#     model_types = metrics_df['model_type'].unique()
#     type_scores = []
#     for model_type in model_types:
#         type_df = metrics_df[metrics_df['model_type'] == model_type]
#         type_scores.append({
#             'Type': model_type,
#             'Avg RMSE': type_df['RMSE'].mean(),
#             'Avg R²': type_df['R2'].mean(),
#             'Count': len(type_df)
#         })
    
#     type_df = pd.DataFrame(type_scores)
#     bars = ax.bar(type_df['Type'], type_df['Avg RMSE'], 
#                   color=['#3498db', '#e74c3c'])
#     ax.set_title('Average RMSE by Model Type', fontsize=16, pad=15)
#     ax.set_ylabel('Avg RMSE (lower is better)', fontsize=12)
#     ax.set_xlabel('Model Type', fontsize=12)
#     ax.grid(axis='y', alpha=0.3, linestyle='--')
    
#     # Add value and count labels with better spacing
#     for bar, count, r2 in zip(bars, type_df['Count'], type_df['Avg R²']):
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2, height + 0.05,
#                 f'RMSE: {float(height):.4f}\nR²: {float(r2):.4f}\n(n={int(count)})', 
#                 ha='center', va='bottom', fontsize=10)
    
#     # Increase spacing between subplots
#     plt.subplots_adjust(hspace=0.3, wspace=0.25)
#     plt.tight_layout(pad=2.0)
#     save_figure(fig, "model_metrics_comparison", output_dir)
    
#     # 3. Generate metrics summary table
#     try:
#         plot_metrics_summary_table(metrics_df)
#     except NameError:
#         # If plot_metrics_summary_table is not defined yet
#         print("Note: metrics_summary_table function not available")
    
#     # 4. Plot residuals for all models
#     try:
#         # Load residuals data
#         residuals = io.load_model("model_residuals.pkl", settings.METRICS_DIR)
        
#         # Plot best model first (this is the default behavior)
#         plot_residuals(output_dir)
        
#         # Then plot each non-sector model individually
#         for model_name in residuals.keys():
#             if not model_name.startswith("Sector_"):
#                 print(f"Generating residual plot for {model_name}...")
#                 plot_residuals(output_dir, best_model_name=model_name)
            
#     except Exception as e:
#         print(f"Error generating residual plots: {e}")
#         # Fall back to just plotting the best model
#         plot_residuals(output_dir)
    
#     # 5. Create statistical test visualizations
#     plot_statistical_tests()
    
#     print(f"Model comparison plots saved to {output_dir}")
#     return fig

# def plot_residuals(output_dir=None, best_model_name=None, top_n=4):
#     """
#     Plot residuals analysis for models.
    
#     Parameters:
#     -----------
#     output_dir : Path or str, optional
#         Directory for saving plots. If None, default directory is used.
#     best_model_name : str, optional
#         Name of specific model to plot. If None, best model by RMSE is used.
#     top_n : int, default=4
#         Number of models to include in multi-model plot.
#     """
#     # Set up style
#     style = setup_visualization_style()
    
#     # Set up output directory
#     if output_dir is None:
#         output_dir = settings.VISUALIZATION_DIR / "performance"
#     io.ensure_dir(output_dir)
    
#     try:
#         # Load residuals data
#         residuals = io.load_model("model_residuals.pkl", settings.METRICS_DIR)
        
#         # Print available models for debugging
#         print("Available models in residuals:")
#         for model in residuals.keys():
#             print(f"  - {model}")
            
#         # Load metrics to find best model if not specified
#         metrics_file = settings.METRICS_DIR / "all_models_comparison.csv"
#         if metrics_file.exists():
#             metrics_df = pd.read_csv(metrics_file)
            
#             # If no specific model requested, use best model by RMSE
#             if best_model_name is None:
#                 best_model_name = metrics_df.sort_values('RMSE').iloc[0]['model_name']
#         else:
#             # If no metrics file and no specific model, use first model
#             if best_model_name is None:
#                 best_model_name = list(residuals.keys())[0]
        
#         # If plotting a specific model
#         if best_model_name is not None and best_model_name in residuals:
#             # Skip sector-specific models
#             if best_model_name.startswith("Sector_"):
#                 return

#             print(f"Processing residual plot for {best_model_name}")
#             # Plot residuals for a single model
#             model_res = residuals[best_model_name]
            
#             # Create figure with multiple plots
#             fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
#             # 1. Predicted vs Actual
#             ax = axes[0, 0]
#             ax.scatter(model_res['y_pred'], model_res['y_test'], 
#                       alpha=0.6, color='#3498db')
            
#             # Add perfect prediction line
#             min_val = min(model_res['y_pred'].min(), model_res['y_test'].min())
#             max_val = max(model_res['y_pred'].max(), model_res['y_test'].max())
#             ax.plot([min_val, max_val], [min_val, max_val], 'r--')
            
#             ax.set_xlabel('Predicted Value')
#             ax.set_ylabel('Actual Value')
#             ax.set_title('Predicted vs Actual Values', fontsize=12)
#             ax.grid(True, alpha=0.3)
            
#             # Add correlation value
#             corr = np.corrcoef(model_res['y_pred'], model_res['y_test'])[0, 1]
#             ax.text(0.05, 0.95, f'Correlation: {corr:.4f}', transform=ax.transAxes,
#                    fontsize=12, va='top')
            
#             # 2. Residuals vs Predicted
#             ax = axes[0, 1]
#             ax.scatter(model_res['y_pred'], model_res['residuals'], 
#                       alpha=0.6, color='#3498db')
            
#             # Add horizontal line at y=0
#             ax.axhline(y=0, color='r', linestyle='--')
            
#             ax.set_xlabel('Predicted Value')
#             ax.set_ylabel('Residual')
#             ax.set_title('Residuals vs Predicted Values', fontsize=12)
#             ax.grid(True, alpha=0.3)
            
#             # 3. Histogram of residuals
#             ax = axes[1, 0]
#             sns.histplot(model_res['residuals'], kde=True, ax=ax, color='#3498db')
            
#             # Add vertical line at x=0
#             ax.axvline(x=0, color='r', linestyle='--')
            
#             ax.set_xlabel('Residual')
#             ax.set_ylabel('Frequency')
#             ax.set_title('Distribution of Residuals', fontsize=12)
            
#             # Add statistics
#             mean_res = model_res['residuals'].mean()
#             std_res = model_res['residuals'].std()
#             ax.text(0.05, 0.95, f'Mean: {mean_res:.4f}\nStd Dev: {std_res:.4f}', 
#                    transform=ax.transAxes, fontsize=12, va='top')
            
#             # 4. Q-Q plot
#             ax = axes[1, 1]
#             from scipy import stats
            
#             # Calculate standardized residuals
#             std_residuals = (model_res['residuals'] - mean_res) / std_res
            
#             # Create Q-Q plot
#             stats.probplot(std_residuals, dist="norm", plot=ax)
#             ax.set_title('Normal Q-Q Plot of Standardized Residuals', fontsize=12)
#             ax.grid(True, alpha=0.3)
            
#             # Set main title
#             plt.suptitle(f'Residuals Analysis for {best_model_name}', fontsize=16, y=1.02)
            
#             plt.tight_layout()

#             # Save the figure and report success
#             save_figure(fig, f"{best_model_name}_residuals_analysis", output_dir)
#             print(f"Successfully saved residual plot for {best_model_name}")
#             plt.close(fig)
            
#         # Create a comparative plot of residuals for top models
#         fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
#         # Get top models by RMSE (or just use the first few if metrics not available)
#         if 'metrics_df' in locals():
#             top_models = metrics_df.sort_values('RMSE').head(top_n)['model_name'].tolist()
#         else:
#             top_models = list(residuals.keys())[:top_n]
            
#         # Filter top_models to only include those that exist in residuals
#         top_models = [model for model in top_models if model in residuals]
        
#         # Skip if no models to plot
#         if not top_models:
#             print("No valid models found for comparison plot.")
#             return None
        
#         # 1. Boxplot of residuals
#         ax = axes[0, 0]
#         boxplot_data = [residuals[model]['residuals'] for model in top_models]
#         ax.boxplot(boxplot_data, labels=top_models, patch_artist=True,
#                   boxprops=dict(facecolor='lightblue', color='blue'),
#                   flierprops=dict(marker='o', markerfacecolor='red', markersize=3))
        
#         ax.set_xticklabels(top_models, rotation=45, ha='right')
#         ax.set_ylabel('Residuals')
#         ax.set_title('Boxplot of Residuals by Model', fontsize=12)
#         ax.grid(axis='y', alpha=0.3)
        
#         # 2. Violin plot of absolute residuals
#         ax = axes[0, 1]
#         violin_data = [residuals[model]['abs_residuals'] for model in top_models]
#         ax.violinplot(violin_data, showmeans=True, showmedians=True)
        
#         ax.set_xticks(range(1, len(top_models) + 1))
#         ax.set_xticklabels(top_models, rotation=45, ha='right')
#         ax.set_ylabel('Absolute Residuals')
#         ax.set_title('Distribution of Absolute Residuals by Model', fontsize=12)
#         ax.grid(axis='y', alpha=0.3)
        
#         # 3. Histograms of residuals
#         ax = axes[1, 0]
#         for i, model in enumerate(top_models):
#             sns.kdeplot(residuals[model]['residuals'], 
#                        label=model, 
#                        ax=ax,
#                        color=style['colors'].get(model, f'C{i}'))
            
#         ax.axvline(x=0, color='black', linestyle='--')
#         ax.set_xlabel('Residuals')
#         ax.set_ylabel('Density')
#         ax.set_title('Kernel Density Estimate of Residuals by Model', fontsize=12)
#         ax.grid(True, alpha=0.3)
#         ax.legend()
        
#         # 4. Scatter of predicted values vs residuals for top model
#         ax = axes[1, 1]
#         best_model = top_models[0]
#         ax.scatter(residuals[best_model]['y_pred'], 
#                   residuals[best_model]['residuals'],
#                   alpha=0.6, 
#                   color=style['colors'].get(best_model, 'C0'))
        
#         ax.axhline(y=0, color='red', linestyle='--')
#         ax.set_xlabel('Predicted Values')
#         ax.set_ylabel('Residuals')
#         ax.set_title(f'Predicted vs Residuals for Best Model ({best_model})', fontsize=12)
#         ax.grid(True, alpha=0.3)
        
#         # Add statistics table
#         stats_table = []
#         for model in top_models:
#             res = residuals[model]['residuals']
#             stats_table.append({
#                 'Model': model,
#                 'Mean': res.mean(),
#                 'Std Dev': res.std(),
#                 'Min': res.min(),
#                 'Max': res.max(),
#                 'RMSE': np.sqrt((res**2).mean())
#             })
        
#         stats_df = pd.DataFrame(stats_table)
#         table_text = []
#         for _, row in stats_df.iterrows():
#             text_row = [
#                 f"{row['Model']}",
#                 f"Mean: {row['Mean']:.4f}",
#                 f"Std: {row['Std Dev']:.4f}",
#                 f"RMSE: {row['RMSE']:.4f}"
#             ]
#             table_text.append(" | ".join(text_row))
        
#         fig.text(0.5, 0.01, "\n".join(table_text), ha='center', fontsize=10, 
#                 family='monospace', bbox=dict(facecolor='white', alpha=0.8))
        
#         # Set main title
#         plt.suptitle('Comparison of Residuals Across Models', fontsize=16, y=1.02)
        
#         plt.tight_layout(rect=[0, 0.05, 1, 0.98])
#         save_figure(fig, "all_models_residuals_comparison", output_dir)
        
#         print(f"Residuals plots saved to {output_dir}")
#         return fig
        
#     except Exception as e:
#         print(f"Error generating residual plots: {e}")
#         import traceback
#         traceback.print_exc()
#         return None


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


# def plot_metrics_summary_table(metrics_df=None):
#     """
#     Create a summary table visualization of model performance metrics.
    
#     Parameters:
#     -----------
#     metrics_df : pandas.DataFrame, optional
#         DataFrame containing model metrics. If None, it will be loaded.
#     """
#     # Set up style
#     style = setup_visualization_style()
    
#     # Load metrics if not provided
#     if metrics_df is None:
#         metrics_file = settings.METRICS_DIR / "all_models_comparison.csv"
#         if metrics_file.exists():
#             metrics_df = pd.read_csv(metrics_file)
#         else:
#             try:
#                 # Try loading from evaluation
#                 from evaluation.metrics import evaluate_models
#                 eval_results = evaluate_models()
#                 metrics_df = eval_results['metrics_df']
#             except:
#                 print("No metrics data found. Please run evaluation first.")
#                 return
    
#     # Set up output directory
#     output_dir = settings.VISUALIZATION_DIR / "performance"
#     io.ensure_dir(output_dir)
    
#     # Select only the necessary columns for the table
#     if 'model_name' in metrics_df.columns:
#         # Rename model_name to Model if it exists
#         metrics_df = metrics_df.rename(columns={'model_name': 'Model'})
#     elif 'index' in metrics_df.columns:
#         # If index column exists, rename to Model
#         metrics_df = metrics_df.rename(columns={'index': 'Model'})
    
#     # Select and order columns
#     table_columns = ['Model', 'MSE', 'MAE', 'R²', 'RMSE']
#     # Make sure R² is spelled correctly
#     if 'R2' in metrics_df.columns and 'R²' not in metrics_df.columns:
#         metrics_df = metrics_df.rename(columns={'R2': 'R²'})
    
#     # Filter columns that exist in the DataFrame
#     available_columns = [col for col in table_columns if col in metrics_df.columns]
#     table_data = metrics_df[available_columns].copy()
    
#     # Create figure
#     plt.figure(figsize=(12, len(table_data) * 0.5 + 1))
    
#     # Reset index to include model names in the table if not already a column
#     if 'Model' not in table_data.columns:
#         table_data = table_data.reset_index()
#         if 'index' in table_data.columns:
#             table_data = table_data.rename(columns={'index': 'Model'})
    
#     # Create a table with no cells, just the data
#     ax = plt.subplot(111, frame_on=False)
#     ax.xaxis.set_visible(False)
#     ax.yaxis.set_visible(False)
    
#     # Create cell colors array - initialize with white
#     colors = [['white' for _ in range(len(table_data.columns))] for _ in range(len(table_data))]
    
#     # Highlight the best value in each metric column
#     best_indices = {}
#     for col in table_data.columns:
#         if col == 'Model':
#             continue
        
#         # Check if the column contains numerics
#         if table_data[col].dtype in [np.float64, np.int64, float, int]:
#             # For MSE, MAE, RMSE lower is better
#             if col in ['MSE', 'MAE', 'RMSE']:
#                 best_indices[col] = table_data[col].idxmin()
#             # For R², higher is better
#             elif col in ['R²']:
#                 best_indices[col] = table_data[col].idxmax()
    
#     # Apply colors to the best values
#     for col, idx in best_indices.items():
#         col_idx = list(table_data.columns).index(col)
#         colors[table_data.index.get_loc(idx)][col_idx] = '#d9ead3'  # Light green
    
#     # Convert values to formatted strings
#     cell_text = []
#     for row in table_data.values:
#         row_text = []
#         for i, val in enumerate(row):
#             if i == 0:  # Model name
#                 row_text.append(str(val))
#             else:  # Numeric value
#                 if isinstance(val, (int, float, np.number)):
#                     row_text.append(f"{val:.4f}")
#                 else:
#                     row_text.append(str(val))
#         cell_text.append(row_text)
    
#     # Create the table
#     table = plt.table(
#         cellText=cell_text,
#         colLabels=table_data.columns,
#         cellColours=colors,
#         cellLoc='center',
#         loc='center'
#     )
#     table.auto_set_font_size(False)
#     table.set_fontsize(12)
#     table.scale(1.2, 1.5)
#     plt.title('Model Performance Metrics Summary', fontsize=16, pad=20)
#     plt.tight_layout()
    
#     # Save the figure
#     save_figure(plt.gcf(), "metrics_summary_table", output_dir)
#     plt.close()
    
#     print(f"Metrics summary table saved to {output_dir}")
#     return

# if __name__ == "__main__":
#     # Run all visualizations
#     plot_model_comparison()
#     plot_residuals()
#     plot_statistical_tests()
#     plot_elasticnet_cv_distribution()
#     plot_metrics_summary_table()