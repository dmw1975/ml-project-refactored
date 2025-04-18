# Define a simplified visualization function that works with your data
def visualize_complete_results(model_metrics, output_dir='linreg_eval_metrics'):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    # Create directories if they don't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define model colors
    MODEL_COLORS = {
        'LR_Base': '#3498db',                    # Base model - blue
        'LR_Base_Random': '#9b59b6',             # Base model with random - purple
        'LR_Yeo': '#2ecc71',                     # Yeo model - green
        'LR_Yeo_Random': '#f39c12',              # Yeo model with random - orange
        'LR_Base_elasticnet': '#e74c3c',         # ElasticNet base - red
        'LR_Base_random_elasticnet': '#1abc9c',  # ElasticNet base with random - teal
        'LR_Yeo_elasticnet': '#d35400',          # ElasticNet yeo - dark orange
        'LR_Yeo_random_elasticnet': '#8e44ad'    # ElasticNet yeo with random - dark purple
    }
    
    # Add RMSE to model_metrics if not present
    for model_name, metrics in model_metrics.items():
        if 'RMSE' not in metrics:
            metrics['RMSE'] = np.sqrt(metrics['MSE'])
    
    # Create DataFrame with all models
    model_names = list(model_metrics.keys())
    
    metrics_data = {
        'Model': model_names,
        'MSE': [model_metrics[model]['MSE'] for model in model_names],
        'MAE': [model_metrics[model]['MAE'] for model in model_names],
        'R²': [model_metrics[model]['R2'] for model in model_names],
        'RMSE': [model_metrics[model]['RMSE'] for model in model_names]
    }
    
    # Convert to DataFrame and set index
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.set_index('Model', inplace=True)
    
    # Save metrics to CSV for reference
    metrics_df.to_csv(f'{output_dir}/model_metrics_summary.csv')
    
    # 1. Create detailed metrics comparison
    plt.figure(figsize=(20, 12))
    
    # Find the best performing model for each metric
    best_mse_model = metrics_df['MSE'].idxmin()
    best_mae_model = metrics_df['MAE'].idxmin()
    best_r2_model = metrics_df['R²'].idxmax()
    best_rmse_model = metrics_df['RMSE'].idxmin()
    
    # Plot MSE
    plt.subplot(2, 2, 1)
    bars = plt.bar(metrics_df.index, metrics_df['MSE'], 
                   color=[MODEL_COLORS.get(model, '#666666') if model != best_mse_model 
                          else '#82ca9d' for model in metrics_df.index])
    plt.title('Mean Squared Error (MSE)', fontsize=14)
    plt.ylabel('MSE (lower is better)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot MAE
    plt.subplot(2, 2, 2)
    bars = plt.bar(metrics_df.index, metrics_df['MAE'],
                   color=[MODEL_COLORS.get(model, '#666666') if model != best_mae_model 
                          else '#82ca9d' for model in metrics_df.index])
    plt.title('Mean Absolute Error (MAE)', fontsize=14)
    plt.ylabel('MAE (lower is better)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.03,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot R²
    plt.subplot(2, 2, 3)
    bars = plt.bar(metrics_df.index, metrics_df['R²'],
                   color=[MODEL_COLORS.get(model, '#666666') if model != best_r2_model 
                          else '#82ca9d' for model in metrics_df.index])
    plt.title('R² Score', fontsize=14)
    plt.ylabel('R² (higher is better)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        text_y = height + 0.05 if height >= 0 else height - 0.15
        plt.text(bar.get_x() + bar.get_width()/2, text_y,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot RMSE
    plt.subplot(2, 2, 4)
    bars = plt.bar(metrics_df.index, metrics_df['RMSE'],
                   color=[MODEL_COLORS.get(model, '#666666') if model != best_rmse_model 
                          else '#82ca9d' for model in metrics_df.index])
    plt.title('Root Mean Squared Error (RMSE)', fontsize=14)
    plt.ylabel('RMSE (lower is better)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.05,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/all_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create radar chart
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(111, polar=True)
    
    # Normalize metrics to 0-100 scale for radar chart
    # For MSE, MAE, and RMSE, lower is better, so we invert the scale
    normalized_metrics = {}
    
    for model in metrics_df.index:
        # Get max values for normalization
        max_mse = metrics_df['MSE'].max()
        max_mae = metrics_df['MAE'].max()
        max_rmse = metrics_df['RMSE'].max()
        
        # Calculate normalized values (inverted for error metrics)
        mse_norm = (1 - metrics_df['MSE'][model] / max_mse) * 100
        mae_norm = (1 - metrics_df['MAE'][model] / max_mae) * 100
        rmse_norm = (1 - metrics_df['RMSE'][model] / max_rmse) * 100
        
        # R² needs special handling since it can be negative
        min_r2 = min(0, metrics_df['R²'].min())  # Take min of 0 or minimum R² value
        max_r2 = max(0.5, metrics_df['R²'].max())  # Take max of 0.5 or maximum R² value
        r2_range = max_r2 - min_r2
        r2_norm = ((metrics_df['R²'][model] - min_r2) / r2_range) * 100
        r2_norm = max(0, min(100, r2_norm))  # Clip to 0-100 range
        
        normalized_metrics[model] = [mse_norm, mae_norm, rmse_norm, r2_norm]
    
    # Create radar chart
    metrics_labels = ['MSE\n(inverted)', 'MAE\n(inverted)', 'RMSE\n(inverted)', 'R² Score']
    angles = np.linspace(0, 2*np.pi, len(metrics_labels) + 1, endpoint=True)
    
    # Add the polar grid
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], metrics_labels, fontsize=12)
    ax.set_rlabel_position(0)
    plt.ylim(0, 100)
    ax.grid(True)
    
    # Plot each model
    for model in metrics_df.index:
        values = normalized_metrics[model]
        values = np.append(values, values[0])  # Close the loop
        color = MODEL_COLORS.get(model, '#666666')
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=10)
    
    plt.title('Model Performance Comparison\n(Higher is Better)', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Create error distribution
    plt.figure(figsize=(14, 8))
    
    # Generate simulated error distributions based on the MSE values
    for model in metrics_df.index:
        # Simulate errors based on the MSE values
        np.random.seed(42)  # For reproducibility
        errors = np.random.normal(0, np.sqrt(metrics_df['MSE'][model]), 1000)
        sns.kdeplot(errors, label=f'{model}', fill=True, alpha=0.3, 
                    color=MODEL_COLORS.get(model, '#666666'))
    
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    plt.title('Error Distribution Comparison', fontsize=15)
    plt.xlabel('Prediction Error')
    plt.ylabel('Density')
    plt.legend(fontsize=9)
    
    # Add annotations for each model's RMSE
    y_pos = 0.95
    plt.text(0.02, 0.98, "RMSE Values:", transform=plt.gca().transAxes, 
             fontsize=12, fontweight='bold')
    
    for i, model in enumerate(metrics_df.index):
        y_pos -= 0.05
        plt.text(0.02, y_pos, 
                f"{model}: {metrics_df['RMSE'][model]:.3f}",
                transform=plt.gca().transAxes, fontsize=10,
                color=MODEL_COLORS.get(model, '#666666'))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Create model improvement analysis
    # Base model for comparison
    base_model = 'LR_Base'  # Assuming LR_Base is our reference model
    
    if base_model in metrics_df.index:
        improvement_data = []
        
        for model in metrics_df.index:
            if model != base_model:
                mse_improvement = ((metrics_df['MSE'][base_model] - metrics_df['MSE'][model]) / 
                                  metrics_df['MSE'][base_model] * 100)
                mae_improvement = ((metrics_df['MAE'][base_model] - metrics_df['MAE'][model]) / 
                                  metrics_df['MAE'][base_model] * 100)
                rmse_improvement = ((metrics_df['RMSE'][base_model] - metrics_df['RMSE'][model]) / 
                                   metrics_df['RMSE'][base_model] * 100)
                
                # Handle R² score improvement calculation differently since R² can be negative
                if metrics_df['R²'][base_model] < 0 and metrics_df['R²'][model] > 0:
                    # If going from negative to positive, we'll report absolute improvement
                    r2_improvement = 100  # Consider it a 100% improvement when going from negative to positive
                else:
                    # Regular percentage calculation if both are positive or both are negative
                    r2_delta = metrics_df['R²'][model] - metrics_df['R²'][base_model]
                    r2_improvement = r2_delta * 100  # Just report absolute percentage point improvement
                
                improvement_data.append({
                    'Model': model,
                    'MSE Improvement (%)': mse_improvement,
                    'MAE Improvement (%)': mae_improvement,
                    'R² Score Improvement': r2_improvement,
                    'RMSE Improvement (%)': rmse_improvement
                })
        
        if improvement_data:
            improvement_df = pd.DataFrame(improvement_data)
            
            plt.figure(figsize=(18, 10))
            
            # Plot the error metrics improvements
            plt.subplot(1, 2, 1)
            plot_cols = ['MSE Improvement (%)', 'MAE Improvement (%)', 'RMSE Improvement (%)']
            melted_improvement = pd.melt(improvement_df, id_vars=['Model'], value_vars=plot_cols,
                                       var_name='Metric', value_name='Improvement (%)')
            
            # Create the barplot with custom colors
            ax = sns.barplot(x='Metric', y='Improvement (%)', hue='Model', data=melted_improvement, 
                            palette={model: MODEL_COLORS.get(model, '#666666') for model in improvement_df['Model']})
            
            # Add a horizontal line at y=0
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            
            plt.title(f'Performance Improvements Compared to {base_model}', fontsize=14)
            plt.ylabel('Improvement (%)')
            plt.xticks(rotation=15)
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            
            # Make the legend more compact
            plt.legend(fontsize=9)
            
            # Plot R² improvement separately
            plt.subplot(1, 2, 2)
            r2_data = improvement_df[['Model', 'R² Score Improvement']]
            bars = plt.bar(r2_data['Model'], r2_data['R² Score Improvement'],
                          color=[MODEL_COLORS.get(model, '#666666') for model in r2_data['Model']])
            
            plt.title(f'R² Score Improvement Compared to {base_model}', fontsize=14)
            plt.ylabel('R² Score Absolute Improvement (percentage points)')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            
            # Add a horizontal line at y=0
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            
            # Add annotations
            for bar in bars:
                height = bar.get_height()
                text_y = height + 2 if height >= 0 else height - 5
                plt.text(bar.get_x() + bar.get_width()/2, text_y,
                        f'{height:.1f} pts', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/improvements_vs_base.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 5. Create metrics summary table
    plt.figure(figsize=(12, len(metrics_df) * 0.5 + 1))
    
    # Reset index to include model names in the table
    table_data = metrics_df.reset_index()
    
    # Create a table with no cells, just the data
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    # Create cell colors array - initialize with white
    colors = [['white' for _ in range(len(table_data.columns))] for _ in range(len(table_data))]
    
    # Highlight the best value in each metric column
    best_indices = {
        'MSE': table_data['MSE'].idxmin(),
        'MAE': table_data['MAE'].idxmin(),
        'R²': table_data['R²'].idxmax(),
        'RMSE': table_data['RMSE'].idxmin()
    }
    
    for col, idx in best_indices.items():
        col_idx = table_data.columns.get_loc(col)
        colors[idx][col_idx] = '#d9ead3'  # Light green
    
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
    plt.savefig(f'{output_dir}/metrics_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Model family comparison
    # Group models by family
    lr_models = [model for model in metrics_df.index if not model.endswith('elasticnet')]
    en_models = [model for model in metrics_df.index if model.endswith('elasticnet')]
    
    base_models = [model for model in metrics_df.index if 'Base' in model]
    yeo_models = [model for model in metrics_df.index if 'Yeo' in model]
    
    random_models = [model for model in metrics_df.index if 'Random' in model or 'random' in model]
    non_random_models = [model for model in metrics_df.index if 'Random' not in model and 'random' not in model]
    
    # Create comparison dataframe
    family_comparison = pd.DataFrame({
        'Model Family': ['Linear Regression', 'ElasticNet', 'Base Features', 'Yeo Features', 
                        'With Random', 'Without Random'],
        'Models': [', '.join(lr_models), ', '.join(en_models), ', '.join(base_models), 
                  ', '.join(yeo_models), ', '.join(random_models), ', '.join(non_random_models)],
        'Avg MSE': [metrics_df.loc[lr_models, 'MSE'].mean() if lr_models else np.nan,
                   metrics_df.loc[en_models, 'MSE'].mean() if en_models else np.nan,
                   metrics_df.loc[base_models, 'MSE'].mean() if base_models else np.nan,
                   metrics_df.loc[yeo_models, 'MSE'].mean() if yeo_models else np.nan,
                   metrics_df.loc[random_models, 'MSE'].mean() if random_models else np.nan,
                   metrics_df.loc[non_random_models, 'MSE'].mean() if non_random_models else np.nan],
        'Avg R²': [metrics_df.loc[lr_models, 'R²'].mean() if lr_models else np.nan,
                  metrics_df.loc[en_models, 'R²'].mean() if en_models else np.nan,
                  metrics_df.loc[base_models, 'R²'].mean() if base_models else np.nan,
                  metrics_df.loc[yeo_models, 'R²'].mean() if yeo_models else np.nan,
                  metrics_df.loc[random_models, 'R²'].mean() if random_models else np.nan,
                  metrics_df.loc[non_random_models, 'R²'].mean() if non_random_models else np.nan]
    })
    
    # Calculate standard deviations to add error bars
    family_comparison['MSE Std'] = [metrics_df.loc[lr_models, 'MSE'].std() if len(lr_models) > 1 else 0,
                                  metrics_df.loc[en_models, 'MSE'].std() if len(en_models) > 1 else 0,
                                  metrics_df.loc[base_models, 'MSE'].std() if len(base_models) > 1 else 0,
                                  metrics_df.loc[yeo_models, 'MSE'].std() if len(yeo_models) > 1 else 0,
                                  metrics_df.loc[random_models, 'MSE'].std() if len(random_models) > 1 else 0,
                                  metrics_df.loc[non_random_models, 'MSE'].std() if len(non_random_models) > 1 else 0]
    
    family_comparison['R² Std'] = [metrics_df.loc[lr_models, 'R²'].std() if len(lr_models) > 1 else 0,
                                  metrics_df.loc[en_models, 'R²'].std() if len(en_models) > 1 else 0,
                                  metrics_df.loc[base_models, 'R²'].std() if len(base_models) > 1 else 0,
                                  metrics_df.loc[yeo_models, 'R²'].std() if len(yeo_models) > 1 else 0,
                                  metrics_df.loc[random_models, 'R²'].std() if len(random_models) > 1 else 0,
                                  metrics_df.loc[non_random_models, 'R²'].std() if len(non_random_models) > 1 else 0]
    
    # Count models in each family
    family_comparison['Count'] = [len(lr_models), len(en_models), len(base_models), 
                                len(yeo_models), len(random_models), len(non_random_models)]
    
    # Create subplot figure
    plt.figure(figsize=(16, 12))
    
    # Plot MSE comparison
    plt.subplot(2, 1, 1)
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    # MSE - lower is better, so we invert the y-axis
    bars = plt.bar(family_comparison['Model Family'], family_comparison['Avg MSE'], 
                  yerr=family_comparison['MSE Std'],
                  color=colors, alpha=0.7)
    
    plt.title('Average MSE by Model Family', fontsize=16)
    plt.ylabel('MSE (lower is better)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels with count
    for bar, count in zip(bars, family_comparison['Count']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                f'{height:.2f}\n(n={count})', ha='center', va='bottom', fontsize=10)
    
    # Plot R² comparison
    plt.subplot(2, 1, 2)
    bars = plt.bar(family_comparison['Model Family'], family_comparison['Avg R²'], 
                  yerr=family_comparison['R² Std'],
                  color=colors, alpha=0.7)
    
    plt.title('Average R² Score by Model Family', fontsize=16)
    plt.ylabel('R² (higher is better)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels with count
    for bar, count in zip(bars, family_comparison['Count']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                f'{height:.3f}\n(n={count})', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_family_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All visualization images have been saved to the '{output_dir}' directory.")
    print("\nFiles generated:")
    print(f"1. {output_dir}/all_metrics_comparison.png - Bar charts showing detailed metrics")
    print(f"2. {output_dir}/radar_comparison.png - Radar chart for multi-metric comparison")
    print(f"3. {output_dir}/error_distribution.png - Error distribution comparison")
    print(f"4. {output_dir}/improvements_vs_base.png - Comparative improvements over base model")
    print(f"5. {output_dir}/metrics_summary_table.png - Summary table of all metrics")
    print(f"6. {output_dir}/model_family_comparison.png - Comparison between model families")
    print(f"7. {output_dir}/model_metrics_summary.csv - CSV file with all metrics")