#!/usr/bin/env python3
"""Script to check and compare model metrics across different sources."""

import pickle
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def print_model_metrics(model_dict, model_name, verbose=True):
    """
    Print metrics for a specific model.
    Returns a dict with extracted metrics for further analysis.
    """
    # Extract metrics
    metrics = {
        'Model': model_name,
        'RMSE': model_dict.get('RMSE', np.nan),
        'MSE': model_dict.get('MSE', np.nan),
        'CV_MSE': model_dict.get('cv_mse', np.nan),
        'CV_MSE_STD': model_dict.get('cv_mse_std', np.nan),
        'Model_Type': model_dict.get('model_type', 'N/A'),
    }
    
    # Calculate sqrt of cv_mse if available
    if 'cv_mse' in model_dict and not np.isnan(model_dict['cv_mse']):
        cv_mse = model_dict['cv_mse']
        metrics['CV_RMSE'] = np.sqrt(cv_mse)
    else:
        metrics['CV_RMSE'] = np.nan
    
    # Print if requested
    if verbose:
        print(f"\nModel: {model_name}")
        print(f"  RMSE: {metrics['RMSE']}")
        print(f"  MSE: {metrics['MSE']}")
        print(f"  CV MSE: {metrics['CV_MSE']}")
        print(f"  CV MSE STD: {metrics['CV_MSE_STD']}")
        print(f"  Model Type: {metrics['Model_Type']}")
        if not np.isnan(metrics.get('CV_RMSE', np.nan)):
            print(f"  sqrt(CV MSE): {metrics['CV_RMSE']}")
    
    return metrics

def load_and_process_models():
    """Load all model files and extract metrics for comparison."""
    # Add project root to path for importing modules
    project_root = Path(__file__).parent.absolute()
    sys.path.append(str(project_root))
    
    # Storage for all metrics
    all_metrics = []
    
    # Check ElasticNet metrics
    try:
        with open('outputs/models/elasticnet_models.pkl', 'rb') as f:
            elasticnet_models = pickle.load(f)
        
        print("============ ElasticNet Models ============")
        for name, model in elasticnet_models.items():
            metrics = print_model_metrics(model, name)
            metrics['Source'] = 'Model File'
            all_metrics.append(metrics)
    except Exception as e:
        print(f"Error loading ElasticNet models: {e}")
    
    # Check tree-based model metrics
    model_files = [
        ('XGBoost', 'outputs/models/xgboost_models.pkl'),
        ('LightGBM', 'outputs/models/lightgbm_models.pkl'),
        ('CatBoost', 'outputs/models/catboost_models.pkl')
    ]
    
    for model_type, file_path in model_files:
        try:
            with open(file_path, 'rb') as f:
                models = pickle.load(f)
            
            print(f"\n============ {model_type} Models ============")
            for name, model in models.items():
                metrics = print_model_metrics(model, name)
                metrics['Source'] = 'Model File'
                all_metrics.append(metrics)
        except Exception as e:
            print(f"Error loading {model_type} models: {e}")
    
    # Get metrics from all_models_comparison.csv
    try:
        print("\n============ Metrics from all_models_comparison.csv ============")
        metrics_df = pd.read_csv('outputs/metrics/all_models_comparison.csv')
        
        # Sort by RMSE (best to worst)
        sorted_df = metrics_df.sort_values('RMSE')
        
        # Print summary and add to our metrics list
        for _, row in sorted_df.iterrows():
            model_name = row['model_name']
            rmse = row['RMSE']
            model_type = row.get('model_type', 'Unknown')
            print(f"Model: {model_name}, RMSE: {rmse}, Type: {model_type}")
            
            # Add to our metrics list
            all_metrics.append({
                'Model': model_name,
                'RMSE': rmse,
                'MSE': row.get('MSE', np.nan),
                'CV_MSE': np.nan,  # Not in the CSV
                'CV_MSE_STD': np.nan,  # Not in the CSV
                'CV_RMSE': np.nan,  # Not in the CSV
                'Model_Type': model_type,
                'Source': 'CSV Summary'
            })
    except Exception as e:
        print(f"Error loading metrics comparison: {e}")
    
    # Check baseline metrics from both possible locations
    baseline_files = [
        'outputs/metrics/baseline_comparison.csv',
        'outputs/visualizations/statistical_tests/baseline_significance_tests.csv'
    ]
    
    baseline_df = None
    for file_path in baseline_files:
        try:
            print(f"\n============ Metrics from {file_path} ============")
            df = pd.read_csv(file_path)
            if not df.empty:
                baseline_df = df
                print(f"Successfully loaded {len(df)} baseline comparison records from {file_path}")
                
                # Display a sample of the data to understand its structure
                print("\nColumns in the file:", df.columns.tolist())
                print("\nSample data (first 3 rows):")
                print(df.head(3))
                break
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Process the baseline comparison data if found
    if baseline_df is not None:
        # Check if this is the baseline_comparison.csv format
        if 'Model' in baseline_df.columns and 'RMSE' in baseline_df.columns and 'Baseline RMSE' in baseline_df.columns:
            # This is likely the baseline_comparison.csv format
            for _, row in baseline_df.iterrows():
                model_name = row['Model'].rsplit('_', 1)[0] if any(suffix in row['Model'] for suffix in ['_random', '_mean', '_median']) else row['Model']
                baseline_type = 'Random'  # Default assumption
                if '_mean' in row['Model']:
                    baseline_type = 'Mean'
                elif '_median' in row['Model']:
                    baseline_type = 'Median'
                
                print(f"Model: {model_name}, Baseline: {baseline_type}, RMSE: {row['RMSE']}, Baseline RMSE: {row['Baseline RMSE']}")
                
                # Add to our metrics list
                all_metrics.append({
                    'Model': model_name,
                    'RMSE': row['RMSE'],
                    'MSE': np.nan,  # Not directly available
                    'CV_MSE': np.nan,  # Not directly available
                    'CV_MSE_STD': np.nan,  # Not directly available
                    'CV_RMSE': row['RMSE'],  # This is from CV metrics
                    'Model_Type': 'Unknown',  # Not directly available
                    'Source': f'Baseline {baseline_type}'
                })
        
        # Check if this is the baseline_significance_tests.csv format
        elif 'Model' in baseline_df.columns and 'Model Mean RMSE' in baseline_df.columns and 'Baseline' in baseline_df.columns:
            # Get just the model names (strip _random, _mean, etc. suffixes)
            baseline_df['Base_Model'] = baseline_df['Model'].apply(
                lambda x: x.rsplit('_', 1)[0] if any(suffix in x for suffix in ['_random', '_mean', '_median']) else x
            )
            
            # Group by base model name and baseline type
            for (base_model, baseline), group in baseline_df.groupby(['Base_Model', 'Baseline']):
                print(f"Model: {base_model}, Baseline: {baseline}, RMSE: {group['Model Mean RMSE'].values[0]}")
                
                # Add to our metrics list - using model's RMSE from baseline tests
                all_metrics.append({
                    'Model': base_model,
                    'RMSE': group['Model Mean RMSE'].values[0],
                    'MSE': np.nan,  # Not in the baseline tests
                    'CV_MSE': np.nan,  # Not directly available
                    'CV_MSE_STD': np.nan,  # Not directly available
                    'CV_RMSE': group['Model Mean RMSE'].values[0],  # This is from CV metrics
                    'Model_Type': 'Unknown',  # Not in the baseline tests
                    'Source': f'Baseline {baseline}'
                })
        else:
            print(f"Unknown baseline file format. Column headers: {baseline_df.columns.tolist()}")
    else:
        print("No baseline comparison files could be loaded.")
    
    return pd.DataFrame(all_metrics)

def analyze_metrics_discrepancy(metrics_df):
    """Analyze discrepancies in metrics across different sources."""
    print("\n\n============= METRICS DISCREPANCY ANALYSIS =============")
    
    # Create a pivot table to compare metrics by model and source
    pivot = metrics_df.pivot_table(
        index='Model', 
        columns='Source', 
        values=['RMSE', 'CV_RMSE'],
        aggfunc='first'
    )
    
    # Check if there are models in both the CSV summary and baseline tests
    models_with_multiple_sources = []
    for model in metrics_df['Model'].unique():
        sources = metrics_df[metrics_df['Model'] == model]['Source'].unique()
        if len(sources) > 1:
            models_with_multiple_sources.append(model)
    
    if not models_with_multiple_sources:
        print("No models with metrics from multiple sources found.")
        return
    
    print(f"Found {len(models_with_multiple_sources)} models with metrics from multiple sources.")
    
    # Calculate discrepancies
    discrepancies = []
    for model in models_with_multiple_sources:
        model_data = metrics_df[metrics_df['Model'] == model]
        
        # Get RMSE from different sources
        csv_rmse = model_data[model_data['Source'] == 'CSV Summary']['RMSE'].values
        model_file_rmse = model_data[model_data['Source'] == 'Model File']['RMSE'].values
        baseline_random_rmse = model_data[model_data['Source'] == 'Baseline Random']['RMSE'].values
        
        # Check for CV RMSE from Model File
        model_file_cv_rmse = model_data[model_data['Source'] == 'Model File']['CV_RMSE'].values
        
        # Only process if we have values to compare
        if (len(csv_rmse) > 0 and 
           (len(model_file_rmse) > 0 or len(baseline_random_rmse) > 0 or len(model_file_cv_rmse) > 0)):
            
            csv_rmse_val = csv_rmse[0] if len(csv_rmse) > 0 else np.nan
            model_file_rmse_val = model_file_rmse[0] if len(model_file_rmse) > 0 else np.nan
            baseline_rmse_val = baseline_random_rmse[0] if len(baseline_random_rmse) > 0 else np.nan
            cv_rmse_val = model_file_cv_rmse[0] if len(model_file_cv_rmse) > 0 else np.nan
            
            # Calculate discrepancies
            csv_vs_model_diff = csv_rmse_val - model_file_rmse_val if not np.isnan(model_file_rmse_val) else np.nan
            csv_vs_baseline_diff = csv_rmse_val - baseline_rmse_val if not np.isnan(baseline_rmse_val) else np.nan
            csv_vs_cv_diff = csv_rmse_val - cv_rmse_val if not np.isnan(cv_rmse_val) else np.nan
            
            # Store results
            discrepancies.append({
                'Model': model,
                'CSV RMSE': csv_rmse_val,
                'Model File RMSE': model_file_rmse_val,
                'Baseline RMSE': baseline_rmse_val,
                'CV RMSE': cv_rmse_val,
                'CSV vs Model File': csv_vs_model_diff,
                'CSV vs Baseline': csv_vs_baseline_diff,
                'CSV vs CV': csv_vs_cv_diff,
                'Model Type': model_data['Model_Type'].iloc[0] if not model_data['Model_Type'].isnull().all() else 'Unknown'
            })
    
    # Convert to DataFrame for analysis
    discrepancy_df = pd.DataFrame(discrepancies)
    
    # Sort by absolute discrepancy (CSV vs Baseline) to find largest issues
    discrepancy_df['Abs CSV vs Baseline'] = discrepancy_df['CSV vs Baseline'].abs()
    sorted_df = discrepancy_df.sort_values('Abs CSV vs Baseline', ascending=False)
    
    # Print summary of discrepancies
    print("\nTop 10 Models with Largest Discrepancies (CSV Summary vs Baseline Tests):")
    print(sorted_df[['Model', 'CSV RMSE', 'Baseline RMSE', 'CSV vs Baseline', 'Model Type']].head(10))
    
    # Print summary statistics by model type
    print("\nDiscrepancy Statistics by Model Type:")
    model_type_stats = discrepancy_df.groupby('Model Type').agg({
        'CSV vs Baseline': ['mean', 'min', 'max', 'count'],
        'CSV vs CV': ['mean', 'min', 'max', 'count']
    })
    print(model_type_stats)
    
    # Visualize the discrepancies
    plt.figure(figsize=(12, 8))
    
    # Filter for models with non-null values in both columns
    vis_df = discrepancy_df.dropna(subset=['CSV RMSE', 'Baseline RMSE'])
    
    # Split into different model types for color coding
    model_types = vis_df['Model Type'].unique()
    
    plt.figure(figsize=(12, 8))
    
    # Create a scatter plot with diagonal line
    plt.plot([0, 5], [0, 5], 'k--', alpha=0.5, label='Perfect Agreement')
    
    # Plot each model type with a different color
    for model_type in model_types:
        subset = vis_df[vis_df['Model Type'] == model_type]
        plt.scatter(
            subset['CSV RMSE'], 
            subset['Baseline RMSE'], 
            label=model_type,
            alpha=0.7,
            s=100  # Larger point size
        )
    
    # Add labels for ElasticNet models
    for _, row in vis_df[vis_df['Model Type'].str.contains('ElasticNet', na=False)].iterrows():
        plt.annotate(
            row['Model'], 
            (row['CSV RMSE'], row['Baseline RMSE']),
            xytext=(5, 5), 
            textcoords='offset points',
            fontsize=8
        )
    
    plt.xlabel('RMSE from Metrics Summary (CSV)', fontsize=12)
    plt.ylabel('RMSE from Baseline Tests (CV-based)', fontsize=12)
    plt.title('Comparison of RMSE Values: Metrics Summary vs. Baseline Tests', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('outputs/metrics/rmse_discrepancy_analysis.png', dpi=300)
    print("\nDiscrepancy visualization saved as 'outputs/metrics/rmse_discrepancy_analysis.png'")
    
    # Add conclusions
    print("\nDISCREPANCY ANALYSIS CONCLUSIONS:")
    elastic_discrepancy = discrepancy_df[discrepancy_df['Model Type'].str.contains('ElasticNet', na=False)]['CSV vs Baseline'].mean()
    tree_models = ['XGBoost', 'LightGBM', 'CatBoost']
    tree_discrepancy = discrepancy_df[discrepancy_df['Model Type'].isin(tree_models)]['CSV vs Baseline'].mean()
    
    print(f"1. ElasticNet models show an average discrepancy of {elastic_discrepancy:.2f} between CSV and Baseline metrics")
    print(f"2. Tree-based models show an average discrepancy of {tree_discrepancy:.2f} between CSV and Baseline metrics")
    
    if abs(elastic_discrepancy) > abs(tree_discrepancy):
        print("3. ElasticNet models have a LARGER discrepancy between test set metrics and CV metrics")
        print("   This suggests the ElasticNet models may be OVERFITTING - they perform well in CV but worse on the test set")
    else:
        print("3. Tree-based models have a larger discrepancy between test set metrics and CV metrics")
    
    # Return the discrepancy DataFrame for further analysis
    return discrepancy_df

def main():
    """Main function to check model metrics."""
    # Load all models and extract metrics
    metrics_df = load_and_process_models()
    
    # Analyze discrepancies
    discrepancy_df = analyze_metrics_discrepancy(metrics_df)
    
    # Save results
    metrics_df.to_csv('outputs/metrics/all_metrics_comparison.csv', index=False)
    if discrepancy_df is not None:
        discrepancy_df.to_csv('outputs/metrics/metrics_discrepancy_analysis.csv', index=False)
    
    print("\nAnalysis complete. Results saved to outputs/metrics directory.")

if __name__ == "__main__":
    main()