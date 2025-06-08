#!/usr/bin/env python3
"""Fix the highlighting bug in metrics summary table."""

import sys
from pathlib import Path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.visualization.core.style import setup_visualization_style

# Copy the collection function from the previous script
def collect_all_metrics_properly():
    """Collect metrics from all model types including Linear Regression."""
    models_dir = settings.MODEL_DIR
    all_metrics = []
    
    # Model files to load
    model_files = {
        'Linear Regression': 'linear_regression_models.pkl',
        'ElasticNet': 'elasticnet_models.pkl',
        'XGBoost': 'xgboost_models.pkl',
        'LightGBM': 'lightgbm_models.pkl',
        'CatBoost': 'catboost_models.pkl'
    }
    
    print("Collecting metrics from all models...")
    
    for model_type, filename in model_files.items():
        filepath = models_dir / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                models = pickle.load(f)
                print(f"\n{model_type}: {len(models)} models")
                
                for name, data in models.items():
                    if isinstance(data, dict):
                        # Extract display name
                        display_name = name.replace('_', ' ')
                        
                        # Handle different metric storage patterns
                        if model_type == 'Linear Regression':
                            # Linear Regression stores metrics at top level
                            if 'RMSE' in data and 'MAE' in data:
                                all_metrics.append({
                                    'Model': display_name,
                                    'Model Type': model_type,
                                    'RMSE': data.get('RMSE'),
                                    'MAE': data.get('MAE'),
                                    'R2': data.get('R2'),
                                    'MSE': data.get('MSE')
                                })
                                print(f"  ✓ {name}: RMSE={data.get('RMSE', 'N/A')}")
                        else:
                            # Other models store metrics in 'metrics' dict
                            if 'metrics' in data:
                                metrics = data['metrics']
                                all_metrics.append({
                                    'Model': display_name,
                                    'Model Type': model_type,
                                    'RMSE': metrics.get('rmse', metrics.get('test_rmse')),
                                    'MAE': metrics.get('mae', metrics.get('test_mae')),
                                    'R2': metrics.get('r2', metrics.get('test_r2')),
                                    'MSE': metrics.get('mse', metrics.get('test_mse'))
                                })
                                print(f"  ✓ {name}: RMSE={metrics.get('rmse', 'N/A')}")
    
    return pd.DataFrame(all_metrics)

def create_correctly_highlighted_table(df, output_path):
    """Create metrics table with CORRECT best model highlighting."""
    # Clean up data
    df = df.dropna(subset=['RMSE'])
    
    # Calculate MSE if missing
    df['MSE'] = df.apply(lambda row: row['MSE'] if pd.notna(row['MSE']) else row['RMSE']**2, axis=1)
    
    # Sort by RMSE
    df = df.sort_values('RMSE').reset_index(drop=True)
    
    # Find the row index of the best model AFTER sorting
    best_rmse_row_index = 0  # Since we sorted by RMSE, the first row is the best
    best_model_name = df.iloc[0]['Model']
    best_rmse_value = df.iloc[0]['RMSE']
    
    print(f"\nBest model: {best_model_name} with RMSE: {best_rmse_value}")
    print(f"This should be in row {best_rmse_row_index + 1} (excluding header)")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = df[['Model', 'RMSE', 'MAE', 'R2', 'MSE']].copy()
    table_data['RMSE'] = table_data['RMSE'].round(3)
    table_data['MAE'] = table_data['MAE'].round(3)
    table_data['R2'] = table_data['R2'].round(3)
    table_data['MSE'] = table_data['MSE'].round(3)
    
    # Create table
    table = ax.table(cellText=table_data.values,
                     colLabels=table_data.columns,
                     cellLoc='center',
                     loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    
    # Color header
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows alternately
    for i in range(1, len(table_data) + 1):
        if i % 2 == 0:
            for j in range(len(table_data.columns)):
                table[(i, j)].set_facecolor('#F2F2F2')
    
    # Highlight the FIRST row (best RMSE) in green since data is sorted by RMSE
    for j in range(len(table_data.columns)):
        table[(1, j)].set_facecolor('#92D050')  # Row 1 (after header) is the best
    
    # Verify what we're highlighting
    print(f"Highlighting row 1 which contains: {table_data.iloc[0]['Model']} with RMSE {table_data.iloc[0]['RMSE']}")
    
    # Title and summary
    plt.title('Model Performance Metrics Summary', fontsize=16, pad=20, weight='bold')
    plt.text(0.5, -0.05, 
             f'Total Models: {len(df)} | Best RMSE: {best_model_name} ({best_rmse_value:.3f})', 
             ha='center', transform=ax.transAxes, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return len(df)

def main():
    """Main function to fix metrics summary table highlighting."""
    print("=" * 80)
    print("FIXING METRICS TABLE HIGHLIGHTING")
    print("=" * 80)
    
    setup_visualization_style()
    
    # Collect all metrics
    df = collect_all_metrics_properly()
    
    print(f"\n\nTotal models with metrics: {len(df)}")
    
    # Debug: Show the top 5 models by RMSE
    df_sorted = df.dropna(subset=['RMSE']).sort_values('RMSE')
    print("\nTop 5 models by RMSE:")
    for idx, row in df_sorted.head(5).iterrows():
        print(f"  {row['Model']}: {row['RMSE']:.4f}")
    
    # Create output directory
    output_dir = settings.VISUALIZATION_DIR / 'performance'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate fixed table
    output_path = output_dir / 'metrics_summary_table.png'
    model_count = create_correctly_highlighted_table(df, output_path)
    
    print(f"\n✓ Fixed metrics summary table created with {model_count} models")
    print(f"✓ Best model correctly highlighted")
    print(f"✓ Saved to: {output_path}")

if __name__ == "__main__":
    main()