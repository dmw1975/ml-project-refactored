#!/usr/bin/env python3
"""Regenerate metrics summary table with ALL 32 models."""

import sys
from pathlib import Path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.visualization.core.style import setup_visualization_style

def collect_all_metrics():
    """Collect metrics from all model types."""
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
                    if isinstance(data, dict) and 'metrics' in data:
                        metrics = data['metrics']
                        # Extract display name
                        display_name = name.replace('_', ' ')
                        
                        all_metrics.append({
                            'Model': display_name,
                            'Model Type': model_type,
                            'RMSE': metrics.get('rmse', metrics.get('test_rmse', None)),
                            'MAE': metrics.get('mae', metrics.get('test_mae', None)),
                            'R2': metrics.get('r2', metrics.get('test_r2', None)),
                            'MSE': metrics.get('mse', metrics.get('test_mse', None))
                        })
                        print(f"  ✓ {name}: RMSE={metrics.get('rmse', 'N/A')}")
                    else:
                        print(f"  ✗ {name}: No metrics found")
        else:
            print(f"\n{model_type}: File not found - {filename}")
    
    return pd.DataFrame(all_metrics)

def create_metrics_summary_table(df, output_path):
    """Create a formatted metrics summary table."""
    # Clean up data
    df = df.dropna(subset=['RMSE'])  # Remove rows without RMSE
    
    # Sort by RMSE
    df = df.sort_values('RMSE')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = df[['Model', 'RMSE', 'MAE', 'R2', 'MSE']].round(3)
    
    # Create table
    table = ax.table(cellText=table_data.values,
                     colLabels=table_data.columns,
                     cellLoc='center',
                     loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
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
    
    # Highlight best values
    best_rmse_idx = df['RMSE'].idxmin()
    best_mae_idx = df['MAE'].idxmin()
    best_r2_idx = df['R2'].idxmax()
    
    # Color best RMSE row
    for j in range(len(table_data.columns)):
        table[(best_rmse_idx + 1, j)].set_facecolor('#92D050')
    
    plt.title('Model Performance Metrics Summary', fontsize=16, pad=20, weight='bold')
    plt.text(0.5, -0.05, f'Total Models: {len(df)} | Best RMSE: {df.loc[best_rmse_idx, "Model"]} ({df.loc[best_rmse_idx, "RMSE"]:.3f})', 
             ha='center', transform=ax.transAxes, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return len(df)

def main():
    """Main function to regenerate metrics summary."""
    print("=" * 80)
    print("REGENERATING METRICS SUMMARY TABLE")
    print("=" * 80)
    
    setup_visualization_style()
    
    # Collect all metrics
    df = collect_all_metrics()
    
    print(f"\n\nTotal models with metrics: {len(df)}")
    print(f"Model types: {df['Model Type'].value_counts().to_dict()}")
    
    if len(df) == 0:
        print("\n✗ No models with metrics found!")
        return
    
    # Create output directory
    output_dir = settings.VISUALIZATION_DIR / 'performance'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate table
    output_path = output_dir / 'metrics_summary_table.png'
    model_count = create_metrics_summary_table(df, output_path)
    
    print(f"\n✓ Metrics summary table created with {model_count} models")
    print(f"✓ Saved to: {output_path}")
    
    # Verify content
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    model_counts = df['Model Type'].value_counts()
    print("\nModels included in summary:")
    for model_type, count in model_counts.items():
        print(f"  {model_type}: {count} models")
    
    expected_counts = {
        'Linear Regression': 4,
        'ElasticNet': 4,
        'XGBoost': 8,
        'LightGBM': 8,
        'CatBoost': 8
    }
    
    print("\nExpected vs Actual:")
    all_complete = True
    for model_type, expected in expected_counts.items():
        actual = model_counts.get(model_type, 0)
        status = "✓" if actual == expected else "✗"
        print(f"  {status} {model_type}: {actual}/{expected}")
        if actual != expected:
            all_complete = False
    
    if all_complete:
        print("\n✓ ALL 32 MODELS INCLUDED IN SUMMARY TABLE")
    else:
        print(f"\n✗ INCOMPLETE: Only {len(df)} of 32 expected models included")

if __name__ == "__main__":
    main()