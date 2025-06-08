#!/usr/bin/env python3
"""Regenerate baseline comparison plots with all models included."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.visualization.plots.baselines import create_metric_baseline_comparison

def regenerate_baseline_plots():
    """Regenerate all baseline comparison plots."""
    
    print("=" * 80)
    print("REGENERATING BASELINE COMPARISON PLOTS")
    print("=" * 80)
    
    # Input and output paths
    baseline_csv = settings.METRICS_DIR / "baseline_comparison.csv"
    output_dir = settings.VISUALIZATION_DIR / "statistical_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots for each baseline type and metric
    baseline_types = ['Mean', 'Median', 'Random']
    
    for baseline_type in baseline_types:
        output_file = output_dir / f"baseline_comparison_{baseline_type.lower()}.png"
        
        print(f"\nGenerating {baseline_type} baseline comparison...")
        
        try:
            result = create_metric_baseline_comparison(
                baseline_data_path=str(baseline_csv),
                output_path=str(output_file),
                metric='RMSE',
                baseline_type=baseline_type,
                figsize=(14, 10),
                dpi=300
            )
            
            if result:
                print(f"✓ Created: {output_file}")
            else:
                print(f"✗ Failed to create {baseline_type} comparison")
                
        except Exception as e:
            print(f"✗ Error creating {baseline_type} comparison: {e}")
            import traceback
            traceback.print_exc()
    
    # Verify results
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    expected_files = [
        'baseline_comparison_mean.png',
        'baseline_comparison_median.png',
        'baseline_comparison_random.png'
    ]
    
    print("Baseline comparison files:")
    for filename in expected_files:
        filepath = output_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} - MISSING")
    
    # Check content by loading CSV to verify model inclusion
    import pandas as pd
    df = pd.read_csv(baseline_csv)
    
    # Check which models are included
    model_types = set()
    for model in df['Model'].unique():
        if 'LightGBM' in model:
            model_types.add('LightGBM')
        elif 'CatBoost' in model:
            model_types.add('CatBoost')
        elif 'XGBoost' in model:
            model_types.add('XGBoost')
        elif 'ElasticNet' in model:
            model_types.add('ElasticNet')
        elif 'LR_' in model or model.startswith('LR'):
            model_types.add('Linear Regression')
    
    print("\nModels included in baseline comparisons:")
    expected_types = ['Linear Regression', 'ElasticNet', 'XGBoost', 'LightGBM', 'CatBoost']
    all_included = True
    for model_type in expected_types:
        if model_type in model_types:
            print(f"  ✓ {model_type}")
        else:
            print(f"  ✗ {model_type} - MISSING")
            all_included = False
    
    if all_included:
        print("\n✓ ALL MODEL TYPES INCLUDED IN BASELINE COMPARISONS")
    else:
        print("\n✗ Some model types are still missing!")

if __name__ == "__main__":
    regenerate_baseline_plots()