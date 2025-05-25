#!/usr/bin/env python3
"""Regenerate all_models_comparison.csv with all model metrics."""

import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import settings

def aggregate_all_metrics():
    """Aggregate metrics from all individual CSV files."""
    metrics_dir = settings.METRICS_DIR
    
    # List of metric files to aggregate
    metric_files = [
        'linear_regression_metrics.csv',
        'elasticnet_metrics.csv',
        'xgboost_metrics.csv',
        'lightgbm_metrics.csv',
        'catboost_categorical_metrics.csv'
    ]
    
    all_metrics = []
    
    for filename in metric_files:
        filepath = metrics_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            all_metrics.append(df)
            print(f"✓ Loaded {len(df)} entries from {filename}")
        else:
            print(f"✗ File not found: {filename}")
    
    # Combine all metrics
    if all_metrics:
        combined_df = pd.concat(all_metrics, ignore_index=True)
        
        # Save to all_models_comparison.csv
        output_file = metrics_dir / 'all_models_comparison.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved {len(combined_df)} total entries to {output_file}")
        
        # Show summary
        print("\nModel counts by type:")
        if 'model_type' in combined_df.columns:
            print(combined_df['model_type'].value_counts())
    else:
        print("No metrics found to aggregate!")

if __name__ == "__main__":
    aggregate_all_metrics()
