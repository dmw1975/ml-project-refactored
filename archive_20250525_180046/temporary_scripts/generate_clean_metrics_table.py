#!/usr/bin/env python3
"""
Generate a clean metrics summary table with all available models
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import settings
from utils import io
import visualization_new as viz
from visualization_new.utils.io import load_all_models


def generate_clean_metrics_table():
    """Generate metrics table with properly loaded models"""
    
    print("\nGenerating clean metrics summary table...")
    
    # Load all models
    all_models = load_all_models()
    
    if not all_models:
        print("No models found!")
        return
    
    print(f"Found {len(all_models)} models total")
    
    # Filter out models without proper metrics
    valid_models = {}
    for name, model_data in all_models.items():
        if isinstance(model_data, dict):
            # Check if it has the required fields
            if 'RMSE' in model_data and model_data['RMSE'] > 0:
                valid_models[name] = model_data
                print(f"✓ {name}: RMSE={model_data['RMSE']:.4f}")
            else:
                print(f"✗ {name}: Missing or invalid metrics")
    
    print(f"\nValid models for visualization: {len(valid_models)}")
    
    if not valid_models:
        print("No valid models found!")
        return
    
    # Create metrics table
    model_list = list(valid_models.values())
    
    try:
        # Use the visualization function
        viz.create_metrics_table(model_list)
        
        # Check if it was created
        metrics_table = settings.VISUALIZATION_DIR / "performance" / "metrics_summary_table.png"
        if metrics_table.exists():
            print(f"\n✓ Metrics table created: {metrics_table}")
        else:
            print("\n✗ Metrics table creation failed")
            
    except Exception as e:
        print(f"\nError creating metrics table: {e}")
        
    # Also create a clean CSV
    print("\nCreating clean metrics CSV...")
    metrics_data = []
    
    for name, model_data in valid_models.items():
        metrics_data.append({
            'model_name': name,
            'RMSE': model_data.get('RMSE', 0),
            'MAE': model_data.get('MAE', 0),
            'MSE': model_data.get('MSE', 0),
            'R2': model_data.get('R2', 0),
            'model_type': model_data.get('model_type', 'Unknown')
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(metrics_data)
    df = df.sort_values('RMSE')
    
    # Save to CSV
    csv_path = settings.METRICS_DIR / "clean_models_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Clean metrics CSV saved: {csv_path}")
    
    # Display summary
    print("\nModel Performance Summary (sorted by RMSE):")
    print("-" * 60)
    print(df[['model_name', 'RMSE', 'R2', 'model_type']].to_string(index=False))
    
    return df


if __name__ == "__main__":
    generate_clean_metrics_table()