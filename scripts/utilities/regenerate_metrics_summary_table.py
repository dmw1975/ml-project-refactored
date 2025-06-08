#!/usr/bin/env python3
"""Regenerate the metrics summary table with updated formatting."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from src.utils.io import load_all_models
from src.visualization.plots.metrics import MetricsTable
from src.config import settings


def regenerate_metrics_table():
    """Regenerate the metrics summary table with sector table formatting."""
    print("Loading all models...")
    
    # Load all models
    models = load_all_models()
    
    if not models:
        print("No models found!")
        return False
    
    # Convert to list of model data
    model_list = list(models.values())
    print(f"Found {len(model_list)} models")
    
    # Create configuration
    config = {
        'save': True,
        'show': False,
        'output_dir': settings.VISUALIZATION_DIR / 'performance',
        'metrics': ['RMSE', 'MAE', 'R2', 'MSE'],
        'dpi': 300,
        'format': 'png'
    }
    
    print("Creating metrics summary table with updated formatting...")
    
    try:
        # Create metrics table
        metrics_table = MetricsTable(model_list, config)
        fig = metrics_table.plot()
        
        print(f"✓ Successfully created metrics summary table at:")
        print(f"  {settings.VISUALIZATION_DIR / 'performance' / 'metrics_summary_table.png'}")
        return True
        
    except Exception as e:
        print(f"✗ Error creating metrics table: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = regenerate_metrics_table()
    sys.exit(0 if success else 1)