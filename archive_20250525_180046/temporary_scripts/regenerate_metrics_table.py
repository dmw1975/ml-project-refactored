#!/usr/bin/env python3
"""Regenerate the metrics summary table with fixed XGBoost metrics."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import required modules
from visualization_new.viz_factory import create_metrics_table
from utils.io import load_all_models
from config import settings

def main():
    """Regenerate metrics summary table."""
    print("Loading all models...")
    all_models = load_all_models()
    
    print(f"Found {len(all_models)} models")
    
    # Create output directory
    output_dir = settings.VISUALIZATION_DIR / "performance"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration for the table
    config = {
        'output_dir': output_dir,
        'save': True,
        'show': False,
        'format': 'png',
        'dpi': 300,
        'metrics': ['RMSE', 'MAE', 'R2', 'MSE']  # Ensure all metrics are included
    }
    
    print("Creating metrics summary table...")
    try:
        # Add model names to the model data dictionaries
        model_list = []
        for model_name, model_data in all_models.items():
            # Make a copy to avoid modifying the original
            model_data_copy = model_data.copy()
            # Add the model name if not present
            if 'model_name' not in model_data_copy:
                model_data_copy['model_name'] = model_name
            model_list.append(model_data_copy)
        
        # Create the metrics table
        fig = create_metrics_table(model_list, config)
        print(f"Metrics table saved to: {output_dir / 'metrics_summary_table.png'}")
    except Exception as e:
        print(f"Error creating metrics table: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()