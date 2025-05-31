"""Regenerate metrics table with proper model names."""

import pickle
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.config.settings import MODEL_DIR, VISUALIZATION_DIR
from src.utils.io import load_all_models
from src.visualization.core.registry import get_adapter_for_model
from src.visualization.plots.metrics import MetricsTable


def regenerate_metrics_table():
    """Regenerate the metrics summary table with proper model names."""
    
    print("Loading all models...")
    
    # Load all models
    all_models = load_all_models()
    
    # Create list of model data
    model_list = []
    
    # Process each model
    for model_name, model_data in all_models.items():
        print(f"  - {model_name}")
        
        # Ensure model_data has model_name field
        if isinstance(model_data, dict) and 'model_name' not in model_data:
            model_data['model_name'] = model_name
        
        # Create adapter
        try:
            adapter = get_adapter_for_model(model_data)
            model_list.append(adapter)
        except Exception as e:
            print(f"    Warning: Could not create adapter for {model_name}: {e}")
            # Fall back to raw model data if it's a dict
            if isinstance(model_data, dict):
                model_list.append(model_data)
    
    print(f"\nTotal models to include in table: {len(model_list)}")
    
    # Create metrics table
    print("\nCreating metrics table...")
    
    config = {
        'save': True,
        'output_dir': VISUALIZATION_DIR / "performance",
        'dpi': 300,
        'format': 'png',
        'metrics': ['RMSE', 'MAE', 'R2', 'MSE']
    }
    
    try:
        metrics_table = MetricsTable(models=model_list, config=config)
        fig = metrics_table.plot()
        print("✓ Metrics table created successfully!")
        
        # Also create a CSV version for debugging
        import pandas as pd
        
        metrics_data = []
        for model in model_list:
            try:
                # Get model metadata
                if hasattr(model, 'get_metadata'):
                    metadata = model.get_metadata()
                    model_name = metadata.get('model_name', 'Unknown')
                else:
                    model_name = model.get('model_name', 'Unknown')
                
                # Get metrics
                if hasattr(model, 'get_metrics'):
                    metrics = model.get_metrics()
                else:
                    metrics = {
                        'RMSE': model.get('RMSE', None),
                        'MAE': model.get('MAE', None),
                        'R2': model.get('R2', None),
                        'MSE': model.get('MSE', None)
                    }
                
                # Add to data
                model_metrics = {'Model': model_name}
                model_metrics.update(metrics)
                metrics_data.append(model_metrics)
                
            except Exception as e:
                print(f"    Warning: Could not extract metrics for model: {e}")
        
        # Save CSV
        metrics_df = pd.DataFrame(metrics_data)
        csv_path = VISUALIZATION_DIR / "performance" / "metrics_summary.csv"
        metrics_df.to_csv(csv_path, index=False)
        print(f"✓ Metrics CSV saved to {csv_path}")
        
    except Exception as e:
        print(f"✗ Error creating metrics table: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    regenerate_metrics_table()