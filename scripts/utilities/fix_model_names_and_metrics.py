#!/usr/bin/env python3
"""
Fix model names in model pickle files and regenerate metrics table.
"""

import sys
from pathlib import Path
import pickle
import pandas as pd

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.utils.io import load_model, save_model
from src.config import settings
from src.visualization.viz_factory import create_metrics_table
from src.utils.io import load_all_models


def fix_model_names():
    """Fix missing model_name in XGBoost and CatBoost models."""
    
    # Fix XGBoost models
    print("Fixing XGBoost model names...")
    xgb_models = load_model("xgboost_models.pkl", settings.MODEL_DIR)
    if xgb_models:
        for model_name, model_data in xgb_models.items():
            if 'model_name' not in model_data or model_data.get('model_name') == 'Unknown':
                model_data['model_name'] = model_name
                print(f"  Fixed: {model_name}")
        save_model(xgb_models, "xgboost_models.pkl", settings.MODEL_DIR)
        print(f"  Saved {len(xgb_models)} XGBoost models with fixed names")
    
    # Fix CatBoost models
    print("\nFixing CatBoost model names...")
    cb_models = load_model("catboost_models.pkl", settings.MODEL_DIR)
    if cb_models:
        for model_name, model_data in cb_models.items():
            if 'model_name' not in model_data or model_data.get('model_name') == 'Unknown':
                model_data['model_name'] = model_name
                print(f"  Fixed: {model_name}")
        save_model(cb_models, "catboost_models.pkl", settings.MODEL_DIR)
        print(f"  Saved {len(cb_models)} CatBoost models with fixed names")
    
    # Check LightGBM models too
    print("\nChecking LightGBM model names...")
    lgb_models = load_model("lightgbm_models.pkl", settings.MODEL_DIR)
    if lgb_models:
        fixed_count = 0
        for model_name, model_data in lgb_models.items():
            if 'model_name' not in model_data or model_data.get('model_name') == 'Unknown':
                model_data['model_name'] = model_name
                fixed_count += 1
                print(f"  Fixed: {model_name}")
        if fixed_count > 0:
            save_model(lgb_models, "lightgbm_models.pkl", settings.MODEL_DIR)
            print(f"  Saved {len(lgb_models)} LightGBM models with {fixed_count} fixes")
        else:
            print("  All LightGBM models already have correct names")


def check_all_models():
    """Check model names in all model files."""
    print("\nChecking all model names:")
    
    model_files = [
        "linear_regression_models.pkl",
        "elasticnet_models.pkl", 
        "xgboost_models.pkl",
        "lightgbm_models.pkl",
        "catboost_models.pkl"
    ]
    
    all_models = []
    
    for model_file in model_files:
        models = load_model(model_file, settings.MODEL_DIR)
        if models:
            print(f"\n{model_file}:")
            for model_name, model_data in models.items():
                stored_name = model_data.get('model_name', 'MISSING')
                match = "✓" if stored_name == model_name else "✗"
                print(f"  {match} {model_name} -> stored as: {stored_name}")
                
                # Collect model info for metrics
                metrics = {}
                metrics['Model'] = model_data.get('model_name', model_name)
                metrics['RMSE'] = model_data.get('RMSE', model_data.get('test_rmse', None))
                metrics['MAE'] = model_data.get('MAE', model_data.get('test_mae', None))
                metrics['R2'] = model_data.get('R2', model_data.get('test_r2', None))
                metrics['MSE'] = model_data.get('MSE', model_data.get('test_mse', None))
                
                # Calculate missing metrics if needed
                if metrics['MSE'] is None and metrics['RMSE'] is not None:
                    metrics['MSE'] = metrics['RMSE'] ** 2
                
                all_models.append(metrics)
    
    return all_models


def regenerate_metrics_table(all_models):
    """Regenerate the metrics summary table with all models."""
    print("\nRegenerating metrics summary table...")
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(all_models)
    
    # Remove rows with no metrics
    metrics_df = metrics_df.dropna(subset=['RMSE', 'MAE', 'R2', 'MSE'], how='all')
    
    # Sort by RMSE
    metrics_df = metrics_df.sort_values('RMSE')
    
    print(f"Total models with metrics: {len(metrics_df)}")
    print("\nModel summary:")
    print(metrics_df[['Model', 'RMSE', 'R2']].to_string(index=False))
    
    # Load all models to pass to create_metrics_table
    all_loaded_models = load_all_models()
    
    # Create visualization config
    from src.visualization.core.interfaces import VisualizationConfig
    config = VisualizationConfig(
        output_dir=settings.VISUALIZATION_DIR / "performance",
        save=True,
        show=False,
        format='png',
        dpi=300
    )
    
    # Create the visualization
    create_metrics_table(list(all_loaded_models.values()), config)
    print("\nMetrics summary table regenerated!")


def main():
    """Main function."""
    # First, fix model names
    fix_model_names()
    
    # Check all models and collect metrics
    all_models = check_all_models()
    
    # Regenerate metrics table
    regenerate_metrics_table(all_models)


if __name__ == "__main__":
    main()