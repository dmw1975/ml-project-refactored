"""Script to regenerate improved contour plots for better readability.

This script regenerates the contour plots for the optimization results
using the improved layout for better readability.

Usage:
    python regenerate_contour_plots.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings
from utils import io
from visualization_new.plots.optimization import plot_improved_contour

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Please install optuna.")
    sys.exit(1)

def regenerate_contour_plots():
    """Regenerate improved contour plots for each model with an Optuna study."""
    print("Regenerating improved contour plots for better readability...")
    
    # Path to models directory where the study objects should be
    models_dir = settings.MODEL_DIR
    
    # Try to load model data
    try:
        # Load XGBoost models
        xgb_models = io.load_model("xgboost_models.pkl", models_dir)
        print(f"Loaded XGBoost models: {len(xgb_models) if xgb_models else 0}")
        
        # Load LightGBM models
        lgbm_models = io.load_model("lightgbm_models.pkl", models_dir)
        print(f"Loaded LightGBM models: {len(lgbm_models) if lgbm_models else 0}")
        
        # Load CatBoost models
        catboost_models = io.load_model("catboost_models.pkl", models_dir)
        print(f"Loaded CatBoost models: {len(catboost_models) if catboost_models else 0}")
        
        # Load ElasticNet models
        elasticnet_models = io.load_model("elasticnet_models.pkl", models_dir)
        print(f"Loaded ElasticNet models: {len(elasticnet_models) if elasticnet_models else 0}")
        
        # Collect all models with study objects
        all_model_configs = []
        
        # Process each model dictionary and check for study objects
        models_to_process = [
            (xgb_models, 'xgboost'),
            (lgbm_models, 'lightgbm'),
            (catboost_models, 'catboost'),
            (elasticnet_models, 'elasticnet')
        ]
        
        for models_dict, model_type in models_to_process:
            if not models_dict or not isinstance(models_dict, dict):
                continue
                
            for model_name, model_data in models_dict.items():
                if not isinstance(model_data, dict):
                    continue
                    
                if 'study' in model_data and model_data['study'] is not None:
                    all_model_configs.append({
                        'model_name': model_name,
                        'study': model_data['study'],
                        'model_type': model_type
                    })
        
        print(f"Total models with optimization studies: {len(all_model_configs)}")
        
        # Generate improved contour plots for each model
        generated_count = 0
        skipped_count = 0
        
        for model_config in all_model_configs:
            model_name = model_config['model_name']
            study = model_config['study']
            model_type = model_config['model_type']
            
            # Create output directory
            output_dir = settings.VISUALIZATION_DIR / "performance" / model_type
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"Generating improved contour plot for {model_name}...")
            
            # Create configuration
            config = {
                'output_dir': output_dir,
                'format': 'png',
                'dpi': 300
            }
            
            # Generate improved contour plot
            try:
                output_path = plot_improved_contour(study, config, model_name)
                if output_path:
                    # Clean up the file name for better consistency
                    new_name = os.path.join(
                        os.path.dirname(output_path),
                        f"{model_name}_contour.{config.get('format', 'png')}"
                    )

                    try:
                        # Rename from *_improved_contour.png to *_contour.png
                        os.rename(output_path, new_name)
                        print(f"Generated and renamed to: {new_name}")
                        generated_count += 1
                    except Exception as e:
                        print(f"Error renaming {output_path} to {new_name}: {e}")
                        print(f"Generated: {output_path}")
                        generated_count += 1
                else:
                    print(f"Failed to generate contour plot for {model_name}")
                    skipped_count += 1
            except Exception as e:
                print(f"Error generating contour plot for {model_name}: {e}")
                skipped_count += 1
        
        print(f"\nSummary:")
        print(f"  - Generated: {generated_count} plots")
        print(f"  - Skipped: {skipped_count} models")
        
    except Exception as e:
        print(f"Error loading models: {e}")

if __name__ == "__main__":
    regenerate_contour_plots()