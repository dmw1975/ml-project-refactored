#!/usr/bin/env python3
"""Fix XGBoost model names and generate residual plots."""

import pickle
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.visualization.plots.residuals import ResidualPlot

def check_and_fix_xgboost_models():
    """Check and fix model_name field in XGBoost models."""
    
    # Load XGBoost models
    xgboost_path = project_root / 'outputs' / 'models' / 'xgboost_models.pkl'
    
    print(f"Loading XGBoost models from: {xgboost_path}")
    
    try:
        with open(xgboost_path, 'rb') as f:
            xgboost_models = pickle.load(f)
        
        print(f"\nFound {len(xgboost_models)} XGBoost models")
        
        # Check and fix model names
        models_fixed = False
        
        for i, (key, model_data) in enumerate(xgboost_models.items()):
            print(f"\n{i+1}. Key: {key}")
            print(f"   Type: {type(model_data)}")
            
            if isinstance(model_data, dict):
                # Check if model_name exists
                if 'model_name' in model_data:
                    print(f"   model_name: {model_data['model_name']}")
                else:
                    print("   model_name: MISSING - Adding...")
                    
                    # Parse the key to determine the model name
                    # Key format may already have XGBoost_ prefix
                    if key.startswith('XGBoost_'):
                        # Already has prefix, just use the key
                        model_name = key
                    else:
                        # Need to add XGBoost prefix
                        parts = key.split('_')
                        
                        if len(parts) >= 2:
                            features = parts[0]
                            
                            # Handle Base_Random and Yeo_Random cases
                            if len(parts) >= 3 and parts[1] == 'Random':
                                features = f"{parts[0]}_{parts[1]}"
                                variant_parts = parts[2:]
                            else:
                                variant_parts = parts[1:]
                            
                            variant = '_'.join(variant_parts)
                            
                            # Construct model name
                            model_name = f"XGBoost_{features}_{variant}"
                        else:
                            print(f"   ERROR: Could not parse key '{key}' to create model_name")
                            continue
                    
                    model_data['model_name'] = model_name
                    models_fixed = True
                    print(f"   Added model_name: {model_name}")
                
                # Show other fields
                print(f"   Fields: {list(model_data.keys())}")
            else:
                print(f"   ERROR: Model data is not a dict, it's {type(model_data)}")
        
        # Save fixed models if any were modified
        if models_fixed:
            print("\nSaving fixed XGBoost models...")
            with open(xgboost_path, 'wb') as f:
                pickle.dump(xgboost_models, f)
            print("Models saved successfully!")
        else:
            print("\nAll models already have model_name field, no fixes needed.")
            
        return xgboost_models
        
    except Exception as e:
        print(f"Error loading/fixing XGBoost models: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_xgboost_residual_plots(models):
    """Generate residual plots for XGBoost models."""
    
    print("\n" + "="*60)
    print("Generating XGBoost residual plots...")
    print("="*60)
    
    if models is None:
        print("No models to process")
        return
    
    plots_dir = project_root / 'outputs' / 'visualization' / 'residuals' / 'xgboost'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    successful_plots = 0
    failed_plots = 0
    
    for key, model_data in models.items():
        if isinstance(model_data, dict) and 'model' in model_data:
            try:
                print(f"\nGenerating residual plot for: {key}")
                model_name = model_data.get('model_name', f'XGBoost_{key}')
                
                # Create ResidualPlot instance with model data
                plotter = ResidualPlot(model_data)
                
                # Generate plot
                plot_path = plots_dir / f"{model_name}_residuals.png"
                fig = plotter.plot()
                
                # Save the figure
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                print(f"   ✓ Saved to: {plot_path}")
                successful_plots += 1
                
            except Exception as e:
                print(f"   ✗ Failed to create plot: {str(e)}")
                import traceback
                traceback.print_exc()
                failed_plots += 1
        else:
            print(f"\nSkipping {key} - missing required data")
            failed_plots += 1
    
    print(f"\n{'='*60}")
    print(f"Residual plot generation complete!")
    print(f"Successful: {successful_plots}")
    print(f"Failed: {failed_plots}")
    print(f"Total: {successful_plots + failed_plots}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Fix model names
    models = check_and_fix_xgboost_models()
    
    # Generate residual plots
    if models:
        generate_xgboost_residual_plots(models)
    else:
        print("\nFailed to load models, cannot generate plots.")