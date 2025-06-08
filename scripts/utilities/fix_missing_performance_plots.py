#!/usr/bin/env python3
"""
Generate missing performance plots for XGBoost and CatBoost models.
"""

import sys
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.utils.io import load_model
from src.config import settings
from src.visualization.plots.optimization import (
    plot_optimization_history,
    plot_param_importance,
    plot_improved_contour,
    plot_basic_vs_optuna,
    plot_optuna_improvement,
    plot_hyperparameter_comparison
)
from src.visualization.core.interfaces import VisualizationConfig


def generate_xgboost_missing_plots():
    """Generate missing performance plots for XGBoost models."""
    print("\nGenerating missing XGBoost performance plots...")
    
    # Load XGBoost models
    xgb_models = load_model("xgboost_models.pkl", settings.MODEL_DIR)
    if not xgb_models:
        print("No XGBoost models found!")
        return
    
    output_dir = settings.VISUALIZATION_DIR / "performance" / "xgboost"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate contour plots for Optuna models
    print("\n1. Generating contour plots for XGBoost Optuna models...")
    optuna_models = {k: v for k, v in xgb_models.items() if 'optuna' in k and 'study' in v}
    
    for model_name, model_data in optuna_models.items():
        try:
            # Ensure model_name is set
            model_data['model_name'] = model_name
            
            # Try to generate contour plot
            if 'study' in model_data and model_data['study'] is not None:
                print(f"  Generating contour plot for {model_name}...")
                config = VisualizationConfig(
                    output_dir=output_dir,
                    save=True,
                    show=False,
                    format='png',
                    dpi=300
                )
                plot_improved_contour(model_data['study'], config, model_name)
                print(f"    ✓ Created contour plot")
            else:
                print(f"    ✗ No study object found for {model_name}")
        except Exception as e:
            print(f"    ✗ Error creating contour plot for {model_name}: {e}")
    
    # 2. Generate basic vs optuna comparison
    print("\n2. Generating XGBoost basic vs Optuna comparison...")
    try:
        config = VisualizationConfig(
            output_dir=output_dir,
            save=True,
            show=False,
            format='png',
            dpi=300
        )
        plot_basic_vs_optuna(list(xgb_models.values()), config, model_family='xgboost')
        print("  ✓ Created basic_vs_optuna_comparison.png")
    except Exception as e:
        print(f"  ✗ Error creating basic vs optuna comparison: {e}")
        traceback.print_exc()
    
    # 3. Generate Optuna improvement plot
    print("\n3. Generating XGBoost Optuna improvement plot...")
    try:
        config = VisualizationConfig(
            output_dir=output_dir,
            save=True,
            show=False,
            format='png',
            dpi=300
        )
        plot_optuna_improvement(list(xgb_models.values()), config, model_family='xgboost')
        print("  ✓ Created optuna_improvement.png")
    except Exception as e:
        print(f"  ✗ Error creating optuna improvement plot: {e}")
        traceback.print_exc()


def generate_catboost_missing_plots():
    """Generate missing performance plots for CatBoost models."""
    print("\n\nGenerating missing CatBoost performance plots...")
    
    # Load CatBoost models
    cb_models = load_model("catboost_models.pkl", settings.MODEL_DIR)
    if not cb_models:
        print("No CatBoost models found!")
        return
    
    output_dir = settings.VISUALIZATION_DIR / "performance" / "catboost"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate optimization history for Optuna models
    print("\n1. Generating optimization history plots for CatBoost Optuna models...")
    optuna_models = {k: v for k, v in cb_models.items() if 'optuna' in k}
    
    for model_name, model_data in optuna_models.items():
        try:
            # Ensure model_name is set
            model_data['model_name'] = model_name
            
            # Check if study exists
            if 'study' in model_data and model_data['study'] is not None:
                print(f"  Generating optimization history for {model_name}...")
                config = VisualizationConfig(
                    output_dir=output_dir,
                    save=True,
                    show=False,
                    format='png',
                    dpi=300
                )
                plot_optimization_history(model_data, config)
                print(f"    ✓ Created optimization history")
            else:
                print(f"    ✗ No study object found for {model_name}")
        except Exception as e:
            print(f"    ✗ Error creating optimization history for {model_name}: {e}")
            traceback.print_exc()
    
    # 2. Generate parameter importance plots
    print("\n2. Generating parameter importance plots for CatBoost Optuna models...")
    for model_name, model_data in optuna_models.items():
        try:
            # Ensure model_name is set
            model_data['model_name'] = model_name
            
            if 'study' in model_data and model_data['study'] is not None:
                print(f"  Generating parameter importance for {model_name}...")
                config = VisualizationConfig(
                    output_dir=output_dir,
                    save=True,
                    show=False,
                    format='png',
                    dpi=300
                )
                plot_param_importance(model_data['study'], config, model_name)
                print(f"    ✓ Created parameter importance")
            else:
                print(f"    ✗ No study object found for {model_name}")
        except Exception as e:
            print(f"    ✗ Error creating parameter importance for {model_name}: {e}")
            traceback.print_exc()
    
    # 3. Generate contour plots
    print("\n3. Generating contour plots for CatBoost Optuna models...")
    for model_name, model_data in optuna_models.items():
        try:
            # Ensure model_name is set
            model_data['model_name'] = model_name
            
            if 'study' in model_data and model_data['study'] is not None:
                print(f"  Generating contour plot for {model_name}...")
                config = VisualizationConfig(
                    output_dir=output_dir,
                    save=True,
                    show=False,
                    format='png',
                    dpi=300
                )
                plot_improved_contour(model_data['study'], config, model_name)
                print(f"    ✓ Created contour plot")
            else:
                print(f"    ✗ No study object found for {model_name}")
        except Exception as e:
            print(f"    ✗ Error creating contour plot for {model_name}: {e}")
    
    # 4. Generate basic vs optuna comparison
    print("\n4. Generating CatBoost basic vs Optuna comparison...")
    try:
        config = VisualizationConfig(
            output_dir=output_dir,
            save=True,
            show=False,
            format='png',
            dpi=300
        )
        plot_basic_vs_optuna(list(cb_models.values()), config, model_family='catboost')
        print("  ✓ Created basic_vs_optuna_comparison.png")
    except Exception as e:
        print(f"  ✗ Error creating basic vs optuna comparison: {e}")
        traceback.print_exc()
    
    # 5. Generate Optuna improvement plot
    print("\n5. Generating CatBoost Optuna improvement plot...")
    try:
        config = VisualizationConfig(
            output_dir=output_dir,
            save=True,
            show=False,
            format='png',
            dpi=300
        )
        plot_optuna_improvement(list(cb_models.values()), config, model_family='catboost')
        print("  ✓ Created optuna_improvement.png")
    except Exception as e:
        print(f"  ✗ Error creating optuna improvement plot: {e}")
        traceback.print_exc()


def check_generated_plots():
    """Check what plots were successfully generated."""
    print("\n\nChecking generated performance plots...")
    
    # Check XGBoost
    xgb_dir = settings.VISUALIZATION_DIR / "performance" / "xgboost"
    if xgb_dir.exists():
        xgb_plots = list(xgb_dir.glob("*.png"))
        print(f"\nXGBoost performance plots ({len(xgb_plots)} total):")
        for plot in sorted(xgb_plots)[:10]:  # Show first 10
            print(f"  - {plot.name}")
        if len(xgb_plots) > 10:
            print(f"  ... and {len(xgb_plots) - 10} more")
    
    # Check CatBoost
    cb_dir = settings.VISUALIZATION_DIR / "performance" / "catboost"
    if cb_dir.exists():
        cb_plots = list(cb_dir.glob("*.png"))
        print(f"\nCatBoost performance plots ({len(cb_plots)} total):")
        for plot in sorted(cb_plots)[:10]:  # Show first 10
            print(f"  - {plot.name}")
        if len(cb_plots) > 10:
            print(f"  ... and {len(cb_plots) - 10} more")


def main():
    """Main function."""
    print("Generating missing performance plots for XGBoost and CatBoost...")
    
    # Generate XGBoost missing plots
    generate_xgboost_missing_plots()
    
    # Generate CatBoost missing plots
    generate_catboost_missing_plots()
    
    # Check what was generated
    check_generated_plots()
    
    print("\nDone!")


if __name__ == "__main__":
    main()