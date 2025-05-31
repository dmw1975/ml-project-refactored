#!/usr/bin/env python3
"""Generate missing performance plots for CatBoost and XGBoost models."""

import os
from pathlib import Path
from visualization_new.utils.io import load_all_models
from visualization_new.plots.optimization import (
    plot_optimization_history, 
    plot_param_importance, 
    plot_contour,
    plot_basic_vs_optuna,
    plot_optuna_improvement
)
from visualization_new.viz_factory import create_hyperparameter_comparison
from visualization_new.core.interfaces import VisualizationConfig
from config import settings

def generate_catboost_performance_plots():
    """Generate missing CatBoost performance visualizations."""
    print("\nGenerating CatBoost performance visualizations...")
    
    # Load all models
    all_models = load_all_models()
    catboost_models = {name: model for name, model in all_models.items() 
                      if 'catboost' in name.lower()}
    
    if not catboost_models:
        print("No CatBoost models found.")
        return
    
    # Set up output directory
    perf_dir = settings.VISUALIZATION_DIR / "performance" / "catboost"
    perf_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = VisualizationConfig(
        output_dir=perf_dir,
        format="png",
        dpi=300,
        save=True,
        show=False
    )
    
    # Generate Optuna visualizations for models with studies
    optuna_models = {name: model for name, model in catboost_models.items() 
                    if 'optuna' in name and 'study' in model}
    
    if optuna_models:
        print("  Generating Optuna visualizations...")
        for model_name, model_data in optuna_models.items():
            study = model_data.get('study')
            if study:
                print(f"    Creating plots for {model_name}...")
                
                # Optimization history
                hist_path = plot_optimization_history(study, config, model_name)
                if hist_path:
                    print(f"      ✓ Optimization history")
                
                # Parameter importance
                param_path = plot_param_importance(study, config, model_name)
                if param_path:
                    print(f"      ✓ Parameter importance")
                
                # Contour plot
                contour_path = plot_contour(study, config, model_name)
                if contour_path:
                    print(f"      ✓ Contour plot")
    
    # Create hyperparameter comparisons
    print("  Generating hyperparameter comparisons...")
    model_data_list = list(catboost_models.values())
    
    for param in ['learning_rate', 'depth', 'iterations', 'l2_leaf_reg']:
        try:
            output_path = create_hyperparameter_comparison(
                model_data_list, param, config, "catboost"
            )
            if output_path:
                print(f"    ✓ {param} comparison")
        except Exception as e:
            print(f"    ✗ Error creating {param} comparison: {e}")
    
    # Create basic vs optuna comparison
    try:
        output_path = plot_basic_vs_optuna(model_data_list, config, "catboost")
        if output_path:
            print("  ✓ Basic vs Optuna comparison")
    except Exception as e:
        print(f"  ✗ Error creating basic vs optuna comparison: {e}")
    
    # Create optuna improvement plot
    try:
        output_path = plot_optuna_improvement(model_data_list, config, "catboost")
        if output_path:
            print("  ✓ Optuna improvement plot")
    except Exception as e:
        print(f"  ✗ Error creating optuna improvement plot: {e}")
    
    print(f"CatBoost performance visualizations saved to: {perf_dir}")

def generate_xgboost_performance_plots():
    """Generate missing XGBoost performance visualizations."""
    print("\nGenerating XGBoost performance visualizations...")
    
    # Load all models
    all_models = load_all_models()
    xgboost_models = {name: model for name, model in all_models.items() 
                     if 'xgboost' in name.lower()}
    
    if not xgboost_models:
        print("No XGBoost models found.")
        return
    
    # Set up output directory
    perf_dir = settings.VISUALIZATION_DIR / "performance" / "xgboost"
    perf_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = VisualizationConfig(
        output_dir=perf_dir,
        format="png",
        dpi=300,
        save=True,
        show=False
    )
    
    # Generate Optuna visualizations for models with studies
    optuna_models = {name: model for name, model in xgboost_models.items() 
                    if 'optuna' in name and 'study' in model}
    
    if optuna_models:
        print("  Generating Optuna visualizations...")
        for model_name, model_data in optuna_models.items():
            study = model_data.get('study')
            if study:
                print(f"    Creating plots for {model_name}...")
                
                # Optimization history
                hist_path = plot_optimization_history(study, config, model_name)
                if hist_path:
                    print(f"      ✓ Optimization history")
                
                # Parameter importance
                param_path = plot_param_importance(study, config, model_name)
                if param_path:
                    print(f"      ✓ Parameter importance")
                
                # Contour plot
                contour_path = plot_contour(study, config, model_name)
                if contour_path:
                    print(f"      ✓ Contour plot")
    
    # Get model data list for comparison plots
    model_data_list = list(xgboost_models.values())
    
    # Create basic vs optuna comparison
    try:
        output_path = plot_basic_vs_optuna(model_data_list, config, "xgboost")
        if output_path:
            print("  ✓ Basic vs Optuna comparison")
    except Exception as e:
        print(f"  ✗ Error creating basic vs optuna comparison: {e}")
    
    # Create optuna improvement plot
    try:
        output_path = plot_optuna_improvement(model_data_list, config, "xgboost")
        if output_path:
            print("  ✓ Optuna improvement plot")
    except Exception as e:
        print(f"  ✗ Error creating optuna improvement plot: {e}")
    
    print(f"XGBoost performance visualizations saved to: {perf_dir}")

def main():
    """Generate all missing performance visualizations."""
    print("Generating missing performance visualizations...")
    
    generate_catboost_performance_plots()
    generate_xgboost_performance_plots()
    
    print("\n✅ Done! Missing performance visualizations have been generated.")

if __name__ == "__main__":
    main()