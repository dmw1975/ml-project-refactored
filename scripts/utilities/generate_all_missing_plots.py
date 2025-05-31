#!/usr/bin/env python3
"""Generate all missing visualizations."""

import os
from pathlib import Path
from visualization_new.core.registry import get_adapter_for_model
from visualization_new.utils.io import load_all_models
from visualization_new.plots.metrics import create_performance_comparison
from visualization_new.plots.features import create_cross_model_feature_importance
from visualization_new.plots.residuals import create_residual_plots
from visualization_new.plots.baselines import create_baseline_comparisons

def generate_all_visualizations():
    """Generate all missing visualizations."""
    print("Loading all models...")
    all_models = load_all_models()
    print(f"Loaded {sum(len(models) for models in all_models.values())} models across {len(all_models)} types")
    
    # Create output directories
    viz_dir = Path("outputs/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate performance comparison plots
    print("\n1. Generating performance comparison plots...")
    try:
        create_performance_comparison(all_models, viz_dir)
        print("✓ Performance comparison plots created")
    except Exception as e:
        print(f"✗ Error creating performance comparison: {e}")
    
    # 2. Generate cross-model feature importance
    print("\n2. Generating cross-model feature importance...")
    try:
        create_cross_model_feature_importance(all_models, viz_dir)
        print("✓ Cross-model feature importance created")
    except Exception as e:
        print(f"✗ Error creating cross-model feature importance: {e}")
    
    # 3. Generate baseline comparisons
    print("\n3. Generating baseline comparisons...")
    try:
        create_baseline_comparisons(all_models, viz_dir)
        print("✓ Baseline comparisons created")
    except Exception as e:
        print(f"✗ Error creating baseline comparisons: {e}")
    
    # 4. Generate residual plots for each model
    print("\n4. Generating residual plots...")
    residuals_dir = viz_dir / "residuals"
    residuals_dir.mkdir(exist_ok=True)
    
    for model_type, models_dict in all_models.items():
        print(f"\n  Processing {model_type} models...")
        for model_name, model_data in models_dict.items():
            try:
                # Create adapter
                adapter = get_adapter_for_model(model_type, model_data)
                
                # Generate residual plot
                output_path = residuals_dir / f"{model_name}_residuals.png"
                if not output_path.exists():
                    create_residual_plots(adapter, output_path)
                    print(f"    ✓ {model_name}")
                else:
                    print(f"    - {model_name} (already exists)")
                    
            except Exception as e:
                print(f"    ✗ {model_name}: {e}")
    
    # 5. List what plots exist
    print("\n5. Checking generated plots...")
    for plot_type in ["metrics", "features", "residuals", "baselines"]:
        plot_dir = viz_dir / plot_type
        if plot_dir.exists():
            plots = list(plot_dir.glob("*.png"))
            print(f"  {plot_type}: {len(plots)} plots")
            if len(plots) > 0 and len(plots) <= 5:
                for plot in plots:
                    print(f"    - {plot.name}")

if __name__ == "__main__":
    generate_all_visualizations()