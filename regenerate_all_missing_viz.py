#!/usr/bin/env python3
"""Regenerate all missing visualizations with fixed models."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.io import load_all_models
from src.evaluation.baseline_significance import run_baseline_significance_analysis
from src.visualization.plots.dataset_comparison import DatasetModelComparisonPlot
from src.data.data import load_scores_data
from src.config import settings

def regenerate_visualizations():
    """Regenerate all visualizations that were missing LightGBM/CatBoost."""
    
    print("Loading all models...")
    all_models = load_all_models()
    print(f"Loaded {len(all_models)} models")
    
    # 1. Regenerate baseline significance analysis
    print("\n1. Regenerating baseline significance analysis...")
    try:
        scores_data = load_scores_data()
    except:
        scores_data = None
    
    stat_test_dir = settings.VISUALIZATION_DIR / "statistical_tests"
    stat_test_dir.mkdir(parents=True, exist_ok=True)
    
    results_df, baseline_plots = run_baseline_significance_analysis(
        models_dict=all_models,
        scores_data=scores_data,
        output_dir=stat_test_dir,
        n_folds=5,
        random_seed=42
    )
    
    if results_df is not None:
        print(f"✓ Generated baseline analysis with {len(results_df)} entries")
        
        # Check if LightGBM and CatBoost are included
        lightgbm_count = results_df['Model'].str.contains('LightGBM').sum()
        catboost_count = results_df['Model'].str.contains('CatBoost').sum()
        print(f"  - LightGBM models: {lightgbm_count}")
        print(f"  - CatBoost models: {catboost_count}")
    
    # 2. Regenerate dataset comparison plots
    print("\n2. Regenerating dataset comparison plots...")
    dc_dir = settings.VISUALIZATION_DIR / "dataset_comparison"
    dc_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert models dict to list of model data
    model_list = list(all_models.values())
    viz = DatasetModelComparisonPlot(model_list)
    
    # Generate plots for each dataset
    for dataset in ['Base', 'Base_Random', 'Yeo', 'Yeo_Random']:
        try:
            fig = viz.plot_dataset_comparison(dataset)
            if fig:
                output_path = dc_dir / f"{dataset}_model_family_comparison.png"
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"✓ Generated {dataset} comparison plot")
        except Exception as e:
            print(f"✗ Error generating {dataset} plot: {e}")
    
    print("\nRegeneration complete!")
    
    # 3. Also regenerate baseline comparison plots
    print("\n3. Regenerating baseline comparison plots...")
    from src.visualization.plots.baselines import visualize_all_baseline_comparisons
    
    try:
        baseline_paths = visualize_all_baseline_comparisons()
        print(f"✓ Generated {len(baseline_paths)} baseline comparison plots")
    except Exception as e:
        print(f"✗ Error generating baseline comparisons: {e}")

if __name__ == "__main__":
    regenerate_visualizations()