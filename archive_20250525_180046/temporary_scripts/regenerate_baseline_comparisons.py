#!/usr/bin/env python3
"""Regenerate baseline comparisons with all models including tree-based models."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import required modules
from utils.io import load_all_models
from evaluation.baselines import run_baseline_evaluation
from visualization_new.plots.baselines import visualize_all_baseline_comparisons
from config import settings

def main():
    """Regenerate baseline comparisons for all models."""
    print("Loading all models...")
    all_models = load_all_models()
    
    print(f"\nFound {len(all_models)} models:")
    
    # Categorize models by type
    model_types = {
        'Linear Regression': [],
        'ElasticNet': [],
        'XGBoost': [],
        'LightGBM': [],
        'CatBoost': []
    }
    
    for model_name in all_models.keys():
        if model_name.startswith('LR_'):
            model_types['Linear Regression'].append(model_name)
        elif model_name.startswith('ElasticNet_'):
            model_types['ElasticNet'].append(model_name)
        elif model_name.startswith('XGBoost_'):
            model_types['XGBoost'].append(model_name)
        elif model_name.startswith('LightGBM_'):
            model_types['LightGBM'].append(model_name)
        elif model_name.startswith('CatBoost_'):
            model_types['CatBoost'].append(model_name)
    
    # Print summary
    for model_type, models in model_types.items():
        print(f"  {model_type}: {len(models)} models")
        for model in models[:3]:  # Show first 3 models of each type
            print(f"    - {model}")
        if len(models) > 3:
            print(f"    ... and {len(models) - 3} more")
    
    # Clear existing baseline comparison file to start fresh
    baseline_path = settings.METRICS_DIR / "baseline_comparison.csv"
    if baseline_path.exists():
        print(f"\nRemoving existing baseline comparison file: {baseline_path}")
        baseline_path.unlink()
    
    # Run baseline evaluation with all models
    print("\nRunning baseline evaluation for all models...")
    print("This will compare each model against random, mean, and median baselines.")
    
    try:
        baseline_comparisons, results_df = run_baseline_evaluation(
            all_models,
            include_mean=True,
            include_median=True
        )
        
        print(f"\nBaseline evaluation complete!")
        print(f"Total comparisons: {len(baseline_comparisons)}")
        print(f"Results saved to: {baseline_path}")
        
        # Show summary of results
        if not results_df.empty:
            print("\nTop 5 models by improvement over random baseline:")
            random_results = results_df[results_df['Baseline Type'] == 'Random'].head(5)
            for idx, row in random_results.iterrows():
                print(f"  {row['Model']}: {row['Improvement (%)']:.2f}% improvement")
        
        # Generate visualizations
        print("\nGenerating baseline comparison visualizations...")
        baseline_figures = visualize_all_baseline_comparisons(create_individual_plots=False)
        
        if baseline_figures:
            print("\nVisualization files created:")
            for name, path in baseline_figures.items():
                print(f"  - {name}: {path}")
        else:
            print("No visualizations were created.")
            
    except Exception as e:
        print(f"Error during baseline evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()