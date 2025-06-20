#!/usr/bin/env python3
"""Run updated SHAP comparison visualizations with separate Base and Yeo plots."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from src.utils.io import load_all_models
from src.visualization.plots.shap_plots import create_separated_model_comparison_shap_plots
from src.config import settings


def run_updated_shap_comparisons():
    """Run the updated SHAP comparison visualizations."""
    
    print("RUNNING UPDATED SHAP COMPARISON VISUALIZATIONS")
    print("=" * 60)
    print("Creating separate plots for Base and Yeo datasets")
    print("=" * 60)
    
    # Load all models
    print("\nLoading all models...")
    all_models = load_all_models()
    print(f"Loaded {len(all_models)} models total")
    
    # Filter for tree-based models with optuna optimization
    tree_models = {}
    for name, data in all_models.items():
        model_type = data.get('model_type', '').lower()
        if any(tree_type in model_type for tree_type in ['xgboost', 'lightgbm', 'catboost', 'xgb', 'lgb']):
            if 'optuna' in name.lower() and 'Random' not in name:
                tree_models[name] = data
    
    print(f"\nFiltered to {len(tree_models)} tree-based Optuna models (excluding Random)")
    
    # Count models by dataset type
    base_count = sum(1 for name in tree_models if 'Base' in name and 'Yeo' not in name)
    yeo_count = sum(1 for name in tree_models if 'Yeo' in name)
    
    print(f"  Base models: {base_count}")
    print(f"  Yeo models: {yeo_count}")
    
    # Set output directory
    shap_dir = settings.VISUALIZATION_DIR / 'shap'
    shap_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the separated plots
    print(f"\nCreating SHAP comparison plots in {shap_dir}...")
    
    try:
        created_paths = create_separated_model_comparison_shap_plots(tree_models, shap_dir)
        
        print(f"\nSuccessfully created {len(created_paths)} SHAP comparison plots:")
        for path in created_paths:
            print(f"  - {path.name}")
            
    except Exception as e:
        print(f"\nError creating SHAP comparison plots: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nDone! Updated SHAP comparison visualizations complete.")
    
    # Also run the feature removal analysis with updated plots
    print("\n" + "=" * 60)
    print("RUNNING FEATURE REMOVAL SHAP COMPARISONS")
    print("=" * 60)
    
    try:
        from xgboost_feature_removal_proper import ProperXGBoostFeatureRemoval
        
        # Check if feature removal analysis has been run
        feature_removal_dir = Path('outputs/feature_removal')
        if not feature_removal_dir.exists():
            print("Feature removal analysis not found. Running analysis first...")
            analyzer = ProperXGBoostFeatureRemoval(
                excluded_feature='top_3_shareholder_percentage',
                n_trials=100
            )
            analyzer.run_analysis()
        else:
            print("Feature removal results found. Regenerating SHAP comparisons...")
            # Just regenerate the SHAP plots
            analyzer = ProperXGBoostFeatureRemoval(
                excluded_feature='top_3_shareholder_percentage',
                n_trials=100
            )
            
            # Load existing results
            all_models = {}
            for dataset in ['Base_Random', 'Yeo_Random']:
                for config in ['with_feature', 'without_feature']:
                    model_name = f"{dataset}_{config}"
                    model_path = feature_removal_dir / 'models' / f'{model_name}.pkl'
                    if model_path.exists():
                        import pickle
                        with open(model_path, 'rb') as f:
                            all_models[model_name] = pickle.load(f)
            
            # Create SHAP comparisons
            shap_dir = feature_removal_dir / 'visualization' / 'shap'
            shap_dir.mkdir(parents=True, exist_ok=True)
            analyzer._create_feature_removal_shap_comparison(all_models, shap_dir)
            
        print("\nFeature removal SHAP comparisons complete!")
        
    except Exception as e:
        print(f"\nError with feature removal SHAP comparisons: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_updated_shap_comparisons()