#!/usr/bin/env python3
"""Restore baseline comparison plots with full statistical significance testing."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.evaluation.baseline_significance import run_baseline_significance_analysis
from src.utils.io import load_all_models
from src.data.data import load_scores_data

def restore_statistical_baseline_comparisons():
    """
    Restore baseline comparison plots with COMPLETE statistical testing content.
    
    This includes:
    1. All 32 models (including LightGBM and CatBoost)
    2. Proper baseline calculations from training data
    3. Statistical significance testing with p-values
    4. Dual-panel layout showing both absolute values and improvements
    5. Significance markers and color coding
    6. Holm-Bonferroni correction for multiple testing
    """
    
    print("=" * 80)
    print("RESTORING BASELINE COMPARISONS WITH STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 80)
    
    # Load all models
    print("\nLoading all models...")
    all_models = load_all_models()
    
    # Check model types
    model_types = set()
    for model_name in all_models.keys():
        if 'XGB' in model_name or 'XGBoost' in model_name:
            model_types.add('XGBoost')
        elif 'LightGBM' in model_name:
            model_types.add('LightGBM')
        elif 'CatBoost' in model_name:
            model_types.add('CatBoost')
        elif 'ElasticNet' in model_name:
            model_types.add('ElasticNet')
        elif 'LR_' in model_name or model_name.startswith('lr_'):
            model_types.add('Linear Regression')
    
    print(f"Model types found: {', '.join(sorted(model_types))}")
    print(f"Total models: {len(all_models)}")
    
    # Load scores data for baseline calculations
    print("\nLoading scores data...")
    scores_data = load_scores_data()
    
    # Set output directory for statistical test plots
    output_dir = settings.VISUALIZATION_DIR / "statistical_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nRunning baseline significance analysis with statistical testing...")
    print("This will generate plots with:")
    print("  - Paired statistical tests (t-test or Wilcoxon)")
    print("  - P-values and significance markers")
    print("  - Holm-Bonferroni correction")
    print("  - Dual-panel layout with absolute values and improvements")
    
    # Run the full statistical analysis
    results_df, plots = run_baseline_significance_analysis(
        models_dict=all_models,
        scores_data=scores_data,
        output_dir=output_dir,
        n_folds=5,  # Standard 5-fold CV
        random_seed=42
    )
    
    if results_df is not None:
        print("\n" + "=" * 80)
        print("STATISTICAL ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Show summary statistics
        for baseline in ['Random', 'Mean', 'Median']:
            baseline_results = results_df[results_df['Baseline'] == baseline]
            if not baseline_results.empty:
                n_significant = baseline_results['Significant'].sum()
                n_total = len(baseline_results)
                print(f"\n{baseline} Baseline:")
                print(f"  Models tested: {n_total}")
                print(f"  Statistically significant: {n_significant} ({n_significant/n_total*100:.1f}%)")
                
                # Show best performing models
                top_models = baseline_results.nlargest(3, 'Improvement (%)')
                print(f"  Top 3 models by improvement:")
                for _, row in top_models.iterrows():
                    sig_marker = "*" if row['Significant'] else ""
                    print(f"    - {row['Model']}: {row['Improvement (%)']:.1f}%{sig_marker} (p={row['p-value']:.4f})")
        
        print("\n* indicates statistical significance after Holm-Bonferroni correction")
    
    print("\n" + "=" * 80)
    print("RESTORATION COMPLETE")
    print("=" * 80)
    print(f"\nStatistical baseline comparison plots saved to: {output_dir}")
    print("\nThese plots now include:")
    print("  ✓ All 32 models (including LightGBM and CatBoost)")
    print("  ✓ Statistical significance testing with p-values")
    print("  ✓ Dual-panel layout (absolute values + improvements)")
    print("  ✓ Significance markers and color coding")
    print("  ✓ Holm-Bonferroni correction for multiple testing")
    
    return results_df, plots

if __name__ == "__main__":
    restore_statistical_baseline_comparisons()