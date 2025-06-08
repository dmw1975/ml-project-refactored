#!/usr/bin/env python3
"""
Regenerate baseline comparisons ensuring CatBoost models are included.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.evaluation.metrics import load_all_models
from src.evaluation.baselines import run_baseline_evaluation
from src.config import settings


def main():
    """Regenerate baseline comparisons with all models including CatBoost."""
    print("Loading all models including CatBoost...")
    all_models = load_all_models()
    
    # Check which models we have
    model_types = {}
    for model_name in all_models.keys():
        if 'CatBoost' in model_name:
            model_types.setdefault('CatBoost', []).append(model_name)
        elif 'XGBoost' in model_name or 'XGB' in model_name:
            model_types.setdefault('XGBoost', []).append(model_name)
        elif 'LightGBM' in model_name:
            model_types.setdefault('LightGBM', []).append(model_name)
        elif 'ElasticNet' in model_name:
            model_types.setdefault('ElasticNet', []).append(model_name)
        elif 'LR_' in model_name:
            model_types.setdefault('LinearRegression', []).append(model_name)
    
    print("\nModels found by type:")
    for model_type, models in model_types.items():
        print(f"  {model_type}: {len(models)} models")
        if model_type == 'CatBoost':
            print(f"    CatBoost models: {', '.join(models[:3])}{'...' if len(models) > 3 else ''}")
    
    if 'CatBoost' not in model_types:
        print("\nWARNING: No CatBoost models found! Please train CatBoost models first.")
        return
    
    print("\nRunning baseline evaluation with all models...")
    baseline_comparison, baseline_summary = run_baseline_evaluation(
        all_models,
        include_mean=True,
        include_median=True
    )
    
    print("\nBaseline evaluation complete!")
    
    # Check if CatBoost is in the results
    baseline_csv_path = settings.METRICS_DIR / "baseline_comparison.csv"
    if baseline_csv_path.exists():
        df = pd.read_csv(baseline_csv_path)
        catboost_rows = df[df['Model'].str.contains('CatBoost', case=False, na=False)]
        print(f"\nCatBoost entries in baseline comparison: {len(catboost_rows)}")
        if len(catboost_rows) > 0:
            print("Sample CatBoost entries:")
            print(catboost_rows[['Model', 'RMSE', 'Baseline RMSE', 'Improvement (%)']].head())
        else:
            print("ERROR: CatBoost models were not included in baseline comparison!")
    
    print("\nBaseline comparison regeneration complete!")


if __name__ == "__main__":
    main()