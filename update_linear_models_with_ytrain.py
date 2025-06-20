"""
Script to retrain Linear Regression and ElasticNet models to include y_train storage.

This ensures all models have consistent data structure for complete statistical testing.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

print("="*80)
print("UPDATING LINEAR MODELS TO INCLUDE Y_TRAIN STORAGE")
print("="*80)

print("\n1. Training Linear Regression models with y_train storage...")
from src.models.linear_regression import train_all_models as train_lr
lr_results = train_lr()
print(f"✓ Trained {len(lr_results)} Linear Regression models")

# Verify y_train is stored
for model_name, model_data in lr_results.items():
    if 'y_train' in model_data:
        print(f"  ✓ {model_name} has y_train")
    else:
        print(f"  ✗ {model_name} MISSING y_train")

print("\n2. Training ElasticNet models with y_train storage...")
from src.models.elastic_net import train_elasticnet_models
elasticnet_results = train_elasticnet_models(use_optuna=True, n_trials=100)
print(f"✓ Trained {len(elasticnet_results)} ElasticNet models")

# Verify y_train is stored
for model_name, model_data in elasticnet_results.items():
    if 'y_train' in model_data:
        print(f"  ✓ {model_name} has y_train")
    else:
        print(f"  ✗ {model_name} MISSING y_train")

print("\n3. Verifying all model types now have y_train...")
from src.utils import io

all_models = io.load_all_models()
models_without_ytrain = []

for model_name, model_data in all_models.items():
    if isinstance(model_data, dict) and 'y_train' not in model_data:
        models_without_ytrain.append(model_name)

if models_without_ytrain:
    print(f"\n⚠️  WARNING: {len(models_without_ytrain)} models still missing y_train:")
    for model in models_without_ytrain[:5]:
        print(f"  - {model}")
    if len(models_without_ytrain) > 5:
        print(f"  ... and {len(models_without_ytrain) - 5} more")
else:
    print("\n✓ SUCCESS: All models now have y_train for complete statistical testing!")

print("\n4. Running improved statistical tests with complete data...")
from src.evaluation.baseline_significance import run_baseline_significance_analysis

results, plots = run_baseline_significance_analysis(all_models, use_improved_method=True)

if results is not None and not results.empty:
    # Count models per baseline
    for baseline in ['Random', 'Mean', 'Median']:
        baseline_df = results[results['Baseline'] == baseline]
        n_models = len(baseline_df)
        n_sig = baseline_df['Significant'].sum()
        print(f"\nvs {baseline}: {n_models} models tested, {n_sig} significant")

print("\n✓ Model update complete! All models now have consistent data structure.")