"""
Verify which models have y_train storage and run statistical tests.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from src.utils import io
from src.evaluation.baseline_significance import run_baseline_significance_analysis

print("="*80)
print("VERIFYING Y_TRAIN STORAGE IN ALL MODELS")
print("="*80)

# Load all models
all_models = io.load_all_models()
print(f"\nTotal models loaded: {len(all_models)}")

# Check y_train availability by model type
model_types = {
    'Linear Regression': [],
    'ElasticNet': [],
    'XGBoost': [],
    'LightGBM': [],
    'CatBoost': []
}

models_with_ytrain = []
models_without_ytrain = []

for model_name, model_data in all_models.items():
    if not isinstance(model_data, dict):
        continue
        
    has_ytrain = 'y_train' in model_data
    
    if has_ytrain:
        models_with_ytrain.append(model_name)
    else:
        models_without_ytrain.append(model_name)
    
    # Categorize by type
    if 'LR_' in model_name or model_name.startswith('lr_'):
        model_types['Linear Regression'].append((model_name, has_ytrain))
    elif 'ElasticNet' in model_name:
        model_types['ElasticNet'].append((model_name, has_ytrain))
    elif 'XGBoost' in model_name:
        model_types['XGBoost'].append((model_name, has_ytrain))
    elif 'LightGBM' in model_name:
        model_types['LightGBM'].append((model_name, has_ytrain))
    elif 'CatBoost' in model_name:
        model_types['CatBoost'].append((model_name, has_ytrain))

# Print summary by model type
print("\nY_TRAIN STORAGE BY MODEL TYPE:")
print("-" * 50)
for model_type, models in model_types.items():
    if models:
        with_ytrain = sum(1 for _, has_it in models if has_it)
        total = len(models)
        print(f"{model_type}: {with_ytrain}/{total} models have y_train")
        
        # Show which ones are missing
        missing = [name for name, has_it in models if not has_it]
        if missing:
            for name in missing[:3]:
                print(f"  ✗ {name}")
            if len(missing) > 3:
                print(f"  ... and {len(missing) - 3} more")

print("\nOVERALL SUMMARY:")
print(f"✓ Models with y_train: {len(models_with_ytrain)}")
print(f"✗ Models without y_train: {len(models_without_ytrain)}")

if models_without_ytrain:
    print("\nMODELS MISSING Y_TRAIN:")
    for model in models_without_ytrain[:10]:
        print(f"  - {model}")
    if len(models_without_ytrain) > 10:
        print(f"  ... and {len(models_without_ytrain) - 10} more")

# Run statistical tests with current models
print("\n" + "="*80)
print("RUNNING STATISTICAL TESTS WITH CURRENT MODELS")
print("="*80)

results, plots = run_baseline_significance_analysis(all_models, use_improved_method=True)

if results is not None and not results.empty:
    # Count models per baseline
    print("\nSTATISTICAL TEST RESULTS:")
    for baseline in ['Random', 'Mean', 'Median']:
        baseline_df = results[results['Baseline'] == baseline]
        n_models = len(baseline_df)
        n_sig = baseline_df['Significant'].sum()
        print(f"\nvs {baseline}: {n_models} models tested, {n_sig} significant ({n_sig/n_models*100:.1f}%)")
        
        # Show which model types are missing
        tested_models = set(baseline_df['Model'].values)
        missing_lr = [m for m in models_without_ytrain if 'LR_' in m and m not in tested_models]
        missing_en = [m for m in models_without_ytrain if 'ElasticNet' in m and m not in tested_models]
        
        if baseline in ['Mean', 'Median'] and (missing_lr or missing_en):
            print(f"  Missing due to no y_train:")
            if missing_lr:
                print(f"    - {len(missing_lr)} Linear Regression models")
            if missing_en:
                print(f"    - {len(missing_en)} ElasticNet models")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)

if models_without_ytrain:
    print("Some models are missing y_train. To fix this:")
    print("1. The code has been updated to store y_train")
    print("2. Retrain the affected models:")
    
    if any('LR_' in m for m in models_without_ytrain):
        print("   python -m src.models.linear_regression")
    if any('ElasticNet' in m for m in models_without_ytrain):
        print("   python -m src.models.elastic_net")
    
    print("\n3. Then re-run statistical tests for complete analysis")
else:
    print("✓ All models have y_train storage!")
    print("✓ Statistical tests are complete for all models")