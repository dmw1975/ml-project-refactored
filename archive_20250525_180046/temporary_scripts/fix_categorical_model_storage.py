#!/usr/bin/env python3
"""
Fix categorical model storage to use the old format.
This script:
1. Updates the saving code to remove individual file saves
2. Consolidates existing categorical models into standard pickle files
3. Removes the separate categorical pickle files
"""

import pickle
import shutil
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import settings


def fix_model_saving_code():
    """Update the model saving code to use only the combined format."""
    print("Updating model saving code...")
    
    # 1. Fix XGBoost categorical
    xgboost_file = Path("models/xgboost_categorical.py")
    with open(xgboost_file, 'r') as f:
        content = f.read()
    
    # Comment out individual saves (lines 96-103)
    lines = content.split('\n')
    new_lines = []
    for i, line in enumerate(lines):
        if i >= 95 and i <= 102:  # Lines 96-103 (0-indexed)
            new_lines.append('                # ' + line if line.strip() else line)
        else:
            new_lines.append(line)
    
    with open(xgboost_file, 'w') as f:
        f.write('\n'.join(new_lines))
    print("✓ Updated xgboost_categorical.py")
    
    # 2. Fix LightGBM categorical
    lightgbm_file = Path("models/lightgbm_categorical.py")
    with open(lightgbm_file, 'r') as f:
        content = f.read()
    
    # Comment out individual saves (lines 96-103)
    lines = content.split('\n')
    new_lines = []
    for i, line in enumerate(lines):
        if i >= 95 and i <= 102:  # Lines 96-103 (0-indexed)
            new_lines.append('                # ' + line if line.strip() else line)
        else:
            new_lines.append(line)
    
    with open(lightgbm_file, 'w') as f:
        f.write('\n'.join(new_lines))
    print("✓ Updated lightgbm_categorical.py")
    
    # 3. Fix CatBoost categorical - change output filename
    catboost_file = Path("models/catboost_categorical.py")
    with open(catboost_file, 'r') as f:
        content = f.read()
    
    # Change line 375 from catboost_categorical_models.pkl to catboost_models.pkl
    content = content.replace(
        'output_file = "catboost_categorical_models.pkl"',
        'output_file = "catboost_models.pkl"'
    )
    
    with open(catboost_file, 'w') as f:
        f.write(content)
    print("✓ Updated catboost_categorical.py")


def consolidate_existing_models():
    """Consolidate existing categorical models into standard pickle files."""
    print("\nConsolidating existing models...")
    
    model_dir = settings.MODEL_DIR
    
    # 1. Consolidate XGBoost models
    xgboost_models = {}
    xgboost_files = [
        'xgboost_base_categorical.pkl',
        'xgboost_base_categorical_basic.pkl', 
        'xgboost_base_categorical_optuna.pkl',
        'xgboost_base_random_categorical.pkl',
        'xgboost_base_random_categorical_basic.pkl',
        'xgboost_base_random_categorical_optuna.pkl',
        'xgboost_yeo_categorical.pkl',
        'xgboost_yeo_categorical_basic.pkl',
        'xgboost_yeo_categorical_optuna.pkl',
        'xgboost_yeo_random_categorical.pkl',
        'xgboost_yeo_random_categorical_basic.pkl',
        'xgboost_yeo_random_categorical_optuna.pkl'
    ]
    
    for filename in xgboost_files:
        filepath = model_dir / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                # Extract model name from the data or filename
                if isinstance(model_data, dict) and 'model_name' in model_data:
                    model_name = model_data['model_name']
                else:
                    # Construct model name from filename
                    model_name = filename.replace('.pkl', '').replace('_', ' ').title()
                    model_name = model_name.replace('Xgboost', 'XGBoost')
                xgboost_models[model_name] = model_data
    
    if xgboost_models:
        with open(model_dir / 'xgboost_models.pkl', 'wb') as f:
            pickle.dump(xgboost_models, f)
        print(f"✓ Consolidated {len(xgboost_models)} XGBoost models")
    
    # 2. Consolidate LightGBM models
    lightgbm_models = {}
    lightgbm_files = [
        'lightgbm_base_categorical.pkl',
        'lightgbm_base_categorical_basic.pkl',
        'lightgbm_base_categorical_optuna.pkl',
        'lightgbm_base_random_categorical.pkl',
        'lightgbm_base_random_categorical_basic.pkl',
        'lightgbm_base_random_categorical_optuna.pkl',
        'lightgbm_yeo_categorical.pkl',
        'lightgbm_yeo_categorical_basic.pkl',
        'lightgbm_yeo_categorical_optuna.pkl',
        'lightgbm_yeo_random_categorical.pkl',
        'lightgbm_yeo_random_categorical_basic.pkl',
        'lightgbm_yeo_random_categorical_optuna.pkl'
    ]
    
    for filename in lightgbm_files:
        filepath = model_dir / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                # Extract model name from the data or filename
                if isinstance(model_data, dict) and 'model_name' in model_data:
                    model_name = model_data['model_name']
                else:
                    # Construct model name from filename
                    model_name = filename.replace('.pkl', '').replace('_', ' ').title()
                    model_name = model_name.replace('Lightgbm', 'LightGBM')
                lightgbm_models[model_name] = model_data
    
    if lightgbm_models:
        with open(model_dir / 'lightgbm_models.pkl', 'wb') as f:
            pickle.dump(lightgbm_models, f)
        print(f"✓ Consolidated {len(lightgbm_models)} LightGBM models")
    
    # 3. Rename CatBoost file if needed
    catboost_categorical = model_dir / 'catboost_categorical_models.pkl'
    catboost_standard = model_dir / 'catboost_models.pkl'
    
    if catboost_categorical.exists() and not catboost_standard.exists():
        shutil.move(catboost_categorical, catboost_standard)
        print("✓ Renamed catboost_categorical_models.pkl to catboost_models.pkl")


def remove_individual_categorical_files():
    """Remove the individual categorical pickle files."""
    print("\nRemoving individual categorical pickle files...")
    
    model_dir = settings.MODEL_DIR
    files_to_remove = []
    
    # List all categorical files
    for pattern in ['*_categorical*.pkl', '*categorical_*.pkl']:
        files_to_remove.extend(model_dir.glob(pattern))
    
    # Exclude the main model files we want to keep
    keep_files = ['catboost_models.pkl', 'lightgbm_models.pkl', 'xgboost_models.pkl']
    files_to_remove = [f for f in files_to_remove if f.name not in keep_files]
    
    # Remove files
    removed_count = 0
    for filepath in files_to_remove:
        try:
            filepath.unlink()
            removed_count += 1
            print(f"  ✓ Removed {filepath.name}")
        except Exception as e:
            print(f"  ✗ Error removing {filepath.name}: {e}")
    
    print(f"\nRemoved {removed_count} individual categorical files")


def update_metrics_aggregation():
    """Update the metrics aggregation to include categorical models."""
    print("\nUpdating metrics aggregation...")
    
    # Create a new script to regenerate all_models_comparison.csv
    script_content = '''#!/usr/bin/env python3
"""Regenerate all_models_comparison.csv with all model metrics."""

import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import settings

def aggregate_all_metrics():
    """Aggregate metrics from all individual CSV files."""
    metrics_dir = settings.METRICS_DIR
    
    # List of metric files to aggregate
    metric_files = [
        'linear_regression_metrics.csv',
        'elasticnet_metrics.csv',
        'xgboost_metrics.csv',
        'lightgbm_metrics.csv',
        'catboost_categorical_metrics.csv'
    ]
    
    all_metrics = []
    
    for filename in metric_files:
        filepath = metrics_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            all_metrics.append(df)
            print(f"✓ Loaded {len(df)} entries from {filename}")
        else:
            print(f"✗ File not found: {filename}")
    
    # Combine all metrics
    if all_metrics:
        combined_df = pd.concat(all_metrics, ignore_index=True)
        
        # Save to all_models_comparison.csv
        output_file = metrics_dir / 'all_models_comparison.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"\\n✓ Saved {len(combined_df)} total entries to {output_file}")
        
        # Show summary
        print("\\nModel counts by type:")
        if 'model_type' in combined_df.columns:
            print(combined_df['model_type'].value_counts())
    else:
        print("No metrics found to aggregate!")

if __name__ == "__main__":
    aggregate_all_metrics()
'''
    
    script_path = Path('regenerate_metrics_comparison.py')
    with open(script_path, 'w') as f:
        f.write(script_content)
    script_path.chmod(0o755)
    print("✓ Created regenerate_metrics_comparison.py")
    
    # Run the script
    import subprocess
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


def main():
    """Run all fixes."""
    print("Fixing categorical model storage...\n")
    
    # 1. Fix the saving code
    fix_model_saving_code()
    
    # 2. Consolidate existing models
    consolidate_existing_models()
    
    # 3. Remove individual files
    remove_individual_categorical_files()
    
    # 4. Update metrics aggregation
    update_metrics_aggregation()
    
    print("\n✅ All fixes completed!")
    print("\nNext steps:")
    print("1. Run the pipeline again to ensure models save correctly")
    print("2. Verify that visualizations now show all models")


if __name__ == "__main__":
    main()