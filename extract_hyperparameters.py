#!/usr/bin/env python3
"""
Extract and Compare Hyperparameters Across Scripts
=================================================

This script extracts hyperparameter definitions from all model scripts
and creates a comparison table.
"""

import re
from pathlib import Path
import pandas as pd
import json

def extract_optuna_params(file_path):
    """Extract Optuna hyperparameter definitions from a Python file."""
    params = {}
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find suggest_ patterns
        patterns = {
            'suggest_int': r"trial\.suggest_int\('(\w+)',\s*(\d+),\s*(\d+)\)",
            'suggest_float': r"trial\.suggest_float\('(\w+)',\s*([\d.e-]+),\s*([\d.e-]+)(?:,\s*log=(True|False))?\)",
        }
        
        for method, pattern in patterns.items():
            matches = re.findall(pattern, content)
            for match in matches:
                param_name = match[0]
                if method == 'suggest_int':
                    params[param_name] = {
                        'type': 'int',
                        'min': int(match[1]),
                        'max': int(match[2]),
                        'log': False
                    }
                else:  # suggest_float
                    params[param_name] = {
                        'type': 'float',
                        'min': float(match[1]),
                        'max': float(match[2]),
                        'log': match[3] == 'True' if len(match) > 3 else False
                    }
        
        # Also extract basic/fixed parameters
        basic_params = {}
        
        # Look for dictionary definitions with basic parameters
        basic_pattern = r"['\"](\w+)['\"]:\s*([\d.]+)"
        basic_matches = re.findall(basic_pattern, content)
        
        for param, value in basic_matches:
            if param in ['max_depth', 'learning_rate', 'n_estimators', 'subsample', 
                        'colsample_bytree', 'num_leaves', 'iterations', 'depth',
                        'alpha', 'l1_ratio']:
                try:
                    basic_params[param] = float(value) if '.' in value else int(value)
                except:
                    pass
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return params, basic_params

def main():
    """Extract and compare hyperparameters from all scripts."""
    
    # Define script locations
    scripts = {
        'XGBoost Enhanced': 'scripts/archive/enhanced_xgboost_categorical.py',
        'XGBoost Feature Removal Basic': 'xgboost_feature_removal_basic.py',
        'XGBoost Feature Removal Enhanced': 'xgboost_feature_removal_enhanced.py',
        'LightGBM Enhanced': 'scripts/archive/enhanced_lightgbm_categorical.py',
        'CatBoost Enhanced': 'scripts/archive/enhanced_catboost_categorical.py',
        'ElasticNet Enhanced': 'scripts/archive/enhanced_elasticnet_optuna.py'
    }
    
    # Extract parameters
    all_params = {}
    for name, path in scripts.items():
        file_path = Path(path)
        if file_path.exists():
            optuna_params, basic_params = extract_optuna_params(file_path)
            all_params[name] = {
                'optuna': optuna_params,
                'basic': basic_params,
                'file': str(file_path)
            }
    
    # Create comparison tables
    print("="*80)
    print("HYPERPARAMETER CONFIGURATION COMPARISON")
    print("="*80)
    
    # XGBoost comparison
    print("\n## XGBoost Parameter Comparison\n")
    xgb_scripts = [k for k in all_params.keys() if 'XGBoost' in k]
    
    if xgb_scripts:
        # Get all unique parameters
        all_xgb_params = set()
        for script in xgb_scripts:
            all_xgb_params.update(all_params[script]['optuna'].keys())
            all_xgb_params.update(all_params[script]['basic'].keys())
        
        # Create comparison table
        comparison_data = []
        for param in sorted(all_xgb_params):
            row = {'Parameter': param}
            for script in xgb_scripts:
                if param in all_params[script]['optuna']:
                    p = all_params[script]['optuna'][param]
                    row[script] = f"[{p['min']}, {p['max']}]" + (" (log)" if p['log'] else "")
                elif param in all_params[script]['basic']:
                    row[script] = str(all_params[script]['basic'][param])
                else:
                    row[script] = "Not specified"
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
    
    # Save detailed results
    output_file = Path("hyperparameter_extraction_results.json")
    with open(output_file, 'w') as f:
        json.dump(all_params, f, indent=2, default=str)
    
    print(f"\n\nDetailed results saved to: {output_file}")
    
    # Configuration consistency check
    print("\n## Configuration Consistency Check\n")
    
    # Check for missing critical parameters in feature removal scripts
    critical_xgb_params = ['max_depth', 'learning_rate', 'n_estimators', 'subsample', 'colsample_bytree']
    
    for script in xgb_scripts:
        if 'Feature Removal' in script:
            missing = []
            for param in critical_xgb_params:
                if param not in all_params[script]['basic'] and param not in all_params[script]['optuna']:
                    missing.append(param)
            
            if missing:
                print(f"⚠️  {script}: Missing parameters: {', '.join(missing)}")
            else:
                print(f"✓ {script}: All critical parameters specified")
    
    # Check n_trials configuration
    print("\n## Optuna Configuration\n")
    
    # Read settings.py for n_trials
    settings_path = Path("src/config/settings.py")
    if settings_path.exists():
        with open(settings_path, 'r') as f:
            content = f.read()
        
        n_trials_matches = re.findall(r'"n_trials":\s*(\d+)', content)
        if n_trials_matches:
            print(f"n_trials defined in settings.py: {n_trials_matches[0]}")
    
    # Check which scripts use Optuna
    for name, data in all_params.items():
        if data['optuna']:
            print(f"✓ {name}: Uses Optuna optimization ({len(data['optuna'])} parameters)")
        else:
            print(f"✗ {name}: No Optuna optimization")

if __name__ == "__main__":
    main()