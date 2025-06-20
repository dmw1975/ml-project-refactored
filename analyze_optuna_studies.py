#!/usr/bin/env python3
"""
Analyze Optuna Studies Across All Models
=======================================

This script analyzes all saved Optuna studies to provide insights into:
- Number of trials completed
- Best parameters found
- Convergence patterns
- Parameter importance
"""

import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import optuna
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

def load_model_files(model_dir: Path) -> Dict[str, Any]:
    """Load all model pickle files."""
    models = {}
    
    model_files = {
        'xgboost': 'xgboost_models.pkl',
        'lightgbm': 'lightgbm_models.pkl',
        'catboost': 'catboost_models.pkl',
        'elasticnet': 'elasticnet_models.pkl'
    }
    
    for model_type, filename in model_files.items():
        filepath = model_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    models[model_type] = pickle.load(f)
                print(f"✓ Loaded {model_type} models from {filename}")
            except Exception as e:
                print(f"✗ Failed to load {model_type}: {e}")
    
    return models

def extract_optuna_studies(models: Dict) -> Dict[str, optuna.Study]:
    """Extract Optuna study objects from model dictionaries."""
    studies = {}
    
    for model_type, model_dict in models.items():
        for model_name, model_data in model_dict.items():
            if 'optuna' in model_name and isinstance(model_data, dict):
                if 'study' in model_data:
                    studies[model_name] = model_data['study']
                    print(f"  Found study: {model_name}")
    
    return studies

def analyze_study(study_name: str, study: optuna.Study) -> Dict:
    """Analyze a single Optuna study."""
    analysis = {
        'study_name': study_name,
        'n_trials': len(study.trials),
        'best_value': study.best_value,
        'best_params': study.best_params,
        'optimization_history': [trial.value for trial in study.trials if trial.value is not None]
    }
    
    # Extract CV information from best trial
    if study.best_trial:
        user_attrs = study.best_trial.user_attrs
        analysis['cv_mean'] = user_attrs.get('cv_mean', study.best_value)
        analysis['cv_std'] = user_attrs.get('cv_std', 0)
        analysis['cv_scores'] = user_attrs.get('cv_scores', [])
    
    # Calculate convergence metrics
    history = analysis['optimization_history']
    if len(history) > 10:
        # Check if best value was found in first half of trials
        best_trial_idx = history.index(min(history))
        analysis['early_convergence'] = best_trial_idx < len(history) / 2
        
        # Calculate improvement over last 10 trials
        if len(history) >= 20:
            early_best = min(history[:10])
            late_best = min(history[-10:])
            analysis['late_improvement'] = (early_best - late_best) / early_best * 100
        else:
            analysis['late_improvement'] = 0
    
    return analysis

def create_convergence_plot(studies: Dict[str, optuna.Study], output_dir: Path):
    """Create convergence plots for all studies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by model type
    model_types = {}
    for study_name in studies.keys():
        model_type = study_name.split('_')[0]
        if model_type not in model_types:
            model_types[model_type] = []
        model_types[model_type].append(study_name)
    
    # Create plot for each model type
    for model_type, study_names in model_types.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, study_name in enumerate(study_names[:4]):  # Max 4 per plot
            if idx < len(axes):
                study = studies[study_name]
                history = [trial.value for trial in study.trials if trial.value is not None]
                
                ax = axes[idx]
                ax.plot(history, 'b-', alpha=0.5, label='Trial values')
                
                # Add cumulative minimum line
                cum_min = np.minimum.accumulate(history)
                ax.plot(cum_min, 'r-', linewidth=2, label='Best so far')
                
                ax.set_xlabel('Trial')
                ax.set_ylabel('Objective Value (RMSE)')
                ax.set_title(study_name.replace('_', ' '))
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_type} Optimization Convergence', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f'{model_type}_convergence.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_parameter_comparison_table(studies: Dict[str, optuna.Study]) -> pd.DataFrame:
    """Create a comparison table of best parameters across studies."""
    rows = []
    
    for study_name, study in studies.items():
        if study.best_trial:
            row = {'study': study_name}
            row.update(study.best_params)
            row['best_value'] = study.best_value
            row['n_trials'] = len(study.trials)
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns
    cols = ['study', 'best_value', 'n_trials'] + [col for col in df.columns if col not in ['study', 'best_value', 'n_trials']]
    df = df[cols]
    
    return df

def main():
    """Main analysis function."""
    print("="*60)
    print("OPTUNA STUDY ANALYSIS")
    print("="*60)
    
    # Load models
    model_dir = Path("outputs/models")
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        return
    
    print("\n1. Loading model files...")
    models = load_model_files(model_dir)
    
    if not models:
        print("No models found!")
        return
    
    # Extract Optuna studies
    print("\n2. Extracting Optuna studies...")
    studies = extract_optuna_studies(models)
    
    if not studies:
        print("No Optuna studies found!")
        return
    
    print(f"\nFound {len(studies)} Optuna studies")
    
    # Analyze each study
    print("\n3. Analyzing studies...")
    analyses = {}
    for study_name, study in studies.items():
        analyses[study_name] = analyze_study(study_name, study)
    
    # Print summary
    print("\n" + "="*60)
    print("STUDY SUMMARY")
    print("="*60)
    
    for study_name, analysis in analyses.items():
        print(f"\n{study_name}:")
        print(f"  Trials: {analysis['n_trials']}")
        print(f"  Best RMSE: {analysis['best_value']:.4f}")
        print(f"  CV Mean ± Std: {analysis['cv_mean']:.4f} ± {analysis['cv_std']:.4f}")
        print(f"  Early convergence: {'Yes' if analysis.get('early_convergence', False) else 'No'}")
        if 'late_improvement' in analysis:
            print(f"  Late improvement: {analysis['late_improvement']:.2f}%")
        
        # Print best parameters
        print("  Best parameters:")
        for param, value in analysis['best_params'].items():
            if isinstance(value, float):
                print(f"    {param}: {value:.4f}")
            else:
                print(f"    {param}: {value}")
    
    # Create visualizations
    print("\n4. Creating visualizations...")
    viz_dir = Path("outputs/optuna_analysis")
    create_convergence_plot(studies, viz_dir)
    print(f"✓ Convergence plots saved to {viz_dir}")
    
    # Create parameter comparison table
    print("\n5. Creating parameter comparison table...")
    param_df = create_parameter_comparison_table(studies)
    param_df.to_csv(viz_dir / "parameter_comparison.csv", index=False)
    print(f"✓ Parameter comparison saved to {viz_dir / 'parameter_comparison.csv'}")
    
    # Check if more trials might help
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    for study_name, analysis in analyses.items():
        if analysis['n_trials'] >= 50:
            if 'late_improvement' in analysis and analysis['late_improvement'] > 1:
                print(f"\n⚠️  {study_name}: Still improving ({analysis['late_improvement']:.1f}% in last 10 trials)")
                print("   Consider running more trials (current: {})".format(analysis['n_trials']))
            else:
                print(f"\n✓ {study_name}: Appears converged (current: {analysis['n_trials']} trials)")
    
    print("\n" + "="*60)
    print("Analysis complete!")

if __name__ == "__main__":
    main()