"""I/O utilities for saving and loading data."""

import pickle
import pandas as pd
import os
import sys
from pathlib import Path

# Import settings
from src.config import settings

def save_model(model, filename, directory):
    """Save a model to a pickle file."""
    path = Path(directory) / filename
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    return path

def load_model(filename, directory):
    """Load a model from a pickle file."""
    path = Path(directory) / filename
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading model from {path}: {e}")
        return None

def save_metrics(metrics_dict, filename, directory):
    """Save metrics to a CSV file."""
    path = Path(directory) / filename
    pd.DataFrame(metrics_dict).to_csv(path)
    return path

def ensure_dir(directory):
    """Ensure a directory exists."""
    os.makedirs(directory, exist_ok=True)
    return directory

def load_all_models():
    """
    Load all trained models from the models directory.
    
    Returns:
        dict: Dictionary containing all loaded models
    """
    all_models = {}
    
    # List of model files to check
    model_files = [
        "linear_regression_models.pkl",
        "linear_regression_models_unified.pkl",  # Also check unified version
        "elasticnet_models.pkl",
        "elasticnet_models_unified.pkl",  # Also check unified version
        "xgboost_models.pkl",
        "xgboost_models_unified.pkl",  # Also check unified version
        "lightgbm_models.pkl",
        "catboost_models.pkl"
    ]
    
    for model_file in model_files:
        try:
            # Try to load the model
            models = load_model(model_file, settings.MODEL_DIR)
            if models is not None and isinstance(models, dict) and models:
                print(f"Loaded {len(models)} models from {model_file}")
                all_models.update(models)
            else:
                print(f"No models found in {model_file} or file not found")
        except Exception as e:
            print(f"Error loading {model_file}: {e}")
    
    print(f"Loaded a total of {len(all_models)} models from all files")
    return all_models


def check_existing_study(model_name, model_file, min_trials=50):
    """
    Check if a complete Optuna study exists for a given model.
    
    Args:
        model_name (str): Name of the model variant (e.g., "XGB_Base_optuna")
        model_file (str): Model file to check (e.g., "xgboost_models.pkl")
        min_trials (int): Minimum number of trials to consider study complete
        
    Returns:
        tuple: (bool, dict or None) - (study_exists, study_info)
    """
    try:
        models = load_model(model_file, settings.MODEL_DIR)
        if not models or not isinstance(models, dict):
            return False, None
            
        if model_name not in models:
            return False, None
            
        model_data = models[model_name]
        
        # Check if study exists and has sufficient trials
        if ('study' in model_data and 
            model_data['study'] is not None and 
            hasattr(model_data['study'], 'trials') and
            len(model_data['study'].trials) >= min_trials):
            
            study_info = {
                'n_trials': len(model_data['study'].trials),
                'best_value': model_data['study'].best_value,
                'best_params': model_data['study'].best_params,
                'model_name': model_name
            }
            return True, study_info
            
        return False, None
        
    except Exception as e:
        print(f"Error checking study for {model_name}: {e}")
        return False, None


def check_existing_studies_for_algorithm(algorithm, datasets=None, min_trials=50):
    """
    Check existing studies for a specific algorithm across all dataset variants.
    
    Args:
        algorithm (str): Algorithm name ("xgboost", "lightgbm", "catboost")
        datasets (list): List of dataset names to check, or None for all
        min_trials (int): Minimum number of trials to consider study complete
        
    Returns:
        dict: Dictionary with model names as keys and study info as values
    """
    model_file_map = {
        'xgboost': 'xgboost_models.pkl',
        'lightgbm': 'lightgbm_models.pkl', 
        'catboost': 'catboost_models.pkl'
    }
    
    if algorithm not in model_file_map:
        raise ValueError(f"Unknown algorithm: {algorithm}")
        
    model_file = model_file_map[algorithm]
    existing_studies = {}
    
    # Default dataset variants if none specified
    if datasets is None or 'all' in datasets:
        dataset_variants = ['Base', 'Yeo', 'Base_Random', 'Yeo_Random']
    else:
        # Convert dataset names like 'LR_Base' to just 'Base'
        dataset_variants = [d.replace('LR_', '') for d in datasets if d != 'all']
    
    algorithm_prefix = {
        'xgboost': 'XGB',
        'lightgbm': 'LightGBM', 
        'catboost': 'CatBoost'
    }[algorithm]
    
    for variant in dataset_variants:
        for training_type in ['basic', 'optuna']:
            model_name = f"{algorithm_prefix}_{variant}_{training_type}"
            
            study_exists, study_info = check_existing_study(model_name, model_file, min_trials)
            if study_exists:
                existing_studies[model_name] = study_info
                
    return existing_studies


def prompt_study_override(existing_studies, algorithm_name="models"):
    """
    Prompt user for confirmation to override existing studies.
    
    Args:
        existing_studies (dict): Dictionary of existing study info
        algorithm_name (str): Name of algorithm for display
        
    Returns:
        bool: True if user wants to override, False otherwise
    """
    if not existing_studies:
        return True
        
    print(f"\n🔍 Found existing {algorithm_name} studies:")
    print("-" * 60)
    
    for model_name, study_info in existing_studies.items():
        print(f"  📊 {model_name}:")
        print(f"     - Trials: {study_info['n_trials']}")
        print(f"     - Best RMSE: {study_info['best_value']:.4f}")
        print(f"     - Best params: {len(study_info['best_params'])} parameters")
        print()
    
    print(f"⚠️  Re-running will overwrite {len(existing_studies)} existing studies.")
    print("💰 This may take significant computational time and resources.")
    
    while True:
        response = input("\n❓ Do you want to proceed and overwrite existing studies? [y/N]: ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no', '']:
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no.")


def report_existing_studies():
    """
    Generate a comprehensive report of all existing Optuna studies.
    
    Returns:
        dict: Summary of all existing studies by algorithm
    """
    algorithms = ['xgboost', 'lightgbm', 'catboost']
    all_studies = {}
    
    print("\n📊 OPTUNA STUDIES REPORT")
    print("=" * 80)
    
    total_studies = 0
    
    for algorithm in algorithms:
        existing_studies = check_existing_studies_for_algorithm(algorithm)
        all_studies[algorithm] = existing_studies
        
        if existing_studies:
            print(f"\n🔬 {algorithm.upper()} Studies ({len(existing_studies)} found):")
            print("-" * 50)
            
            for model_name, study_info in existing_studies.items():
                print(f"  ✅ {model_name}")
                print(f"     📈 Trials: {study_info['n_trials']}")
                print(f"     🎯 Best RMSE: {study_info['best_value']:.4f}")
                
            total_studies += len(existing_studies)
        else:
            print(f"\n🔬 {algorithm.upper()} Studies: None found")
    
    print(f"\n📋 SUMMARY: {total_studies} total Optuna studies found across all algorithms")
    print("=" * 80)
    
    return all_studies


def check_existing_elasticnet_models():
    """
    Check if ElasticNet models exist.
    
    Returns:
        dict: Dictionary with model info if they exist
    """
    existing_models = {}
    
    try:
        models = load_model("elasticnet_models.pkl", settings.MODEL_DIR)
        if models and isinstance(models, dict) and models:
            for model_name, model_data in models.items():
                if 'test_r2' in model_data and 'alpha' in model_data:
                    existing_models[model_name] = {
                        'test_r2': model_data['test_r2'],
                        'alpha': model_data['alpha'],
                        'l1_ratio': model_data.get('l1_ratio', 'N/A'),
                        'model_name': model_name
                    }
    except Exception as e:
        print(f"Error checking ElasticNet models: {e}")
    
    return existing_models


def check_existing_linear_models():
    """
    Check if Linear Regression models exist.
    
    Returns:
        dict: Dictionary with model info if they exist
    """
    existing_models = {}
    
    try:
        models = load_model("linear_regression_models.pkl", settings.MODEL_DIR)
        if models and isinstance(models, dict) and models:
            for model_name, model_data in models.items():
                if 'test_r2' in model_data:
                    existing_models[model_name] = {
                        'test_r2': model_data['test_r2'],
                        'model_name': model_name
                    }
    except Exception as e:
        print(f"Error checking Linear Regression models: {e}")
    
    return existing_models


def check_all_existing_models(datasets=None, min_trials=50):
    """
    Check for existing models across all algorithms.
    
    Args:
        datasets (list): List of dataset names to check, or None for all
        min_trials (int): Minimum number of trials for Optuna studies
        
    Returns:
        dict: Dictionary with algorithm names as keys and existing model info as values
    """
    all_existing = {}
    
    # Check tree-based models with Optuna studies
    tree_algorithms = ['xgboost', 'lightgbm', 'catboost']
    for algorithm in tree_algorithms:
        existing_studies = check_existing_studies_for_algorithm(algorithm, datasets, min_trials)
        if existing_studies:
            all_existing[algorithm] = existing_studies
    
    # Check ElasticNet models
    elasticnet_models = check_existing_elasticnet_models()
    if elasticnet_models:
        all_existing['elasticnet'] = elasticnet_models
    
    # Check Linear Regression models  
    linear_models = check_existing_linear_models()
    if linear_models:
        all_existing['linear_regression'] = linear_models
    
    return all_existing


def prompt_consolidated_retrain(all_existing_models):
    """
    Prompt user for consolidated confirmation to retrain all models.
    
    Args:
        all_existing_models (dict): Dictionary of all existing models by algorithm
        
    Returns:
        bool: True if user wants to retrain all, False otherwise
    """
    if not all_existing_models:
        return True
    
    print(f"\n🔍 Found existing trained models across multiple algorithms:")
    print("=" * 80)
    
    total_models = 0
    estimated_time = 0
    
    for algorithm, models in all_existing_models.items():
        algorithm_display = {
            'xgboost': 'XGBoost',
            'lightgbm': 'LightGBM', 
            'catboost': 'CatBoost',
            'elasticnet': 'ElasticNet',
            'linear_regression': 'Linear Regression'
        }.get(algorithm, algorithm)
        
        print(f"\n🔬 {algorithm_display}: {len(models)} models found")
        
        # Show sample models for tree algorithms with studies
        if algorithm in ['xgboost', 'lightgbm', 'catboost']:
            estimated_time += 45  # ~45 min per tree algorithm
            print("     Sample Optuna studies:")
            for i, (model_name, study_info) in enumerate(list(models.items())[:2]):
                print(f"       📊 {model_name}: {study_info['n_trials']} trials, RMSE {study_info['best_value']:.4f}")
            if len(models) > 2:
                print(f"       ... and {len(models) - 2} more")
        
        # Show ElasticNet models
        elif algorithm == 'elasticnet':
            estimated_time += 15  # ~15 min for ElasticNet
            print("     Sample models:")
            for i, (model_name, model_info) in enumerate(list(models.items())[:2]):
                print(f"       📈 {model_name}: R² {model_info['test_r2']:.4f}, α={model_info['alpha']:.3f}")
            if len(models) > 2:
                print(f"       ... and {len(models) - 2} more")
        
        # Show Linear Regression models
        elif algorithm == 'linear_regression':
            estimated_time += 5  # ~5 min for Linear Regression
            print("     Sample models:")
            for i, (model_name, model_info) in enumerate(list(models.items())[:2]):
                print(f"       📊 {model_name}: R² {model_info['test_r2']:.4f}")
            if len(models) > 2:
                print(f"       ... and {len(models) - 2} more")
        
        total_models += len(models)
    
    print("\n" + "=" * 80)
    print(f"⚠️  RETRAINING WILL OVERWRITE {total_models} EXISTING MODELS")
    print(f"💰 Estimated total time: ~{estimated_time} minutes ({estimated_time//60}h {estimated_time%60}m)")
    print(f"🔄 This includes expensive Optuna hyperparameter optimization")
    print("=" * 80)
    
    while True:
        response = input(f"\n❓ Retrain ALL algorithms and overwrite existing models? [y/N]: ").strip().lower()
        if response in ['y', 'yes']:
            print("\n🔄 Starting full pipeline retraining for all algorithms...")
            return True
        elif response in ['n', 'no', '']:
            print("\n⏭️  Skipping model retraining - using existing models")
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no.")