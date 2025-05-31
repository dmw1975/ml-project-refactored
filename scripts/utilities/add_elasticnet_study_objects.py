"""
Add Optuna study objects to ElasticNet models.

This script loads the ElasticNet model data and CV results, then creates
mock Optuna study objects from the grid search results so that contour
plots can be generated for ElasticNet models too.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings
from utils import io

try:
    import optuna
    from optuna.trial import TrialState
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Please install optuna.")
    sys.exit(1)

def convert_grid_search_to_study(cv_results_df, model_name, direction='minimize'):
    """
    Convert grid search CV results to an Optuna study object.
    
    Args:
        cv_results_df: DataFrame with grid search results
        model_name: Name of the model
        direction: Study direction ('minimize' or 'maximize')
        
    Returns:
        Optuna study object
    """
    # Create new study
    study_name = f"{model_name}_grid_search"
    storage_name = f"sqlite:///{study_name}.db"
    
    if os.path.exists(f"{study_name}.db"):
        os.remove(f"{study_name}.db")
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction=direction,
        load_if_exists=False
    )
    
    # Add trials from grid search results
    for _, row in cv_results_df.iterrows():
        # Create new trial
        trial = optuna.trial.create_trial(
            params={
                'alpha': float(row['alpha']),
                'l1_ratio': float(row['l1_ratio'])
            },
            distributions={
                'alpha': optuna.distributions.LogUniformDistribution(1e-5, 1.0),
                'l1_ratio': optuna.distributions.UniformDistribution(0.0, 1.0)
            },
            value=float(row['mean_rmse']),  # The objective value (using RMSE as the target)
            state=TrialState.COMPLETE
        )
        
        # Add trial to study
        study.add_trial(trial)
    
    return study

def add_elasticnet_study_objects():
    """Add Optuna study objects to ElasticNet models."""
    print("Adding Optuna study objects to ElasticNet models...")
    
    # First, load CV results
    try:
        cv_results = io.load_model("elasticnet_params.pkl", settings.MODEL_DIR)
        print(f"Loaded ElasticNet CV results: {len(cv_results)} datasets")
    except Exception as e:
        print(f"Error loading ElasticNet cross-validation results: {e}")
        return False
    
    # Then, load model data
    try:
        elasticnet_models = io.load_model("elasticnet_models.pkl", settings.MODEL_DIR)
        print(f"Loaded ElasticNet models: {len(elasticnet_models)} models")
    except Exception as e:
        print(f"Error loading ElasticNet models: {e}")
        return False
    
    # Create a dictionary mapping dataset names to CV results
    dataset_to_cv_results = {}
    for result in cv_results:
        dataset = result['dataset']
        dataset_to_cv_results[dataset] = result
    
    # Add study objects to model data
    models_updated = 0
    for model_name, model_data in elasticnet_models.items():
        # Extract dataset name from model name (e.g., "ElasticNet_LR_Base" -> "LR_Base")
        model_parts = model_name.split('_', 1)  # Split at first underscore
        if len(model_parts) < 2:
            print(f"Cannot extract dataset name from model: {model_name}")
            continue
            
        dataset = model_parts[1]  # Get the dataset part
        
        # Check if we have CV results for this dataset
        if dataset not in dataset_to_cv_results:
            print(f"No CV results found for dataset: {dataset}")
            continue
        
        # Get CV results and create study object
        cv_result = dataset_to_cv_results[dataset]
        cv_results_df = cv_result['cv_results']
        
        # Convert grid search results to study object
        study = convert_grid_search_to_study(cv_results_df, model_name)
        
        # Add study object to model data
        model_data['study'] = study
        models_updated += 1
        
        print(f"Added study object to {model_name} with {len(study.trials)} trials")
    
    # Save updated model data
    print(f"Updated {models_updated} of {len(elasticnet_models)} ElasticNet models")
    if models_updated > 0:
        io.save_model(elasticnet_models, "elasticnet_models.pkl", settings.MODEL_DIR)
        print(f"Saved updated ElasticNet models")
        return True
    
    return False

if __name__ == "__main__":
    if add_elasticnet_study_objects():
        print("\nNow run regenerate_contour_plots.py to generate contour plots for ElasticNet models")
    else:
        print("\nFailed to update ElasticNet models with study objects")