"""Central configuration for the ML project."""

import os
import numpy as np
from pathlib import Path

# Root directory of the project
ROOT_DIR = Path(__file__).parent.parent.absolute()

# Data paths
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"

# Output paths
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
FEATURE_IMPORTANCE_DIR = OUTPUT_DIR / "feature_importance"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

# Ensure directories exist
for dir_path in [
    RAW_DATA_DIR, 
    PROCESSED_DATA_DIR, 
    INTERIM_DATA_DIR,
    MODEL_DIR, 
    METRICS_DIR, 
    FEATURE_IMPORTANCE_DIR, 
    VISUALIZATION_DIR / "performance",
    VISUALIZATION_DIR / "features",
    OUTPUT_DIR / "reports"
]:
    os.makedirs(dir_path, exist_ok=True)

# Dataset filenames
DATASET_FILES = {
    "features": "combined_df_for_ml_models.csv",
    "scores": "score.csv"
}

# Model parameters
LINEAR_REGRESSION_PARAMS = {
    "random_state": 42,
    "test_size": 0.2
}

ELASTICNET_PARAMS = {
    "random_state": 42,
    "test_size": 0.2,
    "alpha_grid": list(10 ** np.linspace(-1, 0.2, 15)),  # Logarithmically spaced from 0.1 to ~1.58
    "l1_ratio_grid": list(np.linspace(0, 1, 10))         # Evenly spaced from 0 to 1
}

# Visualization settings
VIZ_STYLE = "whitegrid"
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
FIGURE_SIZE_DEFAULT = (10, 6)

# Color definitions
MODEL_COLORS = {
    'LR_Base': '#3498db',
    'LR_Base_Random': '#9b59b6',
    'LR_Yeo': '#2ecc71',
    'LR_Yeo_Random': '#f39c12',
    'ElasticNet_LR_Base': '#e74c3c',
    'ElasticNet_LR_Base_Random': '#1abc9c',
    'ElasticNet_LR_Yeo': '#d35400',
    'ElasticNet_LR_Yeo_Random': '#8e44ad'
}

# Add these lines to your existing config/settings.py file

# XGBoost parameters
XGBOOST_PARAMS = {
    "random_state": 42,
    "test_size": 0.2,
    "n_trials": 50  # For Optuna optimization
}

# Add XGBoost colors to your existing MODEL_COLORS dictionary
MODEL_COLORS.update({
    'XGB_Base_basic': '#1f77b4',
    'XGB_Base_optuna': '#ff7f0e',
    'XGB_Yeo_basic': '#2ca02c',
    'XGB_Yeo_optuna': '#d62728',
    'XGB_Base_Random_basic': '#9467bd',
    'XGB_Base_Random_optuna': '#8c564b',
    'XGB_Yeo_Random_basic': '#e377c2',
    'XGB_Yeo_Random_optuna': '#7f7f7f'
})

# LightGBM parameters
LIGHTGBM_PARAMS = {
    "random_state": 42,
    "test_size": 0.2,
    "n_trials": 50  # For Optuna optimization
}

# Add LightGBM colors to MODEL_COLORS dictionary
MODEL_COLORS.update({
    'LightGBM_Base_basic': '#17becf',
    'LightGBM_Base_optuna': '#bcbd22',
    'LightGBM_Yeo_basic': '#7f7f7f',
    'LightGBM_Yeo_optuna': '#9467bd',
    'LightGBM_Base_Random_basic': '#8c564b',
    'LightGBM_Base_Random_optuna': '#e377c2',
    'LightGBM_Yeo_Random_basic': '#1f77b4',
    'LightGBM_Yeo_Random_optuna': '#ff7f0e'
})

# CatBoost parameters
CATBOOST_PARAMS = {
    "random_state": 42,
    "test_size": 0.2,
    "n_trials": 50  # For Optuna optimization
}

# Add CatBoost colors to MODEL_COLORS dictionary
MODEL_COLORS.update({
    'CatBoost_Base_basic': '#6a3d9a',
    'CatBoost_Base_optuna': '#cab2d6',
    'CatBoost_Yeo_basic': '#fb9a99',
    'CatBoost_Yeo_optuna': '#e31a1c',
    'CatBoost_Base_Random_basic': '#33a02c',
    'CatBoost_Base_Random_optuna': '#b2df8a',
    'CatBoost_Yeo_Random_basic': '#a6cee3',
    'CatBoost_Yeo_Random_optuna': '#1f78b4'
})