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
    VISUALIZATION_DIR / "summary",
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