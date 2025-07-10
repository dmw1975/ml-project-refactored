"""Data loading utilities."""

import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from src.config import settings

def load_features_data():
    """Load the features dataset."""
    file_path = settings.RAW_DATA_DIR / settings.DATASET_FILES["features"]
    return pd.read_csv(file_path, index_col='issuer_name')

def load_scores_data():
    """Load the ESG scores dataset."""
    file_path = settings.RAW_DATA_DIR / settings.DATASET_FILES["scores"]
    df = pd.read_csv(file_path, index_col='issuer_name')
    # Return as Series, not DataFrame
    return df['esg_score']

def get_base_and_yeo_features(feature_df):
    """Extract Base and Yeo-Johnson transformed features."""
    # Extract column lists
    base_columns = [col for col in feature_df.columns if not col.startswith('yeo_joh_')]
    yeo_columns = [col for col in feature_df.columns if col.startswith('yeo_joh_')]
    
    # Create feature sets
    LR_Base = feature_df[base_columns]
    LR_Yeo = feature_df[yeo_columns]
    
    return LR_Base, LR_Yeo, base_columns, yeo_columns

def add_random_feature(df):
    """Add a random feature to the dataframe."""
    import numpy as np
    
    # Create a copy to avoid modifying original
    df_random = df.copy()
    
    # Add random feature
    np.random.seed(42)  # For reproducibility
    df_random['random_feature'] = np.random.normal(0, 1, size=df.shape[0])
    
    return df_random