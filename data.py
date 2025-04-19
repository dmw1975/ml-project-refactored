"""Data loading and preprocessing utilities."""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings
from utils import io

def load_features_data():
    """Load features dataset."""
    # Try multiple potential locations for the features file
    potential_paths = [
        settings.PROCESSED_DATA_DIR / settings.DATASET_FILES["features"],
        Path("data/processed") / settings.DATASET_FILES["features"],
        Path("data/raw") / settings.DATASET_FILES["features"],  # Added raw data path
        Path("data") / settings.DATASET_FILES["features"],
        Path(settings.DATASET_FILES["features"])
    ]
    
    # Try each path
    for path in potential_paths:
        if path.exists():
            print(f"Loading features from: {path}")
            return pd.read_csv(path)
    
    # If we can't find the file, raise a helpful error
    raise FileNotFoundError(
        f"Could not find features file '{settings.DATASET_FILES['features']}'. "
        f"Tried locations: {[str(p) for p in potential_paths]}"
    )

def load_scores_data():
    """Load target scores dataset."""
    # Try multiple potential locations for the scores file
    potential_paths = [
        settings.PROCESSED_DATA_DIR / settings.DATASET_FILES["scores"],
        Path("data/processed") / settings.DATASET_FILES["scores"],
        Path("data/raw") / settings.DATASET_FILES["scores"],  # Added raw data path
        Path("data") / settings.DATASET_FILES["scores"],
        Path(settings.DATASET_FILES["scores"])
    ]
    
    # Try each path
    for path in potential_paths:
        if path.exists():
            print(f"Loading scores from: {path}")
            scores_df = pd.read_csv(path)
            # Assuming the score is in the first column
            return scores_df.iloc[:, 0]
    
    # If we can't find the file, raise a helpful error
    raise FileNotFoundError(
        f"Could not find scores file '{settings.DATASET_FILES['scores']}'. "
        f"Tried locations: {[str(p) for p in potential_paths]}"
    )

def get_base_and_yeo_features(feature_df):
    """
    Get base features and Yeo-Johnson transformed features from pre-saved pickle files.
    
    Returns:
    --------
    LR_Base : pandas.DataFrame
        DataFrame with base features
    LR_Yeo : pandas.DataFrame
        DataFrame with Yeo-Johnson transformed features
    base_columns : list
        List of column names in LR_Base
    yeo_columns : list
        List of column names in LR_Yeo
    """
    # Try multiple potential locations for the pickle files
    potential_base_paths = [
        Path("data/pkl/base_columns.pkl"),
        Path("data/base_columns.pkl"),
        Path("base_columns.pkl")
    ]
    
    potential_yeo_paths = [
        Path("data/pkl/yeo_columns.pkl"),
        Path("data/yeo_columns.pkl"),
        Path("yeo_columns.pkl")
    ]
    
    base_columns = None
    yeo_columns = None
    
    # Try to load base columns
    for path in potential_base_paths:
        if path.exists():
            print(f"Loading base columns from: {path}")
            with open(path, 'rb') as f:
                base_columns = pickle.load(f)
            break
    
    # Try to load yeo columns
    for path in potential_yeo_paths:
        if path.exists():
            print(f"Loading yeo columns from: {path}")
            with open(path, 'rb') as f:
                yeo_columns = pickle.load(f)
            break
    
    # If we couldn't load from pickle files, detect columns from the dataframe
    if base_columns is None or yeo_columns is None:
        print("Could not load columns from pickle files, detecting columns from data")
        
        # Identify different types of columns
        all_columns = feature_df.columns.tolist()
        
        # Identify categorical columns (typically one-hot encoded)
        categorical_cols = [col for col in all_columns 
                          if col.startswith(('issuer_cntry_', 'cntry_of_risk_', 'gics_sector_', 
                                           'gics_sub_ind_', 'top_1_shareholder_location_',
                                           'top_2_shareholder_location_', 'top_3_shareholder_location_'))]
        
        # Identify columns that are already Yeo-Johnson transformed (prefixed with yeo_joh_)
        yeo_cols = [col for col in all_columns if col.startswith('yeo_joh_')]
        
        # Identify original numerical columns (exclude categorical and Yeo columns)
        exclude_prefixes = ('yeo_joh_', 'issuer_cntry_', 'cntry_of_risk_', 'gics_sector_', 
                           'gics_sub_ind_', 'top_1_shareholder_location_',
                           'top_2_shareholder_location_', 'top_3_shareholder_location_')
        exclude_columns = ('ticker', 'year', 'issuer_name')
        
        orig_numerical_cols = [col for col in all_columns 
                              if not col.startswith(exclude_prefixes) 
                              and col not in exclude_columns]
        
        # Create column lists
        base_columns = categorical_cols + orig_numerical_cols
        yeo_columns = categorical_cols + yeo_cols
    
    print(f"Base features: {len(base_columns)} columns")
    print(f"Yeo features: {len(yeo_columns)} columns")
    
    # Create the dataframes, handling missing columns gracefully
    available_base_columns = [col for col in base_columns if col in feature_df.columns]
    available_yeo_columns = [col for col in yeo_columns if col in feature_df.columns]
    
    if len(available_base_columns) < len(base_columns):
        print(f"Warning: {len(base_columns) - len(available_base_columns)} base columns not found in data")
    
    if len(available_yeo_columns) < len(yeo_columns):
        print(f"Warning: {len(yeo_columns) - len(available_yeo_columns)} yeo columns not found in data")
    
    LR_Base = feature_df[available_base_columns].copy()
    LR_Yeo = feature_df[available_yeo_columns].copy()
    
    return LR_Base, LR_Yeo, base_columns, yeo_columns

def add_random_feature(df):
    """
    Add a random feature to a dataset for feature importance benchmarking.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataset
    
    Returns:
    --------
    df_random : pandas.DataFrame
        Dataset with an additional random feature column
    """
    # Make a copy to avoid modifying the original
    df_random = df.copy()
    
    # Generate random values with the same length as the dataframe
    np.random.seed(42)  # For reproducibility
    random_values = np.random.normal(size=len(df_random))
    
    # Add as a new column
    df_random['random_feature'] = random_values
    
    return df_random

if __name__ == "__main__":
    # Print the current directory and settings to help with debugging
    print(f"Current directory: {os.getcwd()}")
    print(f"Settings DATA_DIR: {settings.DATA_DIR}")
    print(f"Settings PROCESSED_DATA_DIR: {settings.PROCESSED_DATA_DIR}")
    print(f"Settings RAW_DATA_DIR: {settings.RAW_DATA_DIR}")
    print(f"Settings DATASET_FILES: {settings.DATASET_FILES}")
    
    # Test the functions when run directly
    try:
        print("\nTesting data loading and feature extraction...")
        
        feature_df = load_features_data()
        score_df = load_scores_data()
        
        print(f"Loaded features shape: {feature_df.shape}")
        print(f"Loaded scores shape: {score_df.shape}")
        
        LR_Base, LR_Yeo, base_columns, yeo_columns = get_base_and_yeo_features(feature_df)
        
        print(f"Base dataset shape: {LR_Base.shape}")
        print(f"Yeo dataset shape: {LR_Yeo.shape}")
        
        # Test adding random feature
        LR_Base_random = add_random_feature(LR_Base)
        print(f"Added random feature to base features: {LR_Base_random.shape}")
        
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()