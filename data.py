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
    base_path = Path("data/pkl/base_columns.pkl")
    yeo_path = Path("data/pkl/yeo_columns.pkl")
    
    with open(base_path, 'rb') as f:
        base_columns = pickle.load(f)

    with open(yeo_path, 'rb') as f:
        yeo_columns = pickle.load(f)
    
    available_base_columns = [col for col in base_columns if col in feature_df.columns]
    available_yeo_columns = [col for col in yeo_columns if col in feature_df.columns]

    # ALSO keep sector columns
    sector_columns = [col for col in feature_df.columns if col.startswith('gics_sector_') or col.startswith('sector_')]

    # Create LR_Base and LR_Yeo datasets
    LR_Base = feature_df[available_base_columns + sector_columns].copy()
    LR_Yeo = feature_df[available_yeo_columns + sector_columns].copy()

    return LR_Base, LR_Yeo, available_base_columns, available_yeo_columns





"""
def get_base_and_yeo_features(feature_df):
    
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
    
    # Define the correct paths for pickle files
    base_path = Path("data/pkl/base_columns.pkl")
    yeo_path = Path("data/pkl/yeo_columns.pkl")
    
    # Load base columns with proper error handling
    if not base_path.exists():
        raise FileNotFoundError(f"Base columns pickle file not found at: {base_path}")
    
    print(f"Loading base columns from: {base_path}")
    with open(base_path, 'rb') as f:
        base_columns = pickle.load(f)
    
    # Load yeo columns with proper error handling  
    if not yeo_path.exists():
        raise FileNotFoundError(f"Yeo columns pickle file not found at: {yeo_path}")
    
    print(f"Loading yeo columns from: {yeo_path}")
    with open(yeo_path, 'rb') as f:
        yeo_columns = pickle.load(f)
    
    # Print feature counts for verification
    print(f"Base features in pickle: {len(base_columns)} columns")
    print(f"Yeo features in pickle: {len(yeo_columns)} columns")
    
    # Filter columns to only include those available in the dataframe
    available_base_columns = [col for col in base_columns if col in feature_df.columns]
    
    # Create the base dataframe
    LR_Base = feature_df[available_base_columns].copy()
    
    # CORRECTED APPROACH: Properly separate numerical and categorical features
    # First, identify all numerical features that have Yeo-transformed versions
    yeo_prefix = 'yeo_joh_'
    yeo_transformed_columns = [col for col in feature_df.columns if col.startswith(yeo_prefix)]
    
    # Get the original column names from the transformed ones
    original_numerical_columns = [col.replace(yeo_prefix, '') for col in yeo_transformed_columns]
    
    # Identify categorical columns (those in base but not in original numerical)
    categorical_columns = [col for col in available_base_columns 
                          if col not in original_numerical_columns]
    
    # Create Yeo dataset with both transformed numerical and original categorical features
    complete_yeo_columns = yeo_transformed_columns + categorical_columns
    
    # Create the Yeo dataframe
    LR_Yeo = feature_df[complete_yeo_columns].copy()
    
    # Print detailed information about the feature composition
    print(f"\nFeature breakdown:")
    print(f"  LR_Base: {len(available_base_columns)} total features")
    print(f"  LR_Yeo: {len(complete_yeo_columns)} total features")
    print(f"    - {len(yeo_transformed_columns)} Yeo-transformed numerical features")
    print(f"    - {len(categorical_columns)} categorical features")

    available_yeo_columns = [col for col in yeo_columns if col in feature_df.columns]
    print(f"Available Yeo columns (in both pickle and dataframe): {len(available_yeo_columns)}")
    print(f"Missing Yeo columns: {len(yeo_columns) - len(available_yeo_columns)}")
    
    # Verify no overlap between numerical and categorical
    overlap = set(yeo_transformed_columns).intersection(set(categorical_columns))
    if overlap:
        print(f"WARNING: {len(overlap)} overlapping columns between numerical and categorical!")
    
    return LR_Base, LR_Yeo, available_base_columns, complete_yeo_columns

"""

def add_random_feature(df, seed=42):
    """
    Add a random feature to a dataset for feature importance benchmarking.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataset
    seed : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    df_random : pandas.DataFrame
        Dataset with an additional random feature column
    """
    # Make a copy to avoid modifying the original
    df_random = df.copy()
    
    # Generate random values with the same length as the dataframe
    np.random.seed(seed)  # For reproducibility
    random_values = np.random.normal(size=len(df_random))
    
    # Add as a new column
    df_random['random_feature'] = random_values
    
    # Print confirmation
    print(f"Added random feature to dataset: {df_random.shape}")
    
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

        print(f"Actual LR_Base columns: {len(LR_Base.columns)}")
        print(f"Actual LR_Yeo columns: {len(LR_Yeo.columns)}")
        
        # Test adding random feature
        LR_Base_random = add_random_feature(LR_Base)
        print(f"Added random feature to base features: {LR_Base_random.shape}")
        
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()