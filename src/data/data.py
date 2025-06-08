"""Data loading and preprocessing utilities."""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import os

# No need to modify sys.path - use proper imports

from src.config import settings
from src.utils import io

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
            # Return the esg_score column as a Series
            if 'esg_score' in scores_df.columns:
                return scores_df['esg_score']
            else:
                # Fallback: assume the score is in the second column (index 1)
                return scores_df.iloc[:, 1]
    
    # If we can't find the file, raise a helpful error
    raise FileNotFoundError(
        f"Could not find scores file '{settings.DATASET_FILES['scores']}'. "
        f"Tried locations: {[str(p) for p in potential_paths]}"
    )

"""
Corrected fix for Yeo-Johnson transformation implementation in data.py
Replace the get_base_and_yeo_features function with this improved version.
"""

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
    import pickle
    from pathlib import Path
    
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
        yeo_columns_from_pickle = pickle.load(f)
    
    # Print feature counts for verification
    print(f"Base features in pickle: {len(base_columns)} columns")
    print(f"Yeo features in pickle: {len(yeo_columns_from_pickle)} columns")
    
    # Filter columns to only include those available in the dataframe
    available_base_columns = [col for col in base_columns if col in feature_df.columns]
    
    # Create the base dataframe with available columns
    LR_Base = feature_df[available_base_columns].copy()
    
    # CORRECTLY HANDLE YEO TRANSFORMATION:
    yeo_prefix = 'yeo_joh_'
    
    # 1. Identify all columns that have Yeo-transformed versions in the dataframe
    yeo_transformed_columns = [col for col in feature_df.columns if col.startswith(yeo_prefix)]
    
    # 2. Get the original column names from the transformed ones
    original_numerical_columns = [col.replace(yeo_prefix, '') for col in yeo_transformed_columns]
    
    # 3. Identify categorical columns (those in base but not in original numerical)
    categorical_columns = [col for col in available_base_columns 
                          if col not in original_numerical_columns]
    
    print(f"\nFeature type breakdown:")
    print(f"  - Numerical features with Yeo transformations: {len(yeo_transformed_columns)}")
    print(f"  - Categorical features (no transformation): {len(categorical_columns)}")
    
    # 4. Create Yeo dataset with both transformed numerical and original categorical features
    complete_yeo_columns = yeo_transformed_columns + categorical_columns
    
    # 5. Create the Yeo dataframe
    LR_Yeo = feature_df[complete_yeo_columns].copy()
    
    # Additional validation
    print(f"\nFinal dataset dimensions:")
    print(f"  LR_Base: {LR_Base.shape} - {len(LR_Base.columns)} columns")
    print(f"  LR_Yeo: {LR_Yeo.shape} - {len(LR_Yeo.columns)} columns")
    
    # Verify that Yeo column count matches expectations
    expected_yeo_count = len(yeo_transformed_columns) + len(categorical_columns)
    if len(LR_Yeo.columns) != expected_yeo_count:
        print(f"WARNING: Unexpected column count in LR_Yeo.")
        print(f"  Expected: {expected_yeo_count}, Actual: {len(LR_Yeo.columns)}")
    
    return LR_Base, LR_Yeo, available_base_columns, complete_yeo_columns
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