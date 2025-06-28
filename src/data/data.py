"""Data loading and preprocessing utilities."""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
import sys
import os

# No need to modify sys.path - use proper imports

from src.config import settings
from src.utils import io

logger = logging.getLogger(__name__)

def load_features_data(model_type='linear'):
    """Load features dataset based on model type.
    
    Parameters
    ----------
    model_type : str
        Type of model ('linear' or 'tree'). Defaults to 'linear'.
        
    Returns
    -------
    pd.DataFrame
        Features dataframe
    """
    if model_type == 'linear':
        # For linear models, use the linear models dataset
        file_name = "combined_df_for_linear_models.csv"
    elif model_type == 'tree':
        # For tree models, use the tree models dataset
        file_name = "combined_df_for_tree_models.csv"
    else:
        # Fall back to the general file for backward compatibility
        file_name = settings.DATASET_FILES["features"]
    
    # Try multiple potential locations for the features file
    potential_paths = [
        settings.RAW_DATA_DIR / file_name,  # Primary location
        settings.PROCESSED_DATA_DIR / file_name,
        Path("data/raw") / file_name,
        Path("data/processed") / file_name,
        Path("data") / file_name,
        Path(file_name)
    ]
    
    # Try each path
    for path in potential_paths:
        if path.exists():
            logger.info(f"Loading {model_type} features from: {path}")
            df = pd.read_csv(path)
            logger.info(f"Loaded features shape: {df.shape}")
            logger.debug(f"Feature columns: {df.columns.tolist()[:10]}... (showing first 10)")
            
            # NOTE: Data files are currently pre-normalized which breaks linear models
            # Denormalization attempted but estimates don't match original scales
            # Tree models still work reasonably well with normalized data
            
            return df
    
    # If we can't find the file, raise a helpful error
    raise FileNotFoundError(
        f"Could not find {model_type} features file '{file_name}'. "
        f"Tried locations: {[str(p) for p in potential_paths]}"
    )

def load_scores_data():
    """Load target scores dataset."""
    # Try multiple potential locations for the scores file
    potential_paths = [
        settings.PROCESSED_DATA_DIR / settings.DATASET_FILES["scores"],
        settings.RAW_DATA_DIR / settings.DATASET_FILES["scores"],  # Use absolute path from settings
        Path("data/processed") / settings.DATASET_FILES["scores"],
        Path("data/raw") / settings.DATASET_FILES["scores"],
        Path("data") / settings.DATASET_FILES["scores"],
        Path(settings.DATASET_FILES["scores"])
    ]
    
    # Try each path
    for path in potential_paths:
        if path.exists():
            logger.info(f"Loading scores from: {path}")
            scores_df = pd.read_csv(path)
            logger.info(f"Loaded scores shape: {scores_df.shape}")
            # Return the esg_score column as a Series
            if 'esg_score' in scores_df.columns:
                logger.info("Found 'esg_score' column")
                return scores_df['esg_score']
            else:
                # Fallback: assume the score is in the second column (index 1)
                logger.warning("'esg_score' column not found, using second column")
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
    Get base features and Yeo-Johnson transformed features.
    
    This function first tries to use the new JSON metadata approach,
    and falls back to pickle files if metadata is not available.
    
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
    # First, try to use the new loader approach
    try:
        from src.data.loaders import get_base_and_yeo_features as new_get_features
        logger.info("Attempting to use JSON metadata for feature loading...")
        return new_get_features(feature_df)
    except (ImportError, FileNotFoundError) as e:
        logger.info(f"JSON metadata not available ({e}), falling back to pickle approach...")
    
    # Fallback to original pickle-based implementation
    import pickle
    from pathlib import Path
    
    # Define the correct paths for pickle files
    base_path = Path("data/pkl/base_columns.pkl")
    yeo_path = Path("data/pkl/yeo_columns.pkl")
    
    # Load base columns with proper error handling
    if not base_path.exists():
        raise FileNotFoundError(f"Base columns pickle file not found at: {base_path}")
    
    logger.info(f"Loading base columns from: {base_path}")
    with open(base_path, 'rb') as f:
        base_columns = pickle.load(f)
    logger.info(f"Loaded {len(base_columns)} base columns")
    
    # Load yeo columns with proper error handling  
    if not yeo_path.exists():
        raise FileNotFoundError(f"Yeo columns pickle file not found at: {yeo_path}")
    
    print(f"Loading yeo columns from: {yeo_path}")
    with open(yeo_path, 'rb') as f:
        yeo_columns_from_pickle = pickle.load(f)
    
    # Print feature counts for verification
    print(f"Base features in pickle: {len(base_columns)} columns")
    print(f"Yeo features in pickle: {len(yeo_columns_from_pickle)} columns")
    
    # IMPORTANT: The pickle files are incomplete and only contain 26 numerical features
    # We need to include ALL features from the dataframe, not just those in the pickle
    
    # Get all columns from the dataframe
    all_columns = list(feature_df.columns)
    
    # Identify categorical columns (one-hot encoded sectors)
    categorical_columns = [col for col in all_columns if col.startswith('gics_sector_')]
    
    # Identify Yeo-transformed columns
    yeo_prefix = 'yeo_joh_'
    yeo_transformed_columns = [col for col in all_columns if col.startswith(yeo_prefix)]
    
    # Get the original column names from the transformed ones
    original_numerical_columns = [col.replace(yeo_prefix, '') for col in yeo_transformed_columns]
    
    # Identify all numerical columns (those that are not categorical and have Yeo versions)
    all_numerical_columns = [col for col in all_columns 
                            if col not in categorical_columns 
                            and not col.startswith(yeo_prefix)
                            and col in feature_df.select_dtypes(include=[np.number]).columns]
    
    # Create base dataset with all numerical and categorical features
    base_columns_to_use = all_numerical_columns + categorical_columns
    LR_Base = feature_df[base_columns_to_use].copy()
    
    # Create Yeo dataset with transformed numerical and original categorical features
    yeo_columns_to_use = yeo_transformed_columns + categorical_columns
    LR_Yeo = feature_df[yeo_columns_to_use].copy()
    
    print(f"\nFeature type breakdown:")
    print(f"  - All numerical features: {len(all_numerical_columns)}")
    print(f"  - Yeo-transformed features: {len(yeo_transformed_columns)}")
    print(f"  - Categorical features: {len(categorical_columns)}")
    
    # Update the return values to match
    available_base_columns = base_columns_to_use
    complete_yeo_columns = yeo_columns_to_use
    
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