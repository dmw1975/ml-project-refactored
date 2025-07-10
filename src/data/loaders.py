"""Data loaders that use JSON metadata instead of pickle files."""

import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional

from src.config import settings

logger = logging.getLogger(__name__)


class BaseDataLoader:
    """Base class for data loaders."""
    
    def __init__(self, metadata_file: str):
        """
        Initialize data loader with metadata.
        
        Parameters
        ----------
        metadata_file : str
            Name of the metadata JSON file
        """
        self.metadata_file = metadata_file
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Load metadata from JSON file."""
        # Try multiple potential locations
        potential_paths = [
            settings.RAW_DATA_DIR / "metadata" / self.metadata_file,
            settings.DATA_DIR / "metadata" / self.metadata_file,
            Path("data/metadata") / self.metadata_file,
            Path("data/raw/metadata") / self.metadata_file,
        ]
        
        for path in potential_paths:
            if path.exists():
                logger.info(f"Loading metadata from: {path}")
                with open(path, 'r') as f:
                    return json.load(f)
        
        # If metadata not found, raise error
        raise FileNotFoundError(
            f"Could not find metadata file '{self.metadata_file}'. "
            f"Tried locations: {[str(p) for p in potential_paths]}"
        )
    
    def _load_csv_data(self, file_name: str) -> pd.DataFrame:
        """Load CSV data file."""
        potential_paths = [
            settings.RAW_DATA_DIR / file_name,
            settings.PROCESSED_DATA_DIR / file_name,
            Path("data/raw") / file_name,
            Path("data/processed") / file_name,
        ]
        
        for path in potential_paths:
            if path.exists():
                logger.info(f"Loading data from: {path}")
                return pd.read_csv(path)
        
        raise FileNotFoundError(
            f"Could not find data file '{file_name}'. "
            f"Tried locations: {[str(p) for p in potential_paths]}"
        )
    
    def _load_scores(self) -> pd.Series:
        """Load target scores."""
        scores_df = self._load_csv_data(settings.DATASET_FILES["scores"])
        
        if 'esg_score' in scores_df.columns:
            logger.info("Found 'esg_score' column")
            return scores_df['esg_score']
        else:
            logger.warning("'esg_score' column not found, using second column")
            return scores_df.iloc[:, 1]


class LinearModelDataLoader(BaseDataLoader):
    """Data loader for linear models with one-hot encoded features."""
    
    def __init__(self):
        """Initialize linear model data loader."""
        super().__init__("linear_model_columns.json")
        self.data = None
        self._load_data()
        
    def _load_data(self):
        """Load the linear models dataset."""
        self.data = self._load_csv_data("combined_df_for_linear_models.csv")
        logger.info(f"Loaded linear model data shape: {self.data.shape}")
        
        # Remove issuer_name if it's in the features
        issuer_col = self.metadata.get("issuer_identifier", "issuer_name")
        if issuer_col in self.data.columns:
            self.data = self.data.drop(columns=[issuer_col])
            
    def get_base_features(self) -> pd.DataFrame:
        """
        Get base features (original numerical + categorical).
        
        Returns
        -------
        pd.DataFrame
            DataFrame with base numerical and categorical features
        """
        base_numerical = self.metadata.get("base_numerical_features", [])
        one_hot_categorical = self.metadata.get("one_hot_encoded_features", [])
        
        # Combine base numerical with one-hot encoded categorical
        base_columns = base_numerical + one_hot_categorical
        
        # Filter to only include columns that exist in the data
        available_columns = [col for col in base_columns if col in self.data.columns]
        
        logger.info(f"Base features: {len(available_columns)} columns "
                   f"({len(base_numerical)} numerical + {len(one_hot_categorical)} one-hot)")
        
        return self.data[available_columns].copy()
    
    def get_yeo_features(self) -> pd.DataFrame:
        """
        Get Yeo-Johnson transformed features.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with Yeo-transformed numerical and categorical features
        """
        yeo_numerical = self.metadata.get("yeo_numerical_features", [])
        one_hot_categorical = self.metadata.get("one_hot_encoded_features", [])
        
        # Combine Yeo numerical with one-hot encoded categorical
        yeo_columns = yeo_numerical + one_hot_categorical
        
        # Filter to only include columns that exist in the data
        available_columns = [col for col in yeo_columns if col in self.data.columns]
        
        logger.info(f"Yeo features: {len(available_columns)} columns "
                   f"({len(yeo_numerical)} Yeo numerical + {len(one_hot_categorical)} one-hot)")
        
        return self.data[available_columns].copy()
    
    def get_base_and_yeo_features(self) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
        """
        Get both base and Yeo features with column lists.
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]
            Base features, Yeo features, base column names, Yeo column names
        """
        base_df = self.get_base_features()
        yeo_df = self.get_yeo_features()
        
        return base_df, yeo_df, base_df.columns.tolist(), yeo_df.columns.tolist()
    
    def get_categorical_columns_for_stratification(self) -> List[str]:
        """
        Get categorical columns for stratified splitting.
        
        Returns
        -------
        List[str]
            List of one-hot encoded sector columns
        """
        one_hot_columns = self.metadata.get("one_hot_encoded_features", [])
        
        # Filter for sector-related columns
        sector_columns = [col for col in one_hot_columns 
                         if col.startswith("gics_sector_") and col in self.data.columns]
        
        logger.info(f"Found {len(sector_columns)} sector columns for stratification")
        return sector_columns


class TreeModelDataLoader(BaseDataLoader):
    """Data loader for tree models with native categorical features."""
    
    def __init__(self):
        """Initialize tree model data loader."""
        super().__init__("tree_model_columns.json")
        self.data = None
        self._load_data()
        
    def _load_data(self):
        """Load the tree models dataset."""
        self.data = self._load_csv_data("combined_df_for_tree_models.csv")
        logger.info(f"Loaded tree model data shape: {self.data.shape}")
        
        # Remove issuer_name if it's in the features
        issuer_col = self.metadata.get("issuer_identifier", "issuer_name")
        if issuer_col in self.data.columns:
            self.data = self.data.drop(columns=[issuer_col])
            
        # Set categorical features as category dtype
        categorical_features = self.metadata.get("categorical_features", [])
        for cat_feature in categorical_features:
            if cat_feature in self.data.columns:
                self.data[cat_feature] = self.data[cat_feature].astype('category')
                
    def get_base_features(self) -> pd.DataFrame:
        """
        Get base features (original numerical + categorical).
        
        Returns
        -------
        pd.DataFrame
            DataFrame with base numerical and categorical features
        """
        base_numerical = self.metadata.get("base_numerical_features", [])
        categorical = self.metadata.get("categorical_features", [])
        
        # Combine base numerical with categorical
        base_columns = base_numerical + categorical
        
        # Filter to only include columns that exist in the data
        available_columns = [col for col in base_columns if col in self.data.columns]
        
        logger.info(f"Base features: {len(available_columns)} columns "
                   f"({len(base_numerical)} numerical + {len(categorical)} categorical)")
        
        return self.data[available_columns].copy()
    
    def get_yeo_features(self) -> pd.DataFrame:
        """
        Get Yeo-Johnson transformed features.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with Yeo-transformed numerical and categorical features
        """
        yeo_numerical = self.metadata.get("yeo_numerical_features", [])
        categorical = self.metadata.get("categorical_features", [])
        
        # Combine Yeo numerical with categorical
        yeo_columns = yeo_numerical + categorical
        
        # Filter to only include columns that exist in the data
        available_columns = [col for col in yeo_columns if col in self.data.columns]
        
        logger.info(f"Yeo features: {len(available_columns)} columns "
                   f"({len(yeo_numerical)} Yeo numerical + {len(categorical)} categorical)")
        
        return self.data[available_columns].copy()
    
    def get_base_and_yeo_features(self) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
        """
        Get both base and Yeo features with column lists.
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]
            Base features, Yeo features, base column names, Yeo column names
        """
        base_df = self.get_base_features()
        yeo_df = self.get_yeo_features()
        
        return base_df, yeo_df, base_df.columns.tolist(), yeo_df.columns.tolist()
    
    def get_categorical_features(self) -> List[str]:
        """
        Get list of categorical features.
        
        Returns
        -------
        List[str]
            List of categorical feature names
        """
        return self.metadata.get("categorical_features", [])
    
    def get_sector_column(self) -> str:
        """
        Get the main sector column for stratification.
        
        Returns
        -------
        str
            Name of the sector column
        """
        # First check if explicitly defined in metadata
        if "sector_column" in self.metadata:
            return self.metadata["sector_column"]
        
        # Otherwise default to gics_sector
        return "gics_sector"


def add_random_feature(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Add a random feature to a dataset for feature importance benchmarking.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input dataset
    seed : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    pd.DataFrame
        Dataset with an additional random feature column
    """
    df_random = df.copy()
    np.random.seed(seed)
    df_random['random_feature'] = np.random.normal(size=len(df_random))
    
    logger.info(f"Added random feature to dataset: {df_random.shape}")
    return df_random


# Backward compatibility functions
def load_features_data(model_type='linear') -> pd.DataFrame:
    """
    Load features dataset based on model type.
    
    Parameters
    ----------
    model_type : str
        Type of model ('linear' or 'tree')
        
    Returns
    -------
    pd.DataFrame
        Features dataframe
    """
    if model_type == 'linear':
        loader = LinearModelDataLoader()
        # Return the full data for backward compatibility
        return loader.data
    else:
        loader = TreeModelDataLoader()
        return loader.data


def get_base_and_yeo_features(feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Get base and Yeo features from the feature dataframe.
    
    This function provides backward compatibility with the old interface.
    It attempts to use the new metadata-based approach if available.
    
    Parameters
    ----------
    feature_df : pd.DataFrame
        The features dataframe
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]
        Base features, Yeo features, base column names, Yeo column names
    """
    try:
        # Try to determine if this is linear or tree model data
        if len(feature_df.columns) > 100:  # Likely linear model data
            loader = LinearModelDataLoader()
        else:  # Likely tree model data
            loader = TreeModelDataLoader()
        
        return loader.get_base_and_yeo_features()
        
    except FileNotFoundError:
        # Fallback to old logic if metadata not available
        logger.warning("Metadata files not found. Falling back to old pickle-based approach.")
        
        # Import the old function
        from src.data.data import get_base_and_yeo_features as old_get_base_and_yeo_features
        return old_get_base_and_yeo_features(feature_df)