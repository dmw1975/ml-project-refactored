"""
Unified train/test split management for consistent splits across all model types.

This module ensures that all models use the exact same train/test split by:
1. Creating the split once based on company identifiers
2. Saving the split indices to a JSON file
3. Providing functions to load and apply the saved split

This fixes the issue where different model types were getting different
train/test splits due to different indexing strategies.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Union, Optional, List
from sklearn.model_selection import train_test_split
import logging

from src.config import settings

logger = logging.getLogger(__name__)


class UnifiedTrainTestSplit:
    """Manages unified train/test splits across all model types."""
    
    def __init__(self, split_file: Optional[Path] = None):
        """
        Initialize the unified split manager.
        
        Parameters
        ----------
        split_file : Path, optional
            Path to the split indices file. If None, uses default location.
        """
        if split_file is None:
            self.split_file = settings.DATA_DIR / "processed" / "train_test_split.json"
        else:
            self.split_file = Path(split_file)
            
        self.split_data = None
        
    def create_split(self, 
                     data: pd.DataFrame,
                     target: pd.Series,
                     test_size: float = 0.2,
                     random_state: int = 42,
                     stratify_column: Optional[str] = None,
                     use_indices: bool = False) -> dict:
        """
        Create a new train/test split and save the indices.
        
        Parameters
        ----------
        data : pd.DataFrame
            The feature data with companies as index
        target : pd.Series
            The target variable (ESG scores)
        test_size : float
            Proportion of data for test set
        random_state : int
            Random seed for reproducibility
        stratify_column : str, optional
            Column name to use for stratification
        use_indices : bool, optional
            If True, use integer indices instead of company names
            
        Returns
        -------
        dict
            Split information including train/test indices or company names
        """
        # Handle indices based on use_indices flag
        if use_indices:
            # Use integer indices directly
            if not isinstance(data.index[0], (int, np.integer)):
                logger.warning("use_indices=True but data has non-integer index, resetting to integer index")
                data = data.reset_index(drop=True)
                target = target.reset_index(drop=True)
        else:
            # Ensure we have company names as index
            if data.index.name != 'issuer_name':
                if 'issuer_name' in data.columns:
                    data = data.set_index('issuer_name')
                else:
                    # If index already contains company names (strings), use as is
                    if isinstance(data.index[0], str):
                        data.index.name = 'issuer_name'
                    else:
                        raise ValueError("Data must have issuer_name as index or column")
        
        # Align target with data
        target = target.loc[data.index]
        
        # Get stratification column if specified
        stratify = None
        if stratify_column:
            if stratify_column in data.columns:
                stratify = data[stratify_column]
            else:
                # Try to extract sector from sector columns
                sector_cols = [col for col in data.columns if col.startswith('gics_sector_')]
                if sector_cols:
                    # Create sector labels from one-hot encoding
                    stratify = pd.Series(index=data.index, dtype=str)
                    for idx in data.index:
                        for col in sector_cols:
                            if data.loc[idx, col] == 1:
                                stratify.loc[idx] = col.replace('gics_sector_', '')
                                break
                    if stratify.isna().any():
                        logger.warning("Some companies have no sector assignment")
                        stratify = stratify.fillna('Unknown')
        
        # Perform the split
        if stratify is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                data, target, test_size=test_size, random_state=random_state, 
                stratify=stratify
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                data, target, test_size=test_size, random_state=random_state
            )
        
        # Create split data structure
        if use_indices:
            self.split_data = {
                "train_indices": X_train.index.tolist(),
                "test_indices": X_test.index.tolist(),
                "use_indices": True,
                "split_params": {
                    "test_size": test_size,
                    "random_state": random_state,
                    "stratify_column": stratify_column,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "y_train_mean": float(y_train.mean()),
                    "y_train_std": float(y_train.std()),
                    "y_test_mean": float(y_test.mean()),
                    "y_test_std": float(y_test.std())
                },
                "version": "1.1"
            }
        else:
            self.split_data = {
                "train_companies": X_train.index.tolist(),
                "test_companies": X_test.index.tolist(),
                "use_indices": False,
                "split_params": {
                    "test_size": test_size,
                    "random_state": random_state,
                    "stratify_column": stratify_column,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "y_train_mean": float(y_train.mean()),
                    "y_train_std": float(y_train.std()),
                    "y_test_mean": float(y_test.mean()),
                    "y_test_std": float(y_test.std())
                },
                "version": "1.1"
            }
        
        # Save to file
        self.save_split()
        
        logger.info(f"Created train/test split: {len(X_train)} train, {len(X_test)} test samples")
        logger.info(f"Train mean: {y_train.mean():.4f}, Test mean: {y_test.mean():.4f}")
        
        return self.split_data
    
    def save_split(self):
        """Save the current split data to JSON file."""
        if self.split_data is None:
            raise ValueError("No split data to save")
            
        # Ensure directory exists
        self.split_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        with open(self.split_file, 'w') as f:
            json.dump(self.split_data, f, indent=2)
            
        logger.info(f"Saved train/test split to {self.split_file}")
    
    def load_split(self) -> dict:
        """
        Load existing split data from file.
        
        Returns
        -------
        dict
            Split information including train/test company names or indices
        """
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
            
        with open(self.split_file, 'r') as f:
            self.split_data = json.load(f)
            
        logger.info(f"Loaded train/test split from {self.split_file}")
        
        # Handle both old and new formats
        if 'use_indices' in self.split_data and self.split_data['use_indices']:
            logger.info(f"Split contains {len(self.split_data['train_indices'])} train, "
                       f"{len(self.split_data['test_indices'])} test indices")
        else:
            logger.info(f"Split contains {len(self.split_data['train_companies'])} train, "
                       f"{len(self.split_data['test_companies'])} test companies")
        
        return self.split_data
    
    def apply_split(self, 
                    data: pd.DataFrame, 
                    target: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Apply the saved split to new data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Feature data with companies as index or integer index
        target : pd.Series
            Target variable
            
        Returns
        -------
        X_train, X_test, y_train, y_test
            Split datasets
        """
        if self.split_data is None:
            self.load_split()
        
        use_indices = self.split_data.get('use_indices', False)
        
        if use_indices:
            # Use integer indices
            train_indices = self.split_data['train_indices']
            test_indices = self.split_data['test_indices']
            
            # Filter for indices that exist in current data
            max_idx = len(data) - 1
            train_indices = [idx for idx in train_indices if idx <= max_idx]
            test_indices = [idx for idx in test_indices if idx <= max_idx]
            
            # Apply split using iloc
            X_train = data.iloc[train_indices]
            X_test = data.iloc[test_indices]
            y_train = target.iloc[train_indices]
            y_test = target.iloc[test_indices]
        else:
            # Use company names
            # Ensure we have company names as index
            if 'issuer_name' in data.columns:
                # issuer_name is a column, set it as index
                data = data.set_index('issuer_name')
                if target.index.name != 'issuer_name':
                    target = target.set_index(data.index)
            else:
                # Check if index already contains company names (strings)
                if len(data) > 0 and isinstance(data.index[0], str):
                    # Index contains strings, assume they are company names
                    if data.index.name != 'issuer_name':
                        data.index.name = 'issuer_name'
                else:
                    raise ValueError("Data must have issuer_name as index or column")
            
            # Get train and test companies
            train_companies = self.split_data['train_companies']
            test_companies = self.split_data['test_companies']
            
            # Filter for companies that exist in current data
            train_companies = [c for c in train_companies if c in data.index]
            test_companies = [c for c in test_companies if c in data.index]
            
            # Apply split
            X_train = data.loc[train_companies]
            X_test = data.loc[test_companies]
            y_train = target.loc[train_companies]
            y_test = target.loc[test_companies]
        
        # Log statistics
        logger.info(f"Applied split: {len(X_train)} train, {len(X_test)} test samples")
        logger.info(f"Train mean: {y_train.mean():.4f}, Test mean: {y_test.mean():.4f}")
        
        # Verify means match saved values (within tolerance)
        saved_train_mean = self.split_data['split_params']['y_train_mean']
        saved_test_mean = self.split_data['split_params']['y_test_mean']
        
        if abs(y_train.mean() - saved_train_mean) > 0.01:
            logger.warning(f"Train mean mismatch: {y_train.mean():.4f} vs saved {saved_train_mean:.4f}")
        if abs(y_test.mean() - saved_test_mean) > 0.01:
            logger.warning(f"Test mean mismatch: {y_test.mean():.4f} vs saved {saved_test_mean:.4f}")
            
        return X_train, X_test, y_train, y_test
    
    def get_split_indices(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get integer indices for train/test split.
        
        Parameters
        ----------
        data : pd.DataFrame
            Feature data
            
        Returns
        -------
        train_idx, test_idx
            Integer arrays of train and test indices
        """
        if self.split_data is None:
            self.load_split()
            
        train_companies = self.split_data['train_companies']
        test_companies = self.split_data['test_companies']
        
        # Convert company names to integer indices
        if data.index.name == 'issuer_name' or isinstance(data.index[0], str):
            train_idx = np.array([i for i, idx in enumerate(data.index) 
                                 if idx in train_companies])
            test_idx = np.array([i for i, idx in enumerate(data.index) 
                                if idx in test_companies])
        else:
            # If using integer index, need to map back to companies
            raise NotImplementedError("Integer index mapping not yet implemented")
            
        return train_idx, test_idx


def get_or_create_split(data: pd.DataFrame,
                       target: pd.Series,
                       test_size: float = 0.2,
                       random_state: int = 42,
                       stratify_column: Optional[str] = 'sector',
                       force_recreate: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get existing split or create new one if it doesn't exist.
    
    This is the main function that should be used by all models to ensure
    consistent train/test splits.
    
    Parameters
    ----------
    data : pd.DataFrame
        Feature data
    target : pd.Series
        Target variable
    test_size : float
        Test set proportion (only used if creating new split)
    random_state : int
        Random seed (only used if creating new split)
    stratify_column : str
        Column for stratification (only used if creating new split)
    force_recreate : bool
        If True, create new split even if one exists
        
    Returns
    -------
    X_train, X_test, y_train, y_test
        Split datasets
    """
    splitter = UnifiedTrainTestSplit()
    
    # Detect if we should use indices or company names
    use_indices = isinstance(data.index[0], (int, np.integer))
    
    if force_recreate or not splitter.split_file.exists():
        logger.info("Creating new train/test split")
        splitter.create_split(data, target, test_size, random_state, stratify_column, use_indices)
    else:
        logger.info("Using existing train/test split")
        
    return splitter.apply_split(data, target)