#!/usr/bin/env python3
"""
Create a unified data pipeline to ensure all models use the same train/test split.
This solves the inconsistency where different models have different test set sizes.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import sys

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings
from data import load_features_data, load_scores_data


def create_fixed_train_test_split(features_df: pd.DataFrame, 
                                 scores: pd.Series,
                                 test_size: float = 0.2,
                                 random_state: int = 42) -> Dict:
    """
    Create a fixed train/test split that will be used by all models.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        The feature dataframe
    scores : pd.Series
        The target scores
    test_size : float
        Proportion of data for test set
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing train/test indices and metadata
    """
    print("Creating fixed train/test split...")
    
    # Ensure aligned indices
    common_indices = features_df.index.intersection(scores.index)
    features_aligned = features_df.loc[common_indices]
    scores_aligned = scores.loc[common_indices]
    
    print(f"Total samples after alignment: {len(common_indices)}")
    
    # Perform stratified split by sector if available
    stratify_col = None
    sector_columns = [col for col in features_aligned.columns 
                     if col.startswith('gics_sector_') or col == 'gics_sector']
    
    if sector_columns:
        # For one-hot encoded sectors, create a single label
        if len(sector_columns) > 1:  # One-hot encoded
            sector_labels = np.zeros(len(features_aligned), dtype=int)
            for i, col in enumerate(sector_columns):
                sector_labels[features_aligned[col] == 1] = i
            stratify_col = sector_labels
        else:  # Single categorical column
            stratify_col = features_aligned[sector_columns[0]]
        
        print(f"Using stratified split by sector")
    
    # Create train/test split
    indices = np.arange(len(features_aligned))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )
    
    # Store split information
    split_info = {
        'train_indices': train_idx.tolist(),
        'test_indices': test_idx.tolist(),
        'train_size': len(train_idx),
        'test_size': len(test_idx),
        'total_size': len(features_aligned),
        'test_proportion': test_size,
        'random_state': random_state,
        'stratified': stratify_col is not None,
        'index_mapping': {i: idx for i, idx in enumerate(common_indices)},
        'feature_columns': features_aligned.columns.tolist()
    }
    
    print(f"Train size: {split_info['train_size']}")
    print(f"Test size: {split_info['test_size']}")
    
    return split_info


def create_unified_datasets() -> None:
    """
    Create unified datasets for all model types with consistent train/test splits.
    """
    print("="*60)
    print("Creating Unified Datasets for All Models")
    print("="*60)
    
    # Create output directory
    unified_dir = settings.PROCESSED_DATA_DIR / "unified"
    unified_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original data
    print("\n1. Loading original data...")
    features_df = load_features_data()
    scores = load_scores_data()
    
    print(f"Original features shape: {features_df.shape}")
    print(f"Original scores length: {len(scores)}")
    
    # Create fixed split
    print("\n2. Creating fixed train/test split...")
    split_info = create_fixed_train_test_split(
        features_df, 
        scores,
        test_size=settings.LINEAR_REGRESSION_PARAMS['test_size'],
        random_state=settings.LINEAR_REGRESSION_PARAMS['random_state']
    )
    
    # Save split information
    split_path = unified_dir / "train_test_split.pkl"
    with open(split_path, 'wb') as f:
        pickle.dump(split_info, f)
    print(f"Saved split info to: {split_path}")
    
    # Create dataset for linear models (one-hot encoded)
    print("\n3. Creating linear models dataset...")
    linear_features = features_df.copy()
    linear_target = scores.copy()
    
    # Ensure proper alignment
    common_idx = linear_features.index.intersection(linear_target.index)
    linear_features = linear_features.loc[common_idx]
    linear_target = linear_target.loc[common_idx]
    
    # Save linear dataset
    linear_path = unified_dir / "linear_models_unified.csv"
    linear_features.to_csv(linear_path)
    print(f"Saved linear features to: {linear_path}")
    
    # Create dataset for tree models (with categorical features)
    print("\n4. Creating tree models dataset...")
    
    # Identify categorical columns
    categorical_patterns = ['gics_sector', 'country', 'exchange', 'industry']
    categorical_columns = []
    
    # Convert one-hot encoded columns back to categorical
    tree_features = features_df.copy()
    
    for pattern in categorical_patterns:
        # Find one-hot encoded columns
        one_hot_cols = [col for col in tree_features.columns if col.startswith(f"{pattern}_")]
        
        if one_hot_cols:
            print(f"Converting {len(one_hot_cols)} one-hot columns for {pattern}")
            
            # Create categorical column from one-hot
            cat_values = []
            for idx in tree_features.index:
                # Find which one-hot column is 1
                active_cols = [col for col in one_hot_cols if tree_features.loc[idx, col] == 1]
                if active_cols:
                    # Extract category name from column name
                    cat_values.append(active_cols[0].replace(f"{pattern}_", ""))
                else:
                    cat_values.append("Unknown")
            
            # Add categorical column
            tree_features[pattern] = pd.Categorical(cat_values)
            categorical_columns.append(pattern)
            
            # Drop one-hot columns
            tree_features = tree_features.drop(columns=one_hot_cols)
    
    print(f"Categorical columns: {categorical_columns}")
    
    # Save tree dataset
    tree_path = unified_dir / "tree_models_unified.csv"
    tree_features.to_csv(tree_path)
    print(f"Saved tree features to: {tree_path}")
    
    # Save categorical features list
    cat_features_path = unified_dir / "categorical_features.json"
    with open(cat_features_path, 'w') as f:
        json.dump(categorical_columns, f)
    print(f"Saved categorical features list to: {cat_features_path}")
    
    # Save metadata
    metadata = {
        'linear_dataset': {
            'path': str(linear_path),
            'shape': linear_features.shape,
            'features': linear_features.columns.tolist()
        },
        'tree_dataset': {
            'path': str(tree_path),
            'shape': tree_features.shape,
            'features': tree_features.columns.tolist(),
            'categorical_features': categorical_columns
        },
        'split_info': {
            'train_size': split_info['train_size'],
            'test_size': split_info['test_size'],
            'total_size': split_info['total_size']
        }
    }
    
    metadata_path = unified_dir / "unified_datasets_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata to: {metadata_path}")
    
    print("\n" + "="*60)
    print("Unified datasets created successfully!")
    print(f"All models will now use the same {split_info['test_size']} test samples")
    print("="*60)


def load_unified_data(model_type: str = 'linear') -> Tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
    """
    Load unified data with fixed train/test split.
    
    Parameters
    ----------
    model_type : str
        'linear' for one-hot encoded data, 'tree' for categorical data
        
    Returns
    -------
    features : pd.DataFrame
        Feature dataframe
    target : pd.Series
        Target series
    train_idx : np.ndarray
        Training indices
    test_idx : np.ndarray
        Test indices
    """
    unified_dir = settings.PROCESSED_DATA_DIR / "unified"
    
    # Load split info
    with open(unified_dir / "train_test_split.pkl", 'rb') as f:
        split_info = pickle.load(f)
    
    # Load appropriate dataset
    if model_type == 'linear':
        features = pd.read_csv(unified_dir / "linear_models_unified.csv", index_col=0)
    else:
        features = pd.read_csv(unified_dir / "tree_models_unified.csv", index_col=0)
        
        # Convert categorical columns
        with open(unified_dir / "categorical_features.json", 'r') as f:
            cat_features = json.load(f)
        
        for cat_col in cat_features:
            if cat_col in features.columns:
                features[cat_col] = features[cat_col].astype('category')
    
    # Load target
    target = load_scores_data()
    
    # Align with features
    common_idx = features.index.intersection(target.index)
    features = features.loc[common_idx]
    target = target.loc[common_idx]
    
    # Get train/test indices
    train_idx = np.array(split_info['train_indices'])
    test_idx = np.array(split_info['test_indices'])
    
    return features, target, train_idx, test_idx


if __name__ == "__main__":
    create_unified_datasets()