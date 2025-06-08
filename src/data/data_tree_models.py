#!/usr/bin/env python3
"""
Data loading utilities specifically for tree models using the pre-formatted CSV.
This module provides simplified data loading for tree-based models (XGBoost, LightGBM, CatBoost)
using the combined_df_for_tree_models.csv file with native categorical features.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, List
from sklearn.model_selection import train_test_split

from src.config import settings


def load_tree_models_from_csv() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Load tree model data from pre-formatted CSV with categorical features.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, List[str]]
        Features dataframe, target series, and list of categorical column names
    """
    # Load the pre-formatted tree models CSV
    csv_path = Path(settings.DATA_DIR) / 'raw' / 'combined_df_for_tree_models.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Tree models CSV not found at {csv_path}. "
            "Please ensure combined_df_for_tree_models.csv exists."
        )
    
    print(f"Loading tree models data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Load target scores
    scores = load_scores_data()
    
    # Align indices using issuer names
    if 'issuer_name' in df.columns:
        df = df.set_index('issuer_name')
    
    # Ensure scores has issuer_name as index
    if scores.index.name != 'issuer_name' and 'issuer_name' in scores.columns:
        scores = scores.set_index('issuer_name')
    
    # Get common indices
    common_indices = df.index.intersection(scores.index)
    X = df.loc[common_indices].copy()
    
    # Extract target (handle both Series and DataFrame)
    if isinstance(scores, pd.DataFrame) and 'esg_score' in scores.columns:
        y = scores.loc[common_indices, 'esg_score']
    else:
        y = scores.loc[common_indices]
    
    # If y is still a DataFrame, extract the first column
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    
    # Identify categorical features (object dtype columns)
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Remove 'issuer_name' if it's in categorical features
    categorical_features = [col for col in categorical_features if col != 'issuer_name']
    
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # Convert categorical features to category dtype for tree models
    for col in categorical_features:
        X[col] = X[col].astype('category')
    
    return X, y, categorical_features


def get_base_and_yeo_features_tree() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get base and Yeo-Johnson transformed features for tree models.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Base features dataframe and Yeo features dataframe
    """
    # Load full data
    X, _, categorical_features = load_tree_models_from_csv()
    
    # Identify base and yeo columns
    # Base columns: non-yeo numeric columns + categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    base_numeric = [col for col in numeric_cols if not col.startswith('yeo_joh_')]
    yeo_numeric = [col for col in numeric_cols if col.startswith('yeo_joh_')]
    
    # Create base features (original numeric + categorical)
    base_features = X[base_numeric + categorical_features].copy()
    
    # Create yeo features (transformed numeric + categorical)
    yeo_features = X[yeo_numeric + categorical_features].copy()
    
    print(f"Base features shape: {base_features.shape}")
    print(f"Yeo features shape: {yeo_features.shape}")
    
    return base_features, yeo_features


def get_tree_model_datasets(include_random=True) -> dict:
    """
    Get all dataset variants for tree models.
    
    Parameters
    ----------
    include_random : bool
        Whether to include random feature variants
        
    Returns
    -------
    dict
        Dictionary containing all dataset variants with keys:
        'Base', 'Yeo', 'Base_Random', 'Yeo_Random' (if include_random=True)
    """
    # Load target
    _, y, _ = load_tree_models_from_csv()
    
    # Get base and yeo features
    base_features, yeo_features = get_base_and_yeo_features_tree()
    
    # Create datasets dictionary
    datasets = {
        'Base': base_features,
        'Yeo': yeo_features
    }
    
    # Add random feature variants if requested
    if include_random:
        # Note: add_random_feature from data.py already exists and works
        datasets['Base_Random'] = add_random_feature(base_features.copy())
        datasets['Yeo_Random'] = add_random_feature(yeo_features.copy())
    
    return datasets, y


def get_categorical_features_for_dataset(dataset_name: str, X: pd.DataFrame) -> List[str]:
    """
    Get categorical feature names for a specific dataset variant.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset variant ('Base', 'Yeo', etc.)
    X : pd.DataFrame
        The dataset
        
    Returns
    -------
    List[str]
        List of categorical column names in the dataset
    """
    return X.select_dtypes(include=['object']).columns.tolist()


def perform_stratified_split_for_tree_models(X, y, test_size=0.2, random_state=42):
    """
    Performs a stratified train-test split based on GICS sectors for tree models.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Feature data with 'gics_sector' as a categorical column
    y : pandas.Series
        Target variable (e.g., ESG score)
    test_size : float
        Proportion of the dataset to include in the test split
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    X_train, X_test, y_train, y_test : split data
    """
    # Check if gics_sector column exists
    if 'gics_sector' not in X.columns:
        print("Warning: 'gics_sector' column not found. Performing regular train-test split.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Use gics_sector for stratification
    stratify_col = X['gics_sector']
    
    # Perform stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_col
    )
    
    # Print sector distribution for verification
    print("Sector distribution check:")
    train_dist = X_train['gics_sector'].value_counts(normalize=True).sort_index()
    test_dist = X_test['gics_sector'].value_counts(normalize=True).sort_index()
    
    for sector in train_dist.index:
        train_pct = train_dist.get(sector, 0) * 100
        test_pct = test_dist.get(sector, 0) * 100
        print(f"{sector}: Train {train_pct:.1f}%, Test {test_pct:.1f}%")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Test the data loading
    print("Testing tree model data loading...")
    print("=" * 60)
    
    # Test basic loading
    X, y, cat_features = load_tree_models_from_csv()
    print(f"\nBasic load successful!")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Categorical features: {cat_features}")
    
    # Test dataset variants
    print("\n" + "=" * 60)
    print("Testing dataset variants...")
    datasets, y = get_tree_model_datasets()
    
    for name, data in datasets.items():
        print(f"\n{name} dataset:")
        print(f"  Shape: {data.shape}")
        print(f"  Categorical columns: {get_categorical_features_for_dataset(name, data)}")
        if 'random_feature' in data.columns:
            print(f"  Has random feature: Yes")