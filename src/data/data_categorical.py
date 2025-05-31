"""Data loading utilities for categorical datasets."""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional

from src.config.settings import PROCESSED_DATA_DIR


def load_tree_models_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load tree models dataset with native categorical features.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Features dataframe and target series
    """
    # Load tree models dataset
    tree_data_path = PROCESSED_DATA_DIR / 'tree_models_dataset.csv'
    
    if not tree_data_path.exists():
        raise FileNotFoundError(
            f"Tree models dataset not found at {tree_data_path}. "
            "Run create_categorical_datasets.py first."
        )
    
    # Load data
    df = pd.read_csv(tree_data_path)
    
    # Load scores
    from data import load_scores_data
    scores = load_scores_data()
    
    # Set issuer_name as index for both datasets
    if 'issuer_name' in df.columns:
        df = df.set_index('issuer_name')
    if scores.index.name != 'issuer_name' and 'issuer_name' in scores.columns:
        scores = scores.set_index('issuer_name')
    
    # Align indices using issuer names
    common_indices = df.index.intersection(scores.index)
    features = df.loc[common_indices].copy()
    target = scores.loc[common_indices].copy()
    
    # Ensure categorical columns are properly typed
    categorical_features = get_categorical_features()
    for cat_feature in categorical_features:
        if cat_feature in features.columns:
            features[cat_feature] = features[cat_feature].astype('category')
    
    print(f"Loaded tree models data: {features.shape} features, {len(target)} samples")
    print(f"Categorical features: {categorical_features}")
    
    return features, target


def load_linear_models_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load linear models dataset with one-hot encoded features.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Features dataframe and target series
    """
    # Load linear models dataset
    linear_data_path = PROCESSED_DATA_DIR / 'linear_models_dataset.csv'
    
    if not linear_data_path.exists():
        raise FileNotFoundError(
            f"Linear models dataset not found at {linear_data_path}. "
            "Run create_categorical_datasets.py first."
        )
    
    # Load data
    df = pd.read_csv(linear_data_path)
    
    # Load scores
    from data import load_scores_data
    scores = load_scores_data()
    
    # Set issuer_name as index for both datasets
    if 'issuer_name' in df.columns:
        df = df.set_index('issuer_name')
    if scores.index.name != 'issuer_name' and 'issuer_name' in scores.columns:
        scores = scores.set_index('issuer_name')
    
    # Align indices using issuer names
    common_indices = df.index.intersection(scores.index)
    features = df.loc[common_indices].copy()
    target = scores.loc[common_indices].copy()
    
    print(f"Loaded linear models data: {features.shape} features, {len(target)} samples")
    
    return features, target


def get_categorical_features() -> List[str]:
    """
    Get list of categorical feature names.
    
    Returns
    -------
    List[str]
        List of categorical feature column names
    """
    metadata_path = PROCESSED_DATA_DIR / 'datasets_metadata.json'
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        categorical_features = metadata.get('tree_models', {}).get('categorical_features', [])
        return categorical_features
    else:
        # Fallback to default categorical features
        return [
            'gics_sector',
            'gics_sub_ind', 
            'issuer_cntry_domicile',
            'cntry_of_risk',
            'top_1_shareholder_location',
            'top_2_shareholder_location',
            'top_3_shareholder_location'
        ]


def get_categorical_mappings() -> Optional[Dict]:
    """
    Get categorical feature mappings (original one-hot columns for each categorical feature).
    
    Returns
    -------
    Optional[Dict]
        Dictionary mapping categorical feature names to their one-hot columns
    """
    mappings_path = PROCESSED_DATA_DIR / 'categorical_mappings.pkl'
    
    if mappings_path.exists():
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        return mappings
    else:
        print("Warning: Categorical mappings file not found")
        return None


def get_quantitative_features(df: pd.DataFrame) -> List[str]:
    """
    Get list of quantitative feature names from a dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    List[str]
        List of quantitative feature column names
    """
    categorical_features = get_categorical_features()
    
    # Filter out categorical features and target-related columns
    exclude_patterns = ['issuer_name', 'score', 'target']
    quantitative_features = []
    
    for col in df.columns:
        # Skip categorical features
        if col in categorical_features:
            continue
        
        # Skip columns that match exclude patterns
        if any(pattern in col.lower() for pattern in exclude_patterns):
            continue
        
        # Add remaining columns as quantitative
        quantitative_features.append(col)
    
    return quantitative_features


def get_base_and_yeo_features_categorical(use_tree_data: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get base and Yeo-Johnson transformed features for categorical datasets.
    
    Parameters
    ----------
    use_tree_data : bool
        If True, use tree models data. If False, use linear models data.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Base features and Yeo-Johnson transformed features
    """
    if use_tree_data:
        features, _ = load_tree_models_data()
    else:
        features, _ = load_linear_models_data()
    
    # Get quantitative features
    quantitative_features = get_quantitative_features(features)
    categorical_features = get_categorical_features()
    
    # Separate base and Yeo-Johnson features
    base_quant_features = [col for col in quantitative_features if not col.startswith('yeo_joh_')]
    yeo_quant_features = [col for col in quantitative_features if col.startswith('yeo_joh_')]
    
    # Create base dataset (base quantitative + categorical)
    base_features = features[base_quant_features + categorical_features].copy()
    
    # Create Yeo-Johnson dataset (Yeo quantitative + categorical)
    yeo_features = features[yeo_quant_features + categorical_features].copy()
    
    print(f"Base features: {base_features.shape} ({len(base_quant_features)} quantitative + {len(categorical_features)} categorical)")
    print(f"Yeo features: {yeo_features.shape} ({len(yeo_quant_features)} quantitative + {len(categorical_features)} categorical)")
    
    return base_features, yeo_features


def add_random_feature_categorical(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Add a random feature to the categorical dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    random_state : int
        Random state for reproducibility
        
    Returns
    -------
    pd.DataFrame
        Dataframe with added random feature
    """
    np.random.seed(random_state)
    df_with_random = df.copy()
    df_with_random['random_feature'] = np.random.normal(0, 1, len(df))
    
    print(f"Added random feature. Dataset shape: {df_with_random.shape}")
    return df_with_random


def load_datasets_metadata() -> Dict:
    """
    Load metadata about the categorical datasets.
    
    Returns
    -------
    Dict
        Metadata dictionary
    """
    metadata_path = PROCESSED_DATA_DIR / 'datasets_metadata.json'
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    else:
        return {}


def print_categorical_summary():
    """Print summary of categorical datasets."""
    try:
        metadata = load_datasets_metadata()
        
        print("=" * 60)
        print("CATEGORICAL DATASETS SUMMARY")
        print("=" * 60)
        
        if 'tree_models' in metadata:
            tree_meta = metadata['tree_models']
            print(f"Tree Models Dataset:")
            print(f"  Shape: {tree_meta.get('tree_dataset_shape', 'Unknown')}")
            print(f"  Categorical features: {len(tree_meta.get('categorical_features', []))}")
            print(f"  Features reduced: {tree_meta.get('features_reduced', 'Unknown')}")
        
        if 'linear_models' in metadata:
            linear_meta = metadata['linear_models']
            print(f"Linear Models Dataset:")
            print(f"  Shape: {linear_meta.get('dataset_shape', 'Unknown')}")
            print(f"  Encoding: {linear_meta.get('encoding_type', 'Unknown')}")
        
        categorical_features = get_categorical_features()
        print(f"\nCategorical Features ({len(categorical_features)}):")
        for i, feature in enumerate(categorical_features, 1):
            print(f"  {i}. {feature}")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"Error loading categorical summary: {e}")


if __name__ == "__main__":
    # Print summary of categorical datasets
    print_categorical_summary()
    
    # Test loading functions
    try:
        print("\nTesting tree models data loading...")
        tree_features, tree_target = load_tree_models_data()
        
        print("\nTesting linear models data loading...")
        linear_features, linear_target = load_linear_models_data()
        
        print("\nTesting base and Yeo features...")
        base_tree, yeo_tree = get_base_and_yeo_features_categorical(use_tree_data=True)
        
        print("\n✅ All data loading functions working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing data loading: {e}")