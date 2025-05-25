#!/usr/bin/env python3
"""
Create separate datasets for tree models (native categorical) and linear models (one-hot encoded).

This script reads the current one-hot encoded dataset and creates:
1. Tree models dataset: Reconstructed categorical features for XGBoost, CatBoost, LightGBM
2. Linear models dataset: Current one-hot encoded features for ElasticNet, Linear Regression
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Import project modules
from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.io import ensure_dir


def identify_categorical_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify one-hot encoded categorical feature groups in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with one-hot encoded features
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping categorical feature names to their one-hot columns
    """
    categorical_groups = {}
    
    # Define categorical feature patterns
    patterns = {
        'gics_sector': 'gics_sector_',
        'gics_sub_ind': 'gics_sub_ind_',
        'issuer_cntry_domicile': 'issuer_cntry_domicile_name_',
        'cntry_of_risk': 'cntry_of_risk_',
        'top_1_shareholder_location': 'top_1_shareholder_location_',
        'top_2_shareholder_location': 'top_2_shareholder_location_',
        'top_3_shareholder_location': 'top_3_shareholder_location_'
    }
    
    for feature_name, pattern in patterns.items():
        cols = [col for col in df.columns if col.startswith(pattern)]
        if cols:
            categorical_groups[feature_name] = cols
            print(f"Found {len(cols)} categories for {feature_name}")
    
    return categorical_groups


def reconstruct_categorical_features(df: pd.DataFrame, categorical_groups: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Reconstruct categorical features from one-hot encoded columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with one-hot encoded features
    categorical_groups : Dict[str, List[str]]
        Dictionary mapping categorical feature names to their one-hot columns
        
    Returns
    -------
    pd.DataFrame
        Dataframe with reconstructed categorical features
    """
    df_categorical = df.copy()
    
    for feature_name, one_hot_cols in categorical_groups.items():
        # Create categorical feature from one-hot encoded columns
        categorical_values = []
        
        for idx in range(len(df)):
            # Find which one-hot column is 1 for this row
            active_cols = [col for col in one_hot_cols if df.iloc[idx][col] == 1]
            
            if len(active_cols) == 1:
                # Extract category name from column name
                category = active_cols[0].replace(f"{feature_name}_", "").replace("name_", "")
                categorical_values.append(category)
            elif len(active_cols) == 0:
                # No category selected (missing value)
                categorical_values.append(np.nan)
            else:
                # Multiple categories selected (shouldn't happen with proper one-hot encoding)
                print(f"Warning: Multiple categories found for {feature_name} at row {idx}")
                category = active_cols[0].replace(f"{feature_name}_", "").replace("name_", "")
                categorical_values.append(category)
        
        # Add reconstructed categorical feature
        df_categorical[feature_name] = pd.Categorical(categorical_values)
        
        # Remove one-hot encoded columns
        df_categorical = df_categorical.drop(columns=one_hot_cols)
        
        print(f"Reconstructed {feature_name}: {df_categorical[feature_name].nunique()} unique categories")
    
    return df_categorical


def create_tree_models_dataset(input_path: str, output_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Create dataset for tree models with native categorical features.
    
    Parameters
    ----------
    input_path : str
        Path to input CSV file with one-hot encoded features
    output_path : str
        Path to save tree models dataset
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        Tree models dataframe and metadata dictionary
    """
    print("Creating tree models dataset...")
    
    # Read original dataset
    df = pd.read_csv(input_path)
    print(f"Original dataset shape: {df.shape}")
    
    # Identify categorical groups
    categorical_groups = identify_categorical_groups(df)
    
    # Reconstruct categorical features
    df_tree = reconstruct_categorical_features(df, categorical_groups)
    
    # Ensure categorical columns are properly typed
    for feature_name in categorical_groups.keys():
        if feature_name in df_tree.columns:
            df_tree[feature_name] = df_tree[feature_name].astype('category')
    
    print(f"Tree models dataset shape: {df_tree.shape}")
    print(f"Reduced features by: {df.shape[1] - df_tree.shape[1]} columns")
    
    # Save dataset
    ensure_dir(Path(output_path).parent)
    df_tree.to_csv(output_path, index=False)
    print(f"Saved tree models dataset to: {output_path}")
    
    # Create metadata
    metadata = {
        'dataset_type': 'tree_models',
        'categorical_features': list(categorical_groups.keys()),
        'categorical_groups': categorical_groups,
        'original_shape': df.shape,
        'tree_dataset_shape': df_tree.shape,
        'features_reduced': df.shape[1] - df_tree.shape[1]
    }
    
    return df_tree, metadata


def create_linear_models_dataset(input_path: str, output_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Create dataset for linear models with one-hot encoded features.
    
    Parameters
    ----------
    input_path : str
        Path to input CSV file with one-hot encoded features
    output_path : str
        Path to save linear models dataset
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        Linear models dataframe and metadata dictionary
    """
    print("Creating linear models dataset...")
    
    # Read original dataset (keep as-is for linear models)
    df = pd.read_csv(input_path)
    print(f"Linear models dataset shape: {df.shape}")
    
    # Save dataset (copy of original)
    ensure_dir(Path(output_path).parent)
    df.to_csv(output_path, index=False)
    print(f"Saved linear models dataset to: {output_path}")
    
    # Create metadata
    metadata = {
        'dataset_type': 'linear_models',
        'encoding_type': 'one_hot',
        'dataset_shape': df.shape,
        'note': 'Original one-hot encoded dataset for ElasticNet and Linear Regression'
    }
    
    return df, metadata


def save_metadata_and_mappings(tree_metadata: Dict, linear_metadata: Dict, output_dir: str):
    """
    Save metadata and feature mappings for both datasets.
    
    Parameters
    ----------
    tree_metadata : Dict
        Metadata for tree models dataset
    linear_metadata : Dict
        Metadata for linear models dataset
    output_dir : str
        Directory to save metadata files
    """
    ensure_dir(Path(output_dir))
    
    # Save combined metadata
    combined_metadata = {
        'tree_models': tree_metadata,
        'linear_models': linear_metadata,
        'creation_timestamp': pd.Timestamp.now().isoformat(),
        'description': 'Separate datasets for tree-based and linear models with optimal feature encoding'
    }
    
    metadata_path = Path(output_dir) / 'datasets_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(combined_metadata, f, indent=2, default=str)
    print(f"Saved metadata to: {metadata_path}")
    
    # Save categorical mappings for tree models
    if 'categorical_groups' in tree_metadata:
        mappings_path = Path(output_dir) / 'categorical_mappings.pkl'
        with open(mappings_path, 'wb') as f:
            pickle.dump(tree_metadata['categorical_groups'], f)
        print(f"Saved categorical mappings to: {mappings_path}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Creating Categorical Datasets for Tree and Linear Models")
    print("=" * 60)
    
    # Define paths
    input_path = RAW_DATA_DIR / 'combined_df_for_ml_models.csv'
    tree_output_path = PROCESSED_DATA_DIR / 'tree_models_dataset.csv'
    linear_output_path = PROCESSED_DATA_DIR / 'linear_models_dataset.csv'
    
    # Create datasets
    df_tree, tree_metadata = create_tree_models_dataset(str(input_path), str(tree_output_path))
    df_linear, linear_metadata = create_linear_models_dataset(str(input_path), str(linear_output_path))
    
    # Save metadata
    save_metadata_and_mappings(tree_metadata, linear_metadata, str(PROCESSED_DATA_DIR))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original dataset: {input_path}")
    print(f"Tree models dataset: {tree_output_path} ({df_tree.shape})")
    print(f"Linear models dataset: {linear_output_path} ({df_linear.shape})")
    print(f"Categorical features reconstructed: {len(tree_metadata.get('categorical_features', []))}")
    print(f"Features reduced for tree models: {tree_metadata.get('features_reduced', 0)}")
    print("\nCategorical features in tree dataset:")
    for feature in tree_metadata.get('categorical_features', []):
        unique_count = df_tree[feature].nunique() if feature in df_tree.columns else 0
        print(f"  - {feature}: {unique_count} categories")
    print("=" * 60)


if __name__ == "__main__":
    main()