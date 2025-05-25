#!/usr/bin/env python3
"""
Improved categorical feature handling for tree-based models.
This script demonstrates how to better leverage categorical information.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
from config import settings

def identify_categorical_features(df):
    """
    Identify original categorical features from one-hot encoded columns.
    
    Returns:
        dict: mapping from original categorical feature to its one-hot columns
    """
    categorical_groups = {}
    
    # Define patterns for categorical feature groups
    patterns = {
        'gics_sector': 'gics_sector_',
        'gics_industry_group': 'gics_industry_group_',
        'gics_industry': 'gics_industry_',
        'gics_subindustry': 'gics_subindustry_',
        'issuer_cntry_domicile': 'issuer_cntry_domicile_name_',
        'cntry_of_risk': 'cntry_of_risk_',
        'issuer_rating': 'issuer_rating_',
        'crncy': 'crncy_',
    }
    
    for feature_name, pattern in patterns.items():
        cols = [col for col in df.columns if col.startswith(pattern)]
        if cols:
            categorical_groups[feature_name] = cols
            print(f"Found {feature_name}: {len(cols)} categories")
    
    return categorical_groups

def reconstruct_categorical_features(df, categorical_groups):
    """
    Reconstruct original categorical features from one-hot encoded columns.
    
    Args:
        df: DataFrame with one-hot encoded features
        categorical_groups: Dict mapping feature names to their one-hot columns
        
    Returns:
        DataFrame with reconstructed categorical features
    """
    df_cat = df.copy()
    
    for feature_name, onehot_cols in categorical_groups.items():
        # Check if all one-hot columns exist in dataframe
        existing_cols = [col for col in onehot_cols if col in df.columns]
        
        if existing_cols:
            # Reconstruct categorical feature
            categorical_values = []
            for idx in range(len(df)):
                # Find which one-hot column is 1
                active_cols = [col for col in existing_cols if df.iloc[idx][col] == 1]
                if len(active_cols) == 1:
                    # Extract category name from column name
                    category = active_cols[0].replace(f"{existing_cols[0].split('_')[0]}_", "")
                    if len(existing_cols[0].split('_')) > 2:
                        # Handle multi-part prefixes like 'issuer_cntry_domicile_name_'
                        prefix_parts = len(existing_cols[0].split('_')) - 1
                        category = '_'.join(active_cols[0].split('_')[prefix_parts:])
                    categorical_values.append(category)
                elif len(active_cols) == 0:
                    categorical_values.append('Unknown')
                else:
                    # Multiple 1s - take first one
                    category = active_cols[0].replace(f"{existing_cols[0].split('_')[0]}_", "")
                    categorical_values.append(category)
            
            df_cat[feature_name] = categorical_values
            print(f"Reconstructed {feature_name}: {df_cat[feature_name].nunique()} unique values")
    
    return df_cat

def create_improved_catboost_features(df):
    """
    Create improved feature set for CatBoost with native categorical handling.
    
    Args:
        df: Original dataframe
        
    Returns:
        tuple: (X_numerical, X_categorical, categorical_feature_names)
    """
    # Identify categorical groups
    categorical_groups = identify_categorical_features(df)
    
    # Reconstruct categorical features
    df_with_cats = reconstruct_categorical_features(df, categorical_groups)
    
    # Separate numerical and categorical features
    numerical_cols = []
    categorical_cols = list(categorical_groups.keys())
    
    # Get all one-hot columns to exclude
    onehot_cols_to_exclude = []
    for cols in categorical_groups.values():
        onehot_cols_to_exclude.extend(cols)
    
    # Numerical features = all columns except one-hot encoded and reconstructed categorical
    for col in df.columns:
        if col not in onehot_cols_to_exclude and col not in categorical_cols and col != 'issuer_name':
            numerical_cols.append(col)
    
    print(f"Numerical features: {len(numerical_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Excluded one-hot features: {len(onehot_cols_to_exclude)}")
    
    # Create feature matrices
    X_numerical = df_with_cats[numerical_cols]
    X_categorical = df_with_cats[categorical_cols]
    
    return X_numerical, X_categorical, categorical_cols

def improved_catboost_model():
    """
    Example of how to use CatBoost with native categorical features.
    """
    from catboost import CatBoostRegressor
    from sklearn.model_selection import train_test_split
    
    # Load data
    df = pd.read_csv(settings.RAW_DATA_DIR / 'combined_df_for_ml_models.csv')
    
    # Create improved features
    X_numerical, X_categorical, cat_feature_names = create_improved_catboost_features(df)
    
    # Combine features
    X_combined = pd.concat([X_numerical, X_categorical], axis=1)
    
    # For demonstration, create dummy target (replace with actual target)
    y = np.random.randn(len(X_combined))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42
    )
    
    # Get categorical feature indices
    cat_feature_indices = [X_combined.columns.get_loc(col) for col in cat_feature_names]
    
    print(f"Categorical feature indices: {cat_feature_indices}")
    
    # Train CatBoost with native categorical handling
    model = CatBoostRegressor(
        iterations=100,
        cat_features=cat_feature_indices,  # Specify categorical features
        verbose=False,
        random_seed=42
    )
    
    model.fit(X_train, y_train)
    
    print("Model trained successfully with native categorical features!")
    
    # Feature importance will now show categorical features properly
    feature_importance = model.get_feature_importance()
    feature_names = X_combined.columns
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))
    
    return model, importance_df

if __name__ == "__main__":
    # Run example
    model, importance = improved_catboost_model()