#!/usr/bin/env python3
"""
Create an EDA-enhanced version of the combined dataset with both 
categorical and one-hot encoded features.
"""

import pandas as pd
from pathlib import Path

def create_eda_enhanced_dataset():
    """
    Create enhanced versions of the combined dataset for easier EDA.
    
    Creates two files:
    1. combined_df_for_ml_models_enhanced.csv - Original + categorical columns
    2. combined_df_for_eda.csv - Streamlined version for EDA
    """
    
    print("Creating EDA-enhanced datasets...")
    
    # Paths
    raw_dir = Path('data/raw')
    processed_dir = Path('data/processed')
    
    # Load the original combined dataset
    print("Loading original combined dataset...")
    df_combined = pd.read_csv(raw_dir / 'combined_df_for_ml_models.csv')
    print(f"Original dataset shape: {df_combined.shape}")
    
    # Load the tree models dataset which has categorical columns preserved
    print("Loading tree models dataset with categorical features...")
    df_tree = pd.read_csv(processed_dir / 'tree_models_dataset.csv')
    
    # Categorical columns to add
    categorical_cols = [
        'gics_sector', 
        'gics_sub_ind', 
        'issuer_cntry_domicile',
        'cntry_of_risk', 
        'top_1_shareholder_location',
        'top_2_shareholder_location', 
        'top_3_shareholder_location'
    ]
    
    # Set index to align data
    df_combined.set_index('issuer_name', inplace=True)
    df_tree.set_index('issuer_name', inplace=True)
    
    # Ensure we have matching indices
    common_index = df_combined.index.intersection(df_tree.index)
    print(f"Common companies: {len(common_index)}")
    
    # Add categorical columns to the combined dataset with 'cat_' prefix
    print("\nAdding categorical columns...")
    for col in categorical_cols:
        if col in df_tree.columns:
            df_combined[f'cat_{col}'] = df_tree.loc[common_index, col]
            print(f"  Added cat_{col}")
    
    # Reset index
    df_combined.reset_index(inplace=True)
    
    # Save enhanced dataset (Option 1)
    enhanced_path = raw_dir / 'combined_df_for_ml_models_enhanced.csv'
    df_combined.to_csv(enhanced_path, index=False)
    print(f"\nSaved enhanced dataset: {enhanced_path}")
    print(f"Shape: {df_combined.shape}")
    
    # Create streamlined EDA version (Option 2)
    print("\nCreating streamlined EDA version...")
    
    # Select columns for EDA version
    # Start with non one-hot encoded columns
    non_onehot_cols = ['issuer_name']
    
    # Add all columns that aren't one-hot encoded
    one_hot_patterns = [
        '_name_', 'issuer_cntry_domicile_', 'cntry_of_risk_', 
        'gics_sector_', 'gics_sub_ind_', '_shareholder_location_'
    ]
    
    for col in df_combined.columns:
        if col != 'issuer_name' and not any(pattern in col for pattern in one_hot_patterns):
            non_onehot_cols.append(col)
    
    # Add the categorical columns
    cat_cols = [col for col in df_combined.columns if col.startswith('cat_')]
    
    # Combine for EDA dataset
    eda_cols = non_onehot_cols + cat_cols
    df_eda = df_combined[eda_cols].copy()
    
    # Save EDA version
    eda_path = raw_dir / 'combined_df_for_eda.csv'
    df_eda.to_csv(eda_path, index=False)
    print(f"\nSaved EDA dataset: {eda_path}")
    print(f"Shape: {df_eda.shape}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Original dataset: {df_combined.shape[0]} rows × {df_combined.shape[1] - len(categorical_cols)} columns")
    print(f"Enhanced dataset: {df_combined.shape[0]} rows × {df_combined.shape[1]} columns")
    print(f"EDA dataset:      {df_eda.shape[0]} rows × {df_eda.shape[1]} columns")
    print(f"\nReduction: {df_combined.shape[1] - df_eda.shape[1]} one-hot columns removed for EDA")
    
    # Show categorical columns info
    print("\nCategorical columns added:")
    for col in cat_cols:
        n_unique = df_combined[col].nunique()
        n_missing = df_combined[col].isna().sum()
        print(f"  {col}: {n_unique} unique values, {n_missing} missing")
    
    # Example usage
    print("\n" + "="*60)
    print("EXAMPLE USAGE")
    print("="*60)
    print("""
# For EDA - use the streamlined version:
df = pd.read_csv('data/raw/combined_df_for_eda.csv')

# Simple sector analysis
df.groupby('cat_gics_sector')['esg_score'].agg(['mean', 'std', 'count'])

# Country distribution
df['cat_cntry_of_risk'].value_counts()

# For ML - use the enhanced version with both representations:
df = pd.read_csv('data/raw/combined_df_for_ml_models_enhanced.csv')
""")

if __name__ == "__main__":
    create_eda_enhanced_dataset()