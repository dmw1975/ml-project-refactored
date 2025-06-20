#!/usr/bin/env python3
"""
Prepare categorical datasets for XGBoost feature removal analysis.
Creates Base_Random and Yeo_Random datasets with native categorical features.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_categorical_datasets():
    """Prepare categorical datasets for XGBoost."""
    
    # Create output directory
    output_dir = Path('data/pkl')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw data - use the combined tree models data
    logger.info("Loading raw data...")
    tree_data = pd.read_csv('data/raw/combined_df_for_tree_models.csv')
    scores_df = pd.read_csv('data/raw/score.csv')
    
    # Merge with scores
    logger.info("Merging with scores...")
    if 'issuer_name' in tree_data.columns and 'issuer_name' in scores_df.columns:
        data = tree_data.merge(scores_df[['issuer_name', 'esg_score']], on='issuer_name', how='inner')
    else:
        # If issuer_name is index, reset it
        tree_data_reset = tree_data.reset_index() if tree_data.index.name == 'issuer_name' else tree_data
        scores_reset = scores_df.reset_index() if scores_df.index.name == 'issuer_name' else scores_df
        data = tree_data_reset.merge(scores_reset[['issuer_name', 'esg_score']], on='issuer_name', how='inner')
    
    # Define categorical columns
    categorical_columns = [
        'gics_sector', 'gics_sub_ind', 'issuer_cntry_domicile',
        'cntry_of_risk', 'top_1_shareholder_location',
        'top_2_shareholder_location', 'top_3_shareholder_location'
    ]
    
    # Define numeric columns to keep
    numeric_columns = [
        'market_cap_usd', 'net_income_usd', 'hist_pe', 'hist_book_px',
        'hist_fcf_yld', 'hist_ebitda_ev', 'hist_roe', 'hist_roic',
        'hist_roa', 'hist_gross_profit_usd', 'hist_net_chg_lt_debt_usd',
        'hist_net_debt_usd', 'hist_ev_usd', 'hist_asset_turnover',
        'hist_capex_sales', 'hist_capex_depr', 'hist_rd_exp_usd',
        'hist_eps_usd', 'return_usd', 'vola', 'beta',
        'shares_outstanding', 'shares_float',
        'top_1_shareholder_percentage', 'top_2_shareholder_percentage',
        'top_3_shareholder_percentage'
    ]
    
    # Keep only relevant columns
    feature_columns = numeric_columns + categorical_columns
    available_columns = [col for col in feature_columns if col in data.columns]
    
    X = data[available_columns].copy()
    y = data['esg_score'].copy()
    
    # Convert categorical columns to category dtype
    for col in categorical_columns:
        if col in X.columns:
            X[col] = X[col].astype('category')
    
    # Remove rows with missing target
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Categorical columns: {X.select_dtypes(include=['category']).columns.tolist()}")
    logger.info(f"Numeric columns: {X.select_dtypes(include=['number']).columns.tolist()}")
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=X['gics_sector']
    )
    
    # Save Base_Random dataset (original features)
    base_random_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': list(X.columns),
        'categorical_features': [col for col in categorical_columns if col in X.columns]
    }
    
    with open(output_dir / 'Base_Random_data.pkl', 'wb') as f:
        pickle.dump(base_random_data, f)
    logger.info(f"Saved Base_Random dataset to {output_dir / 'Base_Random_data.pkl'}")
    
    # Create Yeo_Random dataset (with Yeo-Johnson transformed features)
    # Apply Yeo-Johnson transformation to numeric features
    from sklearn.preprocessing import PowerTransformer
    
    X_train_yeo = X_train.copy()
    X_test_yeo = X_test.copy()
    
    # Transform numeric columns
    numeric_cols_in_data = [col for col in numeric_columns if col in X.columns]
    
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    
    # Fit on train data
    X_train_yeo[numeric_cols_in_data] = pt.fit_transform(X_train[numeric_cols_in_data])
    X_test_yeo[numeric_cols_in_data] = pt.transform(X_test[numeric_cols_in_data])
    
    # Rename columns to indicate transformation
    rename_dict = {col: f'yeo_joh_{col}' for col in numeric_cols_in_data}
    X_train_yeo = X_train_yeo.rename(columns=rename_dict)
    X_test_yeo = X_test_yeo.rename(columns=rename_dict)
    
    yeo_random_data = {
        'X_train': X_train_yeo,
        'X_test': X_test_yeo,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': list(X_train_yeo.columns),
        'categorical_features': [col for col in categorical_columns if col in X_train_yeo.columns]
    }
    
    with open(output_dir / 'Yeo_Random_data.pkl', 'wb') as f:
        pickle.dump(yeo_random_data, f)
    logger.info(f"Saved Yeo_Random dataset to {output_dir / 'Yeo_Random_data.pkl'}")
    
    # Also save the unified train/test split for consistency
    split_data = {
        'train_indices': X_train.index.tolist(),
        'test_indices': X_test.index.tolist()
    }
    
    with open(output_dir / 'train_test_split.pkl', 'wb') as f:
        pickle.dump(split_data, f)
    
    logger.info("Dataset preparation complete!")
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total samples: {len(X)}")
    print(f"Train samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    print(f"\nBase_Random features: {X_train.shape[1]}")
    print(f"Yeo_Random features: {X_train_yeo.shape[1]}")
    print(f"\nCategorical features: {len([col for col in categorical_columns if col in X.columns])}")
    print(f"Numeric features: {len(numeric_cols_in_data)}")
    
    if 'top_3_shareholder_percentage' in X_train.columns:
        print(f"\n✓ Feature 'top_3_shareholder_percentage' is present in Base_Random")
    if 'yeo_joh_top_3_shareholder_percentage' in X_train_yeo.columns:
        print(f"✓ Feature 'yeo_joh_top_3_shareholder_percentage' is present in Yeo_Random")


if __name__ == "__main__":
    prepare_categorical_datasets()