#!/usr/bin/env python3
"""
FIXED VERSION: Create separate datasets for tree models with BOTH raw and transformed features.
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root))

from src.config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.io import ensure_dir


def main():
    """Create tree models dataset with ALL features (raw + transformed + categorical)."""
    print("Creating FIXED tree models dataset...")
    
    # Use the FIXED tree models file
    input_path = RAW_DATA_DIR / "combined_df_for_tree_models.csv"
    if not input_path.exists():
        print(f"ERROR: Fixed dataset not found at {input_path}")
        print("Run fix_tree_models_dataset.py first!")
        return 1
    
    df = pd.read_csv(input_path)
    print(f"Loaded dataset: {df.shape}")
    
    # Define categorical columns
    categorical_features = [
        'gics_sector', 'gics_sub_ind', 'issuer_cntry_domicile_name', 'cntry_of_risk',
        'top_1_shareholder_location', 'top_2_shareholder_location', 'top_3_shareholder_location'
    ]
    
    # Convert categorical columns to category dtype
    for cat_col in categorical_features:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype('category')
    
    # Save to processed directory
    output_path = PROCESSED_DATA_DIR / "tree_models_dataset.csv"
    ensure_dir(PROCESSED_DATA_DIR)
    df.to_csv(output_path, index=False)
    
    print(f"Saved to: {output_path}")
    print(f"Features: {df.shape[1] - 1} (excluding issuer_name)")
    
    # Verify feature types
    raw_numerical = [c for c in df.columns if not c.startswith('yeo_joh_') 
                     and c not in categorical_features and c != 'issuer_name']
    yeo_numerical = [c for c in df.columns if c.startswith('yeo_joh_')]
    
    print(f"\nFeature breakdown:")
    print(f"  - Categorical: {len(categorical_features)}")
    print(f"  - Raw numerical: {len(raw_numerical)}")
    print(f"  - Yeo-transformed: {len(yeo_numerical)}")
    print(f"  - Total: {len(categorical_features) + len(raw_numerical) + len(yeo_numerical)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
