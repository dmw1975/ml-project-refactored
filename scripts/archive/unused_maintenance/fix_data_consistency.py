"""Fix data consistency between old and new pipelines.

This script ensures both pipelines use exactly the same data for fair comparison.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import pickle
import json
import shutil

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

# Import from old pipeline
from data import load_features_data, load_scores_data, get_base_and_yeo_features, add_random_feature


def main():
    """Fix data consistency between pipelines."""
    print("=== Fixing Data Consistency Between Pipelines ===\n")
    
    # Create directories in new pipeline if they don't exist
    new_data_dir = Path("esg_ml_clean/data")
    new_processed_dir = new_data_dir / "processed"
    new_raw_dir = new_data_dir / "raw"
    
    new_processed_dir.mkdir(parents=True, exist_ok=True)
    new_raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data using old pipeline
    print("Step 1: Loading data from old pipeline...")
    try:
        # Load features and scores
        feature_df = load_features_data()
        scores = load_scores_data()
        
        print(f"  - Loaded features: {feature_df.shape}")
        print(f"  - Loaded scores: {scores.shape}")
        
        # Get base and yeo datasets
        LR_Base, LR_Yeo, base_columns, yeo_columns = get_base_and_yeo_features(feature_df)
        
        print(f"  - Base dataset: {LR_Base.shape}")
        print(f"  - Yeo dataset: {LR_Yeo.shape}")
        
    except Exception as e:
        print(f"ERROR loading data from old pipeline: {e}")
        return
    
    # Step 2: Save processed data to new pipeline location
    print("\nStep 2: Saving data to new pipeline...")
    
    # Save features
    features_base_path = new_processed_dir / "features_base.csv"
    features_yeo_path = new_processed_dir / "features_yeo.csv"
    
    LR_Base.to_csv(features_base_path)
    LR_Yeo.to_csv(features_yeo_path)
    print(f"  - Saved base features to: {features_base_path}")
    print(f"  - Saved yeo features to: {features_yeo_path}")
    
    # Save targets
    targets_path = new_processed_dir / "targets.csv"
    scores_df = pd.DataFrame({'esg_score': scores}, index=scores.index)
    scores_df.to_csv(targets_path)
    print(f"  - Saved targets to: {targets_path}")
    
    # Step 3: Create feature mapping file
    print("\nStep 3: Creating feature mapping...")
    
    feature_mapping = {
        'base': {
            'n_features': len(base_columns),
            'features': base_columns,
            'shape': LR_Base.shape
        },
        'yeo': {
            'n_features': len(yeo_columns),
            'features': yeo_columns,
            'shape': LR_Yeo.shape
        },
        'consistency_check': {
            'features_match_base': len(LR_Base.columns) == len(base_columns),
            'features_match_yeo': len(LR_Yeo.columns) == len(yeo_columns),
            'samples_match': len(LR_Base) == len(LR_Yeo) == len(scores),
            'index_aligned': all(LR_Base.index == scores.index) and all(LR_Yeo.index == scores.index)
        }
    }
    
    mapping_path = new_processed_dir / "feature_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(feature_mapping, f, indent=2)
    print(f"  - Saved feature mapping to: {mapping_path}")
    
    # Step 4: Copy raw data files
    print("\nStep 4: Copying raw data files...")
    
    # Copy score.csv
    src_score = Path("data/raw/score.csv")
    if src_score.exists():
        dst_score = new_raw_dir / "score.csv"
        shutil.copy2(src_score, dst_score)
        print(f"  - Copied score.csv to new location")
    
    # Copy other raw data files
    raw_files = [
        "combined_df_for_linear_models.csv",
        "combined_df_for_ml_models.csv", 
        "combined_df_for_tree_models.csv"
    ]
    
    for fname in raw_files:
        src = Path(f"data/raw/{fname}")
        if src.exists():
            dst = new_raw_dir / fname
            shutil.copy2(src, dst)
            print(f"  - Copied {fname}")
    
    # Step 5: Verify data consistency
    print("\nStep 5: Verifying data consistency...")
    
    # Load saved data to verify
    saved_base = pd.read_csv(features_base_path, index_col=0)
    saved_yeo = pd.read_csv(features_yeo_path, index_col=0)
    saved_targets = pd.read_csv(targets_path, index_col=0).squeeze()
    
    print(f"\nConsistency checks:")
    print(f"  - Base features match: {saved_base.equals(LR_Base)}")
    print(f"  - Yeo features match: {saved_yeo.equals(LR_Yeo)}")
    print(f"  - Targets match: {saved_targets.equals(scores)}")
    print(f"  - Base shape: Original {LR_Base.shape} vs Saved {saved_base.shape}")
    print(f"  - Yeo shape: Original {LR_Yeo.shape} vs Saved {saved_yeo.shape}")
    print(f"  - Index alignment: {all(saved_base.index == saved_targets.index)}")
    
    # Step 6: Create datasets with random features for benchmarking
    print("\nStep 6: Creating random feature variants...")
    
    try:
        # Add random features
        base_random = add_random_feature(LR_Base, seed=42)
        yeo_random = add_random_feature(LR_Yeo, seed=42)
        
        # Save random feature variants
        base_random_path = new_processed_dir / "features_base_random.csv"
        yeo_random_path = new_processed_dir / "features_yeo_random.csv"
        
        base_random.to_csv(base_random_path)
        yeo_random.to_csv(yeo_random_path)
        
        print(f"  - Saved base_random features: {base_random.shape}")
        print(f"  - Saved yeo_random features: {yeo_random.shape}")
    except Exception as e:
        print(f"  - Warning: Could not create random feature variants: {e}")
        print("  - This is optional and won't affect main data consistency")
    
    print("\n=== Data Consistency Fix Complete ===")
    print(f"\nAll data files have been saved to: {new_processed_dir}")
    print("\nNext steps:")
    print("1. Update DataLoader to use the saved files directly")
    print("2. Re-run comparison to verify identical results")


if __name__ == "__main__":
    main()