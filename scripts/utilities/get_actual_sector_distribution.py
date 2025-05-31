#!/usr/bin/env python3
"""Get the actual sector distribution from the tree models dataset."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.data_tree_models import load_tree_models_from_csv, perform_stratified_split_for_tree_models

# Load the data
print("Loading tree models data...")
X, y, cat_features = load_tree_models_from_csv()

# Check if gics_sector exists
if 'gics_sector' in X.columns:
    print(f"\nTotal dataset size: {len(X)}")
    print("\nOverall sector distribution:")
    overall_dist = X['gics_sector'].value_counts(normalize=True).sort_index()
    for sector, pct in overall_dist.items():
        print(f"  {sector}: {pct:.4f} ({int(pct * len(X))} companies)")
    
    # Perform stratified split
    print("\n\nPerforming stratified split (80/20)...")
    X_train, X_test, y_train, y_test = perform_stratified_split_for_tree_models(X, y, test_size=0.2, random_state=42)
    
    # Get train/test distributions
    train_dist = X_train['gics_sector'].value_counts(normalize=True).sort_index()
    test_dist = X_test['gics_sector'].value_counts(normalize=True).sort_index()
    
    # Print distributions for plotting
    print("\n\nData for plotting (proportions of total dataset):")
    print("Sector,Train_Proportion,Test_Proportion")
    
    total_size = len(X)
    for sector in overall_dist.index:
        train_count = (X_train['gics_sector'] == sector).sum()
        test_count = (X_test['gics_sector'] == sector).sum()
        train_prop = train_count / total_size
        test_prop = test_count / total_size
        print(f"{sector},{train_prop:.4f},{test_prop:.4f}")
    
    # Save the actual distributions
    output_path = project_root / "data" / "processed" / "sector_distribution.csv"
    distribution_data = []
    for sector in overall_dist.index:
        train_count = (X_train['gics_sector'] == sector).sum()
        test_count = (X_test['gics_sector'] == sector).sum()
        train_prop = train_count / total_size
        test_prop = test_count / total_size
        
        distribution_data.append({
            'sector': sector,
            'train_proportion': train_prop,
            'test_proportion': test_prop,
            'train_count': train_count,
            'test_count': test_count
        })
    
    dist_df = pd.DataFrame(distribution_data)
    dist_df.to_csv(output_path, index=False)
    print(f"\n\nSaved actual distributions to: {output_path}")
    
else:
    print("ERROR: gics_sector column not found in the dataset!")