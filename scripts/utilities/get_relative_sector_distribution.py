#!/usr/bin/env python3
"""Get the relative sector distribution within train and test sets."""

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
    
    # Perform stratified split
    print("\nPerforming stratified split (80/20)...")
    X_train, X_test, y_train, y_test = perform_stratified_split_for_tree_models(X, y, test_size=0.2, random_state=42)
    
    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Get relative distributions within each set
    train_dist = X_train['gics_sector'].value_counts(normalize=True).sort_index()
    test_dist = X_test['gics_sector'].value_counts(normalize=True).sort_index()
    
    # Also get overall distribution for reference
    overall_dist = X['gics_sector'].value_counts(normalize=True).sort_index()
    
    print("\n\nRelative distributions (percentage within each set):")
    print("=" * 60)
    print(f"{'Sector':<30} {'Overall':<10} {'Train':<10} {'Test':<10} {'Diff':<10}")
    print("-" * 60)
    
    distribution_data = []
    for sector in overall_dist.index:
        overall_pct = overall_dist[sector] * 100
        train_pct = train_dist.get(sector, 0) * 100
        test_pct = test_dist.get(sector, 0) * 100
        diff = abs(train_pct - test_pct)
        
        print(f"{sector:<30} {overall_pct:>8.1f}% {train_pct:>8.1f}% {test_pct:>8.1f}% {diff:>8.2f}%")
        
        distribution_data.append({
            'sector': sector,
            'overall_pct': overall_pct / 100,  # Store as proportion
            'train_pct': train_pct / 100,      # Store as proportion
            'test_pct': test_pct / 100,        # Store as proportion
            'train_count': (X_train['gics_sector'] == sector).sum(),
            'test_count': (X_test['gics_sector'] == sector).sum()
        })
    
    # Calculate statistics
    differences = [abs(d['train_pct'] - d['test_pct']) * 100 for d in distribution_data]
    print("\n" + "=" * 60)
    print(f"Maximum difference: {max(differences):.2f}%")
    print(f"Average difference: {np.mean(differences):.2f}%")
    print(f"Stratification quality: {'EXCELLENT' if max(differences) < 1 else 'GOOD' if max(differences) < 2 else 'FAIR'}")
    
    # Save the relative distributions
    output_path = project_root / "data" / "processed" / "sector_distribution_relative.csv"
    dist_df = pd.DataFrame(distribution_data)
    dist_df.to_csv(output_path, index=False)
    print(f"\n\nSaved relative distributions to: {output_path}")
    
else:
    print("ERROR: gics_sector column not found in the dataset!")