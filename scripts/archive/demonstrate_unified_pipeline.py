#!/usr/bin/env python3
"""
Demonstrate the unified data pipeline with a simple example.
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from create_unified_data_pipeline import load_unified_data
from config import settings
from utils import io


def demonstrate_unified_pipeline():
    """Show that different model types use the same test set."""
    
    print("="*60)
    print("Demonstrating Unified Data Pipeline")
    print("="*60)
    
    # Load data for linear models
    print("\n1. Loading data for linear models (one-hot encoded)...")
    linear_features, linear_target, train_idx, test_idx = load_unified_data(model_type='linear')
    print(f"   Features shape: {linear_features.shape}")
    print(f"   Train size: {len(train_idx)}, Test size: {len(test_idx)}")
    
    # Load data for tree models
    print("\n2. Loading data for tree models (categorical)...")
    tree_features, tree_target, tree_train_idx, tree_test_idx = load_unified_data(model_type='tree')
    print(f"   Features shape: {tree_features.shape}")
    print(f"   Train size: {len(tree_train_idx)}, Test size: {len(tree_test_idx)}")
    
    # Verify same split indices
    print("\n3. Verifying split indices are identical...")
    print(f"   Train indices match: {np.array_equal(train_idx, tree_train_idx)}")
    print(f"   Test indices match: {np.array_equal(test_idx, tree_test_idx)}")
    
    # Train simple models to show they use same test set
    print("\n4. Training simple models on both datasets...")
    
    # Linear model
    lr = LinearRegression()
    lr.fit(linear_features.iloc[train_idx], linear_target.iloc[train_idx])
    lr_pred = lr.predict(linear_features.iloc[test_idx])
    lr_rmse = np.sqrt(mean_squared_error(linear_target.iloc[test_idx], lr_pred))
    
    print(f"\n   Linear Regression:")
    print(f"   - Test RMSE: {lr_rmse:.4f}")
    print(f"   - Test samples: {len(test_idx)}")
    
    # Tree model (using only numeric features for simplicity)
    numeric_cols = tree_features.select_dtypes(include=[np.number]).columns
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    rf.fit(tree_features[numeric_cols].iloc[train_idx], tree_target.iloc[train_idx])
    rf_pred = rf.predict(tree_features[numeric_cols].iloc[test_idx])
    rf_rmse = np.sqrt(mean_squared_error(tree_target.iloc[test_idx], rf_pred))
    
    print(f"\n   Random Forest (on tree dataset):")
    print(f"   - Test RMSE: {rf_rmse:.4f}")
    print(f"   - Test samples: {len(test_idx)}")
    
    # Show that test targets are identical
    print("\n5. Verifying test targets are identical...")
    linear_test_targets = linear_target.iloc[test_idx].values
    tree_test_targets = tree_target.iloc[test_idx].values
    print(f"   Test targets match: {np.array_equal(linear_test_targets, tree_test_targets)}")
    
    # Calculate baseline metrics (should be same for both)
    print("\n6. Calculating baseline metrics...")
    
    # Mean baseline
    train_mean = float(linear_target.iloc[train_idx].mean())
    test_mean = float(linear_target.iloc[test_idx].mean())
    mean_baseline_pred = np.full(len(test_idx), train_mean)
    mean_baseline_rmse = np.sqrt(mean_squared_error(linear_test_targets, mean_baseline_pred))
    
    print(f"\n   Mean baseline:")
    print(f"   - Training set mean: {train_mean:.4f}")
    print(f"   - Test set mean: {test_mean:.4f}")
    print(f"   - Test RMSE (mean baseline): {mean_baseline_rmse:.4f}")
    
    # Random baseline
    np.random.seed(42)
    random_baseline_pred = np.random.uniform(0, 10, len(test_idx))
    random_baseline_rmse = np.sqrt(mean_squared_error(linear_test_targets, random_baseline_pred))
    
    print(f"\n   Random baseline:")
    print(f"   - Baseline RMSE: {random_baseline_rmse:.4f}")
    
    print("\n" + "="*60)
    print("✅ Unified pipeline ensures:")
    print("   - All models use the same train/test split")
    print("   - Baseline values will be identical across model types")
    print("   - Fair comparison between different model architectures")
    print("="*60)


if __name__ == "__main__":
    # Check if unified data exists
    unified_dir = settings.PROCESSED_DATA_DIR / "unified"
    if not (unified_dir / "train_test_split.pkl").exists():
        print("Unified datasets not found. Creating them first...")
        from create_unified_data_pipeline import create_unified_datasets
        create_unified_datasets()
    
    # Run demonstration
    demonstrate_unified_pipeline()