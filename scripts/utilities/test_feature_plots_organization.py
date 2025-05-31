#!/usr/bin/env python3
"""Test script to verify feature plots organization."""

from pathlib import Path
import sys

# Define the features directory
FEATURES_DIR = Path("/mnt/d/ml_project_refactored/outputs/visualizations/features")

def check_organization():
    """Check the current organization of feature plots."""
    
    print("Feature Plots Organization Test")
    print("=" * 50)
    
    if not FEATURES_DIR.exists():
        print(f"Error: Features directory not found: {FEATURES_DIR}")
        return False
    
    # Expected subdirectories
    expected_subdirs = ['catboost', 'lightgbm', 'xgboost', 'elasticnet', 'linear', 'comparisons']
    
    # Check subdirectories
    print("\nChecking subdirectories:")
    all_good = True
    for subdir in expected_subdirs:
        subdir_path = FEATURES_DIR / subdir
        if subdir_path.exists():
            plot_count = len(list(subdir_path.glob("*.png")))
            print(f"  ✓ {subdir}/: {plot_count} plots")
        else:
            print(f"  ✗ {subdir}/: Directory missing!")
            all_good = False
    
    # Check for files in root (there shouldn't be any PNG files)
    root_pngs = list(FEATURES_DIR.glob("*.png"))
    if root_pngs:
        print(f"\n⚠ Warning: Found {len(root_pngs)} PNG files in root directory:")
        for png in root_pngs[:5]:  # Show first 5
            print(f"    - {png.name}")
        if len(root_pngs) > 5:
            print(f"    ... and {len(root_pngs) - 5} more")
        all_good = False
    else:
        print("\n✓ No PNG files in root directory (good!)")
    
    # Check README
    readme_path = FEATURES_DIR / "README.md"
    if readme_path.exists():
        print("\n✓ README.md exists")
    else:
        print("\n✗ README.md missing")
        all_good = False
    
    # Detailed breakdown
    print("\nDetailed breakdown:")
    total_plots = 0
    for subdir in expected_subdirs:
        subdir_path = FEATURES_DIR / subdir
        if subdir_path.exists():
            plots = list(subdir_path.glob("*.png"))
            if plots:
                print(f"\n{subdir}/ ({len(plots)} plots):")
                for plot in sorted(plots)[:3]:  # Show first 3
                    print(f"  - {plot.name}")
                if len(plots) > 3:
                    print(f"  ... and {len(plots) - 3} more")
                total_plots += len(plots)
    
    print(f"\nTotal plots: {total_plots}")
    
    # Check naming conventions
    print("\nChecking naming conventions:")
    naming_issues = []
    
    for subdir in ['catboost', 'lightgbm', 'xgboost', 'elasticnet', 'linear']:
        subdir_path = FEATURES_DIR / subdir
        if subdir_path.exists():
            for plot in subdir_path.glob("*.png"):
                # Check if plot name starts with expected model type
                expected_prefixes = {
                    'catboost': 'CatBoost_',
                    'lightgbm': 'LightGBM_',
                    'xgboost': 'XGBoost_',
                    'elasticnet': 'ElasticNet_',
                    'linear': 'LR_'
                }
                
                prefix = expected_prefixes.get(subdir)
                if prefix and not plot.name.startswith(prefix):
                    naming_issues.append((subdir, plot.name))
    
    if naming_issues:
        print(f"  ✗ Found {len(naming_issues)} naming issues:")
        for subdir, name in naming_issues[:5]:
            print(f"    - {subdir}/{name}")
    else:
        print("  ✓ All plots follow naming conventions")
    
    return all_good

if __name__ == "__main__":
    success = check_organization()
    
    if success:
        print("\n✅ Feature plots are properly organized!")
        sys.exit(0)
    else:
        print("\n❌ Some issues found with feature plots organization")
        sys.exit(1)