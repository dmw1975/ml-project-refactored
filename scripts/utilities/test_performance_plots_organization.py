#!/usr/bin/env python3
"""Test script to verify performance plots organization."""

from pathlib import Path
import sys

# Define the performance directory
PERFORMANCE_DIR = Path("/mnt/d/ml_project_refactored/outputs/visualizations/performance")

def check_organization():
    """Check the current organization of performance plots."""
    
    print("Performance Plots Organization Test")
    print("=" * 50)
    
    if not PERFORMANCE_DIR.exists():
        print(f"Error: Performance directory not found: {PERFORMANCE_DIR}")
        return False
    
    # Expected subdirectories
    expected_subdirs = ['catboost', 'lightgbm', 'xgboost', 'elasticnet', 'linear', 'cv_distributions']
    
    # Check subdirectories
    print("\nChecking subdirectories:")
    all_good = True
    total_plots = 0
    
    for subdir in expected_subdirs:
        subdir_path = PERFORMANCE_DIR / subdir
        if subdir_path.exists():
            plot_count = len(list(subdir_path.glob("*.png")))
            total_plots += plot_count
            print(f"  ✓ {subdir}/: {plot_count} plots")
        else:
            print(f"  ✗ {subdir}/: Directory missing!")
            all_good = False
    
    # Check for files in root (only specific files should be there)
    allowed_root_files = ['metrics_summary_table.png', 'README.md']
    root_pngs = list(PERFORMANCE_DIR.glob("*.png"))
    root_plots = [f for f in root_pngs if f.name not in allowed_root_files]
    
    if root_plots:
        print(f"\n⚠ Warning: Found {len(root_plots)} unexpected PNG files in root directory:")
        for png in root_plots[:5]:  # Show first 5
            print(f"    - {png.name}")
        if len(root_plots) > 5:
            print(f"    ... and {len(root_plots) - 5} more")
        all_good = False
    else:
        print(f"\n✓ Root directory only contains expected files ({len(root_pngs)} PNG files)")
    
    total_plots += len(root_pngs)
    
    # Check README
    readme_path = PERFORMANCE_DIR / "README.md"
    if readme_path.exists():
        print("✓ README.md exists")
    else:
        print("✗ README.md missing")
        all_good = False
    
    # Detailed breakdown
    print("\nDetailed breakdown:")
    
    # Model-specific directories
    for subdir in ['catboost', 'lightgbm', 'xgboost', 'elasticnet']:
        subdir_path = PERFORMANCE_DIR / subdir
        if subdir_path.exists():
            plots = list(subdir_path.glob("*.png"))
            if plots:
                print(f"\n{subdir}/ ({len(plots)} plots):")
                
                # Count plot types
                opt_history = sum(1 for p in plots if 'optimization_history' in p.name)
                param_importance = sum(1 for p in plots if 'param_importance' in p.name)
                contour = sum(1 for p in plots if 'contour' in p.name)
                comparisons = sum(1 for p in plots if 'comparison' in p.name or 'vs' in p.name)
                
                print(f"  - Optimization history: {opt_history}")
                print(f"  - Parameter importance: {param_importance}")
                print(f"  - Contour plots: {contour}")
                print(f"  - Comparison plots: {comparisons}")
    
    # CV distributions
    cv_dir = PERFORMANCE_DIR / 'cv_distributions'
    if cv_dir.exists():
        cv_plots = list(cv_dir.glob("*.png"))
        if cv_plots:
            print(f"\ncv_distributions/ ({len(cv_plots)} plots):")
            for plot in cv_plots:
                print(f"  - {plot.name}")
    
    print(f"\nTotal plots: {total_plots}")
    
    # Check naming conventions
    print("\nChecking naming conventions:")
    naming_issues = []
    
    for subdir in ['catboost', 'lightgbm', 'xgboost', 'elasticnet', 'linear']:
        subdir_path = PERFORMANCE_DIR / subdir
        if subdir_path.exists():
            expected_prefixes = {
                'catboost': 'CatBoost_',
                'lightgbm': 'LightGBM_',
                'xgboost': 'XGBoost_',
                'elasticnet': 'ElasticNet_',
                'linear': 'LR_'
            }
            
            prefix = expected_prefixes.get(subdir)
            if prefix:
                for plot in subdir_path.glob("*.png"):
                    if not plot.name.startswith(prefix) and not plot.name.startswith(f"{subdir}_"):
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
        print("\n✅ Performance plots are properly organized!")
        sys.exit(0)
    else:
        print("\n❌ Some issues found with performance plots organization")
        sys.exit(1)