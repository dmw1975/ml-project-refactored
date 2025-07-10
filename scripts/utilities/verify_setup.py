#!/usr/bin/env python3
"""
Setup Verification Script for ML Pipeline

This script checks that all prerequisites are met for running the ML pipeline:
- Python dependencies are installed
- Required data files exist
- Directory structure is correct
"""

import sys
import os
from pathlib import Path
import importlib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings


def check_python_version():
    """Check if Python version meets requirements."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor}.{version.micro} - Too old (need 3.8+)")
        return False


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    critical_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('xgboost', 'xgboost'),
        ('lightgbm', 'lightgbm'),
        ('catboost', 'catboost'),
        ('optuna', 'optuna'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('shap', 'shap'),
    ]
    
    all_ok = True
    for package_name, import_name in critical_packages:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"  ✅ {package_name} ({version})")
        except ImportError:
            print(f"  ❌ {package_name} - NOT INSTALLED")
            all_ok = False
    
    return all_ok


def check_directory_structure():
    """Check if required directories exist."""
    print("\n📁 Checking directory structure...")
    
    required_dirs = [
        settings.DATA_DIR,
        settings.DATA_DIR / "raw",
        settings.DATA_DIR / "raw" / "metadata",
        settings.DATA_DIR / "processed",
        settings.DATA_DIR / "interim",
        settings.DATA_DIR / "pkl",
        settings.OUTPUT_DIR,
        settings.OUTPUT_DIR / "models",
        settings.OUTPUT_DIR / "metrics",
        settings.OUTPUT_DIR / "visualizations",
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"  ✅ {dir_path.relative_to(project_root)}")
        else:
            print(f"  ❌ {dir_path.relative_to(project_root)} - MISSING")
            all_ok = False
    
    return all_ok


def check_raw_data_files():
    """Check if required raw data files exist."""
    print("\n📊 Checking raw data files...")
    
    required_files = [
        settings.DATA_DIR / "raw" / "combined_df_for_linear_models.csv",
        settings.DATA_DIR / "raw" / "combined_df_for_tree_models.csv",
        settings.DATA_DIR / "raw" / "score.csv",
    ]
    
    optional_files = [
        settings.DATA_DIR / "raw" / "metadata" / "linear_model_columns.json",
        settings.DATA_DIR / "raw" / "metadata" / "tree_model_columns.json",
        settings.DATA_DIR / "raw" / "metadata" / "yeo_johnson_mapping.json",
        settings.DATA_DIR / "raw" / "metadata" / "feature_groups.json",
    ]
    
    all_required_ok = True
    
    print("  Required files:")
    for file_path in required_files:
        if file_path.exists():
            size_mb = file_path.stat().st_size / 1024 / 1024
            print(f"    ✅ {file_path.name} ({size_mb:.1f} MB)")
        else:
            print(f"    ❌ {file_path.name} - MISSING")
            all_required_ok = False
    
    print("\n  Optional metadata files:")
    for file_path in optional_files:
        if file_path.exists():
            print(f"    ✅ {file_path.name}")
        else:
            print(f"    ⚠️  {file_path.name} - Not found (optional)")
    
    return all_required_ok


def check_processed_data_files():
    """Check if processed data files exist."""
    print("\n🔄 Checking processed data files...")
    
    processed_files = [
        settings.DATA_DIR / "processed" / "linear_models_dataset.csv",
        settings.DATA_DIR / "processed" / "tree_models_dataset.csv",
        settings.DATA_DIR / "processed" / "train_test_split.json",
    ]
    
    all_exist = True
    for file_path in processed_files:
        if file_path.exists():
            print(f"  ✅ {file_path.name}")
        else:
            print(f"  ⚠️  {file_path.name} - Not found (will be generated)")
            all_exist = False
    
    return all_exist


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("ML Pipeline Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version()),
        ("Dependencies", check_dependencies()),
        ("Directory Structure", check_directory_structure()),
        ("Raw Data Files", check_raw_data_files()),
        ("Processed Data Files", check_processed_data_files()),
    ]
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_critical_ok = True
    for check_name, result in checks:
        if check_name == "Processed Data Files":
            # Processed files are optional - they can be generated
            status = "✅" if result else "⚠️"
            print(f"{status} {check_name}: {'OK' if result else 'Missing (will be generated)'}")
        else:
            status = "✅" if result else "❌"
            print(f"{status} {check_name}: {'OK' if result else 'FAILED'}")
            if not result and check_name != "Processed Data Files":
                all_critical_ok = False
    
    print("\n" + "=" * 60)
    
    if all_critical_ok:
        print("✅ All critical checks passed! You're ready to run the pipeline.")
        if not checks[4][1]:  # Processed data files missing
            print("\n📝 Note: Processed data files will be generated on first run.")
            print("   Run: python main.py --all")
    else:
        print("❌ Some critical checks failed. Please address the issues above.")
        print("\n📝 Next steps:")
        if not checks[1][1]:  # Dependencies
            print("  1. Install dependencies: pip install -r requirements.txt")
        if not checks[3][1]:  # Raw data
            print("  2. Add raw data files to data/raw/ directory")
            print("     See README.md for data setup instructions")
    
    return 0 if all_critical_ok else 1


if __name__ == "__main__":
    sys.exit(main())