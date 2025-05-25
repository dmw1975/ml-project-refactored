#!/usr/bin/env python3
"""
Repository organization script to clean up temporary files and organize the codebase.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Set


def get_files_to_archive() -> dict:
    """Identify files that should be archived/removed."""
    
    files_to_archive = {
        'temporary_scripts': [],
        'duplicate_models': [],
        'obsolete_viz': [],
        'backup_dirs': [],
        'misc_scripts': []
    }
    
    # Patterns for temporary scripts
    temp_patterns = [
        'test_*.py', 'fix_*.py', 'check_*.py', 'debug_*.py',
        'cleanup_*.py', 'generate_*.py', 'regenerate_*.py',
        'improve_*.py', 'update_*.py', 'create_*.py'
    ]
    
    # Essential scripts to keep
    keep_files = {
        'main.py', 'data_categorical.py', 'create_categorical_datasets.py',
        'config/settings.py', '__init__.py'
    }
    
    # Scan for temporary scripts
    for pattern in temp_patterns:
        for file in Path('.').glob(pattern):
            if file.name not in keep_files and file.name != 'organize_repository.py':
                files_to_archive['temporary_scripts'].append(str(file))
    
    # Duplicate model implementations
    model_duplicates = [
        'models/xgboost_model.py',  # Keep xgboost_categorical.py
        'models/lightgbm_model.py',  # Keep lightgbm_categorical.py
        'models/catboost_model.py',  # Keep catboost_categorical.py
        'models/lightgbm_categorical_original.py',
        'models/xgboost_categorical_original.py'
    ]
    for file in model_duplicates:
        if Path(file).exists():
            files_to_archive['duplicate_models'].append(file)
    
    # Obsolete visualization directories
    obsolete_dirs = ['visualization', 'visualization_legacy']
    for dir_name in obsolete_dirs:
        if Path(dir_name).exists():
            files_to_archive['obsolete_viz'].append(dir_name)
    
    # Backup directories
    for item in Path('.').iterdir():
        if item.is_dir() and 'backup' in item.name:
            files_to_archive['backup_dirs'].append(str(item))
    
    # Miscellaneous one-off scripts
    misc_files = [
        'elasticnet_target_encoding_safe.py',
        'target_encoding_approach.py',
        'improved_categorical_preprocessing.py',
        'model_comparison_shap_plot.py',
        'simple_baseline_evaluation.py',
        'x_code_snippets.py',
        'sector_vis_test.py',
        'elasticnet_shap_visualizations.py',
        'xgboost_shap_visualizations.py',
        'fixed_lightgbm_shap_visualizations.py',
        'improved_shap_visualizations.py',
        'fixed_shap_visualizations.py',
        'improved_elasticnet_plots.py',
        'improved_catboost_plots.py',
        'consolidate_sector_visualizations.py',
        'reorganize_visualizations.py',
        'reorganize_feature_visualizations.py',
        'create_sector_stratification_plot.py',
        'create_feature_importance_charts.py',
        'test_cross_model_feature_importance.py'
    ]
    
    for file in misc_files:
        if Path(file).exists():
            files_to_archive['misc_scripts'].append(file)
    
    return files_to_archive


def print_cleanup_summary(files_to_archive: dict) -> int:
    """Print summary of files to be archived."""
    total_files = 0
    
    print("="*60)
    print("Repository Cleanup Summary")
    print("="*60)
    
    for category, files in files_to_archive.items():
        if files:
            print(f"\n{category.replace('_', ' ').title()}: {len(files)} items")
            for file in sorted(files)[:5]:  # Show first 5
                print(f"  - {file}")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more")
            total_files += len(files)
    
    return total_files


def create_archive_structure() -> Path:
    """Create archive directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = Path(f"archive_{timestamp}")
    
    # Create subdirectories
    subdirs = [
        'temporary_scripts',
        'duplicate_models',
        'obsolete_viz',
        'backup_dirs',
        'misc_scripts',
        'documentation'
    ]
    
    for subdir in subdirs:
        (archive_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return archive_dir


def archive_files(files_to_archive: dict, archive_dir: Path, dry_run: bool = True):
    """Archive files to the archive directory."""
    
    if dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN - No files will be moved")
        print("Run with --execute to actually move files")
        print(f"{'='*60}")
        return
    
    print(f"\nArchiving files to: {archive_dir}")
    
    for category, files in files_to_archive.items():
        for file_path in files:
            src = Path(file_path)
            if src.exists():
                dst = archive_dir / category / src.name
                
                try:
                    if src.is_dir():
                        shutil.move(str(src), str(dst))
                    else:
                        shutil.move(str(src), str(dst))
                    print(f"  Moved: {file_path} -> {dst}")
                except Exception as e:
                    print(f"  Error moving {file_path}: {e}")


def update_main_imports():
    """Update main.py to use only categorical model versions."""
    print("\nChecking main.py imports...")
    
    replacements = [
        ('from models.xgboost_model import', 'from models.xgboost_categorical import'),
        ('from models.lightgbm_model import', 'from models.lightgbm_categorical import'),
        ('from models.catboost_model import', 'from models.catboost_categorical import'),
    ]
    
    main_file = Path('main.py')
    if main_file.exists():
        content = main_file.read_text()
        modified = False
        
        for old, new in replacements:
            if old in content:
                content = content.replace(old, new)
                modified = True
                print(f"  Updated import: {old} -> {new}")
        
        if modified:
            # Create backup
            backup_file = Path('main.py.bak')
            shutil.copy2(main_file, backup_file)
            main_file.write_text(content)
            print(f"  Saved backup to: {backup_file}")


def main():
    """Main cleanup function."""
    import sys
    
    dry_run = '--execute' not in sys.argv
    
    print("Repository Organization Tool")
    print("="*60)
    
    # Get files to archive
    files_to_archive = get_files_to_archive()
    
    # Print summary
    total_files = print_cleanup_summary(files_to_archive)
    
    if total_files == 0:
        print("\nNo files to clean up!")
        return
    
    print(f"\nTotal items to archive: {total_files}")
    
    if dry_run:
        print("\nThis is a DRY RUN. To execute the cleanup, run:")
        print("  python organize_repository.py --execute")
        print("\nRecommended steps:")
        print("1. Review the files listed above")
        print("2. Ensure you have committed any important changes")
        print("3. Run with --execute to move files to archive")
        print("4. Test that main.py still works correctly")
        print("5. If everything works, you can delete the archive folder")
    else:
        print("\nProceeding with archiving...")
        archive_dir = create_archive_structure()
        archive_files(files_to_archive, archive_dir, dry_run=False)
        update_main_imports()
        print(f"\n✅ Files archived to: {archive_dir}")
        print("\nNext steps:")
        print("1. Run 'python main.py' to ensure everything still works")
        print("2. Commit the changes")
        print("3. Once verified, you can delete the archive folder")


if __name__ == "__main__":
    main()