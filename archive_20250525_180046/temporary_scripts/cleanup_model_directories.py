"""Script to clean up model/dataset specific directories.

This script will:
1. Delete all model/dataset combination folders (e.g., LightGBM_Base_optuna/)
2. Keep the type-specific directories (features/, residuals/, etc.)
"""

import shutil
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings

def clean_model_directories():
    """Delete model/dataset specific directories."""
    print("Starting cleanup of model/dataset specific directories...")
    
    # Set up base visualization directory
    viz_dir = settings.VISUALIZATION_DIR
    
    # Get all directories in visualization directory
    all_dirs = [p for p in viz_dir.iterdir() if p.is_dir()]
    
    # Define which directories to keep (type-specific directories)
    dirs_to_keep = [
        'features', 'residuals', 'performance', 'metrics', 'summary', 
        'sectors', 'comparison', 'dataset_comparison', 'statistical_tests', 
        'vif', 'archive'
    ]
    
    # Define model prefixes to identify model/dataset directories
    model_prefixes = ['XGB_', 'LightGBM_', 'CatBoost_', 'ElasticNet_', 'LR_']
    
    # Find model/dataset directories to delete
    dirs_to_delete = []
    for d in all_dirs:
        dir_name = d.name
        
        # Skip if it's a type-specific directory that we want to keep
        if dir_name in dirs_to_keep:
            continue
            
        # Check if it's a model/dataset directory
        if any(dir_name.startswith(prefix) for prefix in model_prefixes):
            dirs_to_delete.append(d)
    
    print(f"Found {len(dirs_to_delete)} model/dataset directories to delete:")
    for d in dirs_to_delete:
        print(f"  - {d.name}")
    
    return dirs_to_delete

if __name__ == "__main__":
    dirs_to_delete = clean_model_directories()
    
    # Automatically delete the directories without asking
    if dirs_to_delete:
        print("\nRemoving model/dataset directories...")
        for dir_path in dirs_to_delete:
            try:
                shutil.rmtree(dir_path)
                print(f"Removed directory: {dir_path}")
            except Exception as e:
                print(f"Error removing directory {dir_path}: {e}")
        print("Cleanup complete!")
    else:
        print("No model/dataset directories found to delete.")