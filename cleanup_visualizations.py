"""Script to reorganize visualization directory structure.

This script addresses the following issues:
1. Moves visualizations from model/dataset specific root dirs to proper category dirs
2. Ensures feature visualizations are only in features directory (not duplicated)
3. Removes empty directories
"""

import shutil
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings

def move_files_safely(source_dir, target_dir, file_pattern="*"):
    """Move files matching pattern from source to target dir."""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Ensure target directory exists
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of files matching pattern
    files = list(source_path.glob(file_pattern))
    
    if not files:
        print(f"No files matching '{file_pattern}' found in {source_dir}")
        return 0
    
    # Move files
    moved_count = 0
    for file_path in files:
        target_file = target_path / file_path.name
        
        # Don't overwrite newer files
        if target_file.exists() and target_file.stat().st_mtime > file_path.stat().st_mtime:
            print(f"Skipping {file_path.name} (newer version exists in target)")
            continue
            
        try:
            shutil.copy2(file_path, target_file)
            moved_count += 1
            # Don't remove original files - we'll clean up directories later
        except Exception as e:
            print(f"Error moving {file_path}: {e}")
    
    return moved_count

def remove_empty_dirs(base_dir):
    """Remove empty directories under base_dir."""
    removed = 0
    for dirpath, dirnames, filenames in os.walk(base_dir, topdown=False):
        # Skip if this is the base directory itself
        if Path(dirpath) == base_dir:
            continue
            
        # Check if directory is empty (no files and no subdirectories)
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
                print(f"Removed empty directory: {dirpath}")
                removed += 1
            except Exception as e:
                print(f"Error removing directory {dirpath}: {e}")
    
    return removed

def consolidate_visualizations():
    """Reorganize the visualization directories."""
    print("Starting visualization directory reorganization...")
    
    # Set up base directories
    viz_dir = settings.VISUALIZATION_DIR
    features_dir = viz_dir / "features"
    residuals_dir = viz_dir / "residuals"
    performance_dir = viz_dir / "performance"
    
    # Ensure directories exist
    features_dir.mkdir(exist_ok=True)
    residuals_dir.mkdir(exist_ok=True)
    performance_dir.mkdir(exist_ok=True)
    
    # Define model types and their feature subdirectories 
    model_types = {
        'XGB': features_dir / 'xgboost',
        'LightGBM': features_dir / 'lightgbm',
        'CatBoost': features_dir / 'catboost',
        'ElasticNet': features_dir / 'elastic_net',
        'LR': features_dir / 'linear'
    }
    
    # Ensure feature type subdirectories exist
    for subdir in model_types.values():
        subdir.mkdir(exist_ok=True)
    
    # Get all directories in visualization directory
    all_dirs = [p for p in viz_dir.iterdir() if p.is_dir()]
    
    # Filter for model/dataset specific directories (those that would match a model name pattern)
    model_dirs = []
    for d in all_dirs:
        dir_name = d.name
        # Skip if it's already a category directory like 'features', 'residuals', etc.
        if dir_name in ['features', 'residuals', 'performance', 'metrics', 'sectors', 
                        'comparison', 'dataset_comparison', 'statistical_tests', 'vif', 'archive']:
            continue
            
        # Check if directory name matches any model pattern
        for model_prefix in model_types.keys():
            if dir_name.startswith(model_prefix):
                model_dirs.append(d)
                break
    
    print(f"Found {len(model_dirs)} model-specific directories to consolidate")
    
    # Process each model directory
    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"\nProcessing {model_name} directory...")
        
        # Determine model type
        model_type = None
        for prefix, subdir in model_types.items():
            if model_name.startswith(prefix):
                model_type = prefix
                model_subdir = subdir
                break
                
        if not model_type:
            print(f"Could not determine model type for {model_name}, skipping")
            continue
            
        # Move feature files to features directory and model type subdirectory
        feature_count = move_files_safely(model_dir, features_dir, "*_features*.png")
        feature_subdir_count = move_files_safely(model_dir, model_subdir, "*_features*.png")
        print(f"Moved {feature_count} feature visualizations to features/ directory")
        
        # Move residual files to residuals directory
        residual_count = move_files_safely(model_dir, residuals_dir, "*residuals*.png")
        print(f"Moved {residual_count} residual visualizations to residuals/ directory")
        
        # Move performance files to performance directory
        perf_count = move_files_safely(model_dir, performance_dir, "*performance*.png")
        print(f"Moved {perf_count} performance visualizations to performance/ directory")
        
        # Move any other visualization files to their respective directories
        # This is a catch-all for other plot types
        for file_path in model_dir.glob("*.png"):
            filename = file_path.name.lower()
            
            if 'feature' in filename and not filename.startswith(('copy_', 'old_')):
                dest_dir = features_dir
                print(f"Moving feature plot {filename} to features/ directory")
                shutil.copy2(file_path, dest_dir / file_path.name)
                
            elif 'residual' in filename and not filename.startswith(('copy_', 'old_')):
                dest_dir = residuals_dir
                print(f"Moving residual plot {filename} to residuals/ directory")
                shutil.copy2(file_path, dest_dir / file_path.name)
                
            elif any(keyword in filename for keyword in ['performance', 'metrics', 'comparison']) and not filename.startswith(('copy_', 'old_')):
                dest_dir = performance_dir
                print(f"Moving performance plot {filename} to performance/ directory")
                shutil.copy2(file_path, dest_dir / file_path.name)
    
    # Don't actually remove directories, just print a message
    print("\nAll files have been consolidated to the appropriate directories.")
    print("You can now safely remove these model-specific directories at the root level if desired.")
    print("To remove them, run the following command in your terminal:")
    for model_dir in model_dirs:
        print(f"rm -rf \"{model_dir}\"")
    
    print("\nReorganization complete!")
    return model_dirs

if __name__ == "__main__":
    model_dirs = consolidate_visualizations()
    
    # Ask for confirmation before removing directories
    if model_dirs:
        print("\nWould you like to remove the empty model-specific directories? (yes/no)")
        confirm = input().strip().lower()
        
        if confirm in ['yes', 'y']:
            for model_dir in model_dirs:
                try:
                    # Use rmtree with ignore_errors=False to catch and report any issues
                    shutil.rmtree(model_dir, ignore_errors=False)
                    print(f"Removed directory: {model_dir}")
                except Exception as e:
                    print(f"Error removing directory {model_dir}: {e}")
            print("Cleanup complete!")
        else:
            print("Directories were not removed. You can remove them manually if needed.")