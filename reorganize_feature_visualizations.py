"""
Script to reorganize feature visualizations into a more structured directory layout.

This will:
1. Create a separate VIF directory for VIF-related visualizations
2. Organize feature visualizations by model type
3. Consolidate redundant visualizations 
4. Create a cross-model comparison directory
"""

import os
import shutil
from pathlib import Path
import re

# Project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
VISUALIZATION_DIR = PROJECT_ROOT / "outputs" / "visualizations"
FEATURES_DIR = VISUALIZATION_DIR / "features"

# Create new directory structure
def create_directories():
    """Create the new directory structure for features."""
    # Create VIF directory
    vif_dir = VISUALIZATION_DIR / "vif"
    vif_dir.mkdir(exist_ok=True)
    
    # Create model subdirectories
    model_dirs = {
        "linear": FEATURES_DIR / "linear",
        "xgboost": FEATURES_DIR / "xgboost", 
        "lightgbm": FEATURES_DIR / "lightgbm",
        "catboost": FEATURES_DIR / "catboost",
        "comparison": FEATURES_DIR / "comparison",  # For cross-model comparisons
    }
    
    for dir_path in model_dirs.values():
        dir_path.mkdir(exist_ok=True, parents=True)
    
    return vif_dir, model_dirs

# Pattern matching functions to categorize files
def is_vif_file(filename):
    """Check if the file is a VIF-related visualization."""
    return "vif" in filename.lower()

def get_model_type(filename):
    """Determine the model type based on filename patterns."""
    filename_lower = filename.lower()
    
    if "elasticnet" in filename_lower or "lr_" in filename_lower:
        return "linear"
    elif "xgb" in filename_lower:
        return "xgboost" 
    elif "lightgbm" in filename_lower:
        return "lightgbm"
    elif "catboost" in filename_lower:
        return "catboost"
    elif any(x in filename_lower for x in ["top_20", "feature_correlation", "random_feature"]):
        return "comparison"
    else:
        return None

def is_redundant_visualization(filename):
    """Identify redundant feature visualizations that can be archived."""
    redundant_patterns = [
        "feature_correlation_clustering", 
        "top_features_correlation", 
        "top_features_distribution", 
        "top_features_heatmap"
    ]
    
    return any(pattern in filename for pattern in redundant_patterns)

def move_files():
    """Move files to their appropriate directories based on pattern matching."""
    # Create new directories
    vif_dir, model_dirs = create_directories()
    
    # Create an archive directory for redundant files
    archive_dir = FEATURES_DIR / "archive"
    archive_dir.mkdir(exist_ok=True)
    
    # Filter out subdirectories from the files list
    feature_files = [f for f in os.listdir(FEATURES_DIR) 
                    if os.path.isfile(FEATURES_DIR / f) and not f.startswith(".")]
    
    # Track stats for summary
    move_count = 0
    archived_count = 0
    
    # Process each file
    for filename in feature_files:
        source_path = FEATURES_DIR / filename
        
        # Skip if it's a directory
        if not source_path.is_file():
            continue
            
        # Determine destination based on file type
        if is_vif_file(filename):
            dest_dir = vif_dir
            dest_path = dest_dir / filename
            shutil.copy2(source_path, dest_path)
            print(f"Moved VIF visualization: {filename} -> {dest_path}")
            move_count += 1
        elif is_redundant_visualization(filename):
            dest_path = archive_dir / filename
            shutil.copy2(source_path, dest_path)
            print(f"Archived redundant visualization: {filename} -> {dest_path}")
            archived_count += 1
        else:
            # Determine model type
            model_type = get_model_type(filename)
            if model_type and model_type in model_dirs:
                dest_dir = model_dirs[model_type]
                dest_path = dest_dir / filename
                shutil.copy2(source_path, dest_path)
                print(f"Moved {model_type} visualization: {filename} -> {dest_path}")
                move_count += 1
            else:
                print(f"Skipping file (couldn't determine category): {filename}")
    
    print(f"\nSummary:\n---------")
    print(f"Moved {move_count} files to organized directories")
    print(f"Archived {archived_count} redundant visualizations")
    
    # Return paths to help with any further operations
    return vif_dir, model_dirs, archive_dir

if __name__ == "__main__":
    # Skip confirmation in automated environment
    print("Starting feature visualization reorganization...")
    
    # Perform reorganization
    vif_dir, model_dirs, archive_dir = move_files()
    
    print("\nReorganization complete!")
    print("\nNew directory structure:")
    print(f"- VIF visualizations: {vif_dir}")
    print(f"- Linear models: {model_dirs['linear']}")
    print(f"- XGBoost models: {model_dirs['xgboost']}")
    print(f"- LightGBM models: {model_dirs['lightgbm']}")
    print(f"- CatBoost models: {model_dirs['catboost']}")
    print(f"- Cross-model comparisons: {model_dirs['comparison']}")
    print(f"- Archived redundant files: {archive_dir}")
    
    print("\nNOTE: Original files were not deleted from the features directory.")
    print("After verifying the reorganization, you can remove the originals if desired.")