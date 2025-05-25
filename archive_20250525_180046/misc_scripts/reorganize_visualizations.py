"""Script to reorganize performance visualizations into a more logical directory structure."""

import os
import shutil
from pathlib import Path
import re

# Define source and target directories
SOURCE_DIR = Path("/mnt/d/ml_project_refactored/outputs/visualizations/performance")
TARGET_DIR = SOURCE_DIR  # We'll use the same parent but reorganize into subdirectories

# Define model pattern mappings
MODEL_PATTERNS = {
    "xgboost": [
        r"XGB_.*\.png",
        r"xgboost_.*\.png",
    ],
    "lightgbm": [
        r"LightGBM_.*\.png",
        r"lightgbm_.*\.png",
        r"LGBM_.*\.png"
    ],
    "catboost": [
        r"CatBoost_.*\.png",
        r"catboost_.*\.png",
        r"CB_.*\.png"
    ],
    "linear": [
        r"LR_.*\.png",
        r"elasticnet_.*\.png",
        r"ElasticNet_.*\.png",
        r"EN_.*\.png"
    ],
    "comparison": [
        r"thesis_model_comparison.*\.png",
        r"metrics_summary_table\.png",
        r"all_models_.*\.png",
        r"model_metrics_comparison\.png",
        r"model_radar_comparison\.png"
    ]
}

# Define subcategory pattern mappings
SUBCATEGORY_PATTERNS = {
    "optimization": [
        r".*optuna_optimization.*\.png",
        r".*optuna_param_importance.*\.png",
        r".*_param_importance\.png",
        r".*_improvement\.png"
    ],
    "hyperparameters": [
        r".*_best_.*comparison\.png", 
        r".*_best_.*_param.*\.png",
        r".*_best_parameters\.png",
        r".*_cv_.*\.png"
    ]
}

def identify_model(filename):
    """Determine which model a file belongs to based on its name."""
    for model, patterns in MODEL_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, filename, re.IGNORECASE):
                return model
    return None

def identify_subcategory(filename, model):
    """Determine subcategory for a file within a model directory."""
    if model == "comparison":
        return None  # Comparison files don't have subcategories
        
    for subcategory, patterns in SUBCATEGORY_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, filename, re.IGNORECASE):
                return subcategory
    return None  # Files without a subcategory go directly in the model directory

def move_file(source, target):
    """Move a file from source to target, ensuring the target directory exists."""
    target_dir = target.parent
    if not target_dir.exists():
        os.makedirs(target_dir)
        
    if target.exists():
        print(f"WARNING: File already exists, skipping: {target}")
        return
        
    try:
        shutil.copy2(source, target)
        print(f"Copied: {source} -> {target}")
    except Exception as e:
        print(f"ERROR copying {source}: {e}")
        
def reorganize_files():
    """Reorganize visualization files into new directory structure."""
    # Get all PNG files in the source directory
    files = list(SOURCE_DIR.glob("*.png"))
    
    # Skip reserved directories to avoid duplicates
    reserved_dirs = ["old", "old_0205", "old_0502", "old_2", "old_3"]
    
    # Also handle subdirectories that aren't the reserved "old" directories
    for subdir in SOURCE_DIR.iterdir():
        if subdir.is_dir() and subdir.name not in reserved_dirs and not any(subdir.name.startswith(model) for model in MODEL_PATTERNS.keys()):
            files.extend(subdir.glob("*.png"))
    
    print(f"Found {len(files)} files to organize")
    
    # Track statistics
    stats = {
        "categorized": 0,
        "uncategorized": 0,
        "models": {model: 0 for model in MODEL_PATTERNS.keys()},
        "subcategories": {subcategory: 0 for subcategory in SUBCATEGORY_PATTERNS.keys()}
    }
    
    # Process each file
    for file_path in files:
        filename = file_path.name
        
        # Skip files that are already in one of our new model directories
        if any(str(model) in str(file_path.parent.name) for model in MODEL_PATTERNS.keys()):
            continue
            
        # Identify which model this file belongs to
        model = identify_model(filename)
        
        if not model:
            print(f"No category found for: {filename}")
            stats["uncategorized"] += 1
            continue
            
        stats["models"][model] += 1
        stats["categorized"] += 1
        
        # Identify subcategory, if any
        subcategory = identify_subcategory(filename, model)
        
        if subcategory:
            stats["subcategories"][subcategory] += 1
            target_path = TARGET_DIR / model / subcategory / filename
        else:
            target_path = TARGET_DIR / model / filename
            
        # Move the file
        move_file(file_path, target_path)
    
    print("\nReorganization Statistics:")
    print(f"Total files processed: {stats['categorized'] + stats['uncategorized']}")
    print(f"Files categorized: {stats['categorized']}")
    print(f"Files uncategorized: {stats['uncategorized']}")
    print("\nFiles per model:")
    for model, count in stats["models"].items():
        print(f"  {model}: {count}")
    print("\nFiles per subcategory:")
    for subcat, count in stats["subcategories"].items():
        print(f"  {subcat}: {count}")
        
if __name__ == "__main__":
    print("Reorganizing visualization files...")
    reorganize_files()
    print("\nDone! Files have been reorganized into model-specific directories.")