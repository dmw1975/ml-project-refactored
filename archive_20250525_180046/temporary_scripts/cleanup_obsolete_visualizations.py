#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cleanup script to remove the obsolete visualization files.
This script removes the specified visualization files that are no longer needed.
"""

import os
from pathlib import Path
from config import settings

def remove_obsolete_files():
    """Remove the specified obsolete visualization files."""
    # Get SHAP directory path
    shap_dir = Path(settings.OUTPUT_DIR) / "visualizations" / "shap"
    
    if not shap_dir.exists():
        print(f"SHAP directory not found: {shap_dir}")
        return
    
    # List of files to remove
    files_to_remove = [
        "lightgbm_feature_importance_bar_fixed.png",
        "model_comparison_shap.png",
        "xgboost_base_feature_importance.png",
        "xgboost_yeo_feature_importance.png"
    ]
    
    # Try to remove each file
    removed_count = 0
    
    for filename in files_to_remove:
        file_path = shap_dir / filename
        if file_path.exists():
            try:
                os.remove(file_path)
                print(f"Removed: {filename}")
                removed_count += 1
            except Exception as e:
                print(f"Error removing {filename}: {e}")
        else:
            print(f"File not found: {filename}")
    
    print(f"Cleanup complete. Removed {removed_count} obsolete files.")

if __name__ == "__main__":
    print("Cleaning up obsolete visualization files...")
    remove_obsolete_files()