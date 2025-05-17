#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cleanup script to remove unused dependence plot files.
This script removes all dependence plot files from the SHAP visualization directory
to save disk space and improve organization.
"""

import os
import glob
from pathlib import Path
from config import settings

def remove_dependence_plots():
    """Remove all dependence plot files from the SHAP visualization directory."""
    # Get SHAP directory path
    shap_dir = Path(settings.OUTPUT_DIR) / "visualizations" / "shap"
    
    if not shap_dir.exists():
        print(f"SHAP directory not found: {shap_dir}")
        return
    
    # Get list of dependence plot files using a glob pattern
    dependence_files = list(shap_dir.glob("*dependence*"))
    
    print(f"Found {len(dependence_files)} dependence plot files to remove")
    
    # Remove each file
    for file_path in dependence_files:
        try:
            os.remove(file_path)
            print(f"Removed: {file_path.name}")
        except Exception as e:
            print(f"Error removing {file_path.name}: {e}")
    
    print(f"Cleanup complete. Removed {len(dependence_files)} dependence plot files.")

if __name__ == "__main__":
    print("Cleaning up unused dependence plot files...")
    remove_dependence_plots()