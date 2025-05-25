#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Consolidate sector visualizations into a single comprehensive plot.
Removes redundant visualizations and streamlines the sector visualization directory.
"""

import os
import shutil
from pathlib import Path

from config import settings
from utils import io

def remove_redundant_subdirectories():
    """Remove empty or redundant subdirectories in the sectors visualization directory."""
    sectors_dir = Path(settings.OUTPUT_DIR) / "visualizations" / "sectors"
    
    # List of directories to check and potentially remove
    subdirs_to_check = ["stratification", "stratification_old"]
    
    for subdir in subdirs_to_check:
        subdir_path = sectors_dir / subdir
        if subdir_path.exists():
            if subdir == "stratification" and len(list(subdir_path.glob("**/*"))) == 0:
                # Remove empty stratification directory
                shutil.rmtree(subdir_path)
                print(f"Removed empty directory: {subdir_path}")
            elif subdir == "stratification_old":
                # Simply remove the stratification_old directory without creating backups
                # (backups are handled by archive_and_clean_visualizations.py)
                shutil.rmtree(subdir_path)
                print(f"Removed redundant directory: {subdir_path}")
    
    print("Redundant subdirectories cleaned up successfully.")

def remove_redundant_sector_plots():
    """Remove all redundant sector weight plots, including the all_models_sector_summary.png."""
    sectors_dir = Path(settings.OUTPUT_DIR) / "visualizations" / "sectors"
    
    # Patterns of files to remove
    patterns = ["*_sector_weights.png", "all_models_sector_summary.png"]
    
    # Track if any files were found
    files_removed = False
    
    # Process each pattern and simply remove the files
    # (backups are handled by archive_and_clean_visualizations.py)
    for pattern in patterns:
        for file_path in sectors_dir.glob(pattern):
            file_path.unlink()
            print(f"Removed redundant plot: {file_path.name}")
            files_removed = True
    
    if files_removed:
        print("Redundant sector plots have been removed.")
    else:
        print("No redundant sector plots found to remove.")

def create_consolidated_sector_plot():
    """
    This function is now deprecated and intentionally does nothing.
    We no longer want to create the all_models_sector_summary.png plot.
    """
    # Skip creating the consolidated sector summary plot
    print("Skipping creation of all_models_sector_summary.png (deprecated)")
    return

def main():
    """Main function to consolidate sector visualizations."""
    # Get sectors directory
    sectors_dir = Path(settings.OUTPUT_DIR) / "visualizations" / "sectors"
    
    # No need to create backups as archive_and_clean_visualizations.py handles this
    print(f"Consolidating sector visualizations...")
    
    # Step 1: Remove redundant subdirectories
    remove_redundant_subdirectories()
    
    # Step 2: Create consolidated sector summary plot (this function is now a no-op)
    create_consolidated_sector_plot()
    
    # Step 3: Remove redundant individual sector plots
    remove_redundant_sector_plots()
    
    print("Sector visualizations have been successfully consolidated.")

if __name__ == "__main__":
    main()