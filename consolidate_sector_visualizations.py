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
                # Create backup of stratification_old if it has content
                if len(list(subdir_path.glob("**/*"))) > 0:
                    backup_dir = sectors_dir.parent / "sectors_stratification_backup"
                    os.makedirs(backup_dir, exist_ok=True)
                    # Copy files to backup location
                    for file_path in subdir_path.glob("**/*"):
                        if file_path.is_file():
                            rel_path = file_path.relative_to(subdir_path)
                            dst_path = backup_dir / rel_path
                            os.makedirs(dst_path.parent, exist_ok=True)
                            shutil.copy2(file_path, dst_path)
                    
                    # Remove the old directory
                    shutil.rmtree(subdir_path)
                    print(f"Backed up and removed redundant directory: {subdir_path}")
                else:
                    # Remove empty directory
                    shutil.rmtree(subdir_path)
                    print(f"Removed empty directory: {subdir_path}")
    
    print("Redundant subdirectories cleaned up successfully.")

def remove_redundant_sector_plots():
    """Remove all redundant sector weight plots, including the all_models_sector_summary.png."""
    sectors_dir = Path(settings.OUTPUT_DIR) / "visualizations" / "sectors"
    
    # Create backup directory for all sector plots
    backup_dir = sectors_dir.parent / "sectors_weights_backup"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Patterns of files to remove
    patterns = ["*_sector_weights.png", "all_models_sector_summary.png"]
    
    # Process each pattern
    for pattern in patterns:
        for file_path in sectors_dir.glob(pattern):
            # Copy to backup
            shutil.copy2(file_path, backup_dir / file_path.name)
            # Remove original
            file_path.unlink()
            print(f"Backed up and removed redundant plot: {file_path.name}")
    
    print("Redundant sector plots have been backed up and removed.")

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
    # Create backup directory for the entire sectors directory
    sectors_dir = Path(settings.OUTPUT_DIR) / "visualizations" / "sectors"
    backup_root = Path(settings.OUTPUT_DIR) / "visualizations_backup"
    os.makedirs(backup_root, exist_ok=True)
    
    # Create timestamped backup of the entire sectors directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_root / f"sectors_backup_{timestamp}"
    
    # Copy entire sectors directory as backup
    if sectors_dir.exists():
        shutil.copytree(sectors_dir, backup_dir)
        print(f"Created backup of sectors directory: {backup_dir}")
    
    # Step 1: Remove redundant subdirectories
    remove_redundant_subdirectories()
    
    # Step 2: Create consolidated sector summary plot
    create_consolidated_sector_plot()
    
    # Step 3: Remove redundant individual sector plots
    remove_redundant_sector_plots()
    
    print("Sector visualizations have been successfully consolidated.")

if __name__ == "__main__":
    main()