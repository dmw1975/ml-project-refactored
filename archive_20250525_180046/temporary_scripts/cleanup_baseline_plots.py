"""
Script to clean up individual baseline plots while preserving them in a backup directory.
"""

import os
import shutil
from pathlib import Path

BASELINE_DIR = Path("/mnt/d/ml_project_refactored/outputs/visualizations/baselines")
BACKUP_DIR = BASELINE_DIR / "backup"
SUMMARY_PLOTS = [
    "model_vs_random_comparison.png",
    "baseline_improvement.png",
    "RMSE_comparison.png",
    "R2_comparison.png", 
    "MSE_comparison.png",
    "MAE_comparison.png",
    "combined_metrics_comparison.png"
]

def cleanup_individual_baseline_plots():
    """
    Moves individual baseline plots to a backup directory while preserving summary plots.
    """
    print(f"Cleaning up individual baseline plots in {BASELINE_DIR}")
    
    # Create backup directory if it doesn't exist
    BACKUP_DIR.mkdir(exist_ok=True)
    
    # Count variables for reporting
    total_files = 0
    moved_files = 0
    skipped_files = 0
    
    # Process all files in the baseline directory
    for file_path in BASELINE_DIR.glob("*.png"):
        total_files += 1
        filename = file_path.name
        
        # Skip summary plots
        if filename in SUMMARY_PLOTS:
            print(f"  Keeping summary plot: {filename}")
            skipped_files += 1
            continue
        
        # Move individual plot to backup
        try:
            # Check if it's an individual model vs baseline plot
            is_individual = (
                "_vs_baseline.png" in filename or 
                "_performance_improvement.png" in filename
            )
            
            if is_individual:
                # Use shutil.move to move the file
                backup_path = BACKUP_DIR / filename
                shutil.move(str(file_path), str(backup_path))
                print(f"  Moved: {filename} → backup/")
                moved_files += 1
            else:
                # Any other files we're not sure about - leave in place
                skipped_files += 1
                print(f"  Skipping file (not baseline pattern): {filename}")
        except Exception as e:
            print(f"  Error moving {filename}: {e}")
            
    print("\nCleanup Summary:")
    print(f"  Total files processed: {total_files}")
    print(f"  Files moved to backup: {moved_files}")
    print(f"  Files kept in place: {skipped_files}")
    print(f"\nBackup directory: {BACKUP_DIR}")
    
if __name__ == "__main__":
    cleanup_individual_baseline_plots()