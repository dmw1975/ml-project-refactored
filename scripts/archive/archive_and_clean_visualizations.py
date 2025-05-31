"""Archive current visualizations and prepare for fresh pipeline run.

This script:
1. Creates a timestamped archive directory
2. Copies all current visualization files to the archive
3. Deletes all visualization files while preserving directory structure
4. Creates a README file in both locations documenting the process
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

# Import settings
from config import settings

def archive_and_clean_visualizations():
    """Archive existing visualizations and prepare for fresh pipeline run."""
    # Get current timestamp for archive directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = settings.OUTPUT_DIR / f"visualizations_archive_{timestamp}"
    viz_dir = settings.VISUALIZATION_DIR
    
    # Ensure the current visualization directory exists
    if not viz_dir.exists():
        print(f"Visualization directory {viz_dir} does not exist. Nothing to archive.")
        return
    
    # Create archive directory
    os.makedirs(archive_dir, exist_ok=True)
    print(f"Created archive directory: {archive_dir}")
    
    # Create README in archive directory
    with open(archive_dir / "README.txt", "w") as f:
        f.write(f"""Archive of visualization files created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This archive contains a snapshot of the visualization directory before running a fresh pipeline.
Original location: {viz_dir}

This archive was created to:
1. Preserve the existing visualizations for reference
2. Allow testing the visualization pipeline with a clean slate
3. Make it easier to identify what the new pipeline generates

The archive includes all subdirectories and files from the original visualization directory.
""")
    
    # Copy all files and directories recursively
    try:
        # Get list of all subdirectories to preserve
        subdirs = [d for d in viz_dir.glob("*") if d.is_dir()]
        
        # Copy all files recursively
        shutil.copytree(viz_dir, archive_dir / "visualizations", dirs_exist_ok=True)
        print(f"Successfully copied all visualization files to {archive_dir / 'visualizations'}")
        
        # Clean up the visualization directory while preserving structure
        for item in viz_dir.glob("**/*"):
            if item.is_file() and item.suffix.lower() in ['.png', '.jpg', '.jpeg', '.svg', '.pdf']:
                try:
                    os.remove(item)
                    print(f"Removed file: {item}")
                except Exception as e:
                    print(f"Error removing {item}: {e}")
        
        # Create README in the cleaned visualization directory
        with open(viz_dir / "README.txt", "w") as f:
            f.write(f"""Visualization directory cleaned on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

All visualization files have been archived to:
{archive_dir}

The directory structure has been preserved, but all image files have been removed.
This allows for testing the visualization pipeline with a clean slate.
""")
        
        print(f"Visualization directory has been cleaned while preserving structure")
        print(f"Archive is available at: {archive_dir}")
        
    except Exception as e:
        print(f"Error during archiving or cleaning: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Starting visualization archiving and cleanup process...")
    success = archive_and_clean_visualizations()
    
    if success:
        print("Archive and cleanup completed successfully!")
        print("\nYou can now run the pipeline to generate fresh visualizations.")
    else:
        print("Archive and cleanup failed.")