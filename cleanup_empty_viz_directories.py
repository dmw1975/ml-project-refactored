"""Script to clean up empty visualization directories.

This script removes empty directories in the visualization output folder to keep
the directory structure clean and organized, while preserving directories that are
needed by the codebase.
"""

import os
import sys
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config import settings

def find_empty_directories(base_dir: Path) -> list:
    """
    Find empty directories under base_dir.
    
    Args:
        base_dir (Path): Base directory to search
        
    Returns:
        list: List of empty directory paths
    """
    empty_dirs = []
    
    for root, dirs, files in os.walk(base_dir, topdown=False):
        # Check if directory is empty (no files and no subdirectories)
        if not dirs and not files:
            empty_dirs.append(Path(root))
    
    return empty_dirs

def should_preserve_directory(dir_path: Path) -> bool:
    """
    Determine if an empty directory should be preserved (not removed).

    Args:
        dir_path (Path): Directory path to check

    Returns:
        bool: True if the directory should be preserved, False if it can be removed
    """
    # Keep the VIF directory - it's used by the multicollinearity analysis
    if dir_path.name == "vif":
        return True

    # Keep model-specific feature directories referenced by the codebase
    model_dirs = ["catboost", "lightgbm", "xgboost", "elasticnet"]

    # Check if this is a model-specific feature directory
    if dir_path.parent.name == "features" and dir_path.name in model_dirs:
        return True

    # All other empty directories can be removed
    return False

def remove_empty_directories(base_dir: Path, dry_run: bool = False) -> list:
    """
    Remove empty directories in the given base directory, except for those that should be preserved.
    
    Args:
        base_dir (Path): The base directory to search for empty directories.
        dry_run (bool): If True, only print directories that would be removed without removing them.
        
    Returns:
        list: List of removed directory paths.
    """
    removed_dirs = []
    preserved_dirs = []
    
    # Walk through all directories in base_dir
    for root, dirs, files in os.walk(base_dir, topdown=False):
        root_path = Path(root)
        
        # Check if directory is empty (no files and no subdirectories)
        if not files and not dirs:
            if should_preserve_directory(root_path):
                preserved_dirs.append(root_path)
                if dry_run:
                    print(f"Would preserve empty directory (needed by code): {root_path}")
                else:
                    print(f"Preserving empty directory (needed by code): {root_path}")
            else:
                if dry_run:
                    print(f"Would remove empty directory: {root_path}")
                else:
                    try:
                        os.rmdir(root_path)
                        print(f"Removed empty directory: {root_path}")
                        removed_dirs.append(root_path)
                    except Exception as e:
                        print(f"Error removing directory {root_path}: {e}")
    
    return removed_dirs

def main(auto_remove=False):
    """
    Run the cleanup process.

    Args:
        auto_remove (bool): If True, automatically remove empty directories without prompting.
    """
    # Setup visualization directory path
    viz_dir = settings.VISUALIZATION_DIR

    # Double-check that we're using the right directory
    # Check if the visualization directory in the project structure exists
    project_viz_dir = project_root / "visualization"
    print(f"Checking visualization directory paths:")
    print(f"1. From settings.VISUALIZATION_DIR: {viz_dir}")
    print(f"2. Project visualization directory: {project_viz_dir}")

    # Use the project visualization directory if it exists
    if project_viz_dir.exists() and viz_dir != project_viz_dir:
        print(f"Using project visualization directory instead of settings.VISUALIZATION_DIR")
        viz_dir = project_viz_dir

    print(f"Checking for empty directories in: {viz_dir}")
    
    # Debug: Let's check if our find_empty_dirs function would find the empty directories
    empty_dirs_check = find_empty_directories(viz_dir)
    if empty_dirs_check:
        print(f"Debug: Found {len(empty_dirs_check)} empty directories using direct check")
        for d in empty_dirs_check:
            print(f"  - {d}")
    else:
        print("Debug: Direct check found no empty directories")

    # Store preserved directories for reporting
    preserved_dirs = []
    removable_dirs = []
    
    # First categorize the empty directories
    for dir_path in empty_dirs_check:
        if should_preserve_directory(dir_path):
            preserved_dirs.append(dir_path)
            print(f"Will preserve empty directory (needed by code): {dir_path}")
        else:
            removable_dirs.append(dir_path)
            print(f"Will remove empty directory: {dir_path}")
    
    if not removable_dirs and not preserved_dirs:
        print("No empty directories found.")
        return

    if auto_remove:
        # Automatically remove the directories
        if removable_dirs:
            print("\nRemoving unnecessary empty directories:")
            removed_count = 0
            for dir_path in removable_dirs:
                try:
                    os.rmdir(dir_path)
                    print(f"Removed: {dir_path}")
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing {dir_path}: {e}")
            
            print(f"\nRemoved {removed_count} empty directories.")
        
        # Report on preserved directories
        if preserved_dirs:
            print(f"Preserved {len(preserved_dirs)} empty directories needed by the codebase.")
    else:
        # Prompt user for confirmation
        try:
            if removable_dirs:
                confirm = input(f"\nDo you want to remove {len(removable_dirs)} empty directories? (y/n): ")
                if confirm.lower() != 'y':
                    print("Cleanup cancelled.")
                    return

                # Remove the directories
                print("\nRemoving empty directories:")
                removed_count = 0
                for dir_path in removable_dirs:
                    try:
                        os.rmdir(dir_path)
                        print(f"Removed: {dir_path}")
                        removed_count += 1
                    except Exception as e:
                        print(f"Error removing {dir_path}: {e}")
                
                print(f"\nRemoved {removed_count} empty directories.")
                
                # Report on preserved directories
                if preserved_dirs:
                    print(f"Preserved {len(preserved_dirs)} empty directories needed by the codebase.")
            else:
                print("\nNo directories to remove. Some empty directories are preserved as they are needed by the code.")
        except (EOFError, KeyboardInterrupt):
            print("\nRunning in non-interactive mode. Use --auto-yes to remove directories automatically.")
            return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean up empty directories in visualization output.")
    parser.add_argument("--auto-yes", "-y", action="store_true", help="Automatically remove empty directories without prompting.")

    args = parser.parse_args()
    main(auto_remove=args.auto_yes)