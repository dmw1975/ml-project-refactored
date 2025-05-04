#!/usr/bin/env python3
"""
Cleanup script to remove CatBoost_* directories from the visualization output.

This script identifies and removes all CatBoost_* directories in the visualization
output directory. It asks for confirmation before performing the removal.
"""

import os
import shutil  # Import shutil at module level to avoid local variable issues
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import settings
from config import settings

def main():
    """Find and remove CatBoost_* directories."""
    # Directory to clean
    viz_dir = settings.VISUALIZATION_DIR
    
    # Ensure directory exists
    if not viz_dir.exists():
        print(f"Visualization directory does not exist: {viz_dir}")
        return
    
    # Find all CatBoost_* directories in the root and model-specific dirs in features/catboost_*
    catboost_dirs = list(viz_dir.glob("CatBoost_*"))
    # Also find model-specific directories under features/ and residuals/
    catboost_feature_dirs = list(viz_dir.glob("features/catboost_*"))
    catboost_residual_dirs = list(viz_dir.glob("residuals/catboost_*"))
    # Also find catboost directories in general/
    general_catboost_dirs = list(viz_dir.glob("general/catboost_*"))
    # Combine all directories to remove
    all_dirs_to_remove = catboost_dirs + catboost_feature_dirs + catboost_residual_dirs + general_catboost_dirs
    
    # If no directories found
    if not all_dirs_to_remove:
        print(f"No CatBoost directories to remove")
        return
    
    # Output information
    print(f"Found {len(all_dirs_to_remove)} CatBoost directories to remove:")
    for d in all_dirs_to_remove:
        print(f" - {d}")
        
        # List contents (optional - helpful to verify what's being removed)
        files = list(d.glob("*"))
        if files:
            print(f"   Contains {len(files)} files:")
            for f in files[:5]:  # Show first 5 files
                print(f"     - {f.name}")
            if len(files) > 5:
                print(f"     - ... and {len(files) - 5} more")
    
    # Add command-line argument support for non-interactive use
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Remove CatBoost_* directories")
    parser.add_argument('--force', '-f', action='store_true', 
                       help="Force removal without confirmation")
    args = parser.parse_args()
    
    # Ask for confirmation if not forced
    if args.force:
        confirmation = 'y'
    else:
        try:
            confirmation = input("\nProceed with removal? (y/n): ").lower()
        except EOFError:
            # Handle running in non-interactive environment
            print("Non-interactive environment detected. Use --force to remove directories.")
            return
    
    if confirmation == 'y':
        # First ensure that the target directories exist
        features_dir = viz_dir / "features" / "catboost"
        residuals_dir = viz_dir / "residuals" / "catboost"
        
        features_dir.mkdir(parents=True, exist_ok=True)
        residuals_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each directory to remove
        for d in all_dirs_to_remove:
            try:
                # Optional: Move important files to the flatter structure before removal
                # Example: Move any feature importance plots to features/catboost/
                feature_files = list(d.glob("*top_features*"))
                residual_files = list(d.glob("*residuals*"))
                
                for file in feature_files:
                    # Create a new name based on variant
                    if 'CatBoost_' in str(d):
                        # For root CatBoost_* directories
                        variant = str(d).split('CatBoost_', 1)[1].lower()
                        new_name = f"top_features_{variant}.{file.suffix}"
                    else:
                        # For subdirectories
                        new_name = file.name
                    
                    # Copy to the target directory
                    target = features_dir / new_name
                    try:
                        shutil.copy2(file, target)
                        print(f"  Copied {file.name} to {target}")
                    except Exception as e:
                        print(f"  Error copying {file.name}: {e}")
                
                for file in residual_files:
                    # Create a new name based on variant
                    if 'CatBoost_' in str(d):
                        # For root CatBoost_* directories
                        variant = str(d).split('CatBoost_', 1)[1].lower()
                        new_name = f"residuals_{variant}.{file.suffix}"
                    else:
                        # For subdirectories
                        new_name = file.name
                    
                    # Copy to the target directory
                    target = residuals_dir / new_name
                    try:
                        shutil.copy2(file, target)
                        print(f"  Copied {file.name} to {target}")
                    except Exception as e:
                        print(f"  Error copying {file.name}: {e}")
                
                # Remove the directory
                shutil.rmtree(d)
                print(f"Removed {d}")
            except Exception as e:
                print(f"Error removing {d}: {e}")
    else:
        print("Operation cancelled.")
        
    # Suggest next steps
    print("\nNext steps:")
    print("1. Run tests to ensure visualizations are still working correctly")
    print("2. Verify that new visualizations are being saved to the correct directories")
    print("   (features/*, residuals/*, performance/*, etc.)")

if __name__ == "__main__":
    main()