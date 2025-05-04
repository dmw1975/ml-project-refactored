#!/usr/bin/env python3
"""
Script to enforce deprecation of the old visualization module.

This script:
1. Adds stronger deprecation warnings to all files in the old visualization module
2. Adds redirection comments pointing to the new implementations
3. (Optionally) Can move the legacy visualization directory to visualization_legacy
"""

import os
import re
import sys
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

# Mapping of old visualization modules to their new replacements
VIZ_MODULE_MAPPING = {
    "xgboost_plots": "visualization_new.adapters.xgboost_adapter",
    "lightgbm_plots": "visualization_new.adapters.lightgbm_adapter",
    "catboost_plots": "visualization_new.adapters.catboost_adapter",
    "elasticnet_plots": "visualization_new.adapters.elasticnet_adapter",
    "feature_plots": "visualization_new.plots.features",
    "metrics_plots": "visualization_new.plots.metrics",
    "statistical_tests": "visualization_new.plots.statistical_tests",
    "dataset_comparison": "visualization_new.plots.dataset_comparison",
    "create_residual_plots": "visualization_new.plots.residuals",
    "sector_plots": "visualization_new.plots.sectors"
}

def add_deprecation_warnings():
    """Add strong deprecation warnings to all files in the old visualization module."""
    viz_dir = project_root / "visualization"
    
    if not viz_dir.exists() or not viz_dir.is_dir():
        print(f"Error: {viz_dir} not found or is not a directory")
        return
    
    # Process all Python files in the visualization directory
    for py_file in viz_dir.glob("**/*.py"):
        if py_file.name == "__init__.py":
            continue  # Skip __init__.py files for now
        
        print(f"Processing {py_file.relative_to(project_root)}...")
        
        # Read the file content
        with open(py_file, "r") as f:
            content = f.read()
        
        # Check if the file already has a deprecation warning
        if "DEPRECATED" in content and "warnings.warn" in content:
            print(f"  Already has deprecation warning")
            continue
        
        # Determine the replacement module
        module_name = py_file.stem
        replacement_module = VIZ_MODULE_MAPPING.get(module_name, "visualization_new")
        
        # Create the deprecation warning block
        deprecation_block = f'''"""This module is DEPRECATED and scheduled for removal.

This module is maintained only for backward compatibility.
Please use {replacement_module} instead.
"""

import warnings

warnings.warn(
    f"This module is deprecated and will be removed. Use {replacement_module} instead.",
    DeprecationWarning,
    stacklevel=2
)

'''
        
        # Check if there's already a docstring
        docstring_pattern = r'^""".*?"""'
        if re.match(docstring_pattern, content, re.DOTALL):
            # Replace the existing docstring
            content = re.sub(docstring_pattern, deprecation_block, content, flags=re.DOTALL)
        else:
            # Add the deprecation block at the beginning of the file
            content = deprecation_block + content
        
        # Write the updated content
        with open(py_file, "w") as f:
            f.write(content)
        
        print(f"  Added deprecation warning")
    
    # Update the __init__.py file
    init_file = viz_dir / "__init__.py"
    if init_file.exists():
        with open(init_file, "r") as f:
            init_content = f.read()
        
        # Add deprecation warning to __init__.py
        if "DEPRECATED" not in init_content:
            deprecation_init = '''"""DEPRECATED visualization module.

This package is deprecated and maintained only for backward compatibility.
Please use visualization_new package instead.
"""

import warnings

warnings.warn(
    "The visualization module is deprecated. Please use visualization_new instead.",
    DeprecationWarning,
    stacklevel=2
)

'''
            
            # Add imports after the deprecation warning
            if "__all__" in init_content:
                # Keep the existing exports
                init_content = deprecation_init + init_content
            else:
                # Create new exports based on the Python files
                exports = []
                for py_file in viz_dir.glob("*.py"):
                    if py_file.name != "__init__.py":
                        exports.append(py_file.stem)
                
                if exports:
                    init_content = deprecation_init + "\n" + "\n".join([f"from visualization import {module}" for module in exports]) + "\n\n__all__ = " + str(exports)
                else:
                    init_content = deprecation_init
            
            with open(init_file, "w") as f:
                f.write(init_content)
            
            print(f"Updated {init_file.relative_to(project_root)} with deprecation warning")

def move_to_legacy(confirm=True):
    """
    Move the old visualization module to visualization_legacy.
    
    Args:
        confirm: If True, ask for confirmation before moving
    """
    viz_dir = project_root / "visualization"
    legacy_dir = project_root / "visualization_legacy"
    
    if not viz_dir.exists() or not viz_dir.is_dir():
        print(f"Error: {viz_dir} not found or is not a directory")
        return
    
    if legacy_dir.exists():
        print(f"Error: {legacy_dir} already exists")
        return
    
    if confirm:
        response = input(f"Are you sure you want to move {viz_dir} to {legacy_dir}? [y/N] ")
        if response.lower() != "y":
            print("Operation cancelled")
            return
    
    # Create a copy of the visualization directory as visualization_legacy
    shutil.copytree(viz_dir, legacy_dir)
    print(f"Copied {viz_dir} to {legacy_dir}")
    
    # Create a new visualization directory that redirects to visualization_new
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create a new __init__.py that redirects to visualization_new
    redirect_init = '''"""DEPRECATED visualization module.

This package is deprecated and maintained only for backward compatibility.
Please use visualization_new package instead.

The original files have been moved to visualization_legacy.
"""

import warnings

warnings.warn(
    "The visualization module is deprecated. Please use visualization_new instead.",
    DeprecationWarning,
    stacklevel=2
)

# Redirect imports to visualization_new
from visualization_new import *
'''
    
    with open(viz_dir / "__init__.py", "w") as f:
        f.write(redirect_init)
    
    print(f"Created redirect in {viz_dir}")

if __name__ == "__main__":
    # Add deprecation warnings to all files
    add_deprecation_warnings()
    
    # Optionally move to legacy directory
    if "--move-to-legacy" in sys.argv:
        # Pass confirm=False to skip confirmation
        move_to_legacy(confirm=False)