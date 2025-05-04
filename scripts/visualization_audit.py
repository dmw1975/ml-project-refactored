#!/usr/bin/env python3
"""
Script to audit visualization functionality between old and new visualization modules.

This script compares function signatures and module structures between the legacy
visualization module and the new visualization_new architecture to identify:
1. Functions that exist in the old but not new architecture
2. Functions that exist in the new but not old architecture
3. Functions that exist in both but may have different implementations
"""

import os
import sys
import inspect
import importlib
import pkgutil
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

# Import both visualization packages
import visualization
import visualization_new


def get_module_functions(module) -> Dict[str, List[str]]:
    """
    Get all functions from a module and its submodules.
    
    Args:
        module: Module to inspect
        
    Returns:
        Dict mapping module names to list of function names
    """
    functions = {}
    
    # Get functions in the module itself
    module_functions = []
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and not name.startswith('_'):
            module_functions.append(name)
    
    if module_functions:
        functions[module.__name__] = module_functions
    
    # Get functions in submodules
    for _, name, is_pkg in pkgutil.iter_modules(module.__path__):
        submodule_name = f"{module.__name__}.{name}"
        
        try:
            submodule = importlib.import_module(submodule_name)
            
            if is_pkg:
                # Recursively get functions from subpackage
                subfuncs = get_module_functions(submodule)
                functions.update(subfuncs)
            else:
                # Get functions from module
                submodule_functions = []
                for fname, obj in inspect.getmembers(submodule):
                    if inspect.isfunction(obj) and not fname.startswith('_'):
                        submodule_functions.append(fname)
                
                if submodule_functions:
                    functions[submodule_name] = submodule_functions
        except ImportError as e:
            print(f"Error importing {submodule_name}: {e}")
    
    return functions


def get_module_files(module_path: Path) -> List[Path]:
    """
    Get all Python files in a module directory.
    
    Args:
        module_path: Path to module directory
        
    Returns:
        List of paths to Python files
    """
    return list(module_path.glob('**/*.py'))


def analyze_visualization_modules():
    """Analyze and compare the old and new visualization modules."""
    print("\n=== Visualization Module Audit ===\n")
    
    # Get paths
    old_viz_path = project_root / 'visualization'
    new_viz_path = project_root / 'visualization_new'
    
    # Get Python files in each module
    old_files = get_module_files(old_viz_path)
    new_files = get_module_files(new_viz_path)
    
    print(f"Legacy visualization: {len(old_files)} Python files")
    print(f"New visualization: {len(new_files)} Python files")
    
    # Get all functions
    try:
        old_functions = get_module_functions(visualization)
        new_functions = get_module_functions(visualization_new)
        
        # Calculate total function counts
        old_function_count = sum(len(funcs) for funcs in old_functions.values())
        new_function_count = sum(len(funcs) for funcs in new_functions.values())
        
        print(f"Legacy visualization: {old_function_count} functions across {len(old_functions)} modules")
        print(f"New visualization: {new_function_count} functions across {len(new_functions)} modules")
        
        # Flatten to sets of function names
        old_function_set = set()
        for module_funcs in old_functions.values():
            old_function_set.update(module_funcs)
            
        new_function_set = set()
        for module_funcs in new_functions.values():
            new_function_set.update(module_funcs)
        
        # Find functions in old but not new
        only_in_old = old_function_set - new_function_set
        only_in_new = new_function_set - old_function_set
        in_both = old_function_set.intersection(new_function_set)
        
        print(f"\nFunctions only in legacy visualization: {len(only_in_old)}")
        print(f"Functions only in new visualization: {len(only_in_new)}")
        print(f"Functions in both architectures: {len(in_both)}")
        
        # Print details
        if only_in_old:
            print("\nFunctions that need to be migrated to new architecture:")
            for func in sorted(only_in_old):
                print(f"  - {func}")
        
        if only_in_new:
            print("\nNew functions only in the new architecture:")
            for func in sorted(only_in_new):
                print(f"  - {func}")
    
    except Exception as e:
        print(f"Error analyzing functions: {e}")
    
    # Analyze file organization by type
    print("\nFile organization analysis:")
    
    # Old visualization organization (typically by model)
    old_by_model = {p.stem: p for p in old_files if p.name.endswith('_plots.py')}
    if old_by_model:
        print("\nLegacy visualization is organized by model type:")
        for model, path in sorted(old_by_model.items()):
            if '_plots' in model:
                model_name = model.replace('_plots', '')
                print(f"  - {model_name}: {path.relative_to(project_root)}")
    
    # New visualization organization (typically by plot type or component)
    plot_types = {p.stem: p for p in new_files if p.parent.name == 'plots'}
    if plot_types:
        print("\nNew visualization is organized by plot type:")
        for plot_type, path in sorted(plot_types.items()):
            print(f"  - {plot_type}: {path.relative_to(project_root)}")
    
    adapters = {p.stem: p for p in new_files if p.parent.name == 'adapters'}
    if adapters:
        print("\nModel adapters in new architecture:")
        for adapter, path in sorted(adapters.items()):
            if '_adapter' in adapter:
                model_name = adapter.replace('_adapter', '')
                print(f"  - {model_name}: {path.relative_to(project_root)}")


if __name__ == "__main__":
    analyze_visualization_modules()