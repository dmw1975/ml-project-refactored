"""Test script to generate enhanced statistical test visualizations."""

import pandas as pd
import os
from pathlib import Path
from visualization_new.plots.statistical_tests import visualize_statistical_tests

def clean_unwanted_visualizations():
    """Remove deprecated or unwanted visualization files."""
    # Check both old and new locations
    old_output_dir_1 = Path('outputs/visualizations/performance/statistical_tests')
    old_output_dir_2 = Path('outputs/visualizations/statistical_tests')
    output_dir = Path('outputs/visualizations/performance/comparison/statistical_tests')
    
    # Files to be removed
    unwanted_files = [
        'model_significance_network.png',
        'model_win_loss_summary.png'
    ]
    
    # Also remove any LR heatmap files
    unwanted_patterns = [
        'model_significant_heatmap_LR_'
    ]
    
    # Collect all files that need to be removed
    files_to_remove = []
    
    # Check all directories
    for dir_path in [old_output_dir_1, old_output_dir_2, output_dir]:
        if dir_path.exists():
            for file in os.listdir(dir_path):
                file_path = dir_path / file
                
                # Check for exact matches
                if file in unwanted_files:
                    files_to_remove.append(file_path)
                    
                # Check for pattern matches
                for pattern in unwanted_patterns:
                    if pattern in file:
                        files_to_remove.append(file_path)
                        
    # Also copy any existing files from old to new location to ensure transition
    if output_dir.exists():
        for old_dir in [old_output_dir_1, old_output_dir_2]:
            if old_dir.exists():
                for file in os.listdir(old_dir):
                    if file.startswith('enhanced_significance_matrix'):
                        old_file = old_dir / file
                        new_file = output_dir / file
                        try:
                            # Use bash to copy the file
                            os.system(f"cp '{old_file}' '{new_file}'")
                            print(f"Copied: {old_file} to {new_file}")
                        except Exception as e:
                            print(f"Error copying {old_file}: {e}")
    
    # Remove the files
    for file_path in files_to_remove:
        try:
            os.remove(file_path)
            print(f"Removed: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")

def main():
    # Clean up unwanted visualizations
    clean_unwanted_visualizations()
    
    # Set paths
    metrics_dir = Path('outputs/metrics')
    tests_file = metrics_dir / "model_comparison_tests.csv"
    
    # Ensure file exists
    if not tests_file.exists():
        print(f"Error: File not found: {tests_file}")
        return
    
    # Create visualizations
    print(f"Creating enhanced statistical test visualizations from {tests_file}...")
    figures = visualize_statistical_tests(tests_file)
    
    print("Done! Check outputs/visualizations/statistical_tests for the updated figures.")

if __name__ == "__main__":
    main()