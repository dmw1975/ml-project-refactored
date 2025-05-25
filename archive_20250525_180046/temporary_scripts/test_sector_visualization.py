#!/usr/bin/env python
"""
Test script to verify sector weight visualizations work after migration.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from visualization_new.plots.sector_weights import plot_sector_weights_distribution

def test_sector_visualization():
    """Test the sector weight visualization."""
    print("Testing sector weight visualization...")
    
    # Try to generate sector weight plots with all models
    plot_sector_weights_distribution()
    
    print("Sector weight visualization test completed.")

if __name__ == "__main__":
    test_sector_visualization()