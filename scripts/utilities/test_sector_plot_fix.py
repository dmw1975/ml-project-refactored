#!/usr/bin/env python3
"""Test the fixed sector distribution plot generation."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.visualization.plots.sectors import create_sector_stratification_plot

# Test directory
test_dir = Path(project_root) / "test_outputs" / "sectors"
test_dir.mkdir(parents=True, exist_ok=True)

print("Testing sector distribution plot generation...")
print(f"Output directory: {test_dir}")

# Generate the plot
success = create_sector_stratification_plot(test_dir)

if success:
    print("\n✓ Plot generated successfully!")
    plot_path = test_dir / "sector_train_test_distribution.png"
    if plot_path.exists():
        print(f"✓ Plot saved to: {plot_path}")
        print(f"  File size: {plot_path.stat().st_size / 1024:.1f} KB")
    else:
        print("✗ Plot file not found after generation")
else:
    print("\n✗ Plot generation failed!")

# Also test the main outputs directory
print("\n\nTesting with main outputs directory...")
main_output_dir = project_root / "outputs" / "visualizations" / "sectors"
main_output_dir.mkdir(parents=True, exist_ok=True)

success2 = create_sector_stratification_plot(main_output_dir)
if success2:
    print("✓ Plot generated successfully in main outputs directory!")
else:
    print("✗ Plot generation failed in main outputs directory!")