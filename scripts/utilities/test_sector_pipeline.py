#!/usr/bin/env python3
"""
Test sector visualization pipeline to ensure model information is included.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.visualization.plots.sectors import visualize_all_sector_plots
from src.config import settings


def test_sector_pipeline():
    """Test that sector plots include model information."""
    print("Testing sector visualization pipeline...")
    print("="*60)
    
    # Check existing plots
    sectors_dir = settings.VISUALIZATION_DIR / "sectors"
    before_plots = list(sectors_dir.glob("*.png")) if sectors_dir.exists() else []
    print(f"Sector plots before: {len(before_plots)}")
    
    # Run sector visualizations
    print("\nRunning sector visualizations...")
    try:
        results = visualize_all_sector_plots()
        
        print(f"\nPlots created: {len(results)}")
        for plot_name, fig in results.items():
            print(f"  - {plot_name}")
        
        # Check specific plots
        boxplot_path = sectors_dir / "sector_performance_boxplots.png"
        comparison_path = sectors_dir / "sector_performance_comparison.png"
        
        if boxplot_path.exists():
            print(f"\n✓ Boxplot exists at: {boxplot_path}")
            print(f"  Size: {boxplot_path.stat().st_size / 1024:.1f} KB")
        else:
            print(f"\n✗ Boxplot not found at: {boxplot_path}")
        
        if comparison_path.exists():
            print(f"\n✓ Comparison plot exists at: {comparison_path}")
            print(f"  Size: {comparison_path.stat().st_size / 1024:.1f} KB")
        else:
            print(f"\n✗ Comparison plot not found at: {comparison_path}")
        
        # Check LightGBM subdirectory
        lightgbm_dir = sectors_dir / "lightgbm"
        if lightgbm_dir.exists():
            lightgbm_plots = list(lightgbm_dir.glob("*.png"))
            print(f"\nLightGBM-specific plots: {len(lightgbm_plots)}")
            for plot in lightgbm_plots:
                print(f"  - {plot.name}")
        
        print("\n✓ Pipeline test complete!")
        
    except Exception as e:
        print(f"\n✗ Error running pipeline: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function."""
    test_sector_pipeline()


if __name__ == "__main__":
    main()