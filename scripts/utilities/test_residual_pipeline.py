#!/usr/bin/env python3
"""
Test residual plot pipeline integration.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.visualization.comprehensive import create_comprehensive_visualizations
from src.config import settings


def test_residual_pipeline():
    """Test that residual plots are generated via the pipeline."""
    print("Testing residual plot pipeline integration...")
    print("="*60)
    
    # Get current residual plots
    residuals_dir = settings.VISUALIZATION_DIR / "residuals"
    before_plots = list(residuals_dir.glob("*.png")) if residuals_dir.exists() else []
    print(f"Residual plots before: {len(before_plots)}")
    
    # Run comprehensive visualizations (simulating --visualize flag)
    print("\nRunning comprehensive visualizations...")
    try:
        results = create_comprehensive_visualizations()
        
        # Check residual plots in results
        if 'residual_plots' in results:
            print(f"\nResidual plots created by pipeline: {len(results['residual_plots'])}")
            
            # Show a few examples
            if results['residual_plots']:
                print("Examples:")
                for plot in results['residual_plots'][:5]:
                    if hasattr(plot, 'name'):
                        print(f"  - {plot.name}")
                    else:
                        print(f"  - {plot}")
        else:
            print("\nNo residual_plots key in results!")
        
        # Check actual files
        after_plots = list(residuals_dir.glob("*.png"))
        print(f"\nResidual plots after: {len(after_plots)}")
        
        # Count by model type
        model_counts = {
            'catboost': 0,
            'xgboost': 0,
            'lightgbm': 0,
            'elasticnet': 0,
            'linear': 0
        }
        
        for plot in after_plots:
            plot_name = plot.stem.lower()
            if 'catboost' in plot_name:
                model_counts['catboost'] += 1
            elif 'xgboost' in plot_name:
                model_counts['xgboost'] += 1
            elif 'lightgbm' in plot_name:
                model_counts['lightgbm'] += 1
            elif 'elasticnet' in plot_name:
                model_counts['elasticnet'] += 1
            elif plot_name.startswith('lr_'):
                model_counts['linear'] += 1
        
        print("\nResidual plots by model type:")
        for model_type, count in model_counts.items():
            print(f"  {model_type}: {count} plots")
        
        # Verify CatBoost and XGBoost
        if model_counts['catboost'] == 0:
            print("\n⚠️  WARNING: No CatBoost residual plots found!")
        else:
            print(f"\n✓ CatBoost residual plots: {model_counts['catboost']}")
            
        if model_counts['xgboost'] == 0:
            print("⚠️  WARNING: No XGBoost residual plots found!")
        else:
            print(f"✓ XGBoost residual plots: {model_counts['xgboost']}")
        
        print("\n✓ Pipeline integration test complete!")
        
    except Exception as e:
        print(f"\n✗ Error running pipeline: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function."""
    test_residual_pipeline()


if __name__ == "__main__":
    main()