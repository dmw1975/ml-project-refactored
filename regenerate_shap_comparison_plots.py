#!/usr/bin/env python3
"""
Regenerate SHAP comparison plots with consistent feature counts and updated titles.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from src.config import settings
from src.visualization.plots.shap_plots import create_separated_model_comparison_shap_plots
from src.utils.io import load_all_models


def main():
    """Regenerate SHAP comparison plots."""
    print("=" * 70)
    print("REGENERATING SHAP COMPARISON PLOTS WITH FIXES")
    print("=" * 70)
    
    try:
        # Load all models
        print("\nLoading all models...")
        models = load_all_models()
        print(f"Loaded {len(models)} models")
        
        # Filter for tree models only
        tree_models = {}
        for name, model_data in models.items():
            model_type = model_data.get('model_type', '').lower()
            if any(tree_type in model_type for tree_type in ['xgboost', 'lightgbm', 'catboost', 'xgb', 'lgb']):
                tree_models[name] = model_data
        
        print(f"Found {len(tree_models)} tree models for SHAP analysis")
        
        # Set output directory
        output_dir = settings.VISUALIZATION_DIR / "shap"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create SHAP comparison plots with fixes
        print("\nGenerating SHAP comparison plots...")
        print("Key improvements:")
        print("- Using only common features across models for fair comparison")
        print("- Showing top 15 features consistently for both Base and Yeo")
        print("- Updated titles to indicate feature count")
        
        paths = create_separated_model_comparison_shap_plots(tree_models, output_dir)
        
        if paths:
            print("\n" + "=" * 70)
            print("✓ SHAP COMPARISON PLOTS REGENERATED SUCCESSFULLY!")
            print("=" * 70)
            print("\nCreated plots:")
            for path in paths:
                print(f"  - {path.name}")
            print(f"\nAll plots saved to: {output_dir}")
        else:
            print("\n✗ Failed to create SHAP comparison plots")
            return 1
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())