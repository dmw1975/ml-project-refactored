#!/usr/bin/env python
"""
Test script for cross-model feature importance visualization by dataset.

This script generates visualizations that compare feature importance
across different model types (LightGBM, XGBoost, CatBoost, ElasticNet, etc.)
for each dataset type (Base, Yeo, Base_Random, Yeo_Random).
"""

import sys
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from visualization_new.viz_factory import create_cross_model_feature_importance_by_dataset


def main():
    """Run the cross-model feature importance analysis."""
    print("Generating cross-model feature importance visualizations by dataset...")
    
    # Print loaded models before feature extraction
    from visualization_new.utils.io import load_all_models
    
    print("\nLoading all models...")
    all_models = load_all_models()
    
    print("\nModels loaded:")
    for model_name in all_models.keys():
        print(f"  - {model_name}")
    
    # Count ElasticNet models
    elasticnet_models = [name for name in all_models.keys() if 'elasticnet' in name.lower()]
    print(f"\nFound {len(elasticnet_models)} ElasticNet models:")
    for model in elasticnet_models:
        print(f"  - {model}")
    
    # Create visualizations
    print("\nCreating cross-model feature importance visualizations...")
    output_paths = create_cross_model_feature_importance_by_dataset(
        format='png',
        dpi=300,
        show=False
    )
    
    # Print the result paths
    print("\nGenerated visualizations:")
    for dataset, paths in output_paths.items():
        print(f"\nDataset: {dataset}")
        for plot_type, path in paths.items():
            print(f"  - {plot_type}: {path}")
            
    # Verify the files were created
    print("\nVerifying files were created:")
    for dataset, paths in output_paths.items():
        for plot_type, path in paths.items():
            if Path(path).exists():
                print(f"  ✓ {path} exists")
            else:
                print(f"  ✗ {path} does not exist")


if __name__ == "__main__":
    main()