"""
Test script for creative dataset comparison visualizations.

This script generates and saves all the visualizations implemented in
visualization_new/plots/creative_dataset_comparison.py.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config import settings
from visualization_new.utils.io import load_all_models, ensure_dir
from visualization_new.core.registry import get_adapter_for_model
from visualization_new.plots.creative_dataset_comparison import (
    create_radar_chart,
    create_performance_delta,
    create_parallel_coordinates,
    create_visual_leaderboard,
    create_sunburst_chart
)


def main():
    """Generate and save all creative visualizations."""
    # Print section header
    print("\n" + "="*80)
    print("GENERATING CREATIVE DATASET COMPARISON VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Create output directory
    output_dir = settings.VISUALIZATION_DIR / "creative_visualizations"
    ensure_dir(output_dir)
    
    # Load all models
    print("Loading models...")
    all_models = load_all_models()
    raw_models = list(all_models.values())
    print(f"Loaded {len(raw_models)} models")
    
    # Convert raw models to model adapters
    print("Converting models to adapters...")
    models = []
    for model_data in raw_models:
        try:
            adapter = get_adapter_for_model(model_data)
            models.append(adapter)
        except Exception as e:
            print(f"Could not create adapter for model {model_data.get('model_name', 'Unknown')}: {e}")
            
    print(f"Created {len(models)} model adapters")
    
    # Get unique datasets
    datasets = set()
    for model in models:
        metadata = model.get_metadata()
        model_name = metadata.get('model_name', '')
            
        if 'Base_Random' in model_name:
            datasets.add('Base_Random')
        elif 'Yeo_Random' in model_name:
            datasets.add('Yeo_Random')
        elif 'Base' in model_name:
            datasets.add('Base')
        elif 'Yeo' in model_name:
            datasets.add('Yeo')
    
    print(f"Found datasets: {', '.join(datasets)}")
    
    # Configuration with output directory
    config = {
        'output_dir': output_dir,
        'dpi': 300,
        'format': 'png'
    }
    
    # 1. Generate radar charts for each dataset
    print("\nGenerating radar charts...")
    for dataset in datasets:
        dataset_config = config.copy()
        dataset_config['dataset'] = dataset
        
        print(f"  - {dataset} dataset")
        fig = create_radar_chart(models, dataset_config)
        plt.close(fig)
    
    # 2. Generate performance delta visualizations
    print("\nGenerating performance delta visualizations...")
    for dataset in datasets:
        dataset_config = config.copy()
        dataset_config['dataset'] = dataset
        
        print(f"  - {dataset} dataset")
        fig = create_performance_delta(models, dataset_config)
        plt.close(fig)
    
    # 3. Generate parallel coordinates plots
    print("\nGenerating parallel coordinates plots...")
    for dataset in datasets:
        dataset_config = config.copy()
        dataset_config['dataset'] = dataset
        
        print(f"  - {dataset} dataset")
        fig = create_parallel_coordinates(models, dataset_config)
        plt.close(fig)
    
    # 4. Generate visual leaderboards
    print("\nGenerating visual leaderboards...")
    for dataset in datasets:
        dataset_config = config.copy()
        dataset_config['dataset'] = dataset
        
        print(f"  - {dataset} dataset")
        fig = create_visual_leaderboard(models, dataset_config)
        plt.close(fig)
    
    # 5. Generate sunburst charts
    print("\nGenerating sunburst charts...")
    
    # Generate one for each metric
    for metric in ['RMSE', 'MAE', 'R2', 'MSE']:
        print(f"  - {metric} metric")
        metric_config = config.copy()
        metric_config['metric'] = metric
        
        fig = create_sunburst_chart(models, metric_config)
        plt.close(fig)
    
    print("\nAll visualizations created successfully!")
    print(f"Files saved to: {output_dir}")


if __name__ == "__main__":
    main()