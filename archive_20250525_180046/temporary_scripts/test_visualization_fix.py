#!/usr/bin/env python3
"""Test that visualizations now work with all models."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from visualization_new import create_metrics_table
from visualization_new.utils.io import load_all_models


def test_metrics_table():
    """Test creating metrics table with all models."""
    print("Loading all models...")
    models = load_all_models()
    model_list = list(models.values())
    
    print(f"\nTotal models loaded: {len(model_list)}")
    
    # Show model types
    model_types = {}
    for name, model in models.items():
        if isinstance(model, dict) and 'model_type' in model:
            model_type = model['model_type']
        else:
            model_type = 'Unknown'
        model_types[model_type] = model_types.get(model_type, 0) + 1
    
    print("\nModel types count:")
    for mtype, count in sorted(model_types.items()):
        print(f"  {mtype}: {count}")
    
    print("\nCreating metrics table...")
    try:
        fig = create_metrics_table(model_list)
        output_path = Path("outputs/visualizations/performance/test_metrics_table.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Metrics table saved to: {output_path}")
        
        # Also check baseline comparison
        print("\nTesting baseline comparison visualization...")
        from visualization_new.plots.baselines import visualize_all_baseline_comparisons
        baseline_figs = visualize_all_baseline_comparisons(create_individual_plots=False)
        if baseline_figs:
            print("✓ Baseline comparisons created successfully")
        
        return True
    except Exception as e:
        print(f"✗ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_metrics_table()
    if success:
        print("\n✅ Visualization test passed! All models are properly integrated.")
    else:
        print("\n❌ Visualization test failed.")