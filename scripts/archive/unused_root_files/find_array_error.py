#!/usr/bin/env python3
"""Find the exact location of the array comparison error."""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Test the annotations module
from src.visualization.components.annotations import add_value_labels
import matplotlib.pyplot as plt

def test_add_value_labels():
    """Test if add_value_labels causes the error with arrays."""
    fig, ax = plt.subplots()
    
    # Create a bar plot with array heights (this might trigger the error)
    x = np.array([1, 2, 3])
    heights = np.array([10, 20, 30])
    
    bars = ax.bar(x, heights)
    
    # Now test add_value_labels which has the problematic code
    try:
        add_value_labels(ax)
        print("✓ add_value_labels works correctly after fix")
    except ValueError as e:
        if "truth value" in str(e) and "array" in str(e):
            print("✗ Found the array comparison error in add_value_labels!")
            print(f"  Error: {e}")
            return False
    except Exception as e:
        print(f"✗ Different error in add_value_labels: {type(e).__name__}: {e}")
        return False
    
    return True

def test_metrics_comparison():
    """Test metrics comparison visualization which might use add_value_labels."""
    from src.visualization.plots.metrics import plot_metrics_comparison
    from src.visualization.utils.io import load_all_models
    
    print("\nTesting metrics comparison visualization...")
    
    try:
        models = load_all_models()
        catboost_models = {k: v for k, v in models.items() if 'catboost' in k.lower()}
        
        if catboost_models:
            config = {'save': False, 'show': False}
            fig = plot_metrics_comparison(catboost_models, config=config)
            print("✓ Metrics comparison works correctly")
            return True
    except ValueError as e:
        if "truth value" in str(e) and "array" in str(e):
            print("✗ Found the array comparison error in metrics comparison!")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"✗ Different error: {type(e).__name__}: {e}")
        return False
    
    return True

def main():
    """Run tests to find array comparison error."""
    print("Testing for array comparison errors...")
    print("=" * 60)
    
    # Test 1: Direct test of add_value_labels
    print("\nTest 1: add_value_labels function")
    test_add_value_labels()
    
    # Test 2: Metrics comparison which might use it
    print("\nTest 2: Metrics comparison visualization")
    test_metrics_comparison()

if __name__ == "__main__":
    main()