"""Test script to verify visualization functionality.

This script contains tests for both the old (deprecated) and new visualization systems.
Use the new visualization architecture for new code.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings

# Old visualization system tests (DEPRECATED)
def test_old_residuals():
    """Test residuals plot with old system (DEPRECATED)."""
    print("Testing old residuals plot function (DEPRECATED)...")
    try:
        from visualization.metrics_plots import plot_residuals
        plot_residuals()
        print("Old residuals plot completed successfully.")
    except Exception as e:
        print(f"Error in old residuals plot: {e}")
        import traceback
        traceback.print_exc()

def test_old_metrics_summary_table():
    """Test metrics table with old system (DEPRECATED)."""
    print("Testing old metrics summary table function (DEPRECATED)...")
    try:
        from visualization.metrics_plots import plot_metrics_summary_table
        plot_metrics_summary_table()
        print("Old metrics summary table completed successfully.")
    except Exception as e:
        print(f"Error in old metrics summary table: {e}")
        import traceback
        traceback.print_exc()

# New visualization system tests
def test_new_residuals():
    """Test residual plot with new visualization architecture."""
    print("Testing new residual plot function...")
    try:
        import visualization_new as viz
        viz.create_residual_plot('XGB_Base_optuna')
        print("New residual plot completed successfully.")
    except Exception as e:
        print(f"Error in new residual plot: {e}")
        import traceback
        traceback.print_exc()

def test_new_metrics_table():
    """Test metrics table with new visualization architecture."""
    print("Testing new metrics table function...")
    try:
        import visualization_new as viz
        from visualization_new.utils.io import load_all_models
        models = load_all_models()
        viz.create_metrics_table(list(models.values()))
        print("New metrics table completed successfully.")
    except Exception as e:
        print(f"Error in new metrics table: {e}")
        import traceback
        traceback.print_exc()

def test_dashboard():
    """Test dashboard with new visualization architecture."""
    print("Testing new dashboard function...")
    try:
        import visualization_new as viz
        viz.create_comparative_dashboard()
        print("New dashboard completed successfully.")
    except Exception as e:
        print(f"Error in new dashboard: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run tests for new visualization system
    test_new_residuals()
    test_new_metrics_table()
    test_dashboard()
    
    # Optionally run old visualization tests
    # test_old_residuals()
    # test_old_metrics_summary_table()