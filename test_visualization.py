"""Test script to verify visualization functionality."""

"""Simple test for residuals plot."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings
from visualization.metrics_plots import plot_residuals

def test_residuals_only():
    """Test just the residuals plot."""
    print("Testing residuals plot function...")
    try:
        plot_residuals()
        print("Residuals plot completed successfully.")
    except Exception as e:
        print(f"Error in residuals plot: {e}")
        import traceback
        traceback.print_exc()

def test_metrics_summary_table():
    """Test the metrics summary table function."""
    print("Testing metrics summary table function...")
    try:
        from visualization.metrics_plots import plot_metrics_summary_table
        plot_metrics_summary_table()
        print("Metrics summary table completed successfully.")
    except Exception as e:
        print(f"Error in metrics summary table: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_residuals_only()