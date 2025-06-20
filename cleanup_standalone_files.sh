#!/bin/bash
# Cleanup script to remove standalone files that aren't integrated into pipeline

echo "FILE CLEANUP - Removing standalone diagnostic and fix scripts"
echo "============================================================"

# Diagnostic test scripts
echo "Removing diagnostic test scripts..."
rm -f test_catboost_feature_importance.py
rm -f test_catboost_models_structure.py
rm -f test_baseline_viz.py
rm -f test_visualization_outputs.py
rm -f test_timing.py
rm -f verify_baseline_comparisons.py
rm -f verify_family_plots.py
rm -f verify_metrics_table.py

# Timing demonstration scripts
echo "Removing timing demonstration scripts..."
rm -f prove_timing_issue.py
rm -f prove_timing_with_logs.py
rm -f demonstrate_fresh_run_issue.py

# Abandoned or applied fix scripts
echo "Removing abandoned/applied fix scripts..."
rm -f fix_baseline_consistency.py
rm -f fix_baseline_evaluation.py
rm -f fix_baseline_viz_adapter.py
rm -f fix_comprehensive_visualizations.py
rm -f fix_cv_distributions_properly.py
rm -f fix_metrics_table_final.py
rm -f fix_metrics_table_properly.py
rm -f fix_remaining_visualizations.py

# Pipeline order fix scripts (already applied)
echo "Removing pipeline order fix scripts..."
rm -f fix_pipeline_order.py
rm -f fix_pipeline_order_precise.py
rm -f add_model_tracking.py

# Temporary and backup files
echo "Removing temporary files..."
rm -f main_fixed.py

# Scripts that should be integrated but aren't
echo "The following scripts contain useful logic but aren't integrated:"
echo "- create_baseline_plots.py (should be part of visualization pipeline)"
echo "- run_missing_visualizations.py (temporary workaround)"
echo ""
echo "Consider integrating their functionality into the main pipeline."

echo ""
echo "Cleanup complete. Removed standalone diagnostic and fix scripts."
echo "Only integrated components remain in the src/ directory."