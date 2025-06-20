#!/bin/bash
# Quick runner script for XGBoost feature removal analysis

echo "Starting XGBoost Feature Removal Analysis"
echo "========================================"
echo ""
echo "This analysis will:"
echo "1. Train XGBoost models with and without 'top_3_shareholder_percentage'"
echo "2. Generate comprehensive visualizations"
echo "3. Create performance comparison metrics"
echo ""
echo "Output directory: outputs/feature_removal_experiment/"
echo ""

# Run the analysis
python xgboost_feature_removal_analysis.py

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Analysis completed successfully!"
    echo ""
    echo "View results in:"
    echo "- Metrics comparison: outputs/feature_removal_experiment/metrics/feature_removal_comparison.csv"
    echo "- Visualizations: outputs/feature_removal_experiment/visualizations/"
    echo "- Summary report: outputs/feature_removal_experiment/ANALYSIS_REPORT.md"
else
    echo ""
    echo "Analysis failed. Check logs for details."
fi