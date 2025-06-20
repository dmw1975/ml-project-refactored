# XGBoost Feature Removal Analysis

## Overview

This analysis evaluates the impact of removing the `top_3_shareholder_percentage` feature from XGBoost models. The implementation is completely isolated from the main pipeline to ensure no interference with existing models or data.

**Important**: The feature exists in two versions in the dataset:
- `top_3_shareholder_percentage` (raw feature)
- `yeo_joh_top_3_shareholder_percentage` (Yeo-Johnson transformed)

Both versions must be removed for a proper analysis.

## Running the Analysis

### Option 1: Using Main Pipeline Command
```bash
python main.py --xgboost-feature-removal
```

### Option 2: Running Corrected Standalone Script
```bash
python xgboost_feature_removal_corrected_v2.py
```

### Option 3: Running Original Analysis Script
```bash
python xgboost_feature_removal_analysis.py
```

## What the Corrected Analysis Does

1. **Data Preparation**:
   - Loads the unified tree models dataset
   - Removes BOTH versions of the feature (raw and Yeo-Johnson transformed)
   - Uses the exact same train/test splits as the original models
   - Maintains complete isolation from main pipeline data

2. **Model Training**:
   - Only trains models WITHOUT features (we already have the WITH feature models)
   - Tests on Optuna-optimized models only:
     - XGBoost_Base_Random_categorical_optuna
     - XGBoost_Yeo_Random_categorical_optuna
   - Uses the same optimized hyperparameters from existing models
   - Total: 2 new models trained (without features only)

3. **Visualization Generation**:
   - Uses main pipeline visualization standards for consistency
   - **Residual Plots**: Shows prediction errors for each model
   - **Performance Metrics**: MAE, MSE, RMSE, R² comparisons
   - **SHAP Analysis**: Feature importance and interactions
   - **Model Comparisons**: Side-by-side performance analysis with proper warnings

4. **Metrics Comparison**:
   - Compares against existing model metrics (no retraining needed)
   - Generates CSV with comprehensive metrics comparison
   - Calculates performance changes and percentage impacts

## Output Structure

```
outputs/feature_removal_experiment/
├── models/                          # Trained model files
│   ├── XGBoost_*_base_original.pkl
│   └── XGBoost_*_base_feature_removed.pkl
├── visualizations/
│   ├── residuals/                   # Residual analysis plots
│   ├── performance/                 # Performance metric plots
│   ├── shap/                       # SHAP visualizations
│   │   └── [model_name]/           # Per-model SHAP plots
│   └── comparisons/                # Model comparison plots
├── metrics/
│   └── feature_removal_comparison.csv  # Performance comparison table
├── logs/                           # Execution logs
└── ANALYSIS_REPORT.md              # Summary report
```

## Key Features

### Complete Isolation
- Uses separate output directory
- Independent data loading
- No shared state with main pipeline
- Custom visualization configuration

### Reuses Existing Infrastructure
- Leverages existing visualization functions
- Uses standard model training code
- Maintains consistent plot formatting
- Compatible with all existing analysis tools

### Safety Features
- Creates data copies (never modifies originals)
- Isolated state management
- Custom output paths prevent overwrites
- Comprehensive error handling

## Interpreting Results

### Metrics Comparison CSV
The `feature_removal_comparison.csv` contains:
- `dataset`: Which dataset variant was used
- `model_type`: base or optuna
- `original_rmse`: RMSE with all features
- `removed_rmse`: RMSE without the feature
- `rmse_change`: Difference (positive = worse performance)
- `original_r2`: R² with all features
- `removed_r2`: R² without the feature
- `r2_change`: Difference (negative = worse performance)

### Expected Outcomes
- **Minimal change**: Feature has low importance
- **Significant degradation**: Feature is important for predictions
- **Improvement**: Feature was adding noise (rare but possible)

## Technical Details

### Data Isolation
```python
# Original data is never modified
X, y = load_tree_models_data()
X_removed = X.copy()  # Create isolated copy
X_removed = X_removed.drop(columns=[excluded_feature])
```

### Visualization Isolation
```python
# Custom output directory for all plots
viz_config = VisualizationConfig(
    output_dir=Path("outputs/feature_removal_experiment/visualizations")
)
create_residual_plots(models, config=viz_config)
```

### State Management Isolation
The analysis bypasses the main pipeline state manager, ensuring no interference with ongoing pipeline operations.

## Extending the Analysis

To analyze different features:
1. Modify the `excluded_feature` parameter
2. Ensure the feature exists in the dataset
3. Run the analysis with custom output directory

To add more datasets:
1. Modify the `datasets` list in the analyzer
2. Ensure datasets are available in the data loading function

## Troubleshooting

### Common Issues

1. **Feature not found**: Check feature name matches exactly
2. **Memory issues**: Reduce Optuna trials or use one dataset at a time
3. **Visualization errors**: Ensure all dependencies are installed

### Debug Mode
Check logs in `outputs/feature_removal_experiment/logs/` for detailed execution information.

## Summary

This isolated analysis provides a safe way to evaluate feature importance through removal experiments without affecting the main ML pipeline. All outputs are contained in a separate directory structure, making it easy to compare results and clean up after analysis.