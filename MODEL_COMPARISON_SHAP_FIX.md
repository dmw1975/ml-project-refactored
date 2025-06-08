# Model Comparison SHAP Plot Fix

## Issue
The `model_comparison_shap.png` plot was not being generated when running `main.py --visualize` or `main.py --all`.

## Root Cause
The `generate_shap_visualizations.py` script was creating individual SHAP plots for each model but was not calling the `create_model_comparison_shap_plot()` function that generates the cross-model comparison heatmap.

## Solution Implemented

### 1. Updated `generate_shap_visualizations.py`
- Added import for `create_model_comparison_shap_plot` function
- Added code at the end of `main()` to generate the comparison plot after processing individual models

### 2. Created Fallback Script
Created `generate_model_comparison_shap_only.py` as a lightweight script that only generates the comparison plot without processing all individual models (useful if the main script times out).

### 3. Enhanced Pipeline Integration
Updated `src/pipelines/visualization.py` to:
- Handle timeouts gracefully in SHAP generation
- Always attempt to create the model comparison plot as a separate step
- Provide clear feedback on success/failure

## The Model Comparison SHAP Plot
This plot shows:
- Feature importance comparison across CatBoost, LightGBM, and XGBoost
- Top 15 features ranked by maximum importance
- Normalized values (0-1 scale) for easy comparison
- Heatmap visualization with annotations
- Excludes "Random" models for cleaner comparison

## Usage
The plot is now generated automatically when running:
```bash
python main.py --visualize
python main.py --all
```

Or manually:
```bash
python scripts/utilities/generate_model_comparison_shap_only.py
```

## Location
The plot is saved to: `outputs/visualizations/shap/model_comparison_shap.png`