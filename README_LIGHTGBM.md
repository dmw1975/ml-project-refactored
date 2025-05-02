# LightGBM Implementation

This document describes the implementation of LightGBM models in the ML project, following the same patterns as the XGBoost implementation.

## Overview

LightGBM has been integrated into the existing ML project architecture, providing both basic and Optuna-optimized variants for different feature sets (Base, Yeo, Base_Random, Yeo_Random). The implementation includes model training, hyperparameter optimization, evaluation, and visualization.

## Files Modified/Added

1. **models/lightgbm_model.py**
   - Implements LightGBM model training and optimization
   - Handles both basic and Optuna-optimized versions
   - Special handling for feature names to avoid JSON character issues

2. **visualization/lightgbm_plots.py**
   - Visualizes LightGBM model performance
   - Creates comparison charts between basic and optimized models
   - Visualizes hyperparameter optimization process
   - Compares LightGBM with other model types

3. **main.py**
   - Updated with command-line arguments for LightGBM models
   - Integrated LightGBM into the main workflow

4. **evaluation/metrics.py**
   - Updated to load and evaluate LightGBM models 
   - Added to model comparison functionality

5. **evaluation/importance.py**
   - Enhanced to support LightGBM feature importance
   - Special handling for feature name mapping

6. **visualization/create_residual_plots.py**
   - Updated to include LightGBM in residual analysis

## Usage

You can train and visualize LightGBM models using the following commands:

```bash
# Train LightGBM models
python main.py --train-lightgbm

# Train with Optuna optimization (specify trials)
python main.py --optimize-lightgbm 50

# Generate visualizations
python main.py --visualize-lightgbm

# Run the full pipeline with all models
python main.py --all
```

For testing and development, use the helper scripts:

```bash
# Train and evaluate a single LightGBM model
python test_lightgbm_importance.py

# Run all LightGBM operations (training, importance, visualization)
python test_run_all_lightgbm.py

# Compare all model types
python test_model_comparison.py
```

## Technical Notes

1. **Feature Name Handling**:
   - LightGBM has strict requirements for feature names and doesn't support special JSON characters
   - The implementation uses numeric feature names internally (feature_0, feature_1, etc.)
   - A mapping is maintained to original feature names for interpretability

2. **Hyperparameter Optimization**:
   - Uses Optuna with 5-fold cross-validation
   - Optimizes key parameters like num_leaves, learning_rate, min_child_samples, etc.
   - Results are saved with optimization history for analysis

3. **Integration with Existing Architecture**:
   - Follows the same patterns as XGBoost implementation
   - Maintains consistent directory structure and naming conventions
   - Ensures compatibility with existing evaluation and visualization workflows

4. **Feature Importance**:
   - Uses LightGBM's native feature_importance() method
   - Maps back to original feature names for interpretability
   - Saves importance results in the same format as other models

## Model Performance

LightGBM models can be compared with existing models (Linear Regression, ElasticNet, XGBoost) using the visualization tools. Key metrics include:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)

The comparison charts in the summary visualization directory show relative performance across all model types.