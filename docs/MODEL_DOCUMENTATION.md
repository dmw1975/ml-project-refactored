# Model-Specific Documentation

Documentation specific to individual ML models

## Table of Contents

1. [Readme Catboost](#readme-catboost)
2. [Readme Lightgbm](#readme-lightgbm)
3. [Tree Models Csv Analysis](#tree-models-csv-analysis)

---

## Readme Catboost

_Source: README_CATBOOST.md (root)_

# CatBoost Model Implementation

This document provides an overview of the CatBoost model implementation in this machine learning project.

## Overview

CatBoost is a powerful gradient boosting library that provides state-of-the-art performance for tabular data. It has been integrated into the project following the same pattern as XGBoost and LightGBM models to ensure consistency and maintainability.

## Features

- Basic CatBoost model with default parameters
- Hyperparameter optimization using Optuna
- Feature importance analysis
- Integration with residual analysis
- Support for different datasets (Base, Yeo, Base_Random, Yeo_Random)

## Implementation Details

The CatBoost implementation consists of the following components:

1. **catboost_model.py**: Main model implementation file containing:
   - `train_basic_catboost()`: Trains a CatBoost model with default parameters
   - `optimize_catboost_with_optuna()`: Finds optimal hyperparameters using Optuna
   - `train_catboost_models()`: Main function that handles training on all datasets

2. **Configuration in settings.py**:
   - `CATBOOST_PARAMS`: Dictionary of CatBoost-specific parameters
   - Color scheme for CatBoost models in visualizations

3. **Feature Importance Analysis**:
   - Integration with existing feature importance code in `evaluation/importance.py`
   - Support for CatBoost-specific feature importance extraction

4. **Residual Analysis**:
   - Compatible with existing residual plot generation in `visualization/create_residual_plots.py`

## Usage

### Training CatBoost Models

To train CatBoost models on all datasets:

```python
from models.catboost_model import train_catboost_models

# Train all CatBoost models
models = train_catboost_models()

# Or train models on specific datasets with custom trials
models = train_catboost_models(
    datasets=['CatBoost_Base', 'CatBoost_Yeo'],
    n_trials=100  # More trials for better optimization
)
```

### Testing

A dedicated test script (`test_catboost.py`) is available to verify the implementation:

```bash
python test_catboost.py
```

This will test:
- Basic model training
- Optuna hyperparameter optimization
- Full training pipeline
- Feature importance analysis

## Hyperparameter Optimization

CatBoost models are optimized using Optuna with the following hyperparameters:

- `learning_rate`: Controls the contribution of each tree
- `depth`: Maximum depth of the trees
- `l2_leaf_reg`: L2 regularization coefficient
- `bagging_temperature`: Bagging temperature parameter for random sampling
- `random_strength`: Amount of randomness to use in the model
- `border_count`: Number of splits for numerical features
- `iterations`: Number of boosting iterations
- `min_data_in_leaf`: Minimum number of samples in a leaf

## Integration with Existing Code

The CatBoost implementation follows the same patterns as XGBoost and LightGBM:

- Consistent model result structure
- Compatible with feature importance analysis
- Works with existing visualization code
- Supports sector-based analysis

## Requirements

- CatBoost library: `pip install catboost`

## Future Improvements

Potential future improvements:
- Add learning curves visualization for CatBoost models
- Implement CatBoost-specific categorical feature handling
- Explore advanced CatBoost features like monotonic constraints
- Add early stopping functionality for training
---

## Readme Lightgbm

_Source: README_LIGHTGBM.md (root)_

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
---

## Tree Models Csv Analysis

_Source: TREE_MODELS_CSV_ANALYSIS.md (root)_

# Analysis of combined_df_for_tree_models.csv

## Summary
You've successfully created a tree-model-friendly dataset with categorical features preserved! This is excellent for tree-based models.

## File Structure

### Dimensions
- **Rows**: 2,202 (same as original)
- **Columns**: 34 (vs 389 in the one-hot encoded version)
- **Size reduction**: 355 columns eliminated (91% reduction!)

### Categorical Columns (8 total)
1. `issuer_name` - Company names (2,202 unique)
2. `issuer_cntry_domicile_name` - Country names (43 unique) 
3. `cntry_of_risk` - Country codes (47 unique)
4. `gics_sector` - Sector classification (11 unique)
5. `gics_sub_ind` - Sub-industry classification (148 unique)
6. `top_1_shareholder_location` - Location codes (24 unique)
7. `top_2_shareholder_location` - Location codes (34 unique)
8. `top_3_shareholder_location` - Location codes (36 unique)

### Numeric Columns (26 total)
All the quantitative features including Yeo-Johnson transformed versions

## Comparison with Original

| Aspect | Original (One-Hot) | Tree Models (Categorical) |
|--------|-------------------|--------------------------|
| Total Columns | 389 | 34 |
| Categorical Columns | 0 (all one-hot) | 8 |
| One-Hot Columns | 336 | 0 |
| File Size | ~15-20 MB | ~1-2 MB |
| Tree Model Friendly | ❌ | ✅ |

## Benefits for Tree Models

1. **Native Categorical Handling**: Tree models can use their built-in categorical splitting algorithms
2. **Memory Efficiency**: 91% reduction in columns means faster training
3. **Better Splits**: Categorical features can be split optimally without binary constraints
4. **No Information Loss**: All category relationships preserved

## Integration with Pipeline

This file is perfect for:
- XGBoost with `enable_categorical=True`
- LightGBM with `categorical_feature` parameter
- CatBoost with native categorical support

## Minor Note
- The file has `issuer_cntry_domicile_name` instead of `issuer_cntry_domicile`
- This is likely the human-readable country name vs country code
- Both work fine for tree models

## Usage Example

```python
# Load for tree models
df = pd.read_csv('data/raw/combined_df_for_tree_models.csv')

# Identify categorical columns
categorical_cols = ['issuer_cntry_domicile_name', 'cntry_of_risk', 
                   'gics_sector', 'gics_sub_ind', 
                   'top_1_shareholder_location', 
                   'top_2_shareholder_location', 
                   'top_3_shareholder_location']

# Use with XGBoost
import xgboost as xgb
model = xgb.XGBRegressor(enable_categorical=True, tree_method='hist')

# Use with LightGBM
import lightgbm as lgb
model = lgb.LGBMRegressor()
model.fit(X, y, categorical_feature=categorical_cols)

# Use with CatBoost
from catboost import CatBoostRegressor
model = CatBoostRegressor(cat_features=categorical_cols)
```

This is an excellent addition to your data pipeline! It provides the ideal format for tree-based models to leverage their native categorical handling capabilities.
---

