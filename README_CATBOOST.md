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