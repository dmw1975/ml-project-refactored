# Main Pipeline Explanation: Best Models Implementation

## Overview
When you run `python main.py --all`, the pipeline executes a comprehensive workflow designed to train the best possible models using appropriate optimization techniques and feature representations.

## Model Training Strategy

### 1. **Linear Models**
**Scripts Used:**
- `models/linear_regression.py` → `train_all_models()`
- `models/elastic_net.py` → `train_elasticnet_models()`

**Rationale:**
- **Linear Regression**: Baseline unregularized models
- **ElasticNet**: Combines L1 and L2 regularization with Optuna optimization
  - Optuna searches for optimal `alpha` (regularization strength) and `l1_ratio` (L1 vs L2 balance)
  - Better than pure linear regression for high-dimensional data
  - Handles multicollinearity and feature selection

**Feature Encoding**: Always uses one-hot encoding (required for linear algebra)

### 2. **Tree-Based Models** 
**Default Behavior (Recommended)**: Native categorical features

#### XGBoost
**Script**: `models/xgboost_categorical.py` → `train_xgboost_categorical_models()`
- Now uses `enhanced_xgboost_categorical.py` with Optuna
- **Rationale**: Native categorical support in XGBoost provides:
  - Better split decisions on categorical variables
  - Reduced memory usage (no one-hot explosion)
  - Faster training
  - Optuna optimizes: max_depth, learning_rate, subsample, colsample_bytree, etc.

#### LightGBM
**Script**: `models/lightgbm_categorical.py` → `train_lightgbm_categorical_models()`
- Now uses `enhanced_lightgbm_categorical.py` with Optuna
- **Rationale**: LightGBM's native categorical handling:
  - Optimal split finding for categorical features
  - No need for label encoding or one-hot encoding
  - Optuna optimizes: num_leaves, learning_rate, feature_fraction, bagging_fraction, etc.

#### CatBoost
**Script**: `models/catboost_categorical.py` → `run_all_catboost_categorical()`
- Already has Optuna optimization built-in
- **Rationale**: CatBoost was designed for categorical features:
  - Advanced categorical encoding techniques
  - Handles high-cardinality features well
  - Optuna optimizes: depth, learning_rate, l2_leaf_reg, etc.

### 3. **Dataset Variants**
For each model type, the pipeline trains on 4 dataset variants:
1. **Base**: Original quantitative features + categorical features
2. **Yeo**: Yeo-Johnson transformed quantitative features + categorical features
3. **Base_Random**: Base + random feature (for importance baseline)
4. **Yeo_Random**: Yeo + random feature (for importance baseline)

**Rationale**: 
- Yeo-Johnson transformation normalizes skewed features
- Random feature provides baseline for feature importance validation

## Evaluation Pipeline

### 1. **Model Evaluation** (`evaluation/metrics.py`)
**What it does:**
- Calculates RMSE, MAE, MSE, R² for all models
- Creates residual analysis
- Performs statistical significance tests between models
- Generates consolidated metrics comparison table

**Rationale**: Comprehensive performance assessment across all metrics

### 2. **Feature Importance** (`evaluation/importance.py`)
**What it does:**
- Extracts feature importance from each model type
- Validates importance against random feature baseline
- Creates consolidated importance rankings

**Rationale**: Understanding which features drive predictions

### 3. **Multicollinearity** (`evaluation/multicollinearity.py`)
**What it does:**
- Calculates Variance Inflation Factors (VIF)
- Identifies highly correlated features

**Rationale**: Validates feature independence and model stability

## Visualization Pipeline

### 1. **Core Visualizations** (New Architecture)
- **Residual Plots**: Model fit quality assessment
- **Metrics Table**: The famous `metrics_summary_table.png` you wanted
- **Feature Importance**: Top features by model
- **Model Comparison**: Performance across model types
- **Statistical Tests**: Significance matrices

### 2. **Advanced Visualizations**
- **SHAP Values**: Model interpretability
- **Cross-Validation Plots**: Training stability
- **Optimization History**: Optuna convergence
- **Baseline Comparisons**: Model vs naive baselines
- **Sector Analysis**: Performance by business sector

## Pipeline Execution Flow

```
1. Data Loading
   ├── Linear models: One-hot encoded data
   └── Tree models: Native categorical data

2. Model Training (with optimization)
   ├── Linear Regression (baseline)
   ├── ElasticNet (Optuna: alpha, l1_ratio)
   ├── XGBoost (Optuna: tree parameters)
   ├── LightGBM (Optuna: tree parameters)
   └── CatBoost (Optuna: tree parameters)

3. Model Evaluation
   ├── Performance metrics (RMSE, R², MAE)
   ├── Feature importance analysis
   └── Statistical significance tests

4. Visualization Generation
   ├── Individual model plots
   ├── Comparative visualizations
   └── Summary tables and dashboards
```

## Key Benefits of This Pipeline

1. **Optimal Models**: Each model type uses the best optimization approach
   - ElasticNet for regularized linear models
   - Optuna for all tree-based models

2. **Appropriate Feature Encoding**:
   - One-hot for linear models (mathematical requirement)
   - Native categorical for trees (performance benefit)

3. **Comprehensive Evaluation**:
   - Multiple metrics for robust assessment
   - Statistical tests for significance
   - Feature importance validation

4. **Rich Visualizations**:
   - Everything from basic metrics to advanced interpretability
   - Publication-ready plots
   - Interactive dashboards where applicable

## Running the Pipeline

```bash
# Run everything with best settings
python main.py --all

# Run with specific optimizations
python main.py --all --optimize-xgboost 100 --optimize-lightgbm 100

# Force retrain even if models exist
python main.py --all --force-retune
```

This pipeline ensures you get the best possible models with comprehensive evaluation and visualization, all automated in a single command.