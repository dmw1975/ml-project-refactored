# ML Project Pipeline - ESG Score Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Commands Reference](#commands-reference)
  - [Complete Pipeline](#complete-pipeline)
  - [Model Training](#model-training)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
  - [Model Evaluation](#model-evaluation)
  - [Visualization](#visualization)
  - [Feature Analysis](#feature-analysis)
  - [Data Options](#data-options)
- [Model Types](#model-types)
- [Feature Engineering](#feature-engineering)
- [Configuration](#configuration)
- [Output Structure](#output-structure)
- [Troubleshooting](#troubleshooting)
- [Known Issues](#known-issues)
- [Best Practices](#best-practices)
- [Contributing](#contributing)

## Data Setup Required

**IMPORTANT**: Before running this pipeline, you must copy the required data files:

1. Copy all files from `esg-score-eda/data/ml_output/` to `/mnt/d/ml_project_refactored/data/raw/`
2. Required files include:
   - `combined_df_for_linear_models.csv`
   - `combined_df_for_tree_models.csv`
   - `score.csv`

```bash
# Example copy command (adjust paths as needed):
cp /path/to/esg-score-eda/data/ml_output/* /mnt/d/ml_project_refactored/data/raw/
```

## Overview

This ML pipeline is designed for ESG (Environmental, Social, and Governance) score prediction using multiple model types with comprehensive evaluation and visualization capabilities. The pipeline supports automated training, hyperparameter optimization, and extensive model comparison features.

### Key Objectives

- Predict ESG scores using various machine learning models
- Compare linear and tree-based model performance
- Analyze feature importance and model interpretability
- Provide sector-specific predictions
- Generate comprehensive visualizations for model analysis

## Features

- **Multiple Model Types**: Linear Regression, ElasticNet, XGBoost, LightGBM, CatBoost
- **Automated Hyperparameter Tuning**: Optuna integration for optimal model performance
- **Native Categorical Support**: Tree models use native categorical features
- **Sector-Specific Models**: Train separate models for each business sector
- **Comprehensive Evaluation**: Multiple metrics, baseline comparisons, statistical tests
- **Rich Visualizations**: SHAP values, residual plots, feature importance, model comparisons
- **Unified Train/Test Split**: Ensures consistent data splits across all models
- **State Management**: Track pipeline progress and resume interrupted runs

## Architecture

```
ml_project_refactored/
├── main.py                         # Main pipeline entry point
├── xgboost_feature_removal_proper.py  # Feature removal analysis
├── src/                           # Core source code
│   ├── config/                    # Configuration files
│   ├── data/                      # Data loading and processing
│   ├── models/                    # Model implementations
│   ├── evaluation/                # Model evaluation modules
│   ├── pipelines/                 # Pipeline orchestration
│   ├── utils/                     # Utility functions
│   └── visualization/             # Visualization system
├── data/                          # Data files (CSV, PKL)
├── outputs/                       # Generated outputs
│   ├── models/                    # Trained models
│   ├── metrics/                   # Evaluation metrics
│   ├── feature_importance/        # Feature analysis
│   └── visualizations/            # Generated plots
├── logs/                          # Execution logs
└── scripts/                       # Utility scripts
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Step-by-Step Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd ml_project_refactored
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify installation**:
```bash
python main.py --help
```

### Dependencies

Key packages include:
- **ML Libraries**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Optimization**: Optuna
- **Visualization**: matplotlib, seaborn, SHAP
- **Data Processing**: pandas, numpy
- **Utilities**: tqdm, joblib

## Quick Start

### First-Time Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd ml-project-refactored
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify setup** (optional but recommended):
```bash
python scripts/utilities/verify_setup.py
```

4. **Add raw data files** to `data/raw/`:
   - Copy `combined_df_for_linear_models.csv`
   - Copy `combined_df_for_tree_models.csv`
   - Copy `score.csv`
   - Copy metadata JSON files to `data/raw/metadata/`

5. **Generate processed datasets**:
```bash
python scripts/utilities/create_categorical_datasets.py
```
Note: This step is now automatic - main.py will run it if needed.

6. **Run the complete pipeline**:
```bash
python main.py --all
```

### Subsequent Runs

After initial setup, you can simply run:
```bash
python main.py --all
```

2. **Train specific models with optimization**:
```bash
python main.py --train-xgboost --optimize-xgboost 100
```

3. **Generate visualizations for existing models**:
```bash
python main.py --visualize
```

4. **Run sector-specific analysis**:
```bash
python main.py --all-sector
```

### Common Workflows

**Workflow 1: Initial Model Training**
```bash
# Train all models with default settings
python main.py --train --train-linear

# Evaluate and visualize results
python main.py --evaluate --visualize
```

**Workflow 2: Hyperparameter Optimization**
```bash
# Optimize tree models with Optuna
python main.py --optimize-xgboost 100 --optimize-lightgbm 100 --optimize-catboost 50

# Force retrain with new parameters
python main.py --train --force-retune
```

**Workflow 3: Feature Analysis**
```bash
# Run feature removal analysis
python main.py --xgboost-feature-removal

# Analyze feature importance
python main.py --importance --vif
```

## Commands Reference

### Complete Pipeline

#### `--all`
**Purpose**: Run the entire ML pipeline (train, evaluate, visualize)  
**Usage**: 
```bash
python main.py --all
```
**Output**: Complete model training, evaluation metrics, and all visualizations  
**Notes**: This is the most comprehensive option, running all components

#### `--all-sector`
**Purpose**: Run complete sector-specific model pipeline  
**Usage**: 
```bash
python main.py --all-sector
```
**Output**: Sector models, sector-specific metrics and visualizations

#### `--sector-only`
**Purpose**: Run only sector models, skipping standard models  
**Usage**: 
```bash
python main.py --sector-only --all-sector
```

### Model Training

#### `--train`
**Purpose**: Train all tree-based models (XGBoost, LightGBM, CatBoost)  
**Usage**: 
```bash
python main.py --train
```
**Output**: Trained models saved to `outputs/models/`

#### `--train-linear`
**Purpose**: Train linear regression models  
**Usage**: 
```bash
python main.py --train-linear
```
**Warning**: Linear models may show poor performance due to pre-normalized data

#### Model-Specific Training

- `--train-xgboost`: Train only XGBoost models
- `--train-lightgbm`: Train only LightGBM models  
- `--train-catboost`: Train only CatBoost models
- `--train-sector`: Train sector-specific linear regression
- `--train-sector-lightgbm`: Train sector-specific LightGBM

**Example**:
```bash
python main.py --train-xgboost --train-lightgbm
```

### Hyperparameter Optimization

#### Optuna-based Optimization

**Commands**:
- `--optimize-elasticnet N`: Optimize ElasticNet (default: 100 trials)
- `--optimize-xgboost N`: Optimize XGBoost (default: 50 trials)
- `--optimize-lightgbm N`: Optimize LightGBM (default: 50 trials)
- `--optimize-catboost N`: Optimize CatBoost (default: 50 trials)

**Usage Example**:
```bash
python main.py --optimize-xgboost 100 --optimize-lightgbm 100
```

**Additional Options**:
- `--force-retune`: Force recreation of Optuna studies
- `--check-studies`: Report existing studies without training
- `--elasticnet-grid`: Use grid search instead of Optuna (legacy)

### Model Evaluation

#### `--evaluate`
**Purpose**: Evaluate all trained models  
**Usage**: 
```bash
python main.py --evaluate
```
**Output**: 
- Metrics CSV files in `outputs/metrics/`
- Baseline comparison results
- Model performance statistics

#### Evaluation Variants
- `--evaluate-sector`: Evaluate sector-specific models
- `--evaluate-sector-lightgbm`: Evaluate sector LightGBM models

### Visualization

#### `--visualize`
**Purpose**: Generate comprehensive visualizations  
**Usage**: 
```bash
python main.py --visualize
```
**Output** in `outputs/visualizations/`:
- Residual plots
- Feature importance plots
- SHAP value visualizations
- Model comparison plots
- Metrics summary tables
- Cross-validation distributions
- Statistical test results

#### Model-Specific Visualization
- `--visualize-xgboost`: XGBoost-specific plots
- `--visualize-lightgbm`: LightGBM-specific plots
- `--visualize-catboost`: CatBoost-specific plots
- `--visualize-sector`: Sector model visualizations

### Feature Analysis

#### `--importance`
**Purpose**: Analyze feature importance across models  
**Usage**: 
```bash
python main.py --importance
```
**Output**: Feature importance rankings and plots

#### `--vif`
**Purpose**: Calculate Variance Inflation Factors for multicollinearity  
**Usage**: 
```bash
python main.py --vif
```
**Output**: VIF analysis results

#### `--xgboost-feature-removal`
**Purpose**: Run XGBoost feature removal analysis  
**Usage**: 
```bash
python main.py --xgboost-feature-removal
```
**Notes**: Analyzes impact of removing specific features

### Data Options

#### `--datasets`
**Purpose**: Specify which datasets to use  
**Options**: 
- `all` (default): Use all datasets
- `LR_Base`: Base features only
- `LR_Yeo`: Yeo-Johnson transformed features
- `LR_Base_Random`: Base + random feature
- `LR_Yeo_Random`: Yeo + random feature

**Usage**: 
```bash
python main.py --train --datasets LR_Base LR_Yeo
```

#### `--use-one-hot`
**Purpose**: Force one-hot encoding for tree models  
**Default**: Tree models use native categorical features  
**Usage**: 
```bash
python main.py --train --use-one-hot
```

### Utility Options

#### `--non-interactive`
**Purpose**: Run without user prompts (for automation)  
**Usage**: 
```bash
python main.py --all --non-interactive
```

## Model Types

### Linear Models

**Linear Regression**
- Standard OLS regression
- Uses one-hot encoded categorical features
- Four variants: Base, Yeo, Base+Random, Yeo+Random

**ElasticNet**
- L1 + L2 regularization
- Optuna hyperparameter optimization
- Handles high-dimensional data better than standard regression

### Tree-Based Models

**XGBoost**
- Gradient boosting with native categorical support
- Optuna optimization for hyperparameters
- High performance on tabular data

**LightGBM**
- Fast gradient boosting
- Efficient handling of categorical features
- Lower memory usage than XGBoost

**CatBoost**
- Specialized for categorical features
- Robust to overfitting
- No need for extensive preprocessing

### Sector Models

- Separate models trained for each business sector
- Available for LinearRegression, ElasticNet, and LightGBM
- Captures sector-specific patterns

## Feature Engineering

### Yeo-Johnson Transformation

The pipeline supports Yeo-Johnson power transformations for numerical features:
- Reduces skewness in features
- Improves linear model performance
- Applied to create "Yeo" dataset variants

### Feature Sets

1. **Base Features**: Original numerical and categorical features
2. **Yeo Features**: Yeo-Johnson transformed numerical features
3. **Random Feature**: Added for baseline comparison
4. **Categorical Features**: Handled natively by tree models or one-hot encoded for linear models

### Feature Removal Analysis

The `xgboost_feature_removal_proper.py` script analyzes:
- Impact of removing specific features
- Model performance comparison with/without features
- SHAP value changes after feature removal

## Configuration

### Main Configuration (`src/config/settings.py`)

Key settings include:
- Data paths and directories
- Model parameters
- Random seeds for reproducibility
- Visualization settings
- Color schemes for plots

### Hyperparameters (`src/config/hyperparameters.py`)

Default hyperparameters for each model type:
```python
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'enable_categorical': True,
    ...
}
```

### Data Configuration

The pipeline supports two data loading mechanisms:
1. **JSON Metadata** (preferred): Uses `data/metadata/` files
2. **Pickle Files** (legacy): Falls back to `data/pkl/` files

## Output Structure

```
outputs/
├── models/                      # Saved model files
│   ├── linear_regression_models.pkl
│   ├── elasticnet_models.pkl
│   ├── xgboost_models.pkl
│   ├── lightgbm_models.pkl
│   └── catboost_models.pkl
├── metrics/                     # Evaluation metrics
│   ├── model_metrics.csv
│   ├── baseline_comparison.csv
│   └── sector_metrics.csv
├── feature_importance/          # Feature analysis
│   └── importance_rankings.csv
└── visualizations/              # Generated plots
    ├── performance/            # Model performance plots
    ├── features/              # Feature importance plots
    ├── residuals/             # Residual analysis
    ├── shap/                  # SHAP visualizations
    ├── statistical_tests/     # Statistical analysis
    └── sectors/               # Sector-specific plots
```

## Troubleshooting

### Common Issues

**1. "No module named 'src'"**
- Ensure you're running from the project root directory
- Check that the virtual environment is activated

**2. "Model file not found"**
- Run training commands before evaluation/visualization
- Check `outputs/models/` directory for saved models

**3. Poor Linear Model Performance**
- Known issue: Input data is pre-normalized
- Tree models are recommended for better performance

**4. Memory Issues with Large Datasets**
- Reduce Optuna trials: `--optimize-xgboost 50`
- Use subset of data: `--datasets LR_Base`

### Debugging Tips

1. **Check logs**: Detailed logs in `logs/` directory
2. **Verify data**: Ensure CSV files exist in `data/raw/`
3. **Test incrementally**: Train one model type at a time
4. **Use non-interactive mode**: Add `--non-interactive` for automation

## Known Issues

### Critical Data Issue

⚠️ **Pre-normalized Data**: The CSV files contain standardized features (mean=0, std=1), causing:
- Linear Regression models to fail (negative R² values)
- ElasticNet models to converge to identical solutions
- Loss of coefficient interpretability

**Recommendation**: Use tree-based models (XGBoost, LightGBM, CatBoost) for reliable predictions.

### Other Limitations

- Linear models require proper non-normalized data
- Some visualizations may fail with very sparse models
- Sector models require sufficient data per sector

## Best Practices

### Performance Optimization

1. **Start with tree models**: Better handling of current data format
2. **Use Optuna optimization**: Significantly improves model performance
3. **Enable native categorical support**: More efficient than one-hot encoding
4. **Monitor memory usage**: Large datasets may require batch processing

### Pipeline Usage

1. **Initial exploration**:
   ```bash
   python main.py --train --evaluate --visualize
   ```

2. **Production training**:
   ```bash
   python main.py --all --optimize-xgboost 100 --optimize-lightgbm 100 --non-interactive
   ```

3. **Feature analysis**:
   ```bash
   python xgboost_feature_removal_proper.py
   python main.py --importance --vif
   ```

### Development Guidelines

- **No standalone scripts**: All functionality integrates with main pipeline
- **Direct modifications**: Change files in place, no patch generators
- **Clean as you go**: Remove temporary files immediately
- **Automatic execution**: All features run without manual intervention

## Special Scripts

### XGBoost Feature Removal Analysis

The `xgboost_feature_removal_proper.py` script provides detailed analysis of feature removal impact:

**Purpose**: Analyze how removing specific features affects model performance
**Location**: `xgboost_feature_removal_proper.py`
**Usage**:
```bash
# Standalone execution
python xgboost_feature_removal_proper.py

# Via main pipeline
python main.py --xgboost-feature-removal
```

**Features**:
- Removes shareholder percentage features by default
- Runs fresh Optuna optimization on modified datasets
- Generates SHAP comparison visualizations
- Outputs detailed metrics comparison

**Output**: Results in `outputs/feature_removal/` including:
- Comparison metrics (JSON and CSV)
- SHAP visualizations showing feature importance changes
- Summary report of findings

### Archive Scripts

The `archive_before_amendments_safe.py` script creates organized archives of project files:

**Purpose**: Archive files while preserving directory structure
**Usage**:
```bash
python archive_before_amendments_safe.py
```

## Frequently Asked Questions (FAQ)

### Q: Why are linear models performing poorly?

A: The input CSV files contain pre-normalized (standardized) data with mean=0 and std=1. This destroys the scale information needed for linear models to work properly. Tree-based models (XGBoost, LightGBM, CatBoost) are recommended.

### Q: How do I force models to retrain?

A: Use the `--force-retune` flag:
```bash
python main.py --all --force-retune
```

### Q: What's the difference between `--visualize` and `--visualize-new`?

A: They are identical. `--visualize-new` exists for backward compatibility but both use the new visualization architecture.

### Q: How can I run only specific models?

A: Use model-specific flags:
```bash
# Train only XGBoost and LightGBM
python main.py --train-xgboost --train-lightgbm

# Evaluate only these models
python main.py --evaluate
```

### Q: What's the recommended workflow for production?

A: For production use:
```bash
python main.py --all --optimize-xgboost 100 --optimize-lightgbm 100 --non-interactive
```

### Q: How do I analyze a specific sector?

A: Use sector-specific commands:
```bash
# Train sector models
python main.py --train-sector --train-sector-lightgbm

# Visualize sector results
python main.py --visualize-sector
```

### Q: What if I run out of memory during training?

A: Try these solutions:
1. Reduce Optuna trials: `--optimize-xgboost 50`
2. Train models individually instead of using `--all`
3. Use subset of datasets: `--datasets LR_Base`

### Q: How can I add new features to the pipeline?

A: Follow these steps:
1. Add feature engineering code to appropriate module in `src/`
2. Update data loading if needed
3. Add command-line argument in `main.py`
4. Ensure outputs go to standard directories
5. Update documentation

### Q: Where are the model files stored?

A: All trained models are saved in `outputs/models/`:
- `linear_regression_models.pkl`
- `elasticnet_models.pkl`
- `xgboost_models.pkl`
- `lightgbm_models.pkl`
- `catboost_models.pkl`

### Q: Can I resume an interrupted pipeline run?

A: Yes, the pipeline includes state management. Simply run the same command again and it will skip completed steps.

## Contributing

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters
- Add docstrings to all functions
- Maintain existing project structure

### Adding New Features

1. Integrate with existing pipeline
2. Add command-line arguments to `main.py`
3. Update configuration files as needed
4. Generate appropriate outputs in standard directories
5. Update this README with new functionality

### Testing

While no formal test suite exists, ensure:
- New features work with `--all` flag
- Models save/load correctly
- Visualizations generate without errors
- Pipeline state management works

---

For questions or issues, please refer to the logs in the `logs/` directory or check the troubleshooting section above.