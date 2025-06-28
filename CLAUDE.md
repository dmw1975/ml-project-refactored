# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Recent Repository Cleanup (2025-06-22)

A comprehensive repository cleanup was performed to improve organization and reduce clutter:

### Cleanup Results
- **Root directory**: Reduced from 150+ files to just 9 essential files
- **Total files archived**: 82 files and 12 directories
- **Total files deleted**: 135 temporary/diagnostic files
- **Disk space saved**: ~5GB from removing old outputs
- **Archive structure**: Created organized archive with 7 categories and 31 subdirectories

### Key Changes
1. All temporary diagnostic scripts moved to `archive/diagnostic_scripts/`
2. Old documentation moved to `archive/documentation/` and `archive/deprecated_docs/`
3. Feature removal scripts consolidated in `archive/feature_removal/`
4. Cleanup operation files stored in `archive/cleanup_operation/`
5. All pre-amendment outputs preserved in `archive/consolidated_outputs/`

### Protected Files (Never Modified)
- `xgboost_feature_removal_proper.py`
- `archive_before_amendments_safe.py`
- `CLAUDE.md`
- `README.md`
- `README-CLAUDE.md`

### Pipeline Status
✅ All pipeline functionality preserved and tested after cleanup
✅ All models, data, and outputs remain accessible
✅ Source code structure unchanged

## Build/Test Commands
- Run all tests: `python test_setup.py && python test_features_data.py && python test_xgboost.py && python test_visualization.py && python test_sector_models.py`
- Run a single test: `python <test_file.py>` (e.g., `python test_xgboost.py`)
- Run with specific flags: `python main.py --<flag>` (common flags: `--train`, `--evaluate`, `--visualize`)
- Format code: `black .`
- Lint code: `flake8`

## Sector Model Training
When running `python main.py --all`, the following sector models are automatically trained:
- **ElasticNet Sector Models**: Trained via `sector_elastic_net_models.py` (generates `sector_models_metrics.csv`)
- **LightGBM Sector Models**: Trained via `sector_lightgbm_models.py` (generates `sector_lightgbm_metrics.csv`)

For individual sector model training:
- `python main.py --train-sector`: Trains LinearRegression sector models (via `sector_models.py`)
- `python main.py --train-sector-lightgbm`: Trains LightGBM sector models only
- `python main.py --all-sector`: Trains all sector model types

## Data Loading Architecture
The data loading system supports both JSON metadata (preferred) and pickle files (legacy):

### ⚠️ CRITICAL ISSUE: Pre-normalized Data
**The CSV files from esg-score-data repository contain pre-normalized (standardized) features with mean=0 and std=1. This causes:**
- Linear Regression models to fail catastrophically (negative R² values)
- ElasticNet models to converge to identical solutions
- Loss of interpretability due to standardized coefficients

**Current Status:**
- Tree models (XGBoost, LightGBM, CatBoost) still work reasonably well
- Linear models should not be used until data normalization is fixed
- Proper solution requires non-normalized data files from esg-score-data

### JSON Metadata Approach (New)
When metadata files are available in `data/metadata/`:
- `linear_model_columns.json`: Defines columns for linear models
- `tree_model_columns.json`: Defines columns for tree models
- `yeo_johnson_mapping.json`: Maps base to Yeo-transformed columns
- `feature_groups.json`: Groups features by business logic

The system uses `src/data/loaders.py` with:
- `LinearModelDataLoader`: Loads data for linear models with one-hot encoding
- `TreeModelDataLoader`: Loads data for tree models with native categorical features

### Pickle Files Approach (Legacy)
Falls back to pickle files if JSON metadata is not available:
- `data/pkl/base_columns.pkl`: Contains base column names (incomplete)
- `data/pkl/yeo_columns.pkl`: Contains Yeo column names (mixed content)

**Note**: The pickle approach requires manual addition of categorical columns in model files.

### Implementation Details
- Data loading attempts JSON approach first, falls back to pickles if needed
- Models automatically detect which approach is being used
- No code changes required when switching between approaches
- Denormalization was attempted but requires accurate original statistics

## Baseline Evaluation
The baseline evaluation is automatically run during `python main.py --evaluate` or `python main.py --all`. It compares all models against Random, Mean, and Median baselines:
- **Random baseline**: Uses test data range
- **Mean/Median baselines**: Uses training data (original methodology)
- Results are saved to `metrics/baseline_comparison.csv`
- The evaluation ensures ALL model types (Linear Regression, ElasticNet, XGBoost, LightGBM, CatBoost) are included

## Cross-Validation (CV) Scores in Models
All tree-based models (LightGBM, XGBoost, CatBoost) now compute and store CV scores during training:
- **CV scores are computed during training** using 5-fold cross-validation
- **Stored in model results** as `cv_scores` (array), `cv_mean`, and `cv_std`
- **ElasticNet models** store `cv_mse` and `cv_mse_std` (not individual fold scores)
- **Linear Regression models** do not currently compute CV scores
- The `fix_add_cv_scores.py` script is no longer needed as CV scores are computed during training

## Metrics Summary Table
The metrics summary table visualization has been enhanced to ensure all models are included:
- **Comprehensive Model Loading**: When run via `main.py --visualize` or `main.py --all`, the table automatically loads all 32 models
- **Proper Highlighting**: The table is sorted by RMSE and the best model (lowest RMSE) is highlighted with a green background
- **Model Count Verification**: The table displays the total model count and identifies the best performing model
- **Implementation Details**:
  - Enhanced `MetricsTable` class in `src/visualization/plots/metrics.py`
  - Added `_collect_all_metrics_comprehensively()` method for loading all model types
  - Handles different metric storage patterns (Linear Regression vs tree models)
  - Calculates missing MSE values when needed
- The `fix_metrics_summary_table.py` and `fix_metrics_table_highlighting.py` scripts are no longer needed

## Code Style Guidelines
- **Imports**: Standard library first, third-party packages second, local modules last
- **Naming**: snake_case for variables/functions, PascalCase for classes, UPPER_SNAKE_CASE for constants
- **Docstrings**: NumPy-style docstrings with parameters and return values documented
- **Formatting**: Follow PEP 8 guidelines, use black for formatting
- **Error Handling**: Use descriptive error messages, validate inputs, use try/except for anticipated errors
- **Project Structure**: Keep code organized in modules (data, models, evaluation, visualization)
- **File I/O**: Use utils.io module for file operations to ensure consistent handling
- **Paths**: Use pathlib.Path for path manipulation, reference paths from settings.py
- **Testing**: Each component should have a corresponding test file (test_*.py)

## Project Structure

```
ml_project_refactored/
├── data/                           # Data directory (in .gitignore)
│   ├── raw/                        # Raw data files
│   ├── processed/                  # Processed data files
│   └── interim/                    # Intermediate data files
│
├── outputs/                        # Output directory for results
│   ├── models/                     # Trained model files (.pkl)
│   │   ├── linear_regression_models.pkl
│   │   ├── elasticnet_models.pkl
│   │   ├── xgboost_models.pkl
│   │   ├── lightgbm_models.pkl
│   │   ├── catboost_models.pkl
│   │   └── sector_lightgbm_models.pkl
│   ├── metrics/                    # Model evaluation metrics
│   ├── feature_importance/         # Feature importance results
│   └── visualizations/             # Generated plots and figures
│       ├── performance/            # Performance plots by model type
│       ├── features/               # Feature importance plots
│       ├── residuals/              # Residual plots
│       ├── shap/                   # SHAP value visualizations
│       ├── statistical_tests/      # Statistical test results
│       ├── baselines/              # Baseline comparison plots
│       └── sectors/                # Sector-specific visualizations
│
├── logs/                           # Pipeline execution logs
│
├── src/                            # Source code
│   ├── config/                     # Configuration
│   │   ├── settings.py             # Central configuration file
│   │   └── hyperparameters.py      # Model hyperparameters
│   ├── data/                       # Data processing
│   │   ├── data_categorical.py     # Categorical data handling
│   │   └── data.py                 # General data utilities
│   ├── models/                     # Model implementations
│   │   ├── linear_regression.py    # Linear regression models
│   │   ├── elastic_net.py          # ElasticNet models
│   │   ├── xgboost_categorical.py  # XGBoost with categorical features
│   │   ├── lightgbm_categorical.py # LightGBM with categorical features
│   │   ├── catboost_categorical.py # CatBoost with categorical features
│   │   └── sector_models.py        # Sector-specific models
│   ├── evaluation/                 # Model evaluation
│   │   ├── metrics.py              # Evaluation metrics
│   │   ├── importance.py           # Feature importance analysis
│   │   ├── baselines.py            # Baseline model comparisons
│   │   └── multicollinearity.py    # VIF analysis
│   ├── utils/                      # Utility functions
│   │   ├── io.py                   # I/O operations
│   │   └── model_check_fix.py      # Model checking utilities
│   └── visualization/              # Visualization module
│       ├── comprehensive.py        # Main visualization pipeline
│       ├── viz_factory.py          # Visualization factory
│       ├── adapters/               # Model-specific adapters
│       └── plots/                  # Plot implementations
│
├── scripts/                        # Utility scripts
│   ├── archive/                    # Enhanced implementations
│   │   ├── enhanced_xgboost_categorical.py
│   │   ├── enhanced_lightgbm_categorical.py
│   │   └── enhanced_catboost_categorical.py
│   └── utilities/                  # Utility scripts
│       └── fix_sklearn_xgboost_compatibility.py
│
├── main.py                         # Main entry point
├── CLAUDE.md                       # This file
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
```

## Key Components

### Main Entry Point
- `main.py`: Central pipeline orchestrator with command-line interface

### Data Processing (`src/data/`)
- Handles both categorical and one-hot encoded data
- Supports tree models with native categorical features
- Provides data loading for linear and tree-based models

### Model Implementations (`src/models/`)
- **Linear Models**: Linear Regression, ElasticNet (with Optuna optimization)
- **Tree Models**: XGBoost, LightGBM, CatBoost (with native categorical support)
- **Sector Models**: Sector-specific implementations

### Command-Line Interface
```bash
# Run entire pipeline
python main.py --all

# Train specific model types
python main.py --train-xgboost
python main.py --train-lightgbm
python main.py --train-catboost

# Force retrain all models
python main.py --all --force-retune

# Run in non-interactive mode
python main.py --all --non-interactive
```

## Recent Fixes and Updates
1. **Scikit-learn Compatibility**: Fixed cross-validation issues with sklearn 1.6.1 using safe CV wrapper
2. **Pipeline Logic**: Individual model checking to properly detect and train missing tree models
3. **Error Reporting**: Enhanced error messages with full tracebacks for debugging

## ML Pipeline Best Practices

1. **Model Persistence**: Always save models with full metadata (parameters, metrics, CV scores)
2. **Data Validation**: Check for NaN values and data types before model training
3. **Reproducibility**: Use consistent random seeds across all models
4. **Memory Efficiency**: Process large datasets in chunks when possible
5. **Pipeline State**: Use the state manager to track long-running operations
6. **Error Recovery**: Pipeline should continue with next model if one fails
7. **Consistent Metrics**: Always compute RMSE, MAE, R² for all models

## Avoiding Common Pitfalls

1. **Don't mix categorical encodings**: Tree models use native categorical, linear models use one-hot
2. **Always check model files exist**: Use model_check_fix utilities before assuming models are trained
3. **Handle Optuna studies carefully**: Check for existing studies before retraining
4. **Cross-validation compatibility**: Use the safe CV wrapper for sklearn 1.6.1+

## Development Best Practices (CRITICAL - Going Forward)

### File Management
1. **NO STANDALONE SCRIPTS**: Everything must integrate into the pipeline
   - Don't create diagnostic scripts that require manual execution
   - Don't create fix scripts that generate patches
   - All functionality must run automatically via main.py

2. **DIRECT MODIFICATIONS ONLY**: Change files in place
   - Use version control for safety, not create new files
   - Modify pipeline components directly
   - No intermediate fix generators

3. **CLEAN AS YOU GO**: Delete temporary files immediately
   - Remove diagnostic scripts after debugging
   - Don't leave abandoned attempts
   - No accumulation of test files

4. **FOLLOW ARCHITECTURE**: Respect project structure
   - Test files go in proper test directories
   - No root-level utility scripts
   - Maintain module organization

### Integration Requirements
1. **Automatic Execution**: All fixes must run without manual intervention
2. **Pipeline Integration**: New functionality must be called by main.py
3. **No Manual Steps**: Everything automated through command-line flags
4. **Proper Imports**: New modules must be imported where needed

### Problem-Solving Approach
1. **Understand First**: Use debugger/logging instead of creating test scripts
2. **Fix In Place**: Modify existing files rather than creating new ones
3. **Test Properly**: Use the existing test framework
4. **Document Changes**: Update this file when making significant changes

### What NOT to Do
- ❌ Create `test_*.py` files in root directory for debugging
- ❌ Create `fix_*.py` scripts that generate patches
- ❌ Create `verify_*.py` scripts for manual checking
- ❌ Leave temporary files after fixing issues
- ❌ Create workarounds instead of permanent solutions

### What TO Do Instead
- ✅ Use logging and debugger for understanding issues
- ✅ Modify source files directly
- ✅ Ensure all changes integrate with main.py
- ✅ Clean up any temporary work immediately
- ✅ Follow the established project structure