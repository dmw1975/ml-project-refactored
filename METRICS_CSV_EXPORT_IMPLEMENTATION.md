# Metrics CSV Export Implementation

## Overview
Enhanced the ML project metrics output to include CSV export functionality with extended metrics for all models.

## Implementation Details

### 1. Enhanced MetricsTable Class
Location: `src/visualization/plots/metrics.py`

#### Key Changes:
- Modified `_collect_all_metrics_comprehensively()` to collect extended metrics including:
  - Number of CV folds
  - Training and testing sample counts
  - Total feature counts
  - Added logging for better debugging

- Added `_calculate_feature_counts()` method to accurately calculate:
  - Quantitative features
  - Qualitative features
  - Handles different counting for:
    - Linear/ElasticNet models (one-hot encoded: ~362 categorical features)
    - Tree-based models (native categorical: 7 features)

- Added `_export_metrics_to_csv()` method to:
  - Export comprehensive metrics to CSV
  - Include all required columns as specified
  - Save single file without timestamp

### 2. CSV Output Format
The CSV file includes the following columns:
- Model Name
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error) 
- R² (R-squared)
- MSE (Mean Square Error)
- Number of CV Folds
- Number of Testing Samples
- Number of Training Samples
- Number of Quantitative Features
- Number of Qualitative Features

### 3. Integration with Main Pipeline
- The CSV export is automatically triggered when running `python main.py --all`
- The comprehensive visualization pipeline sets the `comprehensive: True` flag
- CSV is saved to: `outputs/visualizations/performance/model_metrics_comparison.csv`

### 4. Feature Counting Logic

#### For Linear Regression and ElasticNet Models:
- Use one-hot encoded features
- Total features: 388 (Base) or 389 (with Random)
- Categorical features (one-hot): ~362
- Quantitative features: ~26

#### For Tree-based Models (XGBoost, LightGBM, CatBoost):
- Use native categorical features
- Categorical features: 7
- Quantitative features: Total - 7

### 5. Sample Counts
- All models use 80/20 train/test split
- Training samples: 1761
- Testing samples: 441
- Linear Regression models now correctly report these values

## Verification
Run the test script to verify CSV export:
```bash
python test_metrics_csv_export.py
```

## Usage
The CSV export is automatically generated when running:
```bash
python main.py --all
```
or 
```bash
python main.py --visualize
```

The CSV file will be saved at:
`outputs/visualizations/performance/model_metrics_comparison.csv`

## Benefits
1. **Data Export**: Enables further analysis in Excel, R, or other tools
2. **Extended Metrics**: Provides comprehensive model information including feature counts
3. **Automated**: Integrates seamlessly with existing pipeline
4. **Logging**: Enhanced logging for debugging and monitoring
5. **Accuracy**: Correctly distinguishes between model types for feature counting