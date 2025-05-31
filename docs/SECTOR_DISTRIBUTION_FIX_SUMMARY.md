# Sector Distribution Plot Fix Summary

## Date: 2025-05-31

### Issue
The sector distribution plot in `/outputs/visualizations/sectors/sector_train_test_distribution.png` was showing incorrect values compared to the archived correct plot. The current implementation was using synthetic/hardcoded sector weights instead of actual data distribution from the stratified train/test split.

### Root Cause
The `create_sector_stratification_plot_lightgbm()` function in `src/visualization/plots/sectors.py` was:
1. Using hardcoded sector weights (e.g., Information Technology: 0.26)
2. Adding random noise to these synthetic values
3. Not accessing the actual data distribution from the stratified split

### Solution Implemented

1. **Created a script to extract actual sector distributions**:
   - `scripts/utilities/get_actual_sector_distribution.py`
   - Loads the actual tree models data
   - Performs the same stratified split used in training
   - Saves actual distributions to `data/processed/sector_distribution.csv`

2. **Updated the plotting functions**:
   - Modified `create_sector_stratification_plot()` to check for saved distribution data
   - Created `create_sector_stratification_plot_from_actual_data()` to use the actual data
   - Created `create_sector_stratification_plot_compute()` as a fallback to compute from raw data
   - Replaced synthetic data generation with actual data loading

### Key Changes in `sectors.py`

1. The main function now checks for saved distribution data first:
```python
def create_sector_stratification_plot(output_dir):
    dist_file = Path(settings.DATA_DIR) / 'processed' / 'sector_distribution.csv'
    if dist_file.exists():
        return create_sector_stratification_plot_from_actual_data(output_dir)
    else:
        return create_sector_stratification_plot_compute(output_dir)
```

2. The new function uses actual data proportions:
```python
def create_sector_stratification_plot_from_actual_data(output_dir):
    dist_df = pd.read_csv(dist_file)
    train_props = dist_df['train_proportion'].tolist()
    test_props = dist_df['test_proportion'].tolist()
    # Creates plot with actual values
```

### Results

**Before (Incorrect)**:
- Information Technology: 0.199 (unrealistic)
- Financials: 0.111
- Industrials: 0.072

**After (Correct)**:
- Information Technology: 0.083 (actual)
- Financials: 0.128 (actual)
- Industrials: 0.160 (actual)

### Verification
The fixed plot now matches the archived correct plot and accurately represents the actual sector distribution from the stratified train/test split. The plot is now consistently generated with correct values when running `main.py` with `--all` or `--visualize` flags.

### Files Modified
- `src/visualization/plots/sectors.py` - Updated plotting functions
- `data/processed/sector_distribution.csv` - New file with actual distributions

### Files Created
- `scripts/utilities/get_actual_sector_distribution.py` - Extract actual distributions
- `scripts/utilities/test_sector_plot_fix.py` - Test the fix