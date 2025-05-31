# Sector Distribution Plot - Relative Frequencies Fix Summary

## Date: 2025-05-31

### Issue
The sector distribution plot was showing **absolute proportions** of the total dataset, making it impossible to verify whether stratified splitting preserved the sector distributions between train and test sets.

### Solution
Updated the plot to show **relative frequencies (percentages) within each set**, which properly demonstrates stratification quality.

### Key Changes

1. **Created new data extraction script**: `scripts/utilities/get_relative_sector_distribution.py`
   - Calculates percentages within train set and test set separately
   - Shows that stratification preserved distributions with max difference of only 0.14%

2. **Updated plotting functions in `sectors.py`**:
   - `create_sector_stratification_plot()` - Now checks for relative distribution data first
   - `create_sector_stratification_plot_relative()` - New function that shows percentages within each set
   - Updated compute function to also calculate relative frequencies

### Visual Improvements

1. **Y-axis**: Now shows "Percentage Within Each Set (%)" instead of absolute proportions
2. **Values**: Shows percentages (e.g., "20.0%") instead of proportions (e.g., "0.160")
3. **Differences**: Shows Δ values where train/test percentages differ slightly
4. **Quality indicator**: Shows stratification quality (EXCELLENT/GOOD/FAIR) based on max difference
5. **Interpretation**: Nearly identical bar heights clearly show successful stratification

### Why This Matters

**Before (Absolute Proportions)**:
- Train bars were ~4x taller than test bars (80/20 split)
- Couldn't verify if distributions were preserved
- Misleading for validation purposes

**After (Relative Frequencies)**:
- Train and test bars are nearly identical heights
- Clearly shows each sector has same percentage in both sets
- Properly validates stratified splitting

### Example Values
- Industrials: 20.0% of train set, 20.0% of test set (Δ=0.03%)
- Financials: 16.0% of train set, 16.1% of test set (Δ=0.14%)
- All sectors show <0.15% difference between train and test

### Verification
The stratification quality is EXCELLENT with a maximum difference of only 0.14% between train and test proportions, confirming that the stratified splitting correctly preserved the sector distributions.

This plot is now included when running `main.py` with `--all` or `--visualize` flags and properly demonstrates the success of stratified splitting.