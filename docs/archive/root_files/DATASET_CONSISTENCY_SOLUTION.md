# Dataset Consistency Solution

## Summary

This document describes how we fixed the data consistency issue between the old and new ML pipelines to ensure fair comparison and identical results.

## Problem

The initial comparison between old and new pipelines showed significant differences:
- Old pipeline: Test RMSE = 3.4712, Test R2 = -2.2744
- New pipeline: Test RMSE = 1.8645, Test R2 = 0.1066

Despite both pipelines using the same LinearRegression model from sklearn.

## Root Causes Identified

1. **Data Loading**: The new pipeline was using fallback methods to load data from the old pipeline, but the data wasn't being saved in the expected locations.

2. **Train/Test Split Strategy**: The new pipeline defaulted to stratified splitting by sector, while the old pipeline used simple random splitting.

## Solution Implemented

### 1. Data Migration Script (`fix_data_consistency.py`)

Created a script that:
- Loads data using the old pipeline's exact methods
- Saves the processed datasets to the new pipeline's expected locations
- Creates a feature mapping file for reference
- Verifies data consistency after saving

Key files created:
- `esg_ml_clean/data/processed/features_base.csv` - Base features (362 columns)
- `esg_ml_clean/data/processed/features_yeo.csv` - Yeo-transformed features (26 columns)
- `esg_ml_clean/data/processed/targets.csv` - ESG scores
- `esg_ml_clean/data/processed/feature_mapping.json` - Feature metadata

### 2. DataLoader Updates

Updated the new pipeline's DataLoader to:
- First check for migrated data files
- Use the migrated data when available
- Fall back to legacy loading only if needed

### 3. Comparison Script Fix

Updated `compare_old_vs_new.py` to:
- Use the same train/test split strategy (no stratification)
- Match the exact parameters of the old pipeline

## Verification

After implementing these fixes:
- Both pipelines use identical data (2202 samples, 362 features)
- Both pipelines produce identical train/test splits (1761/441 samples)
- Both pipelines produce identical model results (RMSE: 3.4712, R2: -2.2744)

## Usage

1. Run the data consistency fix (only needed once):
   ```bash
   python fix_data_consistency.py
   ```

2. Verify pipeline equivalence:
   ```bash
   cd esg_ml_clean
   python compare_old_vs_new.py
   ```

## Important Notes

- The new pipeline defaults to stratified splitting by sector, which is generally better for model generalization
- For exact comparison with old results, use `stratify_by=None` in the data configuration
- The migrated data files are now the source of truth for both pipelines
- All feature transformations and preprocessing are preserved exactly as in the old pipeline

## Next Steps

With data consistency verified, you can now:
1. Migrate trained models from the old pipeline
2. Run full pipeline comparisons with all model types
3. Gradually transition to using the new pipeline's improved features (like stratified splitting)