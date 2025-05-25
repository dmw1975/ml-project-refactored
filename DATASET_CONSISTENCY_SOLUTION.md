# Dataset Consistency Solution

## Problem Summary

Currently, different model types use different datasets and thus have different test set sizes:
- **Linear models**: 441 test samples (from 2,202 total)
- **Tree models** (XGBoost/LightGBM): 396 test samples (from 1,977 total)
- **CatBoost**: 441 test samples (but missing predictions)

This creates inconsistent baseline comparisons.

## Root Cause

1. **Linear models** use `combined_df_for_ml_models.csv` with one-hot encoding
2. **Tree models** use `tree_models_dataset.csv` with native categorical features
3. The datasets have different numbers of samples, likely due to:
   - Different preprocessing steps
   - Different handling of missing values
   - Different feature engineering

## Proposed Solution

### Short-term Fix (Immediate)

1. **Fix Linear Models**: Add training data storage to enable proper baseline calculation
   ```bash
   python fix_linear_models_training_data.py
   ```

2. **Document the Difference**: Accept that models use different datasets but ensure fair comparison within each dataset type

### Long-term Solution (Recommended)

1. **Create Unified Dataset Pipeline**:
   ```python
   # Create a unified data loader that ensures all models use the same base dataset
   def load_unified_model_data(encoding_type='categorical'):
       """
       Load data with consistent train/test split for all models.
       
       Parameters:
       -----------
       encoding_type : str
           'categorical' for tree models, 'one-hot' for linear models
       """
       # Load base dataset
       base_data = load_base_dataset()
       
       # Apply consistent preprocessing
       processed_data = preprocess_data(base_data)
       
       # Apply encoding based on model type
       if encoding_type == 'categorical':
           features = encode_categorical_features(processed_data)
       else:
           features = one_hot_encode_features(processed_data)
       
       # Use consistent train/test split with fixed indices
       train_idx, test_idx = get_fixed_split_indices()
       
       return features, targets, train_idx, test_idx
   ```

2. **Store Split Indices**:
   ```python
   # Save split indices to ensure all models use the same split
   split_info = {
       'train_indices': train_idx,
       'test_indices': test_idx,
       'random_state': 42,
       'test_size': 0.2,
       'stratify_by': 'sector'
   }
   save_pickle(split_info, 'data/processed/train_test_split.pkl')
   ```

3. **Modify Model Training**:
   - Update all model training scripts to use the unified data loader
   - Ensure consistent preprocessing steps
   - Use the same train/test indices for all models

## Implementation Steps

1. **Phase 1**: Fix immediate issues
   - Run `fix_linear_models_training_data.py`
   - Re-run baseline evaluation

2. **Phase 2**: Create unified pipeline
   - Implement `create_unified_datasets.py`
   - Update model training scripts
   - Retrain all models with consistent data

3. **Phase 3**: Validate
   - Verify all models have same test set size
   - Ensure baseline values are identical across model types
   - Update documentation

## Benefits

1. **Fair Comparisons**: All models evaluated on exactly the same test set
2. **Consistent Baselines**: Mean, median, and random baselines identical for all models
3. **Reproducibility**: Fixed splits ensure consistent results across runs
4. **Simplified Analysis**: No need to account for different dataset sizes in comparisons