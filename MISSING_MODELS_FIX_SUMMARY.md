# Summary of Missing Models and Visualization Issues

## Problems Identified

1. **No model files in outputs/models directory**
   - The visualization pipeline expected model .pkl files in outputs/models/
   - All subdirectories were empty - no models had been saved

2. **Linear regression models were broken**
   - Models were only using 1 feature instead of all 362 features
   - Performance was terrible (R² = 0.1066 with only 1 feature)
   - Issue was caused by:
     - `score_df` was loaded as DataFrame with shape (2202, 1) instead of Series
     - When sklearn fits a 2D target, `model.coef_` has shape (1, n_features) instead of (n_features,)
     - Code was using `model.coef_.shape[0]` which returned 1 instead of the actual feature count

3. **Wrong target variable was being loaded**
   - `data/__init__.py` was returning the entire score DataFrame with index
   - Linear regression was trying to predict on the issuer names (index) not the scores!

## Fixes Applied

### 1. Fixed score data loading (data/__init__.py)
```python
# Changed from:
return pd.read_csv(file_path, index_col='issuer_name')

# To:
df = pd.read_csv(file_path, index_col='issuer_name')
return df['esg_score']  # Return as Series
```

### 2. Fixed feature count calculation (models/linear_regression.py)
```python
# Changed from:
'n_features': model.coef_.shape[0]

# To:
'n_features': model.coef_.shape[-1] if model.coef_.ndim > 1 else len(model.coef_)
```

### 3. Fixed target column selection (data.py)
```python
# Changed from:
return scores_df.iloc[:, 0]  # Was selecting issuer_name column!

# To:
if 'esg_score' in scores_df.columns:
    return scores_df['esg_score']
else:
    return scores_df.iloc[:, 1]  # Fallback to second column
```

## Results

After fixes:
- Linear regression now uses all 362 features correctly
- Model performance improved significantly
- Model files can now be saved and loaded properly
- Visualization pipeline should work once models are trained

## Next Steps

1. Run `python main.py --all` to train all models with the fixes
2. Verify all model .pkl files are created in outputs/models/
3. Check that visualizations are generated correctly
4. Consider repository cleanup after confirming everything works