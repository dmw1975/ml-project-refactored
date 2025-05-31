# Recommendation: Modifying combined_df_for_ml_models.csv for EDA

## Current Situation

The `combined_df_for_ml_models.csv` file currently contains:
- **393 columns** total
- **336 one-hot encoded columns** from 7 categorical variables
- **No original categorical columns** preserved

## Recommendation: YES, Create an Enhanced Version

### Benefits of Adding Original Categorical Columns:

1. **EDA Convenience**
   - Easier groupby operations (e.g., `df.groupby('gics_sector').mean()`)
   - Simple value counts and distributions
   - Direct categorical plotting without reconstruction
   - Cleaner correlation analysis with target variable

2. **Flexibility**
   - Users can choose between categorical or one-hot representation
   - Supports both tree-based and linear model workflows
   - Enables mixed-type visualizations

3. **Storage Efficiency**
   - Original categorical columns use much less memory
   - The 7 categorical columns would add minimal size vs 336 one-hot columns

4. **Backward Compatibility**
   - Existing code using one-hot columns continues to work
   - New functionality available for categorical analysis

## Proposed Implementation

### Option 1: Enhance Existing File (Recommended)
Add these 7 categorical columns to the existing CSV:
- `gics_sector` (instead of 10 one-hot columns)
- `gics_sub_ind` (instead of 147 one-hot columns)
- `issuer_cntry_domicile` (instead of 42 one-hot columns)
- `cntry_of_risk` (instead of 46 one-hot columns)
- `top_1_shareholder_location` (instead of 23 one-hot columns)
- `top_2_shareholder_location` (instead of 33 one-hot columns)
- `top_3_shareholder_location` (instead of 35 one-hot columns)

### Option 2: Create Separate EDA-Optimized File
Create `combined_df_for_eda.csv` with:
- All quantitative features
- Original categorical columns
- Selected one-hot columns (optional)
- Additional EDA-friendly features (e.g., binned continuous variables)

## Implementation Script

```python
import pandas as pd
import numpy as np

def create_eda_enhanced_dataset():
    """Create an EDA-friendly version of the combined dataset."""
    
    # Load the original combined dataset
    df = pd.read_csv('data/raw/combined_df_for_ml_models.csv')
    
    # Load the tree models dataset which has categorical columns
    tree_df = pd.read_csv('data/processed/tree_models_dataset.csv')
    
    # Categorical columns to add
    categorical_cols = [
        'gics_sector', 'gics_sub_ind', 'issuer_cntry_domicile',
        'cntry_of_risk', 'top_1_shareholder_location',
        'top_2_shareholder_location', 'top_3_shareholder_location'
    ]
    
    # Set index to align data
    df.set_index('issuer_name', inplace=True)
    tree_df.set_index('issuer_name', inplace=True)
    
    # Add categorical columns to the combined dataset
    for col in categorical_cols:
        if col in tree_df.columns:
            df[f'cat_{col}'] = tree_df[col]
    
    # Reset index
    df.reset_index(inplace=True)
    
    # Save enhanced dataset
    df.to_csv('data/raw/combined_df_for_ml_models_enhanced.csv', index=False)
    
    # Optionally create a pure EDA version with fewer columns
    eda_cols = ['issuer_name'] + \
               [col for col in df.columns if not any(cat in col for cat in 
                ['_name_', 'cntry_of_risk_', 'gics_sector_', 'gics_sub_ind_',
                 'shareholder_location_'])] + \
               [f'cat_{col}' for col in categorical_cols]
    
    df_eda = df[eda_cols]
    df_eda.to_csv('data/raw/combined_df_for_eda.csv', index=False)
    
    print(f"Original columns: {len(df.columns)}")
    print(f"EDA version columns: {len(df_eda.columns)}")
    
create_eda_enhanced_dataset()
```

## Usage Examples for EDA

### With Enhanced Dataset:
```python
# Simple sector analysis
df.groupby('cat_gics_sector')['esg_score'].agg(['mean', 'std', 'count'])

# Country risk distribution
df['cat_cntry_of_risk'].value_counts().plot(kind='bar')

# Correlation with categorical features
pd.get_dummies(df['cat_gics_sector']).corrwith(df['esg_score'])

# Multi-level grouping
df.groupby(['cat_gics_sector', 'cat_issuer_cntry_domicile'])['esg_score'].mean()
```

### Current Workaround (Reconstructing from one-hot):
```python
# Much more complex and error-prone
sector_cols = [col for col in df.columns if col.startswith('gics_sector_')]
df['sector'] = df[sector_cols].idxmax(axis=1).str.replace('gics_sector_', '')
```

## Conclusion

Adding the original categorical columns would significantly improve the EDA experience while maintaining full backward compatibility. The storage overhead is minimal (7 additional columns) compared to the convenience gained.