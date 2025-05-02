"""
Verification script for the corrected Yeo-Johnson implementation.
This script compares the original and fixed implementations and validates 
that all categorical features are correctly included.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

# Load the data
print("Loading feature data...")
from data import load_features_data
feature_df = load_features_data()

import pickle

# ========================================================
# 1. First, load pickle files for reference
# ========================================================
print("\n1. LOADING PICKLE FILES FOR REFERENCE")
print("-" * 50)

# Load base columns pickle
base_path = Path("data/pkl/base_columns.pkl")
with open(base_path, 'rb') as f:
    base_columns = pickle.load(f)
print(f"Loaded {len(base_columns)} columns from base_columns.pkl")

# Load yeo columns pickle
yeo_path = Path("data/pkl/yeo_columns.pkl")
with open(yeo_path, 'rb') as f:
    yeo_columns_pickle = pickle.load(f)
print(f"Loaded {len(yeo_columns_pickle)} columns from yeo_columns.pkl")

# ========================================================
# 2. Implement the corrected fix
# ========================================================
print("\n2. IMPLEMENTING CORRECTED YEO FIX")
print("-" * 50)

def get_base_and_yeo_features_corrected(feature_df):
    """Corrected implementation for the Yeo-Johnson transformation."""
    import pickle
    from pathlib import Path
    
    # Define the correct paths for pickle files
    base_path = Path("data/pkl/base_columns.pkl")
    yeo_path = Path("data/pkl/yeo_columns.pkl")
    
    # Load base columns with proper error handling
    if not base_path.exists():
        raise FileNotFoundError(f"Base columns pickle file not found at: {base_path}")
    
    print(f"Loading base columns from: {base_path}")
    with open(base_path, 'rb') as f:
        base_columns = pickle.load(f)
    
    # Load yeo columns with proper error handling  
    if not yeo_path.exists():
        raise FileNotFoundError(f"Yeo columns pickle file not found at: {yeo_path}")
    
    print(f"Loading yeo columns from: {yeo_path}")
    with open(yeo_path, 'rb') as f:
        yeo_columns_from_pickle = pickle.load(f)
    
    # Print feature counts for verification
    print(f"Base features in pickle: {len(base_columns)} columns")
    print(f"Yeo features in pickle: {len(yeo_columns_from_pickle)} columns")
    
    # Filter columns to only include those available in the dataframe
    available_base_columns = [col for col in base_columns if col in feature_df.columns]
    
    # Create the base dataframe with available columns
    LR_Base = feature_df[available_base_columns].copy()
    
    # CORRECTLY HANDLE YEO TRANSFORMATION:
    yeo_prefix = 'yeo_joh_'
    
    # 1. Identify all columns that have Yeo-transformed versions in the dataframe
    yeo_transformed_columns = [col for col in feature_df.columns if col.startswith(yeo_prefix)]
    
    # 2. Get the original column names from the transformed ones
    original_numerical_columns = [col.replace(yeo_prefix, '') for col in yeo_transformed_columns]
    
    # 3. Identify categorical columns (those in base but not in original numerical)
    categorical_columns = [col for col in available_base_columns 
                          if col not in original_numerical_columns]
    
    print(f"\nFeature type breakdown:")
    print(f"  - Numerical features with Yeo transformations: {len(yeo_transformed_columns)}")
    print(f"  - Categorical features (no transformation): {len(categorical_columns)}")
    
    # 4. Create Yeo dataset with both transformed numerical and original categorical features
    complete_yeo_columns = yeo_transformed_columns + categorical_columns
    
    # 5. Create the Yeo dataframe
    LR_Yeo = feature_df[complete_yeo_columns].copy()
    
    # Additional validation
    print(f"\nFinal dataset dimensions:")
    print(f"  LR_Base: {LR_Base.shape} - {len(LR_Base.columns)} columns")
    print(f"  LR_Yeo: {LR_Yeo.shape} - {len(LR_Yeo.columns)} columns")
    
    # Verify that Yeo column count matches expectations
    expected_yeo_count = len(yeo_transformed_columns) + len(categorical_columns)
    if len(LR_Yeo.columns) != expected_yeo_count:
        print(f"WARNING: Unexpected column count in LR_Yeo.")
        print(f"  Expected: {expected_yeo_count}, Actual: {len(LR_Yeo.columns)}")
    
    return LR_Base, LR_Yeo, available_base_columns, complete_yeo_columns


# Run the corrected implementation
LR_Base_corrected, LR_Yeo_corrected, base_columns_corrected, yeo_columns_corrected = get_base_and_yeo_features_corrected(feature_df)

# ========================================================
# 3. Now test the original implementation from data.py
# ========================================================
print("\n3. TESTING ORIGINAL IMPLEMENTATION")
print("-" * 50)

# Import the original implementation
from data import get_base_and_yeo_features
LR_Base_original, LR_Yeo_original, base_columns_original, yeo_columns_original = get_base_and_yeo_features(feature_df)

print(f"Original implementation results:")
print(f"  LR_Base dimensions: {LR_Base_original.shape}")
print(f"  LR_Yeo dimensions: {LR_Yeo_original.shape}")

# ========================================================
# 4. Compare implementations and validate results
# ========================================================
print("\n4. COMPARING IMPLEMENTATIONS")
print("-" * 50)

# Compare dimensions
print(f"Base dimensions - Original: {LR_Base_original.shape}, Corrected: {LR_Base_corrected.shape}")
print(f"Yeo dimensions - Original: {LR_Yeo_original.shape}, Corrected: {LR_Yeo_corrected.shape}")

# Print column counts and compare with pickle file
print(f"\nColumn counts:")
print(f"  Base columns pickle: {len(base_columns)}")
print(f"  Base columns in original implementation: {len(LR_Base_original.columns)}")
print(f"  Base columns in corrected implementation: {len(LR_Base_corrected.columns)}")
print(f"  Yeo columns in original implementation: {len(LR_Yeo_original.columns)}")
print(f"  Yeo columns in corrected implementation: {len(LR_Yeo_corrected.columns)}")

# Identify what exactly is different in the Yeo implementation
original_yeo_columns = set(LR_Yeo_original.columns)
corrected_yeo_columns = set(LR_Yeo_corrected.columns)
missing_from_original = corrected_yeo_columns - original_yeo_columns
extra_in_original = original_yeo_columns - corrected_yeo_columns

print(f"\nDetailed column comparison:")
print(f"  Missing from original implementation: {len(missing_from_original)} columns")
if len(missing_from_original) > 0:
    print(f"  First 5 missing columns: {list(missing_from_original)[:5]}")
    
    # Analyze types of missing columns
    missing_prefixes = {}
    for col in missing_from_original:
        prefix = col.split('_')[0] if '_' in col else 'other'
        missing_prefixes[prefix] = missing_prefixes.get(prefix, 0) + 1
    
    print(f"  Types of missing columns:")
    for prefix, count in missing_prefixes.items():
        print(f"    {prefix}: {count}")

print(f"\nExtra in original implementation: {len(extra_in_original)} columns")
if len(extra_in_original) > 0:
    print(f"  Extra columns: {list(extra_in_original)}")

# ========================================================
# 5. Final validation
# ========================================================
print("\n5. FINAL VALIDATION")
print("-" * 50)

# Check if the corrected implementation matches expectations
yeo_prefix = 'yeo_joh_'
yeo_transformed_count = sum(1 for col in feature_df.columns if col.startswith(yeo_prefix))
categorical_count = len(base_columns) - yeo_transformed_count
expected_yeo_count = yeo_transformed_count + (len(base_columns) - yeo_transformed_count)

print(f"Yeo transformations in dataset: {yeo_transformed_count}")
print(f"Expected categorical columns: {categorical_count}")
print(f"Expected total Yeo columns: {expected_yeo_count}")
print(f"Actual Yeo columns in corrected implementation: {len(LR_Yeo_corrected.columns)}")

# Check if all the original categorical columns are included
def categorize_columns(columns):
    """Categorize columns by their type/prefix."""
    categories = {}
    for col in columns:
        parts = col.split('_')
        category = parts[0] if len(parts) > 1 else 'other'
        categories[category] = categories.get(category, 0) + 1
    return categories

original_categories = categorize_columns(original_yeo_columns)
corrected_categories = categorize_columns(corrected_yeo_columns)

print("\nColumn categories in original implementation:")
for category, count in original_categories.items():
    print(f"  {category}: {count}")

print("\nColumn categories in corrected implementation:")
for category, count in corrected_categories.items():
    print(f"  {category}: {count}")

print("\nVerification complete!")