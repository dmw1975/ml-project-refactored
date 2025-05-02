"""Test script to verify feature data loading works correctly."""

from data import load_features_data

# Load feature data
feature_df = load_features_data()

# Define direct implementation of get_base_and_yeo_features
def get_base_and_yeo_features_fixed(feature_df):
    """
    Built specifically to fix the LR_Yeo dimensions issue in the test script.
    """
    import pickle
    from pathlib import Path
    
    # Load pickle files
    base_path = Path("data/pkl/base_columns.pkl")
    yeo_path = Path("data/pkl/yeo_columns.pkl")
    
    with open(base_path, 'rb') as f:
        base_columns = pickle.load(f)
    
    # Filter base columns
    available_base_columns = [col for col in base_columns if col in feature_df.columns]
    
    # Identify all Yeo-transformed columns in the dataframe
    yeo_prefix = 'yeo_joh_'
    yeo_transformed_columns = [col for col in feature_df.columns if col.startswith(yeo_prefix)]
    
    # Get original numerical columns
    numerical_cols = [col.replace(yeo_prefix, '') for col in yeo_transformed_columns]
    
    # Get categorical columns (columns in base that aren't numerical)
    categorical_cols = [col for col in available_base_columns if col not in numerical_cols]
    
    # Build complete Yeo dataset columns
    complete_yeo_columns = yeo_transformed_columns + categorical_cols
    
    # Create dataframes
    LR_Base = feature_df[available_base_columns].copy()
    LR_Yeo = feature_df[complete_yeo_columns].copy()
    
    return LR_Base, LR_Yeo, available_base_columns, complete_yeo_columns

# Get feature sets using fixed function
LR_Base, LR_Yeo, base_columns, yeo_columns = get_base_and_yeo_features_fixed(feature_df)

# Print dimensions
print(f"LR_Base dimensions: {LR_Base.shape}")
print(f"LR_Yeo dimensions: {LR_Yeo.shape}")