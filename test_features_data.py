from data import load_features_data, get_base_and_yeo_features

# Load feature data
feature_df = load_features_data()

# Get feature sets
LR_Base, LR_Yeo, base_columns, yeo_columns = get_base_and_yeo_features(feature_df)

# Print dimensions
print(f"LR_Base dimensions: {LR_Base.shape}")
print(f"LR_Yeo dimensions: {LR_Yeo.shape}")