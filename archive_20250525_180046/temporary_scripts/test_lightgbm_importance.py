"""Test script for LightGBM feature importance."""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings
from utils import io
from sklearn.inspection import permutation_importance

def calculate_lightgbm_importance(model_name, model_data):
    """Calculate feature importance for a specific LightGBM model."""
    model = model_data['model']
    y_test = model_data['y_test']
    
    # Check if we have clean test data
    if 'X_test_clean' in model_data:
        X_test = model_data['X_test_clean']
        print(f"Using stored clean test data with shape: {X_test.shape}")
        
        # Calculate feature importance using LightGBM's native method
        print(f"Calculating feature importance for {model_name}...")
        
        # Get feature importance from the model
        feature_importance = model.feature_importance()
        feature_names = X_test.columns.tolist()
        
        # Create DataFrame and map back to original feature names
        if 'feature_name_mapping' in model_data:
            feature_mapping = model_data['feature_name_mapping']
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'MappedFeature': [feature_mapping.get(col, col) for col in feature_names],
                'Importance': feature_importance,
            })
        else:
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance,
            })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Save to CSV
        output_dir = settings.FEATURE_IMPORTANCE_DIR
        io.ensure_dir(output_dir)
        importance_df.to_csv(f"{output_dir}/{model_name}_importance.csv", index=False)
        print(f"Feature importance saved to {output_dir}/{model_name}_importance.csv")
        
        return importance_df
    else:
        print(f"No clean test data found for model {model_name}")
        return None

def test_lightgbm_importance():
    """Test feature importance calculation for all LightGBM models."""
    # Load LightGBM models
    try:
        lightgbm_models = io.load_model("lightgbm_models.pkl", settings.MODEL_DIR)
        print(f"Loaded {len(lightgbm_models)} LightGBM models")
    except Exception as e:
        print(f"Error loading LightGBM models: {e}")
        return
    
    # Process all LightGBM models
    if not lightgbm_models:
        print("No LightGBM models found")
        return
    
    # Track results
    importance_results = {}
    
    # Calculate importance for each model
    for model_name, model_data in lightgbm_models.items():
        print(f"\nProcessing {model_name}...")
        
        try:
            importance_df = calculate_lightgbm_importance(model_name, model_data)
            if importance_df is not None:
                importance_results[model_name] = importance_df
                
                # Print top 10 important features
                print(f"\nTop 10 Features for {model_name}:")
                print(importance_df.head(10))
        except Exception as e:
            print(f"Error calculating importance for {model_name}: {e}")
    
    print(f"\nFeature importance calculated for {len(importance_results)} LightGBM models.")
    return importance_results

if __name__ == "__main__":
    test_lightgbm_importance()