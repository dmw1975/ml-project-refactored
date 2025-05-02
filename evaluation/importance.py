"""Feature importance analysis for trained models."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from config import settings
from utils import io
from data import load_features_data, load_scores_data, get_base_and_yeo_features, add_random_feature

def calculate_permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42):
    """Calculate permutation importance for a model."""
    try:
        # Verify feature alignment between model and X_test
        if hasattr(model, 'feature_names_in_'):
            # Check if the model's expected features match the test data
            expected_features = model.feature_names_in_
            
            # Print feature counts for verification
            print(f"Model expects {len(expected_features)} features")
            print(f"X_test has {X_test.shape[1]} features")
            
            # Check for missing features
            if len(expected_features) != X_test.shape[1]:
                print(f"WARNING: Feature count mismatch. Model expects {len(expected_features)} features but X_test has {X_test.shape[1]}")
                
                # Identify missing features in both directions
                missing_in_X = [f for f in expected_features if f not in X_test.columns]
                extra_in_X = [f for f in X_test.columns if f not in expected_features]
                
                if missing_in_X:
                    print(f"Features expected by model but missing in X_test: {missing_in_X[:5] if len(missing_in_X) > 5 else missing_in_X}")
                if extra_in_X:
                    print(f"Extra features in X_test not used by model: {extra_in_X[:5] if len(extra_in_X) > 5 else extra_in_X}")
                
                # Ensure X_test only contains the features the model expects, in the right order
                try:
                    X_test = X_test[expected_features].copy()
                    print("Successfully aligned X_test features with model's expected features")
                except Exception as align_err:
                    print(f"ERROR: Could not align features: {align_err}")
                    raise
            else:
                # Even if counts match, ensure ordering is correct
                X_test = X_test[expected_features].copy()
                print("Features are already aligned (same count and names)")
        
        # Now calculate permutation importance
        perm_importance = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': perm_importance.importances_mean,
            'Std': perm_importance.importances_std
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Print summary of resulting importance
        print(f"Calculated importance for {len(importance_df)} features")
        print(f"Top 5 important features: {', '.join(importance_df['Feature'].head(5).tolist())}")
        
        return importance_df
    except Exception as e:
        print(f"Error in permutation importance calculation: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_feature_importance(all_models=None):
    """Analyze feature importance for all models."""
    if all_models is None:
        # Load evaluation metrics
        from evaluation.metrics import load_all_models
        all_models = load_all_models()
    
    if not all_models:
        print("No models found. Please train models first.")
        return
    
    # Load original data to get feature sets
    feature_df = load_features_data()
    score_df = load_scores_data()
    
    # Dictionary to store the feature sets used during training
    trained_feature_sets = {}
    
    # For each model, store its feature set
    for model_name, model_data in all_models.items():
        model = model_data['model']
        if hasattr(model, 'feature_names_in_'):
            trained_feature_sets[model_name] = model.feature_names_in_
    
    # Calculate importance for each model
    importance_results = {}
    random_feature_stats = []
    
    for model_name, model_data in all_models.items():
        print(f"Calculating feature importance for {model_name}...")
        
        # Get the model
        model = model_data['model']
        
        # Get test data indices
        y_test = model_data['y_test']
        test_indices = y_test.index if hasattr(y_test, 'index') else np.arange(len(y_test))
        
        # Use the original feature dataframe and extract only the needed features
        X_test = None  # Initialize to None to check if it gets set properly
        
        # Special handling for LightGBM models
        if 'LightGBM_' in model_name and 'X_test_clean' in model_data:
            print(f"Using stored clean test data for LightGBM model {model_name}")
            X_test = model_data['X_test_clean']
            print(f"Clean test data shape: {X_test.shape}")
            
            # Use the feature name mapping for the feature names
            if 'feature_name_mapping' in model_data:
                feature_names = list(model_data['feature_name_mapping'].keys())
                print(f"Using {len(feature_names)} feature names from mapping")
            else:
                feature_names = None
        # Special handling for CatBoost models
        elif 'CatBoost_' in model_name and 'X_test' in model_data:
            print(f"Using stored test data for CatBoost model {model_name}")
            X_test = model_data['X_test']
            print(f"Test data shape: {X_test.shape}")
            
            # Use feature_names from the model data
            if 'feature_names' in model_data:
                feature_names = model_data['feature_names']
                print(f"Using {len(feature_names)} feature names from model data")
            else:
                feature_names = None
        elif model_name in trained_feature_sets:
            feature_names = trained_feature_sets[model_name]
            
            # Check if all required features exist in the original dataframe
            missing_features = [f for f in feature_names if f not in feature_df.columns]
            
            # Handle missing features
            if missing_features:
                print(f"Warning: {len(missing_features)} features used in training are missing in original dataset")
                print(f"First few missing: {missing_features[:5]}")
                
                # Check if only the random feature is missing
                if len(missing_features) == 1 and missing_features[0] == 'random_feature':
                    print("Only random feature is missing, attempting to reconstruct...")
                    try:
                        # For Yeo models with random feature, we need to be more careful
                        if "Yeo" in model_name:
                            # Get the feature names from the model (excluding the random feature)
                            model_features = [f for f in feature_names if f != 'random_feature']
                            
                            # Create a DataFrame with all those features from the original data
                            required_features = [f for f in model_features if f in feature_df.columns]
                            
                            # Check if we have all the required features
                            if len(required_features) != len(model_features):
                                print(f"Missing {len(model_features) - len(required_features)} required features, cannot reconstruct")
                                continue  # Skip this model
                                
                            # Build dataset from the original features
                            base_df = feature_df[required_features].copy()
                            
                            # Add random feature
                            X_data = add_random_feature(base_df)
                            
                            # Use only test indices
                            X_test = X_data.loc[test_indices]
                            
                            # Make sure feature order exactly matches what the model expects
                            X_test = X_test[feature_names]
                        else:
                            # For non-Yeo models with random feature
                            base_df, _, _, _ = get_base_and_yeo_features(feature_df)
                            X_data = add_random_feature(base_df)
                            X_test = X_data.loc[test_indices]
                            
                            # Make sure feature order exactly matches what the model expects
                            if not all(f in X_test.columns for f in feature_names):
                                print("Error: Not all required features available after reconstruction")
                                continue  # Skip this model
                                
                            X_test = X_test[feature_names]
                            
                        print(f"Successfully reconstructed dataset with random feature, shape: {X_test.shape}")
                        
                    except Exception as e:
                        print(f"Error reconstructing random feature: {e}")
                        continue  # Skip this model
                else:
                    print("Multiple features missing, skipping importance calculation for this model")
                    continue  # Skip this model
            else:
                # Extract just the test set with the correct features
                X_test = feature_df.loc[test_indices, feature_names]
        else:
            print(f"No feature information available for model {model_name}, skipping")
            continue  # Skip this model
            
        # Verify we have a valid X_test dataset
        if X_test is None or X_test.empty:
            print(f"Error: Failed to create valid test dataset for {model_name}")
            continue  # Skip this model
            
        # Double-check feature alignment
        if feature_names is not None and not all(f in X_test.columns for f in feature_names):
            print(f"Error: Feature mismatch after dataset preparation")
            missing = [f for f in feature_names if f not in X_test.columns]
            print(f"Missing features: {missing[:5]}")
            continue  # Skip this model
            
        # Additional debug output
        print(f"Ready to calculate importance with dataset shape: {X_test.shape}")
        
        # Calculate importance
        try:
            print(f"Starting importance calculation for {model_name}...")
            importance_df = calculate_permutation_importance(
                model, X_test, y_test, 
                n_repeats=10, 
                random_state=settings.LINEAR_REGRESSION_PARAMS['random_state']
            )
            
            if importance_df is None:
                print(f"Skipping {model_name} due to error in importance calculation")
                continue
                
            print(f"Importance calculation successful for {model_name}")
                
            # Save to results
            importance_results[model_name] = importance_df
            
            # Save to CSV
            output_dir = settings.FEATURE_IMPORTANCE_DIR
            io.ensure_dir(output_dir)
            importance_df.to_csv(f"{output_dir}/{model_name}_importance.csv", index=False)
            
            # Check if random feature is present
            if 'random_feature' in importance_df['Feature'].values:
                # Get random feature stats
                random_row = importance_df[importance_df['Feature'] == 'random_feature']
                random_rank = random_row.index[0] + 1
                random_importance = random_row['Importance'].values[0]
                
                random_feature_stats.append({
                    'model_name': model_name,
                    'random_rank': random_rank,
                    'random_importance': random_importance,
                    'total_features': len(importance_df),
                    'percentile': (random_rank / len(importance_df)) * 100
                })
        except Exception as e:
            print(f"Error calculating importance for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save all importance results
    if importance_results:
        output_dir = settings.FEATURE_IMPORTANCE_DIR
        io.ensure_dir(output_dir)
        io.save_model(importance_results, "feature_importance.pkl", output_dir)
        
        # Create random feature stats DataFrame
        if random_feature_stats:
            random_df = pd.DataFrame(random_feature_stats)
            random_df.to_csv(f"{output_dir}/random_feature_stats.csv", index=False)
            
            print("\nRandom Feature Performance:")
            print("==========================")
            for _, row in random_df.iterrows():
                print(f"{row['model_name']}: Rank {row['random_rank']}/{row['total_features']} ({row['percentile']:.1f}%), Importance: {row['random_importance']:.6f}")
        
        # Create consolidated importance table
        consolidated = create_consolidated_importance_table(importance_results)
        
        print("\nFeature importance analysis complete. Results saved to feature_importance directory.")
        return importance_results, consolidated
    else:
        print("No successful importance calculations. Check the errors above.")
        return None, None

def create_consolidated_importance_table(importance_results):
    """Create a consolidated table of feature importance across models."""
    # Get all unique features
    all_features = set()
    for model_name, importance_df in importance_results.items():
        all_features.update(importance_df['Feature'])
    
    # Create DataFrame
    consolidated = pd.DataFrame(index=list(all_features))
    
    # Add importance values for each model
    for model_name, importance_df in importance_results.items():
        # Convert to dictionary for easier lookup
        importance_dict = dict(zip(importance_df['Feature'], importance_df['Importance']))
        
        # Add to consolidated DataFrame
        consolidated[model_name] = consolidated.index.map(lambda x: importance_dict.get(x, 0))
    
    # Add average importance
    consolidated['avg_importance'] = consolidated.mean(axis=1)
    
    # Sort by average importance
    consolidated = consolidated.sort_values('avg_importance', ascending=False)
    
    # Save to CSV
    output_dir = settings.FEATURE_IMPORTANCE_DIR
    consolidated.to_csv(f"{output_dir}/consolidated_importance.csv")
    
    # Print top features
    print("\nTop 10 Features by Average Importance:")
    print("====================================")
    top_features = consolidated.head(10)
    print(top_features[['avg_importance']].round(4))
    
    return consolidated

if __name__ == "__main__":
    # Run feature importance analysis
    analyze_feature_importance()