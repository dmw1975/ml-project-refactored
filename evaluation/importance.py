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
    
    return importance_df

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
    
    # Get feature sets
    LR_Base, LR_Yeo, base_columns, yeo_columns = get_base_and_yeo_features(feature_df)
    
    # Create versions with random features
    LR_Base_random = add_random_feature(LR_Base)
    LR_Yeo_random = add_random_feature(LR_Yeo)
    
    # Dictionary to map model names to datasets
    dataset_map = {
        'LR_Base': LR_Base,
        'LR_Yeo': LR_Yeo,
        'LR_Base_Random': LR_Base_random,
        'LR_Yeo_Random': LR_Yeo_random,
        'ElasticNet_LR_Base': LR_Base,
        'ElasticNet_LR_Yeo': LR_Yeo,
        'ElasticNet_LR_Base_Random': LR_Base_random,
        'ElasticNet_LR_Yeo_Random': LR_Yeo_random
    }
    
    # Calculate importance for each model
    importance_results = {}
    random_feature_stats = []
    
    for model_name, model_data in all_models.items():
        print(f"Calculating feature importance for {model_name}...")
        
        # Get the model
        model = model_data['model']
        
        # Get corresponding dataset
        X_data = dataset_map.get(model_name, None)
        if X_data is None:
            print(f"No matching dataset found for {model_name}, skipping...")
            continue
        
        # Get test data
        y_test = model_data['y_test']
        y_pred = model_data['y_pred']
        
        # We need to align X_test with y_test
        X_test = X_data.loc[y_test.index]
        
        # Calculate importance
        importance_df = calculate_permutation_importance(
            model, X_test, y_test, 
            n_repeats=10, 
            random_state=settings.LINEAR_REGRESSION_PARAMS['random_state']
        )
        
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
    
    # Save all importance results
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