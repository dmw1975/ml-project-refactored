#!/usr/bin/env python3
"""
Test script to compare categorical vs one-hot encoded feature importance.
This demonstrates the impact of switching from one-hot encoding to native categorical features.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def test_categorical_vs_onehot():
    """Compare CatBoost with categorical vs one-hot encoded features."""
    print("🧪 Testing Categorical vs One-Hot Encoded Features")
    print("=" * 60)
    
    # Load both datasets
    print("📊 Loading datasets...")
    
    # One-hot encoded data (current approach) - load from raw CSV
    from config import settings
    features_path = settings.DATA_DIR / "raw" / "combined_df_for_ml_models.csv"
    scores_path = settings.DATA_DIR / "raw" / "score.csv"
    
    X_onehot = pd.read_csv(features_path)
    scores_df = pd.read_csv(scores_path)
    y_onehot = scores_df['esg_score']
    
    print(f"One-hot encoded data: {X_onehot.shape[1]} features, {X_onehot.shape[0]} samples")
    
    # Categorical data (new approach)
    from data_categorical import load_tree_models_data, get_base_and_yeo_features_categorical, get_categorical_features
    tree_features, tree_target = load_tree_models_data()  # Returns tuple (features, target)
    base_features, yeo_features = get_base_and_yeo_features_categorical()
    categorical_features = get_categorical_features()
    
    X_categorical = base_features  # base_features is already a DataFrame
    y_categorical = tree_target    # Use the target from the tuple
    print(f"Categorical data: {X_categorical.shape[1]} features, {X_categorical.shape[0]} samples")
    print(f"Categorical features: {len(categorical_features)} ({categorical_features})")
    
    # Train CatBoost on one-hot encoded data
    print("\n🔢 Training CatBoost with One-Hot Encoded Features...")
    X_train_oh, X_test_oh, y_train_oh, y_test_oh = train_test_split(
        X_onehot, y_onehot, test_size=0.2, random_state=42
    )
    
    model_onehot = CatBoostRegressor(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        verbose=False,
        random_state=42
    )
    model_onehot.fit(X_train_oh, y_train_oh)
    
    y_pred_oh = model_onehot.predict(X_test_oh)
    rmse_oh = np.sqrt(mean_squared_error(y_test_oh, y_pred_oh))
    r2_oh = r2_score(y_test_oh, y_pred_oh)
    
    print(f"One-hot RMSE: {rmse_oh:.4f}")
    print(f"One-hot R²: {r2_oh:.4f}")
    
    # Train CatBoost on categorical data
    print("\n🌳 Training CatBoost with Native Categorical Features...")
    
    # Filter categorical features present in data
    cat_features_present = [col for col in categorical_features if col in X_categorical.columns]
    
    # Handle missing values
    X_cat_clean = X_categorical.copy()
    for cat_feature in cat_features_present:
        if X_cat_clean[cat_feature].dtype.name == 'category':
            if 'Unknown' not in X_cat_clean[cat_feature].cat.categories:
                X_cat_clean[cat_feature] = X_cat_clean[cat_feature].cat.add_categories(['Unknown'])
            X_cat_clean[cat_feature] = X_cat_clean[cat_feature].fillna('Unknown')
        else:
            X_cat_clean[cat_feature] = X_cat_clean[cat_feature].fillna('Unknown').astype('category')
    
    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
        X_cat_clean, y_categorical, test_size=0.2, random_state=42
    )
    
    model_categorical = CatBoostRegressor(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        verbose=False,
        random_state=42,
        cat_features=cat_features_present
    )
    model_categorical.fit(X_train_cat, y_train_cat)
    
    y_pred_cat = model_categorical.predict(X_test_cat)
    rmse_cat = np.sqrt(mean_squared_error(y_test_cat, y_pred_cat))
    r2_cat = r2_score(y_test_cat, y_pred_cat)
    
    print(f"Categorical RMSE: {rmse_cat:.4f}")
    print(f"Categorical R²: {r2_cat:.4f}")
    
    # Compare feature importance
    print("\n📈 Feature Importance Comparison")
    print("-" * 40)
    
    # One-hot encoded feature importance
    importance_oh = model_onehot.get_feature_importance()
    feature_names_oh = X_onehot.columns
    importance_df_oh = pd.DataFrame({
        'feature': feature_names_oh,
        'importance': importance_oh
    }).sort_values('importance', ascending=False)
    
    print("\n🔢 Top 15 One-Hot Encoded Features:")
    print(importance_df_oh.head(15).to_string(index=False))
    
    # Categorical feature importance  
    importance_cat = model_categorical.get_feature_importance()
    feature_names_cat = X_cat_clean.columns
    importance_df_cat = pd.DataFrame({
        'feature': feature_names_cat,
        'importance': importance_cat
    }).sort_values('importance', ascending=False)
    
    print("\n🌳 Top 15 Categorical Features:")
    print(importance_df_cat.head(15).to_string(index=False))
    
    # Analyze categorical feature ranking
    print("\n🔍 Categorical Feature Analysis:")
    categorical_in_top15 = importance_df_cat.head(15)['feature'].isin(cat_features_present).sum()
    print(f"Categorical features in top 15: {categorical_in_top15}")
    
    for cat_feature in cat_features_present:
        if cat_feature in importance_df_cat['feature'].values:
            rank = importance_df_cat[importance_df_cat['feature'] == cat_feature].index[0] + 1
            importance = importance_df_cat[importance_df_cat['feature'] == cat_feature]['importance'].iloc[0]
            print(f"  {cat_feature}: Rank {rank}, Importance {importance:.2f}")
    
    # Summary
    print("\n📊 SUMMARY")
    print("=" * 60)
    print(f"Feature space reduction: {X_onehot.shape[1]} → {X_categorical.shape[1]} features ({100*(X_onehot.shape[1]-X_categorical.shape[1])/X_onehot.shape[1]:.1f}% reduction)")
    print(f"Performance comparison:")
    print(f"  One-hot:     RMSE={rmse_oh:.4f}, R²={r2_oh:.4f}")
    print(f"  Categorical: RMSE={rmse_cat:.4f}, R²={r2_cat:.4f}")
    improvement = ((rmse_oh - rmse_cat) / rmse_oh) * 100
    print(f"  RMSE improvement: {improvement:+.2f}%")
    
    print(f"\nCategorical features now consolidated and visible in feature importance!")
    print(f"Instead of {len([f for f in feature_names_oh if any(cat in f for cat in ['shareholder_location', 'issuer_cntry', 'gics_sector', 'gics_sub_ind', 'cntry_of_risk'])])} scattered binary features,")
    print(f"we have {len(cat_features_present)} meaningful categorical features with unified importance scores.")

if __name__ == "__main__":
    test_categorical_vs_onehot()