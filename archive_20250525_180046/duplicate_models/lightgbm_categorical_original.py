"""LightGBM models with native categorical feature support."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import optuna
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from config import settings
from data_categorical import (
    load_tree_models_data, 
    get_base_and_yeo_features_categorical, 
    add_random_feature_categorical,
    get_categorical_features
)
from utils import io
from sklearn.model_selection import StratifiedShuffleSplit


def train_lightgbm_with_categorical(X, y, dataset_name, categorical_features=None):
    """
    Train LightGBM model with native categorical feature support.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features dataframe with categorical columns
    y : pd.Series
        Target variable
    dataset_name : str
        Name of the dataset for logging
    categorical_features : list, optional
        List of categorical feature column names
        
    Returns
    -------
    dict
        Dictionary containing model, metrics, and predictions
    """
    print(f"\n--- Training LightGBM with Categorical Support on {dataset_name} ---")
    
    if categorical_features is None:
        categorical_features = get_categorical_features()
    
    # Filter categorical features to only those present in X
    cat_features_present = [col for col in categorical_features if col in X.columns]
    print(f"Using {len(cat_features_present)} categorical features: {cat_features_present}")
    
    # Handle missing values in categorical features
    X_clean = X.copy()
    for cat_feature in cat_features_present:
        if cat_feature in X_clean.columns:
            if X_clean[cat_feature].dtype.name == 'category':
                if 'Unknown' not in X_clean[cat_feature].cat.categories:
                    X_clean[cat_feature] = X_clean[cat_feature].cat.add_categories(['Unknown'])
                X_clean[cat_feature] = X_clean[cat_feature].fillna('Unknown')
            else:
                X_clean[cat_feature] = X_clean[cat_feature].fillna('Unknown').astype('category')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.2, random_state=42, stratify=X_clean[cat_features_present[0]] if cat_features_present else None
    )
    
    # Create LightGBM datasets with categorical feature specification
    train_data = lgb.Dataset(
        X_train, 
        label=y_train,
        categorical_feature=cat_features_present,
        free_raw_data=False
    )
    
    # LightGBM parameters optimized for categorical features
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    # Train model with cross-validation
    print("Training with cross-validation...")
    from sklearn.model_selection import KFold
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = lgb.cv(
        params,
        train_data,
        num_boost_round=1000,
        folds=folds,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        return_cvbooster=True
    )
    
    best_iteration = len(cv_results['valid rmse-mean'])
    print(f"Best iteration: {best_iteration}")
    
    # Train final model on full training set
    model = lgb.train(
        params,
        train_data,
        num_boost_round=best_iteration,
        callbacks=[lgb.log_evaluation(0)]
    )
    
    # Make predictions
    y_train_pred = model.predict(X_train, num_iteration=best_iteration)
    y_test_pred = model.predict(X_test, num_iteration=best_iteration)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_clean.columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'feature_importance': feature_importance,
        'categorical_features': cat_features_present,
        'metrics': {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae
        }
    }


def train_lightgbm_categorical_models(datasets=['all']):
    """
    Train LightGBM models with categorical features on specified datasets.
    
    Parameters
    ----------
    datasets : list
        List of dataset names to train on
        
    Returns
    -------
    dict
        Dictionary containing all trained models and results
    """
    print("🌳 Training LightGBM models with native categorical features...")
    
    # Load categorical data
    features, target = load_tree_models_data()
    
    # Get feature sets
    base_features_df, yeo_features_df = get_base_and_yeo_features_categorical()
    
    results = {}
    
    # Define dataset configurations
    dataset_configs = {
        'Base': (base_features_df, 'Base features'),
        'Yeo': (yeo_features_df, 'Yeo-Johnson transformed features'),
    }
    
    # Add random feature variants
    if 'all' in datasets or any('Random' in d for d in datasets):
        for config_name, (feature_subset, description) in list(dataset_configs.items()):
            random_data = add_random_feature_categorical(features)
            dataset_configs[f'{config_name}_Random'] = (
                random_data[feature_subset.columns.tolist() + ['random_feature']], 
                f'{description} + random feature'
            )
    
    # Filter datasets based on request
    if 'all' not in datasets:
        dataset_configs = {k: v for k, v in dataset_configs.items() if k in datasets}
    
    # Train models
    for dataset_name, (X, description) in dataset_configs.items():
        print(f"\n{'='*60}")
        print(f"Training on {dataset_name}: {description}")
        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        
        try:
            result = train_lightgbm_with_categorical(X, target, dataset_name)
            results[f'LightGBM_{dataset_name}_categorical'] = result
            
            # Save model
            model_filename = f'lightgbm_{dataset_name.lower()}_categorical.pkl'
            model_path = io.save_model(result['model'], model_filename, settings.MODEL_DIR)
            print(f"Model saved to: {model_path}")
            
            # Save feature importance
            importance_path = settings.FEATURE_IMPORTANCE_DIR / f'LightGBM_{dataset_name}_categorical_importance.csv'
            result['feature_importance'].to_csv(importance_path, index=False)
            print(f"Feature importance saved to: {importance_path}")
            
        except Exception as e:
            print(f"Error training LightGBM on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ LightGBM categorical training completed! {len(results)} models trained.")
    return results


if __name__ == "__main__":
    # Test the implementation
    results = train_lightgbm_categorical_models()
    print("LightGBM categorical models training completed!")