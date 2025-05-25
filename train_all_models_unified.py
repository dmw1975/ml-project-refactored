#!/usr/bin/env python3
"""
Train all models using the unified data pipeline to ensure consistent train/test splits.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings
from utils import io
from create_unified_data_pipeline import load_unified_data, create_unified_datasets
from data import get_base_and_yeo_features, add_random_feature


def train_linear_regression_unified():
    """Train linear regression models with unified data."""
    print("\n" + "="*60)
    print("Training Linear Regression Models with Unified Data")
    print("="*60)
    
    from sklearn.linear_model import LinearRegression
    
    # Load unified data
    features, target, train_idx, test_idx = load_unified_data(model_type='linear')
    
    # Get base and yeo features
    LR_Base, LR_Yeo, _, _ = get_base_and_yeo_features(features)
    
    # Add random features
    LR_Base_random = add_random_feature(LR_Base)
    LR_Yeo_random = add_random_feature(LR_Yeo)
    
    datasets = {
        'LR_Base': LR_Base,
        'LR_Yeo': LR_Yeo,
        'LR_Base_Random': LR_Base_random,
        'LR_Yeo_Random': LR_Yeo_random
    }
    
    model_results = {}
    
    for name, X in datasets.items():
        print(f"\nTraining {name}...")
        
        # Use unified indices
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = target.iloc[train_idx]
        y_test = target.iloc[test_idx]
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        
        # Metrics
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Test samples: {len(y_test)}")
        
        # Store results
        model_results[name] = {
            'model_name': name,
            'model': model,
            'RMSE': test_rmse,
            'MAE': test_mae,
            'MSE': test_rmse**2,
            'R2': test_r2,
            'n_companies': len(X),
            'n_companies_train': len(X_train),
            'n_companies_test': len(X_test),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_train': y_train,
            'y_train_pred': y_train_pred,
            'X_train': X_train,
            'X_test': X_test,
            'n_features': X.shape[1]
        }
    
    # Save models
    io.save_model(model_results, "linear_regression_models_unified.pkl", settings.MODEL_DIR)
    print("\nLinear regression models saved!")
    return model_results


def train_elasticnet_unified():
    """Train ElasticNet models with unified data."""
    print("\n" + "="*60)
    print("Training ElasticNet Models with Unified Data")
    print("="*60)
    
    from sklearn.linear_model import ElasticNetCV
    
    # Load unified data
    features, target, train_idx, test_idx = load_unified_data(model_type='linear')
    
    # Get base and yeo features
    LR_Base, LR_Yeo, _, _ = get_base_and_yeo_features(features)
    
    # Add random features
    LR_Base_random = add_random_feature(LR_Base)
    LR_Yeo_random = add_random_feature(LR_Yeo)
    
    datasets = {
        'ElasticNet_LR_Base': LR_Base,
        'ElasticNet_LR_Yeo': LR_Yeo,
        'ElasticNet_LR_Base_Random': LR_Base_random,
        'ElasticNet_LR_Yeo_Random': LR_Yeo_random
    }
    
    model_results = {}
    
    for name, X in datasets.items():
        print(f"\nTraining {name}...")
        
        # Use unified indices
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = target.iloc[train_idx]
        y_test = target.iloc[test_idx]
        
        # Train model with CV
        model = ElasticNetCV(
            l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
            alphas=np.logspace(-4, 0, 20),
            cv=5,
            random_state=settings.ELASTICNET_PARAMS['random_state']
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        
        # Metrics
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Alpha: {model.alpha_:.4f}, L1 ratio: {model.l1_ratio_:.4f}")
        print(f"  Test samples: {len(y_test)}")
        
        # Store results
        model_results[name] = {
            'model_name': name,
            'model': model,
            'RMSE': test_rmse,
            'MAE': test_mae,
            'MSE': test_rmse**2,
            'R2': test_r2,
            'alpha': model.alpha_,
            'l1_ratio': model.l1_ratio_,
            'n_companies': len(X),
            'n_companies_train': len(X_train),
            'n_companies_test': len(X_test),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_train': y_train,
            'y_train_pred': y_train_pred,
            'X_train': X_train,
            'X_test': X_test,
            'n_features': X.shape[1]
        }
    
    # Save models
    io.save_model(model_results, "elasticnet_models_unified.pkl", settings.MODEL_DIR)
    print("\nElasticNet models saved!")
    return model_results


def train_xgboost_unified():
    """Train XGBoost models with unified data."""
    print("\n" + "="*60)
    print("Training XGBoost Models with Unified Data")
    print("="*60)
    
    from xgboost import XGBRegressor
    
    # Load unified data for tree models
    features, target, train_idx, test_idx = load_unified_data(model_type='tree')
    
    # Get categorical features
    import json
    unified_dir = settings.PROCESSED_DATA_DIR / "unified"
    with open(unified_dir / "categorical_features.json", 'r') as f:
        cat_features = json.load(f)
    
    # Get base and yeo features (for tree models)
    # Separate quantitative and categorical columns
    quant_cols = [col for col in features.columns if col not in cat_features]
    base_quant_cols = [col for col in quant_cols if not col.startswith('yeo_joh_')]
    yeo_quant_cols = [col for col in quant_cols if col.startswith('yeo_joh_')]
    
    # Create datasets
    base_features = features[base_quant_cols + cat_features]
    yeo_features = features[yeo_quant_cols + cat_features]
    
    # Add random features
    base_random = base_features.copy()
    base_random['random_feature'] = np.random.randn(len(base_random))
    
    yeo_random = yeo_features.copy()
    yeo_random['random_feature'] = np.random.randn(len(yeo_random))
    
    datasets = {
        'XGBoost_Base_categorical': base_features,
        'XGBoost_Yeo_categorical': yeo_features,
        'XGBoost_Base_Random_categorical': base_random,
        'XGBoost_Yeo_Random_categorical': yeo_random
    }
    
    model_results = {}
    
    for name, X in datasets.items():
        print(f"\nTraining {name}...")
        
        # Use unified indices
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = target.iloc[train_idx]
        y_test = target.iloc[test_idx]
        
        # Prepare data for XGBoost
        # Enable categorical support
        enable_categorical = True
        
        # Train model
        model = XGBRegressor(
            n_estimators=100,
            random_state=42,
            enable_categorical=enable_categorical,
            tree_method='hist'
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        
        # Metrics
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Test samples: {len(y_test)}")
        
        # Store results
        model_results[f"{name}_basic"] = {
            'model': model,
            'model_type': 'xgboost',
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_train_pred': y_train_pred,
            'train_score': r2_score(y_train, y_train_pred),
            'test_score': test_r2,
            'metrics': {
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'test_rmse': test_rmse,
                'train_mae': mean_absolute_error(y_train, y_train_pred),
                'test_mae': test_mae,
                'train_r2': r2_score(y_train, y_train_pred),
                'test_r2': test_r2
            },
            'categorical_features': cat_features,
            'n_features': X.shape[1]
        }
    
    # Save models
    io.save_model(model_results, "xgboost_models_unified.pkl", settings.MODEL_DIR)
    print("\nXGBoost models saved!")
    return model_results


def verify_unified_consistency():
    """Verify that all models have the same test set size."""
    print("\n" + "="*60)
    print("Verifying Unified Model Consistency")
    print("="*60)
    
    model_files = [
        "linear_regression_models_unified.pkl",
        "elasticnet_models_unified.pkl",
        "xgboost_models_unified.pkl"
    ]
    
    test_sizes = {}
    
    for file in model_files:
        try:
            models = io.load_model(file, settings.MODEL_DIR)
            for model_name, model_data in models.items():
                if 'y_test' in model_data:
                    test_size = len(model_data['y_test'])
                    model_type = file.replace("_unified.pkl", "")
                    
                    if model_type not in test_sizes:
                        test_sizes[model_type] = []
                    test_sizes[model_type].append((model_name, test_size))
        except:
            print(f"Could not load {file}")
    
    # Check consistency
    all_sizes = []
    for model_type, sizes in test_sizes.items():
        print(f"\n{model_type}:")
        for name, size in sizes:
            print(f"  {name}: {size} test samples")
            all_sizes.append(size)
    
    if len(set(all_sizes)) == 1:
        print(f"\n✅ SUCCESS: All models use the same test set size: {all_sizes[0]}")
    else:
        print(f"\n❌ ERROR: Models have different test set sizes: {set(all_sizes)}")


if __name__ == "__main__":
    # First create unified datasets if they don't exist
    unified_dir = settings.PROCESSED_DATA_DIR / "unified"
    if not (unified_dir / "train_test_split.pkl").exists():
        print("Creating unified datasets...")
        create_unified_datasets()
    
    # Train all models with unified data
    print("\nTraining all models with unified data...")
    
    # Train linear regression
    train_linear_regression_unified()
    
    # Train ElasticNet
    train_elasticnet_unified()
    
    # Train XGBoost
    train_xgboost_unified()
    
    # Verify consistency
    verify_unified_consistency()
    
    print("\n" + "="*60)
    print("All models trained with unified data!")
    print("Run baseline evaluation to see consistent baseline values.")
    print("="*60)