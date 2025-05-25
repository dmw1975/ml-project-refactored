#!/usr/bin/env python3
"""Fix the metrics aggregation to properly combine all model metrics."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import settings


def standardize_metrics():
    """Standardize and aggregate all metrics files."""
    metrics_dir = settings.METRICS_DIR
    
    all_data = []
    
    # 1. Load Linear Regression metrics
    lr_file = metrics_dir / 'linear_regression_metrics.csv'
    if lr_file.exists():
        df = pd.read_csv(lr_file)
        # Rename columns to match standard format
        if 'model' in df.columns:
            df = df.rename(columns={'model': 'model_name'})
        all_data.append(df)
        print(f"✓ Loaded {len(df)} Linear Regression entries")
    
    # 2. Load ElasticNet metrics
    en_file = metrics_dir / 'elasticnet_metrics.csv'
    if en_file.exists():
        df = pd.read_csv(en_file)
        if 'model' in df.columns:
            df = df.rename(columns={'model': 'model_name'})
        all_data.append(df)
        print(f"✓ Loaded {len(df)} ElasticNet entries")
    
    # 3. Load XGBoost metrics
    xgb_file = metrics_dir / 'xgboost_metrics.csv'
    if xgb_file.exists():
        df = pd.read_csv(xgb_file)
        # Add missing categorical XGBoost models
        xgb_categorical = []
        
        # Define the categorical XGBoost configurations
        configs = [
            ('Base', 'basic'),
            ('Base', 'optuna'),
            ('Yeo', 'basic'),
            ('Yeo', 'optuna'),
            ('Base_Random', 'basic'),
            ('Base_Random', 'optuna'),
            ('Yeo_Random', 'basic'),
            ('Yeo_Random', 'optuna')
        ]
        
        for dataset, opt_type in configs:
            xgb_categorical.append({
                'model_name': f'XGBoost_{dataset}_categorical_{opt_type}',
                'RMSE': np.nan,  # Will be filled from model files
                'MAE': np.nan,
                'MSE': np.nan,
                'R2': np.nan,
                'n_companies': 396,  # Tree models use 396 companies
                'n_features': 60,    # Tree models use 60 features
                'model_type': f'XGBoost {opt_type.capitalize()}'
            })
        
        # Combine with existing XGBoost data
        all_data.append(df)
        all_data.append(pd.DataFrame(xgb_categorical))
        print(f"✓ Loaded {len(df)} XGBoost entries + {len(xgb_categorical)} categorical entries")
    
    # 4. Load LightGBM metrics
    lgb_file = metrics_dir / 'lightgbm_metrics.csv'
    if lgb_file.exists():
        df = pd.read_csv(lgb_file)
        # Add categorical LightGBM entries
        lgb_categorical = []
        
        configs = [
            ('Base', 'basic'),
            ('Base', 'optuna'),
            ('Yeo', 'basic'), 
            ('Yeo', 'optuna'),
            ('Base_Random', 'basic'),
            ('Base_Random', 'optuna'),
            ('Yeo_Random', 'basic'),
            ('Yeo_Random', 'optuna')
        ]
        
        for dataset, opt_type in configs:
            lgb_categorical.append({
                'model_name': f'LightGBM_{dataset}_categorical_{opt_type}',
                'RMSE': np.nan,  # Will be filled from model files
                'MAE': np.nan,
                'MSE': np.nan,
                'R2': np.nan,
                'n_companies': 396,
                'n_features': 60,
                'model_type': f'LightGBM {opt_type.capitalize()}'
            })
        
        all_data.append(df)
        all_data.append(pd.DataFrame(lgb_categorical))
        print(f"✓ Loaded {len(df)} LightGBM entries + {len(lgb_categorical)} categorical entries")
    
    # 5. Load CatBoost categorical metrics  
    cb_file = metrics_dir / 'catboost_categorical_metrics.csv'
    if cb_file.exists():
        df = pd.read_csv(cb_file)
        # Transform CatBoost metrics to standard format
        standardized = []
        
        for _, row in df.iterrows():
            # Extract model type (basic or optuna)
            model_type = 'basic' if '_basic' in row['model'] else 'optuna'
            
            standardized.append({
                'model_name': row['model'],
                'RMSE': np.sqrt(row['test_mse']) if 'test_mse' in row else np.nan,
                'MAE': row.get('test_mae', np.nan),
                'MSE': row.get('test_mse', np.nan),
                'R2': row.get('test_r2', np.nan),
                'n_companies': 441,  # CatBoost uses different split
                'n_features': row.get('total_features', 60),
                'model_type': f'CatBoost {model_type.capitalize()}'
            })
        
        all_data.append(pd.DataFrame(standardized))
        print(f"✓ Loaded {len(standardized)} CatBoost entries")
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Load actual metrics from model pickle files to fill NaN values
        try:
            from utils.io import load_model
            
            # Load XGBoost models
            try:
                xgb_models = load_model('xgboost_models.pkl', settings.MODEL_DIR)
                for model_name, model_data in xgb_models.items():
                    if isinstance(model_data, dict) and 'metrics' in model_data:
                        metrics = model_data['metrics']
                        # Update the corresponding row
                        mask = combined_df['model_name'] == model_name
                        if mask.any():
                            combined_df.loc[mask, 'RMSE'] = metrics.get('test_rmse', np.nan)
                            combined_df.loc[mask, 'MAE'] = metrics.get('test_mae', np.nan)
                            combined_df.loc[mask, 'MSE'] = metrics.get('test_mse', np.nan)
                            combined_df.loc[mask, 'R2'] = metrics.get('test_r2', np.nan)
                print("✓ Updated XGBoost categorical metrics from model files")
            except:
                pass
            
            # Load LightGBM models
            try:
                lgb_models = load_model('lightgbm_models.pkl', settings.MODEL_DIR)
                for model_name, model_data in lgb_models.items():
                    if isinstance(model_data, dict) and 'metrics' in model_data:
                        metrics = model_data['metrics']
                        # Update the corresponding row
                        mask = combined_df['model_name'] == model_name
                        if mask.any():
                            combined_df.loc[mask, 'RMSE'] = metrics.get('test_rmse', np.nan)
                            combined_df.loc[mask, 'MAE'] = metrics.get('test_mae', np.nan)
                            combined_df.loc[mask, 'MSE'] = metrics.get('test_mse', np.nan)
                            combined_df.loc[mask, 'R2'] = metrics.get('test_r2', np.nan)
                print("✓ Updated LightGBM categorical metrics from model files")
            except:
                pass
                
        except Exception as e:
            print(f"Warning: Could not load model files for metric updates: {e}")
        
        # Remove rows with all NaN metrics
        metric_cols = ['RMSE', 'MAE', 'MSE', 'R2']
        combined_df = combined_df.dropna(subset=metric_cols, how='all')
        
        # Sort by RMSE
        combined_df = combined_df.sort_values('RMSE', na_position='last')
        
        # Save the cleaned and combined metrics
        output_file = metrics_dir / 'all_models_comparison.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved {len(combined_df)} total entries to {output_file}")
        
        # Print summary
        print("\nModel type summary:")
        if 'model_type' in combined_df.columns:
            print(combined_df['model_type'].value_counts())
        
        # Show best models
        print("\nTop 10 models by RMSE:")
        print(combined_df[['model_name', 'RMSE', 'R2', 'model_type']].head(10))
        
        return combined_df
    else:
        print("No metrics files found!")
        return None


if __name__ == "__main__":
    standardize_metrics()