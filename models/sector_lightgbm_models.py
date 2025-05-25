"""Sector-specific LightGBM models for ESG score prediction."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from config import settings
from data import load_features_data, load_scores_data, get_base_and_yeo_features, add_random_feature
from utils import io

def run_sector_lightgbm_models(feature_df=None, score_df=None, base_columns=None, yeo_columns=None, 
                               LR_Base_random=None, LR_Yeo_random=None,
                               random_state=42, test_size=0.2):
    """
    Run LightGBM models separately for each GICS sector.
    
    Parameters:
    -----------
    feature_df : pandas.DataFrame, optional
        Full feature dataframe. If None, it will be loaded.
    score_df : pandas.Series or DataFrame, optional
        Target variable (ESG scores). If None, it will be loaded.
    base_columns : list, optional
        List of base feature columns. If None, they will be extracted.
    yeo_columns : list, optional
        List of Yeo-Johnson transformed feature columns. If None, they will be extracted.
    LR_Base_random : pandas.DataFrame, optional
        Base dataset with random feature. If None, it will be created.
    LR_Yeo_random : pandas.DataFrame, optional
        Yeo-Johnson dataset with random feature. If None, it will be created.
    random_state : int, default=42
        Reproducibility seed
    test_size : float, default=0.2
        Proportion of data for testing
    """
    # Force reload data module to ensure latest version
    import importlib
    import data
    importlib.reload(data)
    from data import load_features_data, load_scores_data, get_base_and_yeo_features, add_random_feature
    
    print("Loading data for sector-specific LightGBM models...")
    
    # Load data if not provided
    if feature_df is None:
        feature_df = load_features_data()
    
    if score_df is None:
        score_df = load_scores_data()
    
    # Get feature sets if not provided
    if base_columns is None or yeo_columns is None:
        LR_Base, LR_Yeo, base_columns, yeo_columns = get_base_and_yeo_features(feature_df)
        
        # Direct feature count check before continuing
        print(f"\nDIRECT FEATURE COUNT CHECK (AFTER LOADING):")
        print(f"LR_Base column count: {len(LR_Base.columns)}")
        print(f"LR_Yeo column count: {len(LR_Yeo.columns)}")
        
        # If LR_Yeo has less features, fix it directly here
        if len(LR_Yeo.columns) < len(LR_Base.columns):
            print(f"WARNING: LR_Yeo has fewer columns than expected, forcing fix...")
            # Identify all Yeo-transformed columns
            yeo_prefix = 'yeo_joh_'
            yeo_transformed_columns = [col for col in feature_df.columns if col.startswith(yeo_prefix)]
            original_numerical_columns = [col.replace(yeo_prefix, '') for col in yeo_transformed_columns]
            categorical_columns = [col for col in LR_Base.columns if col not in original_numerical_columns]
            complete_yeo_columns = yeo_transformed_columns + categorical_columns
            LR_Yeo = feature_df[complete_yeo_columns].copy()
            print(f"Fixed LR_Yeo column count: {len(LR_Yeo.columns)}")
    
    # Create random feature datasets if not provided
    if LR_Base_random is None:
        if 'LR_Base' not in locals():
            LR_Base, _, _, _ = get_base_and_yeo_features(feature_df)
        LR_Base_random = add_random_feature(LR_Base)
    
    if LR_Yeo_random is None:
        if 'LR_Yeo' not in locals():
            _, LR_Yeo, _, _ = get_base_and_yeo_features(feature_df)
            
            # Check if LR_Yeo needs fixing
            if 'LR_Base' in locals() and len(LR_Yeo.columns) < len(LR_Base.columns):
                print(f"WARNING: LR_Yeo for random features has fewer columns than expected, forcing fix...")
                yeo_prefix = 'yeo_joh_'
                yeo_transformed_columns = [col for col in feature_df.columns if col.startswith(yeo_prefix)]
                original_numerical_columns = [col.replace(yeo_prefix, '') for col in yeo_transformed_columns]
                categorical_columns = [col for col in LR_Base.columns if col not in original_numerical_columns]
                complete_yeo_columns = yeo_transformed_columns + categorical_columns
                LR_Yeo = feature_df[complete_yeo_columns].copy()
                print(f"Fixed LR_Yeo column count: {len(LR_Yeo.columns)}")
        
        LR_Yeo_random = add_random_feature(LR_Yeo)
    
    # Initialize dictionary to store model metrics
    sector_model_results = {}
    
    # Identify sector columns
    sector_columns = [col for col in feature_df.columns if col.startswith('gics_sector_')]
    
    # Define minimum companies needed per sector for modeling
    MIN_COMPANIES = 50
    
    print("\n" + "="*65)
    print("Running LightGBM Models by Sector")
    print("="*65)
    
    # For each sector, create and evaluate models
    for sector_col in sector_columns:
        sector_name = sector_col.replace('gics_sector_', '')
        # Filter for companies in this sector
        sector_mask = feature_df[sector_col] == 1
        X_sector = feature_df[sector_mask]
        y_sector = score_df[sector_mask]
        
        # Check if we have enough companies in this sector
        if len(X_sector) < MIN_COMPANIES:
            print(f"\nSkipping {sector_name} - insufficient data ({len(X_sector)} companies, need {MIN_COMPANIES})")
            continue
        
        print(f"\n{sector_name} Sector - {len(X_sector)} companies")
        print("-" * 50)
        
        # Create feature sets for this sector
        X_sector_base = X_sector[base_columns]
        X_sector_yeo = X_sector[yeo_columns]
        
        # Filter random datasets by the same sector indices
        sector_indices = X_sector.index
        X_sector_base_random = LR_Base_random.loc[sector_indices]
        X_sector_yeo_random = LR_Yeo_random.loc[sector_indices]
        
        # Simple train-test split for regular features
        X_train_base, X_test_base, y_train, y_test = train_test_split(
            X_sector_base, y_sector, test_size=test_size, random_state=random_state
        )
        
        # Use the same indices for the Yeo features
        train_indices = X_train_base.index
        test_indices = X_test_base.index
        X_train_yeo = X_sector_yeo.loc[train_indices]
        X_test_yeo = X_sector_yeo.loc[test_indices]
        
        # Split random feature datasets using the same indices
        X_train_base_random = X_sector_base_random.loc[train_indices]
        X_test_base_random = X_sector_base_random.loc[test_indices]
        X_train_yeo_random = X_sector_yeo_random.loc[train_indices]
        X_test_yeo_random = X_sector_yeo_random.loc[test_indices]
        
        # Train and evaluate all model variations
        model_configs = [
            {
                'name': f"Sector_LightGBM_{sector_name}_Base", 
                'X_train': X_train_base, 
                'X_test': X_test_base, 
                'type': 'Base',
                'feature_list': base_columns
            },
            {
                'name': f"Sector_LightGBM_{sector_name}_Yeo", 
                'X_train': X_train_yeo, 
                'X_test': X_test_yeo, 
                'type': 'Yeo',
                'feature_list': yeo_columns
            },
            {
                'name': f"Sector_LightGBM_{sector_name}_Base_Random", 
                'X_train': X_train_base_random, 
                'X_test': X_test_base_random, 
                'type': 'Base+Random',
                'feature_list': list(X_train_base_random.columns)
            },
            {
                'name': f"Sector_LightGBM_{sector_name}_Yeo_Random", 
                'X_train': X_train_yeo_random, 
                'X_test': X_test_yeo_random, 
                'type': 'Yeo+Random',
                'feature_list': list(X_train_yeo_random.columns)
            }
        ]
        
        # Train and evaluate all models
        for config in model_configs:
            # Clean feature names for LightGBM (remove special characters)
            X_train_clean = config['X_train'].copy()
            X_test_clean = config['X_test'].copy()
            
            # Replace problematic characters in column names
            column_mapping = {}
            for col in X_train_clean.columns:
                # Replace special characters with underscores
                clean_col = col.replace(' ', '_').replace('(', '_').replace(')', '_').replace(',', '_')
                clean_col = clean_col.replace('[', '_').replace(']', '_').replace('{', '_').replace('}', '_')
                clean_col = clean_col.replace('"', '_').replace("'", '_').replace('\\', '_').replace('/', '_')
                clean_col = clean_col.replace(':', '_').replace(';', '_').replace('|', '_').replace('&', '_')
                clean_col = clean_col.replace('?', '_').replace('!', '_').replace('@', '_').replace('#', '_')
                clean_col = clean_col.replace('$', '_').replace('%', '_').replace('^', '_').replace('*', '_')
                clean_col = clean_col.replace('+', '_').replace('=', '_').replace('<', '_').replace('>', '_')
                clean_col = clean_col.replace('~', '_').replace('`', '_').replace('-', '_').replace('.', '_')
                # Remove multiple consecutive underscores
                while '__' in clean_col:
                    clean_col = clean_col.replace('__', '_')
                # Remove leading/trailing underscores
                clean_col = clean_col.strip('_')
                
                column_mapping[col] = clean_col
            
            # Rename columns
            X_train_clean.columns = [column_mapping[col] for col in X_train_clean.columns]
            X_test_clean.columns = [column_mapping[col] for col in X_test_clean.columns]
            
            # Train LightGBM model with basic parameters (no Optuna optimization)
            model_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 100,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': random_state
            }
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train_clean, label=y_train)
            
            # Train model
            model = lgb.train(
                model_params,
                train_data,
                num_boost_round=100,
                callbacks=[lgb.log_evaluation(0)]  # Suppress training output
            )
            
            # Predict
            y_pred = model.predict(X_test_clean)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            print(f"  {config['type']:12} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
            
            # Store results
            sector_model_results[config['name']] = {
                'model_name': config['name'],
                'RMSE': rmse,
                'MAE': mae,
                'MSE': mse,
                'R2': r2,
                'n_companies': len(X_sector),
                'n_features_used': len(config['feature_list']),
                'model_type': 'Sector LightGBM',
                'sector': sector_name,
                'type': config['type'],
                'model': model,
                'y_test': y_test,
                'y_pred': y_pred,
                'feature_list': config['feature_list']
            }
    
    print(f"\n{'='*65}")
    print(f"Sector LightGBM Training Complete: {len(sector_model_results)} models trained")
    print(f"{'='*65}")
    
    # Save models and metrics
    model_file = settings.MODEL_DIR / "sector_lightgbm_models.pkl"
    io.save_model(sector_model_results, "sector_lightgbm_models.pkl", settings.MODEL_DIR)
    print(f"Models saved to: {model_file}")
    
    # Prepare metrics for CSV
    metrics_data = []
    for model_name, results in sector_model_results.items():
        metrics_data.append({
            'model_name': results['model_name'],
            'RMSE': results['RMSE'],
            'MAE': results['MAE'],
            'MSE': results['MSE'],
            'R2': results['R2'],
            'n_companies': results['n_companies'],
            'n_features_used': results['n_features_used'],
            'model_type': results['model_type'],
            'sector': results['sector'],
            'type': results['type']
        })
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_data)
    metrics_file = settings.METRICS_DIR / "sector_lightgbm_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metrics saved to: {metrics_file}")
    
    return sector_model_results


def evaluate_sector_lightgbm_models():
    """Evaluate sector-specific LightGBM models and save metrics."""
    print("Evaluating sector-specific LightGBM models...")
    
    try:
        # Load models
        sector_models = io.load_model("sector_lightgbm_models.pkl", settings.MODEL_DIR)
        if not sector_models:
            print("No sector LightGBM models found. Please train models first.")
            return None
        
        print(f"Loaded {len(sector_models)} sector LightGBM models for evaluation")
        
        # Models are already evaluated during training, but we can add additional analysis here
        # For now, return the loaded models
        return sector_models
        
    except Exception as e:
        print(f"Error evaluating sector LightGBM models: {e}")
        return None


def analyze_sector_lightgbm_importance():
    """Analyze feature importance for sector-specific LightGBM models."""
    print("Analyzing feature importance for sector-specific LightGBM models...")
    
    try:
        # Load models
        sector_models = io.load_model("sector_lightgbm_models.pkl", settings.MODEL_DIR)
        if not sector_models:
            print("No sector LightGBM models found. Please train models first.")
            return None
        
        importance_results = {}
        
        for model_name, model_data in sector_models.items():
            model = model_data['model']
            feature_list = model_data['feature_list']
            
            # Get feature importance from LightGBM model
            importance_scores = model.feature_importance(importance_type='gain')
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_list,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
            
            # Save individual model importance
            sector = model_data['sector']
            model_type = model_data['type']
            filename = f"sector_lightgbm_{sector}_{model_type}_importance.csv"
            filepath = settings.FEATURE_IMPORTANCE_DIR / filename
            importance_df.to_csv(filepath, index=False)
            
            importance_results[model_name] = importance_df
            
            print(f"Feature importance saved for {model_name}")
        
        print(f"Feature importance analysis complete for {len(importance_results)} models")
        return importance_results
        
    except Exception as e:
        print(f"Error analyzing sector LightGBM feature importance: {e}")
        return None


if __name__ == "__main__":
    # Run sector LightGBM models
    sector_results = run_sector_lightgbm_models()
    
    # Evaluate models
    evaluation_results = evaluate_sector_lightgbm_models()
    
    # Analyze feature importance
    importance_results = analyze_sector_lightgbm_importance()
    
    print("\nSector LightGBM modeling complete!")