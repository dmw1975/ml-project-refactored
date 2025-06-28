"""Sector-specific ElasticNet models for ESG score prediction."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from src.config import settings
from src.data.data import load_features_data, load_scores_data, get_base_and_yeo_features, add_random_feature
from src.utils import io

def run_sector_elastic_net_models(feature_df=None, score_df=None, base_columns=None, yeo_columns=None, 
                                  Base_random=None, Yeo_random=None,
                                  random_state=42, test_size=0.2):
    """
    Run ElasticNet models separately for each GICS sector.
    
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
    Base_random : pandas.DataFrame, optional
        Base dataset with random feature. If None, it will be created.
    Yeo_random : pandas.DataFrame, optional
        Yeo-Johnson dataset with random feature. If None, it will be created.
    random_state : int, default=42
        Reproducibility seed
    test_size : float, default=0.2
        Proportion of data for testing
    """
    # Force reload data module to ensure latest version
    import importlib
    import src.data as data
    importlib.reload(data)
    from src.data import load_features_data, load_scores_data, get_base_and_yeo_features, add_random_feature
    
    print("Loading data for sector-specific ElasticNet models...")
    
    # Load data if not provided
    if feature_df is None:
        feature_df = load_features_data(model_type='linear')
    
    if score_df is None:
        score_df = load_scores_data()
    
    # Get feature sets if not provided
    if base_columns is None or yeo_columns is None:
        Base, Yeo, base_columns, yeo_columns = get_base_and_yeo_features(feature_df)
        
        # Direct feature count check before continuing
        print(f"\nDIRECT FEATURE COUNT CHECK (AFTER LOADING):")
        print(f"Base column count: {len(Base.columns)}")
        print(f"Yeo column count: {len(Yeo.columns)}")
        
        # If Yeo has less features, fix it directly here
        if len(Yeo.columns) < len(Base.columns):
            print(f"WARNING: Yeo has fewer columns than expected, forcing fix...")
            # Identify all Yeo-transformed columns
            yeo_prefix = 'yeo_joh_'
            yeo_transformed_columns = [col for col in feature_df.columns if col.startswith(yeo_prefix)]
            original_numerical_columns = [col.replace(yeo_prefix, '') for col in yeo_transformed_columns]
            categorical_columns = [col for col in Base.columns if col not in original_numerical_columns]
            complete_yeo_columns = yeo_transformed_columns + categorical_columns
            Yeo = feature_df[complete_yeo_columns].copy()
            print(f"Fixed Yeo column count: {len(Yeo.columns)}")
    
    # Create random feature datasets if not provided
    if Base_random is None:
        if 'Base' not in locals():
            Base, _, _, _ = get_base_and_yeo_features(feature_df)
        Base_random = add_random_feature(Base)
    
    if Yeo_random is None:
        if 'Yeo' not in locals():
            _, Yeo, _, _ = get_base_and_yeo_features(feature_df)
            
            # Check if Yeo needs fixing
            if 'Base' in locals() and len(Yeo.columns) < len(Base.columns):
                print(f"WARNING: Yeo for random features has fewer columns than expected, forcing fix...")
                yeo_prefix = 'yeo_joh_'
                yeo_transformed_columns = [col for col in feature_df.columns if col.startswith(yeo_prefix)]
                original_numerical_columns = [col.replace(yeo_prefix, '') for col in yeo_transformed_columns]
                categorical_columns = [col for col in Base.columns if col not in original_numerical_columns]
                complete_yeo_columns = yeo_transformed_columns + categorical_columns
                Yeo = feature_df[complete_yeo_columns].copy()
                print(f"Fixed Yeo column count: {len(Yeo.columns)}")
        
        Yeo_random = add_random_feature(Yeo)
    
    # Initialize dictionary to store model metrics
    sector_model_results = {}
    
    # Identify sector columns
    sector_columns = [col for col in feature_df.columns if col.startswith('gics_sector_')]
    
    # Define minimum companies needed per sector for modeling
    MIN_COMPANIES = 50
    
    print("\n" + "="*65)
    print("Running ElasticNet Models by Sector")
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
        X_sector_base_random = Base_random.loc[sector_indices]
        X_sector_yeo_random = Yeo_random.loc[sector_indices]
        
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
                'name': f"Sector_{sector_name}_Base", 
                'X_train': X_train_base, 
                'X_test': X_test_base, 
                'type': 'Base',
                'feature_list': base_columns  # Store feature list used for training
            },
            {
                'name': f"Sector_{sector_name}_Yeo", 
                'X_train': X_train_yeo, 
                'X_test': X_test_yeo, 
                'type': 'Yeo',
                'feature_list': yeo_columns  # Store feature list used for training
            },
            {
                'name': f"Sector_{sector_name}_Base_Random", 
                'X_train': X_train_base_random, 
                'X_test': X_test_base_random, 
                'type': 'Base_Random',
                'feature_list': list(X_train_base_random.columns)  # Store feature list with random feature
            },
            {
                'name': f"Sector_{sector_name}_Yeo_Random", 
                'X_train': X_train_yeo_random, 
                'X_test': X_test_yeo_random, 
                'type': 'Yeo_Random',
                'feature_list': list(X_train_yeo_random.columns)  # Store feature list with random feature
            }
        ]
        
        # Train and evaluate all models
        for config in model_configs:
            # Train ElasticNet model with cross-validation
            model = ElasticNetCV(
                cv=5,
                random_state=random_state,
                l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                alphas=np.logspace(-5, 2, 100),
                max_iter=1000
            )
            model.fit(config['X_train'], y_train)
            y_pred = model.predict(config['X_test'])
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Store results
            sector_model_results[config['name']] = {
                'model_name': config['name'],
                'RMSE': rmse,
                'MAE': mae,
                'MSE': mse,
                'R2': r2,
                'n_companies': len(X_sector),  # Total number of companies in this sector
                'n_companies_train': len(config['X_train']),  # Companies in training set
                'n_companies_test': len(config['X_test']),  # Companies in test set
                'model': model,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_train': y_train,  # Store for baseline comparisons
                'X_test': config['X_test'],  # Store the actual X_test data
                'sector': sector_name,
                'type': config['type'],
                'dataset': config['type'],  # Add dataset field for visualization
                'feature_list': config['feature_list'],  # Store feature list used for training
                'model_type': 'ElasticNet',  # For compatibility with metrics visualizations
                'alpha': model.alpha_,  # Best alpha from CV
                'l1_ratio': model.l1_ratio_,  # Best l1_ratio from CV
                'cv_mse': np.min(model.mse_path_) if hasattr(model, 'mse_path_') else None,  # Best CV MSE
                'cv_mse_std': np.std(model.mse_path_) if hasattr(model, 'mse_path_') else None  # Overall std
            }
            
            # Print results
            print(f"  {config['type']} Model - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")
            print(f"    Best alpha: {model.alpha_:.6f}, Best l1_ratio: {model.l1_ratio_:.2f}")
    
    # Print summary of all sector models
    print("\n" + "="*75)
    print("Summary of Sector-Specific ElasticNet Model Performance")
    print("="*75)
    print(f"{'Sector':<20} {'Model Type':<15} {'RMSE':<8} {'MAE':<8} {'MSE':<8} {'R²':<8} {'n':<6}")
    print("-" * 75)
    
    # Print all sector models sorted by sector then type
    for sector in sorted(set(v['sector'] for v in sector_model_results.values())):
        sector_specific_models = {k: v for k, v in sector_model_results.items() if v['sector'] == sector}
        for model_name, metrics in sorted(sector_specific_models.items(), 
                                         key=lambda x: ('Random' in x[0], x[0])):
            print(f"{metrics['sector']:<20} {metrics['type']:<15} {metrics['RMSE']:.4f}  {metrics['MAE']:.4f}  {metrics['MSE']:.4f}  {metrics['R2']:.4f}  {metrics['n_companies']:<6}")
    
    # Calculate and print averages by model type
    print("\n" + "="*75)
    print("Average Performance by Model Type")
    print("="*75)
    
    model_types = set(v['type'] for v in sector_model_results.values())
    
    for model_type in sorted(model_types):
        models = {k: v for k, v in sector_model_results.items() if v['type'] == model_type}
        if models:
            avg_rmse = np.mean([m['RMSE'] for m in models.values()])
            avg_mae = np.mean([m['MAE'] for m in models.values()])
            avg_mse = np.mean([m['MSE'] for m in models.values()])
            avg_r2 = np.mean([m['R2'] for m in models.values()])
            print(f"{model_type:<15} RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}, MSE: {avg_mse:.4f}, R²: {avg_r2:.4f}")
    
    # Save sector model results
    io.ensure_dir(settings.MODEL_DIR)
    io.save_model(sector_model_results, "sector_elasticnet_models.pkl", settings.MODEL_DIR)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([
        {
            'model_name': name,
            'sector': metrics['sector'],
            'type': metrics['type'],
            'dataset': metrics['dataset'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MSE': metrics['MSE'],
            'R2': metrics['R2'],
            'n_companies': metrics['n_companies'],
            'alpha': metrics['alpha'],
            'l1_ratio': metrics['l1_ratio']
        }
        for name, metrics in sector_model_results.items()
    ])
    
    io.ensure_dir(settings.METRICS_DIR)
    metrics_df.to_csv(f"{settings.METRICS_DIR}/sector_elasticnet_metrics.csv", index=False)
    
    print("\nSector-specific ElasticNet models trained and saved successfully.")
    return sector_model_results


if __name__ == "__main__":
    # Run this file directly to train all sector ElasticNet models
    sector_models = run_sector_elastic_net_models()