"""Sector-specific linear regression models for ESG score prediction."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from src.config import settings
from src.data.data import load_features_data, load_scores_data, get_base_and_yeo_features, add_random_feature
from src.utils import io

def run_sector_models(feature_df=None, score_df=None, base_columns=None, yeo_columns=None, 
                      LR_Base_random=None, LR_Yeo_random=None,
                      random_state=42, test_size=0.2):
    """
    Run linear regression models separately for each GICS sector.
    
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
    import src.data as data
    importlib.reload(data)
    from src.data import load_features_data, load_scores_data, get_base_and_yeo_features, add_random_feature
    
    print("Loading data for sector-specific models...")
    
    # Load data if not provided
    if feature_df is None:
        feature_df = load_features_data(model_type='linear')
    
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
    print("Running Linear Regression Models by Sector")
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
                'type': 'Base+Random',
                'feature_list': list(X_train_base_random.columns)  # Store feature list with random feature
            },
            {
                'name': f"Sector_{sector_name}_Yeo_Random", 
                'X_train': X_train_yeo_random, 
                'X_test': X_test_yeo_random, 
                'type': 'Yeo+Random',
                'feature_list': list(X_train_yeo_random.columns)  # Store feature list with random feature
            }
        ]
        
        # Train and evaluate all models
        for config in model_configs:
            # Train model
            model = LinearRegression()
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
                'X_test': config['X_test'],  # Store the actual X_test data
                'sector': sector_name,
                'type': config['type'],
                'feature_list': config['feature_list'],  # Store feature list used for training
                'model_type': 'Sector Linear Regression'  # For compatibility with metrics visualizations
            }
            
            # Print results
            print(f"  {config['type']} Model - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")
    
    # Print summary of all sector models
    print("\n" + "="*75)
    print("Summary of Sector-Specific Model Performance")
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
    io.save_model(sector_model_results, "sector_models.pkl", settings.MODEL_DIR)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([
        {
            'model_name': name,
            'sector': metrics['sector'],
            'type': metrics['type'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MSE': metrics['MSE'],
            'R2': metrics['R2'],
            'n_companies': metrics['n_companies']
        }
        for name, metrics in sector_model_results.items()
    ])
    
    io.ensure_dir(settings.METRICS_DIR)
    metrics_df.to_csv(f"{settings.METRICS_DIR}/sector_model_metrics.csv", index=False)
    
    print("\nSector-specific models trained and saved successfully.")
    return sector_model_results

def evaluate_sector_models(sector_models=None):
    """
    Run evaluation on sector models, calculating residuals and creating comparison tables.
    
    Parameters:
    -----------
    sector_models : dict, optional
        Dictionary of sector model results. If None, they will be loaded.
    
    Returns:
    --------
    dict
        Dictionary with evaluation results
    """
    from evaluation.metrics import calculate_residuals
    
    print("Evaluating sector-specific models...")
    
    # Load sector models if not provided
    if sector_models is None:
        try:
            sector_models = io.load_model("sector_models.pkl", settings.MODEL_DIR)
            print(f"Loaded {len(sector_models)} sector-specific models")
        except:
            print("No sector models found. Please train sector models first.")
            return None
    
    # Calculate residuals
    print("\nCalculating model residuals...")
    residuals = calculate_residuals(sector_models)
    
    # Removed sector_model_residuals.pkl generation - Analysis showed this file is NEVER READ
    # Residuals are calculated on-the-fly from y_test/y_pred in model PKL files - Date: 2025-01-15
    # io.save_model(residuals, "sector_model_residuals.pkl", settings.METRICS_DIR)
    
    # Create custom comparison table for sector models
    print("\nCreating model comparison table...")
    model_metrics = []
    for model_name, model_data in sector_models.items():
        # Ensure sector information is captured
        metrics = {
            'model_name': model_name,
            'RMSE': model_data.get('RMSE', np.sqrt(model_data.get('MSE', 0))),
            'MAE': model_data.get('MAE', 0),
            'MSE': model_data.get('MSE', 0),
            'R2': model_data.get('R2', 0),
            'n_companies': model_data.get('n_companies', 0),
            'n_features_used': model_data.get('n_features_used', None),
            'alpha': model_data.get('alpha', None),
            'l1_ratio': model_data.get('l1_ratio', None),
            'model_type': 'Sector Linear Regression',
            # Make sure to include these sector-specific columns
            'sector': model_data.get('sector', 'Unknown'),
            'type': model_data.get('type', 'Unknown')
        }
        model_metrics.append(metrics)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(model_metrics)
    
    # Save to CSV - use a consistent name
    io.ensure_dir(settings.METRICS_DIR)
    metrics_df.to_csv(f"{settings.METRICS_DIR}/sector_model_metrics.csv", index=False)
    
    # Print summary
    print("\nSector Model Comparison Summary:")
    print("================================")
    print(metrics_df[['model_name', 'sector', 'type', 'RMSE', 'R2']].sort_values('RMSE'))
    
    print("\nSector model evaluation complete. Results saved to metrics directory.")
    return {
        'sector_models': sector_models,
        'residuals': residuals,
        'metrics_df': metrics_df
    }

def analyze_sector_importance(sector_models=None, n_repeats=10):
    """
    Analyze feature importance for sector-specific models.
    
    Parameters:
    -----------
    sector_models : dict, optional
        Dictionary of sector model results. If None, they will be loaded.
    n_repeats : int, default=10
        Number of permutation repeats for importance calculation
    
    Returns:
    --------
    tuple
        (importance_results, consolidated)
    """
    from evaluation.importance import calculate_permutation_importance, create_consolidated_importance_table
    
    print("Analyzing feature importance for sector-specific models...")
    
    # Load sector models if not provided
    if sector_models is None:
        try:
            sector_models = io.load_model("sector_models.pkl", settings.MODEL_DIR)
            print(f"Loaded {len(sector_models)} sector-specific models")
        except:
            print("No sector models found. Please train sector models first.")
            return None, None
    
    # Calculate importance for each model
    importance_results = {}
    random_feature_stats = []
    
    for model_name, model_data in sector_models.items():
        print(f"Calculating feature importance for {model_name}...")
        
        # Get the model
        model = model_data['model']
        
        # Get the test data DIRECTLY from the stored model data
        # This ensures we use exactly the same data that was used for evaluation
        if 'X_test' in model_data:
            X_test = model_data['X_test']
            y_test = model_data['y_test']
            
            print(f"  Using stored X_test data with {X_test.shape[1]} features")
        else:
            print(f"  X_test not found in model data, attempting to reconstruct...")
            
            # This is the old approach - try to reconstruct the test data
            model_type = model_data['type']
            sector = model_data['sector']
            y_test = model_data['y_test']
            
            # Load full dataset to get feature columns
            feature_df = load_features_data(model_type='linear')
            
            # Get sector mask
            sector_col = f'gics_sector_{sector}'
            if sector_col in feature_df.columns:
                sector_mask = feature_df[sector_col] == 1
                X_sector = feature_df[sector_mask]
                
                # Get the correct feature set based on model type
                LR_Base, LR_Yeo, base_columns, yeo_columns = get_base_and_yeo_features(feature_df)
                
                if 'Base' in model_type and 'Random' not in model_type:
                    X_data = X_sector[base_columns]
                elif 'Yeo' in model_type and 'Random' not in model_type:
                    X_data = X_sector[yeo_columns]
                elif 'Base' in model_type and 'Random' in model_type:
                    X_base = X_sector[base_columns]
                    X_data = add_random_feature(X_base)
                elif 'Yeo' in model_type and 'Random' in model_type:
                    X_yeo = X_sector[yeo_columns]
                    X_data = add_random_feature(X_yeo)
                
                # We need to align X_test with y_test
                X_test = X_data.loc[y_test.index]
                print(f"  Reconstructed X_test with {X_test.shape[1]} features")
        
        if X_test is None or X_test.empty:
            print(f"  Could not get test data for {model_name}, skipping...")
            continue
            
        # Check if the model's feature set matches the test data
        if hasattr(model, 'feature_names_in_') and set(model.feature_names_in_) != set(X_test.columns):
            print(f"  WARNING: Feature mismatch between model and test data for {model_name}")
            print(f"  Model features: {len(model.feature_names_in_)}, Test data features: {len(X_test.columns)}")
            
            # Identify missing features
            model_features = set(model.feature_names_in_)
            test_features = set(X_test.columns)
            
            missing_in_test = model_features - test_features
            extra_in_test = test_features - model_features
            
            if missing_in_test:
                print(f"  Features in model but missing in test data: {len(missing_in_test)}")
                if len(missing_in_test) < 10:
                    print(f"  Missing: {missing_in_test}")
            
            if extra_in_test:
                print(f"  Features in test data but not in model: {len(extra_in_test)}")
                if len(extra_in_test) < 10:
                    print(f"  Extra: {extra_in_test}")
            
            # Fix the test data to match the model
            print(f"  Aligning test data columns with model features...")
            
            # Create a new DataFrame with the model's features
            aligned_X_test = pd.DataFrame(index=X_test.index)
            
            # Add features from X_test that are in the model
            for feature in model.feature_names_in_:
                if feature in X_test.columns:
                    aligned_X_test[feature] = X_test[feature]
                else:
                    # Add a column of zeros for missing features
                    print(f"  Adding zero column for missing feature: {feature}")
                    aligned_X_test[feature] = 0
            
            # Use the aligned data
            X_test = aligned_X_test
            print(f"  Aligned X_test now has {X_test.shape[1]} features matching the model")
            
        # Calculate importance
        try:
            importance_df = calculate_permutation_importance(
                model, X_test, y_test, 
                n_repeats=n_repeats, 
                random_state=settings.LINEAR_REGRESSION_PARAMS['random_state']
            )
            
            # Save to results
            importance_results[model_name] = importance_df
            
            # Save to CSV
            output_dir = settings.FEATURE_IMPORTANCE_DIR
            io.ensure_dir(output_dir)
            importance_df.to_csv(f"{output_dir}/sector_{model_name}_importance.csv", index=False)
            
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
            print(f"  ERROR calculating feature importance for {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Check if we have any importance results
    if not importance_results:
        print("No feature importance results were calculated. Check the errors above.")
        return None, None
    
    # Save all importance results
    output_dir = settings.FEATURE_IMPORTANCE_DIR
    io.ensure_dir(output_dir)
    io.save_model(importance_results, "sector_feature_importance.pkl", output_dir)
    
    # Create random feature stats DataFrame
    if random_feature_stats:
        random_df = pd.DataFrame(random_feature_stats)
        random_df.to_csv(f"{output_dir}/sector_random_feature_stats.csv", index=False)
        
        print("\nRandom Feature Performance in Sector Models:")
        print("============================================")
        for _, row in random_df.iterrows():
            print(f"{row['model_name']}: Rank {row['random_rank']}/{row['total_features']} ({row['percentile']:.1f}%), Importance: {row['random_importance']:.6f}")
    
    # Create consolidated importance table
    try:
        consolidated = create_consolidated_importance_table(importance_results)
        
        # Save with sector prefix
        consolidated.to_csv(f"{output_dir}/sector_consolidated_importance.csv")
    except Exception as e:
        print(f"Error creating consolidated importance table: {e}")
        consolidated = None
    
    print("\nSector model feature importance analysis complete. Results saved to feature_importance directory.")
    return importance_results, consolidated

if __name__ == "__main__":
    # Run this file directly to train all sector models
    sector_models = run_sector_models()
    evaluate_sector_models(sector_models)
    analyze_sector_importance(sector_models)