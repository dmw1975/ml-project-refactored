#!/usr/bin/env python3
"""
Safe target encoding implementation for ElasticNet models.
Addresses potential data leakage and scaling issues.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from category_encoders import TargetEncoder
import warnings

class SafeTargetEncoder:
    """
    Target encoder that prevents data leakage for linear models.
    """
    
    def __init__(self, cv_folds=5, smoothing=1.0, min_samples_leaf=1):
        self.cv_folds = cv_folds
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.encoders = {}
        self.global_means = {}
        
    def fit_transform(self, X, y, categorical_cols):
        """
        Fit and transform using proper cross-validation to prevent leakage.
        """
        X_encoded = X.copy()
        
        # Create CV folds
        kf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        # For stratification, we need sector information
        sector_cols = [col for col in X.columns if col.startswith('gics_sector_')]
        if sector_cols:
            sector_labels = np.zeros(len(X))
            for i, col in enumerate(sector_cols):
                sector_labels[X[col] == 1] = i
        else:
            # Fallback to simple stratification
            sector_labels = pd.qcut(y, q=5, labels=False, duplicates='drop')
        
        for cat_col in categorical_cols:
            if cat_col not in X.columns:
                continue
                
            # Store global mean for this feature
            self.global_means[cat_col] = y.mean()
            
            # Initialize encoded column
            X_encoded[cat_col] = np.nan
            
            # Cross-validation encoding to prevent leakage
            for train_idx, val_idx in kf.split(X, sector_labels):
                # Fit encoder on training fold
                encoder = TargetEncoder(
                    smoothing=self.smoothing,
                    min_samples_leaf=self.min_samples_leaf
                )
                
                X_train_fold = X.iloc[train_idx][[cat_col]]
                y_train_fold = y.iloc[train_idx]
                
                encoder.fit(X_train_fold, y_train_fold)
                
                # Transform validation fold
                X_val_fold = X.iloc[val_idx][[cat_col]]
                encoded_vals = encoder.transform(X_val_fold)[cat_col]
                
                X_encoded.iloc[val_idx, X_encoded.columns.get_loc(cat_col)] = encoded_vals
                
            # Store the final encoder (fitted on all data) for transform()
            final_encoder = TargetEncoder(
                smoothing=self.smoothing,
                min_samples_leaf=self.min_samples_leaf
            )
            final_encoder.fit(X[[cat_col]], y)
            self.encoders[cat_col] = final_encoder
            
        return X_encoded
    
    def transform(self, X):
        """
        Transform new data using fitted encoders.
        """
        X_encoded = X.copy()
        
        for cat_col, encoder in self.encoders.items():
            if cat_col in X.columns:
                try:
                    X_encoded[cat_col] = encoder.transform(X[[cat_col]])[cat_col]
                except:
                    # Fallback to global mean for unseen categories
                    X_encoded[cat_col] = self.global_means[cat_col]
                    
        return X_encoded

def safe_elasticnet_with_target_encoding():
    """
    Demonstrate safe ElasticNet with target encoding.
    """
    from config import settings
    
    # Load data
    df = pd.read_csv(settings.RAW_DATA_DIR / 'combined_df_for_ml_models.csv')
    
    # Create dummy target for demonstration
    np.random.seed(42)
    y = pd.Series(np.random.randn(len(df)))
    
    print("=== SAFE ELASTICNET WITH TARGET ENCODING ===")
    print(f"Original data shape: {df.shape}")
    
    # 1. Identify categorical features to reconstruct
    categorical_patterns = {
        'gics_sector': 'gics_sector_',
        'issuer_country': 'issuer_cntry_domicile_name_',
        'risk_country': 'cntry_of_risk_',
        'currency': 'crncy_',
    }
    
    # 2. Reconstruct categorical features
    cat_features = []
    for cat_name, pattern in categorical_patterns.items():
        onehot_cols = [col for col in df.columns if col.startswith(pattern)]
        if onehot_cols:
            # Reconstruct categorical
            cat_values = []
            for idx in range(len(df)):
                active = [col for col in onehot_cols if df.iloc[idx][col] == 1]
                if active:
                    category = active[0].replace(pattern, '')
                    cat_values.append(category)
                else:
                    cat_values.append('Unknown')
            
            df[cat_name] = cat_values
            cat_features.append(cat_name)
            
            print(f"Reconstructed {cat_name}: {df[cat_name].nunique()} categories")
            
            # Remove one-hot columns
            df = df.drop(columns=onehot_cols)
    
    print(f"Data shape after reconstruction: {df.shape}")
    print(f"Categorical features: {cat_features}")
    
    # 3. Split data first (important for preventing leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42,
        stratify=df['gics_sector'] if 'gics_sector' in cat_features else None
    )
    
    # 4. Apply safe target encoding on training data only
    encoder = SafeTargetEncoder(cv_folds=5, smoothing=1.0)
    X_train_encoded = encoder.fit_transform(X_train, y_train, cat_features)
    X_test_encoded = encoder.transform(X_test)
    
    # 5. Scale features (important for ElasticNet with mixed feature types)
    scaler = StandardScaler()
    
    # Remove non-numeric columns for scaling
    numeric_cols = X_train_encoded.select_dtypes(include=[np.number]).columns
    
    X_train_scaled = X_train_encoded.copy()
    X_test_scaled = X_test_encoded.copy()
    
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train_encoded[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test_encoded[numeric_cols])
    
    print(f"Final feature count: {len(numeric_cols)}")
    print(f"Features reduced from ~245 to {len(numeric_cols)}")
    
    # 6. Train ElasticNet
    model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=5000)
    model.fit(X_train_scaled[numeric_cols], y_train)
    
    # 7. Evaluate
    y_pred = model.predict(X_test_scaled[numeric_cols])
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    n_features_used = np.sum(model.coef_ != 0)
    
    print(f"\n=== RESULTS ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Features used: {n_features_used}/{len(numeric_cols)}")
    
    # 8. Show feature importance
    feature_importance = pd.DataFrame({
        'feature': numeric_cols,
        'coefficient': model.coef_,
        'abs_coefficient': np.abs(model.coef_)
    }).sort_values('abs_coefficient', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10)[['feature', 'coefficient']])
    
    return model, feature_importance, encoder, scaler

if __name__ == "__main__":
    model, importance, encoder, scaler = safe_elasticnet_with_target_encoding()