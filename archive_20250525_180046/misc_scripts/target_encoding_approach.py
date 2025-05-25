#!/usr/bin/env python3
"""
Target encoding approach for better categorical feature utilization.
This is especially useful for high-cardinality categorical features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
from config import settings

class ImprovedCategoricalEncoder:
    """
    Smart categorical encoder that chooses the best encoding strategy per feature.
    """
    
    def __init__(self, target_encoding_threshold=10, cv_folds=5):
        """
        Args:
            target_encoding_threshold: If cardinality > threshold, use target encoding
            cv_folds: Cross-validation folds for target encoding
        """
        self.target_encoding_threshold = target_encoding_threshold
        self.cv_folds = cv_folds
        self.encoders = {}
        self.encoding_strategies = {}
        
    def fit(self, X, y, categorical_features):
        """
        Fit encoders for categorical features.
        
        Args:
            X: Feature matrix
            y: Target variable
            categorical_features: List of categorical feature names
        """
        for feature in categorical_features:
            if feature not in X.columns:
                continue
                
            cardinality = X[feature].nunique()
            
            if cardinality <= 2:
                # Binary features - use label encoding
                self.encoding_strategies[feature] = 'label'
                encoder = LabelEncoder()
                encoder.fit(X[feature].astype(str))
                self.encoders[feature] = encoder
                
            elif cardinality <= self.target_encoding_threshold:
                # Low cardinality - use one-hot encoding
                self.encoding_strategies[feature] = 'onehot'
                # For one-hot, we'll use pandas get_dummies
                
            else:
                # High cardinality - use target encoding
                self.encoding_strategies[feature] = 'target'
                encoder = TargetEncoder(cv=self.cv_folds, smoothing=1.0)
                encoder.fit(X[[feature]], y)
                self.encoders[feature] = encoder
                
        print("Encoding strategies chosen:")
        for feature, strategy in self.encoding_strategies.items():
            cardinality = X[feature].nunique() if feature in X.columns else 0
            print(f"  {feature}: {strategy} (cardinality: {cardinality})")
    
    def transform(self, X):
        """
        Transform categorical features using fitted encoders.
        """
        X_encoded = X.copy()
        
        for feature, strategy in self.encoding_strategies.items():
            if feature not in X.columns:
                continue
                
            if strategy == 'label':
                # Label encoding
                encoder = self.encoders[feature]
                # Handle unseen categories
                X_feature = X[feature].astype(str)
                known_categories = set(encoder.classes_)
                X_feature = X_feature.apply(lambda x: x if x in known_categories else 'unknown')
                
                # Add unknown category to encoder if needed
                if 'unknown' not in encoder.classes_:
                    encoder.classes_ = np.append(encoder.classes_, 'unknown')
                
                X_encoded[feature] = encoder.transform(X_feature)
                
            elif strategy == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(X[feature], prefix=f"{feature}_", dummy_na=True)
                X_encoded = pd.concat([X_encoded.drop(feature, axis=1), dummies], axis=1)
                
            elif strategy == 'target':
                # Target encoding
                encoder = self.encoders[feature]
                X_encoded[feature] = encoder.transform(X[[feature]])[feature]
        
        return X_encoded

def demonstrate_improved_encoding():
    """
    Demonstrate improved categorical encoding on your data.
    """
    # Load data
    df = pd.read_csv(settings.RAW_DATA_DIR / 'combined_df_for_ml_models.csv')
    
    # For demonstration, let's create a dummy target
    # In reality, you'd use your actual ESG score target
    np.random.seed(42)
    y = np.random.randn(len(df))
    
    # Identify categorical features from one-hot encoded columns
    categorical_patterns = {
        'gics_sector': 'gics_sector_',
        'issuer_country': 'issuer_cntry_domicile_name_',
        'risk_country': 'cntry_of_risk_',
        'currency': 'crncy_',
        'issuer_rating': 'issuer_rating_'
    }
    
    # Reconstruct categorical features
    for cat_name, pattern in categorical_patterns.items():
        onehot_cols = [col for col in df.columns if col.startswith(pattern)]
        if onehot_cols:
            # Reconstruct original categorical
            cat_values = []
            for idx in range(len(df)):
                active = [col for col in onehot_cols if df.iloc[idx][col] == 1]
                if active:
                    # Extract category name
                    category = active[0].replace(pattern, '')
                    cat_values.append(category)
                else:
                    cat_values.append('Unknown')
            
            df[cat_name] = cat_values
            # Remove original one-hot columns
            df = df.drop(columns=onehot_cols)
    
    print(f"Data shape after reconstruction: {df.shape}")
    
    # Get categorical features
    categorical_features = list(categorical_patterns.keys())
    print(f"Categorical features: {categorical_features}")
    
    # Apply improved encoding
    encoder = ImprovedCategoricalEncoder(target_encoding_threshold=5)
    encoder.fit(df, y, categorical_features)
    
    df_encoded = encoder.transform(df)
    
    print(f"Encoded data shape: {df_encoded.shape}")
    print(f"Original features: {len(df.columns)}")
    print(f"Encoded features: {len(df_encoded.columns)}")
    
    return df_encoded, encoder

if __name__ == "__main__":
    encoded_data, encoder = demonstrate_improved_encoding()