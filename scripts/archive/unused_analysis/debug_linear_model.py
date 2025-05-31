"""Debug linear model differences between pipelines."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Add path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

# Import from old pipeline
from data import load_features_data, get_base_and_yeo_features, load_scores_data

print("=== Debugging Linear Model Differences ===\n")

# Load data using old pipeline
print("1. Loading data...")
feature_df = load_features_data()
base_features, _, _, _ = get_base_and_yeo_features(feature_df)
scores = load_scores_data()

# Align data
common_index = base_features.index.intersection(scores.index)
X = base_features.loc[common_index]
y = scores.loc[common_index]

print(f"   - Features shape: {X.shape}")
print(f"   - Target shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   - Train shape: {X_train.shape}")
print(f"   - Test shape: {X_test.shape}")

# Check data statistics
print(f"\n2. Data statistics:")
print(f"   - X_train mean: {X_train.mean().mean():.4f}")
print(f"   - X_train std: {X_train.std().mean():.4f}")
print(f"   - y_train mean: {y_train.mean():.4f}")
print(f"   - y_train std: {y_train.std():.4f}")

# Train model
print(f"\n3. Training LinearRegression...")
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_r2 = r2_score(y_train, y_pred_train)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)

print(f"\n4. Model performance:")
print(f"   - Train RMSE: {train_rmse:.4f}")
print(f"   - Train R2: {train_r2:.4f}")
print(f"   - Test RMSE: {test_rmse:.4f}")
print(f"   - Test R2: {test_r2:.4f}")

# Check coefficient statistics
print(f"\n5. Model coefficients:")
print(f"   - Number of coefficients: {len(model.coef_)}")
print(f"   - Mean coefficient: {model.coef_.mean():.6f}")
print(f"   - Std coefficient: {model.coef_.std():.6f}")
print(f"   - Min coefficient: {model.coef_.min():.6f}")
print(f"   - Max coefficient: {model.coef_.max():.6f}")
print(f"   - Intercept: {model.intercept_:.4f}")

# Now test with new pipeline's data loader
print("\n\n=== Testing New Pipeline's Data Loader ===")

# Add new pipeline to path
new_path = Path(__file__).parent / "esg_ml_clean"
sys.path.insert(0, str(new_path))

from src.data.loader import DataLoader

# Create data loader
config = {'data_dir': 'data'}
loader = DataLoader(config)

# Load data
print("\n1. Loading data from new pipeline...")
X_new = loader.load_features('base')
y_new = loader.load_targets()

# Align
common_idx_new = X_new.index.intersection(y_new.index)
X_new = X_new.loc[common_idx_new]
y_new = y_new.loc[common_idx_new]

print(f"   - Features shape: {X_new.shape}")
print(f"   - Target shape: {y_new.shape}")

# Check if data matches
print(f"\n2. Data comparison:")
print(f"   - Features match: {X.equals(X_new)}")
print(f"   - Targets match: {y.equals(y_new)}")

if not X.equals(X_new):
    # Find differences
    print(f"   - Index match: {all(X.index == X_new.index)}")
    print(f"   - Columns match: {all(X.columns == X_new.columns)}")
    
    # Check numerical differences
    diff = (X - X_new).abs()
    print(f"   - Max difference: {diff.max().max()}")
    print(f"   - Mean difference: {diff.mean().mean()}")

# Split with same seed
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X_new, y_new, test_size=0.2, random_state=42
)

# Check if splits match
print(f"\n3. Split comparison:")
print(f"   - X_train match: {X_train.equals(X_train_new)}")
print(f"   - X_test match: {X_test.equals(X_test_new)}")
print(f"   - y_train match: {y_train.equals(y_train_new)}")
print(f"   - y_test match: {y_test.equals(y_test_new)}")

# Train model on new data
print(f"\n4. Training on new pipeline data...")
model_new = LinearRegression()
model_new.fit(X_train_new, y_train_new)

y_pred_test_new = model_new.predict(X_test_new)
test_rmse_new = np.sqrt(mean_squared_error(y_test_new, y_pred_test_new))
test_r2_new = r2_score(y_test_new, y_pred_test_new)

print(f"   - Test RMSE: {test_rmse_new:.4f}")
print(f"   - Test R2: {test_r2_new:.4f}")

# Compare coefficients
print(f"\n5. Coefficient comparison:")
print(f"   - Coefficients match: {np.allclose(model.coef_, model_new.coef_)}")
print(f"   - Intercepts match: {np.isclose(model.intercept_, model_new.intercept_)}")
print(f"   - Max coef difference: {np.max(np.abs(model.coef_ - model_new.coef_))}")

print("\n=== Analysis Complete ===")