"""Debug new pipeline model implementation."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Add paths
new_path = Path(__file__).parent / "esg_ml_clean"
sys.path.insert(0, str(new_path))

# Import new pipeline components
from src.data.loader import DataLoader
from src.models.linear import LinearRegressionModel
from src.models import linear  # Trigger registration

print("=== Debugging New Pipeline Model Implementation ===\n")

# Create data loader
config = {'data_dir': 'data'}
loader = DataLoader(config)

# Get train/test split
print("1. Loading data...")
X_train, X_test, y_train, y_test = loader.get_train_test_split(
    dataset_type='base',
    test_size=0.2,
    random_state=42
)

print(f"   - X_train shape: {X_train.shape}")
print(f"   - X_test shape: {X_test.shape}")

# Create model with new pipeline wrapper
print("\n2. Creating LinearRegressionModel...")
model = LinearRegressionModel(name="linear_regression", normalize=False)

print(f"   - Model name: {model.name}")
print(f"   - Normalize: {model.normalize}")

# Fit model
print("\n3. Training model...")
model.fit(X_train, y_train)

# Make predictions
print("\n4. Making predictions...")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_r2 = r2_score(y_train, y_pred_train)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)

print(f"\n5. Model performance:")
print(f"   - Train RMSE: {train_rmse:.4f}")
print(f"   - Train R2: {train_r2:.4f}")
print(f"   - Test RMSE: {test_rmse:.4f}")
print(f"   - Test R2: {test_r2:.4f}")

# Check underlying sklearn model
print(f"\n6. Checking underlying sklearn model:")
print(f"   - Model type: {type(model.model)}")
print(f"   - Number of coefficients: {len(model.model.coef_)}")
print(f"   - Intercept: {model.model.intercept_:.4f}")

# Test with normalization
print("\n\n=== Testing with Normalization ===")
model_norm = LinearRegressionModel(name="linear_regression", normalize=True)
model_norm.fit(X_train, y_train)

y_pred_test_norm = model_norm.predict(X_test)
test_rmse_norm = np.sqrt(mean_squared_error(y_test, y_pred_test_norm))
test_r2_norm = r2_score(y_test, y_pred_test_norm)

print(f"\nNormalized model performance:")
print(f"   - Test RMSE: {test_rmse_norm:.4f}")
print(f"   - Test R2: {test_r2_norm:.4f}")

print("\n=== Analysis Complete ===")