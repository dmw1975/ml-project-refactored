#!/usr/bin/env python3
"""Final comprehensive fix for the pipeline to ensure all models are properly stored and metrics are collected."""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

print("🔧 Final Pipeline Fix")
print("=" * 60)

# Step 1: Clean up the code files
print("\n1. Cleaning up code files...")
os.system("python fix_categorical_model_storage.py")

# Step 2: Fix model consolidation
print("\n2. Fixing model consolidation...")
os.system("python fix_model_consolidation.py")

# Step 3: Fix model types
print("\n3. Fixing model types...")
os.system("python fix_model_types.py")

# Step 4: Re-run training for categorical models
print("\n4. Re-training categorical models with fixed code...")
print("   This will properly save models in the old format")

# Train XGBoost categorical
print("\n   Training XGBoost categorical models...")
os.system("python -c \"from models.xgboost_categorical import train_xgboost_categorical_models; train_xgboost_categorical_models()\"")

# Train LightGBM categorical
print("\n   Training LightGBM categorical models...")
os.system("python -c \"from models.lightgbm_categorical import train_lightgbm_categorical_models; train_lightgbm_categorical_models()\"")

# Train CatBoost categorical (already uses correct format)
print("\n   Training CatBoost categorical models...")
os.system("python -c \"from models.catboost_categorical import run_all_catboost_categorical; run_all_catboost_categorical()\"")

# Step 5: Regenerate metrics
print("\n5. Regenerating all metrics...")
os.system("python fix_metrics_aggregation.py")

# Step 6: Test visualization
print("\n6. Testing visualization...")
os.system("python test_visualization_fix.py")

print("\n✅ Pipeline fix completed!")
print("\nYou can now run visualizations with:")
print("  python main.py --visualize")