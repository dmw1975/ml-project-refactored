#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Debug script to test SHAP generation for CatBoost and XGBoost models."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import shap
import traceback
from pathlib import Path
from src.visualization.utils.io import load_all_models
from src.visualization.plots.shap_plots import create_shap_visualizations

def test_shap_for_model(model_name, model_data):
    """Test SHAP generation for a specific model."""
    print(f"\n{'='*60}")
    print(f"Testing SHAP for: {model_name}")
    print(f"{'='*60}")
    
    # Check model data
    print(f"Model type: {model_data.get('model_type', 'Unknown')}")
    print(f"Model object type: {type(model_data.get('model'))}")
    print(f"Has X_test: {'X_test' in model_data}")
    print(f"Has y_test: {'y_test' in model_data}")
    
    if 'X_test' in model_data:
        X_test = model_data['X_test']
        print(f"X_test shape: {X_test.shape if hasattr(X_test, 'shape') else 'N/A'}")
        print(f"X_test type: {type(X_test)}")
        if isinstance(X_test, pd.DataFrame):
            print(f"X_test columns: {list(X_test.columns)[:5]}... ({len(X_test.columns)} total)")
    
    # Try to create SHAP explainer
    try:
        model = model_data.get('model')
        if model is None:
            print("ERROR: Model object is None")
            return
            
        X_test = model_data.get('X_test')
        if X_test is None:
            print("ERROR: X_test is None")
            return
            
        # Sample data
        n_samples = min(10, len(X_test))
        sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
        X_sample = X_test.iloc[sample_indices] if isinstance(X_test, pd.DataFrame) else X_test[sample_indices]
        
        print(f"\nCreating SHAP explainer...")
        explainer = shap.TreeExplainer(model)
        print(f"✓ Explainer created successfully")
        
        print(f"\nCalculating SHAP values...")
        shap_values = explainer.shap_values(X_sample)
        print(f"✓ SHAP values calculated")
        print(f"SHAP values shape: {shap_values.shape if hasattr(shap_values, 'shape') else type(shap_values)}")
        
        # Try to create one visualization
        output_dir = Path("test_outputs") / "shap_debug" / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nTrying to create SHAP visualizations...")
        paths = create_shap_visualizations(model_data, output_dir, sample_size=10)
        print(f"✓ Created {len(paths)} visualizations")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nFull traceback:")
        traceback.print_exc()

def main():
    """Main test function."""
    print("Loading models...")
    models = load_all_models()
    print(f"Loaded {len(models)} models")
    
    # Test only CatBoost and XGBoost models
    for model_name, model_data in models.items():
        if 'catboost' in model_name.lower() or 'xgboost' in model_name.lower():
            test_shap_for_model(model_name, model_data)

if __name__ == "__main__":
    main()