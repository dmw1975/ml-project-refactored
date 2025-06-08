#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate missing SHAP visualizations for CatBoost and XGBoost models."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import pickle
from src.visualization.plots.shap_plots import create_shap_visualizations
from src.config.settings import VISUALIZATION_DIR

def generate_shap_for_models():
    """Generate SHAP visualizations for CatBoost and XGBoost models."""
    models_dir = Path("outputs/models")
    shap_dir = VISUALIZATION_DIR / "SHAP"
    
    # Process CatBoost models
    catboost_file = models_dir / "catboost_models.pkl"
    if catboost_file.exists():
        print("Processing CatBoost models...")
        print("="*60)
        
        with open(catboost_file, 'rb') as f:
            catboost_models = pickle.load(f)
        
        for model_name, model_data in catboost_models.items():
            print(f"\nGenerating SHAP for {model_name}...")
            try:
                # Ensure model_name is in the data
                model_data['model_name'] = model_name
                model_data['model_type'] = 'catboost'
                
                # Create SHAP visualizations
                paths = create_shap_visualizations(model_data, shap_dir, sample_size=100)
                print(f"✓ Created {len(paths)} SHAP visualizations for {model_name}")
                
            except Exception as e:
                print(f"❌ Error creating SHAP for {model_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Process XGBoost models
    xgboost_file = models_dir / "xgboost_models.pkl"
    if xgboost_file.exists():
        print("\n\nProcessing XGBoost models...")
        print("="*60)
        
        with open(xgboost_file, 'rb') as f:
            xgboost_models = pickle.load(f)
        
        for model_name, model_data in xgboost_models.items():
            print(f"\nGenerating SHAP for {model_name}...")
            try:
                # Ensure model_name is in the data
                model_data['model_name'] = model_name
                model_data['model_type'] = 'xgboost'
                
                # Create SHAP visualizations
                paths = create_shap_visualizations(model_data, shap_dir, sample_size=100)
                print(f"✓ Created {len(paths)} SHAP visualizations for {model_name}")
                
            except Exception as e:
                print(f"❌ Error creating SHAP for {model_name}: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n\nDone! Check the outputs/visualizations/SHAP/ directory for the generated plots.")

if __name__ == "__main__":
    generate_shap_for_models()