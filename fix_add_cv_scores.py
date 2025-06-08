#!/usr/bin/env python3
"""Add CV scores to LightGBM and CatBoost models that are missing them."""

import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer

def add_cv_scores_to_models():
    """Add CV scores to models that are missing them."""
    
    # RMSE scorer
    rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), 
                              greater_is_better=False)
    
    for model_file in ["lightgbm_models.pkl", "catboost_models.pkl"]:
        print(f"\nProcessing {model_file}...")
        filepath = Path("outputs/models") / model_file
        
        # Load models
        with open(filepath, 'rb') as f:
            models = pickle.load(f)
        
        modified = False
        
        # Process each model
        for model_name, model_data in models.items():
            if 'cv_scores' not in model_data:
                print(f"  Adding CV scores to {model_name}...")
                
                # Get the model and training data
                model = model_data['model']
                X_train = model_data['X_train']
                y_train = model_data['y_train']
                
                try:
                    # Perform cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, 
                                                cv=5, scoring=rmse_scorer)
                    
                    # Note: sklearn returns negative scores for error metrics
                    cv_scores = -cv_scores  # Convert to positive RMSE values
                    
                    # Add to model data
                    model_data['cv_scores'] = cv_scores
                    model_data['cv_mean'] = np.mean(cv_scores)
                    model_data['cv_std'] = np.std(cv_scores)
                    
                    print(f"    CV RMSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
                    modified = True
                    
                except Exception as e:
                    print(f"    Error: {e}")
            else:
                print(f"  {model_name} already has CV scores")
        
        # Save if modified
        if modified:
            print(f"  Saving updated {model_file}...")
            with open(filepath, 'wb') as f:
                pickle.dump(models, f)
            print(f"  ✓ Saved successfully")
        else:
            print(f"  No changes needed for {model_file}")

if __name__ == "__main__":
    add_cv_scores_to_models()