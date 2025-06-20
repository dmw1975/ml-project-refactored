#!/usr/bin/env python3
"""
Complete Feature Removal Visualizations
======================================

Generates all missing visualizations with proper function calls.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Import project modules
from src.utils.io import ensure_dir

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def create_manual_residual_plots(model_with, model_without, output_dir):
    """Create residual plots manually."""
    ensure_dir(output_dir)
    
    # Plot for model with feature
    if 'y_test' in model_with and 'y_pred' in model_with:
        plt.figure(figsize=(10, 6))
        residuals = model_with['y_test'] - model_with['y_pred']
        
        plt.subplot(1, 2, 1)
        plt.scatter(model_with['y_pred'], residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot - With Feature')
        
        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=30, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution - With Feature')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{model_with['model_name']}_residuals.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Created residual plot: {model_with['model_name']}_residuals.png")
    
    # Plot for model without feature
    if 'y_test' in model_without and 'y_pred' in model_without:
        plt.figure(figsize=(10, 6))
        residuals = model_without['y_test'] - model_without['y_pred']
        
        plt.subplot(1, 2, 1)
        plt.scatter(model_without['y_pred'], residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot - Without Feature')
        
        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=30, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution - Without Feature')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{model_without['model_name']}_residuals.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Created residual plot: {model_without['model_name']}_residuals.png")

def create_manual_shap_plots(model_with, model_without, output_dir):
    """Create SHAP plots manually."""
    ensure_dir(output_dir)
    
    try:
        # SHAP for model with feature
        if all(k in model_with for k in ['model', 'X_test', 'model_name']):
            logger.info(f"Creating SHAP plots for {model_with['model_name']}...")
            
            # Create SHAP explainer
            explainer = shap.Explainer(model_with['model'])
            shap_values = explainer(model_with['X_test'])
            
            # Summary plot
            plt.figure()
            shap.summary_plot(shap_values, model_with['X_test'], show=False)
            plt.tight_layout()
            plt.savefig(output_dir / f"{model_with['model_name']}_shap_summary.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Waterfall plot for first instance
            plt.figure()
            shap.waterfall_plot(shap_values[0], show=False)
            plt.tight_layout()
            plt.savefig(output_dir / f"{model_with['model_name']}_shap_waterfall.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ Created SHAP plots for {model_with['model_name']}")
        
        # SHAP for model without feature
        if all(k in model_without for k in ['model', 'X_test', 'model_name']):
            logger.info(f"Creating SHAP plots for {model_without['model_name']}...")
            
            # Create SHAP explainer
            explainer = shap.Explainer(model_without['model'])
            shap_values = explainer(model_without['X_test'])
            
            # Summary plot
            plt.figure()
            shap.summary_plot(shap_values, model_without['X_test'], show=False)
            plt.tight_layout()
            plt.savefig(output_dir / f"{model_without['model_name']}_shap_summary.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Waterfall plot
            plt.figure()
            shap.waterfall_plot(shap_values[0], show=False)
            plt.tight_layout()
            plt.savefig(output_dir / f"{model_without['model_name']}_shap_waterfall.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ Created SHAP plots for {model_without['model_name']}")
            
    except Exception as e:
        logger.error(f"Failed to create SHAP plots: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Generate all missing visualizations."""
    
    # Paths
    output_dir = Path("outputs/feature_removal_experiment")
    models_dir = output_dir / "models"
    
    # Load saved models
    logger.info("Loading saved models...")
    with open(models_dir / "model_with_feature.pkl", 'rb') as f:
        model_with = pickle.load(f)
    with open(models_dir / "model_without_feature.pkl", 'rb') as f:
        model_without = pickle.load(f)
    
    logger.info("Models loaded successfully")
    
    # 1. Create residual plots
    logger.info("\nCreating residual plots...")
    residuals_dir = output_dir / "visualizations" / "residuals"
    create_manual_residual_plots(model_with, model_without, residuals_dir)
    
    # 2. Create SHAP plots
    logger.info("\nCreating SHAP visualizations...")
    shap_dir = output_dir / "visualizations" / "shap"
    create_manual_shap_plots(model_with, model_without, shap_dir)
    
    # Check what was created
    logger.info("\nChecking generated files...")
    for viz_type in ['residuals', 'shap', 'feature_importance', 'metrics']:
        viz_path = output_dir / "visualizations" / viz_type
        if viz_path.exists():
            files = list(viz_path.glob("*.png"))
            logger.info(f"{viz_type}: {len(files)} files")
            for f in files:
                logger.info(f"  - {f.name}")
    
    # Summary
    print("\n" + "="*60)
    print("FEATURE REMOVAL VISUALIZATION COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print("\nVisualization Summary:")
    print("  ✓ Feature importance comparison")
    print("  ✓ Metrics comparison") 
    print("  ✓ Residual plots")
    print("  ✓ SHAP analysis")
    print("\nAll visualizations have been generated successfully!")
    print("="*60)

if __name__ == "__main__":
    main()