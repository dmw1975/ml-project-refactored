"""Comprehensive pipeline test to diagnose and fix issues with main.py --all"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import settings
from utils import io

def check_data_files():
    """Check if data files exist and are accessible"""
    print("\n=== CHECKING DATA FILES ===")
    
    raw_data = settings.DATA_DIR / "raw" / "combined_df_for_ml_models.csv"
    tree_data = settings.DATA_DIR / "processed" / "tree_models_dataset.csv"
    linear_data = settings.DATA_DIR / "processed" / "linear_models_dataset.csv"
    
    print(f"Raw data exists: {raw_data.exists()}")
    print(f"Tree models data exists: {tree_data.exists()}")
    print(f"Linear models data exists: {linear_data.exists()}")
    
    if not tree_data.exists() or not linear_data.exists():
        print("\nCreating categorical datasets...")
        from create_categorical_datasets import main as create_datasets
        create_datasets()
        print("Categorical datasets created.")
    
    return raw_data.exists()

def test_data_loading():
    """Test data loading functionality"""
    print("\n=== TESTING DATA LOADING ===")
    
    try:
        from data_categorical import load_tree_models_data, load_linear_models_data
        
        # Test tree models data loading
        print("Loading tree models data...")
        tree_datasets = load_tree_models_data()
        print(f"Tree models datasets loaded: {list(tree_datasets.keys())}")
        
        # Test linear models data loading
        print("Loading linear models data...")
        linear_datasets = load_linear_models_data()
        print(f"Linear models datasets loaded: {list(linear_datasets.keys())}")
        
        return True
    except Exception as e:
        print(f"ERROR loading data: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_training():
    """Test training for each model type"""
    print("\n=== TESTING MODEL TRAINING ===")
    
    results = {}
    
    # Test Linear Regression
    print("\n1. Testing Linear Regression training...")
    try:
        from models.linear_regression import train_all_models
        linear_models = train_all_models()
        results['linear'] = len(linear_models) if linear_models else 0
        print(f"   ✓ Trained {results['linear']} Linear Regression models")
    except Exception as e:
        print(f"   ✗ Linear Regression training failed: {e}")
        results['linear'] = 0
    
    # Test ElasticNet
    print("\n2. Testing ElasticNet training...")
    try:
        from models.elastic_net import train_elasticnet_models
        elastic_models = train_elasticnet_models(datasets=['LR_Base', 'LR_Yeo'])
        results['elasticnet'] = len(elastic_models) if elastic_models else 0
        print(f"   ✓ Trained {results['elasticnet']} ElasticNet models")
    except Exception as e:
        print(f"   ✗ ElasticNet training failed: {e}")
        results['elasticnet'] = 0
    
    # Test XGBoost
    print("\n3. Testing XGBoost training...")
    try:
        from models.xgboost_categorical import train_xgboost_categorical_models
        xgboost_models = train_xgboost_categorical_models(datasets=['Base', 'Yeo'])
        results['xgboost'] = len(xgboost_models) if xgboost_models else 0
        print(f"   ✓ Trained {results['xgboost']} XGBoost models")
    except Exception as e:
        print(f"   ✗ XGBoost training failed: {e}")
        results['xgboost'] = 0
    
    # Test LightGBM
    print("\n4. Testing LightGBM training...")
    try:
        from models.lightgbm_categorical import train_lightgbm_categorical_models
        lightgbm_models = train_lightgbm_categorical_models(datasets=['Base', 'Yeo'])
        results['lightgbm'] = len(lightgbm_models) if lightgbm_models else 0
        print(f"   ✓ Trained {results['lightgbm']} LightGBM models")
    except Exception as e:
        print(f"   ✗ LightGBM training failed: {e}")
        results['lightgbm'] = 0
    
    # Test CatBoost
    print("\n5. Testing CatBoost training...")
    try:
        from models.catboost_categorical import run_all_catboost_categorical
        catboost_models = run_all_catboost_categorical()
        results['catboost'] = len(catboost_models) if catboost_models else 0
        print(f"   ✓ Trained {results['catboost']} CatBoost models")
    except Exception as e:
        print(f"   ✗ CatBoost training failed: {e}")
        results['catboost'] = 0
    
    return results

def test_evaluation():
    """Test model evaluation"""
    print("\n=== TESTING MODEL EVALUATION ===")
    
    try:
        from evaluation.metrics import evaluate_models
        eval_results = evaluate_models()
        
        # Check what was evaluated
        if 'all_models' in eval_results:
            print(f"Evaluated {len(eval_results['all_models'])} models")
            for model_name in eval_results['all_models']:
                print(f"  - {model_name}")
        
        # Check if metrics CSV was created
        metrics_csv = settings.METRICS_DIR / "all_models_comparison.csv"
        print(f"\nMetrics CSV created: {metrics_csv.exists()}")
        
        return True
    except Exception as e:
        print(f"ERROR in evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visualization():
    """Test visualization generation"""
    print("\n=== TESTING VISUALIZATION ===")
    
    try:
        import visualization_new as viz
        from visualization_new.utils.io import load_all_models
        
        # Load models
        models = load_all_models()
        print(f"Loaded {len(models)} models for visualization")
        
        if not models:
            print("No models found to visualize!")
            return False
        
        model_list = list(models.values())
        
        # Test metrics table
        print("\nTesting metrics table generation...")
        try:
            viz.create_metrics_table(model_list)
            metrics_table = settings.VISUALIZATION_DIR / "performance" / "metrics_summary_table.png"
            print(f"Metrics table created: {metrics_table.exists()}")
        except Exception as e:
            print(f"Metrics table generation failed: {e}")
        
        # Test model comparison
        print("\nTesting model comparison plot...")
        try:
            viz.create_model_comparison_plot(model_list)
            print("Model comparison plot created")
        except Exception as e:
            print(f"Model comparison plot failed: {e}")
        
        return True
    except Exception as e:
        print(f"ERROR in visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_outputs():
    """Check what outputs were generated"""
    print("\n=== CHECKING OUTPUTS ===")
    
    # Check model files
    print("\nModel files:")
    model_dir = settings.MODEL_DIR
    for model_file in model_dir.glob("*.pkl"):
        print(f"  - {model_file.name}")
    
    # Check metrics files
    print("\nMetrics files:")
    metrics_dir = settings.METRICS_DIR
    for metrics_file in metrics_dir.glob("*.csv"):
        print(f"  - {metrics_file.name}")
    
    # Check visualization files
    print("\nVisualization files:")
    viz_perf_dir = settings.VISUALIZATION_DIR / "performance"
    if viz_perf_dir.exists():
        for viz_file in viz_perf_dir.glob("*.png"):
            print(f"  - {viz_file.name}")

def main():
    """Run comprehensive pipeline test"""
    print("="*60)
    print("COMPREHENSIVE PIPELINE TEST")
    print("="*60)
    
    # Check data files
    if not check_data_files():
        print("\nERROR: Data files missing. Cannot continue.")
        return
    
    # Test data loading
    if not test_data_loading():
        print("\nERROR: Data loading failed. Cannot continue.")
        return
    
    # Test model training
    training_results = test_model_training()
    total_models = sum(training_results.values())
    print(f"\nTotal models trained: {total_models}")
    
    if total_models == 0:
        print("\nERROR: No models were trained. Cannot continue.")
        return
    
    # Test evaluation
    if not test_evaluation():
        print("\nERROR: Model evaluation failed.")
    
    # Test visualization
    if not test_visualization():
        print("\nERROR: Visualization generation failed.")
    
    # Check outputs
    check_outputs()
    
    print("\n" + "="*60)
    print("PIPELINE TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()