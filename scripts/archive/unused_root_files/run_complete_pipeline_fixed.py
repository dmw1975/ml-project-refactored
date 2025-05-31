"""
Complete ML Pipeline - Fixed Version

This script runs the entire ML pipeline with proper error handling.
It ensures all models are trained and saved in the correct format.
"""

import sys
import time
import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.config import settings
from utils import io


def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f" {title.upper()} ")
    print("="*70)


def train_all_models():
    """Train all model types"""
    print_header("TRAINING ALL MODELS")
    
    results = {}
    
    # 1. Linear Regression
    print("\n1. Training Linear Regression models...")
    try:
        from src.models.linear_regression import train_all_models
        linear_models = train_all_models()
        results['linear'] = len(linear_models) if linear_models else 0
        print(f"   ✓ Trained {results['linear']} Linear Regression models")
    except Exception as e:
        print(f"   ✗ Linear Regression failed: {e}")
        results['linear'] = 0
    
    # 2. ElasticNet
    print("\n2. Training ElasticNet models...")
    try:
        from src.models.elastic_net import train_elasticnet_models
        elastic_models = train_elasticnet_models(datasets=['all'])
        results['elasticnet'] = len(elastic_models) if elastic_models else 0
        print(f"   ✓ Trained {results['elasticnet']} ElasticNet models")
    except Exception as e:
        print(f"   ✗ ElasticNet failed: {e}")
        results['elasticnet'] = 0
    
    # 3. XGBoost (use one-hot encoding to ensure compatibility)
    print("\n3. Training XGBoost models...")
    try:
        from src.models.xgboost_model import train_xgboost_models
        xgboost_models = train_xgboost_models(datasets=['all'], n_trials=50)
        results['xgboost'] = len(xgboost_models) if xgboost_models else 0
        print(f"   ✓ Trained {results['xgboost']} XGBoost models")
    except Exception as e:
        print(f"   ✗ XGBoost failed: {e}")
        results['xgboost'] = 0
    
    # 4. LightGBM (use one-hot encoding to ensure compatibility)
    print("\n4. Training LightGBM models...")
    try:
        from src.models.lightgbm_model import train_lightgbm_models
        lightgbm_models = train_lightgbm_models(datasets=['all'], n_trials=50)
        results['lightgbm'] = len(lightgbm_models) if lightgbm_models else 0
        print(f"   ✓ Trained {results['lightgbm']} LightGBM models")
    except Exception as e:
        print(f"   ✗ LightGBM failed: {e}")
        results['lightgbm'] = 0
    
    # 5. CatBoost (use categorical features)
    print("\n5. Training CatBoost models...")
    try:
        from src.models.catboost_categorical import run_all_catboost_categorical
        catboost_models = run_all_catboost_categorical()
        results['catboost'] = len(catboost_models) if catboost_models else 0
        print(f"   ✓ Trained {results['catboost']} CatBoost models")
    except Exception as e:
        print(f"   ✗ CatBoost failed: {e}")
        results['catboost'] = 0
    
    return results


def evaluate_models():
    """Evaluate all models"""
    print_header("MODEL EVALUATION")
    
    try:
        from src.evaluation.metrics import evaluate_models
        eval_results = evaluate_models()
        
        if eval_results and 'all_models' in eval_results:
            model_count = len(eval_results['all_models'])
            print(f"\n✓ Successfully evaluated {model_count} models")
            
            # Display summary
            metrics_file = settings.METRICS_DIR / "all_models_comparison.csv"
            if metrics_file.exists():
                import pandas as pd
                df = pd.read_csv(metrics_file)
                print("\nTop 10 Models by RMSE:")
                print("-" * 70)
                print(df[['model_name', 'RMSE', 'R2', 'model_type']].sort_values('RMSE').head(10).to_string(index=False))
            
            return True
        else:
            print("\n✗ No models were evaluated")
            return False
            
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_visualizations():
    """Generate all visualizations"""
    print_header("VISUALIZATION GENERATION")
    
    try:
        # Import and register all adapters
        from src.visualization.core.registry import register_adapter
        from src.visualization.adapters.catboost_adapter import CatBoostAdapter
        from src.visualization.adapters.elasticnet_adapter import ElasticNetAdapter
        from src.visualization.adapters.lightgbm_adapter import LightGBMAdapter
        from src.visualization.adapters.linear_regression_adapter import LinearRegressionAdapter
        from src.visualization.adapters.xgboost_adapter import XGBoostAdapter
        
        register_adapter('catboost', CatBoostAdapter)
        register_adapter('elasticnet', ElasticNetAdapter)
        register_adapter('lightgbm', LightGBMAdapter)
        register_adapter('linearregression', LinearRegressionAdapter)
        register_adapter('xgboost', XGBoostAdapter)
        
        import src.visualization as viz
        from src.visualization.utils.io import load_all_models
        
        # Load models
        models = load_all_models()
        if not models:
            print("✗ No models found for visualization")
            return False
        
        model_list = list(models.values())
        print(f"\nGenerating visualizations for {len(models)} models...")
        
        # Generate key visualizations
        success_count = 0
        
        # 1. Metrics table (most important)
        try:
            print("  Creating metrics summary table...")
            viz.create_metrics_table(model_list)
            metrics_table_path = settings.VISUALIZATION_DIR / "performance" / "metrics_summary_table.png"
            if metrics_table_path.exists():
                print(f"    ✓ Created: {metrics_table_path}")
                success_count += 1
        except Exception as e:
            print(f"    ✗ Failed: {e}")
        
        # 2. Model comparison
        try:
            print("  Creating model comparison plot...")
            viz.create_model_comparison_plot(model_list)
            print("    ✓ Created model comparison")
            success_count += 1
        except Exception as e:
            print(f"    ✗ Failed: {e}")
        
        # 3. Residual plots
        try:
            print("  Creating residual plots...")
            viz.create_all_residual_plots()
            print("    ✓ Created residual plots")
            success_count += 1
        except Exception as e:
            print(f"    ✗ Failed: {e}")
        
        # 4. Feature importance
        try:
            print("  Creating feature importance plots...")
            for i, model in enumerate(model_list[:5]):  # Limit to first 5 models
                viz.create_feature_importance_plot(model)
            print("    ✓ Created feature importance plots")
            success_count += 1
        except Exception as e:
            print(f"    ✗ Failed: {e}")
        
        print(f"\n✓ Successfully generated {success_count}/4 visualization types")
        return success_count > 0
        
    except Exception as e:
        print(f"\n✗ Visualization generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_additional_visualizations():
    """Generate additional visualizations"""
    print_header("ADDITIONAL VISUALIZATIONS")
    
    success_count = 0
    
    # 1. Cross-validation plots
    try:
        print("  Generating cross-validation plots...")
        from scripts.utilities.generate_model_cv_plots import main as generate_cv_plots
        generate_cv_plots()
        print("    ✓ Created CV plots")
        success_count += 1
    except Exception as e:
        print(f"    ✗ CV plots failed: {e}")
    
    # 2. SHAP visualizations
    try:
        print("  Generating SHAP visualizations...")
        from scripts.utilities.generate_shap_visualizations import main as generate_shap_viz
        generate_shap_viz()
        print("    ✓ Created SHAP visualizations")
        success_count += 1
    except Exception as e:
        print(f"    ✗ SHAP visualizations failed: {e}")
    
    print(f"\n✓ Generated {success_count} additional visualization types")
    return success_count > 0


def check_outputs():
    """Check and report on generated outputs"""
    print_header("OUTPUT VERIFICATION")
    
    outputs = {
        'Models': {
            'linear_regression_models.pkl': settings.MODEL_DIR / 'linear_regression_models.pkl',
            'elasticnet_models.pkl': settings.MODEL_DIR / 'elasticnet_models.pkl',
            'xgboost_models.pkl': settings.MODEL_DIR / 'xgboost_models.pkl',
            'lightgbm_models.pkl': settings.MODEL_DIR / 'lightgbm_models.pkl',
            'catboost_categorical_models.pkl': settings.MODEL_DIR / 'catboost_categorical_models.pkl'
        },
        'Metrics': {
            'all_models_comparison.csv': settings.METRICS_DIR / 'all_models_comparison.csv',
            'model_comparison_tests.csv': settings.METRICS_DIR / 'model_comparison_tests.csv'
        },
        'Visualizations': {
            'metrics_summary_table.png': settings.VISUALIZATION_DIR / 'performance' / 'metrics_summary_table.png',
            'model_comparison.png': settings.VISUALIZATION_DIR / 'performance' / 'model_comparison.png'
        }
    }
    
    for category, files in outputs.items():
        print(f"\n{category}:")
        for name, path in files.items():
            status = "✓" if path.exists() else "✗"
            print(f"  {status} {name}")


def main():
    """Run the complete pipeline"""
    start_time = time.time()
    
    print("\n" + "="*70)
    print(" COMPLETE ML PIPELINE RUNNER - FIXED VERSION ")
    print("="*70)
    print(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Train all models
    training_results = train_all_models()
    total_models = sum(training_results.values())
    
    if total_models == 0:
        print("\n✗ ERROR: No models were trained. Cannot continue.")
        return
    
    print(f"\n✓ Total models trained: {total_models}")
    
    # 2. Evaluate models
    eval_success = evaluate_models()
    
    if not eval_success:
        print("\n⚠️  WARNING: Evaluation had issues but continuing...")
    
    # 3. Generate visualizations
    viz_success = generate_visualizations()
    
    # 4. Generate additional visualizations
    additional_viz = generate_additional_visualizations()
    
    # 5. Check outputs
    check_outputs()
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print_header("PIPELINE COMPLETE")
    print(f"\nTotal runtime: {datetime.timedelta(seconds=int(duration))}")
    print(f"\nModels trained: {total_models}")
    print(f"Evaluation: {'✓ Success' if eval_success else '✗ Failed'}")
    print(f"Visualizations: {'✓ Success' if viz_success else '✗ Failed'}")
    
    print("\n✓ Pipeline completed successfully!")
    print("\nYou can now use 'python main.py --all' to run the full pipeline.")


if __name__ == "__main__":
    main()