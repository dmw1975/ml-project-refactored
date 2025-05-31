"""Main entry point for running the ML pipeline using the refactored architecture."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from src.pipelines import TrainingPipeline, EvaluationPipeline, VisualizationPipeline
from src.config import settings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ML model training and evaluation')
    
    # Main pipeline actions
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--all', action='store_true', help='Run the entire pipeline')
    
    # Model-specific training
    parser.add_argument('--train-linear', action='store_true', help='Train linear regression models')
    parser.add_argument('--train-xgboost', action='store_true', help='Train XGBoost models')
    parser.add_argument('--train-lightgbm', action='store_true', help='Train LightGBM models')
    parser.add_argument('--train-catboost', action='store_true', help='Train CatBoost models')
    
    # Optimization options
    parser.add_argument('--optimize-elasticnet', type=int, metavar='N',
                        help='Optimize ElasticNet with Optuna using N trials (default: 100)')
    parser.add_argument('--optimize-xgboost', type=int, metavar='N',
                        help='Optimize XGBoost with Optuna using N trials (default: 50)')
    parser.add_argument('--optimize-lightgbm', type=int, metavar='N',
                        help='Optimize LightGBM with Optuna using N trials (default: 50)')
    parser.add_argument('--optimize-catboost', type=int, metavar='N',
                        help='Optimize CatBoost with Optuna using N trials (default: 50)')
    
    # Other options
    parser.add_argument('--datasets', nargs='+', default=['all'], 
                        help='Datasets to use (e.g., LR_Base LR_Yeo LR_Base_Random LR_Yeo_Random)')
    parser.add_argument('--force-retrain', action='store_true', 
                        help='Force retraining even if models exist')
    parser.add_argument('--use-one-hot', action='store_true', 
                        help='Use one-hot encoded features for tree models (default: native categorical)')
    parser.add_argument('--elasticnet-grid', action='store_true',
                        help='Use grid search instead of Optuna for ElasticNet (legacy)')
    
    # Evaluation options
    parser.add_argument('--importance', action='store_true', help='Analyze feature importance')
    parser.add_argument('--vif', action='store_true', help='Analyze multicollinearity using VIF')
    
    return parser.parse_args()


def create_config(args) -> Dict[str, Any]:
    """Create configuration dictionary from arguments."""
    config = {
        'project_root': settings.ROOT_DIR,
        'data_dir': settings.DATA_DIR,
        'output_dir': settings.OUTPUT_DIR,
        'model_dir': settings.MODEL_DIR,
        'visuals_dir': settings.VISUALS_DIR,
        'random_state': settings.RANDOM_STATE,
        'test_size': settings.TEST_SIZE,
        'cv_folds': settings.CV_FOLDS,
    }
    return config


def determine_models_to_train(args) -> list:
    """Determine which models to train based on arguments."""
    models = []
    
    if args.all or args.train:
        # Train all models
        models = ['linear_regression', 'elasticnet', 'xgboost', 'lightgbm', 'catboost']
    else:
        # Train specific models
        if args.train_linear:
            models.extend(['linear_regression', 'elasticnet'])
        if args.train_xgboost or args.optimize_xgboost:
            models.append('xgboost')
        if args.train_lightgbm or args.optimize_lightgbm:
            models.append('lightgbm')
        if args.train_catboost or args.optimize_catboost:
            models.append('catboost')
            
    return models


def main():
    """Main function."""
    args = parse_args()
    
    # Print configuration info
    print(f"Project root: {settings.ROOT_DIR}")
    print(f"Data directory: {settings.DATA_DIR}")
    print(f"Output directory: {settings.OUTPUT_DIR}")
    
    # Determine encoding preference
    if args.use_one_hot:
        print("🔢 Using one-hot encoded features for all models (legacy mode)")
    else:
        print("🌳 Using native categorical features for tree models (default)")
        print("🔢 Using one-hot encoded features for linear models")
    
    # Create configuration
    config = create_config(args)
    
    # Determine what to run
    run_training = args.all or args.train or any([
        args.train_linear, args.train_xgboost, args.train_lightgbm, args.train_catboost,
        args.optimize_elasticnet, args.optimize_xgboost, args.optimize_lightgbm, args.optimize_catboost
    ])
    run_evaluation = args.all or args.evaluate or args.importance or args.vif
    run_visualization = args.all or args.visualize
    
    # Training pipeline
    if run_training:
        print("\n" + "="*60)
        print("TRAINING PIPELINE")
        print("="*60)
        
        models_to_train = determine_models_to_train(args)
        training_pipeline = TrainingPipeline(config)
        
        trained_models = training_pipeline.run(
            models=models_to_train,
            datasets=args.datasets,
            force_retrain=args.force_retrain,
            use_one_hot=args.use_one_hot,
            elasticnet_grid=args.elasticnet_grid,
            optimize_elasticnet=args.optimize_elasticnet,
            optimize_xgboost=args.optimize_xgboost,
            optimize_lightgbm=args.optimize_lightgbm,
            optimize_catboost=args.optimize_catboost,
            force_retune=args.force_retrain
        )
    
    # Evaluation pipeline
    if run_evaluation:
        print("\n" + "="*60)
        print("EVALUATION PIPELINE")
        print("="*60)
        
        eval_pipeline = EvaluationPipeline(config)
        
        eval_results = eval_pipeline.run(
            evaluate_metrics=args.all or args.evaluate,
            evaluate_baselines=args.all or args.evaluate,
            analyze_importance=args.all or args.importance,
            analyze_vif=args.vif
        )
    
    # Visualization pipeline
    if run_visualization:
        print("\n" + "="*60)
        print("VISUALIZATION PIPELINE")
        print("="*60)
        
        viz_pipeline = VisualizationPipeline(config)
        
        # Determine which plots to generate
        plot_types = None  # None means all plots
        if not args.all:
            # If specific visualizations requested, could add more granular control here
            plot_types = None  # For now, generate all when --visualize is used
            
        generated_plots = viz_pipeline.run(plot_types=plot_types)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print("\nDone!")


if __name__ == "__main__":
    main()