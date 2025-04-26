"""Main entry point for running the ML pipeline."""

import argparse
from pathlib import Path
import sys

# Add the project directory to the path so we can import modules
project_dir = Path(__file__).parent.absolute()
sys.path.append(str(project_dir))

from config import settings

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ML model training and evaluation')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--train-linear', action='store_true', help='Train linear regression models')
    parser.add_argument('--train-linear-elasticnet', action='store_true', 
                        help='Train linear models with optimal ElasticNet parameters')
    parser.add_argument('--importance', action='store_true', help='Analyze feature importance')
    parser.add_argument('--datasets', nargs='+', default=['all'], 
                        help='Datasets to use (e.g., LR_Base LR_Yeo LR_Base_Random LR_Yeo_Random)')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--all', action='store_true', help='Run the entire pipeline')
    # Add these inside your ArgumentParser in parse_args()
    parser.add_argument('--train-sector', action='store_true', help='Train sector-specific models')
    parser.add_argument('--evaluate-sector', action='store_true', help='Evaluate sector-specific models')
    parser.add_argument('--visualize-sector', action='store_true', help='Generate sector-specific visualizations')
    parser.add_argument('--all-sector', action='store_true', help='Run the entire sector model pipeline')
    parser.add_argument('--sector-only', action='store_true', 
                    help='Run only sector models, skipping standard models')
    parser.add_argument('--vif', action='store_true', help='Analyze multicollinearity using VIF')
    parser.add_argument('--train-xgboost', action='store_true', help='Train XGBoost models')
    parser.add_argument('--optimize-xgboost', type=int, metavar='N',
                        help='Optimize XGBoost with Optuna using N trials (default: 50)')
    parser.add_argument('--visualize-xgboost', action='store_true', help='Generate XGBoost visualizations')

    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Determine if we should run standard models
    run_standard = not args.sector_only
    
    # Print configuration info
    print(f"Project root: {settings.ROOT_DIR}")
    print(f"Data directory: {settings.DATA_DIR}")
    print(f"Output directory: {settings.OUTPUT_DIR}")

    # Standard model pipeline
    if run_standard:
        if args.all or args.train or args.train_linear:
            print("\nTraining linear regression models...")
            from models.linear_regression import train_all_models
            linear_models = train_all_models()

        # Add XGBoost section after the standard model pipeline (FIXED INDENTATION)
        if args.all or args.train_xgboost or args.optimize_xgboost:
            print("\nTraining XGBoost models...")
            from models.xgboost_model import train_xgboost_models
            
            # Determine number of trials
            n_trials = args.optimize_xgboost if args.optimize_xgboost else settings.XGBOOST_PARAMS.get('n_trials', 50)
            
            xgboost_models = train_xgboost_models(datasets=args.datasets, n_trials=n_trials)
        
        if args.train_linear_elasticnet:
            print("\nTraining linear models with optimal ElasticNet parameters...")
            from models.linear_regression import train_linear_with_elasticnet_params
            linear_elasticnet_models = train_linear_with_elasticnet_params()
        
        if args.all or args.train:  # Make sure this matches your args
            print("\nTraining ElasticNet models...")
            from models.elastic_net import train_elasticnet_models
            elastic_models = train_elasticnet_models(datasets=args.datasets)
        
        if args.all or args.evaluate:
            print("\nEvaluating models...")
            from evaluation.metrics import evaluate_models
            eval_results = evaluate_models()
            
            print("\nAnalyzing feature importance...")
            from evaluation.importance import analyze_feature_importance
            importance_results = analyze_feature_importance(eval_results['all_models'])
        
        if args.importance:
            print("\nAnalyzing feature importance...")
            from evaluation.importance import analyze_feature_importance
            importance_results = analyze_feature_importance()
            
        if args.all or args.visualize:
            print("\nGenerating visualizations...")
            from visualization.metrics_plots import plot_model_comparison, plot_residuals, plot_statistical_tests, plot_elasticnet_cv_distribution, plot_metrics_summary_table
            print("Creating model performance visualizations...")
            plot_model_comparison()
            plot_residuals()
            plot_statistical_tests()
            plot_elasticnet_cv_distribution()
            plot_metrics_summary_table()
            
            # Add feature visualization calls
            print("Creating feature importance visualizations...")
            from visualization.feature_plots import plot_top_features, plot_feature_importance_by_model, plot_feature_correlations
            plot_top_features()
            plot_feature_importance_by_model()
            plot_feature_correlations()
        
        if args.all or args.visualize_xgboost:
            print("\nGenerating XGBoost visualizations...")
            from visualization.xgboost_plots import visualize_xgboost_models
            visualize_xgboost_models()

    # VIF analysis can be run separately
    if args.all or args.vif:
        print("\nAnalyzing multicollinearity using VIF...")
        from evaluation.multicollinearity import analyze_multicollinearity
        base_vif, yeo_vif = analyze_multicollinearity()

    # Sector model pipeline
    if args.all_sector or args.train_sector:
        print("\nTraining sector-specific models...")
        from models.sector_models import run_sector_models
        sector_models = run_sector_models()

    if args.all_sector or args.evaluate_sector:
        print("\nEvaluating sector-specific models...")
        from models.sector_models import evaluate_sector_models
        sector_eval_results = evaluate_sector_models()
        
        print("\nAnalyzing sector model feature importance...")
        from models.sector_models import analyze_sector_importance
        sector_importance_results = analyze_sector_importance()

    if args.all_sector or args.visualize_sector:
        print("\nGenerating sector-specific visualizations...")
        from visualization.sector_plots import visualize_sector_models
        visualize_sector_models(run_all=True)

    print("\nDone!")

if __name__ == "__main__":
    main()