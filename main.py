"""Main entry point for running the ML pipeline."""

import argparse
from pathlib import Path
import sys

# Add the project directory to the path so we can import modules
project_dir = Path(__file__).parent.absolute()
sys.path.append(str(project_dir))

from config import settings

def get_data_loading_strategy(use_one_hot):
    """Get the appropriate data loading strategy based on encoding preference."""
    if use_one_hot:
        # Use standard one-hot encoded data loading (legacy behavior)
        # Note: data module doesn't have load_data, so we use categorical approach for all
        print("Note: One-hot encoding requested but using standard data loading approach")
        # Fall through to categorical approach which works for all models
    
    # Use categorical data loading (default for tree models)
    try:
        from data_categorical import load_tree_models_data, load_linear_models_data
        return load_tree_models_data, load_linear_models_data
    except ImportError:
        print("⚠️  Categorical data module not found. Creating categorical datasets...")
        # Create categorical datasets if they don't exist
        from create_categorical_datasets import main as create_datasets
        create_datasets()
        from data_categorical import load_tree_models_data, load_linear_models_data
        return load_tree_models_data, load_linear_models_data

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ML model training and evaluation')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--train-linear', action='store_true', help='Train linear regression models')
    parser.add_argument('--train-linear-elasticnet', action='store_true', 
                        help='Train linear models with optimal ElasticNet parameters')
    parser.add_argument('--optimize-elasticnet', type=int, metavar='N',
                        help='Optimize ElasticNet with Optuna using N trials (default: 100)')
    parser.add_argument('--elasticnet-grid', action='store_true',
                        help='Use grid search instead of Optuna for ElasticNet (legacy)')
    parser.add_argument('--importance', action='store_true', help='Analyze feature importance')
    parser.add_argument('--datasets', nargs='+', default=['all'], 
                        help='Datasets to use (e.g., LR_Base LR_Yeo LR_Base_Random LR_Yeo_Random)')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations (uses new architecture)')
    parser.add_argument('--visualize-new', action='store_true', help='Generate visualizations using new architecture (same as --visualize)')
    parser.add_argument('--all', action='store_true', help='Run the entire pipeline')
    # Add these inside your ArgumentParser in parse_args()
    parser.add_argument('--train-sector', action='store_true', help='Train sector-specific models')
    parser.add_argument('--evaluate-sector', action='store_true', help='Evaluate sector-specific models')
    parser.add_argument('--visualize-sector', action='store_true', help='Generate sector-specific visualizations')
    parser.add_argument('--visualize-sector-new', action='store_true', help='Generate sector-specific visualizations using new architecture')
    parser.add_argument('--all-sector', action='store_true', help='Run the entire sector model pipeline')
    parser.add_argument('--train-sector-lightgbm', action='store_true', help='Train sector-specific LightGBM models')
    parser.add_argument('--evaluate-sector-lightgbm', action='store_true', help='Evaluate sector-specific LightGBM models')
    parser.add_argument('--visualize-sector-lightgbm', action='store_true', help='Generate sector-specific LightGBM visualizations')
    parser.add_argument('--sector-only', action='store_true', 
                    help='Run only sector models, skipping standard models')
    parser.add_argument('--vif', action='store_true', help='Analyze multicollinearity using VIF')
    parser.add_argument('--train-xgboost', action='store_true', help='Train XGBoost models')
    parser.add_argument('--optimize-xgboost', type=int, metavar='N',
                        help='Optimize XGBoost with Optuna using N trials (default: 50)')
    parser.add_argument('--visualize-xgboost', action='store_true', help='Generate XGBoost visualizations')
    parser.add_argument('--train-lightgbm', action='store_true', help='Train LightGBM models')
    parser.add_argument('--optimize-lightgbm', type=int, metavar='N',
                        help='Optimize LightGBM with Optuna using N trials (default: 50)')
    parser.add_argument('--visualize-lightgbm', action='store_true', help='Generate LightGBM visualizations')
    parser.add_argument('--train-catboost', action='store_true', help='Train CatBoost models')
    parser.add_argument('--optimize-catboost', type=int, metavar='N',
                        help='Optimize CatBoost with Optuna using N trials (default: 50)')
    parser.add_argument('--visualize-catboost', action='store_true', help='Generate CatBoost visualizations')
    parser.add_argument('--force-retune', action='store_true', 
                        help='Force retraining of Optuna studies even if they exist')
    parser.add_argument('--check-studies', action='store_true', 
                        help='Report existing Optuna studies without training')
    parser.add_argument('--use-one-hot', action='store_true', 
                        help='Use one-hot encoded features for tree models (default: native categorical features)')

    return parser.parse_args()

def main():
    """Main function."""
    # Start timing the execution
    import time
    import datetime
    start_time = time.time()
    
    # Dictionary to track execution time of each step
    step_times = {}
    
    args = parse_args()
    
    # Import settings
    from config import settings
    
    # Handle check-studies flag first
    if args.check_studies:
        print("🔍 Checking existing Optuna studies...")
        from utils.io import report_existing_studies
        report_existing_studies()
        return
    
    # Determine if we should run standard models
    run_standard = not args.sector_only
    
    # Print configuration info
    print(f"Project root: {settings.ROOT_DIR}")
    print(f"Data directory: {settings.DATA_DIR}")
    print(f"Output directory: {settings.OUTPUT_DIR}")
    
    # Set up data loading strategy based on encoding preference
    if args.use_one_hot:
        print("🔢 Using one-hot encoded features for all models (legacy mode)")
    else:
        print("🌳 Using native categorical features for tree models (default)")
        print("🔢 Using one-hot encoded features for linear models")
    
    load_tree_data, load_linear_data = get_data_loading_strategy(args.use_one_hot)

    # Standard model pipeline
    if run_standard:
        # Consolidated confirmation check for all models when using --all
        should_retrain_all = True
        if args.all and not args.force_retune:
            print("\n🔍 Checking for existing trained models across all algorithms...")
            from utils.io import check_all_existing_models, prompt_consolidated_retrain
            
            all_existing_models = check_all_existing_models(datasets=args.datasets)
            if all_existing_models:
                should_retrain_all = prompt_consolidated_retrain(all_existing_models)
            else:
                print("✨ No existing models found - proceeding with full training...")
                should_retrain_all = True
        
        if args.all or args.train or args.train_linear:
            if should_retrain_all or not args.all:
                print("\nTraining linear regression models...")
                step_start = time.time()
                from models.linear_regression import train_all_models
                linear_models = train_all_models()
                step_times["Linear Regression Training"] = time.time() - step_start
            else:
                print("\n⏭️  Skipping Linear Regression training - using existing models")
                step_times["Linear Regression Training"] = 0

        # Add XGBoost section after the standard model pipeline (FIXED INDENTATION)
        if args.all or args.train_xgboost or args.optimize_xgboost:
            if should_retrain_all or not args.all or args.force_retune:
                print("\nTraining XGBoost models...")
                step_start = time.time()
                
                if args.use_one_hot:
                    print("  🔢 Using one-hot encoded XGBoost implementation (legacy mode)")
                    from models.xgboost_categorical import train_xgboost_models
                    # Determine number of trials
                    n_trials = args.optimize_xgboost if args.optimize_xgboost else settings.XGBOOST_PARAMS.get('n_trials', 50)
                    xgboost_models = train_xgboost_models(datasets=args.datasets, n_trials=n_trials, force_retune=args.force_retune)
                else:
                    print("  🌳 Using native categorical XGBoost implementation (default)")
                    from models.xgboost_categorical import train_xgboost_categorical_models
                    xgboost_models = train_xgboost_categorical_models(datasets=args.datasets)
                
                step_times["XGBoost Training"] = time.time() - step_start
            else:
                print("\n⏭️  Skipping XGBoost training - using existing models")
                step_times["XGBoost Training"] = 0
        
        if args.train_linear_elasticnet:
            print("\nTraining linear models with optimal ElasticNet parameters...")
            from models.linear_regression import train_linear_with_elasticnet_params
            linear_elasticnet_models = train_linear_with_elasticnet_params()
        
        if args.all or args.train:  # Make sure this matches your args
            if should_retrain_all or not args.all:
                print("\nTraining ElasticNet models...")
                step_start = time.time()
                from models.elastic_net import train_elasticnet_models
                
                # Determine if using Optuna or grid search
                use_optuna = not args.elasticnet_grid
                n_trials = args.optimize_elasticnet if args.optimize_elasticnet else 100
                
                if use_optuna:
                    print(f"  🎯 Using Optuna optimization with {n_trials} trials")
                else:
                    print("  📊 Using grid search optimization (legacy)")
                
                elastic_models = train_elasticnet_models(
                    datasets=args.datasets,
                    use_optuna=use_optuna,
                    n_trials=n_trials
                )
                step_times["ElasticNet Training"] = time.time() - step_start
            else:
                print("\n⏭️  Skipping ElasticNet training - using existing models")
                step_times["ElasticNet Training"] = 0
        
        if args.all or args.evaluate:
            print("\nEvaluating models...")
            step_start = time.time()
            from evaluation.metrics import evaluate_models
            eval_results = evaluate_models()
            step_times["Model Evaluation"] = time.time() - step_start
            
            print("\nAnalyzing feature importance...")
            from evaluation.importance import analyze_feature_importance
            importance_results = analyze_feature_importance(eval_results['all_models'])
        
        if args.importance:
            print("\nAnalyzing feature importance...")
            from evaluation.importance import analyze_feature_importance
            importance_results = analyze_feature_importance()
            
        if args.all or args.visualize or args.visualize_new:
            print("\nGenerating visualizations using unified architecture...")
            step_start = time.time()
            
            # Import from visualization_new architecture
            import visualization_new as viz
            from visualization_new.utils.io import load_all_models
            
            try:
                # Register LightGBM adapter if not already registered
                try:
                    from visualization_new.adapters.lightgbm_adapter import LightGBMAdapter
                    from visualization_new.core.registry import register_adapter
                    register_adapter('lightgbm', LightGBMAdapter)
                except Exception as e:
                    print(f"Warning: Could not register LightGBM adapter: {e}")
                
                # Load all models once
                models = load_all_models()
                model_list = list(models.values())
                
                # Create all visualizations
                try:
                    print("Creating residual plots...")
                    viz.create_all_residual_plots()
                except Exception as e:
                    print(f"Error creating residual plots: {e}")
                    import traceback
                    traceback.print_exc()
                
                try:
                    print("Creating model comparison visualizations...")
                    viz.create_model_comparison_plot(model_list)
                except Exception as e:
                    print(f"Error creating model comparison: {e}")
                    import traceback
                    traceback.print_exc()
                
                try:
                    print("Creating metrics summary table...")
                    viz.create_metrics_table(model_list)
                except Exception as e:
                    print(f"Error creating metrics table: {e}")
                    import traceback
                    traceback.print_exc()
                
                try:
                    print("Creating feature importance visualizations...")
                    for model in model_list:
                        viz.create_feature_importance_plot(model)
                except Exception as e:
                    print(f"Error creating feature importance plots: {e}")
                    import traceback
                    traceback.print_exc()
                
                try:
                    print("Creating statistical test visualizations...")
                    # First try to create a comparative dashboard
                    viz.create_comparative_dashboard(model_list)
                    
                    # Then create statistical test visualizations
                    # Get the explicit path to the model comparison tests file
                    tests_file = settings.METRICS_DIR / "model_comparison_tests.csv"
                    
                    # Check if file exists before attempting to visualize
                    if tests_file.exists():
                        # Set up output directory
                        output_dir = settings.VISUALIZATION_DIR / "statistical_tests"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Create config with explicit output directory
                        from visualization_new.core.interfaces import VisualizationConfig
                        config = VisualizationConfig(
                            output_dir=output_dir,
                            format="png",
                            dpi=300,
                            save=True,
                            show=False
                        )
                        
                        # Call with explicit parameters
                        viz.visualize_statistical_tests(tests_file=tests_file, config=config)
                        print(f"Statistical test visualizations saved to {output_dir}")
                    else:
                        print(f"Statistical tests file not found: {tests_file}")
                except Exception as e:
                    print(f"Error creating statistical visualizations: {e}")
                    import traceback
                    traceback.print_exc()
                
                # ElasticNet specific visualization
                try:
                    print("Creating ElasticNet visualizations...")
                    elasticnet_models = {name: model for name, model in models.items() 
                                      if 'elasticnet' in name.lower()}
                    
                    if elasticnet_models:
                        # Import necessary modules for ElasticNet visualization
                        from visualization_new.viz_factory import (
                            visualize_model, create_hyperparameter_comparison,
                            create_basic_vs_optuna_comparison, create_optuna_improvement_plot
                        )
                        from visualization_new.adapters.elasticnet_adapter import ElasticNetAdapter
                        from visualization_new.core.registry import register_adapter
                        from visualization_new.plots.features import plot_feature_importance_comparison
                        from pathlib import Path
                        import os
                        
                        # Register ElasticNet adapter if not already registered
                        register_adapter('elasticnet', ElasticNetAdapter)
                        
                        # Visualize each model
                        for name, model in elasticnet_models.items():
                            print(f"Creating visualizations for {name}...")
                            adapter = ElasticNetAdapter(model)
                            output_paths = visualize_model(adapter)
                            print(f"Visualizations for {name}:")
                            for plot_type, path in output_paths.items():
                                print(f"  - {plot_type}: {path}")
                        
                        # Create comparison visualizations
                        print("Generating ElasticNet comparison visualizations...")
                        adapters = [ElasticNetAdapter(model) for model in elasticnet_models.values()]
                        model_data_list = list(elasticnet_models.values())
                        
                        # Set up output directory for performance plots
                        perf_dir = settings.VISUALIZATION_DIR / "performance" / "elasticnet"
                        os.makedirs(perf_dir, exist_ok=True)
                        
                        # Set up output directory for feature plots
                        features_dir = settings.VISUALIZATION_DIR / "features" / "elasticnet"
                        os.makedirs(features_dir, exist_ok=True)
                        
                        # Configuration for performance visualizations
                        from visualization_new.core.interfaces import VisualizationConfig
                        perf_config = VisualizationConfig(
                            output_dir=perf_dir,
                            format="png",
                            dpi=300,
                            save=True,
                            show=False
                        )
                        
                        # Feature importance configuration
                        feature_config = VisualizationConfig(
                            output_dir=features_dir,
                            format="png", 
                            dpi=300,
                            save=True,
                            show=False
                        )
                        
                        # Create feature importance comparisons
                        plot_feature_importance_comparison(adapters, feature_config)
                        
                        # Generate Optuna visualizations if available
                        optuna_models = {name: model for name, model in elasticnet_models.items() 
                                       if 'optuna' in name and 'study' in model}
                        
                        if optuna_models:
                            print("Generating ElasticNet Optuna visualizations...")
                            from visualization_new.plots.optimization import (
                                plot_optimization_history, plot_param_importance, plot_contour
                            )
                            
                            for model_name, model_data in optuna_models.items():
                                study = model_data.get('study')
                                if study:
                                    print(f"  Creating Optuna plots for {model_name}...")
                                    
                                    # Optimization history
                                    hist_path = plot_optimization_history(study, perf_config, model_name)
                                    if hist_path:
                                        print(f"    ✓ Optimization history: {Path(hist_path).name}")
                                    
                                    # Parameter importance
                                    param_path = plot_param_importance(study, perf_config, model_name)
                                    if param_path:
                                        print(f"    ✓ Parameter importance: {Path(param_path).name}")
                                    
                                    # Contour plot
                                    contour_path = plot_contour(study, perf_config, model_name)
                                    if contour_path:
                                        print(f"    ✓ Contour plot: {Path(contour_path).name}")
                        
                        # Create hyperparameter comparisons (one for each important parameter for ElasticNet)
                        for param in ['alpha', 'l1_ratio']:
                            try:
                                output_path = create_hyperparameter_comparison(
                                    model_data_list, param, perf_config, "elasticnet"
                                )
                                if output_path:
                                    print(f"  - {param} comparison: {output_path}")
                            except Exception as e:
                                print(f"Error creating {param} comparison: {e}")
                        
                        # Create basic vs optuna comparison if applicable
                        try:
                            output_path = create_basic_vs_optuna_comparison(
                                model_data_list, perf_config, "elasticnet"
                            )
                            if output_path:
                                print(f"  - Basic vs Optuna comparison: {output_path}")
                        except Exception as e:
                            print(f"Error creating basic vs optuna comparison: {e}")
                        
                        # Create optuna improvement plot if applicable
                        try:
                            output_path = create_optuna_improvement_plot(
                                model_data_list, perf_config, "elasticnet"
                            )
                            if output_path:
                                print(f"  - Optuna improvement: {output_path}")
                        except Exception as e:
                            print(f"Error creating optuna improvement plot: {e}")
                        
                        print(f"ElasticNet visualizations completed successfully using new architecture.")
                    else:
                        print("No ElasticNet models found.")
                except Exception as e:
                    print(f"Error creating ElasticNet visualizations: {e}")
                    import traceback
                    traceback.print_exc()
                
                # LightGBM specific visualization
                try:
                    print("Creating LightGBM visualizations...")
                    lightgbm_models = {name: model for name, model in models.items() 
                                     if 'lightgbm' in name.lower()}
                    
                    if lightgbm_models:
                        from visualization_new.viz_factory import visualize_model
                        from visualization_new.adapters.lightgbm_adapter import LightGBMAdapter
                        from visualization_new.plots.features import plot_feature_importance_comparison
                        import os
                        
                        # Visualize each LightGBM model
                        for name, model_data in lightgbm_models.items():
                            print(f"Creating visualizations for {name}...")
                            adapter = LightGBMAdapter(model_data)
                            output_paths = visualize_model(adapter)
                            
                            # Check if feature importance was created
                            if 'feature_importance' in output_paths:
                                print(f"Feature importance saved to {output_paths['feature_importance']}")
                        
                        # Create comparison visualization for LightGBM models
                        adapters = [LightGBMAdapter(model_data) for model_data in lightgbm_models.values()]
                        performance_dir = settings.VISUALIZATION_DIR / "performance" / "lightgbm"
                        os.makedirs(performance_dir, exist_ok=True)
                        
                        # Create feature importance comparison
                        features_dir = settings.VISUALIZATION_DIR / "features" / "lightgbm"
                        os.makedirs(features_dir, exist_ok=True)
                        
                        # Create config for feature importance comparison
                        from visualization_new.core.interfaces import VisualizationConfig
                        importance_config = VisualizationConfig(
                            output_dir=features_dir,
                            format="png",
                            dpi=300,
                            save=True,
                            show=False,
                            create_heatmap=False  # Explicitly disable heatmap for LightGBM
                        )
                        
                        # Generate feature importance comparison
                        plot_feature_importance_comparison(adapters, importance_config)
                        print(f"LightGBM feature importance comparison saved to {features_dir}")
                    else:
                        print("No LightGBM models found.")
                except Exception as e:
                    print(f"Error creating LightGBM visualizations: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Dataset comparison visualizations
                try:
                    print("Creating dataset comparison visualizations...")
                    viz.plots.dataset_comparison.create_all_dataset_comparisons()
                except Exception as e:
                    print(f"Error creating dataset comparisons: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Create cross-model feature importance comparisons by dataset
                try:
                    print("\nCreating cross-model feature importance comparisons by dataset...")
                    from visualization_new.viz_factory import create_cross_model_feature_importance_by_dataset
                    
                    dataset_paths = create_cross_model_feature_importance_by_dataset(
                        format='png',
                        dpi=300,
                        show=False
                    )
                    
                    if dataset_paths:
                        print("Cross-model feature importance comparisons created successfully.")
                    else:
                        print("No cross-model feature importance comparisons were created.")
                except Exception as e:
                    print(f"Error creating cross-model feature importance comparisons: {e}")
                
                # Add baseline comparison visualizations
                try:
                    print("\nCreating baseline comparison visualizations...")
                    from visualization_new.plots.baselines import visualize_all_baseline_comparisons, create_metric_baseline_comparison
                    from pathlib import Path
                    
                    # Create consolidated plots
                    baseline_figures = visualize_all_baseline_comparisons(create_individual_plots=False)
                    if baseline_figures:
                        print("Baseline comparison visualizations created successfully.")
                    
                    # Also create individual metric baseline comparison plots
                    baseline_data_path = settings.METRICS_DIR / "baseline_comparison.csv"
                    if baseline_data_path.exists():
                        output_dir = settings.VISUALIZATION_DIR / "baselines"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Create plots for each metric
                        for metric in ['RMSE', 'MAE', 'R²']:
                            try:
                                output_path = output_dir / f"baseline_comparison_{metric}.png"
                                create_metric_baseline_comparison(
                                    baseline_data_path=str(baseline_data_path),
                                    output_path=str(output_path),
                                    metric=metric,
                                    baseline_type='Random'
                                )
                                print(f"  Created baseline comparison plot for {metric}")
                            except Exception as e:
                                print(f"  Error creating baseline comparison for {metric}: {e}")
                    else:
                        print("No baseline comparison data found - run baseline evaluation first")
                        
                except Exception as e:
                    print(f"Error creating baseline comparison visualizations: {e}")
                
                print("Visualization completed successfully.")
                step_times["New Visualization"] = time.time() - step_start
            except Exception as e:
                print(f"Error in visualization pipeline: {e}")
                import traceback
                traceback.print_exc()
                step_times["Failed New Visualization"] = time.time() - step_start
                print("\nVisualization failed. Please check the error messages above.")
            
        if args.all or args.visualize_new:
            print("\nGenerating visualizations using new architecture...")
            # Import from new visualization architecture
            import visualization_new as viz
            from visualization_new.utils.io import load_all_models
            
            try:
                # Load all models once
                models = load_all_models()
                model_list = list(models.values())
                
                # Create all visualizations
                try:
                    print("Creating residual plots...")
                    viz.create_all_residual_plots()
                except Exception as e:
                    print(f"Error creating residual plots: {e}")
                
                try:
                    print("Creating model comparison visualizations...")
                    viz.create_model_comparison_plot(model_list)
                except Exception as e:
                    print(f"Error creating model comparison: {e}")
                
                try:
                    print("Creating metrics summary table...")
                    viz.create_metrics_table(model_list)
                except Exception as e:
                    print(f"Error creating metrics table: {e}")
                
                try:
                    print("Creating visualization dashboard...")
                    viz.create_comparative_dashboard()
                except Exception as e:
                    print(f"Error creating dashboard: {e}")
                    
                try:
                    print("Creating dataset-centric model comparisons...")
                    viz.create_all_dataset_comparisons()
                except Exception as e:
                    print(f"Error creating dataset comparisons: {e}")
                
                try:
                    print("Creating statistical test visualizations...")
                    # Get the explicit path to the model comparison tests file
                    tests_file = settings.METRICS_DIR / "model_comparison_tests.csv"
                    
                    # Check if file exists before attempting to visualize
                    if tests_file.exists():
                        # Set up output directory
                        output_dir = settings.VISUALIZATION_DIR / "statistical_tests"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Create config with explicit output directory
                        from visualization_new.core.interfaces import VisualizationConfig
                        config = VisualizationConfig(
                            output_dir=output_dir,
                            format="png",
                            dpi=300,
                            save=True,
                            show=False
                        )
                        
                        # Call with explicit parameters
                        viz.visualize_statistical_tests(tests_file=tests_file, config=config)
                        print(f"Statistical test visualizations saved to {output_dir}")
                    else:
                        print(f"Statistical tests file not found: {tests_file}")
                        print("Run model evaluation with statistical tests first.")
                except Exception as e:
                    print(f"Error creating statistical test visualizations: {e}")
                    import traceback

                # Generate SHAP visualizations for tree models
                try:
                    print("Creating SHAP visualizations...")
                    # Check if SHAP script exists
                    shap_script = Path("generate_shap_visualizations.py")
                    if shap_script.exists():
                        import subprocess
                        result = subprocess.run([sys.executable, str(shap_script)], 
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            print("SHAP visualizations created successfully.")
                        else:
                            print(f"Error creating SHAP visualizations: {result.stderr}")
                    else:
                        print("SHAP visualization script not found. Skipping.")
                except Exception as e:
                    print(f"Error creating SHAP visualizations: {e}")
                
                # Generate CV distribution plots
                try:
                    print("Creating CV distribution plots...")
                    from visualization_new.plots.cv_distributions import plot_cv_distributions
                    
                    # Filter models with CV data
                    cv_models = []
                    for model_data in model_list:
                        if isinstance(model_data, dict) and ('cv_scores' in model_data or 
                                                            'cv_fold_scores' in model_data or
                                                            'cv_mean' in model_data):
                            cv_models.append(model_data)
                    
                    if cv_models:
                        cv_config = {
                            'save': True,
                            'output_dir': settings.VISUALIZATION_DIR / "performance" / "cv_distributions",
                            'dpi': 300,
                            'format': 'png'
                        }
                        cv_figures = plot_cv_distributions(cv_models, cv_config)
                        print(f"Created {len(cv_figures)} CV distribution plots.")
                    else:
                        print("No models with CV data found. Skipping CV distribution plots.")
                except Exception as e:
                    print(f"Error creating CV distribution plots: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Generate performance plots for CatBoost and XGBoost
                try:
                    print("Creating performance optimization plots...")
                    perf_script = Path(__file__).parent / "generate_missing_performance_plots.py"
                    if perf_script.exists():
                        import subprocess
                        result = subprocess.run([sys.executable, str(perf_script)], 
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            print("Performance optimization plots created successfully.")
                        else:
                            print(f"Error creating performance plots: {result.stderr}")
                    else:
                        print("Performance plot generation script not found. Skipping.")
                except Exception as e:
                    print(f"Error creating performance optimization plots: {e}")
                    import traceback
                    traceback.print_exc()
                
                print("New architecture visualizations complete.")
            except Exception as e:
                print(f"Error using new visualization architecture: {e}")
                print("Continuing with legacy visualizations...")
        
        if args.all or args.visualize_xgboost:
            print("\nGenerating XGBoost visualizations...")
            
            # Try to use the new architecture, but fall back to legacy if there are issues
            use_new_system = True
            
            try:
                # Import new architecture components
                from visualization_new.viz_factory import (
                    visualize_model, create_optimization_history_plot, create_param_importance_plot,
                    create_hyperparameter_comparison, create_basic_vs_optuna_comparison,
                    create_optuna_improvement_plot
                )
                from visualization_new.adapters.xgboost_adapter import XGBoostAdapter
                from visualization_new.core.registry import register_adapter
                from visualization_new.utils.io import load_all_models
                import os
                
                # Register XGBoost adapter
                register_adapter('xgboost', XGBoostAdapter)
            except Exception as e:
                print(f"Error importing new visualization modules for XGBoost: {str(e)}")
                print("XGBoost visualization modules not available.")
                use_new_system = False
            
            if use_new_system:
                try:
                    # Load all models and filter for XGBoost models
                    all_models = load_all_models()
                    xgboost_models = {name: model for name, model in all_models.items() 
                                     if 'xgb' in name.lower()}
                    
                    if not xgboost_models:
                        print("No XGBoost models found. Please train models first.")
                    else:
                        # Visualize each model
                        for name, model_data in xgboost_models.items():
                            print(f"Generating visualizations for {name}...")
                            adapter = XGBoostAdapter(model_data)
                            output_paths = visualize_model(adapter)
                            print(f"Generated visualizations for {name}:")
                            for plot_type, path in output_paths.items():
                                print(f"  - {plot_type}: {path}")
                        
                        # Create comparison visualizations
                        print("Generating XGBoost comparison visualizations...")
                        adapters = [XGBoostAdapter(model_data) for model_data in xgboost_models.values()]
                        model_data_list = list(xgboost_models.values())
                        
                        # Set up output directory for performance plots
                        from config import settings
                        perf_dir = settings.VISUALIZATION_DIR / "performance" / "xgboost"
                        os.makedirs(perf_dir, exist_ok=True)
                        
                        # Configuration
                        from visualization_new.core.interfaces import VisualizationConfig
                        config = VisualizationConfig(
                            output_dir=perf_dir,
                            format="png",
                            dpi=300,
                            save=True,
                            show=False
                        )
                        
                        # Create hyperparameter comparisons (one for each important parameter)
                        for param in ['learning_rate', 'max_depth', 'n_estimators', 'subsample', 'colsample_bytree']:
                            try:
                                output_path = create_hyperparameter_comparison(
                                    model_data_list, param, config, "xgboost"
                                )
                                if output_path:
                                    print(f"  - {param} comparison: {output_path}")
                            except Exception as e:
                                print(f"Error creating {param} comparison: {e}")
                        
                        # Create basic vs optuna comparison
                        try:
                            output_path = create_basic_vs_optuna_comparison(
                                model_data_list, config, "xgboost"
                            )
                            if output_path:
                                print(f"  - Basic vs Optuna comparison: {output_path}")
                        except Exception as e:
                            print(f"Error creating basic vs optuna comparison: {e}")
                        
                        # Create optuna improvement plot
                        try:
                            output_path = create_optuna_improvement_plot(
                                model_data_list, config, "xgboost"
                            )
                            if output_path:
                                print(f"  - Optuna improvement: {output_path}")
                        except Exception as e:
                            print(f"Error creating optuna improvement plot: {e}")
                        
                        print(f"XGBoost visualizations completed successfully using new architecture.")
                except Exception as e:
                    print(f"Error using new visualization architecture for XGBoost: {str(e)}")
                    print("XGBoost visualization failed. Please check the error messages above.")
            
        if args.all or args.train_lightgbm or args.optimize_lightgbm:
            if should_retrain_all or not args.all or args.force_retune:
                print("\nTraining LightGBM models...")
                step_start = time.time()
                
                if args.use_one_hot:
                    print("  🔢 Using one-hot encoded LightGBM implementation (legacy mode)")
                    from models.lightgbm_categorical import train_lightgbm_models
                    # Determine number of trials
                    n_trials = args.optimize_lightgbm if args.optimize_lightgbm else settings.LIGHTGBM_PARAMS.get('n_trials', 50)
                    lightgbm_models = train_lightgbm_models(datasets=args.datasets, n_trials=n_trials)
                else:
                    print("  🌳 Using native categorical LightGBM implementation (default)")
                    from models.lightgbm_categorical import train_lightgbm_categorical_models
                    lightgbm_models = train_lightgbm_categorical_models(datasets=args.datasets)
                
                step_times["LightGBM Training"] = time.time() - step_start
            else:
                print("\n⏭️  Skipping LightGBM training - using existing models")
                step_times["LightGBM Training"] = 0
            
        if args.all or args.visualize_lightgbm:
            print("\nGenerating LightGBM visualizations...")
            
            # LightGBM visualization using new architecture
            
            # Try to use the new system, but fall back to the old if there are issues
            use_new_system = True
            
            try:
                # Import new architecture components
                from visualization_new.viz_factory import (
                    visualize_model, create_optimization_history_plot, create_param_importance_plot,
                    create_hyperparameter_comparison, create_basic_vs_optuna_comparison,
                    create_optuna_improvement_plot
                )
                from visualization_new.adapters.lightgbm_adapter import LightGBMAdapter
                from visualization_new.core.registry import register_adapter
                from visualization_new.plots.metrics import plot_model_comparison
                from visualization_new.plots.features import plot_feature_importance_comparison
                from visualization_new.utils.io import load_all_models, load_model_data
                import os
                
                # Register LightGBM adapter
                register_adapter('lightgbm', LightGBMAdapter)
            except Exception as e:
                print(f"Error importing new visualization modules for LightGBM: {str(e)}")
                print("LightGBM visualization failed. Please check the error messages above.")
                use_new_system = False
            
            if use_new_system:
                try:
                    # Load all models and filter for LightGBM models
                    all_models = load_all_models()
                    lightgbm_models = {name: model for name, model in all_models.items() 
                                      if 'lightgbm' in name.lower()}
                    
                    if not lightgbm_models:
                        print("No LightGBM models found. Please train models first.")
                    else:
                        # Visualize each model
                        for name, model_data in lightgbm_models.items():
                            print(f"Generating visualizations for {name}...")
                            adapter = LightGBMAdapter(model_data)
                            output_paths = visualize_model(adapter)
                            print(f"Generated visualizations for {name}:")
                            for plot_type, path in output_paths.items():
                                print(f"  - {plot_type}: {path}")
                        
                        # Create comparison visualizations
                        print("Generating LightGBM comparison visualizations...")
                        adapters = [LightGBMAdapter(model_data) for model_data in lightgbm_models.values()]
                        model_data_list = list(lightgbm_models.values())
                        
                        # Set up separate output directories for performance and feature plots
                        perf_dir = settings.VISUALIZATION_DIR / "performance" / "lightgbm"
                        os.makedirs(perf_dir, exist_ok=True)
                        
                        features_dir = settings.VISUALIZATION_DIR / "features" / "lightgbm"
                        os.makedirs(features_dir, exist_ok=True)
                        
                        # Configuration for performance visualizations
                        from visualization_new.core.interfaces import VisualizationConfig
                        perf_config = VisualizationConfig(
                            output_dir=perf_dir,
                            format="png",
                            dpi=300,
                            save=True,
                            show=False
                        )
                        
                        # Feature importance configuration
                        feature_config = VisualizationConfig(
                            output_dir=features_dir,
                            format="png", 
                            dpi=300,
                            save=True,
                            show=False,
                            create_heatmap=False  # Explicitly disable heatmap for LightGBM
                        )
                        
                        # Create feature importance comparisons
                        plot_feature_importance_comparison(adapters, feature_config)
                        
                        # Create hyperparameter comparisons (one for each important parameter)
                        for param in ['learning_rate', 'num_leaves', 'max_depth', 'min_child_samples', 'feature_fraction', 'bagging_fraction']:
                            try:
                                output_path = create_hyperparameter_comparison(
                                    model_data_list, param, perf_config, "lightgbm"
                                )
                                if output_path:
                                    print(f"  - {param} comparison: {output_path}")
                            except Exception as e:
                                print(f"Error creating {param} comparison: {e}")
                        
                        # Create basic vs optuna comparison
                        try:
                            output_path = create_basic_vs_optuna_comparison(
                                model_data_list, perf_config, "lightgbm"
                            )
                            if output_path:
                                print(f"  - Basic vs Optuna comparison: {output_path}")
                        except Exception as e:
                            print(f"Error creating basic vs optuna comparison: {e}")
                        
                        # Create optuna improvement plot
                        try:
                            output_path = create_optuna_improvement_plot(
                                model_data_list, perf_config, "lightgbm"
                            )
                            if output_path:
                                print(f"  - Optuna improvement: {output_path}")
                        except Exception as e:
                            print(f"Error creating optuna improvement plot: {e}")
                        
                        print(f"LightGBM visualizations completed successfully using new architecture.")
                except Exception as e:
                    print(f"Error using new visualization architecture for LightGBM: {str(e)}")
                    print("Falling back to deprecated visualization...")
                    visualize_lightgbm_models()
            
        if args.all or args.train_lightgbm or args.optimize_lightgbm:
            if should_retrain_all or not args.all or args.force_retune:
                print("\nTraining LightGBM models...")
                step_start = time.time()
                
                if args.use_one_hot:
                    print("  🔢 Using one-hot encoded LightGBM implementation (legacy mode)")
                    from models.lightgbm_categorical import train_lightgbm_models
                    # Determine number of trials
                    n_trials = args.optimize_lightgbm if args.optimize_lightgbm else settings.LIGHTGBM_PARAMS.get('n_trials', 50)
                    lightgbm_models = train_lightgbm_models(datasets=args.datasets, n_trials=n_trials, force_retune=args.force_retune)
                else:
                    print("  🌳 Using native categorical LightGBM implementation (default)")
                    from models.lightgbm_categorical import train_lightgbm_categorical_models
                    lightgbm_models = train_lightgbm_categorical_models(datasets=args.datasets)
                
                step_times["LightGBM Training"] = time.time() - step_start
            else:
                print("\n⏭️  Skipping LightGBM training - using existing models")
                step_times["LightGBM Training"] = 0
            
        if args.all or args.train_catboost or args.optimize_catboost:
            if should_retrain_all or not args.all or args.force_retune:
                print("\nTraining CatBoost models...")
                step_start = time.time()
                
                if args.use_one_hot:
                    print("  🔢 Using one-hot encoded CatBoost implementation (legacy mode)")
                    from models.catboost_categorical import train_catboost_models
                    # Determine number of trials
                    n_trials = args.optimize_catboost if args.optimize_catboost else settings.CATBOOST_PARAMS.get('n_trials', 50)
                    catboost_models = train_catboost_models(datasets=args.datasets, n_trials=n_trials)
                else:
                    print("  🌳 Using native categorical CatBoost implementation (default)")
                    from models.catboost_categorical import train_catboost_categorical_models
                    catboost_models = train_catboost_categorical_models(datasets=args.datasets)
                
                step_times["CatBoost Training"] = time.time() - step_start
            else:
                print("\n⏭️  Skipping CatBoost training - using existing models")
                step_times["CatBoost Training"] = 0
            
        if args.all or args.visualize_catboost:
            print("\nGenerating CatBoost visualizations...")
            
            # CatBoost visualization using new architecture
            
            # Try to use the new system, but fall back to the old if there are issues
            use_new_system = True
            
            try:
                # Import new architecture components
                from visualization_new.viz_factory import (
                    visualize_model, create_optimization_history_plot, create_param_importance_plot,
                    create_hyperparameter_comparison, create_basic_vs_optuna_comparison,
                    create_optuna_improvement_plot
                )
                from visualization_new.adapters.catboost_adapter import CatBoostAdapter
                from visualization_new.core.registry import register_adapter
                from visualization_new.plots.metrics import plot_model_comparison
                from visualization_new.plots.features import plot_feature_importance_comparison
                from visualization_new.utils.io import load_all_models, load_model_data
                import os
                
                # Register CatBoost adapter
                register_adapter('catboost', CatBoostAdapter)
            except Exception as e:
                print(f"Error importing new visualization modules: {str(e)}")
                print("CatBoost visualization failed. Please check the error messages above.")
                use_new_system = False
            
            if use_new_system:
                try:
                    # Load all models and filter for CatBoost models
                    all_models = load_all_models()
                    catboost_models = {name: model for name, model in all_models.items() 
                                     if 'catboost' in name.lower()}
                    
                    if not catboost_models:
                        print("No CatBoost models found. Please train models first.")
                    else:
                        # Visualize each model
                        for name, model_data in catboost_models.items():
                            print(f"Generating visualizations for {name}...")
                            adapter = CatBoostAdapter(model_data)
                            output_paths = visualize_model(adapter)
                            print(f"Generated visualizations for {name}:")
                            for plot_type, path in output_paths.items():
                                print(f"  - {plot_type}: {path}")
                        
                        # Create comparison visualizations
                        print("Generating CatBoost comparison visualizations...")
                        adapters = [CatBoostAdapter(model_data) for model_data in catboost_models.values()]
                        model_data_list = list(catboost_models.values())
                        
                        # Set up separate output directories for performance and feature plots
                        perf_dir = settings.VISUALIZATION_DIR / "performance" / "catboost"
                        os.makedirs(perf_dir, exist_ok=True)
                        
                        features_dir = settings.VISUALIZATION_DIR / "features" / "catboost"
                        os.makedirs(features_dir, exist_ok=True)
                        
                        # Configuration for performance visualizations
                        from visualization_new.core.interfaces import VisualizationConfig
                        perf_config = VisualizationConfig(
                            output_dir=perf_dir,
                            format="png",
                            dpi=300,
                            save=True,
                            show=False
                        )
                        
                        # Feature importance configuration
                        feature_config = VisualizationConfig(
                            output_dir=features_dir,
                            format="png", 
                            dpi=300,
                            save=True,
                            show=False
                        )
                        
                        # Create feature importance comparisons
                        plot_feature_importance_comparison(adapters, feature_config)
                        
                        # Create hyperparameter comparisons (one for each important parameter)
                        for param in ['learning_rate', 'depth', 'iterations', 'l2_leaf_reg', 'border_count', 'bagging_temperature']:
                            try:
                                output_path = create_hyperparameter_comparison(
                                    model_data_list, param, perf_config, "catboost"
                                )
                                if output_path:
                                    print(f"  - {param} comparison: {output_path}")
                            except Exception as e:
                                print(f"Error creating {param} comparison: {e}")
                        
                        # Create basic vs optuna comparison
                        try:
                            output_path = create_basic_vs_optuna_comparison(
                                model_data_list, perf_config, "catboost"
                            )
                            if output_path:
                                print(f"  - Basic vs Optuna comparison: {output_path}")
                        except Exception as e:
                            print(f"Error creating basic vs optuna comparison: {e}")
                        
                        # Create optuna improvement plot
                        try:
                            output_path = create_optuna_improvement_plot(
                                model_data_list, perf_config, "catboost"
                            )
                            if output_path:
                                print(f"  - Optuna improvement: {output_path}")
                        except Exception as e:
                            print(f"Error creating optuna improvement plot: {e}")
                        
                        print(f"CatBoost visualizations completed successfully using new architecture.")
                except Exception as e:
                    print(f"Error using new visualization architecture: {str(e)}")
                    print("CatBoost visualization failed. Continuing...")
                
        # Visualizations using new architecture are handled in a consolidated block above

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
        
        # Legacy sector visualization
        print("\nSector visualizations are not available in the new architecture yet.")
        print("Skipping sector visualizations...")
    
    if args.all_sector or args.visualize_sector_new or args.all or (args.all and args.visualize_new):
        print("\nGenerating sector visualizations using new architecture...")
        try:
            import visualization_new as viz
            viz.visualize_all_sector_plots()
            print("Sector visualizations with new architecture complete.")
        except Exception as e:
            print(f"Error generating sector visualizations with new architecture: {e}")
            print(f"Error details: {str(e)}")
            print("Continuing with legacy sector visualizations...")
    
    # LightGBM Sector model pipeline
    if args.train_sector_lightgbm:
        print("\nTraining sector-specific LightGBM models...")
        from models.sector_lightgbm_models import run_sector_lightgbm_models
        sector_lightgbm_models = run_sector_lightgbm_models()

    if args.evaluate_sector_lightgbm:
        print("\nEvaluating sector-specific LightGBM models...")
        from models.sector_lightgbm_models import evaluate_sector_lightgbm_models
        sector_lightgbm_eval_results = evaluate_sector_lightgbm_models()
        
        print("\nAnalyzing sector LightGBM model feature importance...")
        from models.sector_lightgbm_models import analyze_sector_lightgbm_importance
        sector_lightgbm_importance_results = analyze_sector_lightgbm_importance()

    if args.visualize_sector_lightgbm:
        print("\nGenerating sector-specific LightGBM visualizations...")
        
        # Use the new visualization architecture for LightGBM sectors
        try:
            from visualization_new.plots.sectors import visualize_lightgbm_sector_plots
            
            # Check if LightGBM sector models exist
            lightgbm_metrics_file = settings.METRICS_DIR / "sector_lightgbm_metrics.csv"
            if lightgbm_metrics_file.exists():
                # Generate LightGBM sector visualizations
                lightgbm_figures = visualize_lightgbm_sector_plots()
                
                if lightgbm_figures:
                    print(f"Generated {len(lightgbm_figures)} LightGBM sector visualization plots")
                    print(f"Plots saved to: {settings.VISUALIZATION_DIR / 'sectors' / 'lightgbm'}")
                else:
                    print("No LightGBM sector visualizations were generated")
            else:
                print("No LightGBM sector metrics found. Please train models first with --train-sector-lightgbm")
        except Exception as e:
            print(f"Error generating LightGBM sector visualizations: {e}")
            import traceback
            traceback.print_exc()
    
    # Run additional visualizations when using --all flag
    if args.all:
        step_start = time.time()
        run_additional_visualizations()
        step_times["Additional Visualizations"] = time.time() - step_start
    
    # Calculate and display execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Format as hours, minutes, seconds
    time_formatted = str(datetime.timedelta(seconds=int(execution_time)))
    
    print("\nDone!")
    print(f"\nTotal execution time: {time_formatted} (HH:MM:SS)")
    
    # Display detailed breakdown by step
    if step_times:
        print("\nExecution time breakdown by step:")
        print("-" * 50)
        print(f"{'Step':<35} | {'Time (sec)':<10} | {'Time %':<10}")
        print("-" * 50)
        
        # Sort by execution time (descending)
        sorted_steps = sorted(step_times.items(), key=lambda x: x[1], reverse=True)
        
        for step, step_time in sorted_steps:
            percent = (step_time / execution_time) * 100
            time_formatted = str(datetime.timedelta(seconds=int(step_time)))
            print(f"{step:<35} | {time_formatted:<10} | {percent:6.2f}%")
    
    # If the execution was longer than 5 minutes, provide a simpler breakdown
    if execution_time > 300:
        hours = int(execution_time // 3600)
        minutes = int((execution_time % 3600) // 60)
        seconds = int(execution_time % 60)
        
        print(f"\nTotal runtime: {hours} hours, {minutes} minutes, {seconds} seconds")

def run_additional_visualizations():
    """Run additional visualizations not covered in the standard visualization modules."""
    try:
        # Add cross-validation plots
        print("\nGenerating cross-validation plots...")
        from generate_model_cv_plots import main as generate_cv_plots
        generate_cv_plots()
        
        # Add ElasticNet CV plots
        print("\nGenerating ElasticNet CV plots...")
        from generate_elasticnet_cv_plots import generate_elasticnet_cv_plots
        generate_elasticnet_cv_plots()
        
        # Add stratified splitting plot
        print("\nGenerating sector stratification plots...")
        from create_sector_stratification_plot import create_sector_stratification_plot
        create_sector_stratification_plot()
        
        # Add SHAP visualizations
        try:
            print("\nGenerating SHAP visualizations...")
            from generate_shap_visualizations import main as generate_shap_viz
            generate_shap_viz()
            
            # Add improved CatBoost SHAP visualizations for categorical features
            print("\nGenerating improved CatBoost SHAP visualizations for categorical features...")
            from improved_catboost_shap_categorical import (
                create_categorical_shap_plot, create_mixed_shap_summary, 
                identify_categorical_features
            )
            import shap
            import pickle
            from pathlib import Path
            import numpy as np
            
            # Load CatBoost models
            catboost_path = settings.MODEL_DIR / "catboost_models.pkl"
            if catboost_path.exists():
                with open(catboost_path, 'rb') as f:
                    catboost_models = pickle.load(f)
                
                # Process the best performing CatBoost model (prefer optuna)
                best_model_name = None
                best_model_data = None
                for name, data in catboost_models.items():
                    if 'optuna' in name and 'model' in data and 'X_test' in data:
                        best_model_name = name
                        best_model_data = data
                        break
                
                # Fallback to any available model
                if best_model_data is None:
                    for name, data in catboost_models.items():
                        if 'model' in data and 'X_test' in data:
                            best_model_name = name
                            best_model_data = data
                            break
                
                if best_model_data:
                    print(f"  Using {best_model_name} for improved visualizations")
                    
                    # Get test data sample
                    X_test = best_model_data['X_test']
                    sample_size = min(100, len(X_test))
                    X_sample = X_test.sample(sample_size, random_state=42) if len(X_test) > sample_size else X_test
                    
                    # Calculate SHAP values
                    model = best_model_data['model']
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                    
                    # Identify categorical features
                    categorical_features = identify_categorical_features(X_sample, model)
                    print(f"  Identified {len(categorical_features)} categorical features: {categorical_features[:5]}...")
                    
                    # Create output directory
                    shap_dir = Path(settings.OUTPUT_DIR) / "visualizations" / "shap" / "catboost_improved"
                    shap_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create mixed summary plot
                    mixed_path = shap_dir / "catboost_mixed_shap_summary.png"
                    create_mixed_shap_summary(shap_values, X_sample, categorical_features, mixed_path)
                    print(f"  Created mixed SHAP summary: {mixed_path.name}")
                    
                    # Create individual categorical feature plots for top categorical features
                    if categorical_features:
                        # Calculate importance for categorical features
                        cat_importance = {}
                        for cat_feat in categorical_features:
                            if cat_feat in X_sample.columns:
                                feat_idx = list(X_sample.columns).index(cat_feat)
                                cat_importance[cat_feat] = np.abs(shap_values[:, feat_idx]).mean()
                        
                        # Get top 5 most important categorical features
                        top_categorical = sorted(cat_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                        
                        print(f"  Creating plots for top {len(top_categorical)} categorical features...")
                        for feat_name, importance in top_categorical:
                            cat_plot_path = shap_dir / f"catboost_categorical_{feat_name.replace('/', '_')}.png"
                            create_categorical_shap_plot(shap_values, X_sample, feat_name, cat_plot_path)
                            print(f"    - {feat_name}: {cat_plot_path.name}")
                    
                    print("  Improved CatBoost SHAP visualizations complete!")
                else:
                    print("  No suitable CatBoost model found for improved visualizations")
            else:
                print("  CatBoost models not found, skipping improved visualizations")
                
        except Exception as e:
            print(f"Warning: SHAP visualization generation partially failed: {e}")
            print("Some SHAP plots may have been created successfully. Check outputs/visualizations/shap/")
        
        # Add baseline comparison visualizations
        try:
            print("\nGenerating baseline comparison visualizations...")
            # First try to load and generate from existing metrics
            from visualization_new.plots.baselines import visualize_all_baseline_comparisons
            baseline_figures = visualize_all_baseline_comparisons(create_individual_plots=False)
            
            # If that doesn't work, run the baseline evaluation directly
            if not baseline_figures:
                print("No existing baseline metrics found. Running baseline evaluation...")
                from utils import io
                all_models = io.load_all_models()
                
                if all_models:
                    from evaluation.baselines import run_baseline_evaluation
                    
                    # Run evaluation with all baseline types (random, mean, median)
                    print("Running random, mean, and median baseline evaluations...")
                    _, _ = run_baseline_evaluation(
                        all_models,
                        include_mean=True,
                        include_median=True
                    )
                    
                    # Now try the visualization again
                    baseline_figures = visualize_all_baseline_comparisons(create_individual_plots=False)
                    if baseline_figures:
                        print("Baseline comparison visualizations created successfully.")
                else:
                    print("No models found. Cannot generate baseline comparisons.")
        except Exception as e:
            print(f"Error generating baseline comparison visualizations: {e}")
        
        print("\nAdditional visualizations complete.")
    except Exception as e:
        print(f"Error generating additional visualizations: {e}")

if __name__ == "__main__":
    main()