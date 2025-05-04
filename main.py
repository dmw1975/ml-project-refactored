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
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations (uses new architecture)')
    parser.add_argument('--visualize-new', action='store_true', help='Generate visualizations using new architecture (same as --visualize)')
    parser.add_argument('--all', action='store_true', help='Run the entire pipeline')
    # Add these inside your ArgumentParser in parse_args()
    parser.add_argument('--train-sector', action='store_true', help='Train sector-specific models')
    parser.add_argument('--evaluate-sector', action='store_true', help='Evaluate sector-specific models')
    parser.add_argument('--visualize-sector', action='store_true', help='Generate sector-specific visualizations')
    parser.add_argument('--visualize-sector-new', action='store_true', help='Generate sector-specific visualizations using new architecture')
    parser.add_argument('--all-sector', action='store_true', help='Run the entire sector model pipeline')
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

    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Import settings
    from config import settings
    
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
            
        if args.all or args.visualize or args.visualize_new:
            print("\nGenerating visualizations using unified architecture...")
            
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
                    # Fallback to legacy visualization if needed
                    print("Falling back to legacy residual plots...")
                    from visualization.create_residual_plots import create_all_residual_plots
                    create_all_residual_plots()
                
                try:
                    print("Creating model comparison visualizations...")
                    viz.create_model_comparison_plot(model_list)
                except Exception as e:
                    print(f"Error creating model comparison: {e}")
                    # Fallback to legacy visualization if needed
                    print("Falling back to legacy model comparison...")
                    from visualization.metrics_plots import plot_model_comparison
                    plot_model_comparison()
                
                try:
                    print("Creating metrics summary table...")
                    viz.create_metrics_table(model_list)
                except Exception as e:
                    print(f"Error creating metrics table: {e}")
                    # Fallback to legacy visualization if needed
                    print("Falling back to legacy metrics table...")
                    from visualization.metrics_plots import plot_metrics_summary_table
                    plot_metrics_summary_table()
                
                try:
                    print("Creating feature importance visualizations...")
                    for model in model_list:
                        viz.create_feature_importance_plot(model)
                except Exception as e:
                    print(f"Error creating feature importance plots: {e}")
                    # Fallback to legacy visualization if needed
                    print("Falling back to legacy feature importance plots...")
                    from visualization.feature_plots import plot_top_features, plot_feature_importance_by_model
                    plot_top_features()
                    plot_feature_importance_by_model()
                
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
                    # Fallback to legacy visualization if needed
                    print("Falling back to legacy statistical tests...")
                    from visualization.statistical_tests import visualize_statistical_tests
                    visualize_statistical_tests()
                
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
                    # Fallback to legacy visualization if needed
                    print("Falling back to legacy ElasticNet visualizations...")
                    from visualization.elasticnet_plots import plot_elasticnet_feature_importance
                    plot_elasticnet_feature_importance()
                
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
                    # Fallback to legacy visualization if needed
                    print("Falling back to legacy LightGBM visualizations...")
                    from visualization.lightgbm_plots import plot_lightgbm_feature_importance
                    plot_lightgbm_feature_importance()
                
                # Dataset comparison visualizations
                try:
                    print("Creating dataset comparison visualizations...")
                    viz.plots.dataset_comparison.create_all_dataset_comparisons()
                except Exception as e:
                    print(f"Error creating dataset comparisons: {e}")
                    # Fallback to legacy visualization if needed
                    print("Falling back to legacy dataset comparisons...")
                    from visualization.dataset_comparison import create_all_dataset_comparisons
                    create_all_dataset_comparisons()
                
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
                
                print("Visualization completed successfully.")
            except Exception as e:
                print(f"Error in visualization pipeline: {e}")
                print("Falling back to legacy visualization...")
                
                # Import legacy visualization as a last resort
                print("\nFalling back to legacy visualization module...")
                from visualization.metrics_plots import plot_model_comparison, plot_residuals, plot_statistical_tests_filtered
                from visualization.create_residual_plots import create_all_residual_plots
                from visualization.feature_plots import plot_top_features, plot_feature_importance_by_model
                from visualization.statistical_tests import visualize_statistical_tests
                from visualization.lightgbm_plots import plot_lightgbm_feature_importance
                
                print("Creating model performance visualizations...")
                plot_model_comparison()
                create_all_residual_plots()
                plot_residuals()
                plot_statistical_tests_filtered()
                visualize_statistical_tests()
                
                print("Creating feature importance visualizations...")
                plot_top_features()
                plot_feature_importance_by_model()
                plot_lightgbm_feature_importance()
            
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
                print("Falling back to deprecated visualization...")
                use_new_system = False
                from visualization.xgboost_plots import visualize_xgboost_models
                visualize_xgboost_models()
            
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
                    print("Falling back to deprecated visualization...")
                    from visualization.xgboost_plots import visualize_xgboost_models
                    visualize_xgboost_models()
            
        if args.all or args.train_lightgbm or args.optimize_lightgbm:
            print("\nTraining LightGBM models...")
            from models.lightgbm_model import train_lightgbm_models
            
            # Determine number of trials
            n_trials = args.optimize_lightgbm if args.optimize_lightgbm else settings.LIGHTGBM_PARAMS.get('n_trials', 50)
            
            lightgbm_models = train_lightgbm_models(datasets=args.datasets, n_trials=n_trials)
            
        if args.all or args.visualize_lightgbm:
            print("\nGenerating LightGBM visualizations...")
            
            # Import the fallback option first
            from visualization.lightgbm_plots import visualize_lightgbm_models
            
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
                print("Falling back to deprecated visualization...")
                use_new_system = False
                visualize_lightgbm_models()
            
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
            
        if args.all or args.train_catboost or args.optimize_catboost:
            print("\nTraining CatBoost models...")
            from models.catboost_model import train_catboost_models
            
            # Determine number of trials
            n_trials = args.optimize_catboost if args.optimize_catboost else settings.CATBOOST_PARAMS.get('n_trials', 50)
            
            catboost_models = train_catboost_models(datasets=args.datasets, n_trials=n_trials)
            
        if args.all or args.visualize_catboost:
            print("\nGenerating CatBoost visualizations...")
            
            # Import the fallback option first
            from visualization.catboost_plots import visualize_catboost_models
            
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
                print("Falling back to deprecated visualization...")
                use_new_system = False
                visualize_catboost_models()
            
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
                    print("Falling back to deprecated visualization...")
                    visualize_catboost_models()
                
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
        print("Using legacy visualization module...")
        from visualization.sector_plots import visualize_sector_models
        visualize_sector_models(run_all=True)
    
    if args.all_sector or args.visualize_sector_new or (args.all and args.visualize_new):
        print("\nGenerating sector visualizations using new architecture...")
        try:
            import visualization_new as viz
            viz.visualize_all_sector_plots()
            print("Sector visualizations with new architecture complete.")
        except Exception as e:
            print(f"Error generating sector visualizations with new architecture: {e}")
            print("Continuing with legacy sector visualizations...")

    print("\nDone!")

if __name__ == "__main__":
    main()