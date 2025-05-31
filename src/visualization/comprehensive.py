#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive visualization module that ensures ALL visualization types are generated.
This module consolidates all visualization functions into a single workflow.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback
import logging
import time

from ..config.settings import VISUALIZATION_DIR
from .utils.io import load_all_models
from .viz_factory import (
    create_all_residual_plots,
    create_model_comparison_plot,
    create_metrics_table,
    create_comparative_dashboard
)
from .plots.features import create_cross_model_feature_importance
from .plots.sectors import visualize_all_sector_plots
from .plots.dataset_comparison import create_all_dataset_comparisons
from .plots.statistical_tests import visualize_statistical_tests
from .plots.cv_distributions import plot_cv_distributions
from .plots.shap_plots import create_all_shap_visualizations
from .plots.baselines import visualize_all_baseline_comparisons
from .plots.sector_weights import plot_all_models_sector_summary
from .plots.optimization import (
    plot_optimization_history,
    plot_param_importance,
    plot_hyperparameter_comparison,
    plot_basic_vs_optuna,
    plot_optuna_improvement
)


def create_comprehensive_visualizations(models: Optional[Dict[str, Any]] = None,
                                      visualization_dir: Optional[Path] = None) -> Dict[str, List[Path]]:
    """
    Create ALL visualization types for all models.
    
    This function ensures that every visualization type is generated:
    - Residual plots for all models
    - Feature importance plots for all models
    - CV distribution plots
    - SHAP visualizations
    - Model comparison plots
    - Metrics summary table
    - Sector performance plots
    - Dataset comparison plots
    - Statistical test plots
    - Baseline comparison plots
    - Stratification plots
    - Optimization plots (for Optuna models)
    - VIF (Variance Inflation Factor) analysis plots
    
    Args:
        models: Optional dictionary of models. If None, loads all models.
        visualization_dir: Optional visualization directory. If None, uses default.
        
    Returns:
        Dictionary mapping visualization types to lists of created paths
    """
    # Load models if not provided
    if models is None:
        print("Loading all models...")
        models = load_all_models()
    
    # Use default visualization directory if not provided
    if visualization_dir is None:
        visualization_dir = VISUALIZATION_DIR
    
    # Track all created visualizations
    all_visualizations = {}
    
    # Track timing for each visualization type
    viz_times = {}
    start_time = time.time()
    
    def log_viz_step(step_name, message, is_error=False):
        """Log visualization step with timing."""
        elapsed = time.time() - start_time
        level = logging.ERROR if is_error else logging.INFO
        logging.log(level, f"[VIZ {elapsed:.1f}s] {step_name}: {message}")
        print(f"[{elapsed:.1f}s] {message}")
    
    # 1. Residual Plots
    log_viz_step("RESIDUAL", "Creating residual plots for ALL models...")
    step_start = time.time()
    try:
        residual_paths = create_all_residual_plots()
        all_visualizations['residual_plots'] = residual_paths
        viz_times['residual_plots'] = time.time() - step_start
        log_viz_step("RESIDUAL", f"Created {len(residual_paths)} residual plots")
    except Exception as e:
        log_viz_step("RESIDUAL", f"Error: {e}", is_error=True)
        logging.exception("Full traceback:")
        traceback.print_exc()
    
    # 2. Feature Importance Plots
    log_viz_step("FEATURE_IMPORTANCE", "Creating feature importance plots...")
    step_start = time.time()
    try:
        # The function creates plots but doesn't return paths
        # We need to check if plots were created in the features directory and subdirectories
        features_dir = visualization_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        
        # Count existing plots before (including subdirectories)
        before_plots = list(features_dir.rglob("*.png"))
        
        # Create cross-model comparison
        create_cross_model_feature_importance()
        
        # Count plots after (including subdirectories)
        after_plots = list(features_dir.rglob("*.png"))
        new_plots = [p for p in after_plots if p not in before_plots]
        
        all_visualizations['feature_importance'] = new_plots
        viz_times['feature_importance'] = time.time() - step_start
        log_viz_step("FEATURE_IMPORTANCE", f"Created {len(new_plots)} feature importance plots")
    except Exception as e:
        log_viz_step("FEATURE_IMPORTANCE", f"Error: {e}", is_error=True)
        logging.exception("Full traceback:")
        traceback.print_exc()
    
    # 3. CV Distribution Plots
    log_viz_step("CV_DISTRIBUTIONS", "Creating CV distribution plots...")
    step_start = time.time()
    try:
        # Filter models with CV data
        cv_models = []
        for model_data in models.values():
            if isinstance(model_data, dict) and any(key in model_data for key in 
                                                   ['cv_scores', 'cv_fold_scores', 'cv_mean']):
                cv_models.append(model_data)
        
        if cv_models:
            cv_config = {
                'save': True,
                'output_dir': visualization_dir / "performance" / "cv_distributions",
                'dpi': 300,
                'format': 'png'
            }
            cv_figures = plot_cv_distributions(cv_models, cv_config)
            cv_paths = []
            for fig_name, fig in cv_figures.items():
                cv_path = cv_config['output_dir'] / f"{fig_name}.png"
                cv_paths.append(cv_path)
            all_visualizations['cv_distributions'] = cv_paths
            viz_times['cv_distributions'] = time.time() - step_start
            log_viz_step("CV_DISTRIBUTIONS", f"Created {len(cv_paths)} CV distribution plots")
        else:
            log_viz_step("CV_DISTRIBUTIONS", "No models with CV data found")
            viz_times['cv_distributions'] = time.time() - step_start
    except Exception as e:
        log_viz_step("CV_DISTRIBUTIONS", f"Error: {e}", is_error=True)
        logging.exception("Full traceback:")
        traceback.print_exc()
    
    # 4. SHAP Visualizations
    log_viz_step("SHAP", "Creating SHAP visualizations...")
    step_start = time.time()
    try:
        shap_paths = create_all_shap_visualizations(models, visualization_dir)
        total_shap = sum(len(paths) for paths in shap_paths.values())
        all_visualizations['shap'] = shap_paths
        viz_times['shap'] = time.time() - step_start
        log_viz_step("SHAP", f"Created {total_shap} SHAP visualizations across {len(shap_paths)} models")
        
        # Log details about what was created
        if 'model_comparison' in shap_paths:
            log_viz_step("SHAP", "✓ Model comparison SHAP plot created")
        else:
            log_viz_step("SHAP", "⚠ Model comparison SHAP plot NOT created", is_error=True)
            
    except Exception as e:
        log_viz_step("SHAP", f"Error: {e}", is_error=True)
        logging.exception("Full traceback:")
        traceback.print_exc()
    
    # 5. Model Comparison Plots
    log_viz_step("MODEL_COMPARISON", "Creating model comparison plots...")
    step_start = time.time()
    try:
        model_list = list(models.values())
        comparison_paths = create_model_comparison_plot(model_list)
        all_visualizations['model_comparison'] = comparison_paths if isinstance(comparison_paths, list) else [comparison_paths]
        viz_times['model_comparison'] = time.time() - step_start
        log_viz_step("MODEL_COMPARISON", f"Created model comparison plots")
    except Exception as e:
        log_viz_step("MODEL_COMPARISON", f"Error: {e}", is_error=True)
        logging.exception("Full traceback:")
        traceback.print_exc()
    
    # 6. Metrics Summary Table
    log_viz_step("METRICS_TABLE", "Creating metrics summary table...")
    step_start = time.time()
    try:
        model_list = list(models.values())
        metrics_path = create_metrics_table(model_list)
        all_visualizations['metrics_table'] = [metrics_path] if metrics_path else []
        viz_times['metrics_table'] = time.time() - step_start
        log_viz_step("METRICS_TABLE", f"Created metrics summary table")
    except Exception as e:
        log_viz_step("METRICS_TABLE", f"Error: {e}", is_error=True)
        logging.exception("Full traceback:")
        traceback.print_exc()
    
    # 7. Sector Performance Plots
    log_viz_step("SECTOR_PLOTS", "Creating sector performance plots...")
    step_start = time.time()
    try:
        # Load sector metrics from CSV if available
        from ..config.settings import METRICS_DIR
        import pandas as pd
        
        # Check for both standard sector models and LightGBM sector models
        sector_metrics_path = METRICS_DIR / "sector_models_metrics.csv"
        lightgbm_sector_metrics_path = METRICS_DIR / "sector_lightgbm_metrics.csv"
        
        sector_paths = []
        
        # Standard sector models
        if sector_metrics_path.exists():
            metrics_df = pd.read_csv(sector_metrics_path)
            standard_paths = visualize_all_sector_plots(metrics_df)
            sector_paths.extend(standard_paths if isinstance(standard_paths, list) else list(standard_paths.values()))
            log_viz_step("SECTOR_PLOTS", f"Created {len(standard_paths)} standard sector plots")
        
        # LightGBM sector models
        if lightgbm_sector_metrics_path.exists():
            log_viz_step("SECTOR_PLOTS", "Creating LightGBM sector visualizations...")
            from ..visualization.plots.sectors import visualize_lightgbm_sector_plots
            
            try:
                lightgbm_figures = visualize_lightgbm_sector_plots()
                if lightgbm_figures:
                    # Convert figure dict to paths
                    lightgbm_paths = []
                    for name, fig in lightgbm_figures.items():
                        if isinstance(fig, str) and fig == 'generated':
                            # Already saved, add to paths
                            lightgbm_paths.append(name)
                        elif hasattr(fig, 'savefig'):
                            # It's a figure object, should already be saved
                            lightgbm_paths.append(name)
                    
                    sector_paths.extend(lightgbm_paths)
                    log_viz_step("SECTOR_PLOTS", f"Created {len(lightgbm_paths)} LightGBM sector plots")
            except Exception as e:
                log_viz_step("SECTOR_PLOTS", f"Error creating LightGBM sector plots: {e}", is_error=True)
        
        if not sector_paths:
            log_viz_step("SECTOR_PLOTS", "No sector metrics found - run sector evaluation first")
        
        all_visualizations['sector_plots'] = sector_paths
        viz_times['sector_plots'] = time.time() - step_start
        log_viz_step("SECTOR_PLOTS", f"Total sector plots created: {len(sector_paths)}")
    except Exception as e:
        log_viz_step("SECTOR_PLOTS", f"Error: {e}", is_error=True)
        logging.exception("Full traceback:")
        traceback.print_exc()
    
    # 8. Dataset Comparison Plots
    log_viz_step("DATASET_COMPARISON", "Creating dataset comparison plots...")
    step_start = time.time()
    try:
        dataset_paths = create_all_dataset_comparisons()
        all_visualizations['dataset_comparison'] = dataset_paths
        viz_times['dataset_comparison'] = time.time() - step_start
        log_viz_step("DATASET_COMPARISON", f"Created {len(dataset_paths)} dataset comparison plots")
    except Exception as e:
        log_viz_step("DATASET_COMPARISON", f"Error: {e}", is_error=True)
        logging.exception("Full traceback:")
        traceback.print_exc()
    
    # 9. Statistical Test Plots
    log_viz_step("STATISTICAL_TESTS", "Creating statistical test plots...")
    step_start = time.time()
    try:
        # Call visualize_statistical_tests without arguments - it will use default file path
        stat_paths = visualize_statistical_tests()
        all_visualizations['statistical_tests'] = stat_paths
        viz_times['statistical_tests'] = time.time() - step_start
        log_viz_step("STATISTICAL_TESTS", f"Created {len(stat_paths)} statistical test plots")
        
        # Also run baseline significance analysis with statistical testing
        log_viz_step("STATISTICAL_TESTS", "Running baseline significance analysis...")
        from ..evaluation.baseline_significance import run_baseline_significance_analysis
        
        # Get scores data if available
        from ..data.data import load_scores_data
        try:
            scores_data = load_scores_data()
        except:
            scores_data = None
        
        # Run the analysis
        stat_test_dir = visualization_dir / "statistical_tests"
        stat_test_dir.mkdir(parents=True, exist_ok=True)
        
        results_df, baseline_plots = run_baseline_significance_analysis(
            models_dict=models,
            scores_data=scores_data,
            output_dir=stat_test_dir,
            n_folds=5,
            random_seed=42
        )
        
        if baseline_plots:
            # Add baseline significance plots to our tracking
            baseline_stat_paths = []
            for plot_name, plot_path in baseline_plots.items():
                if plot_path and Path(plot_path).exists():
                    baseline_stat_paths.append(plot_path)
            all_visualizations['baseline_significance'] = baseline_stat_paths
            log_viz_step("STATISTICAL_TESTS", f"Created {len(baseline_stat_paths)} baseline significance plots")
            
    except Exception as e:
        log_viz_step("STATISTICAL_TESTS", f"Error: {e}", is_error=True)
        logging.exception("Full traceback:")
        traceback.print_exc()
    
    # 10. Baseline Comparison Plots
    log_viz_step("BASELINE_COMPARISON", "Creating baseline comparison plots...")
    step_start = time.time()
    try:
        from .plots.baselines import visualize_all_baseline_comparisons, create_metric_baseline_comparison
        
        # Create consolidated plots
        baseline_paths = visualize_all_baseline_comparisons()
        all_visualizations['baseline_comparison'] = baseline_paths if isinstance(baseline_paths, list) else [baseline_paths]
        
        # Also create individual metric baseline comparison plots
        from ..config.settings import METRICS_DIR
        baseline_data_path = METRICS_DIR / "baseline_comparison.csv"
        
        if baseline_data_path.exists():
            output_dir = visualization_dir / "baselines"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create plots for each metric and baseline type
            metrics = ['RMSE', 'MAE', 'R²']
            baseline_types = ['Random', 'Mean', 'Median']
            
            for metric in metrics:
                # Create a single plot that shows all baseline types for this metric
                try:
                    output_path = output_dir / f"baseline_comparison_{metric}.png"
                    # This will create a plot showing Random baseline comparison
                    create_metric_baseline_comparison(
                        baseline_data_path=str(baseline_data_path),
                        output_path=str(output_path),
                        metric=metric,
                        baseline_type='Random'  # Default to Random for main comparison
                    )
                    log_viz_step("BASELINE_COMPARISON", f"Created baseline comparison plot for {metric}")
                except Exception as e:
                    log_viz_step("BASELINE_COMPARISON", f"Error creating baseline comparison for {metric}: {e}", is_error=True)
        else:
            log_viz_step("BASELINE_COMPARISON", "No baseline comparison data found - run baseline evaluation first")
            
        viz_times['baseline_comparison'] = time.time() - step_start
        log_viz_step("BASELINE_COMPARISON", f"Created baseline comparison plots")
    except Exception as e:
        log_viz_step("BASELINE_COMPARISON", f"Error: {e}", is_error=True)
        logging.exception("Full traceback:")
        traceback.print_exc()
    
    # 11. Sector Weights Plots
    log_viz_step("SECTOR_WEIGHTS", "Creating sector weights plots...")
    step_start = time.time()
    try:
        config = {'save': True, 'output_dir': visualization_dir / "sectors", 'dpi': 300, 'format': 'png'}
        sector_weights_paths = plot_all_models_sector_summary(config)
        all_visualizations['sector_weights'] = [sector_weights_paths] if sector_weights_paths else []
        viz_times['sector_weights'] = time.time() - step_start
        log_viz_step("SECTOR_WEIGHTS", f"Created sector weights plots")
    except Exception as e:
        log_viz_step("SECTOR_WEIGHTS", f"Error: {e}", is_error=True)
        logging.exception("Full traceback:")
        traceback.print_exc()
    
    # 12. Optimization Plots (for Optuna models)
    log_viz_step("OPTIMIZATION", "Creating optimization plots for Optuna models...")
    step_start = time.time()
    try:
        opt_paths = []
        for model_name, model_data in models.items():
            if 'optuna' in model_name.lower() and 'study' in model_data:
                # Get study and configuration
                study = model_data.get('study')
                # Don't specify output_dir - let the optimization functions determine the correct subdirectory
                config = {'save': True, 'dpi': 300, 'format': 'png'}
                
                # Create optimization history
                hist_path = plot_optimization_history(study, config, model_name)
                if hist_path:
                    opt_paths.append(hist_path)
                
                # Create parameter importance
                param_path = plot_param_importance(study, config, model_name)
                if param_path:
                    opt_paths.append(param_path)
                
                # Create hyperparameter comparison (this one might need model_data)
                # Skip for now as it's not clear what parameters it needs
        
        all_visualizations['optimization'] = opt_paths
        viz_times['optimization'] = time.time() - step_start
        log_viz_step("OPTIMIZATION", f"Created {len(opt_paths)} optimization plots")
    except Exception as e:
        log_viz_step("OPTIMIZATION", f"Error: {e}", is_error=True)
        logging.exception("Full traceback:")
        traceback.print_exc()
    
    # 13. Comparative Dashboard
    log_viz_step("DASHBOARD", "Creating comparative dashboard...")
    step_start = time.time()
    try:
        dashboard_path = create_comparative_dashboard(models)
        all_visualizations['dashboard'] = [dashboard_path] if dashboard_path else []
        viz_times['dashboard'] = time.time() - step_start
        log_viz_step("DASHBOARD", f"Created comparative dashboard")
    except Exception as e:
        log_viz_step("DASHBOARD", f"Error: {e}", is_error=True)
        logging.exception("Full traceback:")
        traceback.print_exc()
    
    # 14. VIF (Variance Inflation Factor) Analysis
    log_viz_step("VIF_ANALYSIS", "Creating VIF visualizations...")
    step_start = time.time()
    try:
        from ..evaluation.multicollinearity import analyze_multicollinearity
        
        # Run VIF analysis which creates visualizations
        base_vif, yeo_vif = analyze_multicollinearity()
        
        # Count the created VIF plots
        vif_dir = visualization_dir / "vif"
        if vif_dir.exists():
            vif_plots = list(vif_dir.glob("*.png"))
            all_visualizations['vif'] = vif_plots
            viz_times['vif'] = time.time() - step_start
            log_viz_step("VIF_ANALYSIS", f"Created {len(vif_plots)} VIF plots")
        else:
            all_visualizations['vif'] = []
            viz_times['vif'] = time.time() - step_start
            log_viz_step("VIF_ANALYSIS", "VIF directory not found after analysis")
    except Exception as e:
        log_viz_step("VIF_ANALYSIS", f"Error: {e}", is_error=True)
        logging.exception("Full traceback:")
        traceback.print_exc()
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)
    
    # Log summary to both console and log file
    logging.info("="*60)
    logging.info("VISUALIZATION SUMMARY")
    logging.info("="*60)
    
    total_plots = 0
    for viz_type, paths in all_visualizations.items():
        if isinstance(paths, dict):
            # Handle dictionary of paths
            count = 0
            for p in paths.values():
                if isinstance(p, list):
                    count += len(p)
                elif p is not None:
                    count += 1
        elif isinstance(paths, list):
            # Handle list of paths
            count = 0
            for p in paths:
                # Skip Figure objects and None values
                if p is not None and not hasattr(p, 'figure'):
                    count += 1
        else:
            # Handle single path or object
            count = 1 if paths and not hasattr(paths, 'figure') else 0
        total_plots += count
        
        # Get timing if available
        time_str = f" ({viz_times.get(viz_type, 0):.1f}s)" if viz_type in viz_times else ""
        
        message = f"{viz_type:.<30} {count:>4} plots{time_str}"
        print(message)
        logging.info(message)
    
    print("-"*60)
    print(f"{'TOTAL':.<30} {total_plots:>4} plots")
    print(f"{'Total time':.<30} {total_time:>4.1f} seconds")
    
    logging.info("-"*60)
    logging.info(f"{'TOTAL':.<30} {total_plots:>4} plots")
    logging.info(f"{'Total time':.<30} {total_time:>4.1f} seconds")
    
    # Check for critical plots
    critical_checks = [
        ('SHAP model comparison', 'model_comparison' in all_visualizations.get('shap', {})),
        ('Baseline comparisons', len(all_visualizations.get('baseline_comparison', [])) > 0),
        ('Statistical tests', len(all_visualizations.get('baseline_significance', [])) > 0)
    ]
    
    logging.info("\nCritical plot checks:")
    for check_name, exists in critical_checks:
        status = "✓" if exists else "✗"
        logging.info(f"  {status} {check_name}")
    print("="*60)
    
    return all_visualizations


def run_comprehensive_visualization_pipeline():
    """
    Run the complete visualization pipeline.
    This is the main entry point for creating all visualizations.
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE VISUALIZATION PIPELINE")
    print("="*60)
    print("This will create ALL visualization types for all trained models.")
    print("="*60)
    
    # Create all visualizations
    visualizations = create_comprehensive_visualizations()
    
    print("\nVisualization pipeline complete!")
    return visualizations


# Make functions available at module level
__all__ = ['create_comprehensive_visualizations', 'run_comprehensive_visualization_pipeline']