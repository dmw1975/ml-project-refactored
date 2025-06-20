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
import matplotlib.pyplot as plt

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
                                      visualization_dir: Optional[Path] = None,
                                      fail_fast: bool = False) -> Dict[str, List[Path]]:
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
        fail_fast: If True, stop on first error. If False (default), continue with other visualizations.
        
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
        # Validate paths exist
        valid_paths = []
        for path in residual_paths:
            if isinstance(path, (str, Path)) and Path(path).exists():
                valid_paths.append(Path(path))
        
        all_visualizations['residual_plots'] = valid_paths
        viz_times['residual_plots'] = time.time() - step_start
        log_viz_step("RESIDUAL", f"Created {len(valid_paths)} residual plots")
        
        if len(valid_paths) < len(residual_paths):
            log_viz_step("RESIDUAL", f"Warning: {len(residual_paths) - len(valid_paths)} plots failed validation", is_error=True)
            
    except Exception as e:
        log_viz_step("RESIDUAL", f"Error: {e}", is_error=True)
        logging.exception("Full traceback:")
        traceback.print_exc()
        all_visualizations['residual_plots'] = []
    
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
            # Ensure output directory exists
            cv_config['output_dir'].mkdir(parents=True, exist_ok=True)
            
            # Import the function that handles single models
            from src.visualization.plots.cv_distributions import plot_cv_distribution_single
            cv_figures = {}
            cv_paths = []
            
            # Count actual files created
            cv_dir = cv_config['output_dir']
            before_files = set(cv_dir.glob("*.png"))
            
            # Process each model individually
            for idx, model_data in enumerate(cv_models):
                try:
                    # Create subdirectory by model type
                    model_name = model_data.get('model_name', f'model_{idx}')
                    model_type = 'unknown'
                    if 'catboost' in model_name.lower():
                        model_type = 'catboost'
                    elif 'lightgbm' in model_name.lower():
                        model_type = 'lightgbm'
                    elif 'xgboost' in model_name.lower():
                        model_type = 'xgboost'
                    elif 'elasticnet' in model_name.lower():
                        model_type = 'elasticnet'
                    elif model_name.lower().startswith('lr_'):
                        model_type = 'linear'
                    
                    # Update config with model-specific directory
                    model_cv_config = cv_config.copy()
                    model_cv_config['output_dir'] = cv_config['output_dir'].parent / model_type
                    model_cv_config['output_dir'].mkdir(parents=True, exist_ok=True)
                    
                    fig = plot_cv_distribution_single(model_data, model_cv_config)
                    if fig:
                        cv_figures[model_name] = fig
                        # Figure is saved by plot_cv_distribution_single
                        plt.close(fig)
                        log_viz_step("CV_DISTRIBUTIONS", f"Created CV distribution for {model_name}")
                except Exception as e:
                    log_viz_step("CV_DISTRIBUTIONS", f"Error with model {model_name}: {e}", is_error=True)
            
            # Check for new files created
            after_files = set(cv_dir.glob("*.png"))
            new_files = after_files - before_files
            
            # Include both tracked paths and newly created files
            all_cv_files = list(cv_dir.glob("*_cv_distribution.png"))
            
            all_visualizations['cv_distributions'] = all_cv_files
            viz_times['cv_distributions'] = time.time() - step_start
            log_viz_step("CV_DISTRIBUTIONS", f"Created {len(all_cv_files)} CV distribution plots")
            
            # Validate files exist
            for cv_file in all_cv_files:
                if not cv_file.exists():
                    log_viz_step("CV_DISTRIBUTIONS", f"Warning: Expected file not found: {cv_file}", is_error=True)
        else:
            log_viz_step("CV_DISTRIBUTIONS", "No models with CV data found")
            all_visualizations['cv_distributions'] = []
            viz_times['cv_distributions'] = time.time() - step_start
    except Exception as e:
        log_viz_step("CV_DISTRIBUTIONS", f"Error: {e}", is_error=True)
        logging.exception("Full traceback:")
        traceback.print_exc()
        all_visualizations['cv_distributions'] = []
    
    # 4. SHAP Visualizations
    log_viz_step("SHAP", "Creating SHAP visualizations...")
    step_start = time.time()
    try:
        # Split models by type for SHAP compatibility
        shap_compatible_models = {}
        linear_models = {}
        
        for name, model_data in models.items():
            if 'Linear_Regression' in name or 'LR_' in name and 'ElasticNet' not in name:
                linear_models[name] = model_data
            else:
                shap_compatible_models[name] = model_data
        
        if linear_models:
            log_viz_step("SHAP", f"Skipping {len(linear_models)} Linear Regression models (SHAP not supported)")
        
        # Create SHAP visualizations only for compatible models
        if shap_compatible_models:
            shap_paths = create_all_shap_visualizations(shap_compatible_models, visualization_dir)
            total_shap = sum(len(paths) if isinstance(paths, list) else 1 for paths in shap_paths.values() if paths)
            all_visualizations['shap'] = shap_paths
            viz_times['shap'] = time.time() - step_start
            log_viz_step("SHAP", f"Created {total_shap} SHAP visualizations across {len(shap_paths)} models")
            
            # CRITICAL VERIFICATION: Check if CatBoost and LightGBM models got SHAP visualizations
            shap_dir = visualization_dir / "shap"
            if shap_dir.exists():
                # Count directories by model type
                model_type_counts = {"CatBoost": 0, "LightGBM": 0, "XGBoost": 0, "ElasticNet": 0}
                for d in shap_dir.iterdir():
                    if d.is_dir():
                        for model_type in model_type_counts:
                            if model_type in d.name:
                                model_type_counts[model_type] += 1
                                break
                
                # Log verification results
                for model_type, count in model_type_counts.items():
                    if count == 0 and any(model_type in name for name in shap_compatible_models.keys()):
                        log_viz_step("SHAP", f"⚠️ WARNING: No {model_type} SHAP directories created!", is_error=True)
                    elif count > 0:
                        log_viz_step("SHAP", f"✓ {model_type}: {count} SHAP directories created")
                
                # Check total expected vs actual
                expected_models = len(shap_compatible_models)
                actual_dirs = len([d for d in shap_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
                if actual_dirs < expected_models:
                    log_viz_step("SHAP", f"⚠️ WARNING: Expected {expected_models} SHAP dirs but only {actual_dirs} exist", is_error=True)
            
            # Note: SHAP model comparison across different model types is complex
            # and may not be feasible due to different feature spaces and model architectures
            log_viz_step("SHAP", "ℹ️ SHAP model comparison skipped (technical limitation)")
                
        else:
            log_viz_step("SHAP", "No SHAP-compatible models found")
            all_visualizations['shap'] = {}
            viz_times['shap'] = time.time() - step_start
            
    except Exception as e:
        log_viz_step("SHAP", f"Error: {e}", is_error=True)
        logging.exception("Full traceback:")
        traceback.print_exc()
        all_visualizations['shap'] = {}
    
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
        # Use comprehensive loading to ensure all models are included
        config = {
            'comprehensive': True,
            'save': True,
            'output_dir': visualization_dir / "performance",
            'dpi': 300,
            'format': 'png'
        }
        # Pass None to trigger comprehensive loading
        metrics_path = create_metrics_table(None, config)
        all_visualizations['metrics_table'] = [metrics_path] if metrics_path else []
        viz_times['metrics_table'] = time.time() - step_start
        log_viz_step("METRICS_TABLE", f"Created comprehensive metrics summary table")
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
            # baseline_plots contains Figure objects, not paths
            # The plots are already saved inside plot_cv_baseline_tests
            baseline_stat_count = len([fig for fig in baseline_plots.values() if fig is not None])
            log_viz_step("STATISTICAL_TESTS", f"Created {baseline_stat_count} baseline significance plots")
        
        # Display statistical testing results
        if results_df is not None and not results_df.empty:
            log_viz_step("STATISTICAL_TESTS", "Statistical Testing Results Summary:")
            # Group by baseline type and show summary
            for baseline_type in results_df['baseline_type'].unique():
                baseline_results = results_df[results_df['baseline_type'] == baseline_type]
                significant_count = baseline_results['cv_p_value'].apply(lambda p: p < 0.05).sum()
                total_models = len(baseline_results)
                log_viz_step("STATISTICAL_TESTS", 
                            f"  {baseline_type} baseline: {significant_count}/{total_models} models significantly better (p<0.05)")
            
    except Exception as e:
        log_viz_step("STATISTICAL_TESTS", f"Error: {e}", is_error=True)
        logging.exception("Full traceback:")
        traceback.print_exc()
    
    # 10. Baseline Comparison Plots
    log_viz_step("BASELINE_COMPARISON", "Creating baseline comparison plots...")
    step_start = time.time()
    try:
        from .plots.baselines import visualize_all_baseline_comparisons, create_metric_baseline_comparison
        
        # Create consolidated plots only - removes duplicate generation
        baseline_results = visualize_all_baseline_comparisons()
        
        # Handle different return types properly
        baseline_paths = []
        if isinstance(baseline_results, dict):
            # Extract paths from dictionary
            for key, path in baseline_results.items():
                if path and Path(path).exists():
                    baseline_paths.append(Path(path))
        elif isinstance(baseline_results, list):
            baseline_paths = [Path(p) for p in baseline_results if p and Path(p).exists()]
        elif baseline_results:
            # Single path
            if Path(baseline_results).exists():
                baseline_paths = [Path(baseline_results)]
        
        # Also check for baseline plots in statistical_tests directory
        stat_test_dir = visualization_dir / "statistical_tests"
        if stat_test_dir.exists():
            baseline_stat_files = list(stat_test_dir.glob("baseline_*.png"))
            baseline_paths.extend(baseline_stat_files)
        
        # Remove duplicates
        baseline_paths = list(set(baseline_paths))
        
        all_visualizations['baseline_comparison'] = baseline_paths
        viz_times['baseline_comparison'] = time.time() - step_start
        log_viz_step("BASELINE_COMPARISON", f"Created {len(baseline_paths)} baseline comparison plots")
        
        # Validate files exist
        for path in baseline_paths:
            if not path.exists():
                log_viz_step("BASELINE_COMPARISON", f"Warning: Expected file not found: {path}", is_error=True)
                
    except Exception as e:
        log_viz_step("BASELINE_COMPARISON", f"Error: {e}", is_error=True)
        logging.exception("Full traceback:")
        traceback.print_exc()
        all_visualizations['baseline_comparison'] = []
    
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
    actual_file_counts = {}
    
    for viz_type, paths in all_visualizations.items():
        if isinstance(paths, dict):
            # Handle dictionary of paths
            count = 0
            actual_files = []
            for key, p in paths.items():
                if isinstance(p, list):
                    # Validate each path exists
                    for path in p:
                        if isinstance(path, (str, Path)) and Path(path).exists():
                            actual_files.append(path)
                            count += 1
                elif isinstance(p, (str, Path)) and Path(p).exists():
                    actual_files.append(p)
                    count += 1
                elif p is not None and hasattr(p, 'savefig'):
                    # It's a Figure object - assume it was saved
                    count += 1
            actual_file_counts[viz_type] = count
            
        elif isinstance(paths, list):
            # Handle list of paths or figures
            count = 0
            actual_files = []
            for p in paths:
                if isinstance(p, (str, Path)) and Path(p).exists():
                    actual_files.append(p)
                    count += 1
                elif p is not None and hasattr(p, 'savefig'):
                    # It's a Figure object
                    count += 1
            actual_file_counts[viz_type] = count
            
        else:
            # Handle single path or object
            if isinstance(paths, (str, Path)) and Path(paths).exists():
                count = 1
            elif paths and hasattr(paths, 'savefig'):
                count = 1
            else:
                count = 0
            actual_file_counts[viz_type] = count
            
        total_plots += count
        
        # Get timing if available
        time_str = f" ({viz_times.get(viz_type, 0):.1f}s)" if viz_type in viz_times else ""
        
        # Add validation warning if count is 0 but time > 0
        if count == 0 and viz_times.get(viz_type, 0) > 1.0:
            message = f"{viz_type:.<30} {count:>4} plots{time_str} ⚠️ (files may not be tracked)"
        else:
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
        ('Statistical tests', len(all_visualizations.get('statistical_tests', [])) > 0)
    ]
    
    logging.info("\nCritical plot checks:")
    print("\nCritical plot checks:")
    for check_name, exists in critical_checks:
        status = "✓" if exists else "✗"
        logging.info(f"  {status} {check_name}")
        print(f"  {status} {check_name}")
    
    # Track and report any visualization types that completely failed
    failed_types = []
    for viz_type, paths in all_visualizations.items():
        if not paths or (isinstance(paths, dict) and not any(paths.values())):
            if viz_times.get(viz_type, 0) > 0:  # Had time but no output
                failed_types.append(viz_type)
    
    if failed_types:
        print(f"\n⚠️  Warning: {len(failed_types)} visualization types failed completely:")
        logging.warning(f"Failed visualization types: {', '.join(failed_types)}")
        for ft in failed_types:
            print(f"  - {ft}")
    
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