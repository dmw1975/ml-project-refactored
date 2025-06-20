"""
Pipeline orchestrator for creating model visualizations with proper error handling.

This module provides a centralized way to create all visualizations for models
while handling the adapter/dictionary interface mismatches gracefully.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt

from src.visualization.utils.adapter_bridge import ensure_model_dict, ensure_adapter, get_model_type
from src.visualization.plots.cv_distributions import plot_cv_distribution_single
from src.visualization.plots.shap_plots import create_shap_visualizations
from src.visualization.plots.features import plot_feature_importance

logger = logging.getLogger(__name__)


def create_model_visualizations(
    model_name: str, 
    model_data: Any, 
    viz_config: Dict[str, Any]
) -> Dict[str, bool]:
    """
    Create all visualizations for a single model with proper error handling.
    
    This function handles the conversion between adapter and dictionary formats
    and creates CV distributions, SHAP plots, and feature importance plots.
    
    Args:
        model_name: Name of the model
        model_data: Model data (can be dict or adapter)
        viz_config: Base visualization configuration with 'base_dir' key
        
    Returns:
        Dictionary indicating success/failure for each visualization type
    """
    results = {
        'cv_dist': False,
        'shap': False,
        'feature_importance': False
    }
    
    # Ensure we have both dict and adapter formats
    try:
        model_dict = ensure_model_dict(model_data)
        model_adapter = ensure_adapter(model_data)
    except Exception as e:
        logger.error(f"Failed to convert model data for {model_name}: {e}")
        return results
    
    # Get model type
    model_type = get_model_type(model_name)
    base_dir = Path(viz_config.get('base_dir', './outputs/visualizations'))
    
    # CV Distribution
    if 'cv_scores' in model_dict:
        try:
            cv_config = viz_config.copy()
            cv_config['output_dir'] = base_dir / 'cv_distributions' / model_type
            cv_config['output_dir'].mkdir(parents=True, exist_ok=True)
            
            fig = plot_cv_distribution_single(model_dict, cv_config)
            if fig:
                results['cv_dist'] = True
                plt.close(fig)
                logger.info(f"Created CV distribution for {model_name}")
        except Exception as e:
            logger.error(f"CV dist failed for {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # SHAP Analysis
    try:
        shap_config = viz_config.copy()
        shap_config['output_dir'] = base_dir / 'shap' / model_type
        shap_config['output_dir'].mkdir(parents=True, exist_ok=True)
        
        # Ensure model_name is in the dict
        if 'model_name' not in model_dict:
            model_dict['model_name'] = model_name
        
        figs = create_shap_visualizations(model_dict, shap_config)
        if figs:
            results['shap'] = True
            # Close all figures - handle both dict and list return types
            if isinstance(figs, dict):
                for fig in figs.values():
                    if fig:
                        plt.close(fig)
            elif isinstance(figs, list):
                for fig in figs:
                    if fig:
                        plt.close(fig)
            logger.info(f"Created {len(figs)} SHAP plots for {model_name}")
    except Exception as e:
        logger.error(f"SHAP failed for {model_name}: {e}")
        import traceback
        traceback.print_exc()
    
    # Feature Importance
    try:
        fi_config = viz_config.copy()
        fi_config['output_dir'] = base_dir / 'features' / model_type
        fi_config['output_dir'].mkdir(parents=True, exist_ok=True)
        
        fig = plot_feature_importance(model_adapter, fi_config)
        if fig:
            results['feature_importance'] = True
            plt.close(fig)
            logger.info(f"Created feature importance plot for {model_name}")
    except Exception as e:
        logger.error(f"Feature importance failed for {model_name}: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def create_all_model_visualizations(
    models: Dict[str, Any], 
    base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, bool]]:
    """
    Create visualizations for all models in a dictionary.
    
    Args:
        models: Dictionary of model_name -> model_data
        base_config: Base configuration for visualizations
        
    Returns:
        Dictionary of model_name -> visualization results
    """
    from src.config import settings
    
    if base_config is None:
        base_config = {
            'base_dir': settings.VISUALIZATION_DIR,
            'save': True,
            'show': False,
            'format': 'png',
            'dpi': 300,
            'n_samples': 100,  # For SHAP
            'max_display': 15  # For SHAP
        }
    
    all_results = {}
    
    for model_name, model_data in models.items():
        logger.info(f"Processing visualizations for {model_name}")
        results = create_model_visualizations(model_name, model_data, base_config)
        all_results[model_name] = results
        
        # Log summary for this model
        successful = [k for k, v in results.items() if v]
        if successful:
            logger.info(f"{model_name}: Successfully created {', '.join(successful)}")
        else:
            logger.warning(f"{model_name}: No visualizations created")
    
    # Overall summary
    total_models = len(all_results)
    models_with_cv = sum(1 for r in all_results.values() if r['cv_dist'])
    models_with_shap = sum(1 for r in all_results.values() if r['shap'])
    models_with_fi = sum(1 for r in all_results.values() if r['feature_importance'])
    
    logger.info(f"\nVisualization Summary:")
    logger.info(f"  Total models: {total_models}")
    logger.info(f"  CV distributions: {models_with_cv}")
    logger.info(f"  SHAP analyses: {models_with_shap}")
    logger.info(f"  Feature importance: {models_with_fi}")
    
    return all_results


# Test command for pipeline orchestrator:
# python -c "
# from src.utils.io import load_all_models
# from src.visualization.pipeline_orchestrator import create_all_model_visualizations
# models = load_all_models()
# # Test with just CatBoost models
# catboost_models = {k: v for k, v in models.items() if 'CatBoost' in k}
# results = create_all_model_visualizations(catboost_models, {'base_dir': './test_pipeline'})
# print('Pipeline test results:', results)
# "