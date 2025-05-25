"""Visualization factory for easy access to visualization functions."""

from pathlib import Path
import pandas as pd
import sys
from typing import Dict, List, Union, Any, Optional, Tuple

from visualization_new.core.interfaces import ModelData, VisualizationConfig
from visualization_new.core.registry import get_adapter_for_model, load_model
from visualization_new.plots.residuals import plot_residuals, plot_all_residuals
from visualization_new.plots.features import plot_feature_importance, plot_feature_importance_comparison
from visualization_new.plots.metrics import plot_metrics, plot_metrics_table, plot_model_comparison
from visualization_new.plots.sectors import plot_sector_performance, plot_sector_metrics_table, visualize_all_sector_plots
from visualization_new.plots.dataset_comparison import plot_dataset_comparison, create_all_dataset_comparisons
from visualization_new.plots.statistical_tests import plot_statistical_tests, visualize_statistical_tests
from visualization_new.plots.optimization import (
    plot_optimization_history, plot_param_importance, plot_contour,
    plot_hyperparameter_comparison, plot_basic_vs_optuna, plot_optuna_improvement,
    plot_all_optimization_visualizations
)
from visualization_new.utils.io import load_all_models
from visualization_new.adapters.elasticnet_adapter import ElasticNetAdapter


def get_visualization_dir(model_name: str, plot_type: str) -> Path:
    """
    Return standardized directory path for visualizations.
    
    Args:
        model_name: Name of the model
        plot_type: Type of visualization (features, residuals, performance, etc.)
        
    Returns:
        Path: Path to the visualization directory
    """
    # Add project root to path if needed
    project_root = Path(__file__).parent.parent.absolute()
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
        
    # Import settings
    from config import settings
    
    # For residuals, use the top-level residuals directory without model-specific subdirectories
    if plot_type == "residuals":
        output_dir = settings.VISUALIZATION_DIR / plot_type
    else:
        # Extract base model type (e.g., "catboost" from "CatBoost_Base_basic")
        # Handle model name format for various models consistently
        if "xgb" in model_name.lower():
            # All XGBoost models go to xgboost folder
            model_type = 'xgboost'
        elif "lightgbm" in model_name.lower():
            model_type = 'lightgbm'
        elif "catboost" in model_name.lower():
            model_type = 'catboost'
        elif "elasticnet" in model_name.lower():
            # All ElasticNet models go to elasticnet folder
            model_type = 'elasticnet'
        elif "lr_" in model_name.lower():
            # Linear Regression models also go to elasticnet folder for consistency
            model_type = 'elasticnet'
        else:
            # For any other model, skip directory creation to avoid unwanted folders
            print(f"Warning: Unknown model type for '{model_name}' - skipping directory creation")
            return None

        # Create and return the path - using standardized model type, not raw model name
        output_dir = settings.VISUALIZATION_DIR / plot_type / model_type
    
    # Ensure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def create_residual_plot(
    model_data: Union[str, Dict[str, Any], ModelData],
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> Any:
    """
    Create residual analysis plot for a model.
    
    Args:
        model_data: Model name, data dictionary, or ModelData object
        config: Visualization configuration
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Handle string model name
    if isinstance(model_data, str):
        model_data = load_model(model_data)
    
    return plot_residuals(model_data, config)


def create_all_residual_plots(
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> List[Any]:
    """
    Create residual analysis plots for all models.
    
    Args:
        config: Visualization configuration
        
    Returns:
        List[matplotlib.figure.Figure]: List of created figures
    """
    return plot_all_residuals(config=config)


def create_feature_importance_plot(
    model_data: Union[str, Dict[str, Any], ModelData],
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> Any:
    """
    Create feature importance plot for a model.
    
    Args:
        model_data: Model name, data dictionary, or ModelData object
        config: Visualization configuration
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Handle string model name
    if isinstance(model_data, str):
        model_data = load_model(model_data)
    
    return plot_feature_importance(model_data, config)


def create_feature_importance_comparison(
    models: List[Union[str, Dict[str, Any], ModelData]],
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> Any:
    """
    Create feature importance comparison plot.
    
    Args:
        models: List of model names, data dictionaries, or ModelData objects
        config: Visualization configuration
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Handle string model names
    model_list = []
    for model in models:
        if isinstance(model, str):
            model_list.append(load_model(model))
        else:
            model_list.append(model)
    
    return plot_feature_importance_comparison(model_list, config)


def create_model_comparison_plot(
    models: List[Union[str, Dict[str, Any], ModelData]],
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> Any:
    """
    Create model comparison plot.
    
    Args:
        models: List of model names, data dictionaries, or ModelData objects
        config: Visualization configuration
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Handle string model names
    model_list = []
    for model in models:
        if isinstance(model, str):
            model_list.append(load_model(model))
        else:
            model_list.append(model)
    
    return plot_metrics(model_list, config)


def create_metrics_table(
    models: List[Union[str, Dict[str, Any], ModelData]],
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> Any:
    """
    Create metrics summary table.
    
    Args:
        models: List of model names, data dictionaries, or ModelData objects
        config: Visualization configuration
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Handle string model names
    model_list = []
    for model in models:
        if isinstance(model, str):
            model_list.append(load_model(model))
        else:
            model_list.append(model)
    
    return plot_metrics_table(model_list, config)


def create_comparative_dashboard(
    models: Optional[List[Union[str, Dict[str, Any], ModelData]]] = None,
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> Any:
    """
    Create comprehensive dashboard with multiple plots.
    
    Args:
        models: List of model names, data dictionaries, or ModelData objects (if None, use all models)
        config: Visualization configuration
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    import matplotlib.pyplot as plt
    from visualization_new.components.layouts import create_grid_layout
    
    # Use all models if not specified
    if models is None:
        all_models = load_all_models()
        model_list = list(all_models.values())
    else:
        # Handle string model names
        model_list = []
        for model in models:
            if isinstance(model, str):
                model_list.append(load_model(model))
            else:
                model_list.append(model)
    
    # Default configuration
    if config is None:
        config = VisualizationConfig(
            title="Model Performance Dashboard",
            figsize=(16, 12)
        )
    elif isinstance(config, dict):
        config = VisualizationConfig(**config)
    
    # Create dashboard layout with 2x2 grid
    fig, axes = create_grid_layout(
        nrows=2,
        ncols=2,
        figsize=config.get("figsize", (16, 12)),
        suptitle=config.get("title", "Model Performance Dashboard"),
        suptitle_fontsize=16
    )
    
    # 1. Model Comparison (top left)
    try:
        from visualization_new.plots.metrics import plot_model_comparison
        ax1 = axes[0]
        model_comparison_fig = plot_model_comparison(model_list)
        ax1.figure = model_comparison_fig
        ax1.set_title("Model Performance Comparison")
    except Exception as e:
        print(f"Error creating model comparison: {e}")
    
    # 2. Feature Importance (top right)
    try:
        from visualization_new.plots.features import plot_feature_importance_comparison
        ax2 = axes[1]
        feature_fig = plot_feature_importance_comparison(model_list)
        ax2.figure = feature_fig
        ax2.set_title("Feature Importance Comparison")
    except Exception as e:
        print(f"Error creating feature importance: {e}")
    
    # 3. Residual Analysis (bottom left)
    try:
        from visualization_new.plots.residuals import plot_residuals
        ax3 = axes[2]
        if model_list:
            residual_fig = plot_residuals(model_list[0])
            ax3.figure = residual_fig
            ax3.set_title(f"Residual Analysis: {model_list[0].model_name}")
    except Exception as e:
        print(f"Error creating residual plot: {e}")
    
    # 4. Metrics Table (bottom right)
    try:
        from visualization_new.plots.metrics import plot_metrics_table
        ax4 = axes[3]
        table_fig = plot_metrics_table(model_list)
        ax4.figure = table_fig
        ax4.set_title("Metrics Summary")
    except Exception as e:
        print(f"Error creating metrics table: {e}")
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig


def create_optimization_history_plot(
    model_data: Union[str, Dict[str, Any], ModelData],
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> Optional[str]:
    """
    Create optimization history plot for a model.
    
    Args:
        model_data: Model name, data dictionary, or ModelData object
        config: Visualization configuration
        
    Returns:
        Optional[str]: Path to the saved visualization or None if not saved
    """
    # Handle string model name
    if isinstance(model_data, str):
        model_data = load_model(model_data)
    
    # Get model metadata and study object
    study = None
    model_name = None
    
    if isinstance(model_data, ModelData):
        # Use adapter methods
        model_name = model_data.get_metadata().get('model_name', 'Unknown Model')
        # Check if the adapter has get_study method
        if hasattr(model_data, 'get_study'):
            study = model_data.get_study()
        # If not, try to get it from raw model data
        elif hasattr(model_data, 'get_raw_model_data'):
            raw_data = model_data.get_raw_model_data()
            study = raw_data.get('study', None)
    else:
        # Direct dictionary access
        model_name = model_data.get('model_name', 'Unknown Model')
        study = model_data.get('study', None)
    
    if study is None:
        print(f"No optimization study found for {model_name}")
        return None
    
    return plot_optimization_history(study, config, model_name)


def create_param_importance_plot(
    model_data: Union[str, Dict[str, Any], ModelData],
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> Optional[str]:
    """
    Create parameter importance plot for a model.
    
    Args:
        model_data: Model name, data dictionary, or ModelData object
        config: Visualization configuration
        
    Returns:
        Optional[str]: Path to the saved visualization or None if not saved
    """
    # Handle string model name
    if isinstance(model_data, str):
        model_data = load_model(model_data)
    
    # Get model metadata and study object
    study = None
    model_name = None
    
    if isinstance(model_data, ModelData):
        # Use adapter methods
        model_name = model_data.get_metadata().get('model_name', 'Unknown Model')
        # Check if the adapter has get_study method
        if hasattr(model_data, 'get_study'):
            study = model_data.get_study()
        # If not, try to get it from raw model data
        elif hasattr(model_data, 'get_raw_model_data'):
            raw_data = model_data.get_raw_model_data()
            study = raw_data.get('study', None)
    else:
        # Direct dictionary access
        model_name = model_data.get('model_name', 'Unknown Model')
        study = model_data.get('study', None)
    
    if study is None:
        print(f"No optimization study found for {model_name}")
        return None
    
    return plot_param_importance(study, config, model_name)


def create_all_optimization_visualizations(
    model_data: Union[str, Dict[str, Any], ModelData],
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> Dict[str, str]:
    """
    Create all optimization-related visualizations for a model.
    
    Args:
        model_data: Model name, data dictionary, or ModelData object
        config: Visualization configuration
        
    Returns:
        Dict[str, str]: Dictionary of visualization paths
    """
    # Handle string model name
    if isinstance(model_data, str):
        model_data = load_model(model_data)
    
    # Get model metadata and study object
    study = None
    model_name = None
    
    if isinstance(model_data, ModelData):
        # Use adapter methods
        model_name = model_data.get_metadata().get('model_name', 'Unknown Model')
        # Check if the adapter has get_study method
        if hasattr(model_data, 'get_study'):
            study = model_data.get_study()
        # If not, try to get it from raw model data
        elif hasattr(model_data, 'get_raw_model_data'):
            raw_data = model_data.get_raw_model_data()
            study = raw_data.get('study', None)
    else:
        # Direct dictionary access
        model_name = model_data.get('model_name', 'Unknown Model')
        study = model_data.get('study', None)
        
    # Get raw model data for plot_all_optimization_visualizations
    raw_data = model_data
    if isinstance(model_data, ModelData) and hasattr(model_data, 'get_raw_model_data'):
        raw_data = model_data.get_raw_model_data()
    
    return plot_all_optimization_visualizations(raw_data, config)


def create_hyperparameter_comparison(
    models: List[Union[str, Dict[str, Any], ModelData]],
    parameter_name: str,
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None,
    model_family: str = "xgboost"
) -> Optional[str]:
    """
    Create hyperparameter comparison plot across models.
    
    Args:
        models: List of model names, data dictionaries, or ModelData objects
        parameter_name: Name of the hyperparameter to compare
        config: Visualization configuration
        model_family: Model family name (xgboost, lightgbm, catboost, elasticnet)
        
    Returns:
        Optional[str]: Path to the saved visualization or None if not saved
    """
    # Handle string model names
    model_list = []
    for model in models:
        if isinstance(model, str):
            model_list.append(load_model(model))
        elif isinstance(model, ModelData) and hasattr(model, 'get_raw_model_data'):
            model_list.append(model.get_raw_model_data())
        else:
            model_list.append(model)
    
    return plot_hyperparameter_comparison(model_list, parameter_name, config, model_family)


def create_basic_vs_optuna_comparison(
    models: List[Union[str, Dict[str, Any], ModelData]],
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None,
    model_family: str = "xgboost"
) -> Optional[str]:
    """
    Create comparison of basic vs. tuned models.
    
    Args:
        models: List of model names, data dictionaries, or ModelData objects
        config: Visualization configuration
        model_family: Model family name (xgboost, lightgbm, catboost, elasticnet)
        
    Returns:
        Optional[str]: Path to the saved visualization or None if not saved
    """
    # Handle string model names
    model_list = []
    for model in models:
        if isinstance(model, str):
            model_list.append(load_model(model))
        elif isinstance(model, ModelData) and hasattr(model, 'get_raw_model_data'):
            model_list.append(model.get_raw_model_data())
        else:
            model_list.append(model)
    
    return plot_basic_vs_optuna(model_list, config, model_family)


def create_optuna_improvement_plot(
    models: List[Union[str, Dict[str, Any], ModelData]],
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None,
    model_family: str = "xgboost"
) -> Optional[str]:
    """
    Create plot showing improvement from Optuna optimization.
    
    Args:
        models: List of model names, data dictionaries, or ModelData objects
        config: Visualization configuration
        model_family: Model family name (xgboost, lightgbm, catboost, elasticnet)
        
    Returns:
        Optional[str]: Path to the saved visualization or None if not saved
    """
    # Handle string model names
    model_list = []
    for model in models:
        if isinstance(model, str):
            model_list.append(load_model(model))
        elif isinstance(model, ModelData) and hasattr(model, 'get_raw_model_data'):
            model_list.append(model.get_raw_model_data())
        else:
            model_list.append(model)
    
    return plot_optuna_improvement(model_list, config, model_family)


def visualize_model(
    model_data: Union[str, Dict[str, Any], ModelData],
    output_dir: Optional[Union[str, Path]] = None,
    format: str = 'png',
    dpi: int = 300,
    show: bool = False
) -> Dict[str, Path]:
    """
    Create comprehensive visualizations for a model.
    
    Args:
        model_data: Model name, data dictionary, or ModelData object
        output_dir: Output directory
        format: File format
        dpi: Dots per inch
        show: Whether to show plots
        
    Returns:
        Dict[str, Path]: Dictionary of plot types and file paths
    """
    # Close any existing figures to avoid conflicts
    import matplotlib.pyplot as plt
    plt.close('all')
    
    # Handle string model name
    if isinstance(model_data, str):
        model_data = load_model(model_data)
    
    # Get model metadata
    if isinstance(model_data, ModelData):
        metadata = model_data.get_metadata()
        model_name = metadata.get('model_name', 'Unknown Model')
    else:
        model_name = model_data.get('model_name', 'Unknown Model')
    
    # Set up output directory
    if output_dir is None:
        # Default output directory
        from pathlib import Path
        import sys
        
        # Add project root to path if needed
        project_root = Path(__file__).parent.parent.absolute()
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))
            
        # Import settings
        from config import settings
        
        # Use type-based directory structure instead of model-name based
        # This avoids creating unwanted directories
        # Extract model type from model name and use the specific type directory
        model_type = model_name.lower().split('_')[0]
        
        # Map to standard directory names
        if 'catboost' in model_type:
            model_dir = 'catboost'
        elif 'lightgbm' in model_type:
            model_dir = 'lightgbm'
        elif 'xgb' in model_type:
            model_dir = 'xgboost'
        elif 'elasticnet' in model_type or 'lr' in model_type:
            # Route all ElasticNet and Linear Regression models to the elasticnet folder
            model_dir = 'elasticnet'
        else:
            # Skip creating directories for unknown types
            print(f"Warning: Unknown model type '{model_type}' - skipping visualization")
            return {}
            
        output_dir = settings.VISUALIZATION_DIR / "performance" / model_dir
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = VisualizationConfig(
        output_dir=output_dir,
        format=format,
        dpi=dpi,
        show=show
    )
    
    # Create plots
    plots = {}
    
    # Residual plot
    try:
        # Use residuals directory
        residuals_dir = get_visualization_dir(model_name, "residuals")
        if residuals_dir is not None:
            # Update config with the new directory
            residual_config = VisualizationConfig(
                output_dir=residuals_dir,
                format=format,
                dpi=dpi,
                show=show
            )
            residual_fig = create_residual_plot(model_data, residual_config)
            # For CatBoost models, simplify the filename
            if model_name.lower().startswith('catboost'):
                # Extract variant like 'base_basic' from 'CatBoost_Base_basic'
                parts = model_name.split('_', 1)
                variant = parts[1] if len(parts) > 1 else 'default'
                plots['residual'] = residuals_dir / f"residuals_{variant.lower()}.{format}"
            else:
                plots['residual'] = residuals_dir / f"{model_name.lower()}_residuals.{format}"
    except Exception as e:
        print(f"Error creating residual plot: {e}")
    
    # Feature importance plot
    try:
        # Use features directory
        features_dir = get_visualization_dir(model_name, "features")
        if features_dir is not None:
            # Update config with the new directory
            feature_config = VisualizationConfig(
                output_dir=features_dir,
                format=format,
                dpi=dpi,
                show=show
            )
            feature_fig = create_feature_importance_plot(model_data, feature_config)
            # For CatBoost models, simplify the filename
            if model_name.lower().startswith('catboost'):
                # Extract variant like 'base_basic' from 'CatBoost_Base_basic'
                parts = model_name.split('_', 1)
                variant = parts[1] if len(parts) > 1 else 'default'
                plots['feature_importance'] = features_dir / f"top_features_{variant.lower()}.{format}"
            else:
                plots['feature_importance'] = features_dir / f"{model_name.lower()}_top_features.{format}"
    except Exception as e:
        print(f"Error creating feature importance plot: {e}")
    
    # Optimization plots (if applicable)
    if 'optuna' in model_name.lower():
        try:
            # Create optimization visualizations
            optimization_plots = create_all_optimization_visualizations(model_data, config)
            plots.update(optimization_plots)
        except Exception as e:
            print(f"Error creating optimization plots: {e}")
    
    return plots


def create_cross_model_feature_importance_by_dataset(
    format: str = 'png',
    dpi: int = 300,
    show: bool = False,
    tree_based_only: bool = True  # New parameter to exclude linear models
) -> Dict[str, Path]:
    """
    Create feature importance comparisons across model types, grouped by dataset.
    
    This function generates visualizations that show which features are most important
    across different model types (LightGBM, XGBoost, CatBoost) for each dataset type 
    (Base, Yeo, Base_Random, Yeo_Random). By default, only tree-based models are included
    to ensure consistent and comparable feature importance metrics.
    
    Args:
        format: File format
        dpi: Dots per inch
        show: Whether to show plots
        tree_based_only: If True, include only tree-based models (XGBoost, LightGBM, CatBoost)
        
    Returns:
        Dict[str, Path]: Dictionary of dataset names and file paths
    """
    # Set up output directory
    from config import settings
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create a dedicated directory for cross-model feature importance - directly in features dir
    # No longer using subdirectory "cross_model_comparison" to avoid empty directories
    output_dir = settings.VISUALIZATION_DIR / "features"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all models
    all_models = load_all_models()
    
    # Group models by dataset
    dataset_groups = {}
    
    # Import ModelFamily for dataset extraction
    from visualization_new.core.model_family import ModelFamily, get_model_info
    
    # Process each model
    for model_name, model_data in all_models.items():
        # Add model_name to model_data if not present
        if 'model_name' not in model_data:
            model_data['model_name'] = model_name
            
        # Extract dataset name and model family
        family = ModelFamily.from_model_name(model_name)
        dataset = family.get_dataset_from_model_name(model_name)
        model_family_type = family.model_type
        
        # Skip unknown datasets
        if dataset == 'Unknown':
            continue
            
        # Filter out non-tree-based models if requested
        if tree_based_only and model_family_type not in ['XGBoost', 'LightGBM', 'CatBoost']:
            print(f"Skipping {model_name} (type: {model_family_type}) - not a tree-based model")
            continue
        
        # Add to dataset groups
        if dataset not in dataset_groups:
            dataset_groups[dataset] = []
        
        # Create adapter and add to group
        try:
            adapter = get_adapter_for_model(model_data)
            dataset_groups[dataset].append((model_name, adapter))
        except Exception as e:
            print(f"Error creating adapter for {model_name}: {e}")
    
    # Create feature importance comparison for each dataset group
    result_paths = {}
    
    for dataset, model_adapters in dataset_groups.items():
        try:
            if not model_adapters:
                print(f"No models found for dataset: {dataset}")
                continue
            
            # Don't create dataset-specific subdirectories as they're often empty
            # Instead, use a naming convention in the main features directory
            dataset_dir = output_dir
            
            # Extract feature importance from each model
            feature_data = {}
            model_names = []
            model_types = []
            all_features = set()
            
            # Debug: Log all model adapters in this dataset group
            print(f"\nProcessing dataset group: {dataset}")
            print(f"Number of models in this group: {len(model_adapters)}")
            for i, (model_name, _) in enumerate(model_adapters):
                print(f"  {i+1}. {model_name}")
            
            for model_name, adapter in model_adapters:
                # Get model info for colors and type
                info = get_model_info(model_name)
                model_type = info['family']
                
                # Get a simplified model name for display
                if 'optuna' in model_name.lower():
                    display_name = f"{model_type} (Tuned)"
                else:
                    display_name = f"{model_type} (Basic)"
                
                model_names.append(display_name)
                model_types.append(model_type)
                
                # Get feature importance
                try:
                    # Debug: Show pre-importance extraction info
                    print(f"\nProcessing model: {model_name}")
                    print(f"  Display name: {display_name}")
                    print(f"  Model type: {model_type}")
                    
                    # Get raw importance values
                    importance_df = adapter.get_feature_importance()
                    
                    # Debug: Show original importance stats
                    print(f"  Original importance values: min={importance_df['Importance'].min()}, max={importance_df['Importance'].max()}, mean={importance_df['Importance'].mean()}")
                    
                    # Convert to rankings instead of raw importance values
                    # Sort by importance and assign ranks (1 = most important)
                    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
                    importance_df['Rank'] = importance_df.index + 1
                    
                    print(f"  Converted to feature rankings (1 = most important)")
                    print(f"  Top 5 features: {', '.join(importance_df['Feature'].head(5).tolist())}")
                    
                    # Create a dictionary mapping features to their rank
                    feature_dict = dict(zip(importance_df['Feature'], importance_df['Rank']))
                    feature_data[display_name] = feature_dict
                    all_features.update(importance_df['Feature'])
                    
                    # Debug: Confirm model added to feature_data
                    print(f"  ✓ Added {display_name} to feature_data with {len(feature_dict)} features")
                except Exception as e:
                    print(f"  ✗ Error extracting feature importance from {model_name}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Create consolidated DataFrame for ranks
            all_features_list = list(all_features)
            df = pd.DataFrame(index=all_features_list)
            
            # Add rank for each model (using a high rank as default for missing features)
            max_possible_rank = len(all_features) + 1  # One more than total features
            for model_name, rank_dict in feature_data.items():
                df[model_name] = df.index.map(lambda x: rank_dict.get(x, max_possible_rank))
            
            # Add average rank column (lower rank = more important)
            df['Average Rank'] = df.mean(axis=1)
            
            # Sort by average rank (ascending=True because lower rank = more important)
            df = df.sort_values('Average Rank')
            
            # Select top features by rank
            top_n = 20
            top_features = df.head(top_n)
            
            # Create figure for average rank across models
            plt.figure(figsize=(14, 10))
            bars = plt.barh(
                top_features.index[::-1],  # Reverse to have highest at the top
                top_features['Average Rank'][::-1],
                color='#3498db',
                alpha=0.7
            )
            
            # Add value labels (lower rank is better)
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(
                    width + 0.1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.1f}",
                    va='center',
                    fontsize=10
                )
            
            # Set plot properties
            plt.xlabel('Average Rank Across Models (lower is better)', fontsize=12)
            plt.title(f"Top {top_n} Features by Average Rank - {dataset} Dataset (Tree-based Models Only)", fontsize=14)
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            # Save average rank figure
            avg_rank_path = dataset_dir / f"average_feature_rank_{dataset.lower()}.{format}"
            plt.savefig(avg_rank_path, dpi=dpi, format=format, bbox_inches='tight')
            plt.close()
            
            # Create heatmap for detailed model comparison based on ranks
            plt.figure(figsize=(16, 12))
            
            # Drop the Average Rank column for the heatmap
            heatmap_data = top_features.drop('Average Rank', axis=1)
            
            # Create a custom colormap from blue to white
            # For ranks, darker/more intense color = better rank (lower number)
            cmap = LinearSegmentedColormap.from_list(
                'BlueWhite', [(0.2, 0.6, 0.9), (1, 1, 1)], N=100
            )
            
            # Create the heatmap with rank values (integer format)
            sns.heatmap(
                heatmap_data,
                cmap=cmap,
                annot=True,
                fmt='d',  # Integer format for ranks
                linewidths=0.5,
                vmin=1,  # Force scale to start at rank 1
                vmax=min(50, len(all_features))  # Cap at 50 or total features, whichever is smaller
            )
            
            # Set plot properties
            plt.title(f'Feature Ranking Across Tree-based Models - {dataset} Dataset (lower is better)', fontsize=16)
            plt.xlabel('Model Type', fontsize=14)
            plt.ylabel('Features', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save heatmap figure
            heatmap_path = dataset_dir / f"feature_rank_heatmap_{dataset.lower()}.{format}"
            plt.savefig(heatmap_path, dpi=dpi, format=format, bbox_inches='tight')
            plt.close()
            
            # Store the paths
            result_paths[dataset] = {
                'average_rank': avg_rank_path,
                'heatmap': heatmap_path
            }
            
            print(f"Saved cross-model feature rank comparison for {dataset} dataset:")
            print(f"  - Average rank: {avg_rank_path}")
            print(f"  - Rank heatmap: {heatmap_path}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error creating feature importance comparison for dataset {dataset}: {e}")
    
    return result_paths


def visualize_all_models(
    output_dir: Optional[Union[str, Path]] = None,
    format: str = 'png',
    dpi: int = 300,
    show: bool = False
) -> Dict[str, Dict[str, Path]]:
    """
    Create comprehensive visualizations for all models.
    
    Args:
        output_dir: Output directory
        format: File format
        dpi: Dots per inch
        show: Whether to show plots
        
    Returns:
        Dict[str, Dict[str, Path]]: Dictionary of model names and plot paths
    """
    # Load all models
    all_models = load_all_models()
    
    # Create visualizations for each model
    model_plots = {}
    
    for model_name, model_data in all_models.items():
        try:
            # Let visualize_model handle the directory structure consistently
            plots = visualize_model(
                model_data=model_data,
                output_dir=None,  # Let visualize_model use type-based directories
                format=format,
                dpi=dpi,
                show=show
            )
            model_plots[model_name] = plots
        except Exception as e:
            print(f"Error visualizing model {model_name}: {e}")
    
    # Create comparison plots
    try:
        # Place comparison plots directly in the performance directory (not in a subdirectory)
        performance_dir = settings.VISUALIZATION_DIR / "performance"
        performance_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        config = VisualizationConfig(
            output_dir=performance_dir,
            format=format,
            dpi=dpi,
            show=show
        )
        
        # Create comparison plots
        model_list = list(all_models.values())
        
        # Metrics comparison plot - commented out to prevent creating model_metrics_comparison.png
        # metrics_fig = create_model_comparison_plot(model_list, config)
        
        # Metrics table
        table_fig = create_metrics_table(model_list, config)
        
        # Feature importance comparison
        feature_fig = create_feature_importance_comparison(model_list, config)
        
        # Add to plots - use standardized naming and place directly in performance directory
        # Only include the metrics table, not the model metrics comparison
        model_plots['comparison'] = {
            # Commented out to prevent creating model_metrics_comparison.png
            # 'metrics': performance_dir / f"comparison_model_metrics.{format}",
            'table': performance_dir / f"comparison_metrics_table.{format}",
            'feature_importance': performance_dir / f"comparison_top_features.{format}"
        }
    except Exception as e:
        print(f"Error creating comparison plots: {e}")
    
    return model_plots