"""Visualization factory for easy access to visualization functions."""

from pathlib import Path
from typing import Dict, List, Union, Any, Optional, Tuple

from visualization_new.core.interfaces import ModelData, VisualizationConfig
from visualization_new.core.registry import get_adapter_for_model, load_model
from visualization_new.plots.residuals import plot_residuals, plot_all_residuals
from visualization_new.plots.features import plot_feature_importance, plot_feature_importance_comparison
from visualization_new.plots.metrics import plot_metrics, plot_metrics_table, plot_model_comparison
from visualization_new.plots.sectors import plot_sector_performance, plot_sector_metrics_table, visualize_all_sector_plots
from visualization_new.utils.io import load_all_models


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
        figsize=config.figsize,
        suptitle=config.title,
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
        
        # Use default from settings
        output_dir = settings.VISUALIZATION_DIR / model_name
    
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
        residual_fig = create_residual_plot(model_data, config)
        plots['residual'] = Path(output_dir) / f"{model_name}_residuals.{format}"
    except Exception as e:
        print(f"Error creating residual plot: {e}")
    
    # Feature importance plot
    try:
        feature_fig = create_feature_importance_plot(model_data, config)
        plots['feature_importance'] = Path(output_dir) / f"{model_name}_top_features.{format}"
    except Exception as e:
        print(f"Error creating feature importance plot: {e}")
    
    return plots


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
            plots = visualize_model(
                model_data=model_data,
                output_dir=output_dir / model_name if output_dir else None,
                format=format,
                dpi=dpi,
                show=show
            )
            model_plots[model_name] = plots
        except Exception as e:
            print(f"Error visualizing model {model_name}: {e}")
    
    # Create comparison plots
    try:
        # Set up comparison output directory
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
            
            # Use default from settings
            comparison_dir = settings.VISUALIZATION_DIR / "comparison"
        else:
            comparison_dir = Path(output_dir) / "comparison"
        
        # Ensure directory exists
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        config = VisualizationConfig(
            output_dir=comparison_dir,
            format=format,
            dpi=dpi,
            show=show
        )
        
        # Create comparison plots
        model_list = list(all_models.values())
        
        # Metrics comparison plot
        metrics_fig = create_model_comparison_plot(model_list, config)
        
        # Metrics table
        table_fig = create_metrics_table(model_list, config)
        
        # Feature importance comparison
        feature_fig = create_feature_importance_comparison(model_list, config)
        
        # Add to plots
        model_plots['comparison'] = {
            'metrics': comparison_dir / f"model_metrics_comparison.{format}",
            'table': comparison_dir / f"metrics_summary_table.{format}",
            'feature_importance': comparison_dir / f"top_features_avg_importance.{format}"
        }
    except Exception as e:
        print(f"Error creating comparison plots: {e}")
    
    return model_plots