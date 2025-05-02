"""Layout utilities for visualizations."""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, Any, Optional, Union, List, Tuple, Callable

def create_grid_layout(
    nrows: int,
    ncols: int,
    figsize: Optional[Tuple[float, float]] = None,
    height_ratios: Optional[List[float]] = None,
    width_ratios: Optional[List[float]] = None,
    hspace: float = 0.3,
    wspace: float = 0.3,
    suptitle: Optional[str] = None,
    suptitle_fontsize: int = 16
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create a grid layout of subplots.
    
    Args:
        nrows: Number of rows
        ncols: Number of columns
        figsize: Figure size (width, height)
        height_ratios: Ratios of heights
        width_ratios: Ratios of widths
        hspace: Horizontal space between subplots
        wspace: Vertical space between subplots
        suptitle: Super title for figure
        suptitle_fontsize: Font size for super title
        
    Returns:
        Tuple[plt.Figure, List[plt.Axes]]: Figure and list of axes
    """
    # Calculate figure size if not provided
    if figsize is None:
        base_size = 4
        figsize = (base_size * ncols, base_size * nrows)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Create grid specification
    gs = gridspec.GridSpec(
        nrows=nrows,
        ncols=ncols,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=hspace,
        wspace=wspace
    )
    
    # Create list of axes
    axes = []
    for i in range(nrows):
        row_axes = []
        for j in range(ncols):
            ax = fig.add_subplot(gs[i, j])
            row_axes.append(ax)
        axes.append(row_axes)
    
    # Flatten axes list
    axes_flat = [ax for row in axes for ax in row]
    
    # Add super title if provided
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=suptitle_fontsize)
    
    return fig, axes_flat

def create_comparison_layout(
    n_items: int,
    n_metrics: int,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    title_fontsize: int = 16
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create a layout for comparing multiple items by multiple metrics.
    
    Args:
        n_items: Number of items to compare
        n_metrics: Number of metrics to compare
        figsize: Figure size (width, height)
        title: Title for figure
        title_fontsize: Font size for title
        
    Returns:
        Tuple[plt.Figure, List[plt.Axes]]: Figure and list of axes
    """
    # Calculate figure size if not provided
    if figsize is None:
        base_height = 3
        base_width = 5
        figsize = (base_width * n_metrics, base_height * (n_items / 4 + 1))
    
    # Create figure
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    # Ensure axes is a list
    if n_metrics == 1:
        axes = [axes]
    
    # Add title if provided
    if title is not None:
        fig.suptitle(title, fontsize=title_fontsize)
    
    # Add space between subplots
    plt.tight_layout(rect=[0, 0, 1, 0.95] if title else None)
    
    return fig, axes

def create_dashboard_layout(
    n_plots: int,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    title_fontsize: int = 18,
    plot_creators: Optional[List[Callable]] = None
) -> plt.Figure:
    """
    Create a dashboard layout with multiple plots.
    
    Args:
        n_plots: Number of plots
        figsize: Figure size (width, height)
        title: Title for dashboard
        title_fontsize: Font size for title
        plot_creators: List of plot creator functions
        
    Returns:
        plt.Figure: Dashboard figure
    """
    # Calculate number of rows and columns
    if n_plots <= 2:
        nrows, ncols = 1, n_plots
    elif n_plots <= 4:
        nrows, ncols = 2, 2
    elif n_plots <= 6:
        nrows, ncols = 2, 3
    elif n_plots <= 9:
        nrows, ncols = 3, 3
    else:
        nrows = (n_plots + 3) // 4  # Ceiling division
        ncols = 4
    
    # Calculate figure size if not provided
    if figsize is None:
        base_size = 5
        figsize = (base_size * ncols, base_size * nrows)
    
    # Create figure
    fig, axes = create_grid_layout(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        hspace=0.4,
        wspace=0.4,
        suptitle=title,
        suptitle_fontsize=title_fontsize
    )
    
    # Create plots if creators provided
    if plot_creators is not None:
        for i, creator in enumerate(plot_creators):
            if i < len(axes):
                creator(axes[i])
    
    # Hide unused axes
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    return fig