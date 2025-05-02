"""Reusable annotation components for visualizations."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Dict, Any, Optional, Union, List, Tuple

def add_metrics_text(
    fig: plt.Figure,
    metrics: Dict[str, float],
    position: Tuple[float, float] = (0.5, 0.01),
    fontsize: int = 10,
    bbox_props: Optional[Dict[str, Any]] = None
) -> None:
    """
    Add metrics text to figure.
    
    Args:
        fig: Figure to add text to
        metrics: Metrics dictionary
        position: Position (x, y) in figure coordinates
        fontsize: Font size
        bbox_props: Box properties
    """
    # Create text
    lines = []
    
    # Add each metric
    for name, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{name}: {value:.4f}")
        else:
            lines.append(f"{name}: {value}")
    
    # Join lines
    text = "\n".join(lines)
    
    # Default box properties
    if bbox_props is None:
        bbox_props = {
            'boxstyle': 'round',
            'facecolor': 'white',
            'alpha': 0.9
        }
    
    # Add text
    fig.text(
        position[0],
        position[1],
        text,
        ha='center',
        fontsize=fontsize,
        bbox=bbox_props
    )

def add_statistics_text(
    ax: Axes,
    statistics: Dict[str, float],
    position: Tuple[float, float] = (0.05, 0.95),
    fontsize: int = 10,
    bbox_props: Optional[Dict[str, Any]] = None,
    transform: str = 'axes'
) -> None:
    """
    Add statistics text to axes.
    
    Args:
        ax: Axes to add text to
        statistics: Statistics dictionary
        position: Position (x, y) in specified coordinates
        fontsize: Font size
        bbox_props: Box properties
        transform: Coordinate system ('axes', 'figure', 'data')
    """
    # Create text
    lines = []
    
    # Add each statistic
    for name, value in statistics.items():
        if isinstance(value, float):
            lines.append(f"{name}: {value:.4f}")
        else:
            lines.append(f"{name}: {value}")
    
    # Join lines
    text = "\n".join(lines)
    
    # Default box properties
    if bbox_props is None:
        bbox_props = {
            'boxstyle': 'round',
            'facecolor': 'white',
            'alpha': 0.7
        }
    
    # Determine transform
    if transform == 'axes':
        transform_obj = ax.transAxes
    elif transform == 'figure':
        transform_obj = ax.figure.transFigure
    else:  # data
        transform_obj = ax.transData
    
    # Add text
    ax.text(
        position[0],
        position[1],
        text,
        transform=transform_obj,
        verticalalignment='top',
        fontsize=fontsize,
        bbox=bbox_props
    )

def add_value_labels(
    ax: Axes,
    precision: int = 2,
    fontsize: int = 8,
    color: str = 'black',
    vertical_offset: float = 0.01
) -> None:
    """
    Add value labels to bars in bar chart.
    
    Args:
        ax: Axes with bar chart
        precision: Number of decimal places
        fontsize: Font size
        color: Text color
        vertical_offset: Vertical offset as fraction of bar height
    """
    # For each bar in the chart
    for rect in ax.patches:
        # Get height
        height = rect.get_height()
        
        # Skip if height is 0 or NaN
        if height == 0 or np.isnan(height):
            continue
        
        # Add label
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            height + (abs(height) * vertical_offset),
            f"{height:.{precision}f}",
            ha='center',
            va='bottom',
            fontsize=fontsize,
            color=color
        )

def add_significance_stars(
    ax: Axes,
    x: float,
    y: float,
    p_value: float,
    fontsize: int = 12,
    color: str = 'black'
) -> None:
    """
    Add significance stars based on p-value.
    
    Args:
        ax: Axes to add stars to
        x: X position
        y: Y position
        p_value: P-value
        fontsize: Font size
        color: Text color
    """
    # Determine stars
    if p_value < 0.001:
        stars = '***'
    elif p_value < 0.01:
        stars = '**'
    elif p_value < 0.05:
        stars = '*'
    else:
        stars = 'ns'
    
    # Add text
    ax.text(
        x, y, stars,
        ha='center',
        va='center',
        fontsize=fontsize,
        color=color
    )