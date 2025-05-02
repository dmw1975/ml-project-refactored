"""Format utilities for visualizations."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path

def format_figure_for_export(
    fig: Figure,
    title: Optional[str] = None,
    title_fontsize: int = 14,
    tight_layout: bool = True,
    constrained_layout: bool = False,
    pad: float = 1.1,
    theme: str = 'default'
) -> Figure:
    """
    Format figure for export.
    
    Args:
        fig: Figure to format
        title: Title for figure
        title_fontsize: Font size for title
        tight_layout: Whether to use tight layout
        constrained_layout: Whether to use constrained layout
        pad: Padding for tight layout
        theme: Theme to apply ('default', 'publication', 'presentation')
        
    Returns:
        Figure: Formatted figure
    """
    # Add title if provided
    if title is not None:
        fig.suptitle(title, fontsize=title_fontsize)
    
    # Apply layout
    if tight_layout and not constrained_layout:
        fig.tight_layout(pad=pad, rect=[0, 0, 1, 0.95] if title else None)
    elif constrained_layout and not tight_layout:
        fig.set_constrained_layout(True)
        if title:
            fig.set_constrained_layout_pads(h_pad=0.05, w_pad=0.05, hspace=0.05, wspace=0.05)
    
    # Apply theme
    if theme == 'publication':
        # Publication quality theme
        for ax in fig.get_axes():
            # Set font sizes
            ax.title.set_fontsize(12)
            ax.xaxis.label.set_fontsize(10)
            ax.yaxis.label.set_fontsize(10)
            ax.tick_params(axis='both', labelsize=8)
            
            # Set grid style
            ax.grid(alpha=0.3, linestyle='--')
            
            # Set spine visibility
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
    elif theme == 'presentation':
        # Presentation quality theme
        for ax in fig.get_axes():
            # Set font sizes
            ax.title.set_fontsize(16)
            ax.xaxis.label.set_fontsize(14)
            ax.yaxis.label.set_fontsize(14)
            ax.tick_params(axis='both', labelsize=12)
            
            # Set grid style
            ax.grid(alpha=0.3, linestyle='-')
            
            # Set line widths
            for line in ax.get_lines():
                line.set_linewidth(2.5)
            
            # Set spine width
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
    
    return fig

def save_figure(
    fig: Figure,
    filename: str,
    output_dir: Union[str, Path],
    dpi: int = 300,
    format: str = 'png',
    transparent: bool = False,
    facecolor: Optional[str] = None,
    bbox_inches: str = 'tight'
) -> Path:
    """
    Save figure to file.
    
    Args:
        fig: Figure to save
        filename: Filename without extension
        output_dir: Output directory
        dpi: Dots per inch
        format: File format
        transparent: Whether to use transparent background
        facecolor: Face color
        bbox_inches: Bounding box
        
    Returns:
        Path: Path to saved file
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add extension if not present
    if not filename.endswith(f'.{format}'):
        filename = f"{filename}.{format}"
    
    # Create path
    filepath = output_dir / filename
    
    # Save figure
    fig.savefig(
        filepath,
        dpi=dpi,
        format=format,
        transparent=transparent,
        facecolor=facecolor,
        bbox_inches=bbox_inches
    )
    
    return filepath