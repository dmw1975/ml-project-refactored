"""Common styling for visualizations (DEPRECATED).

This module is deprecated and will be removed in a future version.
Please use visualization_new.core.style instead.
"""

import warnings

warnings.warn(
    "This module is deprecated. Please use visualization_new.core.style instead.",
    DeprecationWarning,
    stacklevel=2
)

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from config import settings

def setup_visualization_style():
    """Set up the visualization style."""
    plt.style.use('ggplot')
    sns.set_style(settings.VIZ_STYLE)
    
    # Return common styling elements
    return {
        'colors': settings.MODEL_COLORS,
        'figure_dpi': settings.FIGURE_DPI,
        'figure_format': settings.FIGURE_FORMAT,
        'blue_cmap': create_blue_gradient_cmap()
    }

def create_blue_gradient_cmap():
    """Create a blue gradient colormap."""
    colors = ["#f7fbff", "#08306b"]  # From light blue to dark blue
    return LinearSegmentedColormap.from_list("blue_gradient", colors)

def save_figure(fig, filename, output_dir, dpi=300, format='png'):
    """
    Save matplotlib figure to specified output directory.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to save.
    filename : str
        Name of the file without extension.
    output_dir : Path
        Directory to save the figure.
    dpi : int, optional
        Dots per inch (resolution) for the saved figure. Default is 300.
    format : str, optional
        Format to save ('png', 'pdf', 'svg', etc.). Default is 'png'.
    """
    path = output_dir / f"{filename}.{format}"
    fig.savefig(path, dpi=dpi, format=format, bbox_inches='tight')
    print(f"Figure saved to {path}")
