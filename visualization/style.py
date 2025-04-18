"""Common styling for visualizations."""

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

def save_figure(fig, filename, directory):
    """Save a figure with standard settings."""
    path = f"{directory}/{filename}.{settings.FIGURE_FORMAT}"
    fig.savefig(path, dpi=settings.FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    return path