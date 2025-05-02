"""Enhanced styling for visualizations."""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Any, Optional, List, Tuple, Union
import sys
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
    
# Import settings
from config import settings

# Define color palettes
COLOR_PALETTES = {
    'default': {
        'primary': '#3498db',      # Blue
        'secondary': '#2ecc71',    # Green
        'tertiary': '#e67e22',     # Orange
        'quaternary': '#9b59b6',   # Purple
        'quinary': '#f1c40f',      # Yellow
        'error': '#e74c3c',        # Red
        'success': '#2ecc71',      # Green
        'warning': '#f39c12',      # Orange
        'info': '#3498db',         # Blue
        'neutral': '#95a5a6',      # Gray
    },
    'publication': {
        'primary': '#1f77b4',      # Blue
        'secondary': '#ff7f0e',    # Orange
        'tertiary': '#2ca02c',     # Green
        'quaternary': '#d62728',   # Red
        'quinary': '#9467bd',      # Purple
        'error': '#d62728',        # Red
        'success': '#2ca02c',      # Green
        'warning': '#ff7f0e',      # Orange
        'info': '#1f77b4',         # Blue
        'neutral': '#7f7f7f',      # Gray
    },
    'pastel': {
        'primary': '#a6cee3',      # Light blue
        'secondary': '#b2df8a',    # Light green
        'tertiary': '#fb9a99',     # Light red
        'quaternary': '#cab2d6',   # Light purple
        'quinary': '#ffff99',      # Light yellow
        'error': '#fb9a99',        # Light red
        'success': '#b2df8a',      # Light green
        'warning': '#fdbf6f',      # Light orange
        'info': '#a6cee3',         # Light blue
        'neutral': '#dddddd',      # Light gray
    },
    'dark': {
        'primary': '#1a5276',      # Dark blue
        'secondary': '#196f3d',    # Dark green
        'tertiary': '#af601a',     # Dark orange
        'quaternary': '#6c3483',   # Dark purple
        'quinary': '#b7950b',      # Dark yellow
        'error': '#922b21',        # Dark red
        'success': '#196f3d',      # Dark green
        'warning': '#b9770e',      # Dark orange
        'info': '#1a5276',         # Dark blue
        'neutral': '#424949',      # Dark gray
    }
}

# Define color maps
COLOR_MAPS = {
    'blue_gradient': ['#f7fbff', '#08306b'],     # Light blue to dark blue
    'green_gradient': ['#f7fcf5', '#00441b'],    # Light green to dark green
    'red_gradient': ['#fff5f0', '#67000d'],      # Light red to dark red
    'purple_gradient': ['#fcfbfd', '#3f007d'],   # Light purple to dark purple
    'orange_gradient': ['#fff7ec', '#7f2704'],   # Light orange to dark orange
    'gray_gradient': ['#f7f7f7', '#252525'],     # Light gray to dark gray
    'diverging': ['#1a9641', '#ffffbf', '#d7191c'], # Green to yellow to red
}

def create_color_map(colors: List[str], name: str = 'custom_map') -> LinearSegmentedColormap:
    """
    Create a custom color map.
    
    Args:
        colors: List of colors
        name: Name of color map
        
    Returns:
        matplotlib.colors.LinearSegmentedColormap: Custom color map
    """
    return LinearSegmentedColormap.from_list(name, colors)

def setup_visualization_style(style_name: str = 'whitegrid') -> Dict[str, Any]:
    """
    Set up visualization style.
    
    Args:
        style_name: Name of style
        
    Returns:
        dict: Style configuration
    """
    # Apply style
    plt.style.use('ggplot')
    sns.set_style(style_name)
    
    # Get color palette from settings if available
    palette_name = getattr(settings, 'VIZ_PALETTE', 'default')
    
    # Get color palette
    if palette_name in COLOR_PALETTES:
        colors = COLOR_PALETTES[palette_name]
    else:
        colors = COLOR_PALETTES['default']
    
    # Update with model colors from settings
    if hasattr(settings, 'MODEL_COLORS'):
        model_colors = settings.MODEL_COLORS
        colors.update(model_colors)
    
    # Create color maps for all gradients
    color_maps = {}
    for name, gradient_colors in COLOR_MAPS.items():
        color_maps[name] = create_color_map(gradient_colors, name)
    
    # Return configuration
    return {
        'colors': colors,
        'color_maps': color_maps,
        'figure_dpi': getattr(settings, 'FIGURE_DPI', 300),
        'figure_format': getattr(settings, 'FIGURE_FORMAT', 'png'),
        'figure_size_default': getattr(settings, 'FIGURE_SIZE_DEFAULT', (10, 6)),
    }

def save_figure(fig, filename, output_dir, dpi=None, format=None):
    """
    Save matplotlib figure to specified output directory.
    
    Args:
        fig: Figure to save
        filename: Filename without extension
        output_dir: Output directory
        dpi: Dots per inch
        format: File format
    
    Returns:
        Path: Path to saved file
    """
    from visualization_new.utils.io import ensure_dir
    
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Get DPI and format from settings if not specified
    if dpi is None:
        dpi = getattr(settings, 'FIGURE_DPI', 300)
    
    if format is None:
        format = getattr(settings, 'FIGURE_FORMAT', 'png')
    
    # Create path
    path = Path(output_dir) / f"{filename}.{format}"
    
    # Save figure
    fig.savefig(
        path,
        dpi=dpi,
        format=format,
        bbox_inches='tight'
    )
    
    print(f"Figure saved to {path}")
    return path