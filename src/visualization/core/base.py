"""Base classes for visualizations."""

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

from src.visualization.core.interfaces import ModelData, VisualizationConfig


class BaseViz(ABC):
    """Base class for all visualizations."""
    
    def __init__(self, config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None):
        """
        Initialize base visualization.
        
        Args:
            config: Visualization configuration
        """
        if config is None:
            self.config = VisualizationConfig()
        elif isinstance(config, dict):
            self.config = VisualizationConfig(**config)
        else:
            self.config = config
            
        # Setup figure
        self._setup_figure()
            
    def _setup_figure(self) -> None:
        """Setup figure based on configuration."""
        from src.visualization.core.style import setup_visualization_style
        
        # Apply style
        self.style = setup_visualization_style(self.config.get('style', 'whitegrid'))
        
        # Update rcParams
        plt.rcParams.update({
            'font.size': self.config.get('label_fontsize', 12)
        })
        
    @abstractmethod
    def plot(self) -> plt.Figure:
        """
        Create the visualization.
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        pass
    
    def save(self, fig: plt.Figure, filename: str) -> Path:
        """
        Save figure to file.
        
        Args:
            fig: Figure to save
            filename: Filename without extension
            
        Returns:
            Path: Path to saved file
        """
        from src.visualization.utils.io import ensure_dir
        
        # Ensure output directory exists
        output_dir = self.config.get('output_dir')
        if output_dir is not None:
            output_dir = Path(output_dir)
            ensure_dir(output_dir)
        else:
            # Default output directory
            from pathlib import Path
            import sys
            
            # Add project root to path if needed
            project_root = Path(__file__).parent.parent.parent.absolute()
            if str(project_root) not in sys.path:
                sys.path.append(str(project_root))
                
            # Import settings
            from src.config import settings
            
            # Use default from settings
            output_dir = settings.VISUALIZATION_DIR
            ensure_dir(output_dir)
        
        # File extension
        ext = self.config.get('format', 'png')
        
        # Full path
        filepath = output_dir / f"{filename}.{ext}"
        
        # Save figure
        fig.savefig(
            filepath,
            dpi=self.config.get('dpi', 300),
            bbox_inches='tight',
            format=ext
        )
        
        print(f"Saved {filepath}")
        return filepath
    
    def show(self, fig: plt.Figure) -> None:
        """
        Show figure if configured.
        
        Args:
            fig: Figure to show
        """
        if self.config.get('show', False):
            plt.show()
        else:
            plt.close(fig)


class ModelViz(BaseViz):
    """Base class for single model visualizations."""
    
    def __init__(
        self, 
        model_data: ModelData, 
        config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
    ):
        """
        Initialize model visualization.
        
        Args:
            model_data: Model data
            config: Visualization configuration
        """
        super().__init__(config)
        self.model_data = model_data
        
    @abstractmethod
    def plot(self) -> plt.Figure:
        """
        Create the visualization for a single model.
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        pass


class ComparativeViz(BaseViz):
    """Base class for comparative visualizations."""
    
    def __init__(
        self, 
        models: List[ModelData], 
        config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
    ):
        """
        Initialize comparative visualization.
        
        Args:
            models: List of model data
            config: Visualization configuration
        """
        super().__init__(config)
        self.models = models
        
    @abstractmethod
    def plot(self) -> plt.Figure:
        """
        Create the comparative visualization.
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        pass