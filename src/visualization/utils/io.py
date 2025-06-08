"""I/O utilities for visualization."""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Set

def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """
    Ensure directory exists.
    
    Args:
        dir_path: Directory path
        
    Returns:
        Path: Directory path
    """
    dir_path = Path(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def load_all_models() -> Dict[str, Dict[str, Any]]:
    """
    Load all models from model directory.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of model data
    """
    # Add project root to path if needed
    project_root = Path(__file__).parent.parent.parent.absolute()
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
        
    # Import io and settings
    from src.utils import io
    from src.config import settings
    
    # Load models from each file
    all_models = {}
    model_files = [
        'linear_regression_models.pkl',
        'elasticnet_models.pkl',
        'xgboost_models.pkl',
        'lightgbm_models.pkl',
        'catboost_models.pkl'
    ]
    
    for filename in model_files:
        try:
            models = io.load_model(filename, settings.MODEL_DIR)
            print(f"Loaded {len(models)} models from {filename}")
            all_models.update(models)
        except Exception as e:
            print(f"Error loading models from {filename}: {e}")
    
    return all_models

def find_files_by_pattern(pattern: str, directory: Union[str, Path]) -> List[Path]:
    """
    Find files matching pattern in directory.
    
    Args:
        pattern: File pattern (e.g., "*.py")
        directory: Directory to search
        
    Returns:
        List[Path]: List of matching files
    """
    directory = Path(directory)
    return list(directory.glob(pattern))

def get_model_filenames() -> Dict[str, str]:
    """
    Get mapping of model types to filenames.
    
    Returns:
        Dict[str, str]: Mapping of model types to filenames
    """
    return {
        'linear_regression': 'linear_regression_models.pkl',
        'elasticnet': 'elasticnet_models.pkl',
        'xgboost': 'xgboost_models.pkl',
        'lightgbm': 'lightgbm_models.pkl',
        'catboost': 'catboost_models.pkl'
    }

def load_model_data(model_name: str) -> Dict[str, Any]:
    """
    Load specific model data.
    
    Args:
        model_name: Model name
        
    Returns:
        Dict[str, Any]: Model data
    """
    # Add project root to path if needed
    project_root = Path(__file__).parent.parent.parent.absolute()
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
        
    # Import io and settings
    from src.utils import io
    from src.config import settings
    
    # Determine model type
    model_type = None
    for key in get_model_filenames().keys():
        if key in model_name.lower():
            model_type = key
            break
    
    if model_type is None:
        raise ValueError(f"Cannot determine model type for {model_name}")
    
    # Load model
    filename = get_model_filenames()[model_type]
    models = io.load_model(filename, settings.MODEL_DIR)
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found in {filename}")
    
    # Return model data
    model_data = models[model_name]
    model_data['model_name'] = model_name
    
    return model_data

def save_visualization(fig, filename: str, output_dir: Union[str, Path], 
                      dpi: int = 300, format: str = 'png') -> Path:
    """
    Save visualization to file.
    
    Args:
        fig: Matplotlib figure
        filename: Filename (without extension)
        output_dir: Output directory
        dpi: DPI for raster formats
        format: File format
        
    Returns:
        Path: Output path
    """
    # Ensure directory exists
    output_dir = ensure_dir(output_dir)
    
    # Create output path
    output_path = output_dir / f"{filename}.{format}"
    
    # Save figure
    fig.savefig(output_path, dpi=dpi, format=format, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    
    return output_path