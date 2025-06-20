"""
Adapter bridge utilities for handling conversion between adapter and dictionary formats.

This module provides interface compatibility between visualization code that expects
raw model dictionaries and the adapter pattern used throughout the pipeline.
"""

import logging
from typing import Any, Dict, Union
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_model_dict(model_data: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    """
    Convert adapter or dict to consistent dict format.
    
    Args:
        model_data: Either a model adapter object or a dictionary
        
    Returns:
        Dictionary containing model data
    """
    logger.debug(f"ensure_model_dict: Input type: {type(model_data)}, "
                f"has adapter methods: {hasattr(model_data, 'get_raw_model_data')}")
    
    if hasattr(model_data, 'get_raw_model_data'):
        # It's an adapter
        logger.debug(f"Converting adapter to dict for model: "
                    f"{getattr(model_data, 'model_name', 'unknown')}")
        result = model_data.get_raw_model_data()
        logger.debug(f"Conversion successful, dict has {len(result)} keys")
        return result
    
    # It's already a dict
    logger.debug(f"Input is already a dict with {len(model_data)} keys")
    return model_data


def ensure_adapter(model_data: Union[Dict[str, Any], Any]) -> Any:
    """
    Convert dict or adapter to consistent adapter format.
    
    Args:
        model_data: Either a dictionary or a model adapter object
        
    Returns:
        Model adapter object
    """
    logger.debug(f"ensure_adapter: Input type: {type(model_data)}, "
                f"has adapter methods: {hasattr(model_data, 'get_raw_model_data')}")
    
    if not hasattr(model_data, 'get_raw_model_data'):
        # It's a dict, convert to adapter
        logger.debug(f"Converting dict to adapter for model: "
                    f"{model_data.get('model_name', 'unknown')}")
        from src.visualization.core.registry import get_adapter_for_model
        adapter = get_adapter_for_model(model_data)
        logger.debug(f"Conversion successful, adapter type: {type(adapter).__name__}")
        return adapter
    
    # It's already an adapter
    logger.debug(f"Input is already an adapter of type: {type(model_data).__name__}")
    return model_data


def extract_output_dir(config: Union[Dict[str, Any], Any]) -> Path:
    """
    Extract output directory from various config formats.
    
    Args:
        config: Configuration object (dict, object with attributes, or Path)
        
    Returns:
        Path object for output directory
    """
    from src.config import settings
    
    logger.debug(f"extract_output_dir: Config type: {type(config)}")
    
    # Handle dictionary format
    if isinstance(config, dict):
        output_dir = config.get('output_dir', settings.VISUALIZATION_DIR / 'default')
        logger.debug(f"Extracted from dict: {output_dir}")
        return Path(output_dir)
    
    # Handle object with output_dir attribute
    elif hasattr(config, 'output_dir'):
        output_dir = getattr(config, 'output_dir')
        logger.debug(f"Extracted from object attribute: {output_dir}")
        return Path(output_dir)
    
    # Handle direct Path object
    elif isinstance(config, (Path, str)):
        logger.debug(f"Config is already a path: {config}")
        return Path(config)
    
    # Fallback
    else:
        logger.warning(f"Unknown config format: {type(config)}, using default")
        return settings.VISUALIZATION_DIR / 'default'


def get_model_type(model_name: str) -> str:
    """
    Extract model type from model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model type (catboost, lightgbm, xgboost, elasticnet, linear)
    """
    name_lower = model_name.lower()
    
    if 'catboost' in name_lower:
        return 'catboost'
    elif 'lightgbm' in name_lower:
        return 'lightgbm'
    elif 'xgboost' in name_lower:
        return 'xgboost'
    elif 'elasticnet' in name_lower:
        return 'elasticnet'
    elif name_lower.startswith('lr_'):
        return 'linear'
    else:
        return 'unknown'


# Test commands for this module:
# python -c "
# from src.visualization.utils.adapter_bridge import ensure_model_dict, ensure_adapter
# test_dict = {'model_name': 'test', 'model': 'dummy'}
# result = ensure_model_dict(test_dict)
# print('Dict test:', 'PASSED' if result == test_dict else 'FAILED')
# "