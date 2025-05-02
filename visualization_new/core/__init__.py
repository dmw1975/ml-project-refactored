"""Core components for the visualization package."""

from visualization_new.core.interfaces import ModelData, VisualizationConfig
from visualization_new.core.base import BaseViz, ModelViz, ComparativeViz
from visualization_new.core.style import setup_visualization_style
from visualization_new.core.registry import get_adapter_for_model, register_adapter
