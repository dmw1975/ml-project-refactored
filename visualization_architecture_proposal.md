# Visualization Architecture Proposal

This document outlines a new architecture for the visualization module that makes it more modular, maintainable, and extensible.

## 1. Directory Structure

The proposed directory structure is organized by visualization purpose rather than model type:

```
visualization/
├── __init__.py                 # Package exports
├── core/                       # Core visualization components
│   ├── __init__.py
│   ├── base.py                 # Base visualization classes
│   ├── config.py               # Configuration system
│   ├── style.py                # Enhanced styling system
│   └── registry.py             # Model registry
├── adapters/                   # Model-specific adapters
│   ├── __init__.py
│   ├── xgboost_adapter.py      # XGBoost adapter
│   ├── lightgbm_adapter.py     # LightGBM adapter
│   └── catboost_adapter.py     # CatBoost adapter
├── plots/                      # Topic-based visualization modules
│   ├── __init__.py
│   ├── residuals.py            # Residual plots for all models
│   ├── metrics.py              # Performance metrics plots
│   ├── features.py             # Feature importance plots
│   ├── comparative.py          # Cross-model comparison plots
│   └── optimization.py         # Hyperparameter optimization plots
├── components/                 # Reusable visualization components
│   ├── __init__.py
│   ├── annotations.py          # Text annotations and labels
│   ├── layouts.py              # Layout utilities
│   └── formats.py              # Export format utilities
└── utils/                      # Utility functions
    ├── __init__.py
    ├── data_prep.py            # Data preparation utilities
    └── statistics.py           # Statistical utilities for visualization
```

## 2. Key Architectural Components

### 2.1. Core Visualization System

#### Base Visualization Classes (`core/base.py`)

- `BaseViz`: Abstract base class for all visualizations
- `ModelViz`: Extension of BaseViz for model-specific visualizations
- `ComparativeViz`: Extension of BaseViz for comparing multiple models

#### Styling System (`core/style.py`)

- Enhanced version of current style.py
- Theme-based approach with consistent styling
- Support for customization via configuration

#### Configuration System (`core/config.py`)

- Configuration management for visualizations
- Default configurations that can be overridden
- Support for user-defined configuration

#### Model Registry (`core/registry.py`)

- Central registry for model types
- Automatic discovery of available visualization capabilities
- Registry of adapters for different model types

### 2.2. Adapter Layer

Adapters convert model-specific outputs to standardized formats:

- `XGBoostAdapter` (`adapters/xgboost_adapter.py`)
- `LightGBMAdapter` (`adapters/lightgbm_adapter.py`) 
- `CatBoostAdapter` (`adapters/catboost_adapter.py`)

Each adapter implements:
- Feature importance extraction
- Prediction access
- Residual calculation
- Hyperparameter access
- Training metrics access

### 2.3. Topic-Based Plots

Reorganized by visualization purpose:

- `residuals.py`: Residual analysis plots for all model types
- `metrics.py`: Performance metrics visualizations
- `features.py`: Feature importance visualizations
- `comparative.py`: Cross-model comparisons
- `optimization.py`: Hyperparameter optimization visualizations

### 2.4. Reusable Components

Common visualization components extracted for reuse:

- `annotations.py`: Reusable text annotations and labels
- `layouts.py`: Layout utilities for common plot arrangements
- `formats.py`: Export format standardization

## 3. Interfaces

### 3.1. Model Data Interface

Standardized interface for model data extraction:

```python
class ModelData:
    """Standardized interface for model data."""
    
    def get_predictions(self) -> tuple:
        """Get test set predictions and actual values."""
        pass
        
    def get_residuals(self) -> np.ndarray:
        """Get model residuals."""
        pass
        
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance data."""
        pass
        
    def get_metrics(self) -> dict:
        """Get model performance metrics."""
        pass
        
    def get_hyperparameters(self) -> dict:
        """Get model hyperparameters."""
        pass
        
    def get_metadata(self) -> dict:
        """Get model metadata."""
        pass
```

### 3.2. Visualization Configuration Interface

```python
class VizConfig:
    """Configuration interface for visualizations."""
    
    def get_style(self) -> dict:
        """Get style configuration."""
        pass
        
    def get_format(self) -> dict:
        """Get format configuration."""
        pass
        
    def get_output_dir(self) -> Path:
        """Get output directory."""
        pass
        
    def get_custom_params(self) -> dict:
        """Get custom visualization parameters."""
        pass
```

## 4. Usage Example

```python
# High-level usage
from visualization import viz_factory, models

# Create residual plots for all models with default configuration
viz_factory.create_residual_plots()

# Create feature importance plots for a specific model with custom configuration
model_data = models.load_model("XGB_Base_optuna")
viz_factory.create_feature_importance_plot(
    model_data, 
    config={"top_n": 15, "figsize": (12, 8)}
)

# Create comparative plot for multiple models
model_names = ["XGB_Base_optuna", "LightGBM_Base_optuna", "CatBoost_Base_optuna"]
viz_factory.create_model_comparison_plot(
    model_names,
    metric="RMSE"
)
```

## 5. Migration Strategy

1. Create the new directory structure
2. Implement core components and interfaces
3. Create adapters for existing model types
4. Refactor existing visualization code into topic-based modules
5. Update references to visualization in other parts of the codebase
6. Remove deprecated model-specific visualization files

## 6. Benefits

- **Topic-based organization**: Visualizations are grouped by purpose rather than model type
- **Extensibility**: New model types can be added with minimal code changes
- **Consistency**: Unified styling and interfaces across all visualizations
- **Reusability**: Common visualization components can be reused across different plot types
- **Customization**: Configuration system allows for tailored adjustments without code changes
- **Maintainability**: Decoupled components are easier to maintain and update
- **Readability**: Clear organization makes code easier to understand and navigate