# New Visualization Architecture

This document provides information about the new visualization architecture implemented in the `visualization_new` package.

## Overview

The new visualization architecture is designed to be model-agnostic, modular, and extensible. It provides a standardized way to create visualizations for different model types (XGBoost, LightGBM, CatBoost, ElasticNet, etc.) without duplicating code.

## Directory Structure

```
visualization_new/
├── __init__.py
├── adapters/
│   ├── __init__.py
│   ├── xgboost_adapter.py
│   ├── lightgbm_adapter.py
│   └── catboost_adapter.py
├── core/
│   ├── __init__.py
│   ├── interfaces.py
│   ├── base.py
│   ├── registry.py
│   └── style.py
├── plots/
│   ├── __init__.py
│   ├── residuals.py
│   ├── features.py
│   └── metrics.py
├── components/
│   ├── __init__.py
│   ├── annotations.py
│   ├── layouts.py
│   └── formats.py
├── utils/
│   ├── __init__.py
│   ├── data_prep.py
│   ├── statistics.py
│   └── io.py
└── viz_factory.py
```

## Key Components

### Core

- **interfaces.py**: Defines the `ModelData` interface for standardized model data access and `VisualizationConfig` for configuring visualizations.
- **base.py**: Contains base classes for visualizations (`BaseViz`, `ModelViz`, `ComparativeViz`).
- **registry.py**: Provides a registry for model adapters to support dynamic adapter selection.
- **style.py**: Centralized styling utilities for consistent visualization appearance.

### Adapters

Model-specific adapters that convert model data into the standardized `ModelData` interface:

- **xgboost_adapter.py**: Adapter for XGBoost models
- **lightgbm_adapter.py**: Adapter for LightGBM models
- **catboost_adapter.py**: Adapter for CatBoost models

### Plots

Topic-based visualization modules:

- **residuals.py**: Residual analysis visualizations
- **features.py**: Feature importance visualizations
- **metrics.py**: Performance metrics visualizations

### Components

Reusable visualization components:

- **annotations.py**: Text annotations and statistical labels
- **layouts.py**: Layout utilities for creating multi-plot figures
- **formats.py**: Formatting utilities for consistent styling

### Utils

Utility functions:

- **data_prep.py**: Data preparation functions
- **statistics.py**: Statistical analysis functions
- **io.py**: Input/output utilities

### Factory

- **viz_factory.py**: Provides a simple interface for creating visualizations

## Usage

### Basic Usage

```python
import visualization_new as viz

# Create residual plot for a specific model
viz.create_residual_plot('XGB_Base_optuna')

# Create all residual plots
viz.create_all_residual_plots()

# Create feature importance plot
viz.create_feature_importance_plot('LightGBM_Base_optuna')

# Create model comparison plot
from visualization_new.utils.io import load_all_models
models = load_all_models()
viz.create_model_comparison_plot(list(models.values()))

# Create comprehensive dashboard
viz.create_comparative_dashboard()
```

### Using Command Line

```bash
# Generate visualizations using the new architecture
python main.py --visualize-new

# Run specific model training and visualization
python main.py --train-xgboost --visualize-new

# Run the test script for the new architecture
python test_new_visualization.py
```

## Extending the Architecture

### Adding a New Model Type

1. Create a new adapter in the `adapters/` directory (e.g., `new_model_adapter.py`)
2. Implement the adapter class by extending `ModelData` or implementing the required interface
3. Register the adapter in the registry module

### Adding a New Visualization Type

1. Create a new module in the `plots/` directory (e.g., `new_plot_type.py`)
2. Implement visualization classes extending `BaseViz` or other base classes
3. Add factory functions in `viz_factory.py` to expose the new visualizations

## Benefits

1. **Modularity**: Clear separation of concerns with dedicated modules
2. **Reusability**: Common components are reused across different visualizations
3. **Consistency**: Standardized interfaces and styling
4. **Extensibility**: Easy to add support for new model types or visualizations
5. **Maintainability**: Reduced code duplication and improved organization

## Comparison with Old Architecture

The old visualization architecture had several limitations:

1. **Model-specific code**: Separate modules for each model type (xgboost_plots.py, lightgbm_plots.py, etc.)
2. **Code duplication**: Similar visualization code repeated across files
3. **Inconsistent interfaces**: Different parameters and return values across functions
4. **Limited reusability**: Difficult to extend or reuse components

The new architecture addresses these issues with a modular, component-based design that separates concerns and promotes code reuse.

## Future Improvements

1. Add more adapters for different model types
2. Implement more visualization types
3. Add interactive visualizations
4. Create a dashboard interface
5. Improve documentation with examples