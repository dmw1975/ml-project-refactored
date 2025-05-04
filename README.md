# ML Project Refactored

Machine Learning Project for ESG Score Prediction

## Overview

This project implements various machine learning models to predict ESG scores, including:

- XGBoost
- LightGBM
- CatBoost
- ElasticNet
- Linear Regression

## Project Structure

- `data/`: Data preprocessing and loading
- `models/`: Model implementations
- `evaluation/`: Model evaluation metrics
- `visualization_new/`: New visualization architecture
- `visualization/`: Legacy visualization (deprecated)
- `utils/`: Utility functions

## Visualization System

The project uses a modern, component-based visualization architecture that supports multiple model types. Key features:

- **Model Adapters**: Adapts different model types to a common interface
- **Type-based Directory Structure**: Organizes outputs by visualization type and model
- **Standardized API**: Consistent interface for all visualization functions

For detailed information about the visualization system:
- [README_NEW_VIZ.md](README_NEW_VIZ.md): Overview of the new visualization architecture
- [visualization_new/DIRECTORY_STRUCTURE.md](visualization_new/DIRECTORY_STRUCTURE.md): Directory structure documentation

## Usage

### Running Models

```bash
# Train and evaluate all models
python main.py --train --evaluate

# Train specific models
python main.py --train-xgboost --train-lightgbm

# Evaluate models
python main.py --evaluate

# Generate visualizations
python main.py --visualize
```

### Visualizations

```bash
# Generate all visualizations
python main.py --visualize

# Generate visualizations for specific models
python main.py --visualize-xgboost
python main.py --visualize-lightgbm
python main.py --visualize-catboost
```

## Further Documentation

- [README_XGBOOST.md](README_XGBOOST.md): XGBoost model details
- [README_LIGHTGBM.md](README_LIGHTGBM.md): LightGBM model details
- [README_CATBOOST.md](README_CATBOOST.md): CatBoost model details
