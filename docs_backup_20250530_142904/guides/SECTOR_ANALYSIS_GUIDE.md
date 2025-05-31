# Sector Analysis Guide

This document explains how to run sector analysis in the ML project.

## Overview

Sector analysis in this project involves training sector-specific linear regression models for ESG score prediction. The project supports both training and evaluation of these sector models, as well as various visualizations to analyze their performance.

## Running Sector Analysis

There are several ways to run sector analysis:

### 1. Using the standalone test script

```bash
python test_run_sector_training.py
```

This script will:
- Train all sector-specific models
- Evaluate the models
- Save metrics to a CSV file

### 2. Using the test_sector_models.py script

```bash
python test_sector_models.py
```

This script runs sector model functionality tests:
- Training sector models with test parameters
- Evaluating sector models
- Running feature importance analysis
- Testing sector model visualizations

### 3. Using the main.py script with sector-specific flags

```bash
# Run the entire sector model pipeline
python main.py --all-sector

# Run only sector models, skipping standard models
python main.py --sector-only

# Train sector-specific models
python main.py --train-sector

# Evaluate sector-specific models
python main.py --evaluate-sector

# Generate sector-specific visualizations using legacy visualization module
python main.py --visualize-sector

# Generate sector-specific visualizations using new architecture
python main.py --visualize-sector-new
```

### 4. Direct visualization with sector_vis_test.py

```bash
python sector_vis_test.py
```

This script only runs the sector visualizations.

## Key Components

### Models

The sector model implementation is in `models/sector_models.py` with three main functions:

1. `run_sector_models()` - Trains linear regression models separately for each GICS sector
2. `evaluate_sector_models()` - Evaluates sector models and calculates metrics
3. `analyze_sector_importance()` - Analyzes feature importance for sector-specific models

### Visualizations

Sector visualizations are available in two modules:

1. Legacy implementation: `visualization/sector_plots.py`
   - Contains the deprecated but working visualization functions
   - Creates comparison plots, heatmaps, and metrics tables

2. New architecture: `visualization_new/plots/sectors.py`
   - Model-agnostic implementation using the new visualization framework
   - Creates similar plots with improved architecture

## Output

After running sector analysis, the following output is produced:

1. Trained models:
   - Saved to `MODEL_DIR` as `sector_models.pkl`

2. Metrics:
   - Saved to `METRICS_DIR` as `sector_model_metrics.csv`

3. Feature importance:
   - Saved to `FEATURE_IMPORTANCE_DIR` as `sector_feature_importance.pkl`
   - Individual sector importance files saved as `sector_<SectorName>_importance.csv`
   - Random feature stats saved as `sector_random_feature_stats.csv`
   - Consolidated importance table saved as `sector_consolidated_importance.csv`

4. Visualizations:
   - Saved to `VISUALIZATION_DIR/sectors/`
   - Includes performance comparisons, heatmaps, and metrics tables

## Example Workflow

For a complete sector analysis workflow:

```bash
# Train, evaluate, and visualize sector models with the new architecture
python main.py --train-sector --evaluate-sector --visualize-sector-new
```

This will:
1. Train sector-specific models
2. Evaluate their performance
3. Analyze feature importance
4. Generate visualizations using the new architecture