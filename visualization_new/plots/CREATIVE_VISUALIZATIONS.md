# Creative Dataset Comparison Visualizations

This document explains the different creative visualization approaches implemented in `creative_dataset_comparison.py` for comparing model performance across different datasets.

## Visualization Types

### 1. Radar/Spider Charts

**File:** `create_radar_chart()`

**Description:**
Radar charts display multiple quantitative variables on axes starting from the same point. Each model is represented as a polygon with vertices at the normalized metric values.

**Key Features:**
- Metrics are normalized to a 0-1 scale (with appropriate inversion for metrics where lower is better)
- Line styles differentiate between basic and tuned models
- Colors represent different model families
- Direction indicators (↑/↓) show whether higher or lower values are better
- Multiple datasets displayed in a grid layout

**Best Used For:**
- Comparing overall profiles of models across multiple metrics
- Quickly identifying strengths and weaknesses of different models
- Visual pattern matching for model performance signatures

### 2. Performance Delta Visualizations

**File:** `create_performance_delta()`

**Description:**
Bar charts showing the percentage improvement from basic to tuned versions of models for each metric.

**Key Features:**
- Positive values indicate improvement
- Grouped by model family for easy comparison
- Percentage calculation considers whether higher or lower is better for each metric
- Direct indication of tuning effectiveness

**Best Used For:**
- Assessing the value of tuning for different model families
- Identifying which models benefit most from hyperparameter optimization
- Quantifying the improvement from basic to tuned models

### 3. Parallel Coordinates Plots

**File:** `create_parallel_coordinates()`

**Description:**
A parallel coordinates plot shows each model as a line passing through vertical axes representing different metrics.

**Key Features:**
- Normalized axes so higher is always better (scaled and inverted as needed)
- Line styles differentiate between basic and tuned models
- Colors represent different model families
- Metrics arranged in a logical order

**Best Used For:**
- Visualizing trade-offs between different metrics
- Identifying patterns and correlations between metrics
- Showing model performance across all metrics simultaneously

### 4. Visual Leaderboards

**File:** `create_visual_leaderboard()`

**Description:**
An enhanced bar chart that ranks models for each metric and provides visual indicators of performance.

**Key Features:**
- Horizontal bars for easy comparison
- Medal indicators for top performers
- Rank badges for each model
- Sorted by performance for each metric
- Colors indicate model family

**Best Used For:**
- Ranking models on individual metrics
- Highlighting the best performers for each metric
- Providing an intuitive visual comparison

### 5. Sunburst Charts

**File:** `create_sunburst_chart()`

**Description:**
A hierarchical visualization that shows the relationship between datasets, model families, and models with size indicating performance.

**Key Features:**
- Interactive HTML output with hover information
- Hierarchical organization from datasets to model families to individual models
- Size of segments indicates performance (larger = better)
- Colors represent model families

**Best Used For:**
- Hierarchical exploration of model performance
- Interactive analysis of the model ecosystem
- Understanding the relationship between datasets, model families, and individual models

## Usage

These visualizations can be generated using the included test script:

```bash
python test_creative_visualizations.py
```

This will create all visualization types for each dataset and save them to the configured output directory.

## Implementation Notes

1. All visualizations normalize metrics appropriately so that higher values are better for display consistency
2. For error metrics (RMSE, MAE, MSE), values are inverted during normalization
3. Consistent color scheme is used across all visualizations for model families
4. Line styles consistently differentiate between basic and tuned models
5. Both static and interactive (for sunburst) versions are saved

## Example Output

Output files follow this naming convention:
- `{dataset}_radar_chart.png`
- `{dataset}_performance_delta.png`
- `{dataset}_parallel_coordinates.png`
- `{dataset}_visual_leaderboard.png`
- `dataset_model_sunburst_{metric}.html` (interactive)
- `dataset_model_sunburst_{metric}.png` (static)