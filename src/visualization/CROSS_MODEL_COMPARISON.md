# Cross-Model Feature Importance Comparison

This document explains the approach used for comparing feature importance across different model types.

## Key Changes

1. **Tree-based Models Only**: Cross-model feature comparisons now include only tree-based models (XGBoost, LightGBM, CatBoost) to ensure fair and consistent comparison.

2. **Ranking-based Comparison**: Instead of using raw importance values which vary in scale across model types, we now use feature ranks (1 = most important).

## Rationale for Changes

### Excluding Linear Models (ElasticNet)

Linear models like ElasticNet calculate feature importance using coefficient magnitudes, which:
- Have a fundamentally different statistical basis than tree-based importance
- Required extreme scaling (by 10,000,000×) to appear comparable
- Created misleading comparisons despite scaling

By focusing on tree-based models only, we ensure that all compared models calculate importance using conceptually similar approaches (based on information gain or prediction improvement).

### Using Feature Ranks Instead of Raw Values

Ranks provide several advantages:
- **Model-agnostic**: Works across different model types regardless of their underlying importance scales
- **No arbitrary scaling needed**: Eliminates the need for artificial scaling factors
- **Intuitive interpretation**: "Feature X is the 3rd most important in all models" is immediately meaningful
- **Focus on practical decisions**: Highlights which features should be prioritized

## Visualization Outputs

For each dataset type (Base, Yeo, Base_Random, Yeo_Random), the system generates:

1. **Average Rank Plot**: Shows features with the lowest average rank (most important) across all tree-based models
   - File: `average_feature_rank_{dataset}.png`
   - Lower rank = more important feature

2. **Rank Heatmap**: Detailed view of each feature's rank in different models
   - File: `feature_rank_heatmap_{dataset}.png`
   - Color intensity indicates importance (darker = more important)
   - Numbers show exact rank of each feature in each model

## Interpretation Guidelines

When analyzing these visualizations:

1. **Lower rank = more important**: A feature with rank 1 is the most important
2. **Consistency matters**: Features that rank highly across all models are most reliable
3. **Random feature check**: In *_Random datasets, the random feature should rank last (high number) in all models
4. **Rank spread**: Look for features with consistent ranks across models

This approach provides a more statistically sound and interpretable way to compare feature importance across different tree-based model types.