# Clean Architecture Migration Guide

This guide helps you migrate from the old monolithic pipeline to the new clean architecture.

## Overview

The migration process transfers your:
- **Data files** → New organized structure
- **Trained models** → New model format
- **Configuration** → YAML-based config
- **Outputs** → Organized output directories

## Step-by-Step Migration

### 1. Prepare the New Environment

```bash
cd esg_ml_clean

# Install dependencies
pip install -r requirements.txt

# Verify the setup
python -c "from src.models.registry import ModelRegistry; print('Setup OK')"
```

### 2. Test Basic Functionality

Before migrating, ensure the new pipeline works:

```bash
# Run a simple test
python example_usage.py

# Test visualization
python test_visualization.py

# List available models
./cli.py models
```

### 3. Run Data Migration

Migrate your data files first:

```bash
# Basic data migration
python test_migration.py

# This will:
# - Copy raw data files
# - Create processed datasets (base, yeo)
# - Migrate pickle files
# - Create metadata
```

### 4. Verify Data Migration

Check that data was migrated correctly:

```bash
# Check data directory
ls -la data/processed/

# You should see:
# - features_base.csv
# - features_yeo.csv  
# - targets.csv
# - metadata.json
```

### 5. Compare Old vs New Pipeline

Run the comparison to ensure consistency:

```bash
# Compare basic Linear Regression results
python compare_old_vs_new.py

# Results should be virtually identical
# Small differences (<0.001) are normal
```

### 6. Migrate Models (Optional)

If you have trained models to preserve:

```bash
# Full migration including models
python test_migration.py --models

# Check migrated models
ls -la outputs/models/*_migrated_*.joblib
```

### 7. Run Full Pipeline

Test the complete new pipeline:

```bash
# Run with migrated configuration
./cli.py run --config configs/migrated.yaml

# Or run with default config
./cli.py run
```

## Migration Components

### Data Migration

| Old Location | New Location | Description |
|-------------|--------------|-------------|
| `data/raw/*.csv` | `data/raw/*.csv` | Raw data files |
| `combined_df_for_tree_models.csv` | `data/processed/features_tree.csv` | Tree model features |
| `combined_df_for_linear_models.csv` | `data/processed/features_linear.csv` | Linear model features |
| `score.csv` | `data/processed/targets.csv` | Target values |
| `data/pkl/*.pkl` | `data/processed/pkl/*.pkl` | Pickle files |

### Model Migration

Models are converted to the new format with metadata:

```
old_model.pkl → model_migrated_timestamp.joblib
              → model_migrated_timestamp.json (metadata)
```

### Configuration Migration

Old settings are converted to YAML:

```yaml
# configs/migrated.yaml
project:
  name: "ESG ML Pipeline - Migrated"
  original_path: "/path/to/old/project"

data:
  test_size: 0.2
  random_state: 42
  stratify_by: "sector"
```

## Validation Checklist

- [ ] Data files migrated successfully
- [ ] Comparison shows identical/similar results  
- [ ] Models load and predict correctly
- [ ] Visualizations generate properly
- [ ] Pipeline runs end-to-end

## Common Issues

### 1. Import Errors

If you get import errors when comparing:
```python
# Add to comparison script
import sys
sys.path.insert(0, '/absolute/path/to/old/project')
```

### 2. Data Alignment Issues

If results differ significantly:
- Check that indices match between features and targets
- Verify the same preprocessing is applied
- Ensure consistent train/test splits

### 3. Model Loading Errors

For unsupported model types:
- Manually create wrapper classes
- Or retrain models in new pipeline

## Using Both Pipelines

You can run both pipelines side-by-side:

```bash
# Old pipeline (in old directory)
cd /path/to/old/project
python main.py

# New pipeline (in new directory)  
cd /path/to/esg_ml_clean
./cli.py run
```

## Benefits After Migration

1. **Modular Code**: No more 1,300-line main.py
2. **Easy Extensions**: Add models with decorators
3. **Better Config**: All settings in YAML
4. **Checkpointing**: Resume long runs
5. **Clean CLI**: Intuitive commands
6. **Organized Outputs**: Clear directory structure

## Next Steps

After successful migration:

1. **Update Scripts**: Modify any custom scripts to use new API
2. **Add Features**: Easily add new models/metrics/plots
3. **Document**: Update project documentation
4. **Archive**: Keep old code for reference
5. **Team Training**: Show team the new structure

## Example: Adding a New Model

With the clean architecture, adding models is simple:

```python
# src/models/my_model.py
from ..base import BaseModel
from ..registry import register_model

@register_model("my_awesome_model")
class MyModel(BaseModel):
    def fit(self, X, y):
        # Your implementation
        return self
        
    def predict(self, X):
        # Your implementation  
        return predictions
```

That's it! The model is automatically available in the pipeline.

## Support

If you encounter issues:
1. Check the log files in `outputs/logs/`
2. Run validation scripts
3. Compare with old pipeline results
4. Review migration summary JSON files