# Clean Pipeline Example: Before vs After

## Current Messy Approach

### Running the pipeline now:
```bash
# Confusing, unclear what it does
python main.py --all

# Or with many flags
python main.py --train --evaluate --visualize --optimize_xgboost 100 --elasticnet_grid

# Which plots are generated? Who knows!
# Which models are trained? Check 1379 lines of code!
```

### Current Issues:
1. No idea what `--all` actually does without reading code
2. Plots missing? Add more code to main.py!
3. Want to add a new model? Edit multiple files
4. Failed halfway? Start over from beginning

## Clean Pipeline Approach

### 1. Simple, Clear Commands

```bash
# Train all models
esg-ml train

# Train specific model with optimization
esg-ml train --model xgboost --optimize

# Evaluate and visualize
esg-ml evaluate
esg-ml visualize

# Generate report
esg-ml report
```

### 2. Configuration-Driven Workflow

```yaml
# configs/experiments/full_analysis.yaml
name: "Full ESG Analysis"

data:
  features: "all"
  test_size: 0.2
  stratify: "sector"

models:
  - name: linear_regression
    datasets: ["base", "yeo"]
    
  - name: elasticnet
    datasets: ["base", "yeo"]
    optimize:
      method: "optuna"
      trials: 100
      
  - name: xgboost
    datasets: ["base", "yeo"]
    variants: ["basic", "optimized"]
    
  - name: lightgbm
    datasets: ["base", "yeo"]
    variants: ["basic", "optimized"]

evaluation:
  metrics: ["rmse", "mae", "r2"]
  baselines: ["random", "mean", "median"]
  statistical_tests: true

visualization:
  performance:
    - metrics_table
    - model_comparison
    - cv_distributions
  
  features:
    - importance_plots
    - shap_analysis
    
  diagnostics:
    - residual_plots
    - qq_plots
    
  comparisons:
    - baseline_comparisons
    - optimization_impact
```

### 3. Run Specific Experiment

```bash
# Run full analysis
esg-ml run --config configs/experiments/full_analysis.yaml

# Run only certain stages
esg-ml run --config configs/experiments/full_analysis.yaml --stages training,evaluation

# Resume from checkpoint
esg-ml run --config configs/experiments/full_analysis.yaml --resume
```

### 4. Clear Output Structure

```
outputs/
├── 2025-01-29_full_analysis/
│   ├── models/
│   │   ├── linear_regression_base.pkl
│   │   ├── elasticnet_base_optuna.pkl
│   │   ├── xgboost_base_basic.pkl
│   │   └── ...
│   │
│   ├── figures/
│   │   ├── performance/
│   │   │   ├── metrics_table.png
│   │   │   ├── model_comparison.png
│   │   │   └── cv_distributions/
│   │   ├── features/
│   │   │   ├── importance/
│   │   │   └── shap/
│   │   └── diagnostics/
│   │
│   ├── reports/
│   │   ├── summary.html
│   │   ├── detailed_analysis.pdf
│   │   └── metrics.csv
│   │
│   └── logs/
│       ├── pipeline.log
│       └── checkpoints/
```

### 5. Easy Debugging

```python
# src/pipeline/runner.py
class PipelineRunner:
    def run(self):
        for stage in self.stages:
            try:
                self.logger.info(f"Starting {stage}")
                self.run_stage(stage)
                self.save_checkpoint(stage)
                self.logger.info(f"Completed {stage}")
            except Exception as e:
                self.logger.error(f"Failed at {stage}: {e}")
                if self.config.fail_fast:
                    raise
                else:
                    self.logger.warning(f"Continuing despite error in {stage}")
```

### 6. Adding New Features

#### Before (Messy):
1. Edit main.py (find the right place among 1379 lines)
2. Add new if statement
3. Import your code
4. Hope it doesn't break something else
5. No easy way to test in isolation

#### After (Clean):
```python
# src/models/my_new_model.py
from src.models.base import BaseModel

class MyNewModel(BaseModel):
    """My new model implementation."""
    
    def fit(self, X, y):
        # Implementation
        return self
    
    def predict(self, X):
        # Implementation
        return predictions

# Register it
from src.models.registry import ModelRegistry
ModelRegistry.register("my_new_model", MyNewModel)
```

Then just add to config:
```yaml
models:
  - name: my_new_model
    datasets: ["base"]
```

### 7. Interactive Development

```python
# notebooks/experiment.ipynb
from src.pipeline import load_config, create_pipeline

# Load and modify config
config = load_config("configs/default.yaml")
config.models = ["xgboost"]  # Test single model

# Create and run pipeline
pipeline = create_pipeline(config)
results = pipeline.run()

# Analyze results interactively
results.plot_performance()
results.get_best_model()
```

### 8. Automated Testing

```bash
# Run all tests
make test

# Run specific test
pytest tests/test_models/test_xgboost.py

# Run with coverage
pytest --cov=src tests/
```

### 9. Continuous Integration

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: make install
      - name: Run tests
        run: make test
      - name: Run linting
        run: make lint
      - name: Test pipeline
        run: esg-ml run --config configs/test.yaml
```

### 10. Clear Documentation

```bash
# Generate API docs
make docs

# Serves at http://localhost:8000
make serve-docs
```

## Summary: Why This Is Better

### Current (Messy):
- 🚫 1379-line main.py
- 🚫 Hidden dependencies
- 🚫 Unclear what runs when
- 🚫 Hard to test
- 🚫 Difficult to extend

### New (Clean):
- ✅ Modular components
- ✅ Clear configuration
- ✅ Easy to understand
- ✅ Fully testable
- ✅ Simple to extend
- ✅ Production ready

### For New Developers:
```bash
# Old way: "Good luck understanding main.py!"

# New way:
git clone repo
make install
make test        # Everything works!
esg-ml --help   # Clear commands
make docs       # Comprehensive documentation
```

This clean architecture makes the codebase:
- **Understandable**: New developers productive in hours, not weeks
- **Maintainable**: Clear where to make changes
- **Extensible**: Easy to add new features
- **Reliable**: Comprehensive testing
- **Professional**: Ready for production use