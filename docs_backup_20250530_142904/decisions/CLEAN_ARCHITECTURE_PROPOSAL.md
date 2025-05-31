# Clean Architecture Proposal for ESG ML Project

## 1. Proposed Directory Structure

```
esg-ml-pipeline/
├── README.md                    # Main documentation
├── setup.py                     # Package installation
├── requirements.txt             # Dependencies
├── Makefile                     # Common commands
├── .gitignore                  
├── .env.example                # Environment variables template
│
├── docs/                       # Documentation
│   ├── getting_started.md     # Quick start guide
│   ├── architecture.md        # System architecture
│   ├── api_reference.md       # API documentation
│   ├── model_guide.md         # Model descriptions
│   └── visualization_guide.md # Visualization reference
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── pipeline/             # Pipeline orchestration
│   │   ├── __init__.py
│   │   ├── stages.py         # Pipeline stages
│   │   ├── runner.py         # Main pipeline runner
│   │   └── config.py         # Pipeline configuration
│   │
│   ├── data/                 # Data handling
│   │   ├── __init__.py
│   │   ├── loader.py         # Data loading utilities
│   │   ├── preprocessor.py  # Feature engineering
│   │   ├── splitter.py       # Train/test splitting
│   │   └── schemas.py        # Data validation schemas
│   │
│   ├── models/               # Model implementations
│   │   ├── __init__.py
│   │   ├── base.py           # Base model interface
│   │   ├── linear.py         # Linear models
│   │   ├── tree.py           # Tree-based models
│   │   ├── ensemble.py       # Ensemble methods
│   │   └── registry.py       # Model registry
│   │
│   ├── training/             # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py        # Training orchestration
│   │   ├── optimizer.py      # Hyperparameter optimization
│   │   ├── callbacks.py      # Training callbacks
│   │   └── metrics.py        # Performance metrics
│   │
│   ├── evaluation/           # Model evaluation
│   │   ├── __init__.py
│   │   ├── evaluator.py      # Evaluation logic
│   │   ├── baselines.py      # Baseline comparisons
│   │   ├── statistical.py    # Statistical tests
│   │   └── importance.py     # Feature importance
│   │
│   ├── visualization/        # Visualization (single, clean)
│   │   ├── __init__.py
│   │   ├── plots/            # Plot generators
│   │   │   ├── performance.py
│   │   │   ├── features.py
│   │   │   ├── residuals.py
│   │   │   └── comparisons.py
│   │   ├── styles.py         # Consistent styling
│   │   └── generator.py      # Plot generation logic
│   │
│   └── utils/                # Utilities
│       ├── __init__.py
│       ├── io.py             # File I/O
│       ├── logging.py        # Logging configuration
│       └── validation.py     # Input validation
│
├── configs/                  # Configuration files
│   ├── default.yaml         # Default configuration
│   ├── models/              # Model-specific configs
│   └── experiments/         # Experiment configs
│
├── scripts/                 # Standalone scripts
│   ├── train_model.py      # Train specific model
│   ├── evaluate_all.py     # Run evaluation
│   └── generate_report.py  # Create report
│
├── tests/                   # Unit tests
│   ├── conftest.py         # Test configuration
│   ├── test_data/          # Test data processing
│   ├── test_models/        # Test models
│   ├── test_training/      # Test training
│   └── test_visualization/ # Test plots
│
├── notebooks/              # Jupyter notebooks
│   ├── exploratory/       # EDA notebooks
│   └── examples/          # Usage examples
│
└── outputs/               # Generated outputs
    ├── models/            # Trained models
    ├── figures/           # Visualizations
    ├── reports/           # Analysis reports
    └── logs/              # Execution logs
```

## 2. Key Design Principles

### 2.1 Single Responsibility
Each module has one clear purpose:
- `data/` - Only handles data operations
- `models/` - Only defines model architectures
- `training/` - Only handles training logic
- `visualization/` - Only creates plots

### 2.2 Dependency Injection
```python
# Instead of hard-coded dependencies
def train_model():
    data = load_data()  # Hard-coded
    model = XGBoost()   # Hard-coded

# Use dependency injection
def train_model(data_loader, model_factory, config):
    data = data_loader.load(config.data_path)
    model = model_factory.create(config.model_type)
```

### 2.3 Configuration-Driven
```yaml
# configs/experiments/baseline.yaml
experiment:
  name: "baseline_comparison"
  
data:
  source: "data/processed/features.csv"
  target: "esg_score"
  test_size: 0.2
  
models:
  - type: "linear_regression"
  - type: "elasticnet"
    optimize: true
    trials: 100
  
visualization:
  plots:
    - "performance_metrics"
    - "feature_importance"
    - "residuals"
```

### 2.4 Clear Pipeline Stages
```python
# src/pipeline/stages.py
from enum import Enum

class PipelineStage(Enum):
    DATA_LOADING = "data_loading"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    EVALUATION = "evaluation"
    VISUALIZATION = "visualization"
    REPORTING = "reporting"

class Pipeline:
    def __init__(self, config):
        self.stages = self._build_stages(config)
    
    def run(self, stages=None):
        """Run specified stages or all stages"""
        for stage in self.stages:
            if stages is None or stage.name in stages:
                stage.execute()
```

## 3. Clean Command Interface

### Makefile Commands
```makefile
# Simple, clear commands
install:
	pip install -e .

test:
	pytest tests/

train:
	python -m src.pipeline run --config configs/default.yaml

evaluate:
	python -m src.pipeline run --stages evaluation,visualization

clean:
	rm -rf outputs/*
	find . -type d -name __pycache__ -exec rm -rf {} +

report:
	python scripts/generate_report.py --format html
```

### CLI Interface
```bash
# Clear, intuitive commands
esg-ml train --model xgboost --optimize
esg-ml evaluate --baseline random
esg-ml visualize --plots all
esg-ml report --format pdf
```

## 4. Documentation Structure

### 4.1 README.md
```markdown
# ESG ML Pipeline

A clean, modular machine learning pipeline for ESG score prediction.

## Quick Start
```bash
make install
make train
make report
```

## Features
- Multiple model architectures (Linear, Tree-based, Ensemble)
- Automated hyperparameter optimization
- Comprehensive evaluation metrics
- Publication-ready visualizations

[Links to detailed docs]
```

### 4.2 API Documentation
```python
class ModelTrainer:
    """Handles model training with automatic logging and checkpointing.
    
    Example:
        >>> trainer = ModelTrainer(config)
        >>> model = trainer.train(X_train, y_train)
        >>> metrics = trainer.evaluate(X_test, y_test)
    """
```

## 5. Migration Plan

### Phase 1: Core Structure (Week 1)
1. Create new directory structure
2. Move core functionality to appropriate modules
3. Create base classes and interfaces

### Phase 2: Model Migration (Week 2)
1. Standardize model interfaces
2. Fix naming inconsistencies
3. Create model registry

### Phase 3: Pipeline Integration (Week 3)
1. Build pipeline orchestrator
2. Create configuration system
3. Add logging and monitoring

### Phase 4: Visualization Consolidation (Week 4)
1. Merge visualization modules
2. Standardize plot generation
3. Create consistent styling

### Phase 5: Testing & Documentation (Week 5)
1. Write comprehensive tests
2. Generate API documentation
3. Create user guides

### Phase 6: Cleanup (Week 6)
1. Remove temporary files
2. Archive old code properly
3. Final testing and validation

## 6. Key Improvements

### 6.1 Testability
```python
# Each component can be tested independently
def test_model_trainer():
    config = {"model_type": "xgboost"}
    trainer = ModelTrainer(config)
    
    X, y = generate_test_data()
    model = trainer.train(X, y)
    
    assert model.score(X, y) > 0.8
```

### 6.2 Maintainability
- Clear module boundaries
- Consistent naming conventions
- Comprehensive logging
- Version control for models

### 6.3 Extensibility
```python
# Easy to add new models
class NewModel(BaseModel):
    def fit(self, X, y):
        # Implementation
    
    def predict(self, X):
        # Implementation

# Register and use
ModelRegistry.register("new_model", NewModel)
```

## 7. GitLab/GitHub Structure

### Repository Setup
- Protected main branch
- Feature branch workflow
- CI/CD pipeline
- Automated testing

### CI/CD Pipeline
```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

test:
  script:
    - make test
    - make lint

build:
  script:
    - make build
    
deploy:
  script:
    - make deploy
  only:
    - main
```

This architecture provides:
- **Clarity**: Easy to understand and navigate
- **Modularity**: Components can be developed/tested independently
- **Scalability**: Easy to add new features
- **Maintainability**: Clear ownership and responsibilities
- **Documentation**: Comprehensive and organized