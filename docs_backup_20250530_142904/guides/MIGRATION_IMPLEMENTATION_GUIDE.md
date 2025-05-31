# Migration Implementation Guide

## Immediate Actions (This Week)

### 1. Create Core Structure
```bash
# Create new clean structure
mkdir -p esg_ml_clean/{src,tests,docs,configs,scripts,notebooks}
mkdir -p esg_ml_clean/src/{pipeline,data,models,training,evaluation,visualization,utils}
mkdir -p esg_ml_clean/outputs/{models,figures,reports,logs}
```

### 2. Extract Core Components from main.py

#### 2.1 Pipeline Runner (src/pipeline/runner.py)
```python
from dataclasses import dataclass
from typing import List, Optional
import logging

@dataclass
class PipelineConfig:
    stages: List[str]
    data_path: str
    output_dir: str
    models: List[str]
    
class PipelineRunner:
    """Orchestrates the ML pipeline execution."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def run(self):
        """Execute pipeline stages in order."""
        for stage in self.config.stages:
            self.logger.info(f"Running stage: {stage}")
            getattr(self, f"run_{stage}")()
    
    def run_data_loading(self):
        """Load and validate data."""
        pass
    
    def run_training(self):
        """Train models."""
        pass
    
    def run_evaluation(self):
        """Evaluate models."""
        pass
    
    def run_visualization(self):
        """Generate visualizations."""
        pass
```

#### 2.2 Model Interface (src/models/base.py)
```python
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseModel(ABC):
    """Base interface for all models."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def get_params(self) -> dict:
        """Get model parameters."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name."""
        pass
```

### 3. Standardize Model Naming

Create a mapping script to fix inconsistent names:
```python
# scripts/standardize_models.py
MODEL_NAME_MAPPING = {
    'LR_Base': 'LinearRegression_Base',
    'LR_Yeo': 'LinearRegression_Yeo',
    'ElasticNet_LR_Base_optuna': 'ElasticNet_Base_Optuna',
    # ... etc
}

def standardize_model_names(model_dict):
    """Standardize all model names."""
    return {
        MODEL_NAME_MAPPING.get(k, k): v 
        for k, v in model_dict.items()
    }
```

### 4. Create Configuration System

#### configs/default.yaml
```yaml
pipeline:
  stages:
    - data_loading
    - preprocessing
    - training
    - evaluation
    - visualization

data:
  raw_path: "data/raw/"
  processed_path: "data/processed/"
  features:
    numerical: ["feature1", "feature2"]
    categorical: ["sector", "region"]
  
models:
  linear_regression:
    enabled: true
    
  elasticnet:
    enabled: true
    optimize: true
    optuna_trials: 100
    
  xgboost:
    enabled: true
    variants: ["basic", "optuna"]
    
visualization:
  plots:
    - type: "performance_metrics"
      metrics: ["RMSE", "MAE", "R2"]
    - type: "feature_importance"
      top_k: 20
    - type: "residuals"
    - type: "baseline_comparison"
      
output:
  models_dir: "outputs/models"
  figures_dir: "outputs/figures"
  reports_dir: "outputs/reports"
```

### 5. Create Simple CLI Interface

#### scripts/cli.py
```python
import click
import yaml
from pathlib import Path

@click.group()
def cli():
    """ESG ML Pipeline CLI"""
    pass

@cli.command()
@click.option('--config', default='configs/default.yaml')
@click.option('--stages', multiple=True)
def run(config, stages):
    """Run the pipeline."""
    with open(config) as f:
        cfg = yaml.safe_load(f)
    
    if stages:
        cfg['pipeline']['stages'] = list(stages)
    
    from src.pipeline import PipelineRunner
    runner = PipelineRunner(cfg)
    runner.run()

@cli.command()
@click.option('--model', required=True)
@click.option('--optimize', is_flag=True)
def train(model, optimize):
    """Train a specific model."""
    # Implementation
    pass

@cli.command()
@click.option('--format', default='html')
def report(format):
    """Generate analysis report."""
    # Implementation
    pass

if __name__ == '__main__':
    cli()
```

### 6. Consolidate Visualizations

#### src/visualization/generator.py
```python
class VisualizationGenerator:
    """Central visualization generator."""
    
    def __init__(self, style_config=None):
        self.style_config = style_config or self._default_style()
        self._apply_style()
    
    def generate_all(self, models, output_dir):
        """Generate all configured visualizations."""
        generators = {
            'performance': self.plot_performance,
            'residuals': self.plot_residuals,
            'features': self.plot_features,
            'baseline': self.plot_baseline_comparison
        }
        
        for plot_type, generator in generators.items():
            generator(models, output_dir / plot_type)
    
    def plot_performance(self, models, output_dir):
        """Generate performance plots."""
        # Consolidated from various plot functions
        pass
```

### 7. Archive Cleanup Strategy

```bash
# Create a final archive before cleanup
tar -czf archive_final_20250129.tar.gz archive_* temp_* test_*

# Move to separate archive repository
git init esg-ml-archive
mv archive_final_20250129.tar.gz esg-ml-archive/
cd esg-ml-archive
git add .
git commit -m "Final archive before cleanup"

# In main repo, add to .gitignore
echo "archive_*" >> .gitignore
echo "temp_*" >> .gitignore
echo "test_*" >> .gitignore
```

### 8. Testing Strategy

#### tests/test_pipeline.py
```python
import pytest
from src.pipeline import PipelineRunner

def test_pipeline_stages():
    """Test pipeline executes stages in order."""
    config = {
        'stages': ['data_loading', 'training'],
        'data_path': 'tests/fixtures/test_data.csv'
    }
    
    runner = PipelineRunner(config)
    runner.run()
    
    assert runner.data is not None
    assert runner.models is not None

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    # Implementation
```

## Benefits of This Approach

### 1. **Gradual Migration**
- Can be done alongside existing code
- No disruption to current workflow
- Easy rollback if issues

### 2. **Clear Dependencies**
```python
# requirements.txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0
optuna>=2.10.0
matplotlib>=3.5.0
seaborn>=0.11.0
click>=8.0.0
pyyaml>=6.0
pytest>=7.0.0
```

### 3. **Documentation as Code**
```python
# Every function has clear docstrings
def train_model(X: pd.DataFrame, y: pd.Series, config: dict) -> BaseModel:
    """Train a model with given configuration.
    
    Args:
        X: Feature matrix
        y: Target variable
        config: Model configuration
        
    Returns:
        Trained model instance
        
    Example:
        >>> model = train_model(X_train, y_train, {'type': 'xgboost'})
    """
```

### 4. **Easy to Understand**
- New developers can onboard quickly
- Clear separation of concerns
- Intuitive command structure

### 5. **Production Ready**
- Proper logging
- Error handling
- Configuration management
- Reproducible results

## Next Steps

1. **Week 1**: Set up core structure and base classes
2. **Week 2**: Migrate model implementations
3. **Week 3**: Build pipeline orchestration
4. **Week 4**: Consolidate visualizations
5. **Week 5**: Write tests and documentation
6. **Week 6**: Final cleanup and validation

This approach transforms the messy codebase into a clean, maintainable, and extensible ML pipeline suitable for production use and easy collaboration.