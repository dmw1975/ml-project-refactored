# Clean Architecture Action Plan

## Phase 1: Foundation (Days 1-3)

### Day 1: Create Base Structure
```bash
# 1. Create new clean directory
mkdir esg_ml_clean
cd esg_ml_clean

# 2. Initialize git
git init
echo "# ESG ML Pipeline" > README.md

# 3. Create directory structure
mkdir -p src/{pipeline,data,models,training,evaluation,visualization,utils}
mkdir -p tests/{unit,integration}
mkdir -p docs configs scripts notebooks
mkdir -p outputs/{models,figures,reports,logs}

# 4. Create __init__.py files
find src -type d -exec touch {}/__init__.py \;

# 5. Setup .gitignore
cat > .gitignore << EOF
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.env
.venv
outputs/
*.log
.pytest_cache/
.coverage
htmlcov/
.DS_Store
.idea/
.vscode/
EOF
```

### Day 2: Extract Core Components
```python
# 1. Create base model interface (src/models/base.py)
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

class BaseModel(ABC):
    """Base class for all models."""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.params = kwargs
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """Fit the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        import joblib
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """Load model from disk."""
        import joblib
        return joblib.load(path)
```

### Day 3: Setup Configuration System
```yaml
# configs/default.yaml
project:
  name: "ESG ML Pipeline"
  version: "2.0.0"
  
paths:
  data:
    raw: "data/raw"
    processed: "data/processed"
  outputs:
    models: "outputs/models"
    figures: "outputs/figures"
    reports: "outputs/reports"
    
pipeline:
  stages:
    - data_preparation
    - model_training
    - evaluation
    - visualization
    
models:
  linear_regression:
    enabled: true
  elasticnet:
    enabled: true
    optimize: true
  xgboost:
    enabled: true
    optimize: true
  lightgbm:
    enabled: true
    optimize: true
  catboost:
    enabled: true
    optimize: true
```

## Phase 2: Core Implementation (Days 4-7)

### Day 4: Data Pipeline
```python
# src/data/loader.py
from pathlib import Path
import pandas as pd
from typing import Tuple, Optional

class DataLoader:
    """Handles all data loading operations."""
    
    def __init__(self, config: dict):
        self.config = config
        
    def load_features(self) -> pd.DataFrame:
        """Load feature data."""
        path = Path(self.config['paths']['data']['processed']) / 'features.csv'
        return pd.read_csv(path)
    
    def load_targets(self) -> pd.Series:
        """Load target variable."""
        path = Path(self.config['paths']['data']['processed']) / 'targets.csv'
        return pd.read_csv(path).squeeze()
    
    def get_train_test_split(
        self, 
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Get train/test split."""
        from sklearn.model_selection import train_test_split
        
        X = self.load_features()
        y = self.load_targets()
        
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self._get_stratify_column(X)
        )
```

### Day 5: Model Training Pipeline
```python
# src/training/trainer.py
from typing import Dict, List, Any
import logging
from src.models.base import BaseModel
from src.models.registry import ModelRegistry

class ModelTrainer:
    """Orchestrates model training."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        
    def train_all(self, X_train, y_train) -> Dict[str, BaseModel]:
        """Train all enabled models."""
        for model_name, model_config in self.config['models'].items():
            if model_config.get('enabled', False):
                self.logger.info(f"Training {model_name}")
                model = self._train_model(
                    model_name, 
                    model_config, 
                    X_train, 
                    y_train
                )
                self.models[model_name] = model
        
        return self.models
    
    def _train_model(
        self, 
        name: str, 
        config: dict, 
        X_train, 
        y_train
    ) -> BaseModel:
        """Train a single model."""
        model_class = ModelRegistry.get(name)
        model = model_class(name=name, **config)
        
        if config.get('optimize', False):
            from src.training.optimizer import optimize_hyperparameters
            best_params = optimize_hyperparameters(
                model, X_train, y_train, config
            )
            model.update_params(best_params)
        
        model.fit(X_train, y_train)
        return model
```

### Day 6: Evaluation System
```python
# src/evaluation/evaluator.py
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class ModelEvaluator:
    """Evaluates model performance."""
    
    def __init__(self, config: dict):
        self.config = config
        self.results = {}
        
    def evaluate_all(
        self, 
        models: Dict[str, Any], 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> pd.DataFrame:
        """Evaluate all models."""
        for name, model in models.items():
            self.results[name] = self.evaluate_model(
                model, X_test, y_test
            )
        
        return pd.DataFrame(self.results).T
    
    def evaluate_model(self, model, X_test, y_test) -> Dict[str, float]:
        """Evaluate a single model."""
        y_pred = model.predict(X_test)
        
        return {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred)
        }
```

### Day 7: Visualization System
```python
# src/visualization/plotter.py
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class Plotter:
    """Handles all visualization generation."""
    
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config['paths']['outputs']['figures'])
        self._setup_style()
    
    def _setup_style(self):
        """Set consistent plot style."""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def plot_all(self, models, evaluation_results):
        """Generate all plots."""
        self.plot_metrics_comparison(evaluation_results)
        self.plot_feature_importance(models)
        self.plot_residuals(models)
        # etc...
    
    def plot_metrics_comparison(self, results: pd.DataFrame):
        """Plot metrics comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics = ['RMSE', 'MAE', 'R2', 'MSE']
        for ax, metric in zip(axes.flat, metrics):
            results[metric].plot(kind='bar', ax=ax)
            ax.set_title(f'{metric} by Model')
            ax.set_xlabel('Model')
            ax.set_ylabel(metric)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_comparison.png')
```

## Phase 3: Integration (Days 8-10)

### Day 8: Create Main Pipeline
```python
# src/pipeline/main.py
import logging
from typing import Optional, List
from pathlib import Path

from src.data.loader import DataLoader
from src.training.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.visualization.plotter import Plotter

class Pipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config: dict):
        self.config = config
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging."""
        log_file = Path(self.config['paths']['outputs']['logs']) / 'pipeline.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run(self, stages: Optional[List[str]] = None):
        """Run pipeline stages."""
        stages = stages or self.config['pipeline']['stages']
        
        for stage in stages:
            self.logger.info(f"Running stage: {stage}")
            getattr(self, f"run_{stage}")()
    
    def run_data_preparation(self):
        """Load and prepare data."""
        loader = DataLoader(self.config)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            loader.get_train_test_split()
        
    def run_model_training(self):
        """Train models."""
        trainer = ModelTrainer(self.config)
        self.models = trainer.train_all(self.X_train, self.y_train)
        
    def run_evaluation(self):
        """Evaluate models."""
        evaluator = ModelEvaluator(self.config)
        self.results = evaluator.evaluate_all(
            self.models, self.X_test, self.y_test
        )
        
    def run_visualization(self):
        """Generate visualizations."""
        plotter = Plotter(self.config)
        plotter.plot_all(self.models, self.results)
```

### Day 9: Create CLI Interface
```python
# scripts/cli.py
import click
import yaml
from pathlib import Path
from src.pipeline.main import Pipeline

@click.group()
def cli():
    """ESG ML Pipeline CLI."""
    pass

@cli.command()
@click.option('--config', default='configs/default.yaml', help='Config file path')
@click.option('--stages', multiple=True, help='Stages to run')
def run(config, stages):
    """Run the pipeline."""
    with open(config) as f:
        cfg = yaml.safe_load(f)
    
    pipeline = Pipeline(cfg)
    pipeline.run(stages=stages if stages else None)

@cli.command()
@click.option('--model', help='Model to train')
@click.option('--optimize', is_flag=True, help='Optimize hyperparameters')
def train(model, optimize):
    """Train a specific model."""
    # Implementation
    pass

@cli.command()
def list_models():
    """List available models."""
    from src.models.registry import ModelRegistry
    for name in ModelRegistry.list():
        click.echo(f"- {name}")

if __name__ == '__main__':
    cli()
```

### Day 10: Setup Testing
```python
# tests/unit/test_models.py
import pytest
import pandas as pd
import numpy as np
from src.models.linear import LinearRegressionModel

@pytest.fixture
def sample_data():
    """Generate sample data."""
    n_samples = 100
    n_features = 10
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randn(n_samples))
    return X, y

def test_linear_regression_fit(sample_data):
    """Test linear regression fitting."""
    X, y = sample_data
    model = LinearRegressionModel(name='test')
    
    model.fit(X, y)
    assert model.is_fitted
    
    predictions = model.predict(X)
    assert len(predictions) == len(y)
```

## Phase 4: Migration (Days 11-14)

### Day 11-12: Migrate Existing Code
```python
# migration/migrate_models.py
"""Script to migrate existing models to new structure."""

import pickle
from pathlib import Path
from src.models.registry import ModelRegistry

def migrate_models():
    """Migrate existing model files."""
    old_models_dir = Path('../outputs/models')
    new_models_dir = Path('outputs/models')
    
    for old_file in old_models_dir.glob('*.pkl'):
        with open(old_file, 'rb') as f:
            old_model = pickle.load(f)
        
        # Convert to new format
        new_model = convert_model(old_model)
        
        # Save in new location
        new_path = new_models_dir / f"{new_model.name}.pkl"
        new_model.save(new_path)
```

### Day 13-14: Documentation
```markdown
# docs/getting_started.md
# Getting Started

## Installation
```bash
git clone <repo>
cd esg-ml-pipeline
pip install -e .
```

## Quick Start
```bash
# Run full pipeline
esg-ml run

# Train specific model
esg-ml train --model xgboost --optimize

# Generate report
esg-ml report
```

## Configuration
Edit `configs/default.yaml` to customize the pipeline.
```

## Benefits Summary

### Before:
- 😱 1379-line main.py
- 🤯 43 scattered .md files  
- 😵 Unknown dependencies
- 🙈 Hidden plot generation logic
- 💥 Fragile and hard to modify

### After:
- ✅ Modular, testable components
- ✅ Clear configuration-driven pipeline
- ✅ Comprehensive documentation
- ✅ Easy to understand and extend
- ✅ Production-ready

### For GitHub:
- Professional structure
- Clear README
- CI/CD ready
- Easy for contributors
- Proper versioning

This action plan provides a practical, step-by-step approach to transform the messy codebase into a clean, maintainable architecture suitable for production use and open-source collaboration.