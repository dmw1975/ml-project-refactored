# ML Project for ESG Score Prediction


> **⚠️ FOR CLAUDE/AI ASSISTANTS: Please read README-CLAUDE.md before making any changes**

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Models](#models)
- [Output Structure](#output-structure)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a comprehensive machine learning pipeline for ESG (Environmental, Social, and Governance) score prediction. It features multiple ML models, robust evaluation metrics, and extensive visualization capabilities.

### Key Features
- **Multiple ML Models**: Linear Regression, ElasticNet, XGBoost, LightGBM, and CatBoost
- **Native Categorical Support**: Efficient handling of categorical features for tree-based models
- **Comprehensive Evaluation**: Cross-validation, baseline comparisons, feature importance analysis
- **Rich Visualizations**: SHAP values, residual plots, feature importance, model comparisons
- **Optuna Integration**: Hyperparameter optimization for all models
- **Modular Architecture**: Clean separation of concerns with adapters for different model types

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ml_project_refactored
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare data directories**:
   ```bash
   mkdir -p data/{raw,processed,interim}
   mkdir -p outputs/{models,visualizations,metrics,reports}
   mkdir -p logs
   ```

### Basic Usage

```bash
# Run complete pipeline
python main.py --all

# Train only tree models
python main.py --train

# Train linear models
python main.py --train-linear

# Evaluate all models
python main.py --evaluate

# Generate visualizations
python main.py --visualize
```

## 📘 Usage Guide

### Training Models

#### Train All Models
```bash
python main.py --train --train-linear
```

#### Train Specific Model Types
```bash
# XGBoost models
python main.py --train-xgboost

# LightGBM models
python main.py --train-lightgbm

# CatBoost models
python main.py --train-catboost

# ElasticNet with optimization
python main.py --train-linear-elasticnet --optimize-elasticnet 100
```

### Hyperparameter Optimization

```bash
# Optimize XGBoost with 50 trials
python main.py --optimize-xgboost 50

# Optimize LightGBM with 50 trials
python main.py --optimize-lightgbm 50

# Optimize CatBoost with 50 trials
python main.py --optimize-catboost 50

# Force retuning even if studies exist
python main.py --optimize-xgboost 50 --force-retune
```

### Model Evaluation

```bash
# Evaluate all models
python main.py --evaluate

# Generate feature importance analysis
python main.py --importance

# Analyze multicollinearity
python main.py --vif
```

### Visualization Generation

```bash
# Generate all visualizations
python main.py --visualize

# Generate model-specific visualizations
python main.py --visualize-xgboost
python main.py --visualize-lightgbm
python main.py --visualize-catboost
```

### Dataset Selection

```bash
# Train on specific datasets
python main.py --train --datasets LR_Base LR_Yeo

# Train on all datasets (default)
python main.py --train --datasets all
```

## 📁 Project Structure

```
ml_project_refactored/
├── src/
│   ├── config/           # Configuration settings
│   ├── data/            # Data loading and preprocessing
│   ├── models/          # Model implementations
│   ├── evaluation/      # Evaluation metrics and analysis
│   ├── pipelines/       # Pipeline orchestration
│   ├── utils/           # Utility functions
│   └── visualization/   # Visualization system
│       ├── adapters/    # Model adapters
│       ├── core/        # Core visualization infrastructure
│       ├── plots/       # Plot implementations
│       └── components/  # Reusable components
├── data/
│   ├── raw/            # Original data files
│   ├── processed/      # Processed datasets
│   └── pkl/            # Pickled objects
├── outputs/
│   ├── models/         # Trained models
│   ├── visualizations/ # Generated plots
│   ├── metrics/        # Evaluation results
│   └── reports/        # Analysis reports
├── scripts/            # Utility scripts
├── tests/              # Unit tests
└── docs/               # Documentation
```

## 🤖 Models

### Linear Models
- **Linear Regression**: Basic and Random feature variants
- **ElasticNet**: With Optuna hyperparameter optimization

### Tree-Based Models
- **XGBoost**: Gradient boosting with native categorical support
- **LightGBM**: Fast gradient boosting with categorical features
- **CatBoost**: Gradient boosting optimized for categorical data

### Model Variants
Each model is trained on four dataset variants:
1. **Base**: Original features
2. **Yeo**: Yeo-Johnson transformed features
3. **Base_Random**: Base + random feature
4. **Yeo_Random**: Yeo + random feature

## 📊 Output Structure

### Trained Models
```
outputs/models/
├── <ModelType>_<Dataset>_<Config>.pkl
└── optuna_studies/
    └── <ModelType>_<Dataset>_study.pkl
```

### Visualizations
```
outputs/visualizations/
├── features/           # Feature importance plots
├── performance/        # Model performance plots
├── residuals/         # Residual analysis
├── shap/              # SHAP value visualizations
├── comparisons/       # Model comparisons
└── optimization/      # Optuna optimization plots
```

### Metrics
```
outputs/metrics/
├── model_metrics.csv          # All model metrics
├── baseline_comparison.csv    # Baseline comparisons
├── vif_analysis.csv          # VIF results
└── feature_importance/       # Feature importance by model
```

## 🔧 Troubleshooting

### Common Issues

1. **Pipeline Hanging**
   - Use `python run_pipeline_safe.py` for better error handling
   - Check logs in `logs/` directory

2. **Memory Issues with SHAP**
   - Use memory-safe generation scripts in `scripts/utilities/`
   - Reduce sample size for SHAP calculations

3. **Missing CV Scores**
   - Some models may not have CV scores for Booster objects
   - Check model pickle files for available metrics

4. **Import Errors**
   - Ensure virtual environment is activated
   - Check `requirements.txt` for missing dependencies

### Debug Mode

Enable detailed logging:
```bash
python main.py --all 2>&1 | tee debug.log
```

## 🛠️ Development

### Code Style
- Follow PEP 8 guidelines
- Use NumPy-style docstrings
- Format with `black`
- Lint with `flake8`

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_models.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Adding New Models

1. Create model class in `src/models/`
2. Create adapter in `src/visualization/adapters/`
3. Register in visualization factory
4. Update pipeline configuration

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Contribution Guidelines
- Write tests for new features
- Update documentation
- Follow existing code style
- Add docstrings to new functions

## 📝 License

[Add license information here]

## 🙏 Acknowledgments

- Built with scikit-learn, XGBoost, LightGBM, CatBoost
- Visualization powered by matplotlib, seaborn, and SHAP
- Hyperparameter optimization by Optuna

---

For detailed technical documentation, see the `docs/` directory.