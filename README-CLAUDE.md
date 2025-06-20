# Instructions for Claude - ML Project Refactored

## 🚨 CRITICAL: File Management Policy

**DO NOT CREATE NEW FILES** for individual problems or features. This project follows a strict modular architecture where all functionality must be integrated into the existing pipeline structure.

## 📋 Core Principles

1. **NO NEW FILES**: Do not create new Python files for features or bug fixes
2. **USE EXISTING MODULES**: All code must be added to appropriate existing modules
3. **MAINTAIN STRUCTURE**: The current directory structure is final and complete
4. **EXTEND, DON'T EXPAND**: Extend existing classes and functions rather than creating new files

## 🏗️ Architecture Overview

This is a modular ML pipeline with clear separation of concerns:

```
src/
├── config/           # ALL configuration logic goes here
├── data/            # ALL data operations go here
├── models/          # ALL model definitions go here
├── evaluation/      # ALL metrics and evaluation logic go here
├── pipelines/       # ALL pipeline orchestration goes here
├── utils/           # ALL general utilities go here
└── visualization/   # ALL visualization logic goes here
```

## 📝 Where to Add Code

### Configuration Changes
- **File**: `src/config/`
- **What goes here**: Hyperparameters, paths, experiment settings
- **DO NOT**: Create new config files for specific experiments

### Data Processing
- **File**: `src/data/`
- **What goes here**: Data loaders, preprocessors, transformers
- **DO NOT**: Create separate files for different datasets

### Model Implementation
- **File**: `src/models/`
- **What goes here**: Model architectures, training logic
- **DO NOT**: Create new files for model variants - use inheritance or configuration

### Evaluation Metrics
- **File**: `src/evaluation/`
- **What goes here**: Metrics, analysis functions, performance tracking
- **DO NOT**: Create separate files for different evaluation scenarios

### Pipeline Orchestration
- **File**: `src/pipelines/`
- **What goes here**: End-to-end workflows, experiment runners
- **DO NOT**: Create new pipeline files for different experiments

### Utilities
- **File**: `src/utils/`
- **What goes here**: Helper functions, common operations
- **DO NOT**: Create utility files for specific features

### Visualization System
- **Structure**:
  ```
  visualization/
  ├── adapters/    # Model-specific adapters ONLY
  ├── core/        # Core infrastructure ONLY
  ├── plots/       # Plot implementations ONLY
  └── components/  # Reusable visualization components ONLY
  ```
- **DO NOT**: Create new visualization files outside this structure

## 🔧 Implementation Guidelines

### Adding a New Feature
```python
# WRONG ❌
# Creating feature_x.py

# CORRECT ✅
# Add to existing module:
# src/models/existing_model.py
class ExistingModel:
    def new_feature_method(self):
        """Add new functionality here"""
        pass
```

### Adding a New Model
```python
# WRONG ❌
# Creating src/models/my_new_model.py

# CORRECT ✅
# Add to src/models/__init__.py or existing model file:
class NewModelVariant(BaseModel):
    """Implement as a class in existing structure"""
    pass
```

### Adding Data Processing
```python
# WRONG ❌
# Creating src/data/special_preprocessing.py

# CORRECT ✅
# Add to src/data/preprocessing.py:
def special_preprocessing_function():
    """Add to existing preprocessing module"""
    pass
```

## 📊 Data Directory Usage

- `data/raw/`: Original, immutable data files
- `data/processed/`: Preprocessed datasets only
- `data/pkl/`: Serialized objects only
- **DO NOT**: Create subdirectories for specific experiments

## 📤 Output Organization

- `outputs/models/`: Saved model checkpoints
- `outputs/visualizations/`: Generated plots and figures
- `outputs/metrics/`: Evaluation results in structured format
- `outputs/reports/`: Analysis reports and summaries
- **DO NOT**: Create new output subdirectories

## 🧪 Testing Policy

- Add tests to existing test files in `tests/`
- Match test structure to source structure
- **DO NOT**: Create new test files for individual features

## ⚡ Quick Reference

| Task | Location | File Policy |
|------|----------|-------------|
| New data loader | `src/data/` | Extend existing loader |
| New model architecture | `src/models/` | Add class to existing file |
| New metric | `src/evaluation/` | Add function to existing file |
| New visualization | `src/visualization/plots/` | Extend existing plot classes |
| New utility | `src/utils/` | Add to relevant utility module |
| New pipeline | `src/pipelines/` | Extend existing pipeline class |

## 🚫 Prohibited Actions

1. Creating new Python files in any directory
2. Creating new subdirectories
3. Restructuring the existing architecture
4. Adding standalone scripts outside the `scripts/` directory
5. Creating feature-specific modules

## ✅ Always Do

1. Read existing code before implementing
2. Use inheritance and composition
3. Follow existing naming conventions
4. Maintain backward compatibility
5. Update docstrings in existing files

## 💡 Example Workflow

When asked to "implement feature X":

1. **Identify** the appropriate existing module
2. **Locate** the relevant class or section
3. **Extend** with new methods or functions
4. **Test** using existing test structure
5. **Document** within the same file

Remember: This is a **refactored** project with a carefully designed structure. The architecture is complete - your role is to enhance functionality within the existing framework, not to expand the framework itself.