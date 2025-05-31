# Import Fixes Applied

After restructuring the repository, the following import fixes were applied to make the pipeline work:

## 1. Fixed ROOT_DIR in settings.py
Changed from:
```python
ROOT_DIR = Path(__file__).parent.parent.absolute()
```
To:
```python
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
```
This ensures ROOT_DIR points to `/mnt/d/ml_project_refactored` instead of `/mnt/d/ml_project_refactored/src`

## 2. Updated imports in data modules
- `src/data/data_categorical.py`: Changed `from config.settings` to `from src.config.settings`
- `scripts/utilities/create_categorical_datasets.py`: Updated imports to use `src.` prefix

## 3. Main.py already updated
The main.py file was already automatically updated to use the new import structure when files were moved.

## Result
The pipeline now runs successfully with:
```bash
python main.py --all
```

All paths are correctly resolved and the training process proceeds as expected.