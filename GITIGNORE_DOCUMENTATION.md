# Git Repository Exclusion Documentation

## Overview

This document explains the rationale behind files and folders excluded from version control in this ML project repository. The goal is to maintain a clean, focused repository containing only essential source code and documentation while excluding generated outputs, cache files, and temporary content.

## Repository Size Impact

Before implementing comprehensive exclusions:
- **Total repository size**: ~6.6 GB
- **Generated outputs**: ~3.4 GB (51% of total)
- **Virtual environment**: ~2.0 GB (30% of total)
- **Essential code**: ~2 MB (<1% of total)

After implementing exclusions:
- **Expected repository size**: <50 MB
- **Size reduction**: >99%

## Excluded Categories

### 1. Generated Outputs (`/outputs/`, `/test_outputs/`)
**Rationale**: These folders contain pipeline-generated content that can be recreated by running the code.
- Model visualizations (PNG files)
- Trained model files (PKL files)
- Metrics and reports
- Feature importance plots
- SHAP visualizations

**Size**: ~195 MB (current) + 3.2 GB (archived)

### 2. Archived Outputs (`outputs_pre_amendment_*/`)
**Rationale**: Historical outputs from previous pipeline runs that are no longer needed.
- 13 timestamped folders
- Contains >7,000 files
- Represents multiple versions of the same outputs

**Size**: ~3.2 GB

### 3. Cache and Temporary Files
**Rationale**: Runtime artifacts that are machine-specific and regenerated automatically.
- `catboost_info/`: CatBoost training logs and temp files
- `__pycache__/`: Python bytecode cache
- `logs/`: Application log files
- `*.tmp`, `*.log`: Temporary and log files

**Size**: ~1 MB

### 4. Virtual Environment (`.venv/`)
**Rationale**: Python packages should be installed locally using `requirements.txt`.
- Contains 39,904 files
- Platform and Python version specific
- Can be recreated with `pip install -r requirements.txt`

**Size**: ~2.0 GB

### 5. Data Files (`/data/`)
**Rationale**: Large data files can be stored separately or downloaded as needed.
- Raw CSV files
- Processed data files
- Pickle files with preprocessed data

**Size**: ~20 MB

**Note**: Small reference datasets needed for examples may be kept if essential.

### 6. IDE Settings (`.vscode/`, `.claude/`)
**Rationale**: Personal IDE configurations that vary by developer.
- Editor preferences
- Local project settings
- Extension configurations

**Size**: <10 KB

### 7. Temporary Scripts
**Rationale**: Ad-hoc scripts created during development and debugging.
- `debug_*.py`: Debugging scripts
- `test_*.py`: Temporary test scripts
- `generate_*.py`: One-off generation scripts
- `fix_*.py`: Quick fix scripts
- `compare_amendments_*.sh`: Shell comparison scripts

**Count**: ~30 files

### 8. Archives and Backups
**Rationale**: Old versions and backups that clutter the repository.
- `archive_*/`: Code archives
- `docs_backup_*/`: Documentation backups
- Any folder with `_old`, `_backup`, `_archive` suffixes

**Size**: ~1.5 MB

## What IS Included

### Essential Files Kept in Repository:
1. **Source Code** (`/src/`): Core application code
2. **Scripts** (`/scripts/`): Utility and maintenance scripts (excluding temporary ones)
3. **Documentation** (`/docs/`): Project documentation (excluding backups)
4. **Configuration Files**: 
   - `requirements.txt`: Python dependencies
   - `setup.py`: Package setup
   - `.gitignore`: Version control configuration
   - `README.md`, `DOCUMENTATION.md`, `CLAUDE.md`: Main docs

5. **Tests** (`/tests/`): Unit and integration tests

## Setup Instructions for New Contributors

After cloning the repository, perform these steps:

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Unix/MacOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create necessary directories
mkdir -p outputs/{models,visualizations,metrics,reports}
mkdir -p data/{raw,processed,interim}
mkdir -p logs

# 5. Download any required datasets (if applicable)
# Instructions for data download here
```

## Maintenance Guidelines

### Before Committing:
1. **Check file sizes**: Use `git status --porcelain | xargs -I {} du -h {}` to check sizes
2. **Review new files**: Ensure no generated outputs are being added
3. **Update .gitignore**: Add new patterns for any new types of generated files

### Periodic Cleanup:
1. Remove old archived outputs: `rm -rf outputs_pre_amendment_*/`
2. Clean Python cache: `find . -type d -name __pycache__ -exec rm -rf {} +`
3. Remove old logs: `find logs/ -name "*.log" -mtime +30 -delete`

### If Repository Gets Large:
```bash
# Check what's taking space
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | sort -k3nr | head -20

# Clean up Git history (careful - rewrites history)
git filter-branch --tree-filter 'rm -rf outputs_pre_amendment_*' -- --all
```

## Common Issues and Solutions

### Issue: Accidentally committed large file
**Solution**:
```bash
# Remove from current commit
git rm --cached large_file.pkl
git commit --amend

# Remove from history (if already pushed)
git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch path/to/large_file.pkl' --prune-empty --tag-name-filter cat -- --all
```

### Issue: .gitignore not working
**Solution**:
```bash
# Clear Git cache
git rm -r --cached .
git add .
git commit -m "Reapply .gitignore"
```

### Issue: Need to share outputs with team
**Solution**: Use external storage (cloud storage, shared drives) or create a separate outputs repository.

## Contact

For questions about repository structure or exclusions, contact the project maintainer.