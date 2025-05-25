# Enhanced Archiving Guide

## Overview
I've created enhanced archiving capabilities that allow you to archive the entire `outputs/` directory (not just visualizations) before making amendments. This provides a complete snapshot to compare what changes your amendments create.

## New Scripts

### 1. `archive_outputs_enhanced.py`
A comprehensive archiving tool with multiple features:

**Basic Usage:**
```bash
# Archive entire outputs directory
python archive_outputs_enhanced.py --full

# Archive and clean (remove files after archiving)
python archive_outputs_enhanced.py --full --clean

# Archive with custom name
python archive_outputs_enhanced.py --full --name "pre_cv_fixes"

# Compare two archives
python archive_outputs_enhanced.py --compare "archive1" "archive2"
```

**Features:**
- Archives entire directories with detailed statistics
- Creates both JSON manifest and human-readable README
- Tracks file types, sizes, and directory structure
- Can compare archives to show what changed
- Preserves directory structure when cleaning

### 2. `archive_before_amendments.py`
A simplified script specifically for pre-amendment snapshots that **automatically cleans the outputs directory**:

```bash
# Quick snapshot before amendments + clean outputs
python archive_before_amendments.py
```

**What it does:**
1. Creates timestamped archive of entire outputs directory
2. Shows current statistics (files, size, subdirectories)
3. **CLEANS the outputs directory** (removes all files, preserves structure)
4. Creates a convenience comparison script
5. Provides clear next steps

**Important**: This script now automatically cleans the outputs directory after archiving, giving you a fresh start for your amended pipeline run.

## Workflow Example

### Before Making Changes:
```bash
# 1. Create pre-amendment snapshot AND clean outputs
python archive_before_amendments.py

# Output:
# 📸 CREATING PRE-AMENDMENT SNAPSHOT & CLEANING
# 📊 Current outputs directory statistics:
#   • Total files: 3,394
#   • Total size: 1.4 GB
#   • Subdirectories: visualizations, metrics, models, logs
# 🗄️  Creating archive: outputs_pre_amendment_20250525_120000
# 🧹 Cleaning source directory: outputs/
# ✅ PRE-AMENDMENT SNAPSHOT COMPLETE & OUTPUTS CLEANED!
# 🧹 Outputs directory has been cleaned - ready for fresh results!
```

### After Making Changes:
```bash
# 2. Run your amended pipeline
python main.py --all

# 3. Create post-amendment archive
python archive_outputs_enhanced.py --full --name outputs_post_amendment

# 4. Compare what changed
python archive_outputs_enhanced.py --compare "outputs_pre_amendment_20250525_120000" "outputs_post_amendment"

# Output:
# 📊 Comparing archives:
#   • Size change: +250 MB (increase)
#   • Files change: +156
#   • New file types: .json, .log
#   • New subdirectories: optimization, baseline_comparisons
```

## Archive Structure

Each archive contains:
```
outputs_archive_TIMESTAMP/
├── README.md           # Human-readable description
├── manifest.json       # Machine-readable metadata
└── outputs/            # Complete copy of outputs directory
    ├── visualizations/
    ├── metrics/
    ├── models/
    ├── logs/
    └── ...
```

## Benefits

1. **Complete History**: Full snapshot of all outputs, not just visualizations
2. **Easy Comparison**: See exactly what files were added/removed/changed
3. **Safe Experimentation**: Can always restore to previous state
4. **Documentation**: Each archive self-documents with statistics and metadata
5. **Flexibility**: Can archive/clean selectively or comprehensively

## Tips

- The outputs directory is currently 1.4GB, so archiving may take a minute
- Use `--clean` flag carefully - it removes files after archiving
- Archives are stored in the parent directory of outputs by default
- The comparison feature helps identify unexpected changes
- Keep meaningful archives, delete temporary ones to save space

## Example Use Cases

1. **Before Major Refactoring**:
   ```bash
   python archive_before_amendments.py
   # Make your changes
   python main.py --all
   # Compare results
   ```

2. **Testing Different Configurations**:
   ```bash
   python archive_outputs_enhanced.py --full --name "baseline_config"
   # Change configuration
   python main.py --all
   python archive_outputs_enhanced.py --full --name "new_config"
   python archive_outputs_enhanced.py --compare "baseline_config" "new_config"
   ```

3. **Clean Slate Testing**:
   ```bash
   python archive_outputs_enhanced.py --full --clean
   # Now outputs/ is empty but archived
   python main.py --all
   # Fresh outputs generated
   ```

The enhanced archiving provides complete visibility into how your amendments affect the entire output structure, not just visualizations.