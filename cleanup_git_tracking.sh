#!/bin/bash
# ==============================================================================
# Git Repository Cleanup Script
# ==============================================================================
# This script removes files from Git tracking that should be ignored
# but preserves them locally. Run this after updating .gitignore
# ==============================================================================

echo "Git Repository Cleanup Script"
echo "============================="
echo ""
echo "This script will remove files from Git tracking while preserving them locally."
echo "Make sure you have committed any important changes before proceeding."
echo ""
read -p "Do you want to continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 1
fi

echo ""
echo "Step 1: Backing up current .gitignore..."
cp .gitignore .gitignore.backup

echo "Step 2: Updating .gitignore..."
mv .gitignore_updated .gitignore

echo ""
echo "Step 3: Removing files from Git tracking..."
echo "(Files will be preserved locally)"
echo ""

# Remove archived outputs
echo "Removing archived outputs..."
git rm -r --cached outputs_pre_amendment_*/ 2>/dev/null || true

# Remove test outputs
echo "Removing test outputs..."
git rm -r --cached test_outputs/ 2>/dev/null || true

# Remove CatBoost cache
echo "Removing CatBoost cache..."
git rm -r --cached catboost_info/ 2>/dev/null || true

# Remove logs
echo "Removing log files..."
git rm -r --cached logs/ 2>/dev/null || true
git rm --cached *.log 2>/dev/null || true

# Remove archive folders
echo "Removing archive folders..."
git rm -r --cached archive_*/ 2>/dev/null || true
git rm -r --cached docs_backup_*/ 2>/dev/null || true

# Remove temporary shell scripts
echo "Removing temporary shell scripts..."
git rm --cached compare_amendments_*.sh 2>/dev/null || true

# Remove temporary Python scripts
echo "Removing temporary Python scripts..."
git rm --cached debug_*.py 2>/dev/null || true
git rm --cached test_*.py 2>/dev/null || true
git rm --cached generate_*.py 2>/dev/null || true
git rm --cached fix_*.py 2>/dev/null || true
git rm --cached run_pipeline_safe.py 2>/dev/null || true
git rm --cached direct_run.py 2>/dev/null || true

# Remove IDE settings
echo "Removing IDE settings..."
git rm -r --cached .vscode/ 2>/dev/null || true
git rm -r --cached .claude/ 2>/dev/null || true

# Remove accidental token directory
echo "Removing sensitive directories..."
git rm -r --cached ghp_*/ 2>/dev/null || true

# Remove empty directories
echo "Removing empty directories..."
git rm -r --cached linreg_eval_metrics/ 2>/dev/null || true
git rm -r --cached configs/ 2>/dev/null || true

# Remove large binary files
echo "Removing large binary files..."
find . -name "*.pkl" -size +50M -exec git rm --cached {} \; 2>/dev/null || true

echo ""
echo "Step 4: Creating commit..."
git add .gitignore

echo ""
echo "Cleanup complete!"
echo ""
echo "Next steps:"
echo "1. Review the changes with: git status"
echo "2. Commit the changes with: git commit -m 'Clean up repository: remove non-essential files from tracking'"
echo "3. Push to remote with: git push"
echo ""
echo "Note: The first push after cleanup might be large as it updates the repository structure."
echo ""
echo "Optional: To reduce repository size, you can run:"
echo "  git gc --aggressive --prune=now"
echo ""