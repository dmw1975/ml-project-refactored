#!/bin/bash

# Script to move esg_ml_clean to a separate location

echo "Moving ESG ML Clean Architecture to separate location..."

# Get the parent directory (one level up from ml_project_refactored)
PARENT_DIR=$(dirname "$(pwd)")
TARGET_DIR="$PARENT_DIR/esg_ml_clean"

# Check if target already exists
if [ -d "$TARGET_DIR" ]; then
    echo "Error: Target directory $TARGET_DIR already exists!"
    echo "Please remove it first or choose a different location."
    exit 1
fi

# Move the directory
echo "Moving esg_ml_clean to $TARGET_DIR..."
mv esg_ml_clean "$TARGET_DIR"

# Create a symlink for reference (optional)
ln -s "$TARGET_DIR" esg_ml_clean_link

echo "✅ Move complete!"
echo ""
echo "Clean architecture is now at: $TARGET_DIR"
echo ""
echo "To use it:"
echo "  cd $TARGET_DIR"
echo "  python demo_full_pipeline.py"
echo ""
echo "A symlink 'esg_ml_clean_link' was created for reference."