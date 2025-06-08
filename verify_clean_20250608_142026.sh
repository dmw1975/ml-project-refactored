#!/bin/bash
# Quick verification that outputs is clean and ready

echo "Checking outputs directory..."
echo "Files in outputs/: $(find outputs -type f | wc -l) (should be 0)"
echo "Directories in outputs/: $(find outputs -type d | wc -l)"
echo ""
echo "Checking for pipeline state file..."
if [ -f "outputs/pipeline_state.json" ]; then
    echo "WARNING: pipeline_state.json still exists!"
else
    echo "✓ No pipeline state file found (good)"
fi
echo ""
echo "Checking for model files..."
MODEL_COUNT=$(find outputs -name "*.pkl" | wc -l)
if [ $MODEL_COUNT -gt 0 ]; then
    echo "WARNING: Found $MODEL_COUNT .pkl files!"
else
    echo "✓ No model files found (good)"
fi
