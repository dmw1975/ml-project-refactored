#!/bin/bash

echo "Setting up ESG ML Clean Architecture..."
echo "======================================="

# Navigate to the new location
cd /mnt/d/esg_ml_clean

# Create necessary directories
echo "Creating directories..."
mkdir -p outputs/plots
mkdir -p outputs/models  
mkdir -p outputs/metrics
mkdir -p checkpoints
mkdir -p logs
mkdir -p configs

# Create symbolic links to data
echo "Creating data links..."
mkdir -p data
ln -sf /mnt/d/ml_project_refactored/data/raw data/raw
ln -sf /mnt/d/ml_project_refactored/data/processed data/processed
ln -sf /mnt/d/ml_project_refactored/data/interim data/interim

# Run the Python setup script
echo "Running Python setup..."
python setup_after_move.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "You can now run the pipeline with:"
echo "  cd /mnt/d/esg_ml_clean"
echo "  python demo_full_pipeline.py"
echo ""
echo "Or run tests with:"
echo "  python tests/run_all_tests.py"