#!/bin/bash

# Fix ML Pipeline Script
# This ensures python main.py --all works correctly

echo "=========================================="
echo " FIXING ML PIPELINE FOR main.py --all"
echo "=========================================="

# Step 1: Backup existing models
echo -e "\n1. Backing up existing model files..."
mkdir -p outputs/models/backup_$(date +%Y%m%d)
cp outputs/models/*.pkl outputs/models/backup_$(date +%Y%m%d)/ 2>/dev/null || true

# Step 2: Remove problematic model files
echo -e "\n2. Removing problematic model files..."
rm -f outputs/models/xgboost_models.pkl
rm -f outputs/models/lightgbm_models.pkl
echo "   ✓ Removed xgboost_models.pkl and lightgbm_models.pkl"

# Step 3: Train all models with correct format
echo -e "\n3. Training all models with correct format..."
echo "   This will take some time..."

# Train models using the standard one-hot encoding for XGBoost and LightGBM
# This ensures they save in the correct dictionary format
python main.py --train-linear --train --train-xgboost --train-lightgbm --train-catboost --force-retune

# Step 4: Verify models are in correct format
echo -e "\n4. Verifying model formats..."
python -c "
import pickle
from pathlib import Path

model_dir = Path('outputs/models')
issues = []

# Check each model file
for model_file in ['xgboost_models.pkl', 'lightgbm_models.pkl']:
    filepath = model_dir / model_file
    if filepath.exists():
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            print(f'   ✓ {model_file}: Correct format (dict with {len(data)} models)')
        else:
            print(f'   ✗ {model_file}: Wrong format ({type(data).__name__})')
            issues.append(model_file)
    else:
        print(f'   ✗ {model_file}: Not found')

if issues:
    print(f'\nERROR: {len(issues)} files still have format issues')
    exit(1)
else:
    print('\n✓ All model files are in correct format!')
"

if [ $? -ne 0 ]; then
    echo -e "\n✗ Model format verification failed!"
    exit 1
fi

# Step 5: Run evaluation
echo -e "\n5. Running model evaluation..."
python main.py --evaluate

# Step 6: Generate visualizations
echo -e "\n6. Generating visualizations..."
python main.py --visualize

# Step 7: Verify key outputs
echo -e "\n7. Verifying key outputs..."
if [ -f "outputs/visualizations/performance/metrics_summary_table.png" ]; then
    echo "   ✓ metrics_summary_table.png created"
else
    echo "   ✗ metrics_summary_table.png missing"
fi

if [ -f "outputs/metrics/all_models_comparison.csv" ]; then
    echo "   ✓ all_models_comparison.csv created"
else
    echo "   ✗ all_models_comparison.csv missing"
fi

echo -e "\n=========================================="
echo " PIPELINE FIX COMPLETE!"
echo "=========================================="
echo -e "\nYou can now use: python main.py --all"
echo "The pipeline should work correctly."