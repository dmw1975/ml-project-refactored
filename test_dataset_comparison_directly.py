#!/usr/bin/env python3
"""Test dataset comparison visualization directly."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.io import load_all_models
from src.visualization.plots.dataset_comparison import DatasetModelComparisonPlot

# Load all models
print("Loading all models...")
all_models = load_all_models()
print(f"Loaded {len(all_models)} models")

# Create visualization
print("\nCreating dataset comparison visualization...")
viz = DatasetModelComparisonPlot(all_models)

# Extract metrics to see what's happening
print("\nExtracting model metrics...")
metrics_df = viz.extract_model_metrics()

print(f"\nExtracted metrics for {len(metrics_df)} models")
print(f"Model families found: {sorted(metrics_df['model_family'].unique())}")

# Count by family
family_counts = metrics_df['model_family'].value_counts()
print("\nCounts by model family:")
for family, count in family_counts.items():
    print(f"  {family}: {count}")

# Check which models are missing
all_model_names = set(all_models.keys())
extracted_model_names = set(metrics_df['model_name'].unique())
missing_models = all_model_names - extracted_model_names

if missing_models:
    print(f"\nMISSING MODELS ({len(missing_models)}):")
    for model_name in sorted(missing_models):
        print(f"  - {model_name}")
        # Check why it's missing
        model = all_models[model_name]
        if hasattr(model, 'get_metrics'):
            try:
                metrics = model.get_metrics()
                print(f"    Has get_metrics() but returned: {metrics}")
            except Exception as e:
                print(f"    Error calling get_metrics(): {e}")
        else:
            print(f"    No get_metrics() method")