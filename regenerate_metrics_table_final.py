#!/usr/bin/env python3
"""Regenerate metrics summary table with all models."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.visualization.viz_factory import create_metrics_table
from src.config import settings

# Generate the table
print("Generating metrics summary table with all models...")
output_path = settings.VISUALIZATION_DIR / "performance" / "metrics_summary_table.png"
output_path.parent.mkdir(parents=True, exist_ok=True)

from src.utils.io import load_all_models

try:
    # Load all models
    all_models = load_all_models()
    model_list = list(all_models.values())
    
    # Create the table
    result_path = create_metrics_table(model_list)
    if result_path:
        print(f"✓ Saved metrics summary table to {result_path}")
    else:
        print("✗ Failed to create metrics summary table")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()