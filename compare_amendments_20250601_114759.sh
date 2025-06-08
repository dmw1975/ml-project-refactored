#!/bin/bash
# Convenience script to compare pre and post amendment outputs

echo "Creating post-amendment archive..."
python scripts/archive/archive_outputs_enhanced.py --full --name outputs_post_amendment_20250601_114759

echo "\nComparing archives..."
python scripts/archive/archive_outputs_enhanced.py --compare "/mnt/d/ml_project_refactored/outputs_pre_amendment_20250601_114759" "outputs_post_amendment_20250601_114759"
