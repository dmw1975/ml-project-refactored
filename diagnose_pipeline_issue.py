#!/usr/bin/env python3
"""Diagnose ML pipeline initialization issue."""

import json
from pathlib import Path
from datetime import datetime

# Check pipeline state
state_file = Path("/mnt/d/ml_project_refactored/outputs/pipeline_state.json")
if state_file.exists():
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    print("=== PIPELINE STATE ANALYSIS ===")
    print(f"Pipeline ID: {state['pipeline_id']}")
    print(f"Start Time: {state['start_time']}")
    print(f"End Time: {state['end_time']}")
    
    # Calculate total duration
    if state['start_time'] and state['end_time']:
        start = datetime.fromisoformat(state['start_time'])
        end = datetime.fromisoformat(state['end_time'])
        duration = (end - start).total_seconds()
        print(f"Total Duration: {duration/60:.1f} minutes")
    
    print("\n=== STAGE STATUS ===")
    for stage_name, stage_info in state['stages'].items():
        print(f"\n{stage_name.upper()}:")
        print(f"  Status: {stage_info['status']}")
        print(f"  Start: {stage_info['start_time']}")
        print(f"  End: {stage_info['end_time']}")
        if stage_info['errors']:
            print(f"  Errors: {stage_info['errors']}")

# Check what models exist
models_dir = Path("/mnt/d/ml_project_refactored/outputs/models")
if models_dir.exists():
    print("\n=== EXISTING MODEL FILES ===")
    for model_file in models_dir.glob("*.pkl"):
        print(f"  {model_file.name}")

# Check recent logs for errors
print("\n=== CHECKING FOR ISSUES IN MAIN.PY ===")

# Read main.py to identify the issue
main_py = Path("/mnt/d/ml_project_refactored/main.py")
if main_py.exists():
    with open(main_py, 'r') as f:
        lines = f.readlines()
    
    # Find where stages are managed
    stage_lines = []
    for i, line in enumerate(lines):
        if 'start_stage' in line or 'complete_stage' in line:
            stage_lines.append((i+1, line.strip()))
    
    print("\nStage management in main.py:")
    for line_no, line in stage_lines[:10]:  # Show first 10
        print(f"  Line {line_no}: {line}")

print("\n=== DIAGNOSIS ===")
print("The issue appears to be that:")
print("1. The initialization stage starts at the beginning of main()")
print("2. The initialization stage is only completed at the END of main()")
print("3. This means the entire pipeline execution is considered 'initialization'")
print("4. The proper stage transitions (training, evaluation, visualization) are never marked")
print("\nThis is why the pipeline appears to 'hang' in initialization for over an hour.")
print("\n=== RECOMMENDED FIX ===")
print("The state manager stage transitions need to be properly integrated throughout main.py:")
print("- Complete initialization after setup is done")
print("- Start/complete training stage around model training")
print("- Start/complete evaluation stage around model evaluation")
print("- Start/complete visualization stage around visualization generation")