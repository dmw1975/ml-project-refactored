#!/usr/bin/env python3
"""
EMERGENCY FIX: Fix the state manager transitions in main.py

The problem: The entire pipeline runs inside "initialization" stage!
This script creates a corrected version of main.py with proper stage transitions.
"""

import shutil
from pathlib import Path

def apply_emergency_fix():
    # Backup original
    shutil.copy("main.py", "main.py.backup_emergency")
    print("✅ Backed up main.py to main.py.backup_emergency")
    
    # Read the file
    with open("main.py", "r") as f:
        lines = f.readlines()
    
    # Track modifications
    modifications = []
    
    # Fix 1: Already applied - initialization completion after setup
    # This was done in previous edit
    
    # Fix 2: Add training stage start
    # Look for pattern around line 209-215 where training check happens
    for i in range(200, 250):
        if i < len(lines) and "if args.train or args.all:" in lines[i]:
            # Add state manager start before training
            indent = "    "  # Match the indentation
            insertion = f"{indent}# Start training stage\n{indent}if use_state_manager and state_manager:\n{indent}    state_manager.start_stage(PipelineStage.TRAINING)\n{indent}\n"
            lines.insert(i+1, insertion)
            modifications.append(f"Added training stage start at line {i+1}")
            break
    
    # Fix 3: Add training stage completion
    # Look for "All model training complete" or similar
    for i in range(700, 950):
        if i < len(lines) and "print(\"All model training complete" in lines[i]:
            indent = "        "  # Match the indentation
            insertion = f"\n{indent}# Complete training stage\n{indent}if use_state_manager and state_manager:\n{indent}    state_manager.complete_stage(PipelineStage.TRAINING)\n"
            lines.insert(i+1, insertion)
            modifications.append(f"Added training stage completion at line {i+1}")
            break
    
    # Fix 4: Add evaluation stage start
    # Look for evaluation section
    for i in range(900, 1100):
        if i < len(lines) and "if args.evaluate or args.all:" in lines[i] and "# Evaluation" in lines[i-2:i]:
            indent = "    "
            insertion = f"{indent}# Start evaluation stage\n{indent}if use_state_manager and state_manager:\n{indent}    state_manager.start_stage(PipelineStage.EVALUATION)\n{indent}\n"
            lines.insert(i+1, insertion)
            modifications.append(f"Added evaluation stage start at line {i+1}")
            break
    
    # Fix 5: Add evaluation stage completion
    # Look for "Model evaluation complete"
    for i in range(1000, 1200):
        if i < len(lines) and "print(\"Model evaluation complete" in lines[i]:
            indent = "        "
            insertion = f"\n{indent}# Complete evaluation stage\n{indent}if use_state_manager and state_manager:\n{indent}    state_manager.complete_stage(PipelineStage.EVALUATION)\n"
            lines.insert(i+1, insertion)
            modifications.append(f"Added evaluation stage completion at line {i+1}")
            break
    
    # Fix 6: Add visualization stage start
    # Look for visualization section
    for i in range(1050, 1300):
        if i < len(lines) and "if args.visualize or args.all:" in lines[i] and "# Visualization" in lines[i-5:i]:
            indent = "    "
            insertion = f"{indent}# Start visualization stage\n{indent}if use_state_manager and state_manager:\n{indent}    state_manager.start_stage(PipelineStage.VISUALIZATION)\n{indent}\n"
            lines.insert(i+1, insertion)
            modifications.append(f"Added visualization stage start at line {i+1}")
            break
    
    # Fix 7: Add visualization stage completion
    # Look for "All visualizations generated successfully"
    for i in range(1200, 1500):
        if i < len(lines) and "print(\"All visualizations generated successfully" in lines[i]:
            indent = "        "
            insertion = f"\n{indent}# Complete visualization stage\n{indent}if use_state_manager and state_manager:\n{indent}    state_manager.complete_stage(PipelineStage.VISUALIZATION)\n"
            lines.insert(i+1, insertion)
            modifications.append(f"Added visualization stage completion at line {i+1}")
            break
    
    # Fix 8: Remove the incorrect initialization completion at the end
    # Around line 1505-1506
    for i in range(1500, min(1520, len(lines))):
        if i < len(lines) and "state_manager.complete_stage(PipelineStage.INITIALIZATION)" in lines[i]:
            # Comment it out instead of removing
            lines[i] = "            # " + lines[i].lstrip()
            modifications.append(f"Commented out incorrect initialization completion at line {i}")
            break
    
    # Write the fixed file
    with open("main.py", "w") as f:
        f.writelines(lines)
    
    print("\n✅ Emergency fix applied to main.py")
    print("\n📋 Modifications made:")
    for mod in modifications:
        print(f"  - {mod}")
    
    print("\n🔧 The pipeline will now properly track stages:")
    print("  1. Initialization → Complete after setup")
    print("  2. Training → Start/Complete around model training")  
    print("  3. Evaluation → Start/Complete around evaluation")
    print("  4. Visualization → Start/Complete around visualization")
    print("  5. Completion → At the very end")
    
    print("\n⚡ Next steps:")
    print("  1. Remove old pipeline state: rm outputs/pipeline_state.json")
    print("  2. Run pipeline: python run_pipeline_safe.py --all --non-interactive --extended-timeout")
    print("  3. Monitor the state transitions in the log")

if __name__ == "__main__":
    apply_emergency_fix()