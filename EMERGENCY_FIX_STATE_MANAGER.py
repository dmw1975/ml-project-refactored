#!/usr/bin/env python3
"""
EMERGENCY FIX: Add proper state transitions to main.py

This script patches main.py to add the missing state manager transitions
for training, evaluation, and visualization stages.
"""

import re
from pathlib import Path

def fix_state_transitions():
    """Add missing state transitions to main.py"""
    
    main_file = Path("main.py")
    content = main_file.read_text()
    
    # Fix 1: Complete initialization after setup, not at the end
    # Find the line after logging setup (around line 169)
    pattern1 = r'(logging\.info\("Parsed arguments:"\)\n.*?logging\.info\(f"  {arg}: {value}"\))'
    replacement1 = r'\1\n    \n    # Complete initialization stage\n    if use_state_manager and state_manager:\n        state_manager.complete_stage(PipelineStage.INITIALIZATION)'
    
    # Fix 2: Add training stage transitions
    # Before "if args.train or args.all:" (around line 222)
    pattern2 = r'(# Training\n    if args\.train or args\.all:)'
    replacement2 = r'# Training\n    if args.train or args.all:\n        if use_state_manager and state_manager:\n            state_manager.start_stage(PipelineStage.TRAINING)'
    
    # After training completion (find the end of training block)
    pattern3 = r'(print\("All model training complete!"\))'
    replacement3 = r'\1\n        if use_state_manager and state_manager:\n            state_manager.complete_stage(PipelineStage.TRAINING)'
    
    # Fix 3: Add evaluation stage transitions
    # Before "if args.evaluate or args.all:" (around line 932)
    pattern4 = r'(# Evaluation\n    if args\.evaluate or args\.all:)'
    replacement4 = r'# Evaluation\n    if args.evaluate or args.all:\n        if use_state_manager and state_manager:\n            state_manager.start_stage(PipelineStage.EVALUATION)'
    
    # After evaluation completion
    pattern5 = r'(print\("Model evaluation complete!"\))'
    replacement5 = r'\1\n        if use_state_manager and state_manager:\n            state_manager.complete_stage(PipelineStage.EVALUATION)'
    
    # Fix 4: Add visualization stage transitions
    # Before "if args.visualize or args.all:" (around line 1053)
    pattern6 = r'(# Visualization.*?\n    if args\.visualize or args\.all:)'
    replacement6 = r'\1\n        if use_state_manager and state_manager:\n            state_manager.start_stage(PipelineStage.VISUALIZATION)'
    
    # After visualization completion
    pattern7 = r'(print\("All visualizations generated successfully!"\))'
    replacement7 = r'\1\n        if use_state_manager and state_manager:\n            state_manager.complete_stage(PipelineStage.VISUALIZATION)'
    
    # Fix 5: Remove the incorrect initialization completion at the end
    pattern8 = r'state_manager\.complete_stage\(PipelineStage\.INITIALIZATION\)\n.*?state_manager\.complete_stage\(PipelineStage\.COMPLETION\)'
    replacement8 = r'state_manager.complete_stage(PipelineStage.COMPLETION)'
    
    # Apply all fixes
    content = re.sub(pattern1, replacement1, content, flags=re.DOTALL)
    content = re.sub(pattern2, replacement2, content)
    content = re.sub(pattern3, replacement3, content)
    content = re.sub(pattern4, replacement4, content)
    content = re.sub(pattern5, replacement5, content)
    content = re.sub(pattern6, replacement6, content, flags=re.DOTALL)
    content = re.sub(pattern7, replacement7, content)
    content = re.sub(pattern8, replacement8, content, flags=re.DOTALL)
    
    # Backup original
    import shutil
    shutil.copy("main.py", "main.py.backup_before_state_fix")
    
    # Write fixed version
    main_file.write_text(content)
    
    print("✅ State transitions fixed in main.py")
    print("📄 Original backed up to main.py.backup_before_state_fix")
    print("\nAdded transitions:")
    print("  - Initialization completes after setup")
    print("  - Training stage: start/complete")
    print("  - Evaluation stage: start/complete")
    print("  - Visualization stage: start/complete")
    print("  - Removed incorrect initialization completion at end")

if __name__ == "__main__":
    fix_state_transitions()