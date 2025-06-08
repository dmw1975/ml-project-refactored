#!/usr/bin/env python3
"""Fix the initialization order in main.py and cleanup method in state_manager.py"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def fix_main_initialization_order():
    """Fix the order of operations in main.py"""
    
    main_path = project_root / "main.py"
    
    with open(main_path, 'r') as f:
        content = f.read()
    
    # The correct order should be:
    # 1. Parse args first
    # 2. Then initialize state manager
    # 3. Check force_retune using the parsed args
    
    # Replace the problematic section
    old_section = '''def main():
    """Main function."""

    # Initialize pipeline state manager
    try:
        # Delayed import to avoid circular dependencies
        from src.pipelines.state_manager import get_state_manager, PipelineStage
        state_manager = get_state_manager()
        state_manager.start_stage(PipelineStage.INITIALIZATION)

        
        # Reset state if force-retune is requested
        if hasattr(args, 'force_retune') and args.force_retune:
            print("Force retune requested - resetting pipeline state")
            state_manager.reset_state()
        use_state_manager = True
    except Exception as e:
        print(f"Warning: State manager initialization failed: {e}")
        print("Continuing without pipeline state tracking...")
        state_manager = None
        use_state_manager = False
    args = parse_args()'''
    
    new_section = '''def main():
    """Main function."""
    
    # Parse arguments first
    args = parse_args()
    
    # Initialize pipeline state manager
    try:
        # Delayed import to avoid circular dependencies
        from src.pipelines.state_manager import get_state_manager, PipelineStage
        state_manager = get_state_manager()
        
        # Reset state if force-retune is requested
        if hasattr(args, 'force_retune') and args.force_retune:
            print("Force retune requested - resetting pipeline state")
            state_manager.reset_state()
            
        state_manager.start_stage(PipelineStage.INITIALIZATION)
        use_state_manager = True
    except Exception as e:
        print(f"Warning: State manager initialization failed: {e}")
        print("Continuing without pipeline state tracking...")
        state_manager = None
        use_state_manager = False'''
    
    content = content.replace(old_section, new_section)
    
    # Write back
    with open(main_path, 'w') as f:
        f.write(content)
    
    print("✓ Fixed initialization order in main.py")


def fix_state_manager_cleanup():
    """Fix the indentation error in state_manager.py cleanup method"""
    
    state_manager_path = project_root / "src" / "pipelines" / "state_manager.py"
    
    with open(state_manager_path, 'r') as f:
        content = f.read()
    
    # Fix the indentation issue in _cleanup method
    old_cleanup = '''        print(f"{'='*60}")
    
        def _cleanup(self):
        """Cleanup on exit."""'''
    
    new_cleanup = '''        print(f"{'='*60}")
    
    def _cleanup(self):
        """Cleanup on exit."""'''
    
    content = content.replace(old_cleanup, new_cleanup)
    
    # Write back
    with open(state_manager_path, 'w') as f:
        f.write(content)
    
    print("✓ Fixed cleanup method indentation in state_manager.py")


def main():
    """Apply all fixes"""
    
    print("Fixing pipeline initialization issues...\n")
    
    fix_main_initialization_order()
    fix_state_manager_cleanup()
    
    print("\n✓ Pipeline initialization issues fixed")
    print("\nThe pipeline should now:")
    print("1. Parse arguments before initializing state manager")
    print("2. Properly reset state when using --force-retune")
    print("3. Have correct cleanup method indentation")


if __name__ == "__main__":
    main()