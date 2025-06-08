#!/usr/bin/env python3
"""Update main.py to use the new pipeline state management system."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def update_main_pipeline():
    """Update main.py to integrate pipeline state management."""
    
    main_path = project_root / "main.py"
    
    # Read the current main.py
    with open(main_path, 'r') as f:
        content = f.read()
    
    # Add import for state manager
    if "from src.pipelines.state_manager import" not in content:
        # Find the imports section
        import_section_end = content.find("def main()")
        if import_section_end > 0:
            # Insert the import before main()
            new_import = "from src.pipelines.state_manager import get_state_manager, PipelineStage\n\n"
            content = content[:import_section_end] + new_import + content[import_section_end:]
    
    # Update main() function to initialize state manager
    main_func_start = content.find("def main():")
    if main_func_start > 0:
        # Find where to insert initialization
        func_body_start = content.find("\n", main_func_start) + 1
        indent = "    "
        
        # Check if state manager initialization already exists
        if "state_manager = get_state_manager()" not in content:
            # Find the first line after docstring
            docstring_end = content.find('"""', func_body_start + 10)
            if docstring_end > 0:
                insert_pos = content.find("\n", docstring_end) + 1
            else:
                insert_pos = func_body_start
            
            # Insert state manager initialization
            init_code = f"""
{indent}# Initialize pipeline state manager
{indent}state_manager = get_state_manager()
{indent}state_manager.start_stage(PipelineStage.INITIALIZATION)
"""
            content = content[:insert_pos] + init_code + content[insert_pos:]
    
    # Update the end of main() to print summary
    main_func_end = content.rfind("if __name__ == '__main__':")
    if main_func_end > 0:
        # Find the last line of main()
        last_line_start = content.rfind("\n", 0, main_func_end - 1)
        
        # Check if summary already exists
        if "state_manager.print_summary()" not in content:
            summary_code = f"""
{indent}# Complete pipeline and print summary
{indent}state_manager.complete_stage(PipelineStage.INITIALIZATION)
{indent}state_manager.complete_stage(PipelineStage.COMPLETION)
{indent}state_manager.print_summary()
"""
            content = content[:last_line_start] + summary_code + content[last_line_start:]
    
    # Write the updated content
    with open(main_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Updated {main_path}")
    print("\nChanges made:")
    print("1. Added import for pipeline state manager")
    print("2. Added state manager initialization at start of main()")
    print("3. Added pipeline summary at end of main()")
    print("\nThe pipeline will now track stage completion and dependencies.")


if __name__ == "__main__":
    update_main_pipeline()