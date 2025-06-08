#!/usr/bin/env python3
"""Simple fix for metrics table timing issue."""

import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def add_model_count_verification():
    """Add model count verification to metrics table generation."""
    
    metrics_path = project_root / "src" / "visualization" / "plots" / "metrics.py"
    
    with open(metrics_path, 'r') as f:
        content = f.read()
    
    # Find the create_metrics_table function
    func_start = content.find("def create_metrics_table(")
    if func_start < 0:
        print("Could not find create_metrics_table function")
        return
    
    # Find the first line after the docstring
    func_body_start = content.find('"""', func_start + 50)
    if func_body_start > 0:
        func_body_start = content.find('\n', func_body_start + 3) + 1
    else:
        func_body_start = content.find('\n', func_start) + 1
    
    # Check if verification already exists
    if "expected_model_count" not in content[func_start:func_body_start + 500]:
        # Add verification code
        indent = "    "
        verification_code = f"""
{indent}# Verify we have all expected models
{indent}expected_model_count = 32  # Total expected models
{indent}actual_count = len(models) if isinstance(models, list) else len(list(models.values()))
{indent}
{indent}if actual_count < expected_model_count:
{indent}    print(f"⚠️  Warning: Only {{actual_count}} models found, expected {{expected_model_count}}")
{indent}    print("   Some models may still be training. Waiting 10 seconds...")
{indent}    import time
{indent}    time.sleep(10)
{indent}    
{indent}    # Try to reload models
{indent}    from src.utils.io import load_all_models
{indent}    models = load_all_models()
{indent}    actual_count = len(models) if isinstance(models, list) else len(list(models.values()))
{indent}    print(f"   After reload: {{actual_count}} models found")
"""
        
        content = content[:func_body_start] + verification_code + content[func_body_start:]
    
    # Write back
    with open(metrics_path, 'w') as f:
        f.write(content)
    
    print("✓ Added model count verification to metrics table generation")


def add_visualization_delay():
    """Add a small delay before visualization to ensure models are saved."""
    
    viz_path = project_root / "src" / "visualization" / "__init__.py"
    
    with open(viz_path, 'r') as f:
        content = f.read()
    
    # Find run_comprehensive_visualization_pipeline
    func_name = "run_comprehensive_visualization_pipeline"
    func_start = content.find(f"def {func_name}(")
    
    if func_start > 0:
        # Find the first line after docstring
        func_body_start = content.find('"""', func_start + 50)
        if func_body_start > 0:
            func_body_start = content.find('\n', func_body_start + 3) + 1
        else:
            func_body_start = content.find('\n', func_start) + 1
        
        # Check if delay already exists
        if "Model loading delay" not in content[func_start:func_body_start + 200]:
            delay_code = """
    # Model loading delay to ensure all models are saved
    print("Waiting for model files to be fully written...")
    import time
    time.sleep(5)
"""
            content = content[:func_body_start] + delay_code + content[func_body_start:]
            
            with open(viz_path, 'w') as f:
                f.write(content)
            
            print("✓ Added visualization delay")


def main():
    """Apply timing fixes."""
    print("Applying metrics table timing fixes...\n")
    
    add_model_count_verification()
    add_visualization_delay()
    
    print("\n✓ Timing fixes applied")
    print("\nThe metrics table will now:")
    print("1. Check if all 32 models are present")
    print("2. Wait and retry if models are missing")
    print("3. Show a warning if models are still incomplete")


if __name__ == "__main__":
    main()