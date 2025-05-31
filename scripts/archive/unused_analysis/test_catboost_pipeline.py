#!/usr/bin/env python3
"""Test CatBoost through the main pipeline to reproduce the error."""

import sys
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def main():
    """Run main pipeline visualization for CatBoost only."""
    print("Testing CatBoost visualization through main pipeline...")
    
    # Run the visualization part of main.py
    cmd = [
        sys.executable, 
        "main.py", 
        "--visualize"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run and capture output
    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True
    )
    
    # Check for the specific error
    if "truth value" in result.stderr and "array" in result.stderr:
        print("FOUND THE ERROR!")
        print("\nSTDERR:")
        print(result.stderr)
    else:
        print("No array comparison error found in stderr")
        
    # Also check stdout
    if "truth value" in result.stdout and "array" in result.stdout:
        print("\nFOUND THE ERROR IN STDOUT!")
        print("\nRelevant output:")
        lines = result.stdout.split('\n')
        for i, line in enumerate(lines):
            if "truth value" in line and "array" in line:
                # Print context around the error
                start = max(0, i - 5)
                end = min(len(lines), i + 10)
                for j in range(start, end):
                    print(f"{j}: {lines[j]}")
    
    print(f"\nProcess exit code: {result.returncode}")

if __name__ == "__main__":
    main()