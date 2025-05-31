#!/usr/bin/env python3
"""Demonstrate the archive and clean functionality."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings
import os

def show_outputs_status():
    """Show current status of outputs directory."""
    if not settings.OUTPUT_DIR.exists():
        print("❌ Outputs directory does not exist")
        return
        
    # Count files and subdirectories
    file_count = 0
    dir_count = 0
    total_size = 0
    
    for item in settings.OUTPUT_DIR.rglob('*'):
        if item.is_file():
            file_count += 1
            total_size += item.stat().st_size
        elif item.is_dir():
            dir_count += 1
    
    print(f"📁 Outputs directory status:")
    print(f"   • Files: {file_count}")
    print(f"   • Directories: {dir_count}")
    print(f"   • Total size: {total_size / (1024*1024):.1f} MB")
    
    # Show first few files as examples
    if file_count > 0:
        print("\n   Example files:")
        count = 0
        for item in settings.OUTPUT_DIR.rglob('*'):
            if item.is_file() and count < 5:
                relative_path = item.relative_to(settings.OUTPUT_DIR)
                print(f"     - {relative_path}")
                count += 1
        if file_count > 5:
            print(f"     ... and {file_count - 5} more files")

print("=" * 60)
print("DEMONSTRATING ARCHIVE AND CLEAN FUNCTIONALITY")
print("=" * 60)

print("\n1️⃣  BEFORE archiving:")
show_outputs_status()

print("\n2️⃣  Running archive_before_amendments.py...")
print("   This will:")
print("   • Create a complete archive of outputs/")
print("   • Clean the outputs/ directory")
print("   • Leave you with a fresh, empty outputs/ directory")

print("\n" + "-" * 60)
print("After running: python archive_before_amendments.py")
print("-" * 60)

print("\n3️⃣  AFTER archiving (expected):")
print("📁 Outputs directory status:")
print("   • Files: 0")
print("   • Directories: (only empty structure remains)")
print("   • Total size: 0 MB")
print("\n✅ Ready for fresh pipeline run with clean outputs!")

print("\n💡 The archive preserves everything and can be found in the parent directory")
print("   with a name like: outputs_pre_amendment_YYYYMMDD_HHMMSS/")