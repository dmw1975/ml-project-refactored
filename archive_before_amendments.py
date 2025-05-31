#!/usr/bin/env python3
"""Quick script to archive entire outputs directory before making amendments.

This creates a complete snapshot of the current outputs so you can see
exactly what changes after running the amended pipeline.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

# Import the enhanced archive functionality
from scripts.archive.archive_outputs_enhanced import archive_directory, get_directory_stats, format_size
from src.config import settings


def archive_outputs_before_amendments():
    """Archive the entire outputs directory with a descriptive name and clean it."""
    print("=" * 60)
    print("📸 CREATING PRE-AMENDMENT SNAPSHOT & CLEANING")
    print("=" * 60)
    
    # Check if outputs directory exists
    if not settings.OUTPUT_DIR.exists():
        print(f"⚠️  No outputs directory found at {settings.OUTPUT_DIR}")
        print("Nothing to archive. Run the pipeline first to generate outputs.")
        return None
    
    # Get current stats
    print("\n📊 Current outputs directory statistics:")
    stats = get_directory_stats(settings.OUTPUT_DIR)
    print(f"  • Total files: {stats['total_files']}")
    print(f"  • Total size: {format_size(stats['total_size'])}")
    print(f"  • Subdirectories: {', '.join(stats['subdirectories'])}")
    
    # Create archive with descriptive name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"outputs_pre_amendment_{timestamp}"
    
    print(f"\n🗄️  Creating archive: {archive_name}")
    # Set clean_after=True to automatically clean the directory
    archive_dir = archive_directory(settings.OUTPUT_DIR, archive_name, clean_after=True)
    
    if archive_dir:
        print("\n" + "=" * 60)
        print("✅ PRE-AMENDMENT SNAPSHOT COMPLETE & OUTPUTS CLEANED!")
        print("=" * 60)
        print(f"\n📍 Archive location: {archive_dir}")
        print("🧹 Outputs directory has been cleaned - ready for fresh results!")
        print("\n🔄 Next steps:")
        print("  1. Run your amended pipeline: python main.py --all")
        print("  2. Create post-amendment archive: python scripts/archive/archive_outputs_enhanced.py --full")
        print("  3. Compare the changes:")
        print(f"     python scripts/archive/archive_outputs_enhanced.py --compare \"{archive_dir}\" \"<post_amendment_archive>\"")
        
        # Create a convenience script for comparison
        comparison_script = f"""#!/bin/bash
# Convenience script to compare pre and post amendment outputs

echo "Creating post-amendment archive..."
python scripts/archive/archive_outputs_enhanced.py --full --name outputs_post_amendment_{timestamp}

echo "\\nComparing archives..."
python scripts/archive/archive_outputs_enhanced.py --compare "{archive_dir}" "outputs_post_amendment_{timestamp}"
"""
        
        script_path = project_root / f"compare_amendments_{timestamp}.sh"
        with open(script_path, 'w') as f:
            f.write(comparison_script)
        script_path.chmod(0o755)
        
        print(f"\n💡 Convenience script created: {script_path}")
        print(f"   Run this after your amendments to see what changed")
    
    return archive_dir


if __name__ == "__main__":
    archive_outputs_before_amendments()