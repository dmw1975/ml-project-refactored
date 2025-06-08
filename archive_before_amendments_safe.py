#!/usr/bin/env python3
"""Enhanced script to archive entire outputs directory and ensure complete cleanup.

This creates a complete snapshot of the current outputs and ensures the outputs
directory is completely clean for a fresh pipeline run.
"""

import sys
import os
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from src.config import settings


def get_directory_stats(directory: Path) -> dict:
    """Get comprehensive statistics about a directory."""
    stats = {
        'total_files': 0,
        'total_size': 0,
        'file_types': {},
        'subdirectories': []
    }
    
    if not directory.exists():
        return stats
    
    for item in directory.rglob('*'):
        if item.is_file():
            stats['total_files'] += 1
            stats['total_size'] += item.stat().st_size
            ext = item.suffix.lower()
            stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1
        elif item.is_dir():
            rel_path = item.relative_to(directory)
            if str(rel_path) != '.':
                stats['subdirectories'].append(str(rel_path))
    
    return stats


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def verify_critical_files(directory: Path) -> dict:
    """Check for critical files that must be archived and cleaned."""
    critical_files = {
        'pipeline_state.json': directory / 'pipeline_state.json',
        'model_files': list(directory.glob('models/*.pkl')),
        'all_pkl_files': list(directory.rglob('*.pkl')),
        'all_json_files': list(directory.rglob('*.json'))
    }
    
    report = {
        'pipeline_state_exists': critical_files['pipeline_state.json'].exists(),
        'model_count': len(critical_files['model_files']),
        'total_pkl_count': len(critical_files['all_pkl_files']),
        'total_json_count': len(critical_files['all_json_files'])
    }
    
    return report, critical_files


def complete_clean(directory: Path):
    """Completely clean the outputs directory, removing ALL files and subdirectories."""
    if not directory.exists():
        print(f"⚠️  Directory {directory} does not exist!")
        return
    
    print(f"\n🧹 Performing complete cleanup of {directory}")
    
    # Count items before cleaning
    file_count = sum(1 for _ in directory.rglob('*') if _.is_file())
    dir_count = sum(1 for _ in directory.rglob('*') if _.is_dir())
    
    print(f"  • Files to remove: {file_count}")
    print(f"  • Directories to remove: {dir_count}")
    
    # Remove all contents but preserve the outputs directory itself
    removed_files = 0
    removed_dirs = 0
    errors = []
    
    # First pass: Remove all files
    for item in directory.rglob('*'):
        if item.is_file():
            try:
                os.remove(item)
                removed_files += 1
            except Exception as e:
                errors.append(f"File {item}: {e}")
    
    # Second pass: Remove all directories (bottom-up)
    for item in sorted(directory.rglob('*'), reverse=True):
        if item.is_dir() and item != directory:
            try:
                item.rmdir()
                removed_dirs += 1
            except Exception as e:
                # Try shutil.rmtree for non-empty directories
                try:
                    shutil.rmtree(item)
                    removed_dirs += 1
                except Exception as e2:
                    errors.append(f"Directory {item}: {e2}")
    
    print(f"\n✅ Cleanup complete:")
    print(f"  • Files removed: {removed_files}")
    print(f"  • Directories removed: {removed_dirs}")
    
    if errors:
        print(f"\n⚠️  Errors encountered during cleanup:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  • {error}")
        if len(errors) > 5:
            print(f"  • ... and {len(errors) - 5} more errors")
    
    # Recreate essential subdirectories
    essential_dirs = ['models', 'visualizations', 'metrics', 'reports']
    for subdir in essential_dirs:
        (directory / subdir).mkdir(exist_ok=True)
    print(f"\n📁 Recreated essential subdirectories: {', '.join(essential_dirs)}")


def archive_outputs_before_amendments():
    """Archive the entire outputs directory with a descriptive name and ensure complete cleanup."""
    print("=" * 80)
    print("📸 CREATING PRE-AMENDMENT SNAPSHOT & COMPLETE CLEANUP")
    print("=" * 80)
    
    # Check if outputs directory exists
    if not settings.OUTPUT_DIR.exists():
        print(f"⚠️  No outputs directory found at {settings.OUTPUT_DIR}")
        print("Nothing to archive. Creating outputs directory structure...")
        settings.OUTPUT_DIR.mkdir(exist_ok=True)
        essential_dirs = ['models', 'visualizations', 'metrics', 'reports']
        for subdir in essential_dirs:
            (settings.OUTPUT_DIR / subdir).mkdir(exist_ok=True)
        print("✅ Created outputs directory structure")
        return None
    
    # Get current stats
    print("\n📊 Current outputs directory statistics:")
    stats = get_directory_stats(settings.OUTPUT_DIR)
    print(f"  • Total files: {stats['total_files']}")
    print(f"  • Total size: {format_size(stats['total_size'])}")
    print(f"  • Subdirectories: {len(stats['subdirectories'])}")
    
    # Check for critical files
    print("\n🔍 Checking for critical files:")
    report, critical_files = verify_critical_files(settings.OUTPUT_DIR)
    print(f"  • pipeline_state.json: {'EXISTS' if report['pipeline_state_exists'] else 'NOT FOUND'}")
    print(f"  • Model files (*.pkl): {report['model_count']} files")
    print(f"  • Total PKL files: {report['total_pkl_count']} files")
    print(f"  • Total JSON files: {report['total_json_count']} files")
    
    # Create archive with descriptive name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"outputs_pre_amendment_{timestamp}"
    archive_dir = project_root / archive_name
    
    print(f"\n🗄️  Creating archive: {archive_name}")
    
    # Copy entire outputs directory
    try:
        shutil.copytree(settings.OUTPUT_DIR, archive_dir, dirs_exist_ok=True)
        print("✅ Archive created successfully!")
    except Exception as e:
        print(f"❌ Error creating archive: {e}")
        return None
    
    # Verify archive
    archive_stats = get_directory_stats(archive_dir)
    print(f"\n📋 Archive verification:")
    print(f"  • Files archived: {archive_stats['total_files']}")
    print(f"  • Archive size: {format_size(archive_stats['total_size'])}")
    
    # Create archive manifest
    manifest_path = archive_dir / "archive_manifest.txt"
    with open(manifest_path, 'w') as f:
        f.write(f"Archive created: {datetime.now().isoformat()}\n")
        f.write(f"Source: {settings.OUTPUT_DIR}\n")
        f.write(f"Total files: {stats['total_files']}\n")
        f.write(f"Total size: {format_size(stats['total_size'])}\n")
        f.write(f"\nCritical files archived:\n")
        f.write(f"- pipeline_state.json: {'YES' if report['pipeline_state_exists'] else 'NO'}\n")
        f.write(f"- Model files: {report['model_count']}\n")
        f.write(f"- All PKL files: {report['total_pkl_count']}\n")
        f.write(f"- All JSON files: {report['total_json_count']}\n")
    
    # Perform complete cleanup
    complete_clean(settings.OUTPUT_DIR)
    
    # Verify cleanup
    post_clean_stats = get_directory_stats(settings.OUTPUT_DIR)
    print(f"\n🔍 Post-cleanup verification:")
    print(f"  • Remaining files: {post_clean_stats['total_files']} (should be 0)")
    print(f"  • Remaining size: {format_size(post_clean_stats['total_size'])}")
    
    # Final success message
    print("\n" + "=" * 80)
    print("✅ PRE-AMENDMENT SNAPSHOT COMPLETE & OUTPUTS COMPLETELY CLEANED!")
    print("=" * 80)
    print(f"\n📍 Archive location: {archive_dir}")
    print("🧹 Outputs directory has been completely cleaned - ready for fresh pipeline run!")
    print("\n🔄 Next steps:")
    print("  1. Run: python run_pipeline_safe.py --all --non-interactive --extended-timeout")
    print("  2. Monitor the pipeline execution")
    print("  3. Compare results with the archive if needed")
    
    # Create verification script
    verification_script = f"""#!/bin/bash
# Quick verification that outputs is clean and ready

echo "Checking outputs directory..."
echo "Files in outputs/: $(find outputs -type f | wc -l) (should be 0)"
echo "Directories in outputs/: $(find outputs -type d | wc -l)"
echo ""
echo "Checking for pipeline state file..."
if [ -f "outputs/pipeline_state.json" ]; then
    echo "WARNING: pipeline_state.json still exists!"
else
    echo "✓ No pipeline state file found (good)"
fi
echo ""
echo "Checking for model files..."
MODEL_COUNT=$(find outputs -name "*.pkl" | wc -l)
if [ $MODEL_COUNT -gt 0 ]; then
    echo "WARNING: Found $MODEL_COUNT .pkl files!"
else
    echo "✓ No model files found (good)"
fi
"""
    
    script_path = project_root / f"verify_clean_{timestamp}.sh"
    with open(script_path, 'w') as f:
        f.write(verification_script)
    script_path.chmod(0o755)
    
    print(f"\n💡 Verification script created: {script_path}")
    print(f"   Run this to verify outputs is clean before starting pipeline")
    
    return archive_dir


if __name__ == "__main__":
    archive_outputs_before_amendments()