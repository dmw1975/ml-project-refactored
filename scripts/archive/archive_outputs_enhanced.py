"""Enhanced archive script that can archive the entire outputs directory.

This script:
1. Creates a timestamped archive of the entire outputs directory or just visualizations
2. Optionally cleans the archived directories
3. Creates comprehensive documentation of what was archived
4. Can generate a comparison report between archives
"""

import os
import sys
import shutil
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root))

# Import settings
from src.config import settings


def get_directory_stats(directory: Path) -> Dict:
    """Get statistics about a directory."""
    stats = {
        'total_files': 0,
        'total_dirs': 0,
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
            stats['total_dirs'] += 1
            
    # Get immediate subdirectories
    stats['subdirectories'] = [d.name for d in directory.iterdir() if d.is_dir()]
    
    return stats


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def create_archive_manifest(archive_dir: Path, source_dir: Path, stats: Dict):
    """Create a detailed manifest of the archive."""
    manifest = {
        'archive_timestamp': datetime.now().isoformat(),
        'source_directory': str(source_dir),
        'archive_directory': str(archive_dir),
        'statistics': stats,
        'statistics_formatted': {
            'total_size': format_size(stats['total_size']),
            'total_files': stats['total_files'],
            'total_directories': stats['total_dirs'],
            'file_types': stats['file_types'],
            'subdirectories': stats['subdirectories']
        }
    }
    
    # Save as JSON
    with open(archive_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Save as human-readable README
    with open(archive_dir / 'README.md', 'w') as f:
        f.write(f"""# Archive of {source_dir.name}

## Archive Information
- **Created**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Source**: `{source_dir}`
- **Archive**: `{archive_dir}`

## Statistics
- **Total Size**: {manifest['statistics_formatted']['total_size']}
- **Total Files**: {manifest['statistics_formatted']['total_files']}
- **Total Directories**: {manifest['statistics_formatted']['total_directories']}

## File Types
""")
        for ext, count in sorted(stats['file_types'].items()):
            f.write(f"- `{ext or 'no extension'}`: {count} files\n")
        
        f.write(f"\n## Subdirectories\n")
        for subdir in sorted(stats['subdirectories']):
            f.write(f"- `{subdir}/`\n")
        
        f.write(f"""
## Purpose
This archive was created to:
1. Preserve the current state of the outputs before making changes
2. Allow comparison of outputs before and after pipeline modifications
3. Provide a rollback option if needed

## Usage
- To restore this archive: Copy contents back to `{source_dir}`
- To compare with new outputs: Use the comparison script or manually diff directories
""")


def archive_directory(source_dir: Path, archive_name: str = None, clean_after: bool = False) -> Path:
    """Archive a directory with detailed documentation."""
    # Generate archive name if not provided
    if archive_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"{source_dir.name}_archive_{timestamp}"
    
    # Create archive directory
    archive_dir = source_dir.parent / archive_name
    os.makedirs(archive_dir, exist_ok=True)
    
    print(f"📦 Creating archive: {archive_dir}")
    
    # Get statistics before archiving
    print("📊 Analyzing source directory...")
    stats = get_directory_stats(source_dir)
    
    # Copy entire directory
    if source_dir.exists():
        print(f"📋 Copying {stats['total_files']} files ({format_size(stats['total_size'])})...")
        shutil.copytree(source_dir, archive_dir / source_dir.name, dirs_exist_ok=True)
        print("✅ Copy complete!")
    else:
        print(f"⚠️  Source directory {source_dir} does not exist!")
        return None
    
    # Create manifest
    print("📝 Creating archive manifest...")
    create_archive_manifest(archive_dir, source_dir, stats)
    
    # Clean source directory if requested
    if clean_after:
        print(f"🧹 Cleaning source directory: {source_dir}")
        clean_directory(source_dir)
    
    return archive_dir


def clean_directory(directory: Path, preserve_structure: bool = True, file_types: List[str] = None):
    """Clean a directory, optionally preserving structure."""
    if not directory.exists():
        print(f"⚠️  Directory {directory} does not exist!")
        return
    
    if file_types is None:
        # Default to common output file types
        file_types = ['.png', '.jpg', '.jpeg', '.svg', '.pdf', '.csv', '.pkl', '.json']
    
    removed_count = 0
    for item in directory.rglob('*'):
        if item.is_file():
            if not file_types or item.suffix.lower() in file_types:
                try:
                    os.remove(item)
                    removed_count += 1
                except Exception as e:
                    print(f"❌ Error removing {item}: {e}")
    
    print(f"🗑️  Removed {removed_count} files")
    
    if not preserve_structure:
        # Remove empty directories
        for item in sorted(directory.rglob('*'), reverse=True):
            if item.is_dir() and not any(item.iterdir()):
                try:
                    item.rmdir()
                except Exception as e:
                    print(f"❌ Error removing empty directory {item}: {e}")


def compare_archives(archive1: Path, archive2: Path) -> Dict:
    """Compare two archives and report differences."""
    print(f"\n📊 Comparing archives:")
    print(f"  1️⃣  {archive1}")
    print(f"  2️⃣  {archive2}")
    
    # Load manifests
    manifest1 = {}
    manifest2 = {}
    
    try:
        with open(archive1 / 'manifest.json', 'r') as f:
            manifest1 = json.load(f)
    except:
        print("⚠️  Could not load manifest for archive 1")
    
    try:
        with open(archive2 / 'manifest.json', 'r') as f:
            manifest2 = json.load(f)
    except:
        print("⚠️  Could not load manifest for archive 2")
    
    # Compare statistics
    comparison = {
        'size_change': manifest2.get('statistics', {}).get('total_size', 0) - 
                      manifest1.get('statistics', {}).get('total_size', 0),
        'files_change': manifest2.get('statistics', {}).get('total_files', 0) - 
                       manifest1.get('statistics', {}).get('total_files', 0),
        'new_file_types': [],
        'removed_file_types': [],
        'new_subdirectories': [],
        'removed_subdirectories': []
    }
    
    # Compare file types
    types1 = set(manifest1.get('statistics', {}).get('file_types', {}).keys())
    types2 = set(manifest2.get('statistics', {}).get('file_types', {}).keys())
    comparison['new_file_types'] = list(types2 - types1)
    comparison['removed_file_types'] = list(types1 - types2)
    
    # Compare subdirectories
    dirs1 = set(manifest1.get('statistics', {}).get('subdirectories', []))
    dirs2 = set(manifest2.get('statistics', {}).get('subdirectories', []))
    comparison['new_subdirectories'] = list(dirs2 - dirs1)
    comparison['removed_subdirectories'] = list(dirs1 - dirs2)
    
    # Print comparison report
    print("\n📈 Comparison Results:")
    print(f"  • Size change: {format_size(abs(comparison['size_change']))} "
          f"({'increase' if comparison['size_change'] > 0 else 'decrease'})")
    print(f"  • Files change: {comparison['files_change']:+d}")
    
    if comparison['new_file_types']:
        print(f"  • New file types: {', '.join(comparison['new_file_types'])}")
    if comparison['removed_file_types']:
        print(f"  • Removed file types: {', '.join(comparison['removed_file_types'])}")
    if comparison['new_subdirectories']:
        print(f"  • New subdirectories: {', '.join(comparison['new_subdirectories'])}")
    if comparison['removed_subdirectories']:
        print(f"  • Removed subdirectories: {', '.join(comparison['removed_subdirectories'])}")
    
    return comparison


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Archive and clean output directories')
    parser.add_argument('--full', action='store_true', 
                       help='Archive entire outputs directory (default: visualizations only)')
    parser.add_argument('--clean', action='store_true',
                       help='Clean directories after archiving')
    parser.add_argument('--compare', nargs=2, metavar=('ARCHIVE1', 'ARCHIVE2'),
                       help='Compare two archives')
    parser.add_argument('--name', type=str,
                       help='Custom archive name (default: timestamped)')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare mode
        archive1 = Path(args.compare[0])
        archive2 = Path(args.compare[1])
        compare_archives(archive1, archive2)
    else:
        # Archive mode
        if args.full:
            # Archive entire outputs directory
            print("🗄️  Archiving entire outputs directory...")
            source_dir = settings.OUTPUT_DIR
        else:
            # Archive visualizations only (backward compatibility)
            print("🎨 Archiving visualizations directory...")
            source_dir = settings.VISUALIZATION_DIR
        
        archive_dir = archive_directory(source_dir, args.name, args.clean)
        
        if archive_dir:
            print(f"\n✅ Archive complete!")
            print(f"📍 Location: {archive_dir}")
            print(f"\n💡 Tip: To compare with future outputs, run:")
            print(f"   python {Path(__file__).name} --compare \"{archive_dir}\" \"<future_archive>\"")


if __name__ == "__main__":
    main()