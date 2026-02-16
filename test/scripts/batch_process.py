#!/usr/bin/env python3
"""
Batch process multiple video files in a directory.

Usage:
    python scripts/batch_process.py --video-dir data/videos/ --camera-id cam_001
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    parser = argparse.ArgumentParser(description='Batch process video files')
    parser.add_argument('--video-dir', type=str, required=True,
                       help='Directory containing video files')
    parser.add_argument('--camera-id', type=str, required=True,
                       help='Camera ID (e.g., cam_001)')
    parser.add_argument('--extensions', type=str, nargs='+',
                       default=['.mp4', '.mov', '.avi', '.mkv'],
                       help='Video file extensions to process')
    
    args = parser.parse_args()
    
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"❌ Error: Directory not found: {video_dir}")
        sys.exit(1)
    
    # Find all video files
    video_files = []
    for ext in args.extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
        video_files.extend(video_dir.glob(f"*{ext.upper()}"))
    
    if not video_files:
        print(f"❌ No video files found in {video_dir}")
        print(f"   Looking for extensions: {args.extensions}")
        sys.exit(1)
    
    print("=" * 60)
    print("Batch Video Processing")
    print("=" * 60)
    print(f"Directory: {video_dir}")
    print(f"Camera ID: {args.camera_id}")
    print(f"Found {len(video_files)} video file(s)")
    print()
    
    for i, video_file in enumerate(sorted(video_files), 1):
        print(f"[{i}/{len(video_files)}] Processing: {video_file.name}")
        # TODO: Call process_video.py for each file
        print(f"    ⚠️  Processing not yet implemented")
    
    print()
    print("✅ Batch processing complete!")

if __name__ == "__main__":
    main()


