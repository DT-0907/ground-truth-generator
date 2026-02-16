#!/usr/bin/env python3
"""
Main CLI for video processing.

Usage:
    python process.py <video_path>                    # Process single video
    python process.py --batch <video_dir>             # Process all videos in directory
    python process.py --feedback                      # Analyze corrections, build feedback
    python process.py --export                        # Export COCO dataset for fine-tuning
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.processor import process_video
from src.feedback import analyze_corrections, export_coco


def main():
    parser = argparse.ArgumentParser(
        description='Vehicle Detection & Tracking Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process.py data/videos/traffic.mp4           # Process single video
  python process.py --batch data/videos/              # Process all videos
  python process.py --feedback                        # Build feedback from corrections
  python process.py --export                          # Export COCO dataset
        """
    )

    parser.add_argument('video', nargs='?', help='Path to video file')
    parser.add_argument('--batch', metavar='DIR', help='Process all videos in directory')
    parser.add_argument('--model', default='yolov8m.pt', help='YOLO model (default: yolov8m.pt)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold (default: 0.25)')
    parser.add_argument('--output', default='data/tracks', help='Output directory for tracks')
    parser.add_argument('--feedback', action='store_true', help='Analyze corrections and build feedback')
    parser.add_argument('--export', action='store_true', help='Export COCO dataset for fine-tuning')
    parser.add_argument('--sample-rate', type=int, default=10, help='Frame sample rate for COCO export (default: 10)')

    args = parser.parse_args()

    # Check for feedback file
    feedback_file = Path('config/feedback.json')
    feedback_path = str(feedback_file) if feedback_file.exists() else None

    if args.feedback:
        print("Analyzing corrections...")
        analyze_corrections('data/tracks', 'data/corrections', 'config/feedback.json')
        return

    if args.export:
        print("Exporting COCO dataset...")
        export_coco('data/tracks', 'data/corrections', 'data/videos',
                   'data/exports', args.sample_rate)
        return

    if args.batch:
        # Batch processing
        video_dir = Path(args.batch)
        if not video_dir.exists():
            print(f"Error: Directory not found: {video_dir}")
            sys.exit(1)

        video_files = []
        for ext in ['*.mp4', '*.mov', '*.avi', '*.mkv', '*.MP4', '*.MOV']:
            video_files.extend(video_dir.glob(ext))

        if not video_files:
            print(f"No video files found in {video_dir}")
            sys.exit(1)

        print(f"Found {len(video_files)} videos to process")
        print()

        for i, video_file in enumerate(sorted(video_files), 1):
            print(f"[{i}/{len(video_files)}] Processing: {video_file.name}")
            print("-" * 50)
            try:
                process_video(
                    str(video_file),
                    args.output,
                    args.model,
                    args.conf,
                    feedback_path
                )
            except Exception as e:
                print(f"Error processing {video_file.name}: {e}")
            print()

        print("Batch processing complete!")
        return

    if args.video:
        # Single video processing
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video not found: {video_path}")
            sys.exit(1)

        process_video(
            str(video_path),
            args.output,
            args.model,
            args.conf,
            feedback_path
        )
        return

    # No arguments - show help
    parser.print_help()


if __name__ == '__main__':
    main()
