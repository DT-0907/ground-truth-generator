#!/usr/bin/env python3
"""
View count summaries for processed videos.

Usage:
    python scripts/view_counts.py --camera-id cam_001
"""

import argparse
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    parser = argparse.ArgumentParser(description='View count summaries')
    parser.add_argument('--camera-id', type=str, required=True,
                       help='Camera ID (e.g., cam_001)')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory (default: data)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    analytics_dir = data_dir / 'analytics'
    
    if not analytics_dir.exists():
        print(f"❌ Analytics directory not found: {analytics_dir}")
        print("   Process videos first with: python scripts/process_video.py")
        sys.exit(1)
    
    # Find analytics files for this camera
    analytics_files = list(analytics_dir.glob(f"{args.camera_id}*.csv"))
    analytics_files.extend(list(analytics_dir.glob(f"{args.camera_id}*.json")))
    
    if not analytics_files:
        print(f"❌ No analytics found for camera: {args.camera_id}")
        print(f"   Looked in: {analytics_dir}")
        print("   Process videos first with: python scripts/process_video.py")
        sys.exit(1)
    
    print("=" * 60)
    print(f"Count Summaries for {args.camera_id}")
    print("=" * 60)
    print()
    
    for analytics_file in sorted(analytics_files):
        print(f"📊 {analytics_file.name}")
        print("-" * 60)
        
        if analytics_file.suffix == '.json':
            # JSON format
            try:
                with open(analytics_file, 'r') as f:
                    data = json.load(f)
                print(json.dumps(data, indent=2))
            except Exception as e:
                print(f"   Error reading file: {e}")
        else:
            # CSV format
            try:
                import pandas as pd
                df = pd.read_csv(analytics_file)
                print(df.to_string(index=False))
            except ImportError:
                # Fallback: read CSV manually
                with open(analytics_file, 'r') as f:
                    lines = f.readlines()[:20]  # First 20 lines
                    print(''.join(lines))
                    if len(f.readlines()) > 20:
                        print("   ... (truncated)")
            except Exception as e:
                print(f"   Error reading file: {e}")
        
        print()
    
    print("=" * 60)
    print("💡 Tip: Launch correction UI with: python ui/app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()


