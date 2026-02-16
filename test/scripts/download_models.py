#!/usr/bin/env python3
"""
Pre-download YOLO models to avoid delays on first run.
"""

import os
from pathlib import Path
from ultralytics import YOLO

def download_models():
    """Download recommended YOLO models."""
    
    # Create models directory
    models_dir = Path(__file__).parent.parent / "models" / "yolo"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Models to download (you can customize this list)
    models = [
        "yolov8n.pt",  # Nano - fastest
        "yolov8m.pt",  # Medium - recommended for production
    ]
    
    print("Downloading YOLO models...")
    print(f"Models will be saved to: {models_dir}\n")
    
    for model_name in models:
        model_path = models_dir / model_name
        
        if model_path.exists():
            print(f"✓ {model_name} already exists, skipping...")
            continue
        
        print(f"Downloading {model_name}...")
        try:
            # This will download the model if not present
            model = YOLO(model_name)
            print(f"✓ {model_name} downloaded successfully\n")
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}\n")
    
    print("Model download complete!")
    print(f"\nModels location: {models_dir}")
    print("\nYou can now use these models offline.")

if __name__ == "__main__":
    download_models()


