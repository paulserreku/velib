#!/usr/bin/env python3
"""Resize animation frames to reduce file size"""

from PIL import Image
import os
from pathlib import Path

# Configuration
FRAMES_DIR = "velib_animation_districts_frames"
OUTPUT_DIR = "velib_animation_districts_frames_resized"
SCALE_FACTOR = 0.7  # Reduce to 70% (1400x1200 -> 980x840)

def resize_frames():
    """Resize all PNG frames in the directory"""
    frames_path = Path(FRAMES_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    frames = sorted(frames_path.glob("frame*.png"))
    total = len(frames)
    
    print(f"Resizing {total} frames to {SCALE_FACTOR*100}% of original size...")
    
    for i, frame_file in enumerate(frames, 1):
        img = Image.open(frame_file)
        new_size = (int(img.width * SCALE_FACTOR), int(img.height * SCALE_FACTOR))
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
        
        output_file = output_path / frame_file.name
        img_resized.save(output_file, "PNG", optimize=True)
        
        if i % 50 == 0:
            print(f"Progress: {i}/{total} frames processed")
    
    print(f"Done! Resized frames saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    resize_frames()
