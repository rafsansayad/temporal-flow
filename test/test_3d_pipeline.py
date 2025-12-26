"""
Test 3D Generation Pipeline.

1. Auto-detect object with Owl-ViT
2. Segment and Estimate Depth
3. Generate 3D Point Cloud (.ply)
4. Create Rotating GIF
"""

import cv2
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline import VideoPipeline
from app.processing.point_cloud import depth_to_point_cloud, save_ply, save_rotating_gif

# Config
VIDEO_PATH = "samples/Test vid-2 .mp4"
TEXT_PROMPT = "a person's face"
OUTPUT_DIR = "outputs/3d_test"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Initialize Pipeline
pipeline = VideoPipeline(use_owl_vit=True)

# 2. Get Data
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error reading video")
    sys.exit(1)

# 3. Process Frame
# Auto-detect to init tracker
success = pipeline.auto_detect_and_init(frame, TEXT_PROMPT)
if not success:
    print("Detection failed")
    sys.exit(1)

# Process to get mask and depth
mask, depth, box = pipeline.process_frame(frame)

# 4. Generate 3D Point Cloud
print("Generating 3D data...")

# Convert BGR to RGB for correct colors
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Create point cloud (subsample=4 for speed)
points, colors = depth_to_point_cloud(frame_rgb, depth, mask, subsample=4)

print(f"Generated {len(points)} 3D points.")

# 5. Export
if len(points) > 0:
    # Save PLY
    ply_path = os.path.join(OUTPUT_DIR, "object.ply")
    save_ply(points, colors, ply_path)
    
    # Save GIF
    gif_path = os.path.join(OUTPUT_DIR, "object_3d.gif")
    save_rotating_gif(points, colors, gif_path, frames=20)
else:
    print("No points generated (mask empty?)")

