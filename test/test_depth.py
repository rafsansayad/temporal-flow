"""Test MiDAS depth estimation."""
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.depth import MiDAS
from config import MIDAS_MODEL, DEVICE

# Load model (downloads on first run)
midas = MiDAS(MIDAS_MODEL, DEVICE)
print("MiDAS loaded")

# Create test gradient image
frame = np.zeros((480, 640, 3), dtype=np.uint8)
for i in range(640):
    frame[:, i] = [i // 3, i // 3, i // 3]

# Estimate depth
depth = midas.estimate(frame)
print(f"Depth shape: {depth.shape}, range: [{depth.min()}, {depth.max()}]")

# Save result
cv2.imwrite("outputs/test_depth_input.png", frame)
cv2.imwrite("outputs/test_depth_output.png", depth)
print("Saved to outputs/")

