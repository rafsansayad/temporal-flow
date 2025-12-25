"""Test MobileSAM segmentation."""
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.sam import MobileSAM
from config import WEIGHTS_DIR, SAM_CHECKPOINT, DEVICE

# Load model
checkpoint = os.path.join(WEIGHTS_DIR, SAM_CHECKPOINT)
sam = MobileSAM(checkpoint, DEVICE)
print("MobileSAM loaded")

# Create test image (white circle on black)
frame = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.circle(frame, (320, 240), 100, (255, 255, 255), -1)

# Segment (click center of circle)
mask = sam.segment(frame, point=(320, 240))
print(f"Mask shape: {mask.shape}, unique values: {np.unique(mask)}")

# Save result
cv2.imwrite("outputs/test_sam_input.png", frame)
cv2.imwrite("outputs/test_sam_mask.png", mask)
print("Saved to outputs/")

