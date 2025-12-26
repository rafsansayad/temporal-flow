"""Test Owl-ViT zero-shot object detection."""

import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.owl_vit import OwlViT
from config import DEVICE

# Load model
owl = OwlViT(DEVICE)
print(f"Owl-ViT loaded on {DEVICE}")

# Load test video first frame
video_path = "samples/Test vid-2 .mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error: Could not read video")
    sys.exit(1)

# Detect objects
text_query = "a person's face"
print(f"Detecting: '{text_query}'")

detections = owl.detect(frame, [text_query], threshold=0.1)
print(f"Found {len(detections)} detections")

# Visualize all detections
output = frame.copy()
for i, (x, y, w, h, score) in enumerate(detections):
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(output, f"{score:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print(f"  [{i}] Box: ({x}, {y}, {w}, {h}), Score: {score:.3f}")

# Get best detection
best_box = owl.get_best_detection(frame, text_query, threshold=0.1)
print(f"Best detection: {best_box}")

# Save results
cv2.imwrite("outputs/test_owl_vit_input.png", frame)
cv2.imwrite("outputs/test_owl_vit_detections.png", output)
print("Saved to outputs/")

