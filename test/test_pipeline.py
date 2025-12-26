"""
End-to-End Pipeline Test.

Validates the full video processing workflow:
1. Object Tracking (CSRT)
2. Segmentation (SAM)
3. Optical Flow Refinement (RAFT/Farneback)
4. Depth Filtering (MiDAS)

Outputs a comprehensive visualization video.
"""

import cv2
import numpy as np
import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline import VideoPipeline

# Configuration
DEFAULT_VIDEO = "samples/Test vid-2 .mp4"
OUTPUT_DIR = "outputs/pipeline_test"
MAX_FRAMES = 300

def create_visualization(frame, mask, depth, box):
    """Create a 3-panel visualization: Source+Box, Depth, Refined Mask."""
    h, w = frame.shape[:2]
    
    # 1. Source with Box
    vis_src = frame.copy()
    x, y, bw, bh = box
    cv2.rectangle(vis_src, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
    cv2.putText(vis_src, "Tracker + Source", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 2. Depth Map (Colorized)
    vis_depth = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
    cv2.putText(vis_depth, "MiDAS Depth", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 3. Final Mask Overlay
    vis_mask = frame.copy()
    vis_mask[mask > 0] = vis_mask[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    cv2.putText(vis_mask, "Refined Mask (Flow+Depth)", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return np.hstack([vis_src, vis_depth, vis_mask])

def main():
    print(f"[INFO] Starting Pipeline Test")
    print(f"[INFO] Video: {DEFAULT_VIDEO}")
    
    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pipeline = VideoPipeline()
    
    # Open Video
    cap = cv2.VideoCapture(DEFAULT_VIDEO)
    if not cap.isOpened():
        print("[ERROR] Cannot open video")
        sys.exit(1)
        
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Get first frame for selection
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read first frame")
        sys.exit(1)
        
    # Resize logic
    h, w = first_frame.shape[:2]
    if w > 640:
        scale = 640 / w
        first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
        h, w = first_frame.shape[:2]
        
    print(f"[INFO] Processing at resolution: {w}x{h}")
    
    # User Interaction
    print("\n[INSTRUCTION] Draw a box around the object to track and press ENTER")
    bbox = cv2.selectROI("Select Object", first_frame, fromCenter=False)
    cv2.destroyAllWindows()
    
    if bbox[2] == 0 or bbox[3] == 0:
        print("[ERROR] No selection made")
        sys.exit(1)
        
    # Initialize Pipeline
    pipeline.init_tracker(first_frame, bbox)
    
    # Setup Video Writer
    out_path = os.path.join(OUTPUT_DIR, "pipeline_result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w * 3, h))
    
    # Processing Loop
    print("\n[INFO] Processing video...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    
    while frame_idx < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Ensure frame matches processing size
        frame_resized = cv2.resize(frame, (w, h))
        
        # Run Pipeline
        mask, depth, current_box = pipeline.process_frame(frame_resized)
        
        # Visualize
        vis = create_visualization(frame_resized, mask, depth, current_box)
        out.write(vis)
        
        frame_idx += 1
        if frame_idx % 10 == 0:
            sys.stdout.write(f"\r[PROGRESS] Frame {frame_idx}/{MAX_FRAMES}")
            sys.stdout.flush()
            
    cap.release()
    out.release()
    print(f"\n[SUCCESS] Saved to {out_path}")

if __name__ == "__main__":
    main()

