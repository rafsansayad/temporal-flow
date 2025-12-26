"""
Professional Test: Full Pipeline with OWL-ViT Auto-Detection.

Tests the complete video processing pipeline:
1. OWL-ViT zero-shot object detection
2. CSRT object tracking
3. MobileSAM segmentation
4. Temporal refinement (optical flow)
5. MiDAS depth estimation
"""

import cv2
import sys
import os
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline import VideoPipeline
from config import DEVICE

# Configuration
VIDEO_PATH = "samples/Test 480p 30Fps.mp4"
OUTPUT_DIR = "outputs/pipeline_owl_test/dog"
TEXT_QUERY = "a dog"
DETECTION_THRESHOLD = 0.1
MAX_FRAMES = 350 # Process first 60 frames

def create_visualization(frame: np.ndarray, mask: np.ndarray, depth: np.ndarray, 
                         box: tuple, frame_num: int) -> np.ndarray:
    """Create horizontal side-by-side visualization."""
    h, w = frame.shape[:2]
    
    # Original with bounding box
    frame_viz = frame.copy()
    x, y, w_box, h_box = box
    cv2.rectangle(frame_viz, (x, y), (x + w_box, y + h_box), (0, 255, 0), 3)
    cv2.putText(frame_viz, f"Frame {frame_num}", (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    cv2.putText(frame_viz, "ORIGINAL + BOX", (10, h - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Mask overlay
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    mask_overlay = cv2.addWeighted(frame, 0.6, mask_colored, 0.4, 0)
    cv2.putText(mask_overlay, "MASK", (10, h - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Depth map
    depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
    cv2.putText(depth_colored, "DEPTH", (10, h - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Combine horizontally (side-by-side)
    combined = np.hstack([frame_viz, mask_overlay, depth_colored])
    
    return combined

def main():
    print("=" * 60)
    print("PIPELINE TEST: OWL-ViT Auto-Detection + Full Processing")
    print("=" * 60)
    
    # Setup output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load video
    print(f"\n[1/5] Loading video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
        return 1
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Processing: {min(MAX_FRAMES, total_frames)} frames")
    
    # Read first frame
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read first frame")
        return 1
    
    # Resize if needed
    if width > 640:
        scale = 640 / width
        first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
        height, width = first_frame.shape[:2]
    
    # Initialize pipeline with OWL-ViT
    print(f"\n[2/5] Initializing pipeline on {DEVICE}")
    pipeline = VideoPipeline(use_owl_vit=True)
    
    # Auto-detect object
    print(f"\n[3/5] Auto-detecting: '{TEXT_QUERY}'")
    success = pipeline.auto_detect_and_init( first_frame, 
                                             TEXT_QUERY, 
                                             DETECTION_THRESHOLD, 
                                             redetect_interval=15
                                             )
    
    if not success:
        print("[ERROR] Detection failed. Exiting.")
        return 1
    
    # Setup video writer (3 panels side-by-side)
    print(f"\n[4/5] Setting up video writer")
    output_path = os.path.join(OUTPUT_DIR, "result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 3, height))
    
    # Process video
    print(f"\n[5/5] Processing frames...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
    
    results = {
        "video": VIDEO_PATH,
        "query": TEXT_QUERY,
        "device": DEVICE,
        "frames_processed": 0,
        "initial_detection": pipeline.current_box,
        "frames": []
    }
    
    frame_num = 0
    while frame_num < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize
        if frame.shape[1] > 640:
            scale = 640 / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        # Process frame
        mask, depth, box = pipeline.process_frame(frame)
        
        # Create visualization
        viz = create_visualization(frame, mask, depth, box, frame_num)
        out.write(viz)
        
        # Store results
        results["frames"].append({
            "frame": frame_num,
            "box": box,
            "mask_coverage": float(np.sum(mask > 128) / (mask.shape[0] * mask.shape[1]))
        })
        
        # Progress
        if frame_num % 10 == 0:
            print(f"  Frame {frame_num}/{MAX_FRAMES} | Box: {box}")
        
        frame_num += 1
    
    results["frames_processed"] = frame_num
    
    # Cleanup
    cap.release()
    out.release()
    
    # Save results JSON
    json_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print(f"Frames Processed: {results['frames_processed']}")
    print(f"Initial Detection: {results['initial_detection']}")
    print(f"Output Video: {output_path}")
    print(f"Results JSON: {json_path}")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

