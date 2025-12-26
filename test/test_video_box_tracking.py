"""
Video Segmentation Test with Temporal Consistency.

This script demonstrates the integration of:
1. MobileSAM (Foundation Model) for segmentation
2. CSRT Tracker (OpenCV) for object tracking
3. RAFT/Farneback Optical Flow for temporal mask refinement

Outputs:
- Comparative videos (raw vs. refined)
- Quantitative stability metrics
- Visual samples
"""

import cv2
import numpy as np
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.sam import MobileSAM
from app.processing.flow import TemporalRefiner
from config import WEIGHTS_DIR, SAM_CHECKPOINT, DEVICE

# --- Configuration ---
VIDEO_PATH = "samples/Test vid-2 .mp4"
OUTPUT_DIR = "outputs/video_test"
MAX_FRAMES = 300  # Process up to N frames
ALPHA = 0.15      # Temporal blending factor (0.15 = 15% history)

def calculate_stability(masks):
    """Calculate mean absolute difference between consecutive frames."""
    if len(masks) < 2:
        return 0.0, 0.0
    
    diffs = []
    for i in range(1, len(masks)):
        # Normalize to 0-1 for stability calculation
        prev = masks[i-1].astype(float) / 255.0
        curr = masks[i].astype(float) / 255.0
        diff = np.abs(curr - prev).mean()
        diffs.append(diff)
    
    return np.mean(diffs), np.std(diffs)

def save_metrics(path, metrics_data):
    """Save quantitative results to JSON."""
    with open(path, 'w') as f:
        json.dump(metrics_data, f, indent=4)

def main():
    print(f"[INFO] Starting Video Segmentation Test")
    print(f"[INFO] Video: {VIDEO_PATH}")
    print(f"[INFO] Device: {DEVICE}")
    print(f"-" * 50)

    # 1. Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {VIDEO_PATH}")
        sys.exit(1)
        
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 2. Initialization
    # Resize logic for display/processing if video is too large
    process_width = 640
    scale = process_width / width if width > process_width else 1.0
    process_height = int(height * scale)
    
    print(f"[INFO] Resolution: {width}x{height} -> {process_width}x{process_height}")
    print(f"[INFO] Processing max {MAX_FRAMES} frames")

    # 3. User Interaction (Object Selection)
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read first frame")
        sys.exit(1)
        
    first_frame_resized = cv2.resize(first_frame, (process_width, process_height))
    
    print("\n[INSTRUCTION] Select object to track:")
    print("  1. Draw a box around the object")
    print("  2. Press SPACE or ENTER to confirm")
    
    roi_bbox = cv2.selectROI("Select Object", first_frame_resized, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    
    if roi_bbox[2] == 0 or roi_bbox[3] == 0:
        print("[ERROR] No selection made. Exiting.")
        sys.exit(1)
        
    print(f"[INFO] Initial Bounding Box: {roi_bbox}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 4. Model Loading
    print(f"\n[INFO] Loading MobileSAM...")
    sam = MobileSAM(os.path.join(WEIGHTS_DIR, SAM_CHECKPOINT), DEVICE)
    
    print(f"[INFO] Initializing Trackers...")
    try:
        tracker = cv2.legacy.TrackerCSRT_create()
    except AttributeError:
        try:
            tracker = cv2.TrackerCSRT_create()
        except AttributeError:
            print("[ERROR] CSRT Tracker not found. Install opencv-contrib-python.")
            sys.exit(1)
            
    refiner = TemporalRefiner(alpha=ALPHA, track_point=False)

    # 5. Processing Loop
    print(f"\n[INFO] Processing video...")
    
    frames_processed = []
    masks_raw = []
    masks_refined = []
    
    tracker_initialized = False
    frame_idx = 0
    
    while frame_idx < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_resized = cv2.resize(frame, (process_width, process_height))
        frames_processed.append(frame_resized)
        
        # Track object
        if not tracker_initialized:
            tracker.init(frame_resized, roi_bbox)
            tracker_initialized = True
            curr_box = roi_bbox
        else:
            success, box = tracker.update(frame_resized)
            curr_box = tuple(map(int, box)) if success else curr_box
            
        # A. Raw Segmentation (Independent)
        # Using box + center point logic we added to sam.py
        mask_raw = sam.segment_box(frame_resized, roi_bbox) 
        masks_raw.append(mask_raw)
        
        # B. Refined Segmentation (Tracked + Temporal)
        mask_sam = sam.segment_box(frame_resized, curr_box)
        mask_refined, _ = refiner.refine(frame_resized, mask_sam)
        masks_refined.append(mask_refined)
        
        frame_idx += 1
        if frame_idx % 20 == 0:
            sys.stdout.write(f"\r[PROGRESS] Frame {frame_idx}/{MAX_FRAMES}")
            sys.stdout.flush()
            
    print(f"\n[SUCCESS] Processing complete.")
    cap.release()

    # 6. Metrics & Visualization
    print(f"\n[INFO] calculating stability metrics...")
    
    stab_raw, std_raw = calculate_stability(masks_raw)
    stab_refined, std_refined = calculate_stability(masks_refined)
    improvement = ((stab_raw - stab_refined) / stab_raw * 100) if stab_raw > 0 else 0
    
    results = {
        "video_path": VIDEO_PATH,
        "frames": frame_idx,
        "metrics": {
            "raw_stability_error": round(stab_raw, 4),
            "refined_stability_error": round(stab_refined, 4),
            "improvement_percentage": round(improvement, 2)
        },
        "parameters": {
            "alpha": ALPHA,
            "max_frames": MAX_FRAMES
        },
        "timestamp": datetime.now().isoformat()
    }
    
    save_metrics(os.path.join(OUTPUT_DIR, "results.json"), results)
    
    print(f"-" * 30)
    print(f"RESULTS Summary")
    print(f"Raw Stability Error:     {stab_raw:.4f}")
    print(f"Refined Stability Error: {stab_refined:.4f}")
    print(f"Improvement:             {improvement:.1f}%")
    print(f"-" * 30)

    # 7. Video Generation
    print(f"[INFO] Generating output videos...")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(OUTPUT_DIR, "comparison_demo.mp4")
    out = cv2.VideoWriter(out_path, fourcc, fps, (process_width * 2, process_height))
    
    for i, frame in enumerate(frames_processed):
        # Create visualizations
        vis_raw = frame.copy()
        vis_refined = frame.copy()
        
        # Apply green masks
        mask_r = masks_raw[i] > 0
        mask_f = masks_refined[i] > 0
        
        vis_raw[mask_r] = vis_raw[mask_r] * 0.5 + np.array([0, 0, 255]) * 0.5  # Red for Raw
        vis_refined[mask_f] = vis_refined[mask_f] * 0.5 + np.array([0, 255, 0]) * 0.5  # Green for Refined
        
        # Labels
        cv2.putText(vis_raw, "Baseline (Raw SAM)", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis_refined, f"Ours (Temporal Refined)", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        combined = np.hstack([vis_raw, vis_refined])
        out.write(combined)
        
    out.release()
    print(f"[SUCCESS] Demo saved to {out_path}")

if __name__ == "__main__":
    main()
