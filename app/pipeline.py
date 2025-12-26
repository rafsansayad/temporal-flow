"""
Core Video Processing Pipeline.

Orchestrates:
1. Object Tracking (CSRT)
2. Segmentation (MobileSAM)
3. Temporal Refinement (RAFT/Farneback)
4. Depth Estimation (MiDAS)
"""

import cv2
import numpy as np
import os
import torch
from typing import Tuple, Optional, Generator

from app.models.sam import MobileSAM
from app.models.depth import MiDAS
from app.models.owl_vit import OwlViT
from app.processing.flow import TemporalRefiner
from config import WEIGHTS_DIR, SAM_CHECKPOINT, MIDAS_MODEL, DEVICE

class VideoPipeline:
    def __init__(self, use_owl_vit: bool = False):
        """Initialize the pipeline and load models."""
        print("[INFO] Initializing VideoPipeline...")
        
        # Load Models
        self.sam = MobileSAM(os.path.join(WEIGHTS_DIR, SAM_CHECKPOINT), DEVICE)
        self.midas = MiDAS(MIDAS_MODEL, DEVICE)
        self.owl = OwlViT(DEVICE) if use_owl_vit else None
        
        # Tools
        self.refiner = TemporalRefiner(alpha=0.15, track_point=False)
        self.tracker = None
        
        # State
        self.tracker_initialized = False
        self.current_box = None
        
        # Re-detection settings
        self.redetect_interval = 0  # 0 to disable
        self.redetect_query = ""
        self.redetect_threshold = 0.1
        self.frame_count = 0
        
    def init_tracker(self, frame: np.ndarray, box: Tuple[int, int, int, int]):
        """Initialize the object tracker with a bounding box."""
        try:
            self.tracker = cv2.legacy.TrackerCSRT_create()
        except AttributeError:
            self.tracker = cv2.TrackerCSRT_create()
            
        self.tracker.init(frame, box)
        self.current_box = box
        self.tracker_initialized = True
        self.refiner.reset()
        print(f"[INFO] Tracker initialized at {box}")

    def auto_detect_and_init(self, frame: np.ndarray, text_query: str, threshold: float = 0.1, 
                            keep_owl_loaded: bool = False, redetect_interval: int = 0) -> bool:
        """
        Auto-detect object using OWL-ViT and initialize tracker.
        
        Args:
            frame: First frame
            text_query: What to detect (e.g., "a person's face")
            threshold: Detection confidence threshold
            keep_owl_loaded: If True, OWL-ViT stays in memory for re-detection
            redetect_interval: Run re-detection every N frames (0 to disable)
            
        Returns:
            True if detection successful, False otherwise
        """
        if self.owl is None:
            print("[ERROR] OWL-ViT not initialized. Set use_owl_vit=True")
            return False
        
        # Save settings for re-detection
        self.redetect_query = text_query
        self.redetect_threshold = threshold
        self.redetect_interval = redetect_interval
        
        box = self.owl.get_best_detection(frame, text_query, threshold)
        
        if not keep_owl_loaded and redetect_interval == 0:
            # Unload OWL-ViT immediately to free VRAM
            print("[INFO] Unloading OWL-ViT to free VRAM...")
            self.owl.unload()
            self.owl = None
        else:
            print("[INFO] Keeping OWL-ViT loaded for re-detection.")
        
        if box == (0, 0, 0, 0):
            print(f"[WARNING] No detection found for '{text_query}'")
            return False
        
        print(f"[INFO] Auto-detected: {box}")
        self.init_tracker(frame, box)
        return True

    def filter_by_depth(self, mask: np.ndarray, depth_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Experimental: Remove mask pixels that are significantly farther/closer than the object.
        
        Args:
            mask: Binary segmentation mask
            depth_map: Normalized depth map (0-255)
            threshold: Std dev tolerance multiplier
        """
        if np.sum(mask) == 0:
            return mask

        # Get depth of the object (where mask is True)
        object_depths = depth_map[mask > 128]
        
        if len(object_depths) == 0:
            return mask

        mean_depth = np.mean(object_depths)
        std_depth = np.std(object_depths)
        
        # Define range (e.g., keep pixels within 1.5 standard deviations)
        # This removes outliers (accidental background spills)
        lower_bound = mean_depth - (threshold * std_depth)
        
        # Create depth mask (keep pixels that are "close enough")
        # Note: In MiDAS, higher value = closer. So we want depth > lower_bound.
        depth_mask = depth_map > lower_bound
        
        return cv2.bitwise_and(mask, mask, mask=depth_mask.astype(np.uint8))

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a single frame.
        
        Returns:
            mask: Refined binary mask
            depth: Depth map
            box: Current bounding box
        """
        if not self.tracker_initialized:
            return np.zeros(frame.shape[:2], dtype=np.uint8), \
                   np.zeros(frame.shape[:2], dtype=np.uint8), \
                   (0,0,0,0)

        self.frame_count += 1

        # 0. Periodic Re-Detection (Anti-Drift)
        if (self.redetect_interval > 0 and 
            self.frame_count % self.redetect_interval == 0 and 
            self.owl is not None):
            
            # Try to find object again
            new_box = self.owl.get_best_detection(frame, self.redetect_query, self.redetect_threshold)
            
            # Only update if we found something with high confidence
            # (get_best_detection already filters by threshold)
            if new_box != (0, 0, 0, 0):
                print(f"[INFO] Re-detected object at frame {self.frame_count}: {new_box}")
                self.init_tracker(frame, new_box)
                # Note: init_tracker resets tracker, so next step will work from new box

        # 1. Update Tracker
        success, box = self.tracker.update(frame)
        if success:
            self.current_box = tuple(map(int, box))
        
        # 2. SAM Segmentation (Box + Center Point)
        raw_mask = self.sam.segment_box(frame, self.current_box)
        
        # 3. Temporal Refinement (Optical Flow)
        refined_mask, _ = self.refiner.refine(frame, raw_mask)
        
        # 4. Depth Estimation
        depth_map = self.midas.estimate(frame)
        
        # 5. Depth Filtering (Optional Cleanup)
        # We filter the refined mask using depth to remove background spill
        final_mask = self.filter_by_depth(refined_mask, depth_map, threshold=1.5)
        
        return final_mask, depth_map, self.current_box

    def process_video_generator(self, video_path: str) -> Generator:
        """
        Generator that yields processed frames (for streaming/saving).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize for speed (optional)
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
                
            yield frame, self.process_frame(frame)
            
        cap.release()

