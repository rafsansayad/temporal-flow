import torch
import numpy as np
import cv2
import os


class MobileSAM:
    """
    MobileSAM wrapper for efficient video segmentation.
    
    Uses MobileSAM (10MB) instead of full SAM for faster inference
    on resource-constrained devices.
    """
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Initialize MobileSAM model.
        
        Args:
            checkpoint_path: Path to mobile_sam.pt weights
            device: 'cuda' or 'cpu' (auto-fallback if CUDA unavailable)
        """
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.predictor = None
        
        if self.device == "cpu" and device == "cuda":
            print("Warning: CUDA not available, using CPU")
    
    def _ensure_loaded(self):
        """Lazy load model on first use."""
        if self.predictor is not None:
            return
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        from mobile_sam import sam_model_registry, SamPredictor
        
        model = sam_model_registry["vit_t"](checkpoint=self.checkpoint_path)
        model.to(self.device)
        model.eval()
        self.predictor = SamPredictor(model)
    
    def segment(self, frame: np.ndarray, point: tuple = None) -> np.ndarray:
        """
        Segment object in frame using point prompt.
        
        Args:
            frame: Input image (H, W, 3) in BGR format
            point: (x, y) point prompt. If None, uses image center
            
        Returns:
            Binary mask (H, W) with values 0 or 255
            
        Raises:
            ValueError: If frame is invalid
        """
        self._ensure_loaded()
        
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame: empty or None")
        
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected (H,W,3) frame, got {frame.shape}")
        
        h, w = frame.shape[:2]
        
        # Default to center point
        if point is None:
            point = (w // 2, h // 2)
        
        # Validate point
        if not (0 <= point[0] < w and 0 <= point[1] < h):
            raise ValueError(f"Point {point} outside frame bounds ({w}x{h})")
        
        self.predictor.set_image(frame)
        
        masks, _, _ = self.predictor.predict(
            point_coords=np.array([[point[0], point[1]]]),
            point_labels=np.array([1]),
            multimask_output=False
        )
        
        return masks[0].astype(np.uint8) * 255

