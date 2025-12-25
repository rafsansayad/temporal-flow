import torch
import numpy as np
import cv2
import torch.hub


class MiDAS:
    """
    MiDAS depth estimation wrapper.
    """
    
    def __init__(self, model_type: str = "MiDaS_small", device: str = "cuda"):
        """
        Initialize MiDAS depth estimator.
        
        Args:
            model_type: model variant ('MiDaS_small' recommended)
            device: 'cuda' or 'cpu' 
        """
        self.model_type = model_type
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.transform = None
        
        if self.device == "cpu" and device == "cuda":
            print("Warning: CUDA not available, using CPU")
    
    def _ensure_loaded(self):
        """Lazy load model on first use."""
        if self.model is not None:
            return
        
        # Bypass GitHub API rate limit
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        
        # Load model from torch hub (caches locally)
        self.model = torch.hub.load(
            "intel-isl/MiDaS", 
            self.model_type,
            trust_repo=True,
            skip_validation=True
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Load preprocessing transforms
        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", 
            "transforms",
            trust_repo=True,
            skip_validation=True
        )
        self.transform = midas_transforms.small_transform
    
    @torch.no_grad()
    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from input frame.
        
        Args:
            frame: Input image (H, W, 3) in BGR format
            
        Returns:
            Depth map (H, W) normalized to 0-255 (uint8)
            Brighter = closer, darker = farther
            
        Raises:
            ValueError: If frame is invalid
        """
        self._ensure_loaded()
        
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame: empty or None")
        
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected (H,W,3) frame, got {frame.shape}")
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Transform and predict
        input_batch = self.transform(rgb).to(self.device)
        prediction = self.model(input_batch)
        
        # Resize to original resolution
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()
        
        depth = prediction.cpu().numpy()
        
        # Normalize to 0-255
        depth_min, depth_max = depth.min(), depth.max()
        depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        
        return (depth_normalized * 255).astype(np.uint8)

