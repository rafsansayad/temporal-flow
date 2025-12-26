import cv2
import numpy as np
import torch

# Optical flow backend selection
_flow_backend = None
_raft_model = None

def _init_flow_backend():
    """Initialize optical flow backend (RAFT or Farneback)."""
    global _flow_backend, _raft_model
    
    if _flow_backend is not None:
        return
    
    if not torch.cuda.is_available():
        _flow_backend = "farneback"
        return
    
    try:
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        _raft_model = raft_small(weights=Raft_Small_Weights.DEFAULT)
        _raft_model.to("cuda").eval()
        _flow_backend = "raft"
    except Exception:
        _flow_backend = "farneback"


def _compute_flow_raft(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """RAFT-based optical flow (GPU)."""
    # Convert grayscale to RGB (RAFT expects 3 channels)
    if len(prev_gray.shape) == 2:
        prev_gray = cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2RGB)
        curr_gray = cv2.cvtColor(curr_gray, cv2.COLOR_GRAY2RGB)
    
    orig_h, orig_w = prev_gray.shape[:2]
    
    # RAFT requires dimensions divisible by 8
    pad_h = (8 - orig_h % 8) % 8
    pad_w = (8 - orig_w % 8) % 8
    
    if pad_h > 0 or pad_w > 0:
        prev_gray = cv2.copyMakeBorder(prev_gray, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        curr_gray = cv2.copyMakeBorder(curr_gray, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    
    # Prepare tensors
    img1 = torch.from_numpy(prev_gray).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img2 = torch.from_numpy(curr_gray).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    
    with torch.no_grad():
        flow_predictions = _raft_model(img1.cuda(), img2.cuda())
        flow = flow_predictions[-1][0].permute(1, 2, 0).cpu().numpy()
    
    # Crop back to original size
    if pad_h > 0 or pad_w > 0:
        flow = flow[:orig_h, :orig_w]
    
    return flow


def _compute_flow_farneback(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """Farneback optical flow (CPU)."""
    return cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )


def compute_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """
    Compute dense optical flow between frames.
    
    Args:
        prev_gray: Previous frame (H, W) grayscale
        curr_gray: Current frame (H, W) grayscale
        
    Returns:
        Flow field (H, W, 2) with dx, dy per pixel
    """
    _init_flow_backend()
    
    if _flow_backend == "raft":
        return _compute_flow_raft(prev_gray, curr_gray)
    else:
        return _compute_flow_farneback(prev_gray, curr_gray)


def warp_mask(mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Warp mask using optical flow field.
    
    Args:
        mask: Binary mask (H, W) to warp
        flow: Flow field (H, W, 2) from compute_flow
        
    Returns:
        Warped mask (H, W) aligned with current frame
    """
    h, w = flow.shape[:2]
    
    # Create coordinate grid
    flow_map = np.zeros((h, w, 2), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            flow_map[y, x, 0] = x + flow[y, x, 0]
            flow_map[y, x, 1] = y + flow[y, x, 1]
    
    # Warp mask using remap
    warped = cv2.remap(
        mask,
        flow_map[:, :, 0],
        flow_map[:, :, 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return warped


def temporal_blend(prev_mask: np.ndarray, curr_mask: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Blend previous and current masks for temporal smoothing.
    
    Args:
        prev_mask: Warped previous mask (H, W)
        curr_mask: Current frame mask (H, W)
        alpha: Blending weight (0-1). Higher = more history
        
    Returns:
        Blended mask (H, W)
    """
    blended = (alpha * prev_mask.astype(np.float32) + 
               (1 - alpha) * curr_mask.astype(np.float32))
    return blended.astype(np.uint8)


def visualize_flow(flow: np.ndarray, scale: float = 3.0) -> np.ndarray:
    """
    Visualize optical flow as colored image.
    
    Args:
        flow: Flow field (H, W, 2)
        scale: Scaling factor for visualization
        
    Returns:
        RGB image (H, W, 3) with flow visualization
    """
    h, w = flow.shape[:2]
    
    # Convert flow to polar coordinates
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Create HSV image
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue = direction
    hsv[..., 1] = 255  # Full saturation
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value = magnitude
    
    # Convert to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb


class TemporalRefiner:
    """
    Stateful temporal mask refiner using optical flow.
    
    Tracks previous frame, mask, and point to apply temporal consistency
    across video sequences.
    """
    
    def __init__(self, alpha: float = 0.3, track_point: bool = True):
        """
        Initialize temporal refiner.
        
        Args:
            alpha: Temporal blending weight (0-1). Higher = smoother but more lag
            track_point: If True, propagates point using optical flow
        """
        self.alpha = alpha
        self.track_point = track_point
        self.prev_frame = None
        self.prev_mask = None
        self.prev_point = None
    
    def reset(self):
        """Reset temporal state (call at video start or scene change)."""
        self.prev_frame = None
        self.prev_mask = None
        self.prev_point = None
    
    def refine(self, frame: np.ndarray, mask: np.ndarray, point: tuple = None) -> tuple:
        """
        Apply temporal refinement to mask and optionally track point.
        
        Args:
            frame: Current frame (H, W, 3) BGR
            mask: Current mask (H, W) from segmentation model
            point: Current tracking point (x, y). Required for first frame.
            
        Returns:
            Tuple of (refined_mask, updated_point)
        """
        # First frame: no refinement possible
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.prev_mask = mask
            self.prev_point = point
            return mask, point
        
        # Convert current frame to grayscale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute optical flow
        flow = compute_flow(self.prev_frame, curr_gray)
        
        # Update point using flow
        updated_point = point
        if self.track_point and self.prev_point is not None:
            px, py = self.prev_point
            h, w = flow.shape[:2]
            
            # Ensure point is within bounds
            if 0 <= py < h and 0 <= px < w:
                dx, dy = flow[py, px]
                updated_point = (int(px + dx), int(py + dy))
                
                # Clamp to frame boundaries
                updated_point = (
                    max(0, min(w - 1, updated_point[0])),
                    max(0, min(h - 1, updated_point[1]))
                )
        
        # Warp previous mask
        warped_mask = warp_mask(self.prev_mask, flow)
        
        # Blend with current mask
        refined_mask = temporal_blend(warped_mask, mask, self.alpha)
        
        # Update state
        self.prev_frame = curr_gray
        self.prev_mask = refined_mask
        self.prev_point = updated_point
        
        return refined_mask, updated_point

