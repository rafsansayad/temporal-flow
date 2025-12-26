"""
Owl-ViT: Open-Vocabulary Object Detection.

Zero-shot detection using text prompts.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Tuple


class OwlViT:
    """
    Owl-ViT wrapper for text-to-box detection.
    
    Enables zero-shot object detection with natural language queries.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize Owl-ViT model.
        
        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        
        if self.device == "cpu" and device == "cuda":
            print("[WARNING] CUDA not available, using CPU for Owl-ViT")
    
    def _ensure_loaded(self):
        """Lazy load model on first use from local weights."""
        if self.model is not None:
            return
        
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        
        local_path = "weights/owl"
        print(f"[INFO] Loading Owl-ViT from {local_path}")
        
        self.processor = OwlViTProcessor.from_pretrained(local_path)
        self.model = OwlViTForObjectDetection.from_pretrained(local_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[INFO] Owl-ViT loaded on {self.device}")
    
    def detect(self, frame: np.ndarray, text_queries: List[str], threshold: float = 0.1) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect objects using text queries.
        
        Args:
            frame: Input image (H, W, 3) BGR format
            text_queries: List of text descriptions (e.g., ["a dog", "a person"])
            threshold: Confidence threshold (0-1)
            
        Returns:
            List of detections: [(x, y, w, h, score), ...]
        """
        self._ensure_loaded()
        
        # Convert BGR to RGB
        image_rgb = Image.fromarray(frame[:, :, ::-1])
        
        # Prepare inputs
        inputs = self.processor(text=text_queries, images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([image_rgb.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )[0]
        
        # Convert to (x, y, w, h, score) format
        detections = []
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)
            detections.append((x, y, w, h, float(score)))
        
        return detections
    
    def get_best_detection(self, frame: np.ndarray, text_query: str, threshold: float = 0.1) -> Tuple[int, int, int, int]:
        """
        Get the highest-confidence bounding box for a single query.
        
        Args:
            frame: Input image
            text_query: Text description (e.g., "a dog")
            threshold: Confidence threshold
            
        Returns:
            (x, y, w, h) of best detection, or (0, 0, 0, 0) if none found
        """
        detections = self.detect(frame, [text_query], threshold)
        
        if len(detections) == 0:
            return (0, 0, 0, 0)
        
        # Return highest confidence detection
        best = max(detections, key=lambda d: d[4])
        return best[:4]
    
    def unload(self):
        """Free GPU memory by moving model to CPU and clearing cache."""
        if self.model is not None:
            self.model.to("cpu")
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

