"""
Zero-shot Object Detection using Owl-ViT (Foundation Model).

Allows finding objects by text prompt (e.g., "a dog") to initialize tracking
without manual bounding box selection.
"""

import torch
import numpy as np
from PIL import Image
import gc

class OwlVitDetector:
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        
    def load(self):
        """Load model weights (heavy)."""
        if self.model is not None:
            return

        print(f"[INFO] Loading Owl-ViT Foundation Model on {self.device}...")
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        self.model.to(self.device)
        self.model.eval()
        
    def unload(self):
        """Unload model to free VRAM for SAM/RAFT."""
        if self.model is not None:
            print("[INFO] Unloading Owl-ViT to free VRAM...")
            self.model.to("cpu")
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            gc.collect()
            torch.cuda.empty_cache()
            
    def detect(self, frame: np.ndarray, text_prompt: str, threshold: float = 0.1) -> tuple:
        """
        Detect object matching text prompt.
        Returns: (x, y, w, h) bounding box of best match, or None.
        """
        self.load()
        
        # Convert BGR (OpenCV) to RGB (PIL)
        image = Image.fromarray(frame[..., ::-1])
        texts = [[text_prompt]]
        
        # Preprocess
        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Post-process (get bounding boxes)
        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
        
        # Find best match
        best_score = -1.0
        best_box = None
        
        scores = results["scores"]
        boxes = results["boxes"]
        
        # If multiple detections, pick highest score
        if len(scores) > 0:
            best_idx = torch.argmax(scores).item()
            best_score = scores[best_idx].item()
            box = boxes[best_idx].cpu().numpy() # [xmin, ymin, xmax, ymax]
            
            # Convert to [x, y, w, h] for tracker
            x, y = int(box[0]), int(box[1])
            w, h = int(box[2] - box[0]), int(box[3] - box[1])
            best_box = (x, y, w, h)
            
            print(f"[INFO] Owl-ViT found '{text_prompt}': {best_box} (Confidence: {best_score:.2f})")
        else:
            print(f"[WARN] Owl-ViT could not find '{text_prompt}' (Threshold: {threshold})")
            
        # Clean up immediately
        self.unload()
        
        return best_box

