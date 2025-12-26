"""
Flask REST API for Temporal Flow Pipeline.

Simple production-ready API for object detection, tracking, and 3D reconstruction.
"""

import os
import base64
import uuid
import threading
import json
from pathlib import Path
from datetime import datetime
from io import BytesIO

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np

from app.pipeline import VideoPipeline
from app.processing.point_cloud import depth_to_point_cloud, save_ply
from config import DEVICE

app = Flask(__name__)
CORS(app)

# Global state
pipeline = None
jobs = {}  # In-memory job storage

# Absolute path to outputs directory
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "api"

def init_pipeline():
    """Initialize pipeline on startup."""
    global pipeline
    print(f"[INFO] Initializing pipeline on {DEVICE}...")
    pipeline = VideoPipeline(use_owl_vit=False)
    print("[INFO] Pipeline ready")

def base64_to_image(b64_string: str) -> np.ndarray:
    """Convert base64 string to OpenCV image."""
    img_bytes = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 string."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'device': DEVICE,
        'pipeline_loaded': pipeline is not None
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze single image with OWL-ViT detection + SAM segmentation.
    
    Body: {
        "image": "base64_encoded_image",
        "prompt": "a person's face",
        "threshold": 0.1
    }
    """
    try:
        data = request.json
        img_b64 = data.get('image')
        prompt = data.get('prompt', 'an object')
        threshold = float(data.get('threshold', 0.1))
        
        if not img_b64:
            return jsonify({'error': 'image required'}), 400
        
        # Decode image
        frame = base64_to_image(img_b64)
        
        # Detect with OWL-ViT (load temporarily)
        from app.models.owl_vit import OwlViT
        owl = OwlViT(DEVICE)
        box = owl.get_best_detection(frame, prompt, threshold)
        owl.unload()
        
        if box == (0, 0, 0, 0):
            return jsonify({'error': 'no detection found'}), 404
        
        # Initialize pipeline tracker
        pipeline.init_tracker(frame, box)
        
        # Process frame
        mask, depth, _ = pipeline.process_frame(frame)
        
        # Compute depth stats
        mask_area = mask > 128
        if np.sum(mask_area) > 0:
            depth_stats = {
                'mean': float(np.mean(depth[mask_area])),
                'std': float(np.std(depth[mask_area])),
                'min': float(np.min(depth[mask_area])),
                'max': float(np.max(depth[mask_area]))
            }
        else:
            depth_stats = None
        
        return jsonify({
            'box': box,
            'mask': image_to_base64(mask),
            'depth_stats': depth_stats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_video', methods=['POST'])
def process_video():
    """
    Process video file with full pipeline.
    
    Form data:
        video: file
        prompt: text
        threshold: float (optional)
        max_frames: int (optional)
        output_3d: bool (optional)
    """
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'video file required'}), 400
        
        video_file = request.files['video']
        prompt = request.form.get('prompt', 'an object')
        threshold = float(request.form.get('threshold', 0.1))
        max_frames = int(request.form.get('max_frames', 60))
        output_3d = request.form.get('output_3d', 'false').lower() == 'true'
        
        # Create job
        job_id = str(uuid.uuid4())
        job_dir = OUTPUT_BASE / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded video
        video_path = job_dir / "input.mp4"
        video_file.save(str(video_path))
        
        # Create job entry
        jobs[job_id] = {
            'status': 'processing',
            'created': datetime.now().isoformat(),
            'prompt': prompt,
            'progress': 0
        }
        
        # Start background processing
        thread = threading.Thread(
            target=process_video_task,
            args=(job_id, str(video_path), prompt, threshold, max_frames, output_3d)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'processing'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_video_task(job_id, video_path, prompt, threshold, max_frames, output_3d):
    """Background task for video processing."""
    try:
        job_dir = OUTPUT_BASE / job_id
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Read first frame
        ret, first_frame = cap.read()
        if not ret:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = 'cannot read video'
            return
        
        # Resize if needed
        if width > 640:
            scale = 640 / width
            first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
            height, width = first_frame.shape[:2]
        
        # Auto-detect
        from app.models.owl_vit import OwlViT
        owl = OwlViT(DEVICE)
        box = owl.get_best_detection(first_frame, prompt, threshold)
        owl.unload()
        
        if box == (0, 0, 0, 0):
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = 'no detection found'
            cap.release()
            return
        
        # Initialize tracker
        pipeline.init_tracker(first_frame, box)
        
        # Setup output video
        output_path = job_dir / "result.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width * 3, height))
        
        # Reset video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Process frames
        frame_num = 0
        last_depth = None
        last_mask = None
        
        while frame_num < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame.shape[1] > 640:
                scale = 640 / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            
            # Process
            mask, depth, current_box = pipeline.process_frame(frame)
            
            # Store for 3D output
            if frame_num == 0:
                last_depth = depth
                last_mask = mask
            
            # Create visualization
            viz = create_viz(frame, mask, depth, current_box, frame_num)
            out.write(viz)
            
            # Update progress
            jobs[job_id]['progress'] = int((frame_num / max_frames) * 100)
            frame_num += 1
        
        cap.release()
        out.release()
        
        # Generate 3D if requested
        ply_path = None
        if output_3d and last_depth is not None:
            points, colors = depth_to_point_cloud(first_frame, last_depth, last_mask)
            ply_path = job_dir / "object.ply"
            save_ply(points, colors, str(ply_path))
        
        # Save results
        results = {
            'frames_processed': frame_num,
            'initial_box': box,
            'video_url': f'/outputs/{job_id}/result.mp4',
            'ply_url': f'/outputs/{job_id}/object.ply' if ply_path else None
        }
        
        with open(job_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['results'] = results
        
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)

def create_viz(frame, mask, depth, box, frame_num):
    """Create horizontal visualization."""
    h, w = frame.shape[:2]
    
    # Original + box
    frame_viz = frame.copy()
    x, y, w_box, h_box = box
    cv2.rectangle(frame_viz, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
    cv2.putText(frame_viz, f"Frame {frame_num}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Mask
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    mask_overlay = cv2.addWeighted(frame, 0.6, mask_colored, 0.4, 0)
    
    # Depth
    depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
    
    return np.hstack([frame_viz, mask_overlay, depth_colored])

@app.route('/job/<job_id>', methods=['GET'])
def get_job(job_id):
    """Get job status and results."""
    if job_id not in jobs:
        return jsonify({'error': 'job not found'}), 404
    return jsonify(jobs[job_id])

@app.route('/outputs/<job_id>/<filename>', methods=['GET'])
def get_output(job_id, filename):
    """Download output file."""
    file_path = OUTPUT_BASE / job_id / filename
    if not file_path.exists():
        return jsonify({'error': 'file not found'}), 404
    return send_file(str(file_path), as_attachment=True)

if __name__ == '__main__':
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    init_pipeline()
    app.run(host='0.0.0.0', port=5000, debug=False)

