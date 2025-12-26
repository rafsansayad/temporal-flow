[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

# Temporal Flow: Zero-Shot 3D Video Object Tracking & Reconstruction

**Temporal Flow** is a modular computer vision pipeline designed to perform zero-shot object tracking, segmentation, and 3D reconstruction from monocular video sources.

The system integrates multiple foundation models—**OWL-ViT** for open-vocabulary detection, **MobileSAM** for segmentation, and **MiDAS** for monocular depth into a unified processing stream. It employs optical flow (Farneback/RAFT) for temporal consistency, ensuring stable mask propagation across frames.

The project is fully containerized with **Docker** and exposes a REST API for scalable deployment, demonstrating an end-to-end MLOps workflow.

---



<br>
<div align="center">
  <video src= https://github.com/user-attachments/assets/30a76ce6-562e-4386-a232-ea24928dae5a width="100%" controls autoplay muted loop></video>
</div>
<p align="center"><em>Figure 1: Zero-shot tracking of a dog (OWL-ViT + CSRT) with segmentation and depth estimation.</em></p>
<br>


## 🔬 System Architecture

The pipeline operates in five distinct stages to transform raw video input into 3D spatial data:

### 1. Initialization (Zero-Shot Detection)
The system requires no pre-training on specific object classes. Using **OWL-ViT (Vision Transformer)**, it interprets natural language prompts (e.g., *"a person's face"*) to localize the target object in the initial frame. This provides a high-confidence bounding box to initialize the tracker.

### 2. Temporal Tracking (CSRT)
To maintain computational efficiency, frame-to-frame localization is handled by a **CSRT (Channel and Spatial Reliability Tracker)**. This tracker provides robust short-term position estimation, reducing the need to run heavy detection models on every frame.

### 3. Segmentation (MobileSAM)
The tracker's output serves as a geometric prompt for **MobileSAM (Segment Anything Model)**. This generates a pixel-perfect binary mask of the object. MobileSAM was chosen over the standard SAM to optimize for inference latency while maintaining adequate boundary precision.

### 4. Temporal Refinement (Optical Flow)
Raw segmentation masks often exhibit jitter between frames. We apply **Dense Optical Flow** to warp the previous frame's mask forward. This temporal prior is blended with the current prediction to enforce consistency and smooth out segmentation noise.

### 5. 3D Reconstruction (RGB-D Projection)
**MiDAS** estimates a relative depth map for each frame. By combining the segmentation mask with the depth map, we isolate the object's geometry. Using intrinsic camera approximation, pixels are back-projected into 3D space, generating a colored point cloud (`.ply`) suitable for visualization in tools like MeshLab or Blender.

---

## 🛠️ Technical Stack

- **Language:** Python 3.10
- **Frameworks:** PyTorch, OpenCV, NumPy
- **Models:** 
  - `OWL-ViT` (Open-Vocabulary Detection)
  - `MobileSAM` (Efficient Segmentation)
  - `MiDAS` (Monocular Depth Estimation)
- **Deployment:** Flask (REST API), Docker, NVIDIA CUDA
- **Tools:** Git, Postman

---

## 🚀 Deployment & Usage

### 1. Docker Deployment (Recommended)
The system is packaged as a GPU-accelerated Docker container for consistent deployment across environments.

```bash
# Build the image
docker-compose up

# Check status
docker ps
```

### 2. REST API Interface
The service exposes endpoints for video processing and analysis.

**Health Check:**
```bash
curl http://localhost:5000/health
```

**Process Video Job:**
```bash
curl -X POST http://localhost:5000/process_video \
  -F "video=@sample.mp4" \
  -F "prompt=a person" \
  -F "output_3d=true"
```
*Returns a `job_id` for asynchronous status tracking.*

### 3. Local Development
For research and modification, the pipeline can be run locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the test suite
python test/test_pipeline_owl.py
```

---

## 📊 Results & Performance

- **Low VRAM Usage:** Operates efficiently on just **1.4 GB VRAM**, enabling deployment on edge devices and consumer GPUs (e.g., RTX 3050, laptop GPUs).
- **Hybrid Tracking Architecture:** Intelligently switches between heavy foundation models (OWL-ViT) and lightweight correlation trackers (CSRT) to balance accuracy and speed.
- **Latency:** Optimized for near-real-time processing (approx. 15-20 FPS on RTX 3060).
- **Output:**
  - **Visualization:** Side-by-side video (Original | Mask | Depth)
  - **3D Model:** `.ply` point cloud file
  - **Metadata:** JSON logs of tracking confidence and depth statistics

---

## 📂 Project Structure

```
temporal-flow/
├── app/
│   ├── models/        # Model wrappers (OWL-ViT, SAM, Depth)
│   ├── processing/    # Optical flow & point cloud logic
│   ├── server.py      # Flask REST API implementation
│   └── pipeline.py    # Core orchestration logic
├── docker-compose.yml # Container orchestration
├── Dockerfile         # Build definition
├── requirements.txt   # Dependencies
└── test/              # Unit and integration tests
```

---

*Developed as a prototype for advanced computer vision research applications.*

