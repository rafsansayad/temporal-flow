"""
3D Point Cloud Generation and Visualization.

Converts RGB-D data (Image + Depth) into 3D point clouds.
Supports exporting to .ply and generating rotating GIFs.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import io

def depth_to_point_cloud(
    rgb: np.ndarray, 
    depth: np.ndarray, 
    mask: np.ndarray = None, 
    fov: float = 60.0,
    subsample: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert RGB image and Depth map to 3D point cloud.
    
    Args:
        rgb: RGB image (H, W, 3)
        depth: Depth map (H, W) - Higher value = closer (MiDAS convention)
        mask: Optional binary mask to keep only object points
        fov: Approximate Field of View in degrees
        subsample: Take every Nth pixel (to reduce point count)
        
    Returns:
        points: (N, 3) array of XYZ coordinates
        colors: (N, 3) array of RGB colors (0-1 float)
    """
    h, w = depth.shape
    
    # Intrinsic matrix approximation
    fx = w / (2 * np.tan(np.radians(fov) / 2))
    fy = fx # Assuming square pixels
    cx = w / 2
    cy = h / 2
    
    # Grid of (u, v) coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Subsample to reduce size
    u = u[::subsample, ::subsample]
    v = v[::subsample, ::subsample]
    depth_sub = depth[::subsample, ::subsample]
    rgb_sub = rgb[::subsample, ::subsample]
    
    if mask is not None:
        mask_sub = mask[::subsample, ::subsample]
        valid = (mask_sub > 128)
    else:
        valid = np.ones_like(depth_sub, dtype=bool)
        
    # Filter valid points
    z = depth_sub[valid]
    
    # MiDAS returns relative inverse depth (disparity). 
    # We invert it to get something proportional to real depth.
    # Avoid division by zero
    z = np.maximum(z, 0.1)
    z = 1.0 / z 
    
    # Back-projection
    # x = (u - cx) * z / fx
    # y = (v - cy) * z / fy
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    
    points = np.stack([x, y, z], axis=1)
    
    # Normalize points to be centered and scaled for better visualization
    if len(points) > 0:
        centroid = np.mean(points, axis=0)
        points -= centroid
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points /= max_dist
            
    # Colors
    colors = rgb_sub[valid] / 255.0
    
    return points, colors

def save_ply(points: np.ndarray, colors: np.ndarray, filename: str):
    """
    Save point cloud to .ply file (readable by MeshLab, Blender).
    """
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
""".format(len(points))

    with open(filename, 'w') as f:
        f.write(header)
        for p, c in zip(points, colors):
            r, g, b = (c * 255).astype(int)
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {r} {g} {b}\n")
    
    print(f"[INFO] Saved PLY to {filename}")

def save_rotating_gif(
    points: np.ndarray, 
    colors: np.ndarray, 
    filename: str, 
    frames: int = 30
):
    """
    Render a rotating 3D view and save as GIF.
    """
    print(f"[INFO] Generating rotating GIF ({frames} frames)...")
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Invert Y and Z for better visual orientation (computer vision vs plot coords)
    points_plot = points.copy()
    points_plot[:, 1] *= -1 # Flip Y
    points_plot[:, 2] *= -1 # Flip Z
    
    images = []
    
    angles = np.linspace(0, 360, frames, endpoint=False)
    
    for angle in angles:
        ax.clear()
        ax.scatter(
            points_plot[:, 0], 
            points_plot[:, 2], # Swap Y/Z for plotting
            points_plot[:, 1], 
            c=colors, 
            s=1, 
            alpha=0.8
        )
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_zlim(-0.8, 0.8)
        ax.axis('off')
        ax.view_init(elev=10, azim=angle)
        
        # Save frame to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB"))
        plt.close(fig) # Prevent memory leak
        
        # Re-create figure for next frame (slower but safer)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    if images:
        images[0].save(
            filename,
            save_all=True,
            append_images=images[1:],
            optimize=True,
            duration=100,
            loop=0
        )
        print(f"[INFO] Saved GIF to {filename}")
    
    plt.close()

