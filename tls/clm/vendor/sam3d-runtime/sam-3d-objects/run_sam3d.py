#!/usr/bin/env python3
"""
Validation script for METRIC-SCALED textured OBJ export from SAM3D-Objects.

This script:
1. Loads an example image, mask, and DEPTH from data_online
2. Runs SAM3D inference  
3. [ADDED] Performs ICP-based scale alignment using depth point cloud
4. Exports metric-scaled textured mesh 
5. Validates the output files

The key addition is ICP + scale alignment INSIDE the pipeline:
- Depth image + MASK + INTRINSICS → metric point cloud
- Bounding box-based initial scale estimation
- ICP rigid registration (rotation + translation)
- Scale applied to BOTH mesh vertices AND Gaussian coordinates

CRITICAL FIX: Temporarily normalize mesh/gaussian for texture baking,
then scale back to metric for final export.

Output:
    Creates files in notebook/meshes/single/:
        - model.obj (METRIC-SCALE geometry with UVs)
        - model.mtl (material file)
        - model_texture.png (texture image)
"""

import os
import sys
import numpy as np
import cv2
import torch
import open3d as o3d
import json
import glob
import trimesh
from PIL import Image
from sklearn.decomposition import PCA

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
NOTEBOOK_DIR = os.path.join(PROJECT_ROOT, "notebook")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, NOTEBOOK_DIR)

from inference import Inference, load_image, load_single_mask
from sam3d_objects.utils.export_textured_obj import validate_textured_obj


# =============================================================================
# Multi-GPU Setup for Parallelized Inference + Rendering
# =============================================================================

def setup_multi_gpu():
    """
    Detect and configure available GPUs for parallelized inference.
    
    Strategy (True Dual-GPU):
    - GPU 0: Encoders (ss_generator, slat_generator, ss_decoder, condition_embedders)
    - GPU 1: Decoders (slat_decoder_mesh, slat_decoder_gs) + Texture rendering
    - CPU: ICP optimization (Open3D, sequential)
    
    This prevents OOM during model initialization by placing memory-heavy
    mesh decoder (with ~1GB dense grid) on a separate GPU.
    
    Returns:
        device_inference: torch.device for encoders (cuda:0)
        device_decoder: torch.device for decoders (cuda:1)
    """
    if not torch.cuda.is_available():
        print("[GPU] CUDA not available, using CPU")
        return torch.device('cpu'), torch.device('cpu')
    
    num_gpus = torch.cuda.device_count()
    
    if num_gpus >= 2:
        device_inference = torch.device('cuda:0')
        device_decoder = torch.device('cuda:1')
        print(f"[Multi-GPU] Detected {num_gpus} GPUs")
        print(f"[Multi-GPU] GPU 0: Encoders (ss/slat_generator, condition_embedders)")
        print(f"[Multi-GPU] GPU 1: Decoders (slat_decoder_mesh/gs) + Rendering")
    else:
        device_inference = torch.device('cuda:0')
        device_decoder = torch.device('cuda:0')
        print(f"[Single-GPU] Using GPU 0 for all models")
    
    return device_inference, device_decoder

def apply_multi_gpu_to_pipeline(inference_obj, device_encoder='cuda:0', device_decoder='cuda:1'):
    """
    Legacy function - now handled during Inference initialization.
    
    The true dual-GPU setup is done by passing decoder_device to Inference(),
    which places decoders on GPU 1 during model construction (not post-hoc).
    
    This avoids OOM during initialization and handles cross-GPU tensor transfer
    in the decode_slat() method.
    """
    # Multi-GPU is now handled at initialization time
    print(f"[Multi-GPU] Decoders initialized on {inference_obj._pipeline.decoder_device}")
    print(f"[Multi-GPU] Encoders remain on {inference_obj._pipeline.device}")
    return inference_obj

def batch_process_multi_gpu(image_paths, mask_paths, config_path, output_dirs, devices=['cuda:0', 'cuda:1']):
    """
    Process multiple objects in parallel across GPUs.
    
    Args:
        image_paths: List of image file paths
        mask_paths: List of mask file paths  
        config_path: Pipeline config path
        output_dirs: List of output directories
        devices: List of GPU devices to use
        
    Returns:
        List of results, one per object
    """
    import multiprocessing as mp
    from multiprocessing import Queue
    
    def worker(gpu_id, device, task_queue, result_queue):
        """Worker process for one GPU."""
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = device.split(':')[1]
        
        # Import here to ensure correct GPU binding
        import torch
        from notebook.inference import Inference
        
        # Create inference pipeline on this GPU
        inference = Inference(config_path, compile=False)
        
        while True:
            task = task_queue.get()
            if task is None:  # Poison pill
                break
                
            idx, img_path, mask_path, out_dir = task
            try:
                # Load and process
                from PIL import Image
                import numpy as np
                
                image = np.array(Image.open(img_path))
                mask = np.array(Image.open(mask_path)) > 0
                
                output = inference(image, mask, seed=42)
                result_queue.put((idx, 'success', output))
            except Exception as e:
                result_queue.put((idx, 'error', str(e)))
    
    # Create task and result queues
    task_queue = Queue()
    result_queue = Queue()
    
    # Enqueue all tasks
    for idx, (img, mask, out) in enumerate(zip(image_paths, mask_paths, output_dirs)):
        task_queue.put((idx, img, mask, out))
    
    # Add poison pills
    for _ in devices:
        task_queue.put(None)
    
    # Start workers
    processes = []
    for gpu_id, device in enumerate(devices):
        p = mp.Process(target=worker, args=(gpu_id, device, task_queue, result_queue))
        p.start()
        processes.append(p)
    
    # Collect results
    results = [None] * len(image_paths)
    for _ in range(len(image_paths)):
        idx, status, data = result_queue.get()
        results[idx] = (status, data)
    
    # Wait for all workers
    for p in processes:
        p.join()
    
    return results






# =============================================================================
# [ADDED] ICP + Scale Alignment Functions (Based on notebook/demo.py)
# =============================================================================

def load_depth_pointcloud(depth_path, mask, intrinsics_path, depth_scale=0.001):
    """
    Load depth image and convert masked region to metric point cloud.
    (Based on notebook/demo.py implementation)
    
    Args:
        depth_path: Path to depth PNG (uint16, millimeters)
        mask: Binary mask (H, W) indicating object region
        intrinsics_path: Path to camera intrinsics JSON
        depth_scale: Depth unit conversion (default 0.001 for mm->m)
    
    Returns:
        o3d.geometry.PointCloud: Point cloud in camera coordinates (meters)
    """
    # Load depth
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    depth = depth * depth_scale  # Convert to meters
    
    print(f"[Depth] Image size: {depth.shape[1]}x{depth.shape[0]}")
    print(f"[Depth] Depth range (raw): [{depth.min()/depth_scale:.0f}, {depth.max()/depth_scale:.0f}] (uint16)")
    print(f"[Depth] Depth range (meters): [{depth.min():.3f}, {depth.max():.3f}]")
    
    # Load intrinsics (support both JSON and cam_K.txt formats)
    if intrinsics_path.endswith('.json'):
        with open(intrinsics_path, 'r') as f:
            intr = json.load(f)
        fx, fy = intr['fx'], intr['fy']
        cx, cy = intr['ppx'], intr['ppy']
    elif intrinsics_path.endswith('.txt'):
        # cam_K.txt format: 3x3 camera matrix
        K = np.loadtxt(intrinsics_path)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
    else:
        raise ValueError(f"Unsupported intrinsics format: {intrinsics_path}")
    
    print(f"[Depth] Using camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    # Generate point cloud from masked region
    H, W = depth.shape
    
    # Handle both bool and uint8 mask types
    if mask.dtype == bool:
        mask_bool = mask
    else:
        mask_bool = mask > 0
    
    print(f"[Depth] Mask coverage: {mask_bool.sum()} / {mask_bool.size} pixels ({100*mask_bool.sum()/mask_bool.size:.1f}%)")
    
    ys, xs = np.where(mask_bool)
    zs = depth[mask_bool]
    
    # Filter invalid depths
    valid = (zs > 0) & (zs < 10.0)  # Reasonable depth range
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    
    print(f"[Depth] Valid depth points: {len(xs)} ({100*len(xs)/(H*W):.1f}%)")
    
    # Backproject to 3D
    points_3d = np.zeros((len(xs), 3))
    points_3d[:, 0] = (xs - cx) * zs / fx
    points_3d[:, 1] = (ys - cy) * zs / fy
    points_3d[:, 2] = zs
    
    # Print object extent in 3D
    extent_3d = np.ptp(points_3d, axis=0)
    print(f"[Depth] Object extent in 3D: X={extent_3d[0]:.3f}m, Y={extent_3d[1]:.3f}m, Z={extent_3d[2]:.3f}m")
    print(f"[Depth] Object size (max dimension): {extent_3d.max():.3f}m")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    return pcd


def align_mesh_to_depth_icp(mesh_vertices, depth_pcd, max_iterations=100):
    """
    Align normalized mesh to metric depth point cloud using ICP + scale.
    (Based on notebook/demo.py implementation)
    
    Args:
        mesh_vertices: (N, 3) numpy array, normalized mesh vertices
        depth_pcd: o3d.geometry.PointCloud, metric-scale depth point cloud
        max_iterations: ICP max iterations
    
    Returns:
        aligned_vertices: (N, 3) numpy array, vertices in metric scale
        scale_factor: float, computed scale factor
        transform: (4, 4) transformation matrix (rotation + translation)
    """
    # Convert mesh to point cloud
    mesh_pcd = o3d.geometry.PointCloud()
    mesh_pcd.points = o3d.utility.Vector3dVector(mesh_vertices)
    
    # Step 1: DEPTH-BASED SCALE ESTIMATION (robust, orientation-independent)
    # Uses depth Z-axis extent + PCA principal axis for mesh
    # Handles axis-symmetric objects (e.g., coffee cups) correctly
    
    depth_points = np.asarray(depth_pcd.points)
    
    # 1.1 Robust 3D extent of depth point cloud (all axes)
    # Compute percentile-based extent for X, Y, Z separately
    d_p10 = np.percentile(depth_points, 10, axis=0)  # [X, Y, Z] lower bounds
    d_p90 = np.percentile(depth_points, 90, axis=0)  # [X, Y, Z] upper bounds
    depth_extent_3d = d_p90 - d_p10  # 3D extent vector
    
    # Use maximum dimension as object size (handles arbitrary orientation)
    robust_depth_extent = np.max(depth_extent_3d)  # meters
    
    print(f"\n[DEPTH-SCALE] Depth 3D extent: X={depth_extent_3d[0]*100:.1f}cm, Y={depth_extent_3d[1]*100:.1f}cm, Z={depth_extent_3d[2]*100:.1f}cm")
    print(f"[DEPTH-SCALE] Robust depth extent (max): {robust_depth_extent:.4f}m ({robust_depth_extent*100:.1f}cm)")
    
    # 1.2 Mesh principal axis using PCA (handles arbitrary orientation)
    pca = PCA(n_components=3)
    pca.fit(mesh_vertices)
    principal_axis = pca.components_[0]  # Direction of max variance
    explained_var = pca.explained_variance_ratio_[0]
    
    print(f"[DEPTH-SCALE] Mesh principal axis: [{principal_axis[0]:.2f}, {principal_axis[1]:.2f}, {principal_axis[2]:.2f}]")
    print(f"[DEPTH-SCALE] Explained variance: {explained_var*100:.1f}%")
    
    # 1.3 Project mesh onto principal axis and compute robust extent
    mesh_proj = mesh_vertices @ principal_axis
    m_p10 = np.percentile(mesh_proj, 10)
    m_p90 = np.percentile(mesh_proj, 90)
    robust_mesh_extent = m_p90 - m_p10  # Normalized units
    
    print(f"[DEPTH-SCALE] Mesh extent (principal): {robust_mesh_extent:.4f} units")
    
    # 1.4 Compute scale factor
    scale_factor = robust_depth_extent / robust_mesh_extent
    
    print(f"[DEPTH-SCALE] Final scale factor: {scale_factor:.4f}")
    print(f"[DEPTH-SCALE] Estimated object size: {robust_mesh_extent * scale_factor * 100:.1f}cm")
    
    # Step 2: Apply initial scale
    mesh_pcd_scaled = o3d.geometry.PointCloud()
    mesh_pcd_scaled.points = o3d.utility.Vector3dVector(
        np.asarray(mesh_pcd.points) * scale_factor
    )
    
    # Step 3: ICP rigid alignment (rotation + translation only)
    threshold = 0.02  # 2cm
    trans_init = np.eye(4)
    
    print(f"[ICP] Running ICP registration (threshold={threshold}m)...")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        mesh_pcd_scaled, depth_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )
    
    print(f"[ICP] Fitness: {reg_p2p.fitness:.4f} (fraction of inlier correspondences)")
    print(f"[ICP] RMSE: {reg_p2p.inlier_rmse:.6f} m")
    
    # Step 4: Apply ICP transformation
    transform = reg_p2p.transformation
    aligned_vertices = (mesh_vertices * scale_factor) @ transform[:3, :3].T + transform[:3, 3]
    
    return aligned_vertices, scale_factor, transform


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    print("="*70)
    print("SAM3D-Objects METRIC-SCALED Textured OBJ Export")
    print("="*70)
    
    # Configuration
    TAG = "hf"
    config_path = os.path.join(PROJECT_ROOT, "checkpoints", TAG, "pipeline.yaml")
    
    # Find the largest numbered directory in data_online
    data_online_root = os.path.join(PROJECT_ROOT, "..", "data_online")
    numbered_dirs = []
    for item in os.listdir(data_online_root):
        item_path = os.path.join(data_online_root, item)
        if os.path.isdir(item_path) and item.isdigit():
            numbered_dirs.append(int(item))
    
    if not numbered_dirs:
        print(f"\n❌ ERROR: No numbered directories found in {data_online_root}")
        return 1
    
    max_dir = max(numbered_dirs)
    max_dir_path = os.path.join(data_online_root, str(max_dir))
    
    # All input paths from the largest numbered directory
    # Support two directory structures:
    # Structure 1: preprocess_image/00000_rgb.png, intrinsics.json
    # Structure 2: rgb/*.png, depth/*.png, cam_K.txt
    
    preprocess_image_dir = os.path.join(max_dir_path, "preprocess_image")
    rgb_dir = os.path.join(max_dir_path, "rgb")
    depth_dir = os.path.join(max_dir_path, "depth")
    
    # Check which structure by looking for actual files
    preprocess_has_files = os.path.exists(preprocess_image_dir) and len(glob.glob(os.path.join(preprocess_image_dir, "*.png"))) > 0
    
    if preprocess_has_files:
        # Structure 1: preprocess_image directory
        rgb_files = sorted(glob.glob(os.path.join(preprocess_image_dir, "*rgb*.png")))
        if not rgb_files:
            rgb_files = sorted(glob.glob(os.path.join(preprocess_image_dir, "*.png")))
            rgb_files = [f for f in rgb_files if 'depth' not in os.path.basename(f).lower()]
        
        depth_files = sorted(glob.glob(os.path.join(preprocess_image_dir, "*depth*.png")))
        intrinsics_path = os.path.join(max_dir_path, "intrinsics.json")
        
        if not rgb_files or not depth_files:
            print(f"\n❌ ERROR: RGB or Depth files not found in {preprocess_image_dir}")
            return 1
            
    elif os.path.exists(rgb_dir):
        # Structure 2: separate rgb/ and depth/ directories
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
        depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
        
        if not rgb_files:
            print(f"\n❌ ERROR: No RGB files found in {rgb_dir}")
            return 1
        if not depth_files:
            print(f"\n❌ ERROR: No depth files found in {depth_dir}")
            return 1
        
        # Check for intrinsics file (cam_K.txt or intrinsics.json)
        cam_k_path = os.path.join(max_dir_path, "cam_K.txt")
        intrinsics_json_path = os.path.join(max_dir_path, "intrinsics.json")
        
        if os.path.exists(intrinsics_json_path):
            intrinsics_path = intrinsics_json_path
        elif os.path.exists(cam_k_path):
            intrinsics_path = cam_k_path
        else:
            print(f"\n❌ ERROR: No intrinsics file found (intrinsics.json or cam_K.txt)")
            return 1
    else:
        print(f"\n❌ ERROR: No valid image directory found in {max_dir_path}")
        return 1
    
    image_path = rgb_files[0]
    depth_path = depth_files[0]
    mask_dir = os.path.join(max_dir_path, "masks")
    
    output_dir = os.path.join(max_dir_path, "mesh")
    os.makedirs(output_dir, exist_ok=True)
    base_name = "model"
    
    print(f"\nConfiguration:")
    print(f"  Config:      {config_path}")
    print(f"  Image:       {image_path}")
    print(f"  Depth:       {depth_path}")
    print(f"  Intrinsics:  {intrinsics_path}")
    print(f"  Mask dir:    {mask_dir}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Base name:   {base_name}")
    
    # Check input files
    if not os.path.exists(config_path):
        print(f"\n❌ ERROR: Config not found: {config_path}")
        return 1
    
    if not os.path.exists(image_path):
        print(f"\n❌ ERROR: Image not found: {image_path}")
        return 1
    
    if not os.path.exists(depth_path):
        print(f"\n❌ ERROR: Depth image not found: {depth_path}")
        return 1
    
    if not os.path.exists(intrinsics_path):
        print(f"\n❌ ERROR: Intrinsics file not found: {intrinsics_path}")
        return 1
    
    print(f"\n✓ Input files exist")
    
    # Step 1: Load image and mask
    print(f"\n" + "="*70)
    print("Step 1: Loading image and mask")
    print("="*70)
    
    image = load_image(image_path)
    
    # Auto-detect mask file in mask/ directory
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if not mask_files:
        print(f"\n❌ ERROR: No mask PNG files found in {mask_dir}")
        return 1
    
    mask_path = mask_files[0]  # Use first PNG as mask
    print(f"Auto-detected mask: {os.path.basename(mask_path)}")
    mask = load_image(mask_path)
    
    # Ensure mask is binary (convert to bool if grayscale)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]  # Use first channel if RGB
    if mask.dtype != bool:
        mask = (mask > 127).astype(bool)  # Binarize
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask dtype: {mask.dtype}")
    
    # Step 2: Run inference
    print(f"\n" + "="*70)
    print("Step 2: Running SAM3D inference")
    print("="*70)
    
    # Setup multi-GPU devices
    device_inference, device_decoder = setup_multi_gpu()
    
    # Pass decoder_device to Inference for true dual-GPU initialization
    # This places decoders on GPU 1 DURING model construction, avoiding OOM
    decoder_device_str = str(device_decoder) if device_decoder != device_inference else None
    inference = Inference(config_path, compile=False, decoder_device=decoder_device_str)
    
    # Log the multi-GPU configuration
    if decoder_device_str:
        apply_multi_gpu_to_pipeline(inference, device_encoder=str(device_inference), device_decoder=decoder_device_str)
    
    output = inference(image, mask, seed=42)
    
    print(f"\nInference output keys: {list(output.keys())}")
    
    if "mesh" not in output or len(output["mesh"]) == 0:
        print(f"\n❌ ERROR: No mesh in output")
        return 1
    
    if "gs" not in output:
        print(f"\n❌ ERROR: No Gaussian splats in output")
        return 1
    
    mesh = output["mesh"][0]
    gaussian = output["gs"]
    
    # Note: Gaussian object stores device internally, cannot use .to()
    # Rendering device will be set in export_textured_obj if needed
    if 'device_rendering' in locals() and device_rendering != device_inference:
        print(f"[Multi-GPU] Gaussian ({gaussian.get_xyz.shape[0]} points) ready for rendering on GPU 1")
        # Gaussian device migration handled by rendering engine
    
    print(f"✓ Mesh extracted: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
    print(f"✓ Gaussian splats: {gaussian.get_xyz.shape[0]} points")
    
    # Step 3: Load depth point cloud with camera intrinsics
    print(f"\n" + "="*70)
    print("Step 3: Loading depth point cloud for ICP alignment")
    print("="*70)
    
    try:
        depth_pcd = load_depth_pointcloud(
            depth_path, 
            mask=mask,
            intrinsics_path=intrinsics_path,
            depth_scale=0.001  # mm -> m
        )
        print(f"✓ Depth point cloud loaded: {len(depth_pcd.points)} points")
    except Exception as e:
        print(f"❌ ERROR: Failed to load depth: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 4: ICP + Scale Alignment
    print(f"\n" + "="*70)
    print("Step 4: ICP + Scale Alignment (Normalized → Metric)")
    print("="*70)
    
    # Extract mesh vertices in normalized space
    mesh_vertices_normalized = mesh.vertices.float().cpu().numpy()
    
    # Perform ICP alignment
    try:
        mesh_vertices_aligned, scale_factor, transform_matrix = align_mesh_to_depth_icp(
            mesh_vertices_normalized,
            depth_pcd,
            max_iterations=100
        )
        print(f"\n✓ ICP alignment successful")
        print(f"  Final scale factor: {scale_factor:.6f}")
        print(f"  Mesh dimensions: X={np.ptp(mesh_vertices_aligned[:, 0]):.3f}m, "
              f"Y={np.ptp(mesh_vertices_aligned[:, 1]):.3f}m, "
              f"Z={np.ptp(mesh_vertices_aligned[:, 2]):.3f}m")
    except Exception as e:
        print(f"❌ ERROR: ICP alignment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 5: Export textured mesh with correct texture baking
    print(f"\n" + "="*70)
    print("Step 5: Exporting metric-scaled textured mesh")
    print("="*70)
    print(f"[CRITICAL] Keeping mesh/gaussian in NORMALIZED space for texture baking")
    print(f"[CRITICAL] Will scale final OBJ to metric after texture is baked")
    
    # Import here to use the ORIGINAL normalized mesh/gaussian for texture baking
    from sam3d_objects.utils.export_textured_obj import export_textured_obj
    
    # Move Gaussian to GPU 0 for texture baking (render_utils uses .cuda() which defaults to GPU 0)
    # This is necessary because decoders run on GPU 1 but renderers expect GPU 0
    if device_decoder != device_inference:
        print(f"[Multi-GPU] Moving Gaussian from {device_decoder} to {device_inference} for texture baking")
        try:
            # Gaussian has internal tensors that need to be moved
            import torch
            with torch.cuda.device(device_inference):
                # Core tensor attributes
                gaussian._xyz = gaussian._xyz.to(device_inference)
                gaussian._features_dc = gaussian._features_dc.to(device_inference)
                if hasattr(gaussian, '_features_rest') and gaussian._features_rest is not None:
                    gaussian._features_rest = gaussian._features_rest.to(device_inference)
                gaussian._opacity = gaussian._opacity.to(device_inference)
                gaussian._scaling = gaussian._scaling.to(device_inference)
                gaussian._rotation = gaussian._rotation.to(device_inference)
                gaussian.aabb = gaussian.aabb.to(device_inference)
                
                # Additional internal tensors - bias tensors from setup_functions()
                # These are torch.Tensor objects that need to be moved
                if hasattr(gaussian, 'scale_bias'):
                    if isinstance(gaussian.scale_bias, torch.Tensor):
                        gaussian.scale_bias = gaussian.scale_bias.to(device_inference)
                if hasattr(gaussian, 'rots_bias'):
                    if isinstance(gaussian.rots_bias, torch.Tensor):
                        gaussian.rots_bias = gaussian.rots_bias.to(device_inference)
                if hasattr(gaussian, 'opacity_bias'):  # Note: actual name is opacity_bias, not opacity_bias_tensor
                    if isinstance(gaussian.opacity_bias, torch.Tensor):
                        gaussian.opacity_bias = gaussian.opacity_bias.to(device_inference)
                
                # Update device reference
                gaussian.device = device_inference
            print(f"[Multi-GPU] Gaussian moved successfully (all tensors on {device_inference})")
            # Debug: verify all tensors are on correct device
            print(f"[DEBUG] Verifying Gaussian tensor devices:")
            print(f"  _xyz: {gaussian._xyz.device}")
            print(f"  _features_dc: {gaussian._features_dc.device}")
            print(f"  _opacity: {gaussian._opacity.device}")
            print(f"  _scaling: {gaussian._scaling.device}")
            print(f"  _rotation: {gaussian._rotation.device}")
            print(f"  aabb: {gaussian.aabb.device}")
            if hasattr(gaussian, 'scale_bias') and isinstance(gaussian.scale_bias, torch.Tensor):
                print(f"  scale_bias: {gaussian.scale_bias.device}")
            if hasattr(gaussian, 'rots_bias') and isinstance(gaussian.rots_bias, torch.Tensor):
                print(f"  rots_bias: {gaussian.rots_bias.device}")
            if hasattr(gaussian, 'opacity_bias') and isinstance(gaussian.opacity_bias, torch.Tensor):
                print(f"  opacity_bias: {gaussian.opacity_bias.device}")
            print(f"  gaussian.device: {gaussian.device}")
            # Debug: verify all tensors are on correct device
            print(f"[DEBUG] Verifying Gaussian tensor devices:")
            print(f"  _xyz: {gaussian._xyz.device}")
            print(f"  _features_dc: {gaussian._features_dc.device}")
            print(f"  _opacity: {gaussian._opacity.device}")
            print(f"  _scaling: {gaussian._scaling.device}")
            print(f"  _rotation: {gaussian._rotation.device}")
            print(f"  aabb: {gaussian.aabb.device}")
            if hasattr(gaussian, 'scale_bias') and isinstance(gaussian.scale_bias, torch.Tensor):
                print(f"  scale_bias: {gaussian.scale_bias.device}")
            if hasattr(gaussian, 'rots_bias') and isinstance(gaussian.rots_bias, torch.Tensor):
                print(f"  rots_bias: {gaussian.rots_bias.device}")
            if hasattr(gaussian, 'opacity_bias') and isinstance(gaussian.opacity_bias, torch.Tensor):
                print(f"  opacity_bias: {gaussian.opacity_bias.device}")
            print(f"  gaussian.device: {gaussian.device}")
        except Exception as e:
            print(f"[Multi-GPU] Warning: Could not move Gaussian: {e}")
            print("[Multi-GPU] Texture baking may fail")
    
    # Export with NORMALIZED mesh (this ensures correct camera distance for rendering)
    # Note: Gaussian has been m# =========================
# 2. Python 解释器
# =========================oved to GPU 0, and gaussian_render.py is device-aware
    # So texture baking should work correctly
    
    result = export_textured_obj(
        mesh=mesh,  # Still normalized!
        gaussian=gaussian,  # Now on GPU 0 if multi-GPU!
        output_dir=output_dir,
        base_name="model_normalized",  # Temporary name
        texture_size=1024,
        simplify_ratio=0.95,
        fill_holes=True,
        bake_texture_from_gaussian=True,  # Enable texture baking (Gaussian is on correct device)
        rendering_engine="gsplat",
        verbose=True
    )
    
    # Step 6: Scale the final OBJ to metric units
    print(f"\n" + "="*70)
    print("Step 6: Scaling final mesh to metric units")
    print("="*70)
    
    # Load the normalized OBJ (already in y-up from export_textured_obj)
    mesh_normalized = trimesh.load(result['obj_path'])
    
    print(f"[Scale] Loaded mesh: {len(mesh_normalized.vertices)} vertices")
    print(f"[Scale] Original extent (normalized, y-up): {np.ptp(mesh_normalized.vertices, axis=0)}")
    
    # The mesh from export_textured_obj is already rotated to y-up
    # We need to reverse the rotation to get back to z-up, apply scale, then rotate to y-up again
    rotation_z_to_y = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_y_to_z = rotation_z_to_y.T  # Inverse rotation
    
    # Step 1: Reverse rotation (y-up → z-up) to get back to normalized space
    vertices_z_up = mesh_normalized.vertices @ rotation_y_to_z
    
    # Step 2: Apply scale + ICP transformation in z-up space
    vertices_scaled = (vertices_z_up * scale_factor) @ transform_matrix[:3, :3].T + transform_matrix[:3, 3]
    
    # Step 3: Rotate back to y-up for final export
    mesh_normalized.vertices = vertices_scaled @ rotation_z_to_y
    
    print(f"[Scale] After scaling: {np.ptp(mesh_normalized.vertices, axis=0)}")
    print(f"[Scale] Scale factor applied: {scale_factor:.6f}")
    
    # Save as final model
    final_obj_path = os.path.join(output_dir, f"{base_name}.obj")
    mesh_normalized.export(final_obj_path)
    
    # Rename texture files
    normalized_mtl = os.path.join(output_dir, "model_normalized.mtl")
    normalized_texture = os.path.join(output_dir, "model_normalized_texture.png")
    final_mtl = os.path.join(output_dir, f"{base_name}.mtl")
    final_texture = os.path.join(output_dir, f"{base_name}_texture.png")
    
    if os.path.exists(normalized_mtl):
        os.rename(normalized_mtl, final_mtl)
    if os.path.exists(normalized_texture):
        os.rename(normalized_texture, final_texture)
    
    # Update OBJ to reference correct MTL
    with open(final_obj_path, 'r') as f:
        obj_content = f.read()
    obj_content = obj_content.replace('model_normalized.mtl', f'{base_name}.mtl')
    with open(final_obj_path, 'w') as f:
        f.write(obj_content)
    
    # Update MTL to reference correct texture
    with open(final_mtl, 'r') as f:
        mtl_content = f.read()
    mtl_content = mtl_content.replace('model_normalized_texture.png', f'{base_name}_texture.png')
    with open(final_mtl, 'w') as f:
        f.write(mtl_content)
    
    print(f"✓ Scaled mesh to metric units: {scale_factor:.6f}x")
    print(f"✓ Final files:")
    print(f"  OBJ: {final_obj_path}")
    print(f"  MTL: {final_mtl}")
    print(f"  PNG: {final_texture}")
    
    # Step 7: Validate output
    print(f"\n" + "="*70)
    print("Step 7: Validating output files")
    print("="*70)
    
    validation = validate_textured_obj(output_dir, base_name)
    
    print(f"\n📋 VALIDATION CHECKLIST:")
    print(f"{'='*70}")
    
    checks = [
        ("OBJ file exists", validation['obj_exists']),
        ("MTL file exists", validation['mtl_exists']),
        ("Texture PNG exists", validation['texture_exists']),
        ("OBJ contains 'mtllib' directive", validation['obj_has_mtllib']),
        ("OBJ contains 'usemtl' directive", validation['obj_has_usemtl']),
        ("OBJ contains 'vt' (UV coords)", validation['obj_has_vt']),
        ("MTL contains 'map_Kd' (texture)", validation['mtl_has_map_Kd']),
    ]
    
    for check_name, passed in checks:
        status = "✓" if passed else "❌"
        print(f"  {status} {check_name}")
    
    print(f"{'='*70}")
    
    if validation['all_checks_passed']:
        print(f"\n SUCCESS! All validation checks passed!")
        print(f"\n METRIC SCALE INFO:")
        print(f"  Scale factor: {scale_factor:.6f}")
        print(f"  Mesh dimensions: X={np.ptp(mesh_vertices_aligned[:, 0]):.3f}m, "
              f"Y={np.ptp(mesh_vertices_aligned[:, 1]):.3f}m, "
              f"Z={np.ptp(mesh_vertices_aligned[:, 2]):.3f}m")
        print(f"\nYou can now open the following files in Blender/MeshLab:")
        print(f"  {final_obj_path}")
        print(f"\nThe mesh is now in METRIC SCALE and suitable for robotics applications.")
        return 0
    else:
        print(f"\n⚠️  WARNING: Some validation checks failed")
        print(f"\nFiles created:")
        print(f"  OBJ: {final_obj_path} ({'EXISTS' if os.path.exists(final_obj_path) else 'MISSING'})")
        print(f"  MTL: {final_mtl} ({'EXISTS' if os.path.exists(final_mtl) else 'MISSING'})")
        print(f"  PNG: {final_texture} ({'EXISTS' if os.path.exists(final_texture) else 'MISSING'})")
        
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
