import os
import imageio
import uuid
import open3d as o3d
import numpy as np
import json
import cv2
import trimesh
from PIL import Image

# from IPython.display import Image as ImageDisplay
# 加入了尺寸查看代码get_ply_scale
from inference import Inference, ready_gaussian_for_video_rendering, render_video, load_image, load_single_mask, display_image, make_scene, interactive_visualizer, get_ply_scale

# [ADDED] Import for textured mesh export
try:
    from sam3d_objects.utils.export_textured_obj import export_textured_obj
    HAS_TEXTURED_EXPORT = True
except ImportError:
    HAS_TEXTURED_EXPORT = False
    print("Warning: export_textured_obj not available, using basic export")

PATH = os.getcwd()
TAG = "hf"

# 加入相对读取图片代码，归一化到data_online
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
config_path = os.path.join(PROJECT_ROOT, "sam-3d-objects", "checkpoints", TAG, "pipeline.yaml")

# [MODIFIED] Use structured data paths
DATA_ID = "2"
DATA_DIR = os.path.join(PROJECT_ROOT, "data_online", DATA_ID)
IMAGE_PATH = os.path.join(DATA_DIR, "preprocess_image", "00000_rgb.png")
DEPTH_PATH = os.path.join(DATA_DIR, "preprocess_image", "00000_depth.png")
INTRINSICS_PATH = os.path.join(DATA_DIR, "intrinsics.json")
IMAGE_NAME = os.path.basename(os.path.dirname(IMAGE_PATH))

###############################################################
# Initialize inference
###############################################################
inference = Inference(config_path, compile=False)

# Load image and mask
image = load_image(IMAGE_PATH)
mask = load_single_mask(os.path.dirname(IMAGE_PATH), index=1)
display_image(image, masks=[mask])

# Run model
output = inference(image, mask, seed=42)

# Export gaussian splat (as point cloud)
# output["gs"].save_ply(f"{PATH}/notebook/gaussians/single/{IMAGE_NAME}.ply")


###############################################################
# [ADDED] ICP Scale Alignment Functions
###############################################################

def load_depth_pointcloud(depth_path, mask, intrinsics_path, depth_scale=0.001):
    """
    Load depth image and convert masked region to metric point cloud.
    
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
    
    # Load intrinsics
    with open(intrinsics_path, 'r') as f:
        intr = json.load(f)
    fx, fy = intr['fx'], intr['fy']
    cx, cy = intr['ppx'], intr['ppy']
    
    # Generate point cloud from masked region
    H, W = depth.shape
    mask_bool = mask > 0
    
    ys, xs = np.where(mask_bool)
    zs = depth[mask_bool]
    
    # Filter invalid depths
    valid = (zs > 0) & (zs < 10.0)  # Reasonable depth range
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    
    # Backproject to 3D
    points_3d = np.zeros((len(xs), 3))
    points_3d[:, 0] = (xs - cx) * zs / fx
    points_3d[:, 1] = (ys - cy) * zs / fy
    points_3d[:, 2] = zs
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    return pcd


def align_mesh_to_depth_icp(mesh_vertices, depth_pcd, max_iterations=100):
    """
    Align normalized mesh to metric depth point cloud using ICP + scale.
    
    Args:
        mesh_vertices: (N, 3) numpy array, normalized mesh vertices
        depth_pcd: o3d.geometry.PointCloud, metric-scale depth point cloud
    
    Returns:
        aligned_vertices: (N, 3) numpy array, vertices in metric scale
        scale_factor: float, computed scale factor
        transform: (4, 4) transformation matrix (rotation + translation)
    """
    # Convert mesh to point cloud
    mesh_pcd = o3d.geometry.PointCloud()
    mesh_pcd.points = o3d.utility.Vector3dVector(mesh_vertices)
    
    # Step 1: Estimate initial scale via bounding box ratio
    mesh_bbox = mesh_pcd.get_axis_aligned_bounding_box()
    depth_bbox = depth_pcd.get_axis_aligned_bounding_box()
    
    mesh_extent = np.asarray(mesh_bbox.get_extent())
    depth_extent = np.asarray(depth_bbox.get_extent())
    
    # Use max dimension for robust scale estimation
    scale_factor = np.max(depth_extent) / np.max(mesh_extent)
    
    print(f"[ICP] Initial scale estimate: {scale_factor:.4f}")
    print(f"[ICP] Mesh extent:  {mesh_extent}")
    print(f"[ICP] Depth extent: {depth_extent}")
    
    # Step 2: Apply initial scale
    mesh_pcd_scaled = o3d.geometry.PointCloud()
    mesh_pcd_scaled.points = o3d.utility.Vector3dVector(
        np.asarray(mesh_pcd.points) * scale_factor
    )
    
    # Step 3: ICP rigid alignment (rotation + translation only)
    threshold = 0.02  # 2cm
    trans_init = np.eye(4)
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
        mesh_pcd_scaled, depth_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )
    
    print(f"[ICP] Fitness: {reg_p2p.fitness:.4f}, RMSE: {reg_p2p.inlier_rmse:.6f} m")
    
    # Step 4: Apply ICP transformation
    transform = reg_p2p.transformation
    aligned_vertices = (mesh_vertices * scale_factor) @ transform[:3, :3].T + transform[:3, 3]
    
    return aligned_vertices, scale_factor, transform


###############################################################
# Mesh Generation with Scale Alignment
###############################################################

if "mesh" in output and len(output["mesh"]) > 0:
    from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import to_glb
    
    mesh_extract_result = output["mesh"][0]  # MeshExtractResult 对象
    gaussian = output["gs"]  # Gaussian 对象用于纹理烘焙
    
    print("=" * 70)
    print("正在生成带贴图的 mesh...")
    print("  - 网格后处理: 简化 + 填充孔洞")
    print("  - 颜色来源: 顶点颜色（从原始mesh）")
    print("  - 贴图分辨率: 1024x1024")
    print("=" * 70)
    
    # =========================================================================
    # Stage 1: Mesh postprocessing (simplification + hole filling)
    # =========================================================================
    mesh_trimesh = to_glb(
        app_rep=gaussian,
        mesh=mesh_extract_result,
        simplify=0.95,              # 简化到原始面片的 5%
        fill_holes=True,            # 填充孔洞
        fill_holes_max_size=0.04,   # 最大孔洞尺寸
        texture_size=1024,          # 贴图分辨率
        debug=False,
        verbose=True,
        with_mesh_postprocess=True,    # 启用网格后处理
        with_texture_baking=False,     # 暂时禁用纹理烘焙（后面单独做）
        use_vertex_color=True,         # 保留顶点颜色
        rendering_engine="nvdiffrast"
    )
    
    print(f"\n[Postprocess] Initial mesh: {len(mesh_trimesh.vertices)} vertices, {len(mesh_trimesh.faces)} faces")
    
    # =========================================================================
    # [ADDED] Stage 2: ICP Scale Alignment
    # =========================================================================
    if os.path.exists(DEPTH_PATH) and os.path.exists(INTRINSICS_PATH):
        print("\n" + "=" * 70)
        print("[SCALE ALIGNMENT] Loading depth point cloud...")
        print("=" * 70)
        
        try:
            # Load depth-derived point cloud
            depth_pcd = load_depth_pointcloud(
                DEPTH_PATH, 
                mask, 
                INTRINSICS_PATH,
                depth_scale=0.001  # Assuming depth is in millimeters
            )
            
            print(f"[SCALE] Depth point cloud: {len(depth_pcd.points)} points")
            
            # Perform ICP alignment
            vertices_normalized = np.array(mesh_trimesh.vertices)
            vertices_aligned, scale_factor, transform = align_mesh_to_depth_icp(
                vertices_normalized,
                depth_pcd,
                max_iterations=100
            )
            
            # [CRITICAL] Apply scale to mesh vertices IN-PLACE
            mesh_trimesh.vertices = vertices_aligned
            
            print(f"\n[SCALE] ✓ Mesh rescaled to metric units")
            print(f"[SCALE] Scale factor: {scale_factor:.6f}")
            print(f"[SCALE] Transform:\n{transform}")
            
            # Also scale Gaussian for texture baking consistency
            if hasattr(gaussian, 'points'):
                gaussian_points = gaussian.points.cpu().numpy()
                gaussian_points_scaled = (gaussian_points * scale_factor) @ transform[:3, :3].T + transform[:3, 3]
                gaussian.points = gaussian.points.new_tensor(gaussian_points_scaled)
                print(f"[SCALE] Gaussian also rescaled ({len(gaussian_points)} points)")
            
        except Exception as e:
            print(f"\n[SCALE] Warning: ICP alignment failed: {e}")
            print("[SCALE] Continuing with normalized scale...")
    else:
        print("\n[SCALE] Warning: Depth or intrinsics not found, skipping scale alignment")
    
    # =========================================================================
    # [ADDED] Stage 3: Textured Export (OBJ + MTL + PNG)
    # =========================================================================
    mesh_dir = f"{PATH}/notebook/meshes/single"
    os.makedirs(mesh_dir, exist_ok=True)
    
    base_name = f"{IMAGE_NAME}_aligned"
    
    if HAS_TEXTURED_EXPORT:
        print("\n" + "=" * 70)
        print("[TEXTURE EXPORT] Using full textured pipeline...")
        print("=" * 70)
        
        try:
            # Reconstruct MeshExtractResult from aligned trimesh
            from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.cube2mesh import MeshExtractResult
            import torch
            
            aligned_mesh_result = MeshExtractResult(
                vertices=torch.tensor(mesh_trimesh.vertices, dtype=torch.float32),
                faces=torch.tensor(mesh_trimesh.faces, dtype=torch.int64),
                vertex_attrs=None
            )
            
            # Export with full texture baking
            export_result = export_textured_obj(
                mesh=aligned_mesh_result,
                gaussian=gaussian,
                output_dir=mesh_dir,
                base_name=base_name,
                texture_size=1024,
                simplify_ratio=0.0,  # Already simplified
                fill_holes=False,    # Already filled
                bake_texture_from_gaussian=True,
                rendering_engine="cuda",
                verbose=True
            )
            
            print("\n" + "=" * 70)
            print("✓ TEXTURED EXPORT COMPLETE")
            print("=" * 70)
            print(f"OBJ:     {export_result['obj_path']}")
            print(f"MTL:     {export_result['mtl_path']}")
            print(f"Texture: {export_result['texture_path']}")
            print(f"Vertices: {export_result['num_vertices']}")
            print(f"Faces:    {export_result['num_faces']}")
            print("=" * 70)
            
        except Exception as e:
            print(f"\n[TEXTURE] Warning: Textured export failed: {e}")
            print("[TEXTURE] Falling back to basic OBJ export...")
            HAS_TEXTURED_EXPORT = False
    
    # Fallback: Basic export
    if not HAS_TEXTURED_EXPORT:
        print("\n[EXPORT] Using basic trimesh export...")
        
        glb_path = f"{mesh_dir}/{base_name}.glb"
        obj_path = f"{mesh_dir}/{base_name}.obj"
        
        # Export GLB
        mesh_trimesh.export(glb_path)
        
        # Export OBJ with vertex colors
        mesh_trimesh.export(obj_path)
        
        print(f"✓ Basic export:")
        print(f"  - GLB: {glb_path}")
        print(f"  - OBJ: {obj_path}")
        print(f"  - Vertices: {len(mesh_trimesh.vertices)}")
        print(f"  - Faces:    {len(mesh_trimesh.faces)}")

# Export gaussian splat
PLY_PATH = f"{PATH}/notebook/gaussians/single/{IMAGE_NAME}.ply"
output["gs"].save_ply(PLY_PATH)

# Render gaussian splat
scene_gs = make_scene(output)
scene_gs = ready_gaussian_for_video_rendering(scene_gs)

video = render_video(
    scene_gs,
    r=1,
    fov=60,
    pitch_deg=15,
    yaw_start_deg=-45,
    resolution=512,
)["color"]

# Save video as gif
imageio.mimsave(
    os.path.join(f"{PATH}/notebook/gaussians/single/{IMAGE_NAME}.gif"),
    video,
    format="GIF",
    duration=1000 / 30,  # default assuming 30fps from the input MP4
    loop=0,  # 0 means loop indefinitely
)

get_ply_scale(PLY_PATH)

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
print(f"Gaussian PLY: {PLY_PATH}")
print(f"Gaussian GIF: {PATH}/notebook/gaussians/single/{IMAGE_NAME}.gif")
print("=" * 70)

# notebook display
# ImageDisplay(url=f"gaussians/single/{IMAGE_NAME}.gif?cache_invalidator={uuid.uuid4()}")
# might take a while to load (black screen)