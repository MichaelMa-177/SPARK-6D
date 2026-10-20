"""
Textured OBJ export utility for SAM3D-Objects.

This module provides functions to export meshes with proper UV mapping and textures
in a format compatible with Blender, MeshLab, and other 3D viewers.

Author: SAM3D Export Module
Date: 2025-12-16
"""

import os
import numpy as np
import trimesh
from PIL import Image
from typing import Union, Optional
import torch

from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import Gaussian
from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.cube2mesh import MeshExtractResult
from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import (
    postprocess_mesh,
    parametrize_mesh,
    bake_texture
)
from sam3d_objects.model.backbone.tdfy_dit.utils.render_utils import (
    render_frames,
    sphere_hammersley_sequence,
    yaw_pitch_r_fov_to_extrinsics_intrinsics
)

def export_textured_obj(
    mesh: MeshExtractResult,
    gaussian: Gaussian,
    output_dir: str,
    base_name: str = "model",
    texture_size: int = 1024,
    simplify_ratio: float = 0.95,
    fill_holes: bool = True,
    fill_holes_max_size: float = 0.04,
    bake_texture_from_gaussian: bool = True,
    rendering_engine: str = "gsplat",
    verbose: bool = True
) -> dict:
    """
    Export a textured mesh as OBJ + MTL + PNG with proper UV mapping.
    
    This function performs the complete pipeline:
    1. Mesh postprocessing (simplification + hole filling)
    2. UV parameterization using xatlas
    3. Texture baking from Gaussian splats via multi-view rendering
    4. Export to OBJ with relative paths
    
    Args:
        mesh: MeshExtractResult from SAM3D pipeline
        gaussian: Gaussian splats for texture baking
        output_dir: Output directory path
        base_name: Base filename (without extension)
        texture_size: Texture resolution (default: 1024x1024)
        simplify_ratio: Mesh simplification ratio (0.95 = keep 5%)
        fill_holes: Whether to fill mesh holes
        fill_holes_max_size: Maximum hole area to fill
        bake_texture_from_gaussian: If True, bake texture from Gaussian; 
                                     if False, use vertex colors only
        rendering_engine: "gsplat" or "nvdiffrast" (use gsplat to avoid compilation issues)
        verbose: Print progress
        
    Returns:
        dict with keys:
            - 'obj_path': Path to .obj file
            - 'mtl_path': Path to .mtl file  
            - 'texture_path': Path to texture .png file
            - 'vertices': Final vertex count
            - 'faces': Final face count
            - 'has_uv': Whether UV coordinates exist
            - 'has_texture': Whether texture was baked
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract mesh data
    vertices = mesh.vertices.float().cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    vert_colors = mesh.vertex_attrs[:, :3].cpu().numpy()
    
    if verbose:
        print(f"Original mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Step 1: Mesh postprocessing
    if verbose:
        print("Step 1/4: Mesh postprocessing (simplify + fill holes)...")
    
    vertices, faces = postprocess_mesh(
        vertices,
        faces,
        simplify=simplify_ratio > 0,
        simplify_ratio=simplify_ratio,
        fill_holes=fill_holes,
        fill_holes_max_hole_size=fill_holes_max_size,
        fill_holes_max_hole_nbe=int(250 * np.sqrt(1 - simplify_ratio)),
        fill_holes_resolution=1024,
        fill_holes_num_views=1000,
        debug=False,
        verbose=verbose,
    )
    
    if verbose:
        print(f"After postprocess: {len(vertices)} vertices, {len(faces)} faces")
    
    # Step 2: UV parameterization
    if verbose:
        print("Step 2/4: UV parameterization (xatlas)...")
    
    try:
        vertices, faces, uvs = parametrize_mesh(vertices, faces)
        has_uv = True
        if verbose:
            print(f"UV parameterization successful: {len(uvs)} UV coordinates")
    except Exception as e:
        if verbose:
            print(f"Warning: UV parameterization failed: {e}")
            print("Falling back to vertex colors only")
        has_uv = False
        uvs = None
    
    # Step 3: Texture baking (if requested and UV available)
    texture_img = None
    material = None
    has_texture = False
    
    if bake_texture_from_gaussian and has_uv:
        if verbose:
            print(f"Step 3/4: Texture baking from Gaussian (mode=fast)...")
        
        try:
            # Render multi-view observations
            # Note: render_multiview doesn't accept backend param, but render_frames does
            # We need to use gsplat backend to avoid kernel_size incompatibility
            from sam3d_objects.model.backbone.tdfy_dit.utils.render_utils import (
                render_frames,
                sphere_hammersley_sequence,
                yaw_pitch_r_fov_to_extrinsics_intrinsics
            )
            
            # Generate camera positions (same as render_multiview)
            nviews = 100
            r = 2
            fov = 40
            cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
            yaws = [cam[0] for cam in cams]
            pitchs = [cam[1] for cam in cams]
            # Pass the Gaussian's device to ensure all tensors are on the same device
            extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
                yaws, pitchs, r, fov, device=gaussian.device
            )
            
            # Render with gsplat backend
            res = render_frames(
                gaussian,
                extrinsics,
                intrinsics,
                {"resolution": 1024, "bg_color": (0, 0, 0), "backend": "gsplat"},
            )
            observations = res["color"]
            masks = [np.any(observation > 0, axis=-1) for observation in observations]
            extrinsics = [extrinsics[i].cpu().numpy() for i in range(len(extrinsics))]
            intrinsics = [intrinsics[i].cpu().numpy() for i in range(len(intrinsics))]
            
            # Bake texture using fast mode (opt mode only supports nvdiffrast/pytorch3d)
            texture = bake_texture(
                vertices.astype(np.float32) if hasattr(vertices, 'astype') else vertices,
                faces.astype(np.int32) if hasattr(faces, 'astype') else faces,
                uvs.astype(np.float32) if hasattr(uvs, 'astype') else uvs,
                observations,
                masks,
                [ext.astype(np.float32) if hasattr(ext, 'astype') else ext for ext in extrinsics],
                [intr.astype(np.float32) if hasattr(intr, 'astype') else intr for intr in intrinsics],
                texture_size=texture_size,
                mode="fast",
                lambda_tv=0.01,
                verbose=verbose,
                rendering_engine="cuda"  # fast mode uses RastContext backend
            )
            
            # Create PIL Image (CRITICAL: must be PIL Image for trimesh export)
            texture_img = Image.fromarray(texture.astype(np.uint8), mode='RGB')
            
            # Create PBR material
            material = trimesh.visual.material.PBRMaterial(
                roughnessFactor=1.0,
                baseColorTexture=texture_img,
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            )
            has_texture = True
            
            if verbose:
                print(f"Texture baked: {texture.shape[1]}x{texture.shape[0]} RGB")
                
        except Exception as e:
            if verbose:
                print(f"Warning: Texture baking failed: {e}")
                print("Falling back to vertex colors")
            has_texture = False
    
    # Step 4: Create trimesh and export
    if verbose:
        print("Step 4/4: Creating trimesh and exporting...")
    
    # Rotate mesh from z-up to y-up (SAM3D coordinate system)
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    
    # Create trimesh based on available data
    if has_texture and has_uv:
        # Best case: full textured mesh with UVs
        visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
        mesh_trimesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            visual=visual,
            process=False
        )
    elif has_uv:
        # UV available but no texture: export UV-mapped mesh with white material
        simple_material = trimesh.visual.material.SimpleMaterial(
            diffuse=[255, 255, 255, 255]
        )
        visual = trimesh.visual.TextureVisuals(uv=uvs, material=simple_material)
        mesh_trimesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            visual=visual,
            process=False
        )
    else:
        # Fallback: vertex colors only (no UV)
        mesh_trimesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False
        )
        mesh_trimesh.visual.vertex_colors = vert_colors
    
    # Export OBJ (trimesh handles MTL and PNG automatically)
    obj_path = os.path.join(output_dir, f"{base_name}.obj")
    mesh_trimesh.export(obj_path)
    
    # Check what files were actually created
    mtl_path = os.path.join(output_dir, "material.mtl")
    texture_path = os.path.join(output_dir, "material_0.png")
    
    # Rename to match base_name for clarity
    final_mtl_path = os.path.join(output_dir, f"{base_name}.mtl")
    final_texture_path = os.path.join(output_dir, f"{base_name}_texture.png")
    
    if os.path.exists(mtl_path):
        # Update OBJ to reference correct MTL filename
        with open(obj_path, 'r') as f:
            obj_content = f.read()
        obj_content = obj_content.replace('mtllib material.mtl', f'mtllib {base_name}.mtl')
        with open(obj_path, 'w') as f:
            f.write(obj_content)
        
        # Update MTL to reference correct texture filename and save with new name
        with open(mtl_path, 'r') as f:
            mtl_content = f.read()
        mtl_content = mtl_content.replace('material_0.png', f'{base_name}_texture.png')
        with open(final_mtl_path, 'w') as f:
            f.write(mtl_content)
        
        # Remove old MTL file if different from new one
        if mtl_path != final_mtl_path and os.path.exists(mtl_path):
            os.remove(mtl_path)
        
        # Rename texture file
        if os.path.exists(texture_path):
            # Remove old texture if exists (avoid conflict)
            if os.path.exists(final_texture_path):
                os.remove(final_texture_path)
            os.rename(texture_path, final_texture_path)
    else:
        final_mtl_path = None
        final_texture_path = None
    
    result = {
        'obj_path': obj_path,
        'mtl_path': final_mtl_path,
        'texture_path': final_texture_path,
        'vertices': len(vertices),
        'faces': len(faces),
        'has_uv': has_uv,
        'has_texture': has_texture,
    }
    
    if verbose:
        print("\n" + "="*60)
        print("EXPORT COMPLETE")
        print("="*60)
        print(f"OBJ file:     {obj_path}")
        print(f"MTL file:     {final_mtl_path if final_mtl_path else 'N/A (vertex colors only)'}")
        print(f"Texture file: {final_texture_path if final_texture_path else 'N/A'}")
        print(f"Vertices:     {result['vertices']}")
        print(f"Faces:        {result['faces']}")
        print(f"Has UVs:      {result['has_uv']}")
        print(f"Has Texture:  {result['has_texture']}")
        print("="*60)
    
    return result


def validate_textured_obj(output_dir: str, base_name: str = "model") -> dict:
    """
    Validate that a textured OBJ export is correct and loadable.
    
    Args:
        output_dir: Output directory path
        base_name: Base filename
        
    Returns:
        dict with validation results:
            - 'obj_exists': bool
            - 'mtl_exists': bool
            - 'texture_exists': bool
            - 'obj_has_mtllib': bool
            - 'obj_has_usemtl': bool
            - 'obj_has_vt': bool (UV coordinates)
            - 'mtl_has_map_Kd': bool (texture reference)
            - 'all_checks_passed': bool
    """
    
    obj_path = os.path.join(output_dir, f"{base_name}.obj")
    mtl_path = os.path.join(output_dir, f"{base_name}.mtl")
    texture_path = os.path.join(output_dir, f"{base_name}_texture.png")
    
    result = {
        'obj_exists': os.path.exists(obj_path),
        'mtl_exists': os.path.exists(mtl_path),
        'texture_exists': os.path.exists(texture_path),
        'obj_has_mtllib': False,
        'obj_has_usemtl': False,
        'obj_has_vt': False,
        'mtl_has_map_Kd': False,
        'all_checks_passed': False,
    }
    
    # Check OBJ content
    if result['obj_exists']:
        with open(obj_path, 'r') as f:
            obj_content = f.read()
        result['obj_has_mtllib'] = 'mtllib' in obj_content
        result['obj_has_usemtl'] = 'usemtl' in obj_content
        result['obj_has_vt'] = '\nvt ' in obj_content or obj_content.startswith('vt ')
    
    # Check MTL content
    if result['mtl_exists']:
        with open(mtl_path, 'r') as f:
            mtl_content = f.read()
        result['mtl_has_map_Kd'] = 'map_Kd' in mtl_content
    
    # Overall validation
    result['all_checks_passed'] = (
        result['obj_exists'] and
        result['mtl_exists'] and
        result['texture_exists'] and
        result['obj_has_mtllib'] and
        result['obj_has_usemtl'] and
        result['obj_has_vt'] and
        result['mtl_has_map_Kd']
    )
    
    return result