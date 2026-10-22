"""Refine a frozen single-reference SS hypothesis into SLAT and a metric mesh."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

from .data import sha256
from .export_reference import sam3d_to_camera
from .rgbd import measured_geometry


OFFICIAL_WEIGHTS = {
    "slat_generator.ckpt": "91529bde8e7daa12d09618a66c319e3a5a6398db6b23b958cedcb1c3f28faabb",
    "slat_decoder_mesh.ckpt": "85907b37b67d8ce5b099a96629bdcfbd873eb407dee6b3aa9a75deb15038db33",
}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--coarse-directory", required=True, help="generate_reference_mesh output for one SS step count")
    parser.add_argument("--source", required=True)
    parser.add_argument("--dino-repo", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    inputs, coarse, source, weights, output = map(Path, (
        args.inputs, args.coarse_directory, args.source, args.weights, args.output))
    coarse_report = json.loads((coarse/"report.json").read_text())
    if coarse_report["GT_or_CAD_input"] or coarse_report["prior_sha256"] != sha256(coarse/"prior.npz"):
        raise ValueError("coarse prior provenance differs")
    for key, name in (("rgb_sha256", "first_frame.png"), ("depth_sha256", "first_depth.png"),
                      ("mask_sha256", "mask.png"), ("camera_sha256", "camera.json"),
                      ("prompt_sha256", "user_prompt.json")):
        if sha256(inputs/name) != coarse_report[key]:
            raise ValueError("coarse hypothesis and supplied reference differ")
    prompt = json.loads((inputs/"user_prompt.json").read_text())
    if prompt["prompt_source"] != "human_interactive_click" or prompt["rgb_sha256"] != coarse_report["rgb_sha256"]:
        raise ValueError("expected the exact saved human reference")
    for name, expected in OFFICIAL_WEIGHTS.items():
        if sha256(weights/name) != expected:
            raise ValueError(f"official SLAT weight hash mismatch: {name}")
    rgb = cv2.cvtColor(cv2.imread(str(inputs/"first_frame.png")), cv2.COLOR_BGR2RGB)
    depth, mask = cv2.imread(str(inputs/"first_depth.png"), -1), cv2.imread(str(inputs/"mask.png"), 0) > 0
    camera = json.loads((inputs/"camera.json").read_text())
    intrinsics = np.asarray(camera["cam_K"]).reshape(3, 3)
    _, xyz, valid = measured_geometry(rgb, depth, intrinsics, camera["depth_scale_m"])
    origin = xyz[mask & valid].mean(0)
    np.testing.assert_allclose(origin, coarse_report["memory_origin_m"], atol=1e-7, rtol=0)
    with np.load(coarse/"prior.npz", allow_pickle=False) as saved:
        logits = saved["occupancy_logits"].copy()
        layout = {key: saved["layout_"+key].copy() for key in ("scale", "rotation", "translation")}
    if logits.shape != (64, 64, 64) or not np.isfinite(logits).all() or not (logits > 0).any():
        raise ValueError("invalid coarse occupancy")
    output.mkdir(parents=True, exist_ok=False)
    (output/"generate_slat_mesh.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4)
    os.environ.update(SAM3D_DINO_REPO=str(args.dino_repo),
                      SAM3D_DINO_WEIGHTS=str(weights/"dinov2_vitl14_reg4_pretrain.pth"),
                      ATTN_BACKEND="sdpa", SPARSE_ATTN_BACKEND="sdpa", LIDRA_SKIP_INIT="1")
    sys.path.insert(0, str(source))
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sam3d_objects.pipeline.inference_utils import prune_sparse_structure, downsample_sparse_structure
    cfg = OmegaConf.load(weights/"pipeline.yaml")
    cfg.workspace_dir = str(weights)
    cfg.compile_model = False
    cfg.low_vram = True
    cfg.decode_formats = ["mesh"]
    cfg.depth_model = {"_target_": "torch.nn.Identity"}
    cfg.rendering_engine = "pytorch3d"
    started = time.perf_counter()
    with torch.device("cuda"):
        pipeline = instantiate(cfg)
    rgba = np.concatenate((rgb, (mask.astype(np.uint8)*255)[..., None]), axis=-1)
    item = pipeline.preprocess_image(rgba, pipeline.slat_preprocessor)
    coordinates = torch.argwhere(torch.from_numpy(logits).cuda()[None, None] > 0)[:, [0, 2, 3, 4]].int()
    original_count = len(coordinates)
    if pipeline.downsample_ss_dist > 0:
        coordinates = prune_sparse_structure(coordinates, max_neighbor_axes_dist=pipeline.downsample_ss_dist)
    coordinates, downsample_factor = downsample_sparse_structure(coordinates)
    layout["scale"] *= downsample_factor
    pipeline._load_low_vram_stage2()
    torch.cuda.synchronize()
    loaded = time.perf_counter()
    torch.manual_seed(args.seed)
    slat = pipeline.sample_slat(item, coordinates, inference_steps=args.steps, use_distillation=False)
    torch.cuda.synchronize()
    sampled = time.perf_counter()
    np.savez_compressed(output/"slat.npz", coordinates=slat.coords.cpu().numpy(),
                        features=slat.feats.float().cpu().numpy(),
                        **{"layout_"+key: value for key, value in layout.items()})
    pipeline._load_low_vram_decoders(["mesh"])
    mesh_output = pipeline.decode_slat(slat, ["mesh"])["mesh"][0]
    torch.cuda.synchronize()
    decoded = time.perf_counter()
    if not mesh_output.success:
        raise ValueError("SLAT decoder returned an empty mesh")
    canonical = mesh_output.vertices.float().cpu().numpy()
    faces = mesh_output.faces.cpu().numpy()
    attributes = mesh_output.vertex_attrs.float().cpu().numpy() if mesh_output.vertex_attrs is not None else np.empty((len(canonical), 0))
    vertices_camera = sam3d_to_camera(canonical, layout["scale"], layout["rotation"], layout["translation"])
    from pytorch3d.transforms import quaternion_to_matrix
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
    tensors = {key: torch.as_tensor(value, device="cuda") for key, value in layout.items()}
    transform = compose_transform(tensors["scale"], quaternion_to_matrix(tensors["rotation"]), tensors["translation"])
    check = transform.transform_points(torch.from_numpy(canonical).cuda()[None])[0].cpu().numpy()
    check[:, :2] *= -1
    transform_error = float(np.abs(check-vertices_camera).max())
    if not np.isfinite(vertices_camera).all() or transform_error > 1e-5:
        raise ValueError("metric SLAT mesh transform failed")
    # Match the SS control's color protocol. Generated attributes are preserved
    # separately, so the first comparison changes geometry rather than texture.
    projected = vertices_camera @ intrinsics.T
    uv = np.rint(projected[:, :2]/np.maximum(projected[:, 2:], 1e-6)).astype(int)
    inside = ((vertices_camera[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < rgb.shape[1])
              & (uv[:, 1] >= 0) & (uv[:, 1] < rgb.shape[0]))
    ids = np.flatnonzero(inside)
    u, v = uv[ids].T
    supported = mask[v, u] & valid[v, u] & (np.abs(xyz[v, u, 2]-vertices_camera[ids, 2]) < .01)
    colored = ids[supported]
    median = np.median(rgb[mask & valid], axis=0).astype(np.uint8)
    colors = np.tile(np.r_[median, np.uint8(255)], (len(canonical), 1))
    colors[colored, :3] = rgb[uv[colored, 1], uv[colored, 0]]
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices_camera-origin, faces=faces, vertex_colors=colors, process=False)
    mesh.export(output/"generated.ply")
    np.savez_compressed(output/"decoded_canonical.npz", vertices=canonical, faces=faces, attributes=attributes)
    first_pose = np.eye(4)
    first_pose[:3, 3] = origin
    report = {
        "source": "fixed supplied RGB-D SS hypothesis -> frozen SLAT -> mesh",
        "GT_or_CAD_input": False, "SAM_segmentation_executed": False, "units": "metres",
        "memory_origin_m": origin.tolist(), "first_camera_from_memory": first_pose.tolist(),
        "coordinate_convention": "first-camera axes; measured reference centroid origin",
        "rgb_sha256": coarse_report["rgb_sha256"], "depth_sha256": coarse_report["depth_sha256"],
        "mask_sha256": coarse_report["mask_sha256"], "camera_sha256": coarse_report["camera_sha256"],
        "prompt_sha256": coarse_report["prompt_sha256"], "coarse_report_sha256": sha256(coarse/"report.json"),
        "coarse_prior_sha256": sha256(coarse/"prior.npz"), "coarse_SS_steps": coarse_report["steps"],
        "SLAT_steps": args.steps, "SLAT_seed": args.seed, "SLAT_distillation": False,
        "original_SS_coordinates": original_count, "SLAT_coordinates": len(coordinates),
        "downsample_factor": downsample_factor, "layout": {key: value.tolist() for key, value in layout.items()},
        "max_transform_error_m": transform_error, "vertices": len(mesh.vertices), "faces": len(mesh.faces),
        "extent_m": mesh.extents.tolist(), "reference_projected_vertex_fraction": len(colored)/len(canonical),
        "unobserved_vertex_color": "median measured foreground RGB, not generated texture",
        "seconds": {"initialization": loaded-started, "SLAT_sampling": sampled-loaded,
                    "mesh_decoder_load_and_inference": decoded-sampled},
        "mesh_sha256": sha256(output/"generated.ply"), "slat_sha256": sha256(output/"slat.npz"),
        "weight_sha256": {**OFFICIAL_WEIGHTS,
                          "dinov2_vitl14_reg4_pretrain.pth": sha256(weights/"dinov2_vitl14_reg4_pretrain.pth")},
        "code_sha256": sha256(__file__),
        "vendor_pipeline_sha256": sha256(source/"sam3d_objects/pipeline/inference_pipeline.py"),
        "config_sha256": {name: sha256(weights/name) for name in
                          ("pipeline.yaml", "slat_generator.yaml", "slat_decoder_mesh.yaml")},
    }
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
