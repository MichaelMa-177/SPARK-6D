"""Decode a saved single-frame SS occupancy field into a metric inference mesh."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .data import sha256
from .export_reference import sam3d_to_camera
from .rgbd import measured_geometry


def mesh_from_fields(logits, layout, rgb, xyz, valid, mask, k):
    from skimage.measure import marching_cubes
    import trimesh
    if logits.shape != (64, 64, 64) or not np.isfinite(logits).all():
        raise ValueError("expected finite 64-cubed SS decoder field")
    measured = xyz[mask & valid]
    if len(measured) < 12:
        raise ValueError("reference has insufficient measured geometry")
    origin = measured.mean(0)
    # Field nodes represent voxel centres, matching the upstream mesh voxelizer.
    vertices, faces, _, _ = marching_cubes(np.pad(logits, 1, constant_values=-30), level=0)
    canonical = (vertices-1+.5)/64-.5
    vertices_camera = sam3d_to_camera(canonical, layout["scale"], layout["rotation"], layout["translation"])
    projected = vertices_camera @ k.T
    uv = np.rint(projected[:, :2]/projected[:, 2:]).astype(int)
    inside = (vertices_camera[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < rgb.shape[1]) & (
        uv[:, 1] >= 0) & (uv[:, 1] < rgb.shape[0])
    ids = np.where(inside)[0]
    u, v = uv[ids].T
    observed = mask[v, u] & valid[v, u] & (np.abs(xyz[v, u, 2]-vertices_camera[ids, 2]) < .01)
    colored = ids[observed]
    median = np.median(rgb[mask & valid], axis=0).astype(np.uint8)
    colors = np.tile(np.r_[median, np.uint8(255)], (len(vertices_camera), 1))
    colors[colored, :3] = rgb[uv[colored, 1], uv[colored, 0]]
    mesh = trimesh.Trimesh(vertices=vertices_camera-origin, faces=faces,
                           vertex_colors=colors, process=False)
    reference_pose = np.eye(4)
    reference_pose[:3, 3] = origin
    return mesh, {
        "GT_or_CAD_input": False, "units": "metres",
        "memory_origin_m": origin.tolist(), "first_camera_from_memory": reference_pose.tolist(),
        "coordinate_convention": "memory axes equal the first RGB camera; mesh vertices are in memory coordinates",
        "surface": "isosurface of generated surface-occupancy logits; approximate geometry, not measured CAD",
        "reference_projected_vertex_fraction": len(colored)/len(vertices_camera),
        "unobserved_vertex_color": "median measured foreground RGB, not inferred texture",
        "vertices": len(mesh.vertices), "faces": len(mesh.faces), "extent_m": mesh.extents.tolist(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior", required=True, help="saved coarse_shape and layout_* NPZ")
    p.add_argument("--occupancy", required=True, help="verified decoder logits NPZ")
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    prior, occupancy, inputs, output = map(Path, (args.prior, args.occupancy, args.inputs, args.output))
    prior_report = json.loads(prior.with_suffix(".json").read_text())
    decoder_report = json.loads((occupancy.parent/"report.json").read_text())
    if sha256(prior) != decoder_report["source_latent_sha256"]:
        raise ValueError("occupancy decoder source differs from saved prior")
    if sha256(occupancy) != decoder_report["output_sha256"][occupancy.name]:
        raise ValueError("occupancy differs from verified decoder output")
    if sha256(inputs/"first_frame.png") != prior_report["rgb_sha256"] or (
        sha256(inputs/"mask.png") != prior_report["human_mask_sha256"]):
        raise ValueError("saved human reference differs from the generated latent")
    rgb = cv2.cvtColor(cv2.imread(str(inputs/"first_frame.png")), cv2.COLOR_BGR2RGB)
    depth, mask = cv2.imread(str(inputs/"first_depth.png"), -1), cv2.imread(str(inputs/"mask.png"), 0) > 0
    camera = json.loads((inputs/"camera.json").read_text())
    k = np.array(camera["cam_K"]).reshape(3, 3)
    _, xyz, valid = measured_geometry(rgb, depth, k, camera["depth_scale_m"])
    measured = xyz[mask & valid]
    with np.load(occupancy.parent/"measured.npz", allow_pickle=False) as original:
        np.testing.assert_allclose(measured, original["xyz_m"], atol=1e-6, rtol=1e-6)
    with np.load(occupancy, allow_pickle=False) as data:
        logits = data["logits"]
    with np.load(prior, allow_pickle=False) as data:
        layout = {key: data["layout_"+key] for key in ("scale", "rotation", "translation")}
    mesh, geometry_report = mesh_from_fields(logits, layout, rgb, xyz, valid, mask, k)
    output.mkdir(parents=True, exist_ok=False)
    mesh_path = output/"generated.ply"
    mesh.export(mesh_path)
    report = {
        **geometry_report,
        "source": "single human-selected RGB-D reference -> frozen SS latent -> decoded occupancy",
        "source_prior_sha256": sha256(prior), "source_occupancy_sha256": sha256(occupancy),
        "mask_sha256": sha256(inputs/"mask.png"), "depth_sha256": sha256(inputs/"first_depth.png"),
        "rgb_sha256": sha256(inputs/"first_frame.png"), "mesh_sha256": sha256(mesh_path),
        "code_sha256": sha256(__file__),
    }
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
