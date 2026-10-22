"""Change only vertex colors on an already frozen, metric SLAT mesh."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .data import sha256
from .export_reference import sam3d_to_camera
from .rgbd import measured_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-directory", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--mode", choices=("generated", "hybrid"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import trimesh
    root, inputs, output = map(Path, (args.mesh_directory, args.inputs, args.output))
    report = json.loads((root/"report.json").read_text())
    if report["GT_or_CAD_input"] or report["units"] != "metres" or sha256(root/"generated.ply") != report["mesh_sha256"]:
        raise ValueError("expected the verified metric SLAT mesh")
    for key, name in (("rgb_sha256", "first_frame.png"), ("depth_sha256", "first_depth.png"),
                      ("mask_sha256", "mask.png"), ("camera_sha256", "camera.json")):
        if sha256(inputs/name) != report[key]:
            raise ValueError("reference differs")
    mesh = trimesh.load(root/"generated.ply", force="mesh", process=False)
    with np.load(root/"decoded_canonical.npz", allow_pickle=False) as data:
        canonical, faces, attributes = data["vertices"], data["faces"], data["attributes"]
    layout = {key: np.asarray(value, np.float32) for key, value in report["layout"].items()}
    camera_vertices = sam3d_to_camera(canonical, layout["scale"], layout["rotation"], layout["translation"])
    np.testing.assert_array_equal(faces, mesh.faces)
    np.testing.assert_allclose(camera_vertices-np.asarray(report["memory_origin_m"]), mesh.vertices, atol=1e-7, rtol=0)
    if attributes.ndim != 2 or attributes.shape[0] != len(mesh.vertices) or attributes.shape[1] < 3 or not np.isfinite(attributes).all():
        raise ValueError("missing generated vertex attributes")
    # Upstream postprocessing_utils.py also reads vertex_attrs[:, :3] as RGB.
    colors = np.concatenate((np.rint(np.clip(attributes[:, :3], 0, 1)*255).astype(np.uint8),
                             np.full((len(mesh.vertices), 1), 255, dtype=np.uint8)), axis=1)
    supported = np.empty(0, dtype=int)
    if args.mode == "hybrid":
        rgb = cv2.cvtColor(cv2.imread(str(inputs/"first_frame.png")), cv2.COLOR_BGR2RGB)
        depth, mask = cv2.imread(str(inputs/"first_depth.png"), -1), cv2.imread(str(inputs/"mask.png"), 0) > 0
        camera = json.loads((inputs/"camera.json").read_text())
        intrinsics = np.asarray(camera["cam_K"]).reshape(3, 3)
        _, xyz, valid = measured_geometry(rgb, depth, intrinsics, camera["depth_scale_m"])
        projected = camera_vertices @ intrinsics.T
        uv = np.rint(projected[:, :2]/np.maximum(projected[:, 2:], 1e-6)).astype(int)
        inside = ((camera_vertices[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < rgb.shape[1])
                  & (uv[:, 1] >= 0) & (uv[:, 1] < rgb.shape[0]))
        ids = np.flatnonzero(inside)
        u, v = uv[ids].T
        good = mask[v, u] & valid[v, u] & (np.abs(xyz[v, u, 2]-camera_vertices[ids, 2]) < .01)
        supported = ids[good]
        colors[supported, :3] = rgb[uv[supported, 1], uv[supported, 0]]
    mesh.visual.vertex_colors = colors
    output.mkdir(parents=True, exist_ok=False)
    mesh.export(output/"generated.ply")
    reloaded = trimesh.load(output/"generated.ply", force="mesh", process=False)
    np.testing.assert_array_equal(reloaded.vertices, mesh.vertices)
    np.testing.assert_array_equal(reloaded.faces, mesh.faces)
    (output/"slat_color_ablation.py").write_bytes(Path(__file__).read_bytes())
    report.update(source="same frozen SLAT geometry; vertex-color ablation",
                  color_mode=args.mode, source_mesh_report_sha256=sha256(root/"report.json"),
                  source_mesh_sha256=sha256(root/"generated.ply"),
                  decoded_canonical_sha256=sha256(root/"decoded_canonical.npz"),
                  mesh_sha256=sha256(output/"generated.ply"),
                  reference_projected_vertex_fraction=len(supported)/len(mesh.vertices),
                  unobserved_vertex_color="SLAT inferred RGB; not a measured back surface",
                  geometry_unchanged=True, color_ablation_code_sha256=sha256(__file__))
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"mode": args.mode, "mesh_sha256": report["mesh_sha256"],
                      "geometry_unchanged": True, "reference_colored_fraction": report["reference_projected_vertex_fraction"]}))


if __name__ == "__main__":
    main()
