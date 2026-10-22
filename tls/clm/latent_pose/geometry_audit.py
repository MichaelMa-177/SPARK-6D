"""GT-assisted generated-shape diagnosis; no fitting and no inference output."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .bop import annotation_pose, object_annotation
from .data import sha256
from .rgbd import measured_geometry


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True)
    p.add_argument("--reference-scene", required=True)
    p.add_argument("--cad", required=True, help="official millimetre BOP evaluation mesh")
    p.add_argument("--object-id", type=int, default=15)
    p.add_argument("--mesh-directories", nargs="+", required=True)
    p.add_argument("--refinement-reports", nargs="+")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.refinement_reports and len(args.refinement_reports) != len(args.mesh_directories):
        p.error("each mesh must have its corresponding refinement report")
    import trimesh
    inputs, scene, output = Path(args.inputs), Path(args.reference_scene), Path(args.output)
    frames = sorted((scene/"rgb").glob("*.png"))
    if not frames or sha256(frames[0]) != sha256(inputs/"first_frame.png"):
        raise ValueError("reference scene differs from input")
    camera = json.loads((inputs/"camera.json").read_text())
    rgb = cv2.cvtColor(cv2.imread(str(inputs/"first_frame.png")), cv2.COLOR_BGR2RGB)
    depth, mask = cv2.imread(str(inputs/"first_depth.png"), -1), cv2.imread(str(inputs/"mask.png"), 0) > 0
    _, xyz, valid = measured_geometry(rgb, depth, np.asarray(camera["cam_K"]).reshape(3, 3), camera["depth_scale_m"])
    measured = xyz[mask & valid]
    gt = json.loads((scene/"scene_gt.json").read_text())
    truth = annotation_pose(object_annotation(gt[str(int(frames[0].stem))], args.object_id)[1])
    cad = trimesh.load(args.cad, force="mesh")
    cad.apply_scale(.001)
    def samples(mesh):
        np.random.seed(0)
        return trimesh.sample.sample_surface(mesh, 100000)[0]
    def transform(points, pose):
        return points @ pose[:3, :3].T + pose[:3, 3]
    cad_points = transform(samples(cad), truth)
    cad_tree = cKDTree(cad_points)
    def distances(values):
        mm = values*1000
        return {"median_mm": float(np.median(mm)), "p90_mm": float(np.percentile(mm, 90)),
                "within_5mm": float((mm <= 5).mean()), "within_10mm": float((mm <= 10).mean())}
    reports = []
    for index, name in enumerate(args.mesh_directories):
        folder = Path(name)
        report = json.loads((folder/"report.json").read_text())
        if report["rgb_sha256"] != sha256(inputs/"first_frame.png") or report["mesh_sha256"] != sha256(folder/"generated.ply"):
            raise ValueError("generated mesh provenance mismatch")
        points = samples(trimesh.load(folder/"generated.ply", force="mesh"))
        transforms = {"original_layout": np.asarray(report["first_camera_from_memory"])}
        if args.refinement_reports:
            refinement = json.loads(Path(args.refinement_reports[index]).read_text())
            if refinement["mesh_sha256"] != report["mesh_sha256"]:
                raise ValueError("refinement uses a different mesh")
            transforms["reference_only_refinement"] = np.asarray(refinement["calibrated_first_camera_from_generated"])
        metrics = {}
        for key, pose in transforms.items():
            generated = transform(points, pose)
            generated_tree = cKDTree(generated)
            metrics[key] = {"generated_to_CAD": distances(cad_tree.query(generated, workers=4)[0]),
                            "CAD_to_generated": distances(generated_tree.query(cad_points, workers=4)[0]),
                            "measured_to_generated": distances(generated_tree.query(measured, workers=4)[0])}
        reports.append({"mesh_directory": str(folder), "mesh_sha256": report["mesh_sha256"], "metrics": metrics})
    output.mkdir(parents=True, exist_ok=False)
    report = {"diagnostic_only": True, "GT_and_CAD_used_only_for_evaluation": True,
              "GT_alignment_or_shape_fitting": False, "surface_sampling": "100000 triangle-area samples, seed 0",
              "measured_to_CAD": distances(cad_tree.query(measured, workers=4)[0]),
              "cad_sha256": sha256(args.cad), "GT_sha256": sha256(scene/"scene_gt.json"),
              "code_sha256": sha256(__file__), "reports": reports}
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(reports), flush=True)


if __name__ == "__main__":
    main()
