"""Offline FoundationPose reference trajectory; GT loaded only after prediction."""

import argparse
import json
import logging
from pathlib import Path
import time

import cv2
import numpy as np

from .bop import annotation_pose, object_annotation
from .data import sha256
from .evaluate import pose_errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--mesh", required=True, help="BOP mesh vertices in millimetres")
    p.add_argument("--inputs", required=True, help="saved exact human reference RGB/depth/mask/prompt")
    p.add_argument("--object-id", type=int, default=15)
    p.add_argument("--register-iterations", type=int, default=5)
    p.add_argument("--track-iterations", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    scene, inputs, output = Path(args.scene), Path(args.inputs), Path(args.output)
    images = sorted((scene/"rgb").glob("*.png"))
    if args.max_frames:
        images = images[:args.max_frames]
    if not images:
        raise ValueError("no RGB frames found")
    prompt = json.loads((inputs/"user_prompt.json").read_text())
    if prompt["prompt_source"] != "human_interactive_click":
        raise ValueError("teacher reference must reuse a saved human-selected mask")
    if sha256(images[0]) != prompt["rgb_sha256"] or sha256(inputs/"first_frame.png") != prompt["rgb_sha256"]:
        raise ValueError("first scene frame differs from the confirmed prompt")
    if not np.array_equal(cv2.imread(str(inputs/"first_depth.png"), -1),
                          cv2.imread(str(scene/"depth"/images[0].name), -1)):
        raise ValueError("reference depth mismatch")
    mask = cv2.imread(str(inputs/"mask.png"), 0) > 0
    cameras = json.loads((scene/"scene_camera.json").read_text())
    saved_camera = json.loads((inputs/"camera.json").read_text())
    first_camera = cameras[str(int(images[0].stem))]
    if not np.allclose(saved_camera["cam_K"], first_camera["cam_K"]) or not np.isclose(
            saved_camera["depth_scale_m"], first_camera["depth_scale"] / 1000):
        raise ValueError("reference camera calibration mismatch")
    output.mkdir(parents=True, exist_ok=False)
    # Import the teacher only in its separate environment.
    import torch
    import trimesh
    import nvdiffrast.torch as dr
    import warp as wp
    from estimater import FoundationPose
    from Utils import set_seed
    torch.set_num_threads(4)
    set_seed(0)
    logging.getLogger().setLevel(logging.WARNING)
    wp.config.kernel_cache_dir = str(Path(output).parents[1]/"warp-cache")
    mesh = trimesh.load(args.mesh, force="mesh")
    mesh.apply_scale(.001)
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                         mesh=mesh, glctx=dr.RasterizeCudaContext(), debug=0,
                         debug_dir=str(output/"debug"))
    if est.kalman_filter is not None:
        raise RuntimeError("teacher must produce per-frame neural refinements without Kalman")
    poses, rows = [], []
    for index, path in enumerate(images):
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        camera = cameras[str(int(path.stem))]
        depth = cv2.imread(str(scene/"depth"/path.name), -1).astype(np.float32)
        depth *= camera["depth_scale"] / 1000
        k = np.array(camera["cam_K"]).reshape(3, 3)
        torch.cuda.synchronize()
        started = time.perf_counter()
        if index == 0:
            pose = est.register(K=k, rgb=rgb, depth=depth, ob_mask=mask,
                                iteration=args.register_iterations)
        else:
            pose = est.track_one(rgb=rgb, depth=depth, K=k, iteration=args.track_iterations)
        torch.cuda.synchronize()
        if not np.isfinite(pose).all():
            raise RuntimeError(f"non-finite teacher output for frame {path.stem}")
        poses.append(pose.copy())
        np.savetxt(output/f"{path.stem}.txt", pose)
        row = {"frame_id": int(path.stem), "index": index, "rgb_sha256": sha256(path),
               "mode": "register" if index == 0 else "neural_refine",
               "seconds": time.perf_counter() - started}
        rows.append(row)
        print(json.dumps(row), flush=True)
    np.savez_compressed(output/"predictions.npz", poses=np.stack(poses),
                        frame_ids=np.array([r["frame_id"] for r in rows]))
    # The complete teacher trajectory exists before any GT pose file is read.
    gt = json.loads((scene/"scene_gt.json").read_text())
    metrics = []
    for row, pose in zip(rows, poses):
        _, annotation = object_annotation(gt[str(row["frame_id"])], args.object_id)
        metrics.append({"frame_id": row["frame_id"], **pose_errors(
            pose, annotation_pose(annotation), np.eye(4)[None])})
    # Drill object 15 has no exact discrete symmetry in this development run.
    if args.object_id != 15:
        raise ValueError("add official object symmetry metadata before scoring other objects")
    gaps = np.diff([r["frame_id"] for r in rows])
    dense = bool(len(gaps) and np.all(gaps == 1))
    repo = Path(__file__).resolve().parents[3]
    weights = repo/"weights"
    report = {
        "label_type": "FoundationPose pseudo/reference trajectory", "GT_loaded_after_predictions": True,
        "kalman": False, "skipped_refinements": 0,
        "input_sampling": "consecutive frames" if dense else "available sparse keyframes",
        "max_frame_id_gap": int(gaps.max()) if len(gaps) else 0,
        "weights_sha256": {str(p.relative_to(weights)): sha256(p) for p in sorted(weights.rglob("*"))
                           if p.is_file() and p.suffix in {".pth", ".yml", ".yaml"}},
        "source_sha256": {str(p.relative_to(repo)): sha256(p) for p in
                           (repo/"estimater.py", Path(__file__).resolve())},
        "pose_convention": "original BOP CAD -> current camera; metres",
        "object_id": args.object_id, "mesh_sha256": sha256(args.mesh),
        "mask_sha256": sha256(inputs/"mask.png"), "prompt_sha256": sha256(inputs/"user_prompt.json"),
        "register_iterations": args.register_iterations, "track_iterations": args.track_iterations,
        "frames": len(rows),
        "rotation_median_deg": float(np.median([x["rotation_deg"] for x in metrics])),
        "translation_median_mm": float(np.median([x["translation_mm_at_best_rotation"] for x in metrics])),
        "pass_10mm_5deg_all_frames": float(np.mean([x["pass_10mm_5deg"] for x in metrics])),
        "records": rows, "per_frame_GT_errors": metrics,
    }
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"records", "per_frame_GT_errors"}}), flush=True)


if __name__ == "__main__":
    main()
