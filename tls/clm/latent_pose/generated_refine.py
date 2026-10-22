"""Frozen FoundationPose refinement against a single-frame generated mesh."""

import argparse
import json
import logging
from pathlib import Path
import time

import cv2
import numpy as np

from .bop import annotation_pose, object_annotation
from .cli import write_json
from .data import sha256
from .video_evaluate import trajectory_metrics


def centered_seed(camera_from_memory, memory_to_centered):
    """Convert an original-mesh pose to the centered-mesh pose FP expects."""
    return np.asarray(camera_from_memory) @ np.linalg.inv(memory_to_centered)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--reference-scene", required=True)
    p.add_argument("--mesh-directory", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--initial-predictions", help="memory-frame poses and frame_ids NPZ; independent refinement")
    p.add_argument("--calibrate-reference", action="store_true",
                   help="anchor generated geometry using refinement on the supplied reference only")
    p.add_argument("--calibration-rounds", type=int, default=1,
                   help="reference-only refinement rounds before fixing the shared memory gauge")
    args = p.parse_args()
    if args.iterations <= 0 or args.calibration_rounds < 1 or args.max_frames < 0 or args.max_frames in {1, 2}:
        p.error("positive iterations and at least three frames required")
    scene, reference_scene, mesh_directory, inputs, output = map(
        Path, (args.scene, args.reference_scene, args.mesh_directory, args.inputs, args.output))
    mesh_path = mesh_directory/"generated.ply"
    mesh_report = json.loads((mesh_directory/"report.json").read_text())
    if mesh_report["GT_or_CAD_input"] or mesh_report["units"] != "metres":
        raise ValueError("expected a metric generated mesh without CAD or GT input")
    if sha256(mesh_path) != mesh_report["mesh_sha256"]:
        raise ValueError("generated mesh hash differs")
    for key, name in (("rgb_sha256", "first_frame.png"), ("depth_sha256", "first_depth.png"),
                      ("mask_sha256", "mask.png")):
        if sha256(inputs/name) != mesh_report[key]:
            raise ValueError("generated mesh reference differs")
    prompt = json.loads((inputs/"user_prompt.json").read_text())
    if prompt["prompt_source"] != "human_interactive_click" or prompt["rgb_sha256"] != mesh_report["rgb_sha256"]:
        raise ValueError("expected the saved exact human reference")
    reference_images = sorted((reference_scene/"rgb").glob("*.png"))
    if not reference_images or sha256(reference_images[0]) != mesh_report["rgb_sha256"]:
        raise ValueError("reference scene does not start at the saved reference")
    if not np.array_equal(cv2.imread(str(inputs/"first_depth.png"), -1),
                          cv2.imread(str(reference_scene/"depth"/reference_images[0].name), -1)):
        raise ValueError("reference depth differs")
    cameras = json.loads((scene/"scene_camera.json").read_text())
    ref_cameras = json.loads((reference_scene/"scene_camera.json").read_text())
    saved_camera = json.loads((inputs/"camera.json").read_text())
    ref_camera = ref_cameras[str(int(reference_images[0].stem))]
    if not np.allclose(saved_camera["cam_K"], ref_camera["cam_K"]) or not np.isclose(
            saved_camera["depth_scale_m"], ref_camera["depth_scale"]/1000):
        raise ValueError("reference calibration differs")
    first_pose = np.asarray(mesh_report["first_camera_from_memory"])
    images = sorted((scene/"rgb").glob("*.png"))
    seeds = None
    if args.initial_predictions:
        with np.load(args.initial_predictions, allow_pickle=False) as data:
            if str(data["reference_rgb_sha256"].item()) != mesh_report["rgb_sha256"] or not np.allclose(
                    data["memory_origin_m"], mesh_report["memory_origin_m"], atol=1e-7, rtol=0):
                raise ValueError("seed poses and generated mesh have different reference gauges")
            if len(np.unique(data["frame_ids"])) != len(data["frame_ids"]) or data["poses"].shape != (
                    len(data["frame_ids"]), 4, 4):
                raise ValueError("seed frame identifiers must uniquely match 4x4 poses")
            if "valid" in data and not data["valid"].all():
                raise ValueError("invalid seed poses require an explicit recovery policy")
            rotations = data["poses"][:, :3, :3]
            if not np.isfinite(data["poses"]).all() or not np.allclose(
                    data["poses"][:, 3], [0, 0, 0, 1], atol=1e-5) or not np.allclose(
                    rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-3) or not np.allclose(
                    np.linalg.det(rotations), 1, atol=1e-3):
                raise ValueError("seed poses must be finite proper rigid transforms")
            seeds = {int(f): pose.copy() for f, pose in zip(data["frame_ids"], data["poses"])}
        images = [im for im in images if int(im.stem) in seeds]
    elif scene.resolve() != reference_scene.resolve():
        raise ValueError("a different video requires independent predicted initial poses")
    if args.max_frames:
        images = images[:args.max_frames]
    if len(images) < 3:
        raise ValueError("need at least three query frames")
    output.mkdir(parents=True, exist_ok=False)
    import torch
    import trimesh
    import nvdiffrast.torch as dr
    import warp as wp
    from estimater import FoundationPose
    from Utils import set_seed
    torch.set_num_threads(4)
    set_seed(0)
    logging.getLogger().setLevel(logging.WARNING)
    wp.config.kernel_cache_dir = str(output.parents[1]/"warp-cache")
    mesh = trimesh.load(mesh_path, force="mesh")
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                         mesh=mesh, glctx=dr.RasterizeCudaContext(), debug=0,
                         debug_dir=str(output/"debug"))
    centered = est.get_tf_to_centered_mesh().cpu().numpy()
    if est.kalman_filter is not None:
        raise RuntimeError("expected per-frame refinement without Kalman")
    est.pose_last = torch.as_tensor(centered_seed(first_pose, centered), dtype=torch.float32, device="cuda")
    np.testing.assert_allclose((est.pose_last @ est.get_tf_to_centered_mesh()).cpu().numpy(),
                               first_pose, atol=1e-6)
    generated_from_memory = np.eye(4)
    calibrated_reference = first_pose.copy()
    calibration_history = []
    calibration_seconds = 0.
    if args.calibrate_reference:
        ref_rgb = cv2.cvtColor(cv2.imread(str(inputs/"first_frame.png")), cv2.COLOR_BGR2RGB)
        ref_depth = cv2.imread(str(inputs/"first_depth.png"), -1).astype(np.float32)*saved_camera["depth_scale_m"]
        torch.cuda.synchronize()
        started = time.perf_counter()
        for round_index in range(args.calibration_rounds):
            previous = calibrated_reference.copy()
            calibrated_reference = est.track_one(
                rgb=ref_rgb, depth=ref_depth, K=np.asarray(saved_camera["cam_K"]).reshape(3, 3),
                iteration=args.iterations)
            relative_rotation = calibrated_reference[:3, :3] @ previous[:3, :3].T
            calibration_history.append({
                "round": round_index+1, "pose": calibrated_reference.tolist(),
                "rotation_change_deg": float(np.rad2deg(np.arccos(np.clip((np.trace(relative_rotation)-1)/2, -1, 1)))),
                "translation_change_mm": float(np.linalg.norm(calibrated_reference[:3, 3]-previous[:3, 3])*1000),
            })
        torch.cuda.synchronize()
        calibration_seconds = time.perf_counter()-started
        generated_from_memory = np.linalg.inv(calibrated_reference) @ first_pose
        np.testing.assert_allclose(calibrated_reference @ generated_from_memory, first_pose, atol=1e-6)
    poses, rows = [], []
    with (output/"inference.jsonl").open("x", buffering=1) as log:
        for index, path in enumerate(images):
            rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            camera = cameras[str(int(path.stem))]
            depth = cv2.imread(str(scene/"depth"/path.name), -1).astype(np.float32)*camera["depth_scale"]/1000
            k = np.asarray(camera["cam_K"]).reshape(3, 3)
            torch.cuda.synchronize()
            started = time.perf_counter()
            if seeds is not None:
                seed = seeds[int(path.stem)]
                if not np.isfinite(seed).all():
                    raise ValueError("nonfinite seed")
                raw_seed = seed @ np.linalg.inv(generated_from_memory)
                est.pose_last = torch.as_tensor(centered_seed(raw_seed, centered), dtype=torch.float32, device="cuda")
            # Reference pose follows from the chosen coordinate system; it needs no GT.
            pose = first_pose.copy() if seeds is None and index == 0 else est.track_one(
                rgb=rgb, depth=depth, K=k, iteration=args.iterations) @ generated_from_memory
            torch.cuda.synchronize()
            if not np.isfinite(pose).all():
                raise RuntimeError(f"nonfinite refinement at {path.name}")
            poses.append(pose)
            row = {"frame_id": int(path.stem), "rgb_sha256": sha256(path),
                   "seconds": time.perf_counter()-started}
            rows.append(row)
            log.write(json.dumps(row)+"\n")
            if index % 100 == 0:
                print(json.dumps({"frames_done": index+1, "total": len(images)}), flush=True)
    poses = np.stack(poses)
    frame_ids = np.array([r["frame_id"] for r in rows])
    np.savez_compressed(output/"predictions.npz", poses=poses, frame_ids=frame_ids)
    # Evaluation annotations enter only after the entire inference result is saved.
    ref_gt = json.loads((reference_scene/"scene_gt.json").read_text())
    gt = json.loads((scene/"scene_gt.json").read_text())
    cad_pose = annotation_pose(object_annotation(ref_gt[str(int(reference_images[0].stem))], 15)[1])
    object_from_memory = np.linalg.inv(cad_pose) @ first_pose
    truth = np.stack([annotation_pose(object_annotation(gt[str(f)], 15)[1]) @ object_from_memory for f in frame_ids])
    skip = 1 if seeds is None else 0
    metrics = trajectory_metrics(poses[skip:], np.ones(len(poses)-skip, dtype=bool), truth[skip:], frame_ids[skip:])
    write_json(output/"GT_errors.json", metrics.pop("per_frame"))
    repo = Path(__file__).resolve().parents[3]
    weights = repo/"weights"
    report = {
        "protocol": "generated single-frame mesh + frozen FoundationPose neural refinement; drill development",
        "GT_or_CAD_inference_input": False, "GT_loaded_after_predictions": True,
        "previous_pose_used": seeds is None, "kalman": False, "iterations": args.iterations,
        "input_sampling": "consecutive frames" if np.all(np.diff(frame_ids) == 1) else "sparse frames",
        "max_frame_id_gap": int(np.diff(frame_ids).max()),
        "initialization": "coordinate-defined reference pose" if seeds is None else "independent predicted poses",
        "initial_predictions_sha256": sha256(args.initial_predictions) if seeds is not None else None,
        "reference_frame_excluded_from_metrics": seeds is None,
        "reference_calibration": args.calibrate_reference,
        "reference_calibration_history": calibration_history,
        "reference_calibration_seconds": calibration_seconds,
        "calibrated_first_camera_from_generated": calibrated_reference.tolist(),
        "generated_from_memory": generated_from_memory.tolist(),
        "mesh_report_sha256": sha256(mesh_directory/"report.json"), "mesh_sha256": sha256(mesh_path),
        "pose_convention": "first-camera axes, measured reference centroid origin -> current camera; metres",
        "metrics": metrics,
        "refinement_ms_median": float(np.median([r["seconds"] for r in rows[skip:]])*1000),
        "source_sha256": {str(f.relative_to(repo)): sha256(f) for f in (Path(__file__).resolve(), repo/"estimater.py")},
        "weights_sha256": {str(f.relative_to(weights)): sha256(f) for f in sorted(weights.rglob("*"))
                           if f.is_file() and f.suffix in {".pth", ".yml", ".yaml"}},
    }
    write_json(output/"report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
