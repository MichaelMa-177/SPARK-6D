"""Raw consecutive RGB-D evaluation with absolute and motion-error diagnostics."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .bop import annotation_pose, object_annotation
from .cli import write_json
from .data import sha256
from .evaluate import pose_errors
from .features import FrozenDINO
from .runtime import PoseRuntime


def rotation_degrees(rotation):
    return np.rad2deg(np.arccos(np.clip((np.trace(rotation, axis1=-2, axis2=-1)-1)/2, -1, 1)))


def trajectory_metrics(predictions, valid, truth, frame_ids):
    if len(predictions) < 2 or len(predictions) != len(truth) or len(valid) != len(frame_ids):
        raise ValueError("trajectory diagnostics require at least two paired frames")
    errors = [pose_errors(p, t, np.eye(4)[None]) for p, t in zip(predictions, truth)]
    consecutive = (np.diff(frame_ids) == 1) & valid[1:] & valid[:-1]
    pred_delta = predictions[1:, :3, :3] @ predictions[:-1, :3, :3].transpose(0, 2, 1)
    gt_delta = truth[1:, :3, :3] @ truth[:-1, :3, :3].transpose(0, 2, 1)
    rotation_motion = rotation_degrees(pred_delta @ gt_delta.transpose(0, 2, 1))
    translation_motion = np.linalg.norm(np.diff(predictions[:, :3, 3], axis=0) -
                                       np.diff(truth[:, :3, 3], axis=0), axis=-1)*1000
    gt_small_motion = (rotation_degrees(gt_delta) <= .1) & (
        np.linalg.norm(np.diff(truth[:, :3, 3], axis=0), axis=-1)*1000 <= .25)
    def quantiles(values):
        return dict(zip(("median", "p90", "p95"), map(float, np.percentile(values, [50, 90, 95])))) if len(values) else None
    return {
        "frames": len(valid), "valid_fraction": float(valid.mean()),
        "pass_10mm_5deg_all_frames": float(np.mean([v and e["pass_10mm_5deg"] for v, e in zip(valid, errors)])),
        "absolute_rotation_deg_valid": quantiles(np.array([e["rotation_deg"] for e in errors])[valid]),
        "absolute_translation_mm_valid": quantiles(np.array([e["translation_mm_at_best_rotation"] for e in errors])[valid]),
        "consecutive_valid_pairs": int(consecutive.sum()),
        "rotation_increment_error_deg": quantiles(rotation_motion[consecutive]),
        "origin_displacement_error_mm": quantiles(translation_motion[consecutive]),
        "small_GT_motion_pair_thresholds": {"rotation_deg": .1, "translation_mm": .25},
        "small_GT_motion_valid_pairs": int((consecutive & gt_small_motion).sum()),
        "small_GT_motion_rotation_increment_error_deg": quantiles(rotation_motion[consecutive & gt_small_motion]),
        "small_GT_motion_origin_displacement_error_mm": quantiles(translation_motion[consecutive & gt_small_motion]),
        "per_frame": errors,
    }


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--dino-repo", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--teacher-predictions")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--solver", choices=["weighted", "robust", "consensus"], default="consensus")
    p.add_argument("--fps", type=float, default=30, help="timestamp assumption; errors reported per frame")
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args()
    if args.fps <= 0 or args.max_frames < 0 or args.max_frames in {1, 2}:
        p.error("invalid FPS or frame limit")
    torch.set_num_threads(4)
    scene, inputs, output = Path(args.scene), Path(args.inputs), Path(args.output)
    images = sorted((scene/"rgb").glob("*.png"))
    if args.max_frames:
        images = images[:args.max_frames]
    if not images:
        raise ValueError("no frames")
    prompt = json.loads((inputs/"user_prompt.json").read_text())
    if prompt["prompt_source"] != "human_interactive_click" or sha256(images[0]) != prompt["rgb_sha256"]:
        raise ValueError("reference must be the exact saved human-selected first frame")
    ref_rgb = cv2.cvtColor(cv2.imread(str(inputs/"first_frame.png")), cv2.COLOR_BGR2RGB)
    ref_depth = cv2.imread(str(inputs/"first_depth.png"), -1)
    mask = cv2.imread(str(inputs/"mask.png"), 0)
    cameras = json.loads((scene/"scene_camera.json").read_text())
    first_camera = cameras[str(int(images[0].stem))]
    saved_camera = json.loads((inputs/"camera.json").read_text())
    if sha256(inputs/"first_frame.png") != prompt["rgb_sha256"] or not np.array_equal(
            ref_depth, cv2.imread(str(scene/"depth"/images[0].name), -1)):
        raise ValueError("saved reference RGB/depth mismatch")
    if not np.allclose(first_camera["cam_K"], saved_camera["cam_K"]) or not np.isclose(
            first_camera["depth_scale"]/1000, saved_camera["depth_scale_m"]):
        raise ValueError("saved camera mismatch")
    output.mkdir(parents=True, exist_ok=False)
    extractor = FrozenDINO(args.dino_repo, args.weights, args.device)
    runtime = PoseRuntime.from_checkpoint(args.checkpoint, args.device,
                                          feature_extractor=extractor, retrieve_region=True,
                                          solver=args.solver)
    if runtime.model.config.memory_mode != "observed_only":
        raise ValueError("this audit currently supports the observed-memory baseline")
    registered = runtime.register_object(
        "drill", ref_rgb, ref_depth, mask, np.array(first_camera["cam_K"]).reshape(3, 3),
        first_camera["depth_scale"]/1000, mask_source="saved human-interactive SAM mask")
    stream = runtime.start_video(scene.name, "drill")
    poses, rows, valid = [], [], []
    with (output/"inference.jsonl").open("x", buffering=1) as log:
        for index, path in enumerate(images):
            rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(str(scene/"depth"/path.name), -1)
            camera = cameras[str(int(path.stem))]
            result = stream.estimate(rgb, depth, np.array(camera["cam_K"]).reshape(3, 3),
                                     camera["depth_scale"]/1000, index/args.fps)
            pose = result.pop("pose")
            poses.append(pose if pose is not None else np.eye(4))
            valid.append(result["pose_valid"])
            row = {**result, "frame_id": int(path.stem), "rgb_sha256": sha256(path)}
            rows.append(row)
            log.write(json.dumps(row, allow_nan=False)+"\n")
            if index % 100 == 0:
                print(json.dumps({"frames_done": index+1, "total": len(images)}), flush=True)
    poses, valid = np.stack(poses), np.array(valid)
    frame_ids = np.array([r["frame_id"] for r in rows])
    np.savez_compressed(output/"predictions.npz", poses=poses, valid=valid, frame_ids=frame_ids)
    # All query predictions are persisted before opening the GT pose file.
    gt = json.loads((scene/"scene_gt.json").read_text())
    first_pose = annotation_pose(object_annotation(gt[str(frame_ids[0])], 15)[1])
    first_from_memory = np.eye(4)
    first_from_memory[:3, 3] = registered.memory.origin_m[0].cpu().numpy()
    object_from_memory = np.linalg.inv(first_pose) @ first_from_memory
    truth = np.stack([annotation_pose(object_annotation(gt[str(f)], 15)[1]) @ object_from_memory for f in frame_ids])
    # Exclude the reference frame from accuracy and motion summaries.
    metrics = trajectory_metrics(poses[1:], valid[1:], truth[1:], frame_ids[1:])
    write_json(output/"GT_errors.json", metrics.pop("per_frame"))
    report = {
        "protocol": "dense scene59 drill development; one human reference; independent localization",
        "GT_loaded_after_predictions": True, "previous_pose_used": False, "smoothing": False,
        "checkpoint_sha256": sha256(args.checkpoint), "mask_sha256": sha256(inputs/"mask.png"),
        "prompt_sha256": sha256(inputs/"user_prompt.json"), "fps_assumption": args.fps,
        "reference_frame_excluded_from_metrics": True, "metrics": metrics,
        "query_latency_ms_median": float(np.median([r["raw_frame_to_output_ms"] for r in rows[1:]])),
        "timing_scope": "raw RGB-D to CPU pose including retrieval and two DINO passes; excludes disk reads",
    }
    if args.teacher_predictions:
        teacher = np.load(args.teacher_predictions, allow_pickle=False)
        by_frame = {int(f): p for f, p in zip(teacher["frame_ids"], teacher["poses"])}
        teacher_poses = np.stack([by_frame[int(f)] @ object_from_memory for f in frame_ids])
        teacher_metrics = trajectory_metrics(teacher_poses[1:], np.ones(len(frame_ids)-1, dtype=bool),
                                             truth[1:], frame_ids[1:])
        teacher_metrics.pop("per_frame")
        report["teacher_metrics_same_memory_origin"] = teacher_metrics
        report["teacher_predictions_sha256"] = sha256(args.teacher_predictions)
    write_json(output/"report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
