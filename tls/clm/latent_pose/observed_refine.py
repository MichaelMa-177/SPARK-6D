"""Projective point-to-plane refinement using only the supplied reference surface.

This diagnostic keeps the original memory gauge. It never fits generated surfaces,
uses query masks, or updates memory. Query GT is read after all predictions exist.
"""

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .bop import annotation_pose, object_annotation
from .cli import write_json
from .data import sha256
from .rgbd import measured_geometry
from .video_evaluate import trajectory_metrics


def normal_map(xyz, valid, max_depth_jump=.02):
    """Central-difference normals, oriented towards the observing camera."""
    dx = xyz[1:-1, 2:] - xyz[1:-1, :-2]
    dy = xyz[2:, 1:-1] - xyz[:-2, 1:-1]
    normals = np.cross(dx, dy)
    lengths = np.linalg.norm(normals, axis=-1)
    good = (valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2]
            & valid[2:, 1:-1] & valid[:-2, 1:-1] & (lengths > 1e-9)
            & (np.abs(dx[..., 2]) < max_depth_jump) & (np.abs(dy[..., 2]) < max_depth_jump))
    normals /= np.maximum(lengths[..., None], 1e-12)
    away = np.sum(normals * xyz[1:-1, 1:-1], axis=-1) > 0
    normals[away] *= -1
    result, support = np.zeros_like(xyz), np.zeros_like(valid)
    result[1:-1, 1:-1] = normals
    support[1:-1, 1:-1] = good
    return result, support


def point_plane_step(points, targets, normals, pivot, scale=.1):
    """Solve observable directions of a robust metric linearization at pivot."""
    residual = np.einsum("ij,ij->i", points-targets, normals)
    jacobian = np.concatenate((np.cross(points-pivot, normals)/scale, normals), axis=1)
    weights = np.minimum(1., .003/np.maximum(np.abs(residual), 1e-12))
    hessian = jacobian.T @ (weights[:, None]*jacobian)
    gradient = jacobian.T @ (weights*residual)
    eigenvalues, vectors = np.linalg.eigh(hessian)
    supported = eigenvalues > max(eigenvalues[-1]*1e-3, 1e-8)
    if supported.sum() < 3:
        return None, int(supported.sum())
    update = -(vectors[:, supported] @ ((vectors[:, supported].T @ gradient)/eigenvalues[supported]))
    update[:3] /= scale
    return update, int(supported.sum())


def projective_refine(source, source_normals, xyz, normals, valid, intrinsics, initial,
                      iterations=20, max_distance=.02):
    pose = np.asarray(initial, dtype=np.float64).copy()
    h, w = valid.shape
    history = []

    def correspondences(transform):
        points = source @ transform[:3, :3].T + transform[:3, 3]
        oriented = source_normals @ transform[:3, :3].T
        uv = points @ intrinsics.T
        uv = np.rint(uv[:, :2]/np.maximum(uv[:, 2:3], 1e-6)).astype(int)
        visible = ((points[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w)
                   & (uv[:, 1] >= 0) & (uv[:, 1] < h)
                   & (np.sum(oriented * -points, axis=1) > .1*np.linalg.norm(points, axis=1)))
        indices = np.flatnonzero(visible)
        pixels = uv[indices]
        good = valid[pixels[:, 1], pixels[:, 0]]
        indices, pixels = indices[good], pixels[good]
        targets, target_normals = xyz[pixels[:, 1], pixels[:, 0]], normals[pixels[:, 1], pixels[:, 0]]
        good = ((np.linalg.norm(points[indices]-targets, axis=1) < max_distance)
                & (points[indices, 2]-targets[:, 2] < .008)
                & (np.sum(oriented[indices]*target_normals, axis=1) > .5))
        return points[indices[good]], targets[good], target_normals[good]

    first_points, first_targets, first_normals = correspondences(pose)
    first_count = len(first_points)
    for _ in range(iterations):
        points, targets, matched_normals = correspondences(pose)
        if len(points) < 30:
            break
        update, rank = point_plane_step(points, targets, matched_normals, pose[:3, 3])
        if update is None or not np.isfinite(update).all():
            break
        factor = min(1., np.deg2rad(2)/max(np.linalg.norm(update[:3]), 1e-12),
                     .002/max(np.linalg.norm(update[3:]), 1e-12))
        update *= factor
        candidate = pose.copy()
        candidate[:3, :3] = cv2.Rodrigues(update[:3])[0] @ pose[:3, :3]
        candidate[:3, 3] += update[3:]
        total_angle = np.linalg.norm(cv2.Rodrigues(candidate[:3, :3] @ initial[:3, :3].T)[0])
        total_shift = np.linalg.norm(candidate[:3, 3]-initial[:3, 3])
        if total_angle > np.deg2rad(12) or total_shift > .015:
            break
        history.append({"correspondences": len(points), "observable_rank": rank,
                        "residual_median_mm": float(np.median(np.abs(
                            np.sum((points-targets)*matched_normals, axis=1)))*1000)})
        pose = candidate
        if np.linalg.norm(update[:3]) < 1e-5 and np.linalg.norm(update[3:]) < 1e-6:
            break
    final_points, final_targets, final_normals = correspondences(pose)
    accepted = bool(history) and len(final_points) >= max(30, first_count//2)
    # Assess improvement on the original pairs, so dropping hard points cannot
    # by itself pass this acceptance test. The scoring pairs are all measured.
    before = after = None
    if first_count:
        local = (first_points-initial[:3, 3]) @ initial[:3, :3]
        moved = local @ pose[:3, :3].T + pose[:3, 3]
        def huber_mean(residual):
            magnitude = np.abs(residual)
            return float(np.mean(np.where(magnitude <= .003, .5*magnitude**2,
                                          .003*(magnitude-.0015))))
        before = huber_mean(np.sum((first_points-first_targets)*first_normals, axis=1))
        after = huber_mean(np.sum((moved-first_targets)*first_normals, axis=1))
        accepted &= after <= before
    return (pose if accepted else np.asarray(initial).copy()), {
        "accepted": accepted, "initial_correspondences": first_count,
        "final_correspondences": len(final_points), "initial_fixed_pair_loss": before,
        "final_fixed_pair_loss": after, "iterations": history,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--reference-scene", required=True)
    p.add_argument("--initial-directory", required=True, help="generated_refine output with provenance")
    p.add_argument("--output", required=True)
    p.add_argument("--iterations", type=int, default=20)
    args = p.parse_args()
    if args.iterations < 1:
        p.error("iterations must be positive")
    cv2.setNumThreads(4)
    inputs, scene, reference_scene, initial_dir, output = map(
        Path, (args.inputs, args.scene, args.reference_scene, args.initial_directory, args.output))
    initial_report = json.loads((initial_dir/"report.json").read_text())
    if initial_report["GT_or_CAD_inference_input"] or initial_report["previous_pose_used"]:
        raise ValueError("expected independent generated-mesh initial predictions without CAD/GT")
    # Resolve the mesh report from its recorded hash before trusting the gauge.
    mesh_reports = list(initial_dir.parent.glob("generated-mesh*/report.json"))
    mesh_reports += list(initial_dir.parent.glob("generated-mesh*/steps_*/report.json"))
    matches = [path for path in mesh_reports if sha256(path) == initial_report["mesh_report_sha256"]]
    if len(matches) != 1:
        raise ValueError("need exactly one matching generated-mesh provenance report")
    mesh_report = json.loads(matches[0].read_text())
    for key, name in (("rgb_sha256", "first_frame.png"), ("depth_sha256", "first_depth.png"),
                      ("mask_sha256", "mask.png")):
        if sha256(inputs/name) != mesh_report[key]:
            raise ValueError("reference differs from predicted gauge")
    prompt = json.loads((inputs/"user_prompt.json").read_text())
    if prompt["prompt_source"] != "human_interactive_click" or prompt["rgb_sha256"] != mesh_report["rgb_sha256"]:
        raise ValueError("expected the exact saved human reference")
    reference_images = sorted((reference_scene/"rgb").glob("*.png"))
    if not reference_images or sha256(reference_images[0]) != mesh_report["rgb_sha256"]:
        raise ValueError("reference scene does not match")
    rgb = cv2.imread(str(inputs/"first_frame.png"))
    depth, mask = cv2.imread(str(inputs/"first_depth.png"), -1), cv2.imread(str(inputs/"mask.png"), 0) > 0
    camera = json.loads((inputs/"camera.json").read_text())
    reference_camera = json.loads((reference_scene/"scene_camera.json").read_text())[str(int(reference_images[0].stem))]
    if not np.allclose(camera["cam_K"], reference_camera["cam_K"]) or not np.isclose(
            camera["depth_scale_m"], reference_camera["depth_scale"]/1000):
        raise ValueError("reference calibration differs")
    _, xyz, valid = measured_geometry(rgb, depth, np.asarray(camera["cam_K"]).reshape(3, 3), camera["depth_scale_m"])
    origin = xyz[mask & valid].mean(0)
    np.testing.assert_allclose(origin, mesh_report["memory_origin_m"], atol=1e-7, rtol=0)
    normals, normal_valid = normal_map(xyz, valid)
    selection = normal_valid & (cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    source, source_normals = xyz[selection]-origin, normals[selection]
    _, unique = np.unique(np.floor(source/.003).astype(int), axis=0, return_index=True)
    source, source_normals = source[unique], source_normals[unique]
    with np.load(initial_dir/"predictions.npz", allow_pickle=False) as data:
        initial, frame_ids = data["poses"].copy(), data["frame_ids"].copy()
    initial_rows = [json.loads(line) for line in (initial_dir/"inference.jsonl").read_text().splitlines()]
    hashes = {row["frame_id"]: row["rgb_sha256"] for row in initial_rows}
    if len(frame_ids) < 3 or len(np.unique(frame_ids)) != len(frame_ids) or len(hashes) != len(frame_ids) or initial.shape != (len(frame_ids), 4, 4) or not np.isfinite(initial).all():
        raise ValueError("invalid initial trajectory")
    rotations = initial[:, :3, :3]
    if not np.allclose(initial[:, 3], [0, 0, 0, 1], atol=1e-5) or not np.allclose(
            rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-3) or not np.allclose(
            np.linalg.det(rotations), 1, atol=1e-3):
        raise ValueError("initial trajectory must contain proper rigid transforms")
    cameras = json.loads((scene/"scene_camera.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    (output/"observed_refine.py").write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(output/"memory.npz", source=source, normals=source_normals, origin_m=origin)
    poses, records = [], []
    for frame, seed in zip(frame_ids, initial):
        rgb_path = scene/"rgb"/f"{frame:06d}.png"
        if sha256(rgb_path) != hashes[int(frame)]:
            raise ValueError("query scene or frame differs")
        rgb, depth = cv2.imread(str(rgb_path)), cv2.imread(str(scene/"depth"/rgb_path.name), -1)
        camera = cameras[str(frame)]
        intrinsics = np.asarray(camera["cam_K"]).reshape(3, 3)
        started = time.perf_counter()
        _, xyz, valid = measured_geometry(rgb, depth, intrinsics, camera["depth_scale"]/1000)
        normals, normal_valid = normal_map(xyz, valid)
        pose, diagnostic = projective_refine(source, source_normals, xyz, normals,
                                             normal_valid, intrinsics, seed, args.iterations)
        poses.append(pose)
        records.append({"frame_id": int(frame), "seconds": time.perf_counter()-started, **diagnostic})
    poses = np.stack(poses)
    np.savez_compressed(output/"predictions.npz", poses=poses, frame_ids=frame_ids)
    write_json(output/"inference.json", records)
    # These labels are used only now, after the final trajectory has been saved.
    ref_gt = json.loads((reference_scene/"scene_gt.json").read_text())
    gt = json.loads((scene/"scene_gt.json").read_text())
    reference_pose = annotation_pose(object_annotation(ref_gt[str(int(reference_images[0].stem))], 15)[1])
    first_pose = np.eye(4)
    first_pose[:3, 3] = origin
    object_from_memory = np.linalg.inv(reference_pose) @ first_pose
    truth = np.stack([annotation_pose(object_annotation(gt[str(f)], 15)[1]) @ object_from_memory for f in frame_ids])
    metrics = trajectory_metrics(poses, np.ones(len(poses), bool), truth, frame_ids)
    write_json(output/"GT_errors.json", metrics.pop("per_frame"))
    write_json(output/"report.json", {
        "protocol": "frozen generated-mesh initial pose + observed reference projective ICP; drill development",
        "GT_or_CAD_inference_input": False, "query_GT_pose_annotations_loaded_after_predictions": True,
        "initial_report_metrics_used": False,
        "query_mask_used": False, "previous_pose_used": False, "online_memory_update": False,
        "source_points": len(source), "max_iterations": args.iterations,
        "refinement_accepted_fraction": float(np.mean([r["accepted"] for r in records])),
        "refinement_ms_median": float(np.median([r["seconds"] for r in records])*1000),
        "metrics": metrics, "initial_report_sha256": sha256(initial_dir/"report.json"),
        "initial_predictions_sha256": sha256(initial_dir/"predictions.npz"),
        "reference_rgb_sha256": mesh_report["rgb_sha256"], "source_sha256": sha256(__file__),
    })
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
