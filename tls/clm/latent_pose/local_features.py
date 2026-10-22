"""Single-reference SIFT + metric 3D consensus baseline for overlap diagnosis."""

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from .cli import write_json
from .data import read_packet, sha256
from .evaluate import pose_errors, summarize_errors
from .geometry import consensus_rigid
from .rgbd import measured_geometry


def sample_keypoints(keypoints, xyz, valid):
    pixels = np.rint([p.pt for p in keypoints]).astype(int).reshape(-1, 2)
    pixels[:, 0] = pixels[:, 0].clip(0, xyz.shape[1]-1)
    pixels[:, 1] = pixels[:, 1].clip(0, xyz.shape[0]-1)
    return xyz[pixels[:, 1], pixels[:, 0]], valid[pixels[:, 1], pixels[:, 0]]


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--ratio", type=float, default=.75)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if not 0 < args.ratio < 1:
        p.error("ratio must lie between zero and one")
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    inputs, path, raw, output = map(Path, (args.inputs, args.manifest, args.data_root, args.output))
    document = json.loads(path.read_text())
    rows = document["samples"]
    if document["split"] != "development" or len({r["reference"] for r in rows}) != 1:
        raise ValueError("this diagnosis expects one fixed reference and a development manifest")
    rgb = cv2.cvtColor(cv2.imread(str(inputs/"first_frame.png")), cv2.COLOR_BGR2RGB)
    depth, mask = cv2.imread(str(inputs/"first_depth.png"), -1), cv2.imread(str(inputs/"mask.png"), 0)
    prompt = json.loads((inputs/"user_prompt.json").read_text())
    if prompt["prompt_source"] != "human_interactive_click" or sha256(inputs/"first_frame.png") != prompt["rgb_sha256"]:
        raise ValueError("reference must match the saved human prompt")
    camera = json.loads((inputs/"camera.json").read_text())
    _, xyz, depth_valid = measured_geometry(rgb, depth, np.asarray(camera["cam_K"]).reshape(3, 3),
                                            camera["depth_scale_m"])
    origin = xyz[(mask > 0) & depth_valid].mean(0)
    with np.load(path.parent/rows[0]["reference"], allow_pickle=False) as reference:
        np.testing.assert_allclose(origin, reference["origin_m"], atol=1e-7, rtol=0)
    detector = cv2.SIFT_create(nfeatures=4000, contrastThreshold=.01)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    ref_kp, ref_desc = detector.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), mask)
    if ref_desc is None:
        raise ValueError("reference contains no SIFT features")
    ref_xyz, ref_valid = sample_keypoints(ref_kp, xyz, depth_valid & (mask > 0))
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"memory.npz", descriptors=ref_desc, points_m=ref_xyz-origin,
                        valid=ref_valid, origin_m=origin)
    cameras, poses, valid, records = {}, [], [], []
    for index, row in enumerate(rows):
        scene = raw/f'{row["scene_id"]:06d}'
        if row["scene_id"] not in cameras:
            cameras[row["scene_id"]] = json.loads((scene/"scene_camera.json").read_text())
        camera = cameras[row["scene_id"]][str(row["frame_id"])]
        rgb_path = scene/"rgb"/f'{row["frame_id"]:06d}.png'
        if sha256(rgb_path) != row["rgb_sha256"]:
            raise ValueError("query source mismatch")
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(scene/"depth"/rgb_path.name), -1)
        started = time.perf_counter()
        _, xyz, depth_valid = measured_geometry(rgb, depth, np.asarray(camera["cam_K"]).reshape(3, 3),
                                                camera["depth_scale"]/1000)
        region = np.zeros(depth.shape, np.uint8)
        x0, y0, x1, y1 = row["box_xyxy"]
        region[y0:y1, x0:x1] = 255  # existing predicted ROI; no query segmentation input
        kp, desc = detector.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), region)
        source, target, confidence = [], [], []
        matched = 0
        if desc is not None and len(desc) >= 2:
            points, point_valid = sample_keypoints(kp, xyz, depth_valid)
            reverse = {m.queryIdx: m.trainIdx for m in matcher.match(desc, ref_desc)}
            for candidates in matcher.knnMatch(ref_desc, desc, k=2):
                if len(candidates) != 2:
                    continue
                a, b = candidates
                if a.distance < args.ratio*b.distance and reverse.get(a.trainIdx) == a.queryIdx:
                    matched += 1
                    if ref_valid[a.queryIdx] and point_valid[a.trainIdx]:
                        source.append(ref_xyz[a.queryIdx]-origin)
                        target.append(points[a.trainIdx])
                        confidence.append(1-a.distance/max(b.distance, 1e-6))
        pose, ok = np.eye(4), False
        if len(source) >= 6:
            tensors = [torch.tensor(np.asarray(a), device=args.device, dtype=torch.float32)[None]
                       for a in (source, target, confidence)]
            solved = consensus_rigid(*tensors, min_points=6, radius_m=.01, trials=256)
            pose, ok = solved["pose"][0].cpu().numpy(), bool(solved["pose_valid"][0])
        poses.append(pose)
        valid.append(ok)
        records.append({"frame_id": row["frame_id"], "scene_id": row["scene_id"],
                        "keypoints": len(kp), "mutual_ratio_matches": matched,
                        "matches_with_valid_depth": len(source), "seconds": time.perf_counter()-started})
    np.savez_compressed(output/"predictions.npz", poses=np.stack(poses), valid=np.array(valid))
    write_json(output/"inference.json", records)
    errors = []
    for row, pose, ok in zip(rows, poses, valid):
        label = read_packet(path.parent/row["query"], labels=True)
        error = pose_errors(pose, label["pose"].numpy(), label["symmetry_transforms"].numpy())
        error["pass_10mm_5deg"] &= ok
        errors.append({**error, "scene_id": row["scene_id"], "pose_valid": ok})
    write_json(output/"errors.json", errors)
    report = {**summarize_errors(errors), "protocol": "SIFT reference mask + predicted query ROI + metric 3D consensus",
              "query_GT_loaded_after_predictions": True, "previous_pose_used": False,
              "reference_keypoints": len(ref_kp), "ratio_threshold": args.ratio,
              "manifest_sha256": sha256(path), "reference_rgb_sha256": sha256(inputs/"first_frame.png"),
              "code_sha256": sha256(__file__), "opencv_version": cv2.__version__,
              "by_scene": {str(s): summarize_errors([e for e in errors if e["scene_id"] == s])
                           for s in sorted({e["scene_id"] for e in errors})}}
    write_json(output/"report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
