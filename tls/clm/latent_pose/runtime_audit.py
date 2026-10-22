"""Compare actual raw RGB-D runtime to cached-packet inference before GT scoring."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .bop import object_annotation
from .data import read_packet, sha256, to_device
from .evaluate import pose_errors
from .features import FrozenDINO
from .runtime import PoseRuntime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--dino-repo", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=3)
    args = p.parse_args()
    if args.limit < 1:
        p.error("limit must be positive")
    torch.set_num_threads(4)
    path, raw, output = Path(args.manifest), Path(args.data_root), Path(args.output)
    document = json.loads(path.read_text())
    if document["split"] == "train":
        raise ValueError("runtime audit requires held-out or development queries")
    rows = document["samples"][:args.limit]
    output.mkdir(parents=True, exist_ok=False)
    features = FrozenDINO(args.dino_repo, args.weights, args.device)
    runtime = PoseRuntime.from_checkpoint(args.checkpoint, device=args.device, feature_extractor=features,
                                          retrieve_region=True)
    registered, reports, poses = {}, [], []
    for index, row in enumerate(rows):
        ref_name = row["reference"]
        if ref_name not in registered:
            ref_scene = raw/f'{row["reference_scene"]:06d}'
            frame = row["reference_frame"]
            # Annotation metadata selects the supplied reference mask only.
            annotations = json.loads((ref_scene/"scene_gt.json").read_text())
            mask_id, _ = object_annotation(annotations[str(frame)], int(row["object_id"].split(":")[1]))
            camera = json.loads((ref_scene/"scene_camera.json").read_text())[str(frame)]
            ref_rgb_path = ref_scene/"rgb"/f"{frame:06d}.png"
            if sha256(ref_rgb_path) != row["reference_rgb_sha256"]:
                raise ValueError("reference RGB mismatch")
            ref_rgb = cv2.cvtColor(cv2.imread(str(ref_rgb_path)), cv2.COLOR_BGR2RGB)
            ref_depth = cv2.imread(str(ref_scene/"depth"/f"{frame:06d}.png"), -1)
            mask = cv2.imread(str(ref_scene/"mask_visib"/f"{frame:06d}_{mask_id:06d}.png"), 0)
            packet = read_packet(path.parent/ref_name, reference=True)
            shape = {}
            if runtime.model.config.memory_mode != "observed_only":
                shape = {k: packet[k].numpy() for k in ("shape_latent", "latent_xyz_m", "shape_geometry")
                         if k in packet}
            registered[ref_name] = runtime.register_object(
                ref_name, ref_rgb, ref_depth, mask, np.array(camera["cam_K"]).reshape(3, 3),
                camera["depth_scale"]/1000, mask_source="official BOP reference mask_visib", **shape)
        scene = raw/f'{row["scene_id"]:06d}'
        camera = json.loads((scene/"scene_camera.json").read_text())[str(row["frame_id"])]
        rgb_path = scene/"rgb"/f'{row["frame_id"]:06d}.png'
        if sha256(rgb_path) != row["rgb_sha256"]:
            raise ValueError("query RGB mismatch")
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(scene/"depth"/rgb_path.name), -1)
        # New stream per query explicitly tests reuse of the same fixed memory.
        stream = runtime.start_video(f"independent-video-{index}", ref_name)
        result = stream.estimate(rgb, depth, np.array(camera["cam_K"]).reshape(3, 3),
                                 camera["depth_scale"]/1000, 0.)
        with torch.no_grad():
            cached_memory = runtime.model.build_memory(
                to_device(read_packet(path.parent/ref_name, reference=True), args.device, batch=True), cache=True)
            cached = runtime.model.decode(
                to_device(read_packet(path.parent/row["query"]), args.device, batch=True), cached_memory)
        raw_pose = result.pop("pose")
        poses.append(raw_pose if raw_pose is not None else np.eye(4))
        cached_pose = cached["pose"][0].cpu().numpy()
        same_box = list(result["query_box_xyxy"]) == row["box_xyxy"]
        same_valid = result["pose_valid"] == bool(cached["pose_valid"][0])
        max_error = float(np.abs(poses[-1]-cached_pose).max())
        reports.append(dict(index=index, benchmark_object_id=row["object_id"], **result,
                            region_matches_export=same_box, validity_matches_cached=same_valid,
                            raw_vs_cached_pose_max_abs_error=max_error))
        if not same_box or not same_valid or (result["pose_valid"] and max_error > 5e-4):
            raise RuntimeError("raw runtime differs from the frozen packet pipeline")
    np.savez_compressed(output/"predictions.npz", poses=np.stack(poses),
                        valid=np.array([r["pose_valid"] for r in reports]))
    # Query GT arrays are opened only after all actual raw-frame predictions exist.
    for row, pose, report in zip(rows, poses, reports):
        label = read_packet(path.parent/row["query"], labels=True)
        report["GT_errors"] = pose_errors(pose, label["pose"].numpy(), label["symmetry_transforms"].numpy())
        report["GT_errors"]["pass_10mm_5deg"] &= report["pose_valid"]
    summary = dict(checkpoint_sha256=sha256(args.checkpoint), manifest_sha256=sha256(path),
                   query_GT_loaded_after_predictions=True,
                   reference_pose_used=False, reference_mask_source="official BOP mask_visib",
                   single_frame_shape_prior="cached generated latent" if runtime.model.config.memory_mode != "observed_only" else None,
                   timing_scope="raw query RGB-D to CPU pose, including two DINO passes and retrieval; no warmup",
                   records=reports)
    (output/"report.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(reports), flush=True)


if __name__ == "__main__":
    main()
