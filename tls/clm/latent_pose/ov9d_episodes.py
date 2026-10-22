"""Independent synthetic training episodes, with saturated depth rejected."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .bop import annotation_pose, cad_from_memory
from .bop_episodes import evenly, symmetry_matrices
from .config import ModelConfig
from .data import sha256
from .features import FrozenDINO
from .rgbd import prepare_query, prepare_reference, reference_crop_box
from .shape_prior import ShapePriorEncoder


def sanitize_depth(depth, mask):
    """Do not treat uint16 saturation as a measured surface."""
    valid = np.isfinite(depth) & (depth > 0)
    if np.issubdtype(depth.dtype, np.integer):
        valid &= depth < np.iinfo(depth.dtype).max
    clean = np.where(valid, depth, 0)
    fraction = float((valid & mask).sum()/max(int(mask.sum()), 1))
    return clean, fraction


def jittered_training_box(mask, seed):
    """Training label supplies a crop, with fixed center/size perturbations."""
    y, x = np.where(mask)
    rng = np.random.default_rng(seed)
    side = max(float(np.ptp(x)), float(np.ptp(y)), 16.) * rng.uniform(1.4, 1.9)
    cx, cy = (x.min()+x.max())/2, (y.min()+y.max())/2
    cx, cy = np.array([cx, cy]) + rng.uniform(-.1, .1, 2)*side
    return (max(0, int(cx-side/2)), max(0, int(cy-side/2)),
            min(mask.shape[1], int(cx+side/2)+1), min(mask.shape[0], int(cy+side/2)+1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--models-info", required=True)
    p.add_argument("--dino-repo", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--shape-source", required=True)
    p.add_argument("--shape-weights", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--queries", type=int, default=24)
    p.add_argument("--references", type=int, default=2)
    p.add_argument("--max-objects", type=int, default=0, help="smoke limit; no final training manifest")
    args = p.parse_args()
    if min(args.queries, args.references) < 1 or args.max_objects < 0:
        p.error("invalid episode counts")
    torch.set_num_threads(4)
    raw, output = Path(args.data_root), Path(args.output)
    download_policy = json.loads((raw/"policy.json").read_text())
    if not args.max_objects and not (raw/"complete.json").exists():
        raise ValueError("full frozen training download must complete before final export")
    output.mkdir(parents=True, exist_ok=False)
    names = [n.removesuffix(".tgz") for n in download_policy["archives"]]
    if args.max_objects:
        names = [n for n in names if (raw/"receipts"/(n+".tgz.json")).exists()][:args.max_objects]
    models = json.loads(Path(args.models_info).read_text())
    policy = {
        "schema_version": 1, "purpose": "independent OV9D synthetic training; no YCB-V images",
        "download_policy_sha256": sha256(raw/"policy.json"),
        "source_sha256": sha256(__file__), "models_info_sha256": sha256(args.models_info),
        "reference_policy": "evenly spaced valid training views; one reference per episode",
        "query_policy": "evenly spaced other views; training GT crop with deterministic jitter",
        "reference_mask_source": "author-provided mask_visib; no SAM or human click generated",
        "invalid_depth": "zero, nonfinite and integer saturation; reject frames with <95% valid foreground",
        "args": vars(args), "smoke_only": bool(args.max_objects),
    }
    (output/"policy.json").write_text(json.dumps(policy, indent=2)+"\n")
    config = ModelConfig(memory_mode="observed_only")
    dino = FrozenDINO(args.dino_repo, args.weights, args.device)
    shape = ShapePriorEncoder(args.shape_source, args.dino_repo, args.shape_weights, args.device)
    samples, audit = [], []
    for object_index, name in enumerate(names):
        scene = raw/"objects"/name/name
        cameras = json.loads((scene/"scene_camera.json").read_text())
        annotations = json.loads((scene/"scene_gt.json").read_text())
        observations = {}
        rejected = []
        for frame in sorted(annotations, key=int):
            if len(annotations[frame]) != 1:
                raise ValueError("expected single-object training renders")
            image = scene/"rgb"/f"{int(frame):06d}.png"
            rgb = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(str(scene/"depth"/image.name), -1)
            mask = cv2.imread(str(scene/"mask_visib"/f"{int(frame):06d}_000000.png"), 0) > 0
            depth, fraction = sanitize_depth(depth, mask)
            if fraction < .95 or ((depth > 0) & mask).sum() < 100:
                rejected.append({"frame": int(frame), "valid_foreground_fraction": fraction})
                continue
            camera = cameras[frame]
            observations[int(frame)] = (rgb, depth, mask, np.array(camera["cam_K"]).reshape(3, 3),
                                         camera["depth_scale"]/1000, image)
        object_audit = {"name": name, "available_frames": len(annotations),
                        "depth_rejected_frames": rejected, "accepted_frames": len(observations)}
        if len(observations) < args.references + 2:
            audit.append({**object_audit, "used": False, "reason": "insufficient nonsaturated depth views"})
            continue
        frames = sorted(observations)
        refs = evenly(frames, args.references)
        queries = evenly([f for f in frames if f not in refs], args.queries)
        object_id = int(annotations[str(frames[0])][0]["obj_id"])
        if str(object_id) not in models:
            raise ValueError("missing author symmetry/model metadata")
        query_packets = {}
        query_boxes = {}
        for frame in queries:
            rgb, depth, mask, k, depth_scale, _ = observations[frame]
            seed = int.from_bytes(hashlib.sha256(f"{name}/{frame}".encode()).digest()[:4], "little")
            box = jittered_training_box(mask, seed)
            packet = prepare_query(rgb, depth, k, depth_scale, config, box)
            x0, y0, x1, y1 = box
            packet["foundation_features"] = dino(rgb[y0:y1, x0:x1])
            packet["target_mask"] = cv2.resize(mask[y0:y1, x0:x1].astype(np.uint8),
                                               (128, 128), interpolation=cv2.INTER_NEAREST)[None]
            query_packets[frame], query_boxes[frame] = packet, box
        for ref_id, frame in enumerate(refs):
            rgb, depth, mask, k, depth_scale, image = observations[frame]
            reference = prepare_reference(rgb, depth, mask, k, depth_scale, config)
            box = reference_crop_box(mask)
            x0, y0, x1, y1 = box
            reference["foundation_features"] = dino(rgb[y0:y1, x0:x1])
            prior, prior_report = shape(rgb, depth, mask, k, depth_scale)
            reference.update(prior)
            fixed = cad_from_memory(annotation_pose(annotations[str(frame)][0]), reference["origin_m"])
            symmetries = symmetry_matrices(models[str(object_id)], fixed)
            ref_name = f"references/{name}_{ref_id}.npz"
            (output/ref_name).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output/ref_name, **reference)
            (output/ref_name).with_suffix(".json").write_text(json.dumps(prior_report, indent=2)+"\n")
            for qframe, packet in query_packets.items():
                qname = f"queries/{name}_{ref_id}_{qframe:06d}.npz"
                (output/qname).parent.mkdir(parents=True, exist_ok=True)
                labelled = {**packet, "pose": (annotation_pose(annotations[str(qframe)][0])@fixed).astype(np.float32),
                            "symmetry_transforms": symmetries}
                np.savez_compressed(output/qname, **labelled)
                samples.append({
                    "reference": ref_name, "query": qname, "object_id": f"ov9d:{object_id:06d}:{name}",
                    "scene_id": name, "frame_id": qframe, "reference_frame": frame,
                    "rgb_sha256": sha256(observations[qframe][-1]), "reference_rgb_sha256": sha256(image),
                    "box_xyxy": list(query_boxes[qframe]), "cad_from_memory": fixed.tolist(),
                })
        audit.append({**object_audit, "used": True, "references": refs, "queries": queries})
        print(json.dumps({"object": object_index+1, "total": len(names), "name": name,
                          "pairs": len(samples), "depth_rejected": len(rejected)}), flush=True)
    policy["foundation_features"] = {"backbone": "frozen DINOv2 ViT-L/14 registers",
                                     "weights_sha256": sha256(args.weights)}
    policy["shape_prior"] = shape.provenance
    filename = "smoke_pairs.json" if args.max_objects else "train.json"
    (output/filename).write_text(json.dumps(dict(schema_version=1, split="train",
                                                provenance=policy, samples=samples), indent=2)+"\n")
    (output/"data_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    (output/"complete.json").write_text(json.dumps({
        "pairs": len(samples), "used_objects": sum(r["used"] for r in audit),
        "rejected_objects": sum(not r["used"] for r in audit), "smoke_only": bool(args.max_objects),
        "manifest_sha256": sha256(output/filename)}, indent=2)+"\n")


if __name__ == "__main__":
    main()
