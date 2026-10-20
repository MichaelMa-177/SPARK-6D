"""Reference-conditioned RGB-D region proposals; no query labels enter retrieval."""

import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from .bop import object_annotation
from .config import ModelConfig
from .data import read_packet, sha256, to_device
from .features import FrozenDINO
from .rgbd import prepare_query


@torch.no_grad()
def propose_region(reference, query, image_shape, intrinsics):
    """Cluster mutual appearance matches in metric 3D; size ROI from the reference."""
    h, w = image_shape
    intrinsics = np.asarray(intrinsics)
    valid = query["valid"][0, 0, ::4, ::4].flatten() > .5
    reference_mask = reference["mask"][0, 0, ::4, ::4].flatten() > .5
    mask = reference_mask & (reference["valid"][0, 0, ::4, ::4].flatten() > .5)
    foreground = F.normalize(reference["foundation_features"][0, mask], dim=-1)
    background = F.normalize(reference["foundation_features"][0, ~reference_mask], dim=-1)
    descriptors = F.normalize(query["foundation_features"][0], dim=-1)
    similarity = foreground @ descriptors.T
    if not valid.any() or len(foreground) < 3:
        return (0, 0, w, h), {"fallback": "no_valid_geometry", "matches": 0}
    best_query = similarity.masked_fill(~valid[None], -1e4).argmax(-1)
    best_ref = similarity.argmax(0)
    positive = similarity.max(0).values
    negative = (background @ descriptors.T).max(0).values if len(background) else torch.zeros_like(positive)
    score = (20 * (positive-negative) + 5 * (positive-.6)).sigmoid()
    mutual = best_ref[best_query] == torch.arange(len(foreground), device=score.device)
    ids = best_query[mutual]
    xyz = query["xyz"][0, :, ::4, ::4].flatten(1).T
    fallback = None
    if len(ids) < 3:
        ids = score.masked_fill(~valid, -1).topk(min(32, int(valid.sum()))).indices
        fallback = "few_mutual_matches"
    points = xyz[ids]
    scale = float(reference["scale_m"][0, 0])
    weights = score[ids].clamp_min(.01)
    distance = torch.cdist(points[None], points[None])[0]
    support = (distance < .65 * scale).float() @ weights
    center_id = int(support.argmax())
    close = distance[center_id] < .65 * scale
    selected = ids[close]
    selected_weights = weights[close]
    point = (xyz[selected] * selected_weights[:, None]).sum(0) / selected_weights.sum()
    side = int(valid.numel() ** .5)
    # This grid matches full-frame XYZ nearest-neighbour samples.
    pixels = torch.stack(((selected % side) * (w/side), (selected // side) * (h/side)), -1)
    center = (pixels * selected_weights[:, None]).sum(0) / selected_weights.sum()
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    z = max(float(point[2]), .05)
    extent = float((pixels.max(0).values - pixels.min(0).values).max()) if len(pixels) > 1 else 0
    span = max(32., 1.6 * max(fx, fy) * scale / z, 1.6 * extent)
    cx, cy = center.tolist()
    box = (max(0, int(np.floor(cx-span/2))), max(0, int(np.floor(cy-span/2))),
           min(w, int(np.ceil(cx+span/2))), min(h, int(np.ceil(cy+span/2))))
    return box, {"fallback": fallback, "matches": int(mutual.sum()), "cluster_support": int(close.sum()),
                 "foreground_score_mean": float(selected_weights.mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet-root", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--dino-repo", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--manifest-names", nargs="+", default=["train", "validation"])
    args = p.parse_args()
    torch.set_num_threads(4)
    root, raw, output = Path(args.packet_root), Path(args.data_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if any(Path(name).name != name or name in {".", ".."} for name in args.manifest_names):
        p.error("manifest names must be plain filename stems")
    documents = {split: json.loads((root/f"{split}.json").read_text()) for split in args.manifest_names}
    references, cameras, proposals = {}, {}, {}
    for document in documents.values():
        for row in document["samples"]:
            scene_id = row["scene_id"]
            scene = raw/f"{scene_id:06d}"
            if scene_id not in cameras:
                cameras[scene_id] = json.loads((scene/"scene_camera.json").read_text())
            camera = cameras[scene_id][str(row["frame_id"])]
            k = np.array(camera["cam_K"]).reshape(3, 3)
            if row["reference"] not in references:
                references[row["reference"]] = to_device(
                    read_packet(root/row["reference"], reference=True), args.device, batch=True)
            query = to_device(read_packet(root/row["query"]), args.device, batch=True)
            shape = cv2.imread(str(scene/"rgb"/f'{row["frame_id"]:06d}.png')).shape[:2]
            box, record = propose_region(references[row["reference"]], query, shape, k)
            proposals[row["query"]] = dict(box_xyxy=list(box), **record)
    # Persist EVERY proposal before opening GT annotations or target-mask arrays.
    (output/"proposals.json").write_text(json.dumps({
        "query_GT_used": False, "source_sha256": sha256(__file__), "records": proposals,
        "input_manifest_sha256": {s: sha256(root/f"{s}.json") for s in documents},
    }, indent=2)+"\n")
    print(json.dumps({"proposals_frozen": len(proposals), "GT_read": False}), flush=True)
    extractor = FrozenDINO(args.dino_repo, args.weights, args.device)
    config = ModelConfig(memory_mode="observed_only", foundation_feature_dim=1024)
    feature_cache, annotations, metrics = {}, {}, {}
    for name in references:
        (output/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name, output/name)
    for split, document in documents.items():
        quality = []
        for i, row in enumerate(document["samples"]):
            scene_id, frame_id = row["scene_id"], row["frame_id"]
            scene = raw/f"{scene_id:06d}"
            if scene_id not in annotations:
                annotations[scene_id] = json.loads((scene/"scene_gt.json").read_text())
            rgb_path = scene/"rgb"/f"{frame_id:06d}.png"
            if sha256(rgb_path) != row["rgb_sha256"]:
                raise ValueError("query RGB identity mismatch")
            rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(str(scene/"depth"/rgb_path.name), -1)
            camera = cameras[scene_id][str(frame_id)]
            k = np.array(camera["cam_K"]).reshape(3, 3)
            box = proposals[row["query"]]["box_xyxy"]
            packet = prepare_query(rgb, depth, k, camera["depth_scale"]/1000, config, box)
            x0, y0, x1, y1 = box
            cache_key = (scene_id, frame_id, tuple(box))
            if cache_key not in feature_cache:
                feature_cache[cache_key] = extractor(rgb[y0:y1, x0:x1])
            packet["foundation_features"] = feature_cache[cache_key]
            index, _ = object_annotation(annotations[scene_id][str(frame_id)], int(row["object_id"].split(":")[1]))
            mask = cv2.imread(str(scene/"mask_visib"/f"{frame_id:06d}_{index:06d}.png"), 0) > 0
            label = read_packet(root/row["query"], labels=True)
            packet.update(pose=label["pose"].numpy(), symmetry_transforms=label["symmetry_transforms"].numpy(),
                          target_mask=cv2.resize(mask[y0:y1, x0:x1].astype(np.uint8), (128, 128),
                                                 interpolation=cv2.INTER_NEAREST)[None])
            target = output/row["query"]
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **packet)
            row["original_box_xyxy"], row["box_xyxy"] = row["box_xyxy"], box
            row["query_sha256"] = sha256(target)
            quality.append(dict(object_id=row["object_id"], query=row["query"],
                                mask_coverage=float(mask[y0:y1, x0:x1].sum()/max(1, mask.sum())),
                                foreground_grid_points=int((packet["target_mask"][:, ::4, ::4] *
                                                           packet["valid"][:, ::4, ::4]).sum())))
            if i % 100 == 0:
                print(json.dumps({"split": split, "done": i+1, "total": len(document["samples"])}), flush=True)
        document["provenance"] = {
            **document["provenance"], "query_roi": "frozen-DINO mutual-match metric cluster",
            "query_GT_used_for_ROI": False, "proposals_sha256": sha256(output/"proposals.json"),
            "failed_proposals_retained": True, "no_sample_filtering": True,
        }
        (output/f"{split}.json").write_text(json.dumps(document, indent=2)+"\n")
        metrics[split] = {
            "samples": len(quality), "mask_coverage_median": float(np.median([r["mask_coverage"] for r in quality])),
            "at_least_90pct_mask_coverage": float(np.mean([r["mask_coverage"] >= .9 for r in quality])),
            "fewer_than_six_valid_target_points": sum(r["foreground_grid_points"] < 6 for r in quality),
            "per_sample": quality,
        }
    (output/"roi_quality.json").write_text(json.dumps(metrics, indent=2)+"\n")
    print(json.dumps({k: {a: b for a, b in v.items() if a != "per_sample"} for k, v in metrics.items()}), flush=True)


if __name__ == "__main__":
    main()
