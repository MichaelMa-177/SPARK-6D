"""Adapted One2Any baseline using the same predicted ROIs and frozen mask prior."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from .bop import object_annotation
from .cli import write_json
from .data import read_packet, sha256, to_device
from .evaluate import pose_errors, summarize_errors
from .geometry import consensus_rigid
from .one2any_baseline import load_one2any
from .rgbd import measured_geometry, reference_crop_box


def crop_resize(value, box, size=192):
    x0, y0, x1, y1 = box
    return cv2.resize(value[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_NEAREST)


def author_reference_crop(value, mask, size=192):
    """Match the upstream 1.5x square, integer center, zero-padded crop."""
    y, x = np.where(mask)
    width, height = int(x.max()-x.min()+1), int(y.max()-y.min()+1)
    cx, cy = int(x.min()+width/2), int(y.min()+height/2)
    side = int(min(1.5*max(width, height), 480))
    half = side//2
    x0, x1 = max(0, cx-half), min(value.shape[1], cx+half)
    y0, y1 = max(0, cy-half), min(value.shape[0], cy+half)
    canvas = np.zeros((side, side, *value.shape[2:]), dtype=value.dtype)
    left, top = half-(cx-x0), half-(cy-y0)
    canvas[top:top+y1-y0, left:left+x1-x0] = value[y0:y1, x0:x1]
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_NEAREST)


def reference_condition(rgb, depth, mask, k, depth_scale, author_crop=False):
    rgb, xyz, valid = measured_geometry(rgb, depth, k, depth_scale)
    mask = (mask > 0) & valid
    depth_mean = xyz[mask, 2].mean()
    mask &= np.abs(xyz[:, :, 2]-depth_mean) <= .1  # authors' reference depth trimming
    points = xyz[mask]
    if len(points) < 12:
        raise ValueError("insufficient reference depth for One2Any normalization")
    center = (points.max(0)+points.min(0))/2
    scale = float(np.ptp(points, axis=0).max())*1.1
    nocs = np.zeros_like(xyz)
    nocs[mask] = np.clip((points-center)/scale+.5, 0, 1)
    box = reference_crop_box(mask)
    crop = (lambda value: author_reference_crop(value, mask)) if author_crop else (lambda value: crop_resize(value, box))
    condition = np.concatenate((crop(rgb).astype(np.float32)/255,
                                crop(nocs), crop(mask.astype(np.float32))[..., None]), -1)
    return condition.transpose(2, 0, 1), center, scale


def foreground_prior(reference, query, side=64):
    fg = F.interpolate(reference["mask"]*reference["valid"], (32, 32), mode="nearest").flatten() > .5
    bg = (F.interpolate(reference["valid"], (32, 32), mode="nearest").flatten() > .5) & ~fg
    features = F.normalize(reference["foundation_features"][0], dim=-1)
    q = F.normalize(query["foundation_features"][0], dim=-1)
    positive = (q @ features[fg].T).max(-1).values
    negative = (q @ features[bg].T).max(-1).values if bg.any() else torch.zeros_like(positive)
    logits = 20*(positive-negative)+5*(positive-.6)
    return F.interpolate(logits.reshape(1, 1, 32, 32), (side, side), mode="nearest").flatten(1)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--ldm-source", required=True)
    p.add_argument("--taming-source", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--reference-inputs", help="saved human reference for the fixed drill protocol")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--author-reference-crop", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    path, raw, output = Path(args.manifest), Path(args.data_root), Path(args.output)
    document = json.loads(path.read_text())
    if document["split"] not in {"validation", "development"}:
        raise ValueError("adapted baseline development requires a non-test held-out manifest")
    samples = document["samples"][:args.limit] if args.limit else document["samples"]
    output.mkdir(parents=True, exist_ok=False)
    model, provenance = load_one2any(args.source, args.ldm_source, args.taming_source,
                                     args.checkpoint, args.device)
    references, poses, valid, details, reference_checks = {}, [], [], [], {}
    for index, row in enumerate(samples):
        name = row["reference"]
        if name not in references:
            reference = to_device(read_packet(path.parent/name, reference=True), args.device, batch=True)
            if args.reference_inputs:
                inputs = Path(args.reference_inputs)
                rgb_path, depth_path, mask_path = inputs/"first_frame.png", inputs/"first_depth.png", inputs/"mask.png"
                camera = json.loads((inputs/"camera.json").read_text())
                scale = camera["depth_scale_m"]
                prompt = json.loads((inputs/"user_prompt.json").read_text())
                if prompt["prompt_source"] != "human_interactive_click" or sha256(rgb_path) != prompt["rgb_sha256"]:
                    raise ValueError("fixed reference differs from the saved human prompt")
            else:
                scene = raw/f'{row["reference_scene"]:06d}'
                frame = row["reference_frame"]
                # Reference metadata selects its supplied mask only; no pose enters the condition.
                annotation = json.loads((scene/"scene_gt.json").read_text())[str(frame)]
                mask_id, _ = object_annotation(annotation, int(row["object_id"].split(":")[1]))
                rgb_path, depth_path = scene/"rgb"/f"{frame:06d}.png", scene/"depth"/f"{frame:06d}.png"
                mask_path = scene/"mask_visib"/f"{frame:06d}_{mask_id:06d}.png"
                camera = json.loads((scene/"scene_camera.json").read_text())[str(frame)]
                scale = camera["depth_scale"]/1000
                if sha256(rgb_path) != row["reference_rgb_sha256"]:
                    raise ValueError("reference image differs from the fixed protocol")
            rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
            depth, mask = cv2.imread(str(depth_path), -1), cv2.imread(str(mask_path), 0)
            condition, center, scale = reference_condition(rgb, depth, mask,
                                                            np.array(camera["cam_K"]).reshape(3, 3), scale,
                                                            args.author_reference_crop)
            condition = torch.from_numpy(condition)[None].to(args.device)
            latent = model.text_adapter(condition)
            references[name] = reference, condition, latent, torch.tensor(center, device=args.device), scale
            self_nocs = model(condition[:, :3], condition[:, 6].bool(), ref_cond=condition)["pred_nocs"]
            source_self = ((self_nocs[:, :, ::3, ::3]-.5)*scale).flatten(2).transpose(1, 2)
            source_self += torch.tensor(center, device=args.device)-reference["origin_m"][:, None]
            target_self = ((condition[:, 3:6, ::3, ::3]-.5)*scale).flatten(2).transpose(1, 2)
            target_self += torch.tensor(center, device=args.device)
            foreground_self = condition[:, 6, ::3, ::3].flatten(1)
            solved_self = consensus_rigid(source_self, target_self, foreground_self, 6, .01, 256)
            known_self = np.eye(4)
            known_self[:3, 3] = reference["origin_m"][0].cpu().numpy()
            reference_checks[name] = {
                "pose_valid": bool(solved_self["pose_valid"][0]),
                "reference_only_no_GT_object_pose": True,
                **pose_errors(solved_self["pose"][0].cpu().numpy(), known_self, np.eye(4)[None]),
            }
        reference, condition, latent, center, scale = references[name]
        scene = raw/f'{row["scene_id"]:06d}'
        rgb_path = scene/"rgb"/f'{row["frame_id"]:06d}.png'
        if sha256(rgb_path) != row["rgb_sha256"]:
            raise ValueError("query image differs from fixed ROI protocol")
        camera = json.loads((scene/"scene_camera.json").read_text())[str(row["frame_id"])]
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(scene/"depth"/rgb_path.name), -1)
        rgb, xyz, qvalid = measured_geometry(rgb, depth, np.array(camera["cam_K"]).reshape(3, 3),
                                            camera["depth_scale"]/1000)
        box = row["box_xyxy"]
        image = torch.from_numpy(crop_resize(rgb, box).transpose(2, 0, 1).copy())[None].to(args.device).float()/255
        feature = model.encoder(image*2-1, latent)
        nocs = model.last_layer_nocs(model.decoder([feature])).clamp(0, 1)
        if index == 0:
            official = model(image, torch.ones_like(image[:, 0], dtype=torch.bool), ref_cond=condition)["pred_nocs"]
            torch.testing.assert_close(nocs, official, atol=1e-6, rtol=1e-5)
            provenance["cached_reference_forward_matches_upstream"] = True
        source = nocs[:, :, ::3, ::3].flatten(2).transpose(1, 2)
        source = (source-.5)*scale + center - reference["origin_m"][:, None]
        target = torch.from_numpy(crop_resize(xyz, box)[::3, ::3].reshape(1, -1, 3).copy()).to(args.device)
        eligible = torch.from_numpy(crop_resize(qvalid.astype(np.uint8), box)[::3, ::3].reshape(1, -1).copy()).to(args.device).bool()
        query = to_device(read_packet(path.parent/row["query"]), args.device, batch=True)
        logits = foreground_prior(reference, query)
        weights = logits.sigmoid() * (logits >= 0) * eligible
        solved = consensus_rigid(source, target, weights, 6, .01, 256)
        poses.append(solved["pose"][0].cpu().numpy())
        valid.append(bool(solved["pose_valid"][0]))
        details.append((source.cpu(), target.cpu(), eligible.cpu()))
        if index % 25 == 0:
            print(json.dumps({"done": index+1, "total": len(samples)}), flush=True)
    np.savez_compressed(output/"predictions.npz", poses=np.stack(poses), valid=np.array(valid))
    # Labels are opened after primary predictions; the oracle-mask result below
    # is explicitly diagnostic and never replaces the primary result.
    errors, oracle_errors = [], []
    for index, (row, pose, ok, detail) in enumerate(zip(samples, poses, valid, details)):
        label = read_packet(path.parent/row["query"], labels=True)
        error = pose_errors(pose, label["pose"].numpy(), label["symmetry_transforms"].numpy())
        error["pass_10mm_5deg"] &= ok
        errors.append({"index": index, "pose_valid": ok, "scene_id": row["scene_id"],
                       "object_id": row["object_id"], **error})
        source, target, eligible = [v.to(args.device) for v in detail]
        mask = F.interpolate(label["target_mask"][None].to(args.device), (64, 64),
                             mode="nearest").flatten(1) > .5
        solved = consensus_rigid(source, target, (mask & eligible).float(), 6, .01, 256)
        error = pose_errors(solved["pose"][0].cpu().numpy(), label["pose"].numpy(),
                            label["symmetry_transforms"].numpy())
        ok = bool(solved["pose_valid"][0])
        error["pass_10mm_5deg"] &= ok
        oracle_errors.append({"pose_valid": ok, **error})
    write_json(output/"errors.json", errors)
    report = {
        **summarize_errors(errors), "provenance": provenance,
        "protocol": "adapted One2Any with shared predicted ROI, frozen DINO membership and consensus; not official published scores",
        "reference_latent_tokens": list(references[next(iter(references))][2].shape),
        "query_GT_loaded_after_predictions": True, "reference_GT_pose_used_in_network": False,
        "GT_mask_diagnostic_only": summarize_errors(oracle_errors),
        "manifest_sha256": sha256(path), "previous_pose_used": False,
        "author_reference_crop": args.author_reference_crop, "reference_self_checks": reference_checks,
        "by_scene": {str(s): summarize_errors([r for r in errors if r["scene_id"] == s])
                     for s in sorted({r["scene_id"] for r in errors})},
    }
    write_json(output/"report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}), flush=True)


if __name__ == "__main__":
    main()
