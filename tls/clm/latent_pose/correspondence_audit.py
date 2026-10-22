"""GT-assisted failure diagnosis only; never produces an inference result."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .cli import load_model, write_json
from .data import PairDataset, query_inputs, read_packet, sha256, to_device
from .geometry import camera_to_memory, consensus_rigid
from .experiment import errors


@torch.no_grad()
def run(args):
    torch.set_num_threads(4)
    dataset = PairDataset(args.manifest)
    if dataset.split not in {"development", "validation"}:
        raise ValueError("failure diagnosis must not tune on reserved test data")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model, _ = load_model(args.checkpoint, args.device)
    references, rows = {}, []
    for row in dataset.samples:
        name = row["reference"]
        if name not in references:
            ref = to_device(read_packet(dataset.path.parent/name, reference=True), args.device, batch=True)
            visible = (ref["mask"].bool() & ref["valid"].bool()).flatten()
            cloud = ref["xyz"].flatten(2).transpose(1, 2)[0, visible] - ref["origin_m"][0]
            references[name] = ref, model.build_memory(ref, cache=True), cloud
        ref, memory, cloud = references[name]
        # Explicitly diagnostic: GT is used to measure correspondence errors,
        # never fed through build_memory/decode.
        packet = to_device(read_packet(dataset.path.parent/row["query"], labels=True),
                           args.device, batch=True)
        prediction = model.decode(query_inputs(packet), memory, solve=False)
        gt_points = camera_to_memory(prediction["query_xyz_m"], packet["pose"], ref["scale_m"])
        foreground = (model._grid(packet["target_mask"]).squeeze(-1) > .5) & prediction["query_valid"]
        coordinates = prediction["coordinates"] * ref["scale_m"][:, None]
        truth = gt_points * ref["scale_m"][:, None]
        distances = torch.cdist(truth, cloud[None]).min(-1).values
        overlap = distances <= .01
        coordinate_error = (coordinates-truth).norm(dim=-1)*1000
        oracle = consensus_rigid(coordinates, prediction["query_xyz_m"], foreground.float(),
                                 model.config.min_correspondences, .01, 256)
        rot, mm, joint = errors(oracle["pose"], oracle["pose_valid"], packet["pose"],
                               packet["symmetry_transforms"])
        def median(mask):
            values = coordinate_error[mask]
            return float(values.median()) if values.numel() else None
        rows.append({
            "sample": row, "foreground_grid_points": int(foreground.sum()),
            "within_10mm_reference_surface_fraction": float(overlap[foreground].float().mean())
                if foreground.any() else None,
            "coordinate_error_median_mm": median(foreground),
            "observed_surface_coordinate_error_median_mm": median(foreground & overlap),
            "unobserved_surface_coordinate_error_median_mm": median(foreground & ~overlap),
            "GT_mask_consensus_valid": bool(oracle["pose_valid"][0]),
            "GT_mask_consensus_pass_10mm_5deg": bool(joint[0]),
            "GT_mask_consensus_rotation_deg": float(rot[0]),
            "GT_mask_consensus_translation_mm": float(mm[0]),
        })
    groups = {}
    for scene in sorted({str(r["sample"]["scene_id"]) for r in rows}):
        selected = [r for r in rows if str(r["sample"]["scene_id"]) == scene]
        groups[scene] = {"samples": len(selected)}
        for key in ("within_10mm_reference_surface_fraction", "coordinate_error_median_mm",
                    "observed_surface_coordinate_error_median_mm", "unobserved_surface_coordinate_error_median_mm",
                    "GT_mask_consensus_rotation_deg", "GT_mask_consensus_translation_mm"):
            values = [r[key] for r in selected if r[key] is not None]
            groups[scene][key+"_median"] = float(np.median(values)) if values else None
        groups[scene]["GT_mask_consensus_pass_fraction"] = float(np.mean(
            [r["GT_mask_consensus_pass_10mm_5deg"] for r in selected]))
    write_json(output/"diagnostics.json", rows)
    report = {"diagnostic_only": True, "query_GT_used_for_analysis": True,
              "GT_used_in_network": False, "checkpoint_sha256": sha256(args.checkpoint),
              "manifest_sha256": sha256(args.manifest), "by_scene": groups}
    write_json(output/"report.json", report)
    print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    run(p.parse_args())


if __name__ == "__main__":
    main()
