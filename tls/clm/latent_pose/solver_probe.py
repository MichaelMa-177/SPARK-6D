"""Freeze network predictions, compare predeclared solvers on development frames."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .cli import load_model, write_json
from .data import read_packet, sha256, to_device
from .evaluate import pose_errors
from .geometry import robust_rigid, weighted_rigid


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.set_num_threads(4)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    if manifest["split"] != "development":
        raise ValueError("solver selection is restricted to a development manifest")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model, _ = load_model(args.checkpoint, args.device)
    variants = {"weighted": (None, False), "threshold_050": (.5, False),
                "robust_025": (.25, True), "robust_050": (.5, True), "robust_075": (.75, True)}
    predictions = {name: [] for name in variants}
    memory, last_reference = None, None
    for row in manifest["samples"]:
        path = manifest_path.parent / row["reference"]
        if path != last_reference:
            memory = model.build_memory(to_device(read_packet(path, reference=True), args.device, batch=True),
                                        cache=True)
            last_reference = path
        query = to_device(read_packet(manifest_path.parent / row["query"]), args.device, batch=True)
        result = model.decode(query, memory, solve=False)
        for name, (threshold, robust) in variants.items():
            w = result["weights"]
            if threshold is not None:
                w = w * (result["membership_logits"].sigmoid() >= threshold)
            solver = robust_rigid if robust else weighted_rigid
            solved = solver(result["coordinates"] * memory.scale_m[:, None], result["query_xyz_m"], w)
            predictions[name].append((solved["pose"][0].cpu().numpy(), bool(solved["pose_valid"][0])))
    # Persist the complete prediction set before reading any scoring labels.
    for name, values in predictions.items():
        np.savez_compressed(output / f"{name}.npz", pose=np.stack([v[0] for v in values]),
                            valid=np.array([v[1] for v in values]))
    labels = [read_packet(manifest_path.parent / r["query"], labels=True) for r in manifest["samples"]]
    reports = {}
    for name, values in predictions.items():
        metrics = [pose_errors(pose, label["pose"].numpy(), label["symmetry_transforms"].numpy())
                   if valid else None for (pose, valid), label in zip(values, labels)]
        valid = [r for r in metrics if r is not None]
        reports[name] = {
            "frames": len(metrics), "valid_fraction": len(valid)/len(metrics),
            "pass_10mm_5deg_all_frames": sum(r["pass_10mm_5deg"] for r in valid)/len(metrics),
            "rotation_median_deg": float(np.median([r["rotation_deg"] for r in valid])) if valid else None,
            "translation_median_mm": float(np.median([r["translation_mm_at_best_rotation"] for r in valid])) if valid else None,
        }
    write_json(output/"report.json", {"checkpoint_sha256": sha256(args.checkpoint),
                                     "manifest_sha256": sha256(args.manifest), "split": "development",
                                     "GT_loaded_after_predictions": True, "previous_pose_used": False,
                                     "robust_radius_m": .01, "robust_iterations": 4, "variants": reports})
    print(json.dumps(reports), flush=True)


if __name__ == "__main__":
    main()
