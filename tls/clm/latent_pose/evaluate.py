"""Independent pair inference, then label loading and gauge-consistent scoring."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .cli import load_model, write_json
from .data import read_packet, sha256, to_device


def pose_errors(predicted, truth, symmetries):
    # The loss supervises S*x_object; the matching pose is T_true*inverse(S).
    candidates = truth[None] @ np.linalg.inv(symmetries)
    relative = predicted[None, :3, :3] @ candidates[:, :3, :3].transpose(0, 2, 1)
    rotation = np.rad2deg(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    translation = np.linalg.norm(predicted[None, :3, 3] - candidates[:, :3, 3], axis=-1) * 1000
    index = int(np.argmin(rotation))
    return {
        "rotation_deg": float(rotation[index]),
        "translation_mm_at_best_rotation": float(translation[index]),
        "pass_10mm_5deg": bool(((rotation <= 5) & (translation <= 10)).any()),
    }


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    path = Path(args.manifest)
    document = json.loads(path.read_text())
    if document.get("schema_version") != 1 or document["split"] not in {"test", "validation", "development"}:
        raise ValueError("expected a non-training version-1 pair manifest")
    samples = document["samples"]
    if not samples:
        raise ValueError("empty evaluation manifest")
    model, checkpoint = load_model(args.checkpoint, args.device)
    shared = set(checkpoint.get("training_object_ids", [])) & {str(s["object_id"]) for s in samples}
    if shared and document["split"] != "development":
        raise ValueError("held-out evaluation contains a training object")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    poses, valid = [], []
    memory, last_reference = None, None
    # This entire pass reads only reference data and unlabelled query arrays.
    for row in samples:
        reference_path = (path.parent / row["reference"]).resolve()
        if reference_path != last_reference:
            ref = to_device(read_packet(reference_path, reference=True), args.device, batch=True)
            memory = model.build_memory(ref, cache=True)
            last_reference = reference_path
        query = to_device(read_packet(path.parent / row["query"]), args.device, batch=True)
        result = model.decode(query, memory)
        poses.append(result["pose"][0].cpu().numpy())
        valid.append(bool(result["pose_valid"][0]))
    np.savez_compressed(output / "predictions.npz", poses=np.stack(poses), valid=np.array(valid))
    # All predictions are now persisted; only now request label arrays.
    rows = []
    for index, (row, pose, ok) in enumerate(zip(samples, poses, valid)):
        label = read_packet(path.parent / row["query"], labels=True)
        errors = pose_errors(pose, label["pose"].numpy(), label["symmetry_transforms"].numpy()) if ok else {
            "rotation_deg": None, "translation_mm_at_best_rotation": None, "pass_10mm_5deg": False,
        }
        rows.append({"index": index, "object_id": row["object_id"], "pose_valid": ok, **errors})
    write_json(output / "errors.json", rows)
    report = {
        "samples": len(rows), "valid_pose_fraction": float(np.mean(valid)),
        "pass_10mm_5deg_all_queries": float(np.mean([r["pass_10mm_5deg"] for r in rows])),
        "rotation_median_deg_valid_only": float(np.median([r["rotation_deg"] for r in rows if r["pose_valid"]])) if any(valid) else None,
        "split": document["split"], "training_object_overlap": sorted(shared),
        "GT_loaded_after_predictions": True, "previous_pose_used": False,
        "manifest_sha256": sha256(path), "checkpoint_sha256": sha256(args.checkpoint),
        "protocol": "fixed first-reference gauge; discrete symmetry-aware diagnostics, NOT official BOP AR",
    }
    write_json(output / "report.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
