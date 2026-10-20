"""Development baseline: frozen RGB features, mutual matches, metric RANSAC."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import read_packet, sha256, to_device
from .evaluate import pose_errors
from .geometry import transform_points, weighted_rigid


def grid(value):
    return value[..., ::4, ::4].flatten(2).transpose(1, 2)


@torch.no_grad()
def descriptor_pose(reference, query, generator, trials=512, radius=.01):
    device = reference["rgb"].device
    ri = (grid(reference["mask"] * reference["valid"])[0, :, 0] > .5).nonzero().flatten()
    qi = (grid(query["valid"])[0, :, 0] > .5).nonzero().flatten()
    failed = dict(pose=torch.eye(4, device=device), pose_valid=False, matches=0, inliers=0)
    if min(len(ri), len(qi)) < 6:
        return failed
    a = F.normalize(reference["foundation_features"][0, ri], dim=-1)
    b = F.normalize(query["foundation_features"][0, qi], dim=-1)
    similarity = a @ b.T
    best_b = similarity.argmax(-1)
    best_a = similarity.argmax(0)
    mutual = best_a[best_b] == torch.arange(len(a), device=device)
    mutual &= similarity[torch.arange(len(a), device=device), best_b] >= .5
    rid, qid = ri[mutual], qi[best_b[mutual]]
    count = len(rid)
    failed["matches"] = count
    if count < 6:
        return failed
    source = grid(reference["xyz"])[0, rid] - reference["origin_m"][0]
    target = grid(query["xyz"])[0, qid]
    # Random triples are reproducible; duplicates/collinearity fail the rigid solver.
    sample = torch.randint(count, (trials, 3), generator=generator, device=device)
    hypotheses = weighted_rigid(source[sample], target[sample], torch.ones(trials, 3, device=device),
                                min_points=3)
    residual = (transform_points(source[None], hypotheses["pose"]) - target[None]).norm(dim=-1)
    inliers = (residual < radius) & hypotheses["pose_valid"][:, None]
    support = inliers.sum(-1)
    # Prefer lower residual when support is tied.
    score = support.float() - (residual.clamp_max(radius) / radius).mean(-1) * .001
    chosen = int(score.argmax())
    mask = inliers[chosen]
    solved = weighted_rigid(source[None], target[None], mask[None].float())
    for _ in range(2):
        residual = (transform_points(source[None], solved["pose"]) - target[None]).norm(dim=-1)[0]
        mask = residual < radius
        solved = weighted_rigid(source[None], target[None], mask[None].float())
    return dict(pose=solved["pose"][0], pose_valid=bool(solved["pose_valid"][0]),
                matches=count, inliers=int(mask.sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(4)
    path, output = Path(args.manifest), Path(args.output)
    document = json.loads(path.read_text())
    if document["split"] not in {"development", "validation"}:
        raise ValueError("descriptor probe cannot tune on a test manifest")
    output.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    memories, predictions, records = {}, [], []
    with torch.no_grad():
        for row in document["samples"]:
            refpath = path.parent/row["reference"]
            if refpath not in memories:
                memories[refpath] = to_device(read_packet(refpath, reference=True), args.device, batch=True)
            query = to_device(read_packet(path.parent/row["query"]), args.device, batch=True)
            result = descriptor_pose(memories[refpath], query, generator)
            predictions.append(result.pop("pose").cpu().numpy())
            records.append({**row, **result})
    np.savez_compressed(output/"predictions.npz", poses=np.stack(predictions))
    # Prediction file is persisted before loading any labels.
    for record, pose in zip(records, predictions):
        label = read_packet(path.parent/record["query"], labels=True)
        metrics = pose_errors(pose, label["pose"].numpy(), label["symmetry_transforms"].numpy())
        metrics["pass_10mm_5deg"] &= record["pose_valid"]
        record.update(metrics)
    grouped = {}
    for field in ("scene_id", "object_id"):
        by_group = {}
        for group in sorted({r[field] for r in records}):
            selected = [r for r in records if r[field] == group]
            valid = [r for r in selected if r["pose_valid"]]
            by_group[str(group)] = {
                "frames": len(selected), "valid_fraction": len(valid)/len(selected),
                "pass_10mm_5deg": float(np.mean([r["pass_10mm_5deg"] for r in selected])),
                "rotation_median_deg_valid": float(np.median([r["rotation_deg"] for r in valid])) if valid else None,
                "translation_median_mm_valid": float(np.median([r["translation_mm_at_best_rotation"] for r in valid])) if valid else None,
                "matches_median": float(np.median([r["matches"] for r in selected])),
            }
        grouped[field] = by_group
    report = dict(protocol="frozen-DINO visible-surface diagnostic; no trained pose network",
                  GT_loaded_after_predictions=True, query_mask_used=False, previous_pose_used=False,
                  parameters=dict(trials=512, radius_m=.01, minimum_cosine=.5, seed=args.seed),
                  manifest_sha256=sha256(path), source_sha256=sha256(__file__),
                  grouped=grouped, records=records)
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(grouped), flush=True)


if __name__ == "__main__":
    main()
