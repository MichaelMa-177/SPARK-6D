"""Step-based training with separate coordinate, segmentation and pose diagnostics."""

import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .cli import load_model, save_new, write_json
from .config import ModelConfig
from .data import PairDataset, collate_pairs, query_inputs, read_packet, sha256, to_device
from .geometry import weighted_rigid
from .losses import coordinate_loss
from .model import LatentPoseModel
from .augmentation import random_memory_basis


class IndexedPairStore:
    """Deduplicate reference/query tensors on CPU; transfer only each batch."""

    def __init__(self, dataset, device):
        self.device = device
        self.references, self.queries, self.targets, self.rows = {}, {}, [], []
        for row in dataset.samples:
            ref_path = (dataset.path.parent/row["reference"]).resolve()
            if ref_path not in self.references:
                self.references[ref_path] = read_packet(ref_path, reference=True)
            packet = read_packet(dataset.path.parent/row["query"], labels=True)
            query = query_inputs(packet)
            key = (row.get("rgb_sha256", row["query"]), tuple(row.get("box_xyxy", [])))
            if key in self.queries:
                cached = self.queries[key]
                if cached.keys() != query.keys() or any(not torch.equal(cached[k], query[k]) for k in cached):
                    raise ValueError("query cache key maps to different inference tensors")
            else:
                self.queries[key] = query
            self.targets.append({k: packet[k] for k in ("pose", "target_mask", "symmetry_transforms")})
            self.rows.append((ref_path, key))

    def __len__(self):
        return len(self.rows)

    def batch(self, indices):
        if isinstance(indices, slice):
            indices = range(*indices.indices(len(self)))
        elif isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().tolist()
        batch = [(self.references[self.rows[i][0]], self.queries[self.rows[i][1]], self.targets[i])
                 for i in indices]
        return [to_device(part, self.device) for part in collate_pairs(batch)]


def preload(dataset, device):
    references, rows = {}, []
    for row in dataset.samples:
        ref_path = (dataset.path.parent / row["reference"]).resolve()
        if ref_path not in references:
            references[ref_path] = read_packet(ref_path, reference=True)
        packet = read_packet(dataset.path.parent / row["query"], labels=True)
        rows.append((references[ref_path], query_inputs(packet),
                     {k: packet[k] for k in ("pose", "target_mask", "symmetry_transforms")}))
    return [to_device(x, device) for x in collate_pairs(rows)]


def subset(data, indices):
    if isinstance(data, IndexedPairStore):
        return data.batch(indices)
    return [{k: v[indices] for k, v in part.items()} for part in data]


def errors(pose, valid, truth, symmetries):
    candidates = truth[:, None] @ torch.linalg.inv(symmetries)
    rotations = pose[:, None, :3, :3] @ candidates[..., :3, :3].transpose(-1, -2)
    cosine = ((rotations.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    degrees = cosine.acos().rad2deg()
    mm = (pose[:, None, :3, 3] - candidates[..., :3, 3]).norm(dim=-1) * 1000
    best = degrees.argmin(-1)
    index = torch.arange(len(pose), device=pose.device)
    return degrees[index, best], mm[index, best], ((degrees <= 5) & (mm <= 10)).any(-1) & valid


@torch.no_grad()
def diagnose(model, data, batch_size):
    model.eval()
    gathered = {k: [] for k in ("rotation", "translation", "joint", "valid", "oracle_rotation",
                                "oracle_translation", "foreground_weight_fraction", "mask_iou")}
    losses = []
    generated_fractions = []
    count = len(data) if isinstance(data, IndexedPairStore) else len(data[0]["rgb"])
    for start in range(0, count, batch_size):
        ref, query, target = subset(data, slice(start, start + batch_size))
        prediction = model(ref, query, solve=True)
        losses.append(float(coordinate_loss(prediction, target["pose"], ref["scale_m"],
                                            target["target_mask"], target["symmetry_transforms"],
                                            model.config.rigidity_weight)["loss"]))
        rot, mm, joint = errors(prediction["pose"], prediction["pose_valid"], target["pose"],
                                target["symmetry_transforms"])
        side = model.config.image_size // 4
        mask = F.interpolate(target["target_mask"].float(), (side, side),
                             mode="nearest").flatten(1) > .5
        mask = mask & prediction["query_valid"]
        if "generated_coordinate_fraction" in prediction and mask.any():
            generated_fractions.append(prediction["generated_coordinate_fraction"][mask].detach().cpu())
        predicted_mask = (prediction["membership_logits"] > 0) & prediction["query_valid"]
        # Diagnostic only: GT mask never modifies the reported model pose.
        oracle = weighted_rigid(prediction["coordinates"] * ref["scale_m"][:, None],
                                prediction["query_xyz_m"], mask.float())
        orot, omm, _ = errors(oracle["pose"], oracle["pose_valid"], target["pose"],
                              target["symmetry_transforms"])
        orot = torch.where(oracle["pose_valid"], orot, torch.full_like(orot, float("nan")))
        omm = torch.where(oracle["pose_valid"], omm, torch.full_like(omm, float("nan")))
        fraction = (prediction["weights"] * mask).sum(-1) / prediction["weights"].sum(-1).clamp_min(1e-12)
        iou = (mask & predicted_mask).sum(-1) / (mask | predicted_mask).sum(-1).clamp_min(1)
        for key, value in zip(gathered, (rot, mm, joint, prediction["pose_valid"], orot, omm, fraction, iou)):
            gathered[key].append(value.detach().cpu())
    values = {k: torch.cat(v) for k, v in gathered.items()}
    valid = values["valid"]
    def median(t):
        finite = t[torch.isfinite(t)]
        return float(finite.median()) if len(finite) else None
    return {
        "loss": sum(losses) / len(losses), "frames": count,
        "valid_fraction": float(valid.float().mean()),
        "pass_10mm_5deg_all_frames": float(values["joint"].float().mean()),
        "rotation_median_deg": median(values["rotation"][valid]),
        "translation_median_mm": median(values["translation"][valid]),
        "oracle_mask_rotation_median_deg_diagnostic": median(values["oracle_rotation"]),
        "oracle_mask_translation_median_mm_diagnostic": median(values["oracle_translation"]),
        "foreground_weight_fraction_mean": float(values["foreground_weight_fraction"].mean()),
        "mask_iou_mean": float(values["mask_iou"].mean()),
        "generated_coordinate_fraction_foreground_mean": float(torch.cat(generated_fractions).mean())
            if generated_fractions else None,
    }


def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    train_set = PairDataset(args.train, expected_split="train")
    val_set = PairDataset(args.validation) if args.validation else None
    if val_set and val_set.split not in {"validation", "development"}:
        raise ValueError("checkpoint selection cannot use a test manifest")
    shared = train_set.object_ids & val_set.object_ids if val_set else set()
    if shared and not (args.allow_seen_object_development and val_set.split == "development"):
        raise ValueError("seen-object validation requires explicit development mode")
    if val_set:
        train_queries = {(train_set.path.parent / r["query"]).resolve() for r in train_set.samples}
        val_queries = {(val_set.path.parent / r["query"]).resolve() for r in val_set.samples}
        if train_queries & val_queries:
            raise ValueError("training and validation query files overlap")
        train_hashes = {r["rgb_sha256"] for r in train_set.samples if "rgb_sha256" in r}
        val_hashes = {r["rgb_sha256"] for r in val_set.samples if "rgb_sha256" in r}
        if train_hashes & val_hashes:
            raise ValueError("training and validation query images overlap")
    config = ModelConfig.from_dict(json.loads(Path(args.config).read_text())) if args.config else ModelConfig()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    source_root = Path(__file__).parent
    for source in sorted(source_root.rglob("*.py")):
        relative = source.relative_to(source_root)
        saved_source = output/"source"/relative
        saved_source.parent.mkdir(parents=True, exist_ok=True)
        saved_source.write_bytes(source.read_bytes())
        source_hashes[str(relative)] = sha256(saved_source)
    model = LatentPoseModel(config).to(args.device)
    inherited_objects = set()
    if args.initialize:
        initialized, initialized_metadata = load_model(args.initialize, args.device)
        if initialized.config != config:
            raise ValueError("initialization config mismatch")
        model.load_state_dict(initialized.state_dict())
        inherited_objects = set(initialized_metadata.get("training_object_ids", []))
        shared |= inherited_objects & val_set.object_ids if val_set else set()
        if shared and not (args.allow_seen_object_development and val_set.split == "development"):
            raise ValueError("initialization checkpoint has seen validation objects")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    train_data = IndexedPairStore(train_set, args.device) if args.cpu_unique_cache else preload(train_set, args.device)
    val_data = preload(val_set, args.device) if val_set else None
    diagnostic_indices = None
    diagnostic_data = train_data
    if args.train_diagnostic_samples and args.train_diagnostic_samples < len(train_set):
        diagnostic_indices = torch.linspace(0, len(train_set)-1, args.train_diagnostic_samples).round().long()
        diagnostic_data = subset(train_data, diagnostic_indices)
    provenance = {
        "schema_version": 1, "config": config.to_dict(), "seed": args.seed,
        "training_object_ids": sorted(train_set.object_ids | inherited_objects),
        "training_manifest_sha256": sha256(args.train),
        "validation_manifest_sha256": sha256(args.validation) if val_set else None,
        "validation_object_overlap": sorted(shared),
        "protocol": "seen-object development" if shared else "object-disjoint validation",
        "initialization_sha256": sha256(args.initialize) if args.initialize else None,
        "foundation_features": train_set.provenance.get("foundation_features", {
            "weights_sha256": train_set.provenance.get("dino_weights_sha256")}),
        "shape_prior": train_set.provenance.get("shape_prior"),
        "source_sha256": source_hashes,
        "sampling_protocol": "seed reset after initialization/preload; shared batch and basis draws",
        "training_diagnostic_indices": diagnostic_indices.tolist() if diagnostic_indices is not None else "all",
        "args": vars(args),
    }
    write_json(output / "run.json", provenance)
    n = len(train_set)
    best = -1.
    # Optional modules consume different amounts of RNG during construction.
    # Keep batch order and basis augmentation matched across ablations.
    torch.manual_seed(args.seed)
    start_time = time.perf_counter()
    with (output / "history.jsonl").open("x", buffering=1) as history:
        for step in range(args.steps + 1):
            if step > 0:
                model.train()
                indices = torch.randperm(n, device=args.device)[:args.batch_size]
                ref, query, target = subset(train_data, indices)
                if args.augment_memory_basis:
                    ref, target = random_memory_basis(ref, target)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(ref, query)
                loss = coordinate_loss(prediction, target["pose"], ref["scale_m"],
                                       target["target_mask"], target["symmetry_transforms"],
                                       model.config.rigidity_weight)["loss"]
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
            if step % args.eval_every != 0 and step != args.steps:
                continue
            row = {"step": step, "seconds": time.perf_counter() - start_time,
                   "train": diagnose(model, diagnostic_data, args.batch_size),
                   "validation": diagnose(model, val_data, args.batch_size) if val_data else None}
            history.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)
            if step == 0:
                continue
            checkpoint = {
                **provenance, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "training_steps": step, "metrics": row,
            }
            # A new checkpoint file per evaluation step preserves every audited state.
            filename = f"step_{step:06d}.pt"
            save_new(output / filename, checkpoint)
            score = (row["validation"] or row["train"])["pass_10mm_5deg_all_frames"]
            if score > best:
                best = score
                (output / "best.json").write_text(json.dumps({"checkpoint": filename, "score": score}) + "\n")
    write_json(output / "complete.json", {"steps": args.steps, "best_score": best,
                                         "seconds": time.perf_counter() - start_time})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--validation")
    p.add_argument("--allow-seen-object-development", action="store_true")
    p.add_argument("--config")
    p.add_argument("--initialize")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--augment-memory-basis", action="store_true")
    p.add_argument("--cpu-unique-cache", action="store_true",
                   help="deduplicate large training datasets in host RAM; only transfer selected batches")
    p.add_argument("--train-diagnostic-samples", type=int, default=0,
                   help="fixed evenly spaced training diagnostics; 0 evaluates all training pairs")
    args = p.parse_args()
    if min(args.steps, args.eval_every, args.batch_size, args.lr) <= 0:
        p.error("steps, intervals, batch size and learning rate must be positive")
    if args.train_diagnostic_samples < 0:
        p.error("training diagnostic count cannot be negative")
    run(args)


if __name__ == "__main__":
    main()
