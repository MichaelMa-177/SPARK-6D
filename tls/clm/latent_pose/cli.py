"""Train, cache an object memory, and independently estimate query poses."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ModelConfig
from .data import PairDataset, collate_pairs, read_packet, sha256, to_device
from .losses import coordinate_loss
from .model import LatentPoseModel, ObjectMemory


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def save_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(payload, stream)


def load_model(path, device, allow_untrained=False, solver_mode=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported model checkpoint schema")
    if checkpoint.get("training_steps", 0) < 1 and not allow_untrained:
        raise ValueError("untrained checkpoint: use --allow-untrained only for a wiring smoke test")
    config = ModelConfig.from_dict(checkpoint["config"])
    if solver_mode is not None:
        config = replace(config, solver_mode=solver_mode)
    model = LatentPoseModel(config).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint


def train(args):
    torch.manual_seed(args.seed)
    config = ModelConfig.from_dict(json.loads(Path(args.config).read_text())) if args.config else ModelConfig()
    dataset = PairDataset(args.manifest, expected_split="train")
    validation = PairDataset(args.validation, expected_split="validation") if args.validation else None
    if validation and dataset.object_ids & validation.object_ids:
        raise ValueError("training and validation object IDs overlap; use a held-out object split")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model = LatentPoseModel(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        collate_fn=collate_pairs)
    steps = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        totals = []
        for ref, query, target in loader:
            ref, query, target = [to_device(x, args.device) for x in (ref, query, target)]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(ref, query, solve=False)
            losses = coordinate_loss(prediction, target["pose"], ref["scale_m"],
                                     target["target_mask"], target["symmetry_transforms"],
                                     model.config.rigidity_weight)
            if not torch.isfinite(losses["loss"]):
                raise RuntimeError("non-finite training loss")
            losses["loss"].backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            totals.append(float(losses["loss"].detach()))
        row = {"epoch": epoch + 1, "steps": steps, "train_loss": float(np.mean(totals)),
               "last_grad_norm": float(grad)}
        if validation:
            model.eval()
            values = []
            with torch.no_grad():
                for ref, query, target in DataLoader(validation, batch_size=args.batch_size,
                                                     collate_fn=collate_pairs):
                    ref, query, target = [to_device(x, args.device) for x in (ref, query, target)]
                    result = model(ref, query)
                    values.append(float(coordinate_loss(result, target["pose"], ref["scale_m"],
                                                        target["target_mask"], target["symmetry_transforms"],
                                                        model.config.rigidity_weight)["loss"]))
            row["validation_loss"] = float(np.mean(values))
        history.append(row)
        print(json.dumps(row), flush=True)
    checkpoint = {
        "schema_version": 1, "config": config.to_dict(), "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "training_steps": steps, "seed": args.seed,
        "training_manifest_sha256": sha256(args.manifest),
        "training_object_ids": sorted(dataset.object_ids),
        "validation_manifest_sha256": sha256(args.validation) if validation else None,
        "purpose": "research framework; training steps do not establish benchmark accuracy",
    }
    save_new(output / "last.pt", checkpoint)
    write_json(output / "history.json", history)
    write_json(output / "config.json", config.to_dict())


@torch.inference_mode()
def cache(args):
    model, checkpoint = load_model(args.checkpoint, args.device, args.allow_untrained, getattr(args, "solver", None))
    reference = to_device(read_packet(args.reference, reference=True), args.device, batch=True)
    memory = model.build_memory(reference)
    tensors = {k: v.cpu() for k, v in asdict(memory).items() if k != "kv" and v is not None}
    save_new(args.output, {
        "schema_version": 1, "memory": tensors, "config": model.config.to_dict(),
        "checkpoint_sha256": sha256(args.checkpoint), "reference_sha256": sha256(args.reference),
        "training_steps": checkpoint["training_steps"],
        "coordinate_convention": "object origin=reference centroid; axes=first RGB camera; units=metres",
    })
    print(json.dumps({"output": args.output, "tokens": memory.tokens.shape[1],
                      "generated_tokens": int((memory.kinds[0] == 1).sum())}))


@torch.inference_mode()
def infer(args):
    model, checkpoint = load_model(args.checkpoint, args.device, args.allow_untrained, getattr(args, "solver", None))
    saved = torch.load(args.memory, map_location=args.device, weights_only=True)
    if saved.get("schema_version") != 1 or saved["config"] != model.config.to_dict():
        raise ValueError("memory schema/config mismatch")
    if saved["checkpoint_sha256"] != sha256(args.checkpoint):
        raise ValueError("memory was encoded by a different checkpoint; rebuild it")
    memory = ObjectMemory(**saved["memory"])
    model.cache_memory(memory)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    predictions = []
    for index, path in enumerate(args.query):
        query = to_device(read_packet(path), args.device, batch=True)  # excludes GT arrays
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = model.decode(query, memory)
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        np.savez_compressed(output / f"{index:06d}.npz", **{
            k: v.cpu().numpy() for k, v in result.items()
        })
        valid = bool(result["pose_valid"][0])
        predictions.append({
            "query": str(path), "query_sha256": sha256(path), "pose_valid": valid,
            "residual_m": float(result["residual_m"][0]) if valid else None,
            "model_ms": elapsed, "training_steps": checkpoint["training_steps"],
        })
    write_json(output / "report.json", {
        "predictions": predictions, "checkpoint_sha256": sha256(args.checkpoint),
        "GT_read": False, "previous_pose_used": False,
        "timing_scope": "ready device tensors -> output tensors; excludes loading, detection, I/O; no warmup",
        "accuracy_evaluated": False,
    })
    print(json.dumps({"queries": len(predictions), "output": str(output), "accuracy_evaluated": False}))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    t = commands.add_parser("train")
    t.add_argument("--manifest", required=True)
    t.add_argument("--validation")
    t.add_argument("--config")
    t.add_argument("--output", required=True)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cpu")
    t.set_defaults(function=train)
    for name, function in (("cache", cache), ("infer", infer)):
        sub = commands.add_parser(name)
        sub.add_argument("--checkpoint", required=True)
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--output", required=True)
        sub.add_argument("--allow-untrained", action="store_true")
        sub.add_argument("--solver", choices=["weighted", "robust", "consensus"])
        if name == "cache":
            sub.add_argument("--reference", required=True)
        else:
            sub.add_argument("--memory", required=True)
            sub.add_argument("--query", nargs="+", required=True)
        sub.set_defaults(function=function)
    return p


def main():
    args = parser().parse_args()
    torch.set_num_threads(4)
    if args.command == "train" and min(args.epochs, args.batch_size, args.lr) <= 0:
        raise ValueError("epochs, batch size and learning rate must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
