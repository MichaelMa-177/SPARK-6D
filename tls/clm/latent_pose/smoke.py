"""Check real-reference inference wiring. Random weights imply NO accuracy claim."""

import argparse
import json
from pathlib import Path
import time

import torch

from .config import ModelConfig
from .data import read_packet, to_device
from .model import LatentPoseModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(4)
    ref = to_device(read_packet(args.reference, reference=True), args.device, batch=True)
    query = to_device(read_packet(args.query), args.device, batch=True)
    model = LatentPoseModel().to(args.device).eval()
    with torch.inference_mode():
        memory = model.build_memory(ref, cache=True)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        begin = time.perf_counter()
        result = model.decode(query, memory)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - begin) * 1000
        required = ("coordinates", "membership_logits", "log_variance", "weights")
        if not all(torch.isfinite(result[k]).all() for k in required):
            raise RuntimeError("non-finite model output")
        report = {
            "status": "wiring_smoke_passed", "accuracy_evaluated": False,
            "weights": "random initialization", "device": args.device,
            "parameters": sum(x.numel() for x in model.parameters()),
            "input_latent": list(ref["shape_latent"].shape),
            "memory_tokens": list(memory.tokens.shape), "query_tokens": result["coordinates"].shape[1],
            "measured_tokens": int((memory.kinds[0] == 0).sum()),
            "generated_tokens": int((memory.kinds[0] == 1).sum()),
            "all_decoder_outputs_finite": True, "pose_solver_valid": bool(result["pose_valid"][0]),
            "single_cold_forward_ms": elapsed, "previous_pose_used": False, "GT_read": False,
        }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.save({"schema_version": 1, "config": ModelConfig().to_dict(), "model": model.state_dict(),
                "training_steps": 0}, output / "untrained.pt")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
