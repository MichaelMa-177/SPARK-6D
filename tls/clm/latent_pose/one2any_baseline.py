"""Strict loader for the authors' external One2Any checkpoint and source tree."""

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import types

import torch

from .data import sha256


def load_one2any(source, ldm_source, taming_source, checkpoint, device="cuda"):
    for path in (source, ldm_source, taming_source):
        sys.path.insert(0, str(Path(path).resolve()))
    # The upstream LDM imports the historical Lightning location. The callable
    # itself is unchanged; no forward computation is replaced.
    import pytorch_lightning.utilities.rank_zero as rank_zero
    compatibility = types.ModuleType("pytorch_lightning.utilities.distributed")
    compatibility.rank_zero_only = rank_zero.rank_zero_only
    sys.modules.setdefault("pytorch_lightning.utilities.distributed", compatibility)
    import models.model as upstream
    from ldm.util import instantiate_from_config
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = weights["model"]
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    if any(k.startswith("dino.") for k in state):
        raise ValueError("this loader supports the official default non-DINO checkpoint only")

    def construct_for_full_checkpoint(config):
        config = copy.deepcopy(config)
        # Every retained inference parameter is loaded strictly below. Avoid
        # downloading redundant SD and CLIP weights that the final model replaces
        # or immediately deletes.
        config.params.ckpt_path = None
        config.params.cond_stage_config = {"target": "torch.nn.Identity"}
        return instantiate_from_config(config)

    old_factory = upstream.instantiate_from_config
    old_cwd = Path.cwd()
    try:
        upstream.instantiate_from_config = construct_for_full_checkpoint
        os.chdir(source)  # upstream uses its own relative v1-inference.yaml
        model = upstream.One2Any(args=types.SimpleNamespace(
            data_path="", data_name=None, dino=False, num_deconv=3,
            num_filters=[32, 32, 32], deconv_kernels=[2, 2, 2]))
    finally:
        upstream.instantiate_from_config = old_factory
        os.chdir(old_cwd)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval().requires_grad_(False)
    provenance = {
        "source": "https://github.com/lmy1001/One2Any",
        "checkpoint_sha256": sha256(checkpoint), "all_inference_weights_loaded_strictly": True,
        "source_sha256": {str(p.relative_to(source)): sha256(p)
                          for p in sorted(Path(source).rglob("*.py"))},
        "ldm_source_sha256": {str(p.relative_to(ldm_source)): sha256(p)
                              for p in sorted(Path(ldm_source).rglob("*.py"))},
        "compatibility": "Lightning rank_zero_only import alias; unused SD/CLIP initialization skipped",
        "checkpoint_training_objects": "official pretrained baseline; not verified object-disjoint with custom splits",
    }
    return model, provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--ldm-source", required=True)
    p.add_argument("--taming-source", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.set_num_threads(4)
    model, report = load_one2any(args.source, args.ldm_source, args.taming_source,
                                 args.checkpoint, args.device)
    with torch.inference_mode():
        x = torch.zeros(1, 3, 192, 192, device=args.device)
        reference = torch.zeros(1, 7, 192, 192, device=args.device)
        result = model(x, torch.ones_like(x[:, 0], dtype=torch.bool), ref_cond=reference)
    report.update(smoke_only=True, accuracy_evaluated=False,
                  output_shape=list(result["pred_nocs"].shape),
                  output_finite=bool(torch.isfinite(result["pred_nocs"]).all()))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"load_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("source_sha256")}), flush=True)


if __name__ == "__main__":
    main()
