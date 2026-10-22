"""Generate and freeze mesh hypotheses from the exact supplied human RGB-D reference."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import sha256
from .generated_mesh import mesh_from_fields
from .rgbd import measured_geometry
from .shape_prior import ShapePriorEncoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--dino-repo", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--steps", type=int, nargs="+", default=[1, 8, 25])
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if min(args.steps) < 1 or len(set(args.steps)) != len(args.steps):
        p.error("steps must be distinct positive integers")
    torch.set_num_threads(4)
    inputs, output = Path(args.inputs), Path(args.output)
    prompt = json.loads((inputs/"user_prompt.json").read_text())
    if prompt["prompt_source"] != "human_interactive_click" or sha256(inputs/"first_frame.png") != prompt["rgb_sha256"]:
        raise ValueError("input differs from the confirmed human reference")
    rgb = cv2.cvtColor(cv2.imread(str(inputs/"first_frame.png")), cv2.COLOR_BGR2RGB)
    depth, mask = cv2.imread(str(inputs/"first_depth.png"), -1), cv2.imread(str(inputs/"mask.png"), 0) > 0
    camera = json.loads((inputs/"camera.json").read_text())
    k = np.asarray(camera["cam_K"]).reshape(3, 3)
    _, xyz, valid = measured_geometry(rgb, depth, k, camera["depth_scale_m"])
    output.mkdir(parents=True, exist_ok=False)
    encoder = ShapePriorEncoder(args.source, args.dino_repo, args.weights, args.device, args.steps[0])
    for steps in args.steps:
        encoder.steps = steps
        encoder.provenance["steps"] = steps
        prior, report = encoder(rgb, depth, mask, k, camera["depth_scale_m"], return_dense=True)
        layout = {key: np.asarray(value, dtype=np.float32) for key, value in report["layout"].items()}
        mesh, geometry_report = mesh_from_fields(prior["occupancy_logits"], layout, rgb, xyz, valid, mask, k)
        folder = output/f"steps_{steps:02d}"
        folder.mkdir()
        np.savez_compressed(folder/"prior.npz", **prior, **{"layout_"+k: v for k, v in layout.items()})
        mesh.export(folder/"generated.ply")
        report = {**report, **geometry_report, "source": "supplied human RGB-D -> frozen SS -> occupancy mesh",
                  "rgb_sha256": sha256(inputs/"first_frame.png"), "mask_sha256": sha256(inputs/"mask.png"),
                  "depth_sha256": sha256(inputs/"first_depth.png"), "camera_sha256": sha256(inputs/"camera.json"),
                  "prompt_sha256": sha256(inputs/"user_prompt.json"), "mesh_sha256": sha256(folder/"generated.ply"),
                  "prior_sha256": sha256(folder/"prior.npz"), "exporter_sha256": sha256(__file__)}
        (folder/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({"steps": steps, "seconds": report["seconds"],
                          "colored_fraction": report["reference_projected_vertex_fraction"],
                          "extent_m": report["extent_m"]}), flush=True)


if __name__ == "__main__":
    main()
