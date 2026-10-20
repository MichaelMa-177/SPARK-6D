"""Single supplied RGB-D + mask -> aligned SAM3D SS latent; no pose/CAD input."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
import torch

from .bop import object_annotation
from .data import sha256
from .export_reference import sam3d_to_camera
from .rgbd import measured_geometry


class ShapePriorEncoder:
    def __init__(self, source, dino_repo, weights, device="cuda", steps=1):
        source, weights = Path(source), Path(weights)
        self.device, self.steps = device, steps
        os.environ.update(SAM3D_DINO_REPO=str(dino_repo),
                          SAM3D_DINO_WEIGHTS=str(weights/"dinov2_vitl14_reg4_pretrain.pth"),
                          ATTN_BACKEND="sdpa", SPARSE_ATTN_BACKEND="sdpa", LIDRA_SKIP_INIT="1")
        sys.path[:0] = [str(source/"pytorch3d"), str(source)]
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(weights/"pipeline.yaml")
        cfg.workspace_dir = str(weights)
        cfg.compile_model = False
        cfg.low_vram = True
        cfg.decode_formats = ["mesh"]
        cfg.depth_model = {"_target_": "torch.nn.Identity"}
        cfg.rendering_engine = "pytorch3d"
        with torch.device(device):
            self.pipeline = instantiate(cfg)
        self.provenance = {
            "source": "SAM3D frozen SS spatial latent", "steps": steps,
            "weight_sha256": {name: sha256(weights/name) for name in
                              ("ss_generator.ckpt", "ss_decoder.ckpt", "dinov2_vitl14_reg4_pretrain.pth")},
            "pipeline_config_sha256": sha256(weights/"pipeline.yaml"),
            "code_sha256": sha256(__file__),
            "vendor_pipeline_sha256": sha256(source/"sam3d_objects/pipeline/inference_pipeline.py"),
            "GT_pose_or_CAD_input": False, "mask_is_supplied": True, "SAM_segmentation_executed": False,
            "geometry": "generated volume cells, not observed or exact CAD surfaces",
        }

    @torch.no_grad()
    def __call__(self, rgb, depth, mask, intrinsics, depth_scale_m):
        from pytorch3d.transforms import quaternion_to_matrix
        from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
        _, xyz, valid = measured_geometry(rgb, depth, intrinsics, depth_scale_m)
        mask = np.asarray(mask) > 0
        if mask.shape != valid.shape or (mask & valid).sum() < 12:
            raise ValueError("supplied mask needs at least 12 valid registered depth pixels")
        xyz[~valid] = np.nan
        xyz[:, :, :2] *= -1
        rgba = np.concatenate((rgb, (mask.astype(np.uint8)*255)[..., None]), -1)
        torch.manual_seed(0)
        torch.cuda.synchronize()
        started = time.perf_counter()
        pipeline = self.pipeline
        pointmap = pipeline._clip_pointmap(torch.from_numpy(xyz).to(self.device).permute(2, 0, 1),
                                           torch.from_numpy(mask.astype(np.float32)).to(self.device))
        item = pipeline.preprocess_image(rgba, pipeline.ss_preprocessor, pointmap=pointmap)
        generator = pipeline.models["ss_generator"]
        generator.inference_steps = self.steps
        generator.no_shortcut = False
        generator.reverse_fn.strength = generator.reverse_fn.strength_pm = 0
        with torch.autocast("cuda", dtype=torch.float16):
            args, kwargs = pipeline.get_condition_input(pipeline.condition_embedders["ss_condition_embedder"],
                                                        item, pipeline.ss_condition_input_mapping)
            shapes = {key: (1, value.pos_emb.shape[0], value.input_layer.in_features)
                      for key, value in generator.reverse_fn.backbone.latent_mapping.items()}
            raw = generator(shapes, self.device, *args, **kwargs)
        layout = pipeline.pose_decoder(raw, scene_scale=item.get("pointmap_scale"),
                                        scene_shift=item.get("pointmap_shift"))
        latent = raw["shape"][0].float().cpu().numpy()
        if latent.shape != (4096, 8) or not np.isfinite(latent).all():
            raise ValueError("expected finite ordered 16^3 x 8 spatial latent")
        grid = np.stack(np.meshgrid(*([np.arange(16)]*3), indexing="ij"), -1).reshape(-1, 3)
        canonical = (grid.astype(np.float32)+.5)/16-.5
        fields = {k: layout[k].float().cpu().numpy() for k in ("scale", "rotation", "translation")}
        points = sam3d_to_camera(canonical, fields["scale"], fields["rotation"], fields["translation"])
        # Verify all cell centres against the actual upstream row-vector transform.
        transform = compose_transform(layout["scale"], quaternion_to_matrix(layout["rotation"]),
                                      layout["translation"])
        check = transform.transform_points(torch.from_numpy(canonical).to(self.device)[None])[0].cpu().numpy()
        check[:, :2] *= -1
        error = float(np.abs(check-points).max())
        if error > 1e-5 or not np.isfinite(points).all():
            raise ValueError("generated latent coordinate alignment failed")
        torch.cuda.synchronize()
        return dict(shape_latent=latent, latent_xyz_m=points), {
            **self.provenance, "seconds": time.perf_counter()-started,
            "layout": {k: v.tolist() for k, v in fields.items()}, "max_transform_error_m": error,
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet-root", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--dino-repo", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--max-references", type=int, default=0, help="smoke-only limit; does not write manifests")
    args = p.parse_args()
    if args.steps < 1 or args.max_references < 0:
        p.error("steps must be positive and max-references nonnegative")
    torch.set_num_threads(4)
    root, raw, output = Path(args.packet_root), Path(args.data_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    documents = {s: json.loads((root/f"{s}.json").read_text()) for s in ("train", "validation")}
    rows = {row["reference"]: row for d in documents.values() for row in d["samples"]}
    encoder = ShapePriorEncoder(args.source, args.dino_repo, args.weights, args.device, args.steps)
    print(json.dumps({"event": "shape_encoder_loaded", "references": len(rows)}), flush=True)
    reports = {}
    for i, (name, row) in enumerate(rows.items()):
        if args.max_references and i >= args.max_references:
            break
        scene = raw/f'{row["reference_scene"]:06d}'
        frame = row["reference_frame"]
        rgb_path = scene/"rgb"/f"{frame:06d}.png"
        if sha256(rgb_path) != row["reference_rgb_sha256"]:
            raise ValueError("reference image differs from the training protocol")
        # GT metadata is used only to find the supplied reference-mask file.
        # No annotation rotation or translation is passed to ShapePriorEncoder.
        annotations = json.loads((scene/"scene_gt.json").read_text())
        index, _ = object_annotation(annotations[str(frame)], int(row["object_id"].split(":")[1]))
        mask_path = scene/"mask_visib"/f"{frame:06d}_{index:06d}.png"
        mask = cv2.imread(str(mask_path), 0) > 0
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        depth_path = scene/"depth"/rgb_path.name
        depth = cv2.imread(str(depth_path), -1)
        camera = json.loads((scene/"scene_camera.json").read_text())[str(frame)]
        prior, report = encoder(rgb, depth, mask, np.array(camera["cam_K"]).reshape(3, 3),
                                camera["depth_scale"]/1000)
        with np.load(root/name, allow_pickle=False) as saved:
            packet = {k: saved[k].copy() for k in saved.files}
        packet.update(prior)
        destination = output/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **packet)
        report.update(mask_source="official BOP reference mask_visib; not a human prompt",
                      rgb_sha256=sha256(rgb_path), mask_sha256=sha256(mask_path),
                      depth_sha256=sha256(depth_path), camera=camera,
                      reference_packet_sha256=sha256(destination), object_id=row["object_id"])
        destination.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
        reports[name] = report
        print(json.dumps({"done": i+1, "reference": name, "seconds": report["seconds"],
                          "transform_error_m": report["max_transform_error_m"]}), flush=True)
    if not args.max_references:
        for split, document in documents.items():
            for row in document["samples"]:
                destination = output/row["query"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root/row["query"], destination)
                row["reference_sha256"] = reports[row["reference"]]["reference_packet_sha256"]
            document["provenance"]["shape_prior"] = encoder.provenance
            (output/f"{split}.json").write_text(json.dumps(document, indent=2)+"\n")
    (output/"shape_prior_summary.json").write_text(json.dumps({
        "references": len(reports), "smoke_only": bool(args.max_references),
        "seconds_mean": float(np.mean([v["seconds"] for v in reports.values()])),
        "provenance": encoder.provenance}, indent=2)+"\n")


if __name__ == "__main__":
    main()
