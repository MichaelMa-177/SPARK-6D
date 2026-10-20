"""Frozen local DINOv2 features aligned with the measured RGB-D sample grid."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from .data import sha256


def geometry_sample_grid(width, height, image_size, device):
    """Match two nearest-neighbour resizes used by XYZ, including pixel centres."""
    side = image_size // 4
    x = torch.floor(torch.arange(side, device=device) * 4 * width / image_size)
    y = torch.floor(torch.arange(side, device=device) * 4 * height / image_size)
    yy, xx = torch.meshgrid((y + .5) / height * 2 - 1,
                            (x + .5) / width * 2 - 1, indexing="ij")
    return torch.stack((xx, yy), -1)[None]


class FrozenDINO:
    def __init__(self, repo, weights, device="cuda"):
        self.device = device
        self.weights_sha256 = sha256(weights)
        self.model = torch.hub.load(str(repo), "dinov2_vitl14_reg", source="local", pretrained=False)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        self.model.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, rgb_crop, image_size=128):
        h, w = rgb_crop.shape[:2]
        value = torch.from_numpy(np.array(rgb_crop, copy=True)).to(self.device).permute(2, 0, 1)[None].float() / 255
        value = F.interpolate(value, (518, 518), mode="bicubic", align_corners=False, antialias=True)
        mean = value.new_tensor([.485, .456, .406])[None, :, None, None]
        std = value.new_tensor([.229, .224, .225])[None, :, None, None]
        with torch.autocast("cuda", enabled=str(self.device).startswith("cuda"), dtype=torch.float16):
            features = self.model.get_intermediate_layers((value - mean) / std, n=1, reshape=True)[0]
        grid = geometry_sample_grid(w, h, image_size, self.device)
        features = F.grid_sample(features.float(), grid, mode="bilinear", padding_mode="border",
                                 align_corners=False).flatten(2).transpose(1, 2)
        return features[0].cpu().numpy().astype(np.float16)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet-root", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--reference-rgb", required=True)
    p.add_argument("--reference-provenance", required=True)
    p.add_argument("--dino-repo", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.set_num_threads(4)
    root, output, raw = Path(args.packet_root), Path(args.output), Path(args.data_root)
    documents = {f.name: json.loads(f.read_text()) for f in root.glob("*.json")
                 if "samples" in json.loads(f.read_text())}
    provenance = json.loads(Path(args.reference_provenance).read_text())
    if sha256(args.reference_rgb) != provenance["rgb_sha256"]:
        raise ValueError("reference RGB hash mismatch")
    if sha256(root/"reference.npz") != provenance["files"]["reference.npz"]:
        raise ValueError("reference packet differs from verified provenance")
    output.mkdir(parents=True, exist_ok=False)
    extractor = FrozenDINO(args.dino_repo, args.weights, args.device)
    feature_provenance = {
        "backbone": "frozen DINOv2 ViT-L/14 registers",
        "weights_sha256": sha256(args.weights), "feature_source": "RGB only; no GT input",
        "code_sha256": sha256(__file__), "input_size": 518, "feature_dim": 1024,
        "upstream_packet_root": str(root.resolve()),
        "dino_source_sha256": {str(f.relative_to(args.dino_repo)): sha256(f)
                              for f in sorted(Path(args.dino_repo).rglob("*.py"))},
    }
    items = {"reference.npz": (Path(args.reference_rgb), provenance["reference_box_xyxy"], provenance["rgb_sha256"])}
    for document in documents.values():
        for row in document["samples"]:
            items[row["query"]] = (raw/f'{row["scene_id"]:06d}'/"rgb"/f'{row["frame_id"]:06d}.png',
                                    row["box_xyxy"], row["rgb_sha256"])
    hashes = {}
    for index, (name, (image_path, box, image_hash)) in enumerate(items.items()):
        if sha256(image_path) != image_hash:
            raise ValueError(f"source RGB mismatch: {image_path}")
        with np.load(root/name, allow_pickle=False) as loaded:
            packet = {k: loaded[k].copy() for k in loaded.files}
        rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        x0, y0, x1, y1 = box
        crop = rgb[y0:y1, x0:x1]
        size = packet["rgb"].shape[-1]
        expected = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA).transpose(2, 0, 1).astype(np.float32)/255
        if not np.array_equal(expected, packet["rgb"]):
            raise ValueError("raw crop and packet pixels do not match")
        packet["foundation_features"] = extractor(crop, size)
        path = output/name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **packet)
        hashes[name] = sha256(path)
        if index % 25 == 0:
            print(json.dumps({"done": index + 1, "total": len(items)}), flush=True)
    for name, document in documents.items():
        document["provenance"]["foundation_features"] = feature_provenance
        document["provenance"]["reference_sha256"] = hashes["reference.npz"]
        for row in document["samples"]:
            row["query_sha256"] = hashes[row["query"]]
        (output/name).write_text(json.dumps(document, indent=2)+"\n")
    (output/"features_provenance.json").write_text(json.dumps(feature_provenance, indent=2)+"\n")
    print(json.dumps({"completed": len(items), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
