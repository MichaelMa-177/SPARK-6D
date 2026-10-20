"""Versioned NPZ interchange. No dataset paths or ground truth enter the model."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_packet(path, reference=False, labels=False):
    """Select only requested keys. Inference never opens pose/mask label arrays."""
    keys = ["rgb", "xyz", "valid"]
    if reference:
        keys += ["mask", "origin_m", "scale_m", "shape_latent", "latent_xyz_m"]
    if labels:
        keys += ["pose", "target_mask"]
    with np.load(path, allow_pickle=False) as packet:
        if int(packet["schema_version"]) != 1 or str(packet["units"]) != "metres":
            raise ValueError("expected schema_version=1 and units='metres'")
        value = {k: torch.from_numpy(np.array(packet[k], dtype=np.float32, copy=True)) for k in keys}
        if labels:
            sym = np.array(packet["symmetry_transforms"], dtype=np.float32) if "symmetry_transforms" in packet else np.eye(4, dtype=np.float32)[None]
            value["symmetry_transforms"] = torch.from_numpy(sym)
    if labels:
        pose = value["pose"]
        if pose.shape != (4, 4) or not torch.isfinite(pose).all():
            raise ValueError("label pose must be a finite 4x4 metric rigid transform")
        if not torch.allclose(pose[3], torch.tensor([0., 0., 0., 1.]), atol=1e-5):
            raise ValueError("invalid homogeneous pose")
        rotation = pose[:3, :3]
        if not torch.allclose(rotation.T @ rotation, torch.eye(3), atol=1e-4) or torch.det(rotation) < .999:
            raise ValueError("pose rotation must lie in SO(3)")
        sym = value["symmetry_transforms"]
        if sym.ndim != 3 or sym.shape[1:] != (4, 4) or not torch.isfinite(sym).all():
            raise ValueError("symmetry_transforms must be finite [S,4,4]")
        if not torch.allclose(sym[:, 3], torch.tensor([0., 0., 0., 1.]).expand(len(sym), -1), atol=1e-5):
            raise ValueError("symmetry transforms must be homogeneous")
        rotations = sym[:, :3, :3]
        if not torch.allclose(rotations.transpose(-1, -2) @ rotations,
                              torch.eye(3).expand(len(sym), -1, -1), atol=1e-4) or (torch.det(rotations) < .999).any():
            raise ValueError("symmetry rotations must lie in SO(3)")
        if not torch.isclose(sym, torch.eye(4), atol=1e-5).all(-1).all(-1).any():
            raise ValueError("symmetry transforms must include identity")
    return value


def to_device(value, device, batch=False):
    if isinstance(value, dict):
        return {k: to_device(v, device, batch) for k, v in value.items()}
    return (value.unsqueeze(0) if batch else value).to(device)


class PairDataset(Dataset):
    def __init__(self, manifest, expected_split=None):
        self.path = Path(manifest).resolve()
        document = json.loads(self.path.read_text())
        if document.get("schema_version") != 1:
            raise ValueError("manifest schema_version must be 1")
        self.split = document["split"]
        if expected_split is not None and self.split != expected_split:
            raise ValueError(f"expected split={expected_split}; got {self.split}")
        self.samples = document["samples"]
        if not self.samples:
            raise ValueError("manifest contains no pairs")
        self.object_ids = {str(s["object_id"]) for s in self.samples}
        self.provenance = document.get("provenance", {})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        root = self.path.parent
        reference = read_packet(root / row["reference"], reference=True)
        packet = read_packet(root / row["query"], labels=True)
        query = {k: packet[k] for k in ("rgb", "xyz", "valid")}
        target = {k: packet[k] for k in ("pose", "target_mask", "symmetry_transforms")}
        return reference, query, target


def collate_pairs(rows):
    references, queries, targets = zip(*rows)
    max_sym = max(len(t["symmetry_transforms"]) for t in targets)
    padded = []
    for target in targets:
        t = dict(target)
        sym = t["symmetry_transforms"]
        t["symmetry_transforms"] = torch.cat((sym, torch.eye(4).repeat(max_sym - len(sym), 1, 1)))
        padded.append(t)
    return default_collate(references), default_collate(queries), default_collate(padded)
