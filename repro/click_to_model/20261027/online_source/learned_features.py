"""Pinned XFeat features for observation-conditioned seeds, not output poses.

The pretrained CVPR 2024 model remains unmodified. This adapter isolates its
imports, uses safe tensor-only weight loading, and provides fixed-size GPU
selection (optional compilation) with a CPU feature/matching interface. Masking
restricts detector candidates, never masks the RGB seen by the network. Sparse
sampling follows the upstream interpolation coordinate convention. Timings of
this component alone are NOT end-to-end tracking FPS.
"""
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

UPSTREAM_COMMIT = "e92685f57f8318b18725c5c8c0bd28c7fe188d9a"
WEIGHT_SHA256 = "0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b"
MODEL_SOURCE_SHA256 = "d9a665f18fcea5eaf3e278925e1a92103afcba9051e05b2334f3daa29f411964"
UPSTREAM_ROOT = Path(__file__).resolve().parent / "third_party/accelerated_features"


@dataclass(frozen=True)
class FeatureSet:
    keypoints: np.ndarray
    descriptors: np.ndarray
    scores: np.ndarray

    def __post_init__(self):
        p, d, s = (np.asarray(v, dtype=np.float32) for v in
                   (self.keypoints, self.descriptors, self.scores))
        if p.ndim != 2 or p.shape[1:] != (2,) or d.ndim != 2 or d.shape[0] != len(p) or d.shape[1] < 1 or s.shape != (len(p),):
            raise ValueError("features require Nx2 points, NxD descriptors and N scores")
        if not all(np.isfinite(v).all() for v in (p, d, s)):
            raise ValueError("features must be finite")
        if len(d) and not np.allclose(np.linalg.norm(d, axis=1), 1, atol=2e-3, rtol=0):
            raise ValueError("descriptors must be L2 normalized")
        for name, value in zip(("keypoints", "descriptors", "scores"), (p, d, s)):
            value = np.array(value, copy=True, order="C")
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    def subset(self, indices):
        return FeatureSet(self.keypoints[indices], self.descriptors[indices], self.scores[indices])


def match_features(a, b, min_similarity=0.8, ratio=0.9):
    """Mutual nearest cosine matches with strict L2 ratio and unique endpoints.

    The ratio uses Euclidean descriptor distance, not a ratio of similarities.
    With fewer than two destination descriptors the ratio is undefined: return
    no matches. Exact duplicate ambiguities also fail the strict ratio test.
    """
    if not isinstance(a, FeatureSet) or not isinstance(b, FeatureSet):
        raise TypeError("matching inputs must be FeatureSet")
    if a.descriptors.shape[1] != b.descriptors.shape[1]:
        raise ValueError("descriptor dimensions disagree")
    if not -1 <= min_similarity <= 1 or not 0 < ratio <= 1:
        raise ValueError("require cosine threshold in [-1,1] and ratio in (0,1]")
    if not len(a.keypoints) or len(b.keypoints) < 2:
        return np.empty((0, 2), dtype=np.int64)
    similarities = a.descriptors @ b.descriptors.T
    dest = similarities.argmax(axis=1)
    source = similarities.argmax(axis=0)
    rows = np.arange(len(dest))
    best = np.clip(similarities[rows, dest], -1, 1)
    second = np.clip(np.partition(similarities, -2, axis=1)[:, -2], -1, 1)
    accepted = ((source[dest] == rows) & (best >= min_similarity)
                & (1 - best < ratio * ratio * (1 - second)))
    return np.column_stack((rows[accepted], dest[accepted])).astype(np.int64)


class _SparseNetwork(nn.Module):
    def __init__(self, backbone, max_features, detection_threshold, spatial_quota=0):
        super().__init__()
        self.backbone = backbone
        self.max_features = max_features
        self.detection_threshold = detection_threshold
        self.spatial_quota = spatial_quota

    def forward(self, image, mask):
        _, _, h, w = image.shape
        descriptors, logits, reliability = self.backbone(image)
        descriptors = F.normalize(descriptors, dim=1)
        heat = F.softmax(logits, dim=1)[:, :64]
        heat = heat.permute(0, 2, 3, 1).reshape(1, h // 8, w // 8, 8, 8)
        heat = heat.permute(0, 1, 3, 2, 4).reshape(1, 1, h, w)
        peaks = (heat == F.max_pool2d(heat, 5, stride=1, padding=2)) & (heat > self.detection_threshold)
        # Upstream samples reliability with align_corners=False at coordinates
        # normalized by W-1/H-1. Use the same convention for ranking and samples.
        yy, xx = torch.meshgrid(torch.arange(h, device=image.device),
                                torch.arange(w, device=image.device), indexing="ij")
        grid = torch.stack((2 * xx / (w - 1) - 1, 2 * yy / (h - 1) - 1), dim=-1).float()[None]
        reliability = F.grid_sample(reliability, grid, mode="bilinear", align_corners=False)
        # Upstream nearest heatmap sampling is identical at these integer points
        # except half-pixel border behavior. Exclude image borders explicitly.
        inside = (xx > 0) & (yy > 0) & (xx < w - 1) & (yy < h - 1)
        scores = torch.where(peaks & mask & inside[None, None], heat * reliability, 0).flatten()
        if self.spatial_quota:
            # A global top-k can allocate every keypoint to textured background,
            # while masked memory admitted weak target points. Reserve an equal
            # per-32px-cell quota in both memory and query extraction instead.
            cells = (h // 32) * (w // 32)
            quota = max(1, min(self.spatial_quota, self.max_features // cells))
            tiled = scores.reshape(h // 32, 32, w // 32, 32).permute(0, 2, 1, 3).reshape(cells, 1024)
            local_scores, local_ids = tiled.topk(quota, dim=1)
            cell_ids = torch.arange(cells, device=image.device)[:, None]
            px = (cell_ids % (w // 32)) * 32 + local_ids % 32
            py = (cell_ids // (w // 32)) * 32 + local_ids // 32
            indices, scores = (py * w + px).flatten(), local_scores.flatten()
            # max_features smaller than number of cells cannot cover all cells;
            # cap explicitly, without falsely claiming full coverage.
            if cells > self.max_features:
                scores, keep = scores.topk(self.max_features)
                indices = indices[keep]
        else:
            scores, indices = scores.topk(min(self.max_features, h * w))
        points = torch.stack((indices % w, indices // w), dim=-1).float()
        sample_grid = (2 * points / points.new_tensor([w - 1, h - 1]) - 1)[None, :, None]
        features = F.grid_sample(descriptors, sample_grid, mode="bicubic", align_corners=False)
        features = F.normalize(features[0, :, :, 0].T, dim=-1)
        return points, features, scores


class XFeatExtractor:
    def __init__(self, device="cpu", max_features=2048, compile_model=False,
                 detection_threshold=0.05, upstream_root=None, spatial_quota=0):
        if type(max_features) is not int or max_features < 1:
            raise ValueError("max_features must be a positive integer")
        if not 0 < detection_threshold < 1:
            raise ValueError("detection_threshold must be in (0,1)")
        if type(spatial_quota) is not int or not 0 <= spatial_quota <= 1024:
            raise ValueError("spatial_quota must be integer in [0,1024]")
        self.device = torch.device(device)
        root = Path(upstream_root) if upstream_root is not None else UPSTREAM_ROOT
        source, weight = root / "modules/model.py", root / "weights/xfeat.pt"
        if not source.is_file() or not weight.is_file():
            raise FileNotFoundError("Fetch pinned verlab/accelerated_features into third_party/accelerated_features first")
        if hashlib.sha256(weight.read_bytes()).hexdigest() != WEIGHT_SHA256:
            raise ValueError("XFeat pretrained weight checksum mismatch")
        if hashlib.sha256(source.read_bytes()).hexdigest() != MODEL_SOURCE_SHA256:
            raise ValueError("XFeat model source checksum mismatch; refusing dynamic import")
        module_name = "_spark6d_pinned_xfeat_model"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, source)
            module = importlib.util.module_from_spec(spec)
            # Dynamo resolves class definitions by module name during tracing.
            # Reuse this exact module for later instances: replacing sys.modules
            # breaks Dynamo's cached global aliases on a second object/sequence.
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                del sys.modules[module_name]
                raise
            module._spark_source_sha256 = MODEL_SOURCE_SHA256
        elif getattr(module, "_spark_source_sha256", None) != MODEL_SOURCE_SHA256:
            raise RuntimeError("conflicting XFeat module in process import registry")
        backbone = module.XFeatModel().to(self.device).eval()
        backbone.load_state_dict(torch.load(weight, map_location=self.device, weights_only=True), strict=True)
        self.network = _SparseNetwork(backbone, max_features, detection_threshold, spatial_quota).to(self.device).eval()
        self.parameters = sum(p.numel() for p in backbone.parameters())
        self.provenance = {"upstream": "https://github.com/verlab/accelerated_features",
                           "commit": UPSTREAM_COMMIT, "weight_sha256": WEIGHT_SHA256,
                           "model_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                           "compile_model": bool(compile_model), "max_features": max_features,
                           "detection_threshold": detection_threshold, "parameters": self.parameters,
                           "spatial_quota_per_32px_cell": spatial_quota,
                           "device": str(self.device), "precision": "float32",
                           "adapter": "masked_fixed_topk_border_excluded_upstream_sampling"}
        if compile_model:
            self.network = torch.compile(self.network, mode="reduce-overhead", fullgraph=True)

    @torch.inference_mode()
    def extract(self, rgb, mask=None):
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 32:
            raise ValueError("XFeat input must be uint8 RGB HxWx3 with H,W >=32")
        h, w = rgb.shape[:2]
        hh, ww = h // 32 * 32, w // 32 * 32
        if mask is None:
            mask = np.ones((h, w), dtype=np.uint8)
        mask = np.asarray(mask)
        if mask.shape != (h, w) or mask.dtype.kind not in "buif" or not np.isfinite(mask).all():
            raise ValueError("mask must be a finite numeric/bool array matching RGB")
        image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).to(self.device).float()[None] / 255.
        if (h, w) != (hh, ww):
            image = F.interpolate(image, (hh, ww), mode="bilinear", align_corners=False)
            mask = cv2.resize((mask != 0).astype(np.uint8), (ww, hh), interpolation=cv2.INTER_NEAREST)
        selected = torch.from_numpy(np.ascontiguousarray(mask != 0)).to(self.device)[None, None]
        points, descriptors, scores = self.network(image, selected)
        # Copy before the next compiled invocation can reuse CUDA-graph outputs.
        points, descriptors, scores = (x.cpu().numpy().copy() for x in (points, descriptors, scores))
        valid = scores > 0
        points = points[valid] * np.array([w / ww, h / hh], dtype=np.float32)
        return FeatureSet(points, descriptors[valid], scores[valid])
