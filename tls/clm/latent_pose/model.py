from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .geometry import weighted_rigid


def conv(a, b, stride=1):
    return nn.Sequential(nn.Conv2d(a, b, 3, stride, 1), nn.GroupNorm(4, b), nn.GELU())


@dataclass
class ObjectMemory:
    tokens: torch.Tensor
    positions: torch.Tensor  # normalized coordinates in the first-frame object gauge
    kinds: torch.Tensor  # 0: measured surface; 1: generated volume cell
    scale_m: torch.Tensor
    origin_m: torch.Tensor  # first-camera centroid; metadata, never a query pose prior
    kv: Optional[list] = None


class RGBDEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.rgb = nn.Sequential(conv(3, width // 2, 2), conv(width // 2, width // 2, 2))
        self.xyz = nn.Sequential(conv(4, width // 2, 2), conv(width // 2, width // 2, 2))
        self.fuse = nn.Sequential(conv(width, width), nn.Conv2d(width, width, 1))

    def forward(self, rgb, xyz, valid, center, scale):
        local = torch.where(valid, (xyz - center[..., None, None]) / scale[..., None, None], 0)
        features = self.fuse(torch.cat((
            self.rgb(rgb - .5),
            self.xyz(torch.cat((local.clamp(-4, 4), valid.float()), 1)),
        ), 1))
        return features.flatten(2).transpose(1, 2)


class MemoryAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm_q, self.norm_m = nn.LayerNorm(width), nn.LayerNorm(width)
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, width * 2)
        self.out = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 4),
                                nn.GELU(), nn.Linear(width * 4, width))

    def split(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)

    def prepare(self, memory):
        k, v = self.kv(self.norm_m(memory)).chunk(2, -1)
        return self.split(k), self.split(v)

    def forward(self, query, memory, kv=None):
        k, v = self.prepare(memory) if kv is None else kv
        q = self.split(self.q(self.norm_q(query)))
        context = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
        context = context.transpose(1, 2).reshape_as(query)
        query = query + self.out(context)
        return query + self.ff(query)


class LatentPoseModel(nn.Module):
    """One reference -> reusable memory; each query is estimated independently.

    This is a trainable architecture, not a pretrained pose estimator. Generated
    volume-cell centers serve as positional encodings, NOT surface matches.
    The decoder predicts continuous reference surface coordinates.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.encoder = RGBDEncoder(c.width)
        self.shape_adapter = nn.Sequential(
            nn.Conv3d(c.latent_channels, c.width, 3, padding=1),
            nn.GroupNorm(8, c.width), nn.GELU(),
            nn.Conv3d(c.width, c.width, 3, padding=1),
        )
        self.position = nn.Sequential(nn.Linear(3, c.width), nn.GELU(), nn.Linear(c.width, c.width))
        self.kind = nn.Embedding(2, c.width)
        self.layers = nn.ModuleList([MemoryAttention(c.width, c.heads) for _ in range(c.decoder_layers)])
        self.coordinate_head = nn.Sequential(nn.LayerNorm(c.width), nn.Linear(c.width, c.width),
                                             nn.GELU(), nn.Linear(c.width, 3))
        self.membership_head = nn.Linear(c.width, 1)
        self.uncertainty_head = nn.Linear(c.width, 1)

    def _image(self, item):
        rgb, xyz, valid = item["rgb"], item["xyz"], item["valid"]
        n = self.config.image_size
        if rgb.ndim != 4 or rgb.shape[1:] != (3, n, n) or xyz.shape != rgb.shape:
            raise ValueError(f"rgb/xyz must have shape [B,3,{n},{n}]")
        if valid.shape != (rgb.shape[0], 1, n, n):
            raise ValueError("valid must have shape [B,1,H,W]")
        if not torch.isfinite(rgb).all() or (rgb < 0).any() or (rgb > 1).any():
            raise ValueError("rgb must be finite and in [0,1]")
        valid = valid.bool() & torch.isfinite(xyz).all(1, keepdim=True) & (xyz[:, 2:3] > 0)
        return rgb, torch.where(valid, xyz, 0), valid

    def _grid(self, value):
        size = self.config.image_size // 4
        return F.interpolate(value.float(), (size, size), mode="nearest").flatten(2).transpose(1, 2)

    def build_memory(self, reference, cache=False):
        c = self.config
        rgb, xyz, valid = self._image(reference)
        scale, origin = reference["scale_m"], reference["origin_m"]
        b = rgb.shape[0]
        if scale.shape != (b, 1) or origin.shape != (b, 3):
            raise ValueError("scale_m/origin_m must be [B,1]/[B,3]")
        if not torch.isfinite(scale).all() or (scale <= 0).any() or not torch.isfinite(origin).all():
            raise ValueError("reference scale must be positive and origin finite")
        if reference["mask"].shape != valid.shape:
            raise ValueError("reference mask shape mismatch")
        encoded = self.encoder(rgb, xyz, valid, origin, scale)
        positions = (self._grid(xyz) - origin[:, None]) / scale[:, None]
        eligible = self._grid(valid & reference["mask"].bool()).squeeze(-1) > .5
        indices = []
        for row in eligible:
            ids = row.nonzero().flatten()
            if len(ids) < 3:
                raise ValueError("reference requires at least 3 valid masked grid points")
            indices.append(ids[torch.linspace(0, len(ids) - 1, c.observed_tokens, device=rgb.device).round().long()])
        indices = torch.stack(indices)
        observed_f = encoded.gather(1, indices[..., None].expand(-1, -1, c.width))
        observed_p = positions.gather(1, indices[..., None].expand(-1, -1, 3))
        fs, ps = [observed_f], [observed_p]
        if c.memory_mode != "observed_only":
            z, zp = reference["shape_latent"], reference["latent_xyz_m"]
            cells = c.latent_side ** 3
            if z.shape != (b, cells, c.latent_channels) or zp.shape != (b, cells, 3):
                raise ValueError("latent tensors must match the configured ordered XYZ grid")
            if not torch.isfinite(z).all() or not torch.isfinite(zp).all():
                raise ValueError("latent features and coordinates must be finite")
            z = z.transpose(1, 2).reshape(b, c.latent_channels, *([c.latent_side] * 3))
            features = self.shape_adapter(z)
            if c.memory_mode == "geometry_only":
                features = features * 0  # same decoder/capacity, no latent feature information
            features = F.adaptive_avg_pool3d(features, c.memory_side).flatten(2).transpose(1, 2)
            points = ((zp - origin[:, None]) / scale[:, None]).transpose(1, 2)
            points = points.reshape(b, 3, *([c.latent_side] * 3))
            points = F.adaptive_avg_pool3d(points, c.memory_side).flatten(2).transpose(1, 2)
            fs.append(features)
            ps.append(points)
        features, positions = torch.cat(fs, 1), torch.cat(ps, 1)
        kinds = torch.zeros(features.shape[:2], dtype=torch.long, device=rgb.device)
        kinds[:, c.observed_tokens:] = 1
        tokens = features + self.position(positions) + self.kind(kinds)
        memory = ObjectMemory(tokens, positions, kinds, scale, origin)
        if cache:
            self.cache_memory(memory)
        return memory

    def cache_memory(self, memory):
        if self.training or torch.is_grad_enabled():
            raise ValueError("cache_memory requires model.eval() and torch.no_grad()")
        memory.kv = [layer.prepare(memory.tokens) for layer in self.layers]
        return memory

    def decode(self, query, memory, solve=True):
        rgb, xyz, valid = self._image(query)
        if rgb.shape[0] != memory.tokens.shape[0]:
            raise ValueError("query and memory batch sizes must match")
        count = valid.sum((2, 3)).clamp_min(1)
        center = xyz.sum((2, 3)) / count
        features = self.encoder(rgb, xyz, valid, center, memory.scale_m)
        if self.training and memory.kv is not None:
            raise ValueError("cached inference memory cannot be used for training")
        for i, layer in enumerate(self.layers):
            features = layer(features, memory.tokens, None if memory.kv is None else memory.kv[i])
        coords = self.coordinate_head(features)
        logits = self.membership_head(features).squeeze(-1)
        log_variance = self.uncertainty_head(features).squeeze(-1).clamp(-6, 6)
        query_xyz = self._grid(xyz)
        query_valid = self._grid(valid).squeeze(-1) > .5
        weights = logits.sigmoid() * torch.exp(-log_variance) * query_valid
        output = dict(coordinates=coords, membership_logits=logits, log_variance=log_variance,
                      weights=weights, query_xyz_m=query_xyz, query_valid=query_valid)
        if solve:
            # The solver does not backpropagate through potentially repeated singular values.
            with torch.no_grad():
                output.update(weighted_rigid(coords * memory.scale_m[:, None], query_xyz, weights,
                                             self.config.min_correspondences))
        return output

    def forward(self, reference, query, solve=False):
        return self.decode(query, self.build_memory(reference), solve=solve)
