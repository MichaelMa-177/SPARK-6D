from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .geometry import consensus_rigid, robust_rigid, weighted_rigid


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
    observed_descriptors: Optional[torch.Tensor] = None
    background_descriptors: Optional[torch.Tensor] = None


class RGBDEncoder(nn.Module):
    def __init__(self, width, foundation_dim=0):
        super().__init__()
        self.rgb = nn.Sequential(conv(3, width // 2, 2), conv(width // 2, width // 2, 2))
        self.xyz = nn.Sequential(conv(4, width // 2, 2), conv(width // 2, width // 2, 2))
        self.fuse = nn.Sequential(conv(width, width), nn.Conv2d(width, width, 1))
        self.foundation_dim = foundation_dim
        if foundation_dim:
            self.foundation = nn.Sequential(nn.LayerNorm(foundation_dim), nn.Linear(foundation_dim, width))

    def forward(self, rgb, xyz, valid, center, scale, foundation_features=None):
        local = torch.where(valid, (xyz - center[..., None, None]) / scale[..., None, None], 0)
        features = self.fuse(torch.cat((
            self.rgb(rgb - .5),
            self.xyz(torch.cat((local.clamp(-4, 4), valid.float()), 1)),
        ), 1))
        features = features.flatten(2).transpose(1, 2)
        if self.foundation_dim:
            expected = (*features.shape[:2], self.foundation_dim)
            if foundation_features is None or foundation_features.shape != expected:
                raise ValueError(f"foundation_features must have shape {expected}")
            if not torch.isfinite(foundation_features).all():
                raise ValueError("foundation_features must be finite")
            features = features + self.foundation(foundation_features)
        return features


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
        self.encoder = RGBDEncoder(c.width, c.foundation_feature_dim)
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
        if c.coordinate_mode == "memory_residual":
            self.match_query = nn.Linear(c.width, c.width)
            self.match_key = nn.Linear(c.width, c.width)
            nn.init.zeros_(self.match_query.weight)
            nn.init.zeros_(self.match_query.bias)
            nn.init.zeros_(self.coordinate_head[-1].weight)
            nn.init.zeros_(self.coordinate_head[-1].bias)
        if c.membership_mode == "appearance_prior":
            nn.init.zeros_(self.membership_head.weight)
            nn.init.zeros_(self.membership_head.bias)

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
        encoded = self.encoder(rgb, xyz, valid, origin, scale, reference.get("foundation_features"))
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
        if "memory_basis" in reference:
            basis = reference["memory_basis"]
            if basis.shape != (b, 3, 3) or not torch.isfinite(basis).all():
                raise ValueError("memory_basis must be finite [B,3,3]")
            if not torch.allclose(basis.transpose(1, 2) @ basis,
                                  torch.eye(3, device=basis.device).expand(b, -1, -1), atol=1e-4) or (torch.det(basis) < .999).any():
                raise ValueError("memory_basis must contain proper rotations")
            positions = positions @ basis.transpose(1, 2)
        kinds = torch.zeros(features.shape[:2], dtype=torch.long, device=rgb.device)
        kinds[:, c.observed_tokens:] = 1
        tokens = features + self.position(positions) + self.kind(kinds)
        memory = ObjectMemory(tokens, positions, kinds, scale, origin)
        if c.coordinate_mode == "memory_residual":
            memory.observed_descriptors = reference["foundation_features"].gather(
                1, indices[..., None].expand(-1, -1, c.foundation_feature_dim))
        if c.membership_mode == "appearance_prior":
            backgrounds = []
            unmasked = (self._grid(valid).squeeze(-1) > .5) & ~eligible
            for row, descriptors in zip(unmasked, reference["foundation_features"]):
                ids = row.nonzero().flatten()
                if len(ids):
                    selected = ids[torch.linspace(0, len(ids)-1, c.observed_tokens, device=rgb.device).round().long()]
                    backgrounds.append(descriptors[selected])
                else:
                    backgrounds.append(descriptors.new_zeros(c.observed_tokens, c.foundation_feature_dim))
            memory.background_descriptors = torch.stack(backgrounds)
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
        features = self.encoder(rgb, xyz, valid, center, memory.scale_m, query.get("foundation_features"))
        if self.training and memory.kv is not None:
            raise ValueError("cached inference memory cannot be used for training")
        for i, layer in enumerate(self.layers):
            features = layer(features, memory.tokens, None if memory.kv is None else memory.kv[i])
        coords = self.coordinate_head(features)
        coordinate_residual = None
        if self.config.coordinate_mode == "memory_residual":
            # A coordinate must be read from this object's measured memory first.
            # The learned residual can explain unseen surfaces, conditioned on
            # the full memory (including generated cells when configured).
            count = self.config.observed_tokens
            appearance = F.normalize(query["foundation_features"], dim=-1) @ F.normalize(
                memory.observed_descriptors, dim=-1).transpose(1, 2)
            learned = self.match_query(features) @ self.match_key(memory.tokens[:, :count]).transpose(1, 2)
            attention = (appearance / .05 + learned / self.config.width ** .5).softmax(-1)
            coordinate_residual = coords
            coords = attention @ memory.positions[:, :count] + coordinate_residual
        logits = self.membership_head(features).squeeze(-1)
        membership_prior = None
        if self.config.membership_mode == "appearance_prior":
            background = F.normalize(query["foundation_features"], dim=-1) @ F.normalize(
                memory.background_descriptors, dim=-1).transpose(1, 2)
            positive = appearance.max(-1).values
            negative = background.max(-1).values
            membership_prior = 20 * (positive - negative) + 5 * (positive - .6)
            # Bound the learned correction so the reference comparison remains
            # relevant on unseen objects.
            logits = membership_prior + 2 * logits.tanh()
        log_variance = self.uncertainty_head(features).squeeze(-1).clamp(-6, 6)
        query_xyz = self._grid(xyz)
        query_valid = self._grid(valid).squeeze(-1) > .5
        weights = logits.sigmoid() * torch.exp(-log_variance) * query_valid
        output = dict(coordinates=coords, membership_logits=logits, log_variance=log_variance,
                      weights=weights, query_xyz_m=query_xyz, query_valid=query_valid)
        if coordinate_residual is not None:
            output["coordinate_residual"] = coordinate_residual
        if membership_prior is not None:
            output["membership_prior_logits"] = membership_prior
        if solve:
            # The solver does not backpropagate through potentially repeated singular values.
            with torch.no_grad():
                if self.config.solver_mode != "weighted":
                    selected = weights * (logits.sigmoid() >= self.config.membership_threshold)
                    solver = consensus_rigid if self.config.solver_mode == "consensus" else robust_rigid
                    count = self.config.consensus_trials if self.config.solver_mode == "consensus" else self.config.robust_iterations
                    output.update(solver(coords * memory.scale_m[:, None], query_xyz, selected,
                                         self.config.min_correspondences, self.config.inlier_radius_m, count))
                else:
                    output.update(weighted_rigid(coords * memory.scale_m[:, None], query_xyz, weights,
                                                 self.config.min_correspondences))
        return output

    def forward(self, reference, query, solve=False):
        return self.decode(query, self.build_memory(reference), solve=solve)
