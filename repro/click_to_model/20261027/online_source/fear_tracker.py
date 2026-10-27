"""Fixed-template FEAR-XS *2D proposals*, not a 6D tracker or a FPS claim.

Inference-only adaptation of https://github.com/PinataFarms/FEARTracker
at 0a3bd039918909c79c1b7e55a4bfb7807520abde (MIT, Copyright 2022
PiñataFarms; full notice: third_party/FEARTracker/LICENSE).
FBNet-C blocks/architecture are adapted from Facebook mobile-vision at
51804a6873ae1029257cf652179c960cceeecc75, Copyright (c) Facebook, Inc.
and its affiliates. All Rights Reserved. Its CC BY-NC 4.0 license applies
to these adapted portions: third_party/fear_mobile_vision/LICENSE and
https://creativecommons.org/licenses/by-nc/4.0/. This is NOT an all-MIT
adapter: noncommercial restriction and attribution must be retained.
Changes: remove training/dependency registries and unused classifier;
tensor-only encoder/head; safe pinned loading; transactional tracking state.

The public FEAR-XS-NoEmbs checkpoint has NO dynamic template update. Template
features and initialization padding color remain fixed, including on commit.
The upstream Python default has smooth=False (no Hann/penalty/box smoothing).
raw_score is its sigmoid activation, not calibrated P(tracking success), and
raw_logit is the unnormalized activation. An external, independently validated
6D controller must choose commit/reject; this module makes no trust decision.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch import nn


FEAR_COMMIT = "0a3bd039918909c79c1b7e55a4bfb7807520abde"
MOBILE_VISION_COMMIT = "51804a6873ae1029257cf652179c960cceeecc75"
CHECKPOINT_SHA256 = "8efe7dfd3498e385f332fd655f360a848d723f5fd77c2d433b2c084029616be5"
DEFAULT_CHECKPOINT = (Path(__file__).resolve().parent / "third_party/FEARTracker/"
                      "evaluate/checkpoints/FEAR-XS-NoEmbs.ckpt")


class _ConvBNReLU(nn.Module):
    """mobile-vision ConvBNRelu, inference subset (CC BY-NC 4.0)."""

    def __init__(self, cin, cout, kernel=1, stride=1, groups=1, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel, stride, kernel // 2,
                              groups=groups, bias=True)
        self.bn = nn.BatchNorm2d(cout, momentum=0.003)
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _IRF(nn.Module):
    """FBNet-C inverted residual, without unused SE/shuffle (CC BY-NC 4.0)."""

    def __init__(self, cin, cout, kernel, stride, expansion):
        super().__init__()
        middle = cin * expansion
        self.pw = _ConvBNReLU(cin, middle) if middle != cin else nn.Identity()
        self.dw = _ConvBNReLU(middle, middle, kernel, stride, groups=middle)
        self.pwl = _ConvBNReLU(middle, cout, relu=False)
        self.residual = cin == cout and stride == 1

    def forward(self, x):
        y = self.pwl(self.dw(self.pw(x)))
        return y + x if self.residual else y


def _fbnet_c_prefix():
    # Official flat xif names; FEAR max_layer=4 executes [:18], including skip.
    stages = OrderedDict([("xif0_0", _ConvBNReLU(3, 16, 3, 2)),
                          ("xif1_0", _IRF(16, 16, 3, 1, 1))])
    definitions = [
        [(24, 3, 2, 6), (24, 0, 1, 1), (24, 3, 1, 1), (24, 3, 1, 1)],
        [(32, 5, 2, 6), (32, 5, 1, 3), (32, 5, 1, 6), (32, 3, 1, 6)],
        [(64, 5, 2, 6), (64, 5, 1, 3), (64, 5, 1, 6), (64, 5, 1, 6),
         (112, 5, 1, 6), (112, 5, 1, 6), (112, 5, 1, 6), (112, 5, 1, 3)],
    ]
    cin = 16
    for stage, blocks in enumerate(definitions, 2):
        for index, (cout, kernel, stride, expansion) in enumerate(blocks):
            stages[f"xif{stage}_{index}"] = (
                _IRF(cin, cout, kernel, stride, expansion) if kernel else nn.Identity())
            cin = cout
    return nn.Sequential(stages)


class _SepConv(nn.Module):
    def __init__(self, cin, cout, bias=True):
        super().__init__()
        self.depthwise = nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=bias)
        self.pointwise = nn.Conv2d(cin, cout, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class _AdjustLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample = nn.Sequential(nn.Conv2d(112, 256, 1, bias=False), nn.BatchNorm2d(256))

    def forward(self, x):
        return self.downsample(x)


class _MatrixMobile(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix11_s = nn.Sequential(_SepConv(256, 256, bias=False),
                                        nn.BatchNorm2d(256), nn.ReLU(inplace=True))

    def forward(self, z, x):
        return z.reshape(z.shape[0], z.shape[1], -1), self.matrix11_s(x)


class _MobileCorrelation(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(_SepConv(320, 256), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

    def forward(self, z, x):
        b, c, h, w = x.shape
        correlation = torch.matmul(z.permute(0, 2, 1), x.reshape(b, c, -1)).reshape(b, -1, h, w)
        return self.enc(torch.cat([x, correlation], dim=1))


class _BoxTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_encode, self.reg_encode = _MatrixMobile(), _MatrixMobile()
        self.cls_dw, self.reg_dw = _MobileCorrelation(), _MobileCorrelation()
        self.bbox_tower = nn.Sequential(*[layer for _ in range(2) for layer in
                                        (_SepConv(256, 256), nn.BatchNorm2d(256), nn.ReLU())])
        self.cls_tower = nn.Sequential(*[layer for _ in range(2) for layer in
                                       (_SepConv(256, 256), nn.BatchNorm2d(256), nn.ReLU())])
        self.bbox_pred, self.cls_pred = _SepConv(256, 4), _SepConv(256, 1)
        self.adjust = nn.Parameter(0.1 * torch.ones(1))
        self.bias = nn.Parameter(torch.ones(1, 4, 1, 1))

    def forward(self, search, template):
        cls_z, cls_x = self.cls_encode(template, search)
        reg_z, reg_x = self.reg_encode(template, search)
        reg = self.bbox_tower(self.reg_dw(reg_z, reg_x))
        cls = self.cls_tower(self.cls_dw(cls_z, cls_x))
        return torch.exp(self.adjust * self.bbox_pred(reg) + self.bias), 0.1 * self.cls_pred(cls)


class FEARExtractor(nn.Module):
    """FP32 tensor-only FEAR-XS encoder/head suitable for torch.compile.

    Use from_checkpoint(), not random constructor weights, for actual proposals.
    forward(search[1,3,256,256], template_features[1,256,8,8]) returns
    positive LTRB distances [1,4,16,16] and logits [1,1,16,16].
    """

    def __init__(self):
        super().__init__()
        self.encoder = _fbnet_c_prefix()
        self.neck = _AdjustLayer()
        self.connect_model = _BoxTower()
        self.checkpoint_provenance = None

    def encode(self, normalized_rgb):
        return self.neck(self.encoder(normalized_rgb))

    def forward(self, normalized_search, template_features):
        return self.connect_model(self.encode(normalized_search), template_features)

    @classmethod
    def from_checkpoint(cls, checkpoint=DEFAULT_CHECKPOINT):
        path = Path(checkpoint).resolve()
        # Hash exactly the bytes subsequently loaded (no hash/load TOCTOU).
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != CHECKPOINT_SHA256:
            raise ValueError(f"FEAR checkpoint SHA256 mismatch: {digest}")
        data = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        if not isinstance(data, dict) or not isinstance(data.get("state_dict"), dict):
            raise ValueError("FEAR checkpoint has no state_dict")
        used, ignored = {}, []
        backbone_prefix = "model.encoder.model.backbone.stages."
        unused = tuple(f"{backbone_prefix}xif5_{i}." for i in range(5)) + (
            f"{backbone_prefix}xif6_0.", "model.encoder.model.head.conv.")
        for key, value in data["state_dict"].items():
            if not isinstance(key, str) or not isinstance(value, torch.Tensor):
                raise ValueError("FEAR state_dict must contain named tensors only")
            if key.startswith(unused):
                ignored.append(key)
                continue
            if key.startswith(backbone_prefix):
                target = "encoder." + key[len(backbone_prefix):]
            elif key.startswith(("model.neck.", "model.connect_model.")):
                target = key[len("model."):]
            else:
                raise ValueError(f"Unexpected FEAR checkpoint key: {key}")
            if target in used:
                raise ValueError(f"Duplicate FEAR checkpoint mapping: {target}")
            used[target] = value
        model = cls()
        model.load_state_dict(used, strict=True)
        model.requires_grad_(False).eval()
        model.checkpoint_provenance = {
            "checkpoint": str(path), "checkpoint_sha256": digest,
            "fear_commit": FEAR_COMMIT, "mobile_vision_commit": MOBILE_VISION_COMMIT,
            "licenses": ["FEAR: MIT", "mobile-vision adapted portions: CC BY-NC 4.0"],
            "loaded_state_entries": len(used), "unused_classifier_state_keys": sorted(ignored),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "template_update": "none_public_NoEmbs_checkpoint", "smooth": False,
            "role": "2d_proposal_only_not_6d_inference_or_end_to_end_fps",
        }
        return model


def _validate_rgb(rgb):
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
            or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 3):
        raise ValueError("Expected HxWx3 uint8 RGB image, with height/width >= 3")


def _clamp_bbox(bbox: Sequence[float], shape):
    values = np.asarray(bbox, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all() or np.any(values[2:] <= 0):
        raise ValueError("bbox must be finite [x, y, positive width, positive height]")
    # Preserve upstream ensure_bbox_boundaries ordering, including negative x/y.
    x, y, w, h = values
    x, y = min(max(0, x), shape[1]), min(max(0, y), shape[0])
    x2, y2 = min(max(0, x + w), shape[1]), min(max(0, y + h), shape[0])
    x, y, w, h = np.asarray([x, y, x2 - x, y2 - y], dtype=np.int32).tolist()
    if w < 3:
        w = 3
        x -= max(0, x + w - shape[1])
    if h < 3:
        h = 3
        y -= max(0, y + h - shape[0])
    return (x, y, w, h)


def _extended_crop(rgb, bbox, size, offset, padding_color):
    x, y, w, h = bbox
    context = np.asarray([x - w * offset, y - h * offset,
                          w * (1.0 + offset + offset), h * (1.0 + offset + offset)],
                         dtype=np.int32)
    cx, cy, cw, ch = map(int, context)
    left, top = max(-cx, 0), max(-cy, 0)
    right, bottom = max(cx + cw - rgb.shape[1], 0), max(cy + ch - rgb.shape[0], 0)
    crop = rgb[cy + top:cy + ch - bottom, cx + left:cx + cw - right]
    padded = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=tuple(map(float, padding_color)))
    return cv2.resize(padded, (size, size), interpolation=cv2.INTER_LINEAR), tuple(map(int, context))


def _normalize_rgb(rgb):
    # Albumentations 1.0.0 Normalize: mean/std scaled BEFORE subtraction.
    # https://github.com/albumentations-team/albumentations/blob/1.0.0/
    # albumentations/augmentations/functional.py#L122
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
    denominator = np.reciprocal(np.asarray([0.229, 0.224, 0.225], dtype=np.float32) * 255.0)
    normalized = (rgb.astype(np.float32) - mean) * denominator
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)[None]))


@dataclass(frozen=True)
class FEARProposal:
    bbox: tuple[int, int, int, int]
    raw_score: float
    raw_logit: float
    search_context_xywh: tuple[int, int, int, int]
    peak_rc: tuple[int, int]


class FEARTracker:
    """Caller-validated transaction around a fixed-template 2D proposal model.

    initialize(rgb, xywh); p = propose(rgb); commit(p) or reject(p).
    Reject freezes bbox/template/initial padding color. A new proposal cannot
    replace a pending one; stale/cross-tracker proposals cannot be committed.
    reset_box(xywh) explicitly resynchronizes only the search box from an
    externally validated observation; initialize() instead replaces template.
    No automatic update(), confidence threshold, GT input, or 6D pose output.
    """

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, *, device="cpu", compile_model=False,
                 extractor=None):
        self.device = torch.device(device)
        self.extractor = (FEARExtractor.from_checkpoint(checkpoint) if extractor is None else extractor)
        self.extractor.to(device=self.device, dtype=torch.float32).requires_grad_(False).eval()
        self._infer = (torch.compile(self.extractor, mode="reduce-overhead", fullgraph=True, dynamic=False)
                       if compile_model else self.extractor)
        self.compile_model = bool(compile_model)
        self._bbox = None
        self._template_features = None
        self._mean_color = None
        self._shape = None
        self._pending = None

    @property
    def bbox(self):
        return self._bbox

    @property
    def pending(self):
        return self._pending

    def provenance(self):
        import copy
        return {
            "extractor": copy.deepcopy(getattr(self.extractor, "checkpoint_provenance", None)),
            "device": str(self.device), "compile_model": self.compile_model,
            "precision": "float32", "template_size": 128, "search_size": 256,
            "search_context_offset": 2.0, "template_context_offset": 0.2,
            "normalization": "RGB_uint8_Albumentations_1.0.0_ImageNet",
            "smooth": False, "dynamic_template_update": False,
            "raw_score_semantics": "upstream_sigmoid_activation_not_calibrated_success_probability",
        }

    @torch.inference_mode()
    def initialize(self, rgb, bbox):
        _validate_rgb(rgb)
        clamped = _clamp_bbox(bbox, rgb.shape)
        mean = np.mean(rgb, axis=(0, 1))
        crop, _ = _extended_crop(rgb, clamped, 128, 0.2, mean)
        features = self.extractor.encode(_normalize_rgb(crop).to(self.device))
        if features.shape != (1, 256, 8, 8) or not torch.isfinite(features).all():
            raise ValueError("FEAR template features must be finite [1,256,8,8]")
        # Initialization changes state only after all validation/inference passes.
        self._template_features = features.detach().clone()
        self._bbox, self._mean_color, self._shape = clamped, mean, rgb.shape
        self._pending = None

    @torch.inference_mode()
    def propose(self, rgb):
        if self._bbox is None:
            raise RuntimeError("initialize must precede propose")
        if self._pending is not None:
            raise RuntimeError("commit or reject the pending proposal before proposing again")
        _validate_rgb(rgb)
        if rgb.shape != self._shape:
            raise ValueError("Image shape changed; explicitly reinitialize tracker")
        crop, context = _extended_crop(rgb, self._bbox, 256, 2.0, self._mean_color)
        regression, logits = self._infer(_normalize_rgb(crop).to(self.device), self._template_features)
        if regression.shape != (1, 4, 16, 16) or logits.shape != (1, 1, 16, 16):
            raise ValueError("Invalid FEAR prediction shape")
        # Copy results before a future compiled call can reuse graph output storage.
        distances = regression.detach().float().cpu().numpy().copy()
        activations = logits.detach().float().cpu()
        scores = activations.sigmoid().numpy().copy()
        logit_values = activations.numpy().copy()
        if (not np.isfinite(distances).all() or not np.isfinite(logit_values).all()
                or np.any(distances <= 0)):
            raise ValueError("FEAR prediction contains invalid/nonpositive LTRB or logits")
        row, col = np.unravel_index(np.argmax(scores[0, 0]), (16, 16))
        left, top, right, bottom = map(float, distances[0, :, row, col])
        cx, cy, cw, ch = context
        # Upstream grid is [0,16,...,240], NOT half-stride pixel centers.
        box = [round((col * 16 - left) * cw / 256 + cx),
               round((row * 16 - top) * ch / 256 + cy),
               max(3, round((left + right) * cw / 256)),
               max(3, round((top + bottom) * ch / 256))]
        proposal = FEARProposal(
            _clamp_bbox(box, rgb.shape), float(scores[0, 0, row, col]),
            float(logit_values[0, 0, row, col]), context, (int(row), int(col)))
        self._pending = proposal
        return proposal

    def _check_pending(self, proposal):
        if self._pending is None:
            raise RuntimeError("No pending FEAR proposal")
        if proposal is not None and proposal is not self._pending:
            raise ValueError("Stale or foreign FEAR proposal")

    def commit(self, proposal=None):
        self._check_pending(proposal)
        self._bbox = self._pending.bbox
        self._pending = None
        return self._bbox

    def reject(self, proposal=None):
        self._check_pending(proposal)
        self._pending = None
        return self._bbox

    def reset_box(self, bbox):
        """Set an externally validated xywh search box, retaining fixed template.

        This is an explicit caller-authorized state change, not a confidence
        check. Resolve any pending transaction first. No RGB/depth/GT is read.
        """
        if self._bbox is None:
            raise RuntimeError("initialize must precede reset_box")
        if self._pending is not None:
            raise RuntimeError("commit or reject the pending proposal before reset_box")
        clamped = _clamp_bbox(bbox, self._shape)
        self._bbox = clamped
        return clamped
