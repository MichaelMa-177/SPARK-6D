"""First-frame mask proposal admission without CAD, future frames, or GT.

Largest admissible clicked component is a whole-object heuristic, not identity
or object-completeness evidence. Distinct alternatives remain explicitly marked
ambiguous; callers can refuse automatic selection instead.
"""
from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class MeshlessMaskConfig:
    min_pixels: int = 100
    min_depth_fraction: float = .25
    min_depth_coherence: float = .80
    depth_band_absolute_m: float = .05
    depth_band_relative: float = .25
    border_margin: int = 2
    equivalent_mask_iou: float = .95

    def __post_init__(self):
        if type(self.min_pixels) is not int or self.min_pixels < 1:
            raise ValueError("min_pixels must be a positive integer")
        if type(self.border_margin) is not int or self.border_margin < 0:
            raise ValueError("border_margin must be a nonnegative integer")
        for name in ("min_depth_fraction", "min_depth_coherence", "equivalent_mask_iou"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be finite in (0,1]")
        for name in ("depth_band_absolute_m", "depth_band_relative"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite positive")


def select_meshless_mask(masks, scores, depth_m, click_xy, *, ambiguity_policy="largest", config=None):
    config = config or MeshlessMaskConfig()
    if ambiguity_policy not in ("largest", "reject"):
        raise ValueError("ambiguity_policy must be largest or reject")
    depth = np.asarray(depth_m, dtype=float)
    if depth.ndim != 2 or not len(masks) or len(masks) != len(scores):
        raise ValueError("Require an HxW depth image and equally many masks/scores")
    xy = np.asarray(click_xy, float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("Click must be finite x,y")
    x, y = np.rint(xy).astype(int)
    h, w = depth.shape
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError("Click is outside the image")
    patch = depth[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
    valid_patch = patch[np.isfinite(patch) & (patch > 1e-4)]
    reference_depth = float(np.median(valid_patch)) if len(valid_patch) else None
    band = (max(config.depth_band_absolute_m, config.depth_band_relative * reference_depth)
            if reference_depth is not None else None)
    records, components = [], []
    for index, (raw, score) in enumerate(zip(masks, scores)):
        record = {"index": index, "sam_score": None, "raw_pixels": 0, "component_pixels": 0,
                  "valid_depth_pixels": 0, "valid_depth_fraction": 0., "depth_coherence": None,
                  "reasons": [], "admissible": False}
        component = np.zeros(depth.shape, bool)
        raw = np.asarray(raw)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = math.nan
        if not np.isfinite(score):
            record["reasons"].append("nonfinite_sam_score")
        else:
            record["sam_score"] = float(score)
        if (raw.shape != depth.shape or raw.dtype.kind not in "buif"
                or not np.isfinite(raw).all() or not np.isin(raw, [0, 1, 255]).all()):
            record["reasons"].append("invalid_mask_shape_or_values")
        else:
            raw = raw != 0
            record["raw_pixels"] = int(raw.sum())
            if not raw[y, x]:
                record["reasons"].append("click_outside_mask")
            else:
                _, labels = cv2.connectedComponents(raw.astype(np.uint8), connectivity=8)
                component = labels == labels[y, x]
                count = int(component.sum())
                record["component_pixels"] = count
                if count < config.min_pixels:
                    record["reasons"].append("too_few_component_pixels")
                yy, xx = np.nonzero(component)
                margin = config.border_margin
                if (xx.min() <= margin or yy.min() <= margin
                        or xx.max() >= w - 1 - margin or yy.max() >= h - 1 - margin):
                    record["reasons"].append("truncated_at_image_border")
                values = depth[component]
                valid = values[np.isfinite(values) & (values > 1e-4)]
                record["valid_depth_pixels"] = int(len(valid))
                record["valid_depth_fraction"] = len(valid) / max(count, 1)
                if len(valid) < config.min_pixels or record["valid_depth_fraction"] < config.min_depth_fraction:
                    record["reasons"].append("insufficient_depth")
                if reference_depth is None:
                    record["reasons"].append("missing_click_neighborhood_depth")
                elif len(valid):
                    coherence = float(np.mean(np.abs(valid - reference_depth) <= band))
                    record["depth_coherence"] = coherence
                    record["depth_p10_p50_p90_m"] = np.percentile(valid, [10, 50, 90]).tolist()
                    if coherence < config.min_depth_coherence:
                        record["reasons"].append("depth_inconsistent_with_click_neighborhood")
        record["admissible"] = not record["reasons"]
        records.append(record)
        components.append(component)
    admissible = [r["index"] for r in records if r["admissible"]]
    comparisons = []
    for n, a in enumerate(admissible):
        for b in admissible[n + 1:]:
            overlap = float(np.count_nonzero(components[a] & components[b]) /
                            np.count_nonzero(components[a] | components[b]))
            comparisons.append({"a": a, "b": b, "iou": overlap})
    ambiguous = any(p["iou"] < config.equivalent_mask_iou for p in comparisons)
    selected = None
    if admissible and not (ambiguous and ambiguity_policy == "reject"):
        selected = max(admissible, key=lambda i: (records[i]["component_pixels"], records[i]["sam_score"], -i))
    status = ("no_admissible_candidate" if not admissible else
              "ambiguous_rejected" if selected is None else "selected_with_ambiguity" if ambiguous else "selected")
    return {"schema": "meshless_mask_selection/1", "status": status, "selected_index": selected,
            "ambiguous": ambiguous, "ambiguity_policy": ambiguity_policy, "config": asdict(config),
            "selection_rule": "admit clicked component with sufficient coherent depth and no border truncation; maximize area, SAM score only tie-break",
            "selection_is_heuristic": True, "identity_or_object_completeness_verified": False,
            "reference_click_depth_m": reference_depth, "depth_band_m": band,
            "candidates": records, "admissible_pairwise_iou": comparisons,
            "input_sources": ["current RGB SAM2 candidates", "current aligned depth", "click"],
            "cad_supplied_mask_or_gt_used": False}, components
