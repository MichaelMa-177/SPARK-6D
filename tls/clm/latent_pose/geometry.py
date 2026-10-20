"""All metric coordinates are in metres; T maps object memory -> query camera."""

import torch


def transform_points(points, pose):
    return points @ pose[..., :3, :3].transpose(-1, -2) + pose[..., None, :3, 3]


def weighted_rigid(source, target, weights, min_points=6):
    """Weighted Kabsch with reflection correction and explicit failure status.

    Planar surfaces (rank two) are accepted; collinear/empty sets are rejected.
    Invalid rows are removed by zero weighting. No identity fallback is presented
    as a successful pose. Use this solver for inference; coordinate supervision
    avoids differentiating an SVD at degenerate initial predictions.
    """
    if source.ndim != 3 or source.shape != target.shape or source.shape[-1] != 3:
        raise ValueError("source/target must have matching [B,N,3] shapes")
    if weights.shape != source.shape[:2]:
        raise ValueError("weights must have shape [B,N]")
    source, target, weights = source.float(), target.float(), weights.float()
    finite = (
        torch.isfinite(source).all(-1)
        & torch.isfinite(target).all(-1)
        & torch.isfinite(weights)
        & (weights > 0)
    )
    w = torch.where(finite, weights, 0)
    a = torch.where(finite[..., None], source, 0)
    b = torch.where(finite[..., None], target, 0)
    total = w.sum(-1)
    wn = w / total.clamp_min(1e-12)[:, None]
    ac = (a * wn[..., None]).sum(1)
    bc = (b * wn[..., None]).sum(1)
    aa, bb = a - ac[:, None], b - bc[:, None]
    cov = (aa * wn[..., None]).transpose(1, 2) @ bb
    u, s, vh = torch.linalg.svd(cov)
    v = vh.transpose(-1, -2)
    correction = torch.ones_like(s)
    correction[:, -1] = torch.where(torch.det(v @ u.transpose(-1, -2)) < 0, -1., 1.)
    rotation = (v * correction[:, None]) @ u.transpose(-1, -2)
    translation = bc - (rotation @ ac[..., None]).squeeze(-1)
    pose = torch.eye(4, device=a.device).expand(a.shape[0], -1, -1).clone()
    pose[:, :3, :3], pose[:, :3, 3] = rotation, translation
    effective = total.square() / w.square().sum(-1).clamp_min(1e-12)
    valid = (
        (finite.sum(-1) >= min_points)
        & (effective >= 3)
        & (total > 1e-8)
        & (s[:, 1] > 1e-10)
        & (s[:, 1] / s[:, 0].clamp_min(1e-12) > 1e-5)
    )
    residual = ((transform_points(a, pose) - b).square().sum(-1) * wn).sum(-1).sqrt()
    identity = torch.eye(4, device=a.device).expand_as(pose)
    return {
        "pose": torch.where(valid[:, None, None], pose, identity),
        "pose_valid": valid,
        "residual_m": torch.where(valid, residual, torch.full_like(residual, float("inf"))),
        "singular_values": s,
        "effective_correspondences": effective,
    }


def camera_to_memory(points, pose, scale):
    """Inverse rigid transform, followed by the fixed reference scale."""
    return ((points - pose[:, None, :3, 3]) @ pose[:, :3, :3]) / scale[:, None]
