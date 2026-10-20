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


def robust_rigid(source, target, weights, min_points=6, radius_m=.01, iterations=4):
    """Cauchy IRLS followed by a measured residual inlier check.

    All selection uses predicted weights and current geometry. No target mask,
    known pose, previous frame, or randomness enters this solver.
    """
    if radius_m <= 0 or iterations < 1:
        raise ValueError("positive inlier radius and iteration count required")
    finite = torch.isfinite(source).all(-1) & torch.isfinite(target).all(-1) & torch.isfinite(weights)
    base = torch.where(finite & (weights > 0), weights, 0).float()
    a, b = torch.nan_to_num(source.float()), torch.nan_to_num(target.float())
    result = weighted_rigid(a, b, base, min_points)
    for _ in range(iterations):
        residual = (transform_points(a, result["pose"]) - b).norm(dim=-1)
        current = base / (1 + (residual / radius_m).square())
        result = weighted_rigid(a, b, current, min_points)
    residual = (transform_points(a, result["pose"]) - b).norm(dim=-1)
    inliers = (residual <= radius_m) & (base > 0)
    result = weighted_rigid(a, b, base * inliers, min_points)
    result["inlier_count"] = inliers.sum(-1)
    return result


def consensus_rigid(source, target, weights, min_points=6, radius_m=.01, trials=256):
    """Deterministic per-query RANSAC for heavy predicted-correspondence outliers."""
    if source.ndim != 3 or source.shape != target.shape or source.shape[-1] != 3 or weights.shape != source.shape[:2]:
        raise ValueError("expected matching [B,N,3] points and [B,N] weights")
    if trials < 1 or radius_m <= 0:
        raise ValueError("positive trials and radius required")
    results = []
    for original_a, original_b, original_w in zip(source, target, weights):
        finite = torch.isfinite(original_a).all(-1) & torch.isfinite(original_b).all(-1)
        finite &= torch.isfinite(original_w) & (original_w > 0)
        a, b = torch.where(finite[:, None], original_a, 0).float(), torch.where(finite[:, None], original_b, 0).float()
        w = torch.where(finite, original_w, 0).float()
        if int(finite.sum()) < min_points:
            result = weighted_rigid(a[None], b[None], torch.zeros_like(w)[None], min_points)
            result["inlier_count"] = torch.zeros(1, device=a.device, dtype=torch.long)
            results.append(result)
            continue
        generator = torch.Generator(device=a.device).manual_seed(0)
        sample = torch.multinomial(w.sqrt(), trials * 3, replacement=True,
                                   generator=generator).reshape(trials, 3)
        hypotheses = weighted_rigid(a[sample], b[sample], torch.ones(trials, 3, device=a.device), 3)
        residual = (transform_points(a[None], hypotheses["pose"]) - b[None]).norm(dim=-1)
        inliers = (residual < radius_m) & finite[None] & hypotheses["pose_valid"][:, None]
        count = inliers.sum(-1)
        tie = (residual.clamp_max(radius_m)/radius_m).mean(-1)
        selected = inliers[(count.float() - .001*tie).argmax()]
        result = weighted_rigid(a[None], b[None], (w*selected)[None], min_points)
        for _ in range(2):
            residual = (transform_points(a[None], result["pose"]) - b[None]).norm(dim=-1)[0]
            selected = (residual < radius_m) & finite & result["pose_valid"][0]
            result = weighted_rigid(a[None], b[None], (w*selected)[None], min_points)
        result["inlier_count"] = selected.sum()[None]
        results.append(result)
    return {key: torch.cat([result[key] for result in results]) for key in results[0]}
