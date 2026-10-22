import torch
from torch.nn import functional as F

from .geometry import camera_to_memory


def coordinate_loss(output, pose, scale_m, target_mask, symmetry_transforms=None, rigidity_weight=0.):
    """Supervise memory-frame coordinates, target membership and uncertainty.

    symmetry_transforms: optional [B,S,4,4] metric rigid transforms expressed
    about the SAME memory origin. Identity must be included. Choose one symmetry
    for the whole object, never a different symmetry independently per pixel.
    """
    n = output["coordinates"].shape[1]
    side = int(n ** .5)
    if side * side != n:
        raise ValueError("query tokens must form a square grid")
    mask = F.interpolate(target_mask.float(), (side, side), mode="nearest").flatten(1) > .5
    valid = output["query_valid"]
    foreground = valid & mask
    truth = camera_to_memory(output["query_xyz_m"], pose, scale_m)
    candidates = truth[:, None]
    if symmetry_transforms is not None:
        sym = symmetry_transforms
        candidates = (
            torch.einsum("bnj,bskj->bsnk", truth * scale_m[:, None], sym[..., :3, :3])
            + sym[..., None, :3, 3]
        ) / scale_m[:, None, None]
    error = F.smooth_l1_loss(
        output["coordinates"][:, None].expand_as(candidates), candidates, reduction="none", beta=.05
    ).sum(-1)
    selection = (error * foreground[:, None]).sum(-1).argmin(-1)
    error = error[torch.arange(len(error), device=error.device), selection]
    uncertainty = output["log_variance"]
    # Fixed coefficient prevents an unconstrained confidence-only objective.
    zero = output["coordinates"].sum() * 0
    coordinate = error[foreground].mean() if foreground.any() else zero
    nll = (torch.exp(-uncertainty) * error + uncertainty)[foreground].mean() if foreground.any() else zero
    bce = F.binary_cross_entropy_with_logits(output["membership_logits"], mask.float(), reduction="none")
    membership = bce[valid].mean() if valid.any() else output["membership_logits"].sum() * 0
    residual = output.get("coordinate_residual")
    regularization = .01 * residual[foreground].square().mean() if residual is not None and foreground.any() else zero
    rigidity = zero
    if rigidity_weight:
        # A rigid coordinate field preserves pair distances, including on unseen
        # surfaces. This uses training foreground labels, without an SVD gradient.
        # Fixed uniform grid sampling leaves the batch/augmentation RNG unchanged.
        ids = torch.linspace(0, n-1, min(n, 256), device=foreground.device).round().long()
        selected = foreground[:, ids]
        pairs = selected[:, :, None] & selected[:, None, :]
        pairs &= torch.ones(len(ids), len(ids), device=pairs.device, dtype=torch.bool).triu(1)[None]
        predicted_distances = torch.cdist(output["coordinates"][:, ids], output["coordinates"][:, ids],
                                          compute_mode="donot_use_mm_for_euclid_dist")
        camera_points = output["query_xyz_m"][:, ids]/scale_m[:, None]
        measured_distances = torch.cdist(camera_points, camera_points,
                                         compute_mode="donot_use_mm_for_euclid_dist")
        if pairs.any():
            rigidity = F.smooth_l1_loss(predicted_distances[pairs], measured_distances[pairs], beta=.05)
    return {
        "loss": coordinate + .1 * nll + .2 * membership + regularization + rigidity_weight*rigidity,
        "coordinate": coordinate,
        "uncertainty": nll,
        "membership": membership,
        "residual_regularization": regularization,
        "rigidity": rigidity,
    }
