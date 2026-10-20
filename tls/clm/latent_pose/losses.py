import torch
from torch.nn import functional as F

from .geometry import camera_to_memory


def coordinate_loss(output, pose, scale_m, target_mask, symmetry_transforms=None):
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
    if not foreground.any():
        raise ValueError("training batch has no valid foreground depth")
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
    coordinate = error[foreground].mean()
    nll = (torch.exp(-uncertainty) * error + uncertainty)[foreground].mean()
    bce = F.binary_cross_entropy_with_logits(output["membership_logits"], mask.float(), reduction="none")
    membership = bce[valid].mean()
    residual = output.get("coordinate_residual")
    regularization = .01 * residual[foreground].square().mean() if residual is not None else coordinate * 0
    return {
        "loss": coordinate + .1 * nll + .2 * membership + regularization,
        "coordinate": coordinate,
        "uncertainty": nll,
        "membership": membership,
        "residual_regularization": regularization,
    }
