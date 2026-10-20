"""Training-only random memory basis; physical RGB-D observations stay unchanged."""

import torch


def random_memory_basis(reference, target):
    """Use p_new = Q p_old, T_new = T_old Q^-1, S_new = Q S_old Q^-1."""
    b = reference["rgb"].shape[0]
    device = reference["rgb"].device
    q, r = torch.linalg.qr(torch.randn(b, 3, 3, device=device))
    q = q * r.diagonal(dim1=-2, dim2=-1).sign()[:, None]
    q[:, :, -1] *= torch.det(q)[:, None]
    change = torch.eye(4, device=device)[None].repeat(b, 1, 1)
    change[:, :3, :3] = q
    inverse = change.transpose(1, 2)  # pure homogeneous rotation
    transformed = dict(target)
    transformed["pose"] = target["pose"] @ inverse
    transformed["symmetry_transforms"] = change[:, None] @ target["symmetry_transforms"] @ inverse[:, None]
    return {**reference, "memory_basis": q}, transformed
