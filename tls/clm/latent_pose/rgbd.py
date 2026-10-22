"""Registered RGB-D frames -> network packets, without labels or segmentation."""

import cv2
import numpy as np

from .export_reference import crop_packet


def measured_geometry(rgb, depth, intrinsics, depth_scale_m):
    """Lift registered depth with a rectified pinhole RGB camera calibration.

    RGB is HWC uint8. Depth may be integer sensor units or floating point;
    depth_scale_m explicitly converts either representation to metres.
    """
    rgb, depth = np.asarray(rgb), np.asarray(depth)
    k = np.asarray(intrinsics, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be HWC uint8 RGB")
    if depth.ndim != 2 or depth.shape != rgb.shape[:2] or min(depth.shape) < 2:
        raise ValueError("depth must be registered to RGB with the same H,W")
    if depth.dtype.kind not in "uif":
        raise ValueError("depth must be a numeric array")
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("intrinsics must be a finite 3x3 pinhole camera matrix")
    if not np.allclose(k[2], [0, 0, 1]) or not np.allclose([k[0, 1], k[1, 0]], 0):
        raise ValueError("rectified zero-skew pinhole intrinsics are required")
    if not np.isfinite(depth_scale_m) or depth_scale_m <= 0:
        raise ValueError("depth_scale_m must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore"):
        z = depth.astype(np.float32) * depth_scale_m
        y, x = np.indices(depth.shape, dtype=np.float32)
        xyz = np.stack(((x - k[0, 2]) * z / k[0, 0],
                        (y - k[1, 2]) * z / k[1, 1], z), -1).astype(np.float32)
    valid = np.isfinite(xyz).all(-1) & (z > 0)
    xyz[~valid] = 0
    return rgb, xyz, valid


def checked_box(box, shape):
    """Half-open XYXY crop. XYZ stays in the original camera coordinates."""
    h, w = shape[:2]
    if box is None:
        return (0, 0, w, h)
    values = np.asarray(box)
    if values.shape != (4,) or values.dtype.kind not in "iu":
        raise ValueError("box must contain four integer XYXY coordinates")
    x0, y0, x1, y1 = (int(v) for v in values)
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("box must be nonempty and inside the registered image")
    return x0, y0, x1, y1


def reference_crop_box(mask, padding=.25):
    mask = np.asarray(mask)
    y, x = np.where(mask)
    if not len(x) or not np.isfinite(padding) or padding < 0:
        raise ValueError("nonempty mask and nonnegative padding required")
    pad = int(max(x.max() - x.min(), y.max() - y.min()) * padding) + 4
    return (max(0, int(x.min()) - pad), max(0, int(y.min()) - pad),
            min(mask.shape[1], int(x.max()) + pad + 1),
            min(mask.shape[0], int(y.max()) + pad + 1))


def prepare_query(rgb, depth, intrinsics, depth_scale_m, config, box=None):
    """No mask, previous pose, or GT is required; default input is the full frame."""
    rgb, xyz, valid = measured_geometry(rgb, depth, intrinsics, depth_scale_m)
    box = checked_box(box, rgb.shape)
    return crop_packet(rgb, xyz, valid, box, config.image_size)


def prepare_reference(rgb, depth, mask, intrinsics, depth_scale_m, config,
                      shape_latent=None, latent_xyz_m=None, padding=.25, shape_geometry=None):
    """Build a packet from ONE supplied mask and RGB-D observation.

    Spatial modes require the generated latent AND its cell centres, aligned
    to this reference RGB camera in metres. This function does not generate a
    shape prior or claim that a provided mask was human-confirmed.
    """
    rgb, xyz, valid = measured_geometry(rgb, depth, intrinsics, depth_scale_m)
    mask = np.asarray(mask)
    if mask.shape != valid.shape or not np.isin(mask, [0, 1, 255]).all():
        raise ValueError("mask must be binary H,W (0/1 or 0/255)")
    if not np.isfinite(padding) or padding < 0:
        raise ValueError("padding must be finite and nonnegative")
    mask = mask > 0
    points = xyz[mask & valid]
    if len(points) < 12:
        raise ValueError("reference requires at least 12 valid masked depth pixels")
    origin = points.mean(0)
    scale = np.array([max(float(np.ptp(points, axis=0).max()), .01)], np.float32)
    box = reference_crop_box(mask, padding)
    packet = crop_packet(rgb, xyz, valid, box, config.image_size)
    x0, y0, x1, y1 = box
    packet["mask"] = cv2.resize(mask[y0:y1, x0:x1].astype(np.uint8),
                               (config.image_size, config.image_size),
                               interpolation=cv2.INTER_NEAREST)[None]
    packet.update(origin_m=origin, scale_m=scale)
    if config.memory_mode != "observed_only":
        cells = config.latent_side ** 3
        z, positions = np.asarray(shape_latent), np.asarray(latent_xyz_m)
        if z.shape != (cells, config.latent_channels) or positions.shape != (cells, 3):
            raise ValueError("spatial memory requires aligned shape_latent and latent_xyz_m")
        if z.dtype.kind not in "uif" or positions.dtype.kind not in "uif":
            raise ValueError("shape latent and positions must be numeric")
        z, positions = z.astype(np.float32), positions.astype(np.float32)
        if not np.isfinite(z).all() or not np.isfinite(positions).all():
            raise ValueError("shape latent and positions must be finite")
        packet.update(shape_latent=z, latent_xyz_m=positions)
        if config.shape_geometry_channels:
            field = np.asarray(shape_geometry, dtype=np.float32)
            if field.shape != (cells, config.shape_geometry_channels) or not np.isfinite(field).all():
                raise ValueError("shape_geometry must match latent cells and configured channels")
            if (field < 0).any() or (field > 1).any() or (field[:, 1] > field[:, 0]+1e-6).any():
                raise ValueError("invalid occupancy/boundary fractions")
            packet["shape_geometry"] = field
    return packet
