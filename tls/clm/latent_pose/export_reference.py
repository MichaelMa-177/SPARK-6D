"""Reuse the exact saved scene59 user prompt; export no new segmentation."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .data import sha256


def lift(depth, intrinsics, scale):
    z = depth.astype(np.float32) * scale
    y, x = np.indices(z.shape, dtype=np.float32)
    xyz = np.stack(((x - intrinsics[0, 2]) * z / intrinsics[0, 0],
                    (y - intrinsics[1, 2]) * z / intrinsics[1, 1], z), -1)
    valid = np.isfinite(xyz).all(-1) & (z > .05) & (z < 3)
    xyz[~valid] = 0
    return xyz, valid


def crop_packet(rgb, xyz, valid, box, size):
    x0, y0, x1, y1 = box
    rgb = cv2.resize(rgb[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
    xyz = cv2.resize(xyz[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_NEAREST)
    valid = cv2.resize(valid[y0:y1, x0:x1].astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return dict(schema_version=np.int64(1), units=np.str_("metres"),
                rgb=rgb.transpose(2, 0, 1).astype(np.float32) / 255,
                xyz=xyz.transpose(2, 0, 1), valid=valid[None].astype(np.float32))


def sam3d_to_camera(canonical, scale, quaternion, translation):
    """Match this project's PyTorch3D row-vector compose_transform exactly."""
    q = quaternion.reshape(4).astype(np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    rotation = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
    ])
    points = (canonical * scale.reshape(1, 3)) @ rotation + translation.reshape(1, 3)
    points[:, :2] *= -1
    return points.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", default=str(Path(__file__).resolve().parents[1] /
                                          "experiments/first-memory-fast-refine/inputs"))
    p.add_argument("--prior", required=True)
    p.add_argument("--scene", required=True, help="BOP scene59 directory; only RGB/depth/cameras are read")
    p.add_argument("--output", required=True)
    p.add_argument("--image-size", type=int, default=128)
    args = p.parse_args()
    inputs, prior_path, scene = Path(args.inputs), Path(args.prior), Path(args.scene)
    report = json.loads(prior_path.with_suffix(".json").read_text())
    prompt = json.loads((inputs / "user_prompt.json").read_text())
    camera = json.loads((inputs / "camera.json").read_text())
    if prompt["prompt_source"] != "human_interactive_click":
        raise ValueError("this exporter only reuses an existing human-selected reference")
    rgb_sha = sha256(inputs / "first_frame.png")
    if rgb_sha != prompt["rgb_sha256"] or rgb_sha != report["rgb_sha256"]:
        raise ValueError("reference image identity mismatch")
    if sha256(inputs / "mask.png") != report["human_mask_sha256"]:
        raise ValueError("saved confirmed mask mismatch")
    if sha256(prior_path) != report["file_sha256"]:
        raise ValueError("prior file checksum mismatch")
    frame = int(prompt["frame_id"])
    if sha256(scene / "rgb" / f"{frame:06d}.png") != rgb_sha:
        raise ValueError("scene does not match the saved human reference")
    cameras = json.loads((scene / "scene_camera.json").read_text())
    k = np.array(camera["cam_K"], np.float32).reshape(3, 3)
    record_camera = cameras[str(frame)]
    if not np.allclose(k.ravel(), record_camera["cam_K"]) or not np.isclose(
        camera["depth_scale_m"], record_camera["depth_scale"] / 1000
    ):
        raise ValueError("saved camera/depth scale differs from the source scene")
    rgb = cv2.cvtColor(cv2.imread(str(inputs / "first_frame.png")), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(inputs / "first_depth.png"), -1)
    if not np.array_equal(depth, cv2.imread(str(scene / "depth" / f"{frame:06d}.png"), -1)):
        raise ValueError("saved depth differs from source frame")
    mask = cv2.imread(str(inputs / "mask.png"), 0) > 0
    xyz, valid = lift(depth, k, camera["depth_scale_m"])
    points = xyz[mask & valid]
    if len(points) < 12:
        raise ValueError("insufficient measured reference geometry")
    origin = points.mean(0)
    scale = np.array([max(float(np.ptp(points, axis=0).max()), .01)], np.float32)
    y, x = np.where(mask)
    pad = int(max(x.max() - x.min(), y.max() - y.min()) * .25) + 4
    box = (max(0, x.min()-pad), max(0, y.min()-pad),
           min(rgb.shape[1], x.max()+pad+1), min(rgb.shape[0], y.max()+pad+1))
    ref = crop_packet(rgb, xyz, valid, box, args.image_size)
    x0, y0, x1, y1 = box
    ref["mask"] = cv2.resize(mask[y0:y1, x0:x1].astype(np.uint8),
                            (args.image_size, args.image_size), interpolation=cv2.INTER_NEAREST)[None]
    with np.load(prior_path, allow_pickle=False) as prior:
        z = prior["coarse_shape"][0].astype(np.float32)
        side = round(len(z) ** (1/3))
        if side ** 3 != len(z) or z.shape[1] != 8:
            raise ValueError("expected ordered cubic SS latent with 8 channels")
        grid = np.stack(np.meshgrid(*([np.arange(side)] * 3), indexing="ij"), -1).reshape(-1, 3)
        canonical = (grid.astype(np.float32) + .5) / side - .5
        layout = [prior["layout_" + key] for key in ("scale", "rotation", "translation")]
        generated = sam3d_to_camera(canonical, *layout)
        # Verify the coordinate convention against saved actual inference anchors.
        anchors = sam3d_to_camera(prior["boundary_coords"].astype(np.float32) / 64 - .5, *layout)
        error = float(np.abs(anchors - prior["points"]).max())
        if error > 1e-5:
            raise ValueError(f"SAM3D coordinate transform mismatch: {error} m")
    ref.update(origin_m=origin, scale_m=scale, shape_latent=z, latent_xyz_m=generated)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "reference.npz", **ref)
    # Full-frame query is intentional: no previous pose, GT crop or new prompt.
    queries = []
    for path in sorted((scene / "rgb").glob("*.png"))[1:4]:
        idx = int(path.stem)
        row = cameras[str(idx)]
        query_rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        query_depth = cv2.imread(str(scene / "depth" / path.name), -1)
        query_xyz, query_valid = lift(query_depth, np.array(row["cam_K"]).reshape(3, 3),
                                      row["depth_scale"] / 1000)
        h, w = query_depth.shape
        packet = crop_packet(query_rgb, query_xyz, query_valid, (0, 0, w, h), args.image_size)
        name = f"query_{idx:06d}.npz"
        np.savez_compressed(output / name, **packet)
        queries.append(name)
    provenance = {
        "source": "saved scene59 human reference", "split": "development",
        "prior_sha256": sha256(prior_path), "rgb_sha256": rgb_sha,
        "mask_sha256": sha256(inputs / "mask.png"),
        "depth_sha256": sha256(inputs / "first_depth.png"),
        "camera_sha256": sha256(inputs / "camera.json"), "user_prompt": prompt,
        "reference_box_xyxy": [int(v) for v in box], "query_ROI": "full frame",
        "layout_anchor_max_error_m": error, "latent_side": side,
        "latent_coordinates": "volume cell centers, not reconstructed surface points",
        "GT_read": False, "queries": queries,
        "files": {f.name: sha256(f) for f in output.glob("*.npz")},
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"output": str(output), "latent_tokens": len(z), "queries": queries,
                      "layout_anchor_max_error_m": error, "GT_read": False}))


if __name__ == "__main__":
    main()
