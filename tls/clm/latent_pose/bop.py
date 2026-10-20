"""Export supervised BOP pairs in the fixed single-reference memory gauge."""

import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from .cli import write_json
from .config import ModelConfig
from .data import sha256
from .rgbd import prepare_query


def annotation_pose(annotation):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(annotation["cam_R_m2c"]).reshape(3, 3)
    pose[:3, 3] = np.asarray(annotation["cam_t_m2c"]) / 1000.
    if not np.isfinite(pose).all() or not np.allclose(pose[:3, :3].T @ pose[:3, :3],
                                                    np.eye(3), atol=1e-4):
        raise ValueError("invalid BOP rigid pose")
    if np.linalg.det(pose[:3, :3]) < .999:
        raise ValueError("BOP pose contains a reflection")
    return pose


def cad_from_memory(reference_pose, reference_origin_m):
    """Memory axes equal reference camera axes; its origin is the measured centroid."""
    reference_from_memory = np.eye(4, dtype=np.float64)
    reference_from_memory[:3, 3] = reference_origin_m
    return np.linalg.inv(reference_pose) @ reference_from_memory


def object_annotation(rows, object_id):
    matches = [(i, row) for i, row in enumerate(rows) if row["obj_id"] == object_id]
    if len(matches) != 1:
        raise ValueError("expected exactly one selected object instance")
    return matches[0]


def export(args):
    root, output = Path(args.data_root), Path(args.output)
    ref_scene = root / f"{args.reference_scene:06d}"
    ref_gt_path = ref_scene / "scene_gt.json"
    _, ref_annotation = object_annotation(json.loads(ref_gt_path.read_text())[str(args.reference_frame)],
                                         args.object_id)
    with np.load(args.reference, allow_pickle=False) as packet:
        origin = packet["origin_m"].copy()
    fixed = cad_from_memory(annotation_pose(ref_annotation), origin)
    config = ModelConfig(image_size=args.image_size)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.reference, output / "reference.npz")
    (output / "queries").mkdir()
    rows, sources, counts = [], {}, []
    for scene_id in args.scenes:
        scene = root / f"{scene_id:06d}"
        cameras = json.loads((scene / "scene_camera.json").read_text())
        gt = json.loads((scene / "scene_gt.json").read_text())
        sources[str(scene_id)] = {name: sha256(scene / name)
                                 for name in ("scene_gt.json", "scene_camera.json")}
        for rgb_path in sorted((scene / "rgb").glob("*.png")):
            frame_id = int(rgb_path.stem)
            if (scene_id, frame_id) == (args.reference_scene, args.reference_frame):
                continue  # reference observation never counted as a query
            idx, annotation = object_annotation(gt[str(frame_id)], args.object_id)
            mask_path = scene / "mask_visib" / f"{frame_id:06d}_{idx:06d}.png"
            mask = cv2.imread(str(mask_path), 0)
            if mask is None:
                raise FileNotFoundError(mask_path)
            mask = mask > 0
            rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(str(scene / "depth" / rgb_path.name), -1)
            camera = cameras[str(frame_id)]
            h, w = depth.shape
            box = (0, 0, w, h)
            if args.roi == "gt-box-diagnostic":
                y, x = np.where(mask)
                pad = int(max(np.ptp(x), np.ptp(y)) * .3) + 4
                box = (max(0, int(x.min()) - pad), max(0, int(y.min()) - pad),
                       min(w, int(x.max()) + pad + 1), min(h, int(y.max()) + pad + 1))
            packet = prepare_query(rgb, depth, np.asarray(camera["cam_K"]).reshape(3, 3),
                                   camera["depth_scale"] / 1000, config, box=box)
            x0, y0, x1, y1 = box
            target_mask = cv2.resize(mask[y0:y1, x0:x1].astype(np.uint8),
                                     (args.image_size, args.image_size),
                                     interpolation=cv2.INTER_NEAREST)[None]
            packet.update(pose=(annotation_pose(annotation) @ fixed).astype(np.float32),
                          target_mask=target_mask)
            foreground = int(((target_mask[0, ::4, ::4] > 0) &
                              (packet["valid"][0, ::4, ::4] > 0)).sum())
            if foreground < 3:
                raise ValueError(f"too few visible target grid points: {scene_id}/{frame_id}")
            name = f"queries/{scene_id:06d}_{frame_id:06d}.npz"
            np.savez_compressed(output / name, **packet)
            rows.append({
                "reference": "reference.npz", "query": name,
                "object_id": f"ycbv:{args.object_id:06d}", "scene_id": scene_id,
                "frame_id": frame_id, "rgb_sha256": sha256(rgb_path),
                "query_sha256": sha256(output / name), "box_xyxy": list(box),
            })
            counts.append(foreground)
    provenance = {
        "dataset": "BOP YCB-V", "label_source": "official_scene_gt",
        "purpose": "development experiment on previously inspected scenes, not official benchmark",
        "query_roi": args.roi, "reference_sha256": sha256(args.reference),
        "reference_scene": args.reference_scene, "reference_frame": args.reference_frame,
        "reference_origin_m": origin.tolist(), "cad_from_memory": fixed.tolist(),
        "reference_gt_sha256": sha256(ref_gt_path), "scene_metadata_sha256": sources,
        "GT_read_for_export": True, "GT_query_mask_is_supervision": True,
        "new_SAM_segmentation": False, "temporal_sampling": "downloaded sparse keyframes",
        "foreground_grid_points_min_median_max": [min(counts), float(np.median(counts)), max(counts)],
    }
    def manifest(name, split, selected):
        if selected:
            write_json(output / name, dict(schema_version=1, split=split,
                                           provenance=provenance, samples=selected))
    # All are explicitly development protocols; never claim unseen-object performance.
    reference_scene_rows = [r for r in rows if r["scene_id"] == args.reference_scene]
    fit_train = [r for i, r in enumerate(reference_scene_rows) if i % 4 != 0][:32]
    fit_validation = [r for i, r in enumerate(reference_scene_rows) if i % 4 == 0]
    manifest("fit_train.json", "train", fit_train)
    manifest("fit_validation.json", "development", fit_validation)
    manifest("cross_scene_train.json", "train", [r for r in rows if r["scene_id"] != args.reference_scene])
    manifest("cross_scene_development.json", "development", reference_scene_rows)
    manifest("all_development.json", "development", rows)
    write_json(output / "provenance.json", provenance)
    print(json.dumps({"pairs": len(rows), "fit_train": len(fit_train),
                      "fit_validation": len(fit_validation), "output": str(output),
                      "foreground_grid_points": provenance["foreground_grid_points_min_median_max"]}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True, help="BOP test scenes root; development use is recorded")
    p.add_argument("--reference", required=True, help="verified reference NPZ")
    p.add_argument("--reference-scene", type=int, default=59)
    p.add_argument("--reference-frame", type=int, default=1)
    p.add_argument("--object-id", type=int, default=15)
    p.add_argument("--scenes", nargs="+", type=int, default=[50, 54, 59])
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--roi", choices=["full", "gt-box-diagnostic"], default="full")
    p.add_argument("--output", required=True)
    export(p.parse_args())


if __name__ == "__main__":
    main()
