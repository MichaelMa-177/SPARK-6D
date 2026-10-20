"""Small object-disjoint development curriculum with multiple training references.

This uses the public YCB-V test partition for CUSTOM DEVELOPMENT, not an official
BOP benchmark. Held-out objects/scenes are reserved before training starts.
Masks are dataset annotations, never represented as human/SAM prompts.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .bop import annotation_pose, cad_from_memory, object_annotation
from .config import ModelConfig
from .data import sha256
from .features import FrozenDINO
from .rgbd import prepare_query, prepare_reference


TRAIN_OBJECTS = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 16, 17, 19]
TRAIN_SCENES = list(range(49, 57))
VALIDATION = {7: [58], 8: [58], 18: [59]}
RESERVED_TEST = {20: [57], 21: [57]}


def symmetry_matrices(info, fixed):
    """BOP millimetre symmetries -> memory gauge, continuous sampled every 5°."""
    matrices = [np.eye(4)]
    for value in info.get("symmetries_discrete", []):
        transform = np.asarray(value, dtype=np.float64).reshape(4, 4).copy()
        transform[:3, 3] /= 1000
        matrices.append(transform)
    for record in info.get("symmetries_continuous", []):
        axis = np.array(record["axis"], dtype=np.float64)
        axis /= np.linalg.norm(axis)
        offset = np.array(record["offset"], dtype=np.float64)/1000
        skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rotations = []
        for angle in np.arange(72) * (2*np.pi/72):
            rotation = np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*(skew@skew)
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = offset - rotation@offset
            rotations.extend([transform@discrete for discrete in matrices])
        matrices = rotations
    return np.stack([np.linalg.inv(fixed) @ s @ fixed for s in matrices]).astype(np.float32)


def reference_box(mask):
    y, x = np.where(mask)
    pad = int(max(np.ptp(x), np.ptp(y)) * .25) + 4
    return (max(0, int(x.min())-pad), max(0, int(y.min())-pad),
            min(mask.shape[1], int(x.max())+pad+1), min(mask.shape[0], int(y.max())+pad+1))


def evenly(rows, count):
    ids = np.linspace(0, len(rows)-1, min(count, len(rows))).round().astype(int)
    return [rows[i] for i in ids]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--models-info", required=True)
    p.add_argument("--dino-repo", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--queries-per-object", type=int, default=32)
    p.add_argument("--training-references", type=int, default=3)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.set_num_threads(4)
    root, output = Path(args.data_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = ModelConfig(memory_mode="observed_only", foundation_feature_dim=1024, solver_mode="robust")
    models = json.loads(Path(args.models_info).read_text())
    policy = {
        "schema_version": 1, "purpose": "custom object-disjoint development, not official BOP test",
        "training_objects": TRAIN_OBJECTS, "training_scenes": TRAIN_SCENES,
        "validation": VALIDATION, "reserved_test_not_exported": RESERVED_TEST,
        "mask_source": "official BOP mask_visib annotation", "SAM_executed": False,
        "initialization": "fresh pose network; frozen pretrained DINOv2",
        "test_object_CAD_used_in_training": False, "reference_policy": "first acceptable for validation; evenly spaced training references",
        "query_policy": "evenly spaced acceptable queries, excluding all selected reference RGB frames",
        "continuous_symmetry_discretization_deg": 5,
        "models_info_sha256": sha256(args.models_info), "dino_weights_sha256": sha256(args.weights),
        "source_sha256": sha256(__file__), "args": vars(args),
    }
    # Freeze the split and sampling policy before generating features or training.
    (output/"policy.json").write_text(json.dumps(policy, indent=2)+"\n")
    extractor = FrozenDINO(args.dino_repo, args.weights, args.device)
    metadata, feature_cache, manifests = {}, {}, {}
    for scene_id in sorted(set(TRAIN_SCENES + [58, 59])):
        scene = root/f"{scene_id:06d}"
        metadata[scene_id] = (json.loads((scene/"scene_camera.json").read_text()),
                              json.loads((scene/"scene_gt.json").read_text()))

    def observation(scene_id, frame_id, object_id):
        scene = root/f"{scene_id:06d}"
        camera = metadata[scene_id][0][str(frame_id)]
        index, annotation = object_annotation(metadata[scene_id][1][str(frame_id)], object_id)
        image_path = scene/"rgb"/f"{frame_id:06d}.png"
        rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(scene/"depth"/image_path.name), -1)
        mask = cv2.imread(str(scene/"mask_visib"/f"{frame_id:06d}_{index:06d}.png"), 0) > 0
        return rgb, depth, mask, np.array(camera["cam_K"]).reshape(3, 3), camera["depth_scale"]/1000, annotation, image_path

    def features(rgb, scene_id, frame_id, box):
        key = (scene_id, frame_id, tuple(box))
        if key not in feature_cache:
            x0, y0, x1, y1 = box
            feature_cache[key] = extractor(rgb[y0:y1, x0:x1])
        return feature_cache[key]

    for split, objects in (("train", {o: TRAIN_SCENES for o in TRAIN_OBJECTS}), ("validation", VALIDATION)):
        samples = []
        for object_id, scene_ids in objects.items():
            candidates = []
            for scene_id in scene_ids:
                for frame in sorted(metadata[scene_id][1], key=int):
                    annotations = metadata[scene_id][1][frame]
                    matches = [i for i, a in enumerate(annotations) if a["obj_id"] == object_id]
                    if len(matches) != 1:
                        continue
                    mask = cv2.imread(str(root/f"{scene_id:06d}"/"mask_visib"/f"{int(frame):06d}_{matches[0]:06d}.png"), 0)
                    small = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
                    if (small[::4, ::4] > 0).sum() >= 8:
                        candidates.append((scene_id, int(frame)))
            if len(candidates) < 4:
                raise ValueError(f"insufficient visible examples for object {object_id}")
            refs = evenly(candidates, args.training_references) if split == "train" else candidates[:1]
            queries = evenly([x for x in candidates if x not in refs], args.queries_per_object)
            for reference_id, (scene_id, frame_id) in enumerate(refs):
                rgb, depth, mask, k, depth_scale, annotation, image_path = observation(scene_id, frame_id, object_id)
                reference = prepare_reference(rgb, depth, mask, k, depth_scale, config)
                box = reference_box(mask)
                reference["foundation_features"] = features(rgb, scene_id, frame_id, box)
                fixed = cad_from_memory(annotation_pose(annotation), reference["origin_m"])
                name = f"references/{object_id:06d}_{reference_id:03d}.npz"
                (output/name).parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output/name, **reference)
                symmetries = symmetry_matrices(models[str(object_id)], fixed)
                for query_scene, query_frame in queries:
                    qrgb, qdepth, qmask, qk, qscale, qann, qpath = observation(query_scene, query_frame, object_id)
                    packet = prepare_query(qrgb, qdepth, qk, qscale, config)
                    packet["target_mask"] = (cv2.resize(qmask.astype(np.uint8), (128, 128),
                                                        interpolation=cv2.INTER_NEAREST)[None]).astype(np.float32)
                    if (packet["target_mask"][:, ::4, ::4]*packet["valid"][:, ::4, ::4]).sum() < 6:
                        continue
                    qbox = (0, 0, qrgb.shape[1], qrgb.shape[0])
                    packet.update(foundation_features=features(qrgb, query_scene, query_frame, qbox),
                                  pose=(annotation_pose(qann)@fixed).astype(np.float32),
                                  symmetry_transforms=symmetries)
                    qname = f"queries/{object_id:06d}_{reference_id:03d}_{query_scene:06d}_{query_frame:06d}.npz"
                    (output/qname).parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(output/qname, **packet)
                    samples.append(dict(reference=name, query=qname, object_id=f"ycbv:{object_id:06d}",
                                        scene_id=query_scene, frame_id=query_frame, rgb_sha256=sha256(qpath),
                                        reference_scene=scene_id, reference_frame=frame_id,
                                        reference_rgb_sha256=sha256(image_path), cad_from_memory=fixed.tolist(),
                                        box_xyxy=list(qbox)))
            print(json.dumps({"split": split, "object_id": object_id, "pairs_so_far": len(samples)}), flush=True)
        document = dict(schema_version=1, split=split, provenance=policy, samples=samples)
        (output/f"{split}.json").write_text(json.dumps(document, indent=2)+"\n")
        manifests[split] = samples
    # Object and scene separation are deliberate, not a random per-frame split.
    train_images = {r["rgb_sha256"] for r in manifests["train"]}
    val_images = {r["rgb_sha256"] for r in manifests["validation"]}
    if train_images & val_images:
        raise RuntimeError("training and validation RGB images overlap")
    (output/"complete.json").write_text(json.dumps({k: len(v) for k, v in manifests.items()})+"\n")


if __name__ == "__main__":
    main()
