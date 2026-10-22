"""Export already predicted memory poses for generated-mesh refinement, without GT."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .config import ModelConfig
from .data import sha256
from .rgbd import prepare_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--evaluation", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    manifest, evaluation, inputs, output = map(Path, (args.manifest, args.evaluation, args.inputs, args.output))
    document = json.loads(manifest.read_text())
    report = json.loads((evaluation/"report.json").read_text())
    if report["manifest_sha256"] != sha256(manifest):
        raise ValueError("predictions belong to a different manifest")
    rows = document["samples"]
    references = {str((manifest.parent/r["reference"]).resolve()) for r in rows}
    if len(references) != 1:
        raise ValueError("one fixed reference is required")
    reference_path = Path(next(iter(references)))
    reference_hash = sha256(reference_path)
    if any(r["reference_sha256"] != reference_hash for r in rows):
        raise ValueError("reference packet differs from prediction protocol")
    camera = json.loads((inputs/"camera.json").read_text())
    reference = prepare_reference(
        cv2.cvtColor(cv2.imread(str(inputs/"first_frame.png")), cv2.COLOR_BGR2RGB),
        cv2.imread(str(inputs/"first_depth.png"), -1), cv2.imread(str(inputs/"mask.png"), 0),
        np.asarray(camera["cam_K"]).reshape(3, 3), camera["depth_scale_m"],
        ModelConfig(memory_mode="observed_only"))
    with np.load(reference_path, allow_pickle=False) as saved:
        for key, value in reference.items():
            if np.asarray(value).dtype.kind in "SUbiu":
                np.testing.assert_array_equal(saved[key], value,
                                               err_msg=f"reference differs: {key}")
            else:
                np.testing.assert_allclose(saved[key], value, atol=1e-6, rtol=1e-6,
                                           err_msg=f"reference differs: {key}")
    with np.load(evaluation/"predictions.npz", allow_pickle=False) as data:
        poses, valid = data["poses"].copy(), data["valid"].copy()
    if poses.shape != (len(rows), 4, 4) or valid.shape != (len(rows),):
        raise ValueError("prediction count differs from manifest")
    output.mkdir(parents=True, exist_ok=False)
    for scene in sorted({r["scene_id"] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r["scene_id"] == scene]
        np.savez_compressed(output/f"{scene:06d}.npz", poses=poses[indices], valid=valid[indices],
                            frame_ids=np.array([rows[i]["frame_id"] for i in indices]),
                            memory_origin_m=reference["origin_m"],
                            reference_rgb_sha256=np.array(sha256(inputs/"first_frame.png")))
    (output/"report.json").write_text(json.dumps({
        "query_GT_or_CAD_read": False, "reference_reconstructed_and_verified": True,
        "manifest_sha256": sha256(manifest), "prediction_sha256": sha256(evaluation/"predictions.npz"),
        "reference_packet_sha256": reference_hash, "code_sha256": sha256(__file__),
    }, indent=2)+"\n")


if __name__ == "__main__":
    main()
