"""Official BOP surface/projection errors on a declared development subset.

This does not claim an official full-dataset BOP score.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from .data import read_packet, sha256


def summarize(rows):
    mssd_thresholds = np.arange(.05, .51, .05)
    mspd_thresholds = np.arange(5, 51, 5)
    mssd = np.array([r["mssd_diameter"] if r["pose_valid"] else np.inf for r in rows])
    mspd = np.array([r["mspd_px_at_width640"] if r["pose_valid"] else np.inf for r in rows])
    r_mssd = (mssd[:, None] < mssd_thresholds).mean(0)
    r_mspd = (mspd[:, None] < mspd_thresholds).mean(0)
    result = dict(frames=len(rows), valid_fraction=float(np.mean([r["pose_valid"] for r in rows])),
                MSSD_recall_thresholds=mssd_thresholds.tolist(), MSSD_recalls=r_mssd.tolist(),
                MSPD_recall_thresholds_px=mspd_thresholds.tolist(), MSPD_recalls=r_mspd.tolist(),
                mean_recall_MSSD=float(r_mssd.mean()), mean_recall_MSPD=float(r_mspd.mean()))
    if "vsd" in rows[0]:
        values = np.array([r["vsd"] if r["pose_valid"] else np.ones(10) for r in rows])
        recalls = (values[:, :, None] < mssd_thresholds).mean(0)
        result.update(VSD_taus=mssd_thresholds.tolist(), VSD_correct_thresholds=mssd_thresholds.tolist(),
                      VSD_recalls=recalls.tolist(), mean_recall_VSD=float(recalls.mean()))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--predictions", required=True, help="directory written by latent_pose.evaluate")
    p.add_argument("--models", required=True, help="official BOP models_eval directory; millimetres")
    p.add_argument("--data-root", required=True)
    p.add_argument("--toolkit", required=True, help="clean, pinned official bop_toolkit checkout")
    p.add_argument("--output", required=True)
    p.add_argument("--vsd", action="store_true", help="official vispy renderer and YCB-V delta=15 mm")
    args = p.parse_args()
    toolkit = Path(args.toolkit).resolve()
    revision = subprocess.check_output(["git", "-C", str(toolkit), "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(["git", "-C", str(toolkit), "diff", "--exit-code", "HEAD"], check=True,
                   stdout=subprocess.DEVNULL)
    sys.path.insert(0, str(toolkit))
    from bop_toolkit_lib import misc, pose_error
    import trimesh
    path, predictions, models = Path(args.manifest), Path(args.predictions), Path(args.models)
    manifest = json.loads(path.read_text())
    audited = json.loads((predictions/"report.json").read_text())
    if audited["manifest_sha256"] != sha256(path) or not audited["GT_loaded_after_predictions"]:
        raise ValueError("prediction audit does not match this manifest")
    with np.load(predictions/"predictions.npz", allow_pickle=False) as data:
        poses, valid = data["poses"], data["valid"]
    if len(poses) != len(manifest["samples"]):
        raise ValueError("prediction count differs from manifest")
    info = json.loads((models/"models_info.json").read_text())
    meshes, symmetries, cameras, rows = {}, {}, {}, []
    renderers = {}
    for index, (row, pose, ok) in enumerate(zip(manifest["samples"], poses, valid)):
        object_id = int(row["object_id"].split(":")[1])
        label = read_packet(path.parent/row["query"], labels=True)
        fixed = np.array(row.get("cad_from_memory", manifest["provenance"].get("cad_from_memory")))
        if fixed.shape != (4, 4):
            raise ValueError("a fixed reference-to-CAD evaluation mapping is required")
        estimate, truth = pose @ np.linalg.inv(fixed), label["pose"].numpy() @ np.linalg.inv(fixed)
        estimate[:3, 3] *= 1000
        truth[:3, 3] *= 1000
        if object_id not in meshes:
            mesh_path = models/f"obj_{object_id:06d}.ply"
            meshes[object_id] = np.asarray(trimesh.load(mesh_path, force="mesh", process=False).vertices)
            symmetries[object_id] = misc.get_symmetry_transformations(info[str(object_id)], .01)
            identity_error = pose_error.mssd(truth[:3, :3], truth[:3, 3:4], truth[:3, :3],
                                             truth[:3, 3:4], meshes[object_id], symmetries[object_id])
            if identity_error > 1e-6:
                raise RuntimeError("official error identity check failed")
        scene = Path(args.data_root)/f'{row["scene_id"]:06d}'
        if row["scene_id"] not in cameras:
            cameras[row["scene_id"]] = json.loads((scene/"scene_camera.json").read_text())
        k = np.array(cameras[row["scene_id"]][str(row["frame_id"])]["cam_K"]).reshape(3, 3)
        height, width = cv2.imread(str(scene/"rgb"/f'{row["frame_id"]:06d}.png')).shape[:2]
        if args.vsd and (width, height) not in renderers:
            from bop_toolkit_lib.rendering.renderer import create_renderer
            renderers[width, height] = (create_renderer(width, height, "vispy", mode="depth"), set())
        if args.vsd:
            renderer, added = renderers[width, height]
            if object_id not in added:
                renderer.add_object(object_id, str(models/f"obj_{object_id:06d}.ply"))
                added.add(object_id)
        record = dict(index=index, object_id=row["object_id"], pose_valid=bool(ok),
                      scene_id=row["scene_id"], frame_id=row["frame_id"])
        if ok:
            mssd = pose_error.mssd(estimate[:3, :3], estimate[:3, 3:4], truth[:3, :3],
                                   truth[:3, 3:4], meshes[object_id], symmetries[object_id])
            mspd = pose_error.mspd(estimate[:3, :3], estimate[:3, 3:4], truth[:3, :3],
                                   truth[:3, 3:4], k, meshes[object_id], symmetries[object_id])
            record.update(mssd_mm=float(mssd), mssd_diameter=float(mssd/info[str(object_id)]["diameter"]),
                          mspd_px=float(mspd), mspd_px_at_width640=float(mspd*640/width))
        else:
            record.update(mssd_mm=None, mssd_diameter=None, mspd_px=None, mspd_px_at_width640=None)
        if args.vsd:
            camera = cameras[row["scene_id"]][str(row["frame_id"])]
            depth = cv2.imread(str(scene/"depth"/f'{row["frame_id"]:06d}.png'), -1).astype(np.float32)
            depth *= camera["depth_scale"]  # BOP raw sensor units -> millimetres
            if ok:
                record["vsd"] = pose_error.vsd(
                    estimate[:3, :3], estimate[:3, 3:4], truth[:3, :3], truth[:3, 3:4],
                    depth, k, 15, list(np.arange(.05, .51, .05)), True,
                    info[str(object_id)]["diameter"], renderer, object_id)
            else:
                record["vsd"] = None
        rows.append(record)
    per_object = {obj: summarize([r for r in rows if r["object_id"] == obj])
                  for obj in sorted({r["object_id"] for r in rows})}
    report = dict(
        protocol="official BOP error implementations on custom development subset, not an official benchmark submission",
        VSD_computed=args.vsd, VSD_delta_mm=15 if args.vsd else None,
        split=manifest["split"], toolkit_revision=revision,
        toolkit_source_sha256={name: sha256(toolkit/"bop_toolkit_lib"/name)
                               for name in ("pose_error.py", "misc.py", "transform.py")},
        manifest_sha256=sha256(path), predictions_sha256=sha256(predictions/"predictions.npz"),
        models_info_sha256=sha256(models/"models_info.json"),
        model_sha256={str(obj): sha256(models/f"obj_{obj:06d}.ply") for obj in meshes},
        max_sym_disc_step=.01, translation_units="millimetres for official toolkit",
        all_queries=summarize(rows), per_object=per_object,
        macro_mean_recall_MSSD=float(np.mean([x["mean_recall_MSSD"] for x in per_object.values()])),
        macro_mean_recall_MSPD=float(np.mean([x["mean_recall_MSPD"] for x in per_object.values()])),
        records=rows,
    )
    if args.vsd:
        report["macro_mean_recall_VSD"] = float(np.mean([x["mean_recall_VSD"] for x in per_object.values()]))
        report["macro_mean_three_error_recall"] = float(np.mean([
            report["macro_mean_recall_MSSD"], report["macro_mean_recall_MSPD"], report["macro_mean_recall_VSD"]]))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k in
                     {"macro_mean_recall_MSSD", "macro_mean_recall_MSPD", "macro_mean_recall_VSD",
                      "macro_mean_three_error_recall", "protocol"}}))


if __name__ == "__main__":
    main()
