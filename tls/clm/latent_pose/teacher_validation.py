"""Calibrate the teacher, export development pseudo-labels, and score three ways."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .data import read_packet, sha256
from .evaluate import pose_errors


def teacher_data(directory):
    directory = Path(directory)
    report = json.loads((directory/"report.json").read_text())
    if report["object_id"] != 15:
        raise ValueError("this development protocol currently supports asymmetric object 15")
    if report["input_sampling"] != "consecutive frames" or not report["GT_loaded_after_predictions"]:
        raise ValueError("only the audited consecutive-frame teacher is accepted")
    with np.load(directory/"predictions.npz", allow_pickle=False) as packet:
        poses = dict(zip(packet["frame_ids"].tolist(), packet["poses"]))
    records = {r["frame_id"]: r for r in report["records"]}
    if set(poses) != set(records):
        raise ValueError("teacher prediction and provenance frame mismatch")
    return poses, records, report


def checked_teacher(row, poses, records, reference_scene):
    if row["object_id"] != "ycbv:000015" or row["scene_id"] != reference_scene:
        raise ValueError("teacher does not cover this object/scene")
    frame = row["frame_id"]
    if row["rgb_sha256"] != records[frame]["rgb_sha256"]:
        raise ValueError("teacher/query RGB identity mismatch")
    return poses[frame]


def export(args):
    root, output = Path(args.packet_root), Path(args.output)
    train = json.loads((root/"fit_train.json").read_text())
    val = json.loads((root/"fit_validation.json").read_text())
    provenance = train["provenance"]
    poses, records, report = teacher_data(args.teacher)
    frame = provenance["reference_frame"]
    anchor = np.eye(4)
    anchor[:3, 3] = provenance["reference_origin_m"]
    pseudo_cad_from_memory = np.linalg.inv(poses[frame]) @ anchor
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(root/"reference.npz", output/"reference.npz")
    for document, filename in ((train, "train.json"), (val, "validation.json")):
        training = filename == "train.json"
        for row in document["samples"]:
            teacher_pose = checked_teacher(row, poses, records, provenance["reference_scene"])
            target = output/row["query"]
            target.parent.mkdir(parents=True, exist_ok=True)
            with np.load(root/row["query"], allow_pickle=False) as packet:
                copied = {k: packet[k].copy() for k in packet.files}
            if training:
                copied["pose"] = (teacher_pose@pseudo_cad_from_memory).astype(np.float32)
            np.savez_compressed(target, **copied)
            row["query_sha256"] = sha256(target)
        document["provenance"] = {
            **provenance,
            "pose_label_source": "FoundationPose relative to its first-frame estimate" if training else "official BOP GT",
            "target_mask_source": "official BOP mask_visib supervision",
            "teacher_predictions_sha256": sha256(Path(args.teacher)/"predictions.npz"),
            "teacher_report_sha256": sha256(Path(args.teacher)/"report.json"),
            "teacher_calibration_pass_fraction": report["pass_10mm_5deg_all_frames"],
            "per_frame_GT_filtering": False,
            "pseudo_cad_from_memory": pseudo_cad_from_memory.tolist(),
        }
        (output/filename).write_text(json.dumps(document, indent=2)+"\n")
    print(json.dumps({"training_pairs": len(train["samples"]), "development_pairs": len(val["samples"])}))


def summarize(errors, validity):
    valid = [x for x, ok in zip(errors, validity) if ok]
    return {
        "frames": len(errors), "valid_fraction": float(np.mean(validity)),
        "pass_10mm_5deg_all_frames": float(np.mean([x["pass_10mm_5deg"] and ok for x, ok in zip(errors, validity)])),
        "rotation_median_deg_valid": float(np.median([x["rotation_deg"] for x in valid])) if valid else None,
        "translation_median_mm_valid": float(np.median([x["translation_mm_at_best_rotation"] for x in valid])) if valid else None,
    }


def compare(args):
    path, output = Path(args.manifest), Path(args.output)
    document = json.loads(path.read_text())
    evaluation = json.loads((Path(args.student)/"report.json").read_text())
    if evaluation["manifest_sha256"] != sha256(path) or not evaluation["GT_loaded_after_predictions"]:
        raise ValueError("student predictions were not independently evaluated on this exact manifest")
    with np.load(Path(args.student)/"predictions.npz", allow_pickle=False) as saved:
        students, validity = saved["poses"], saved["valid"]
    if len(students) != len(document["samples"]):
        raise ValueError("student prediction count mismatch")
    teacher, records, _ = teacher_data(args.teacher)
    p = document["provenance"]
    fixed = np.array(p["cad_from_memory"])
    anchor = np.eye(4)
    anchor[:3, 3] = p["reference_origin_m"]
    teacher_fixed = np.linalg.inv(teacher[p["reference_frame"]]) @ anchor
    result = {name: [] for name in ("student_vs_GT_CAD", "teacher_vs_GT_CAD",
                                   "student_vs_teacher_CAD", "teacher_relative_memory_vs_GT_memory")}
    for row, student in zip(document["samples"], students):
        fp = checked_teacher(row, teacher, records, p["reference_scene"])
        label = read_packet(path.parent/row["query"], labels=True)
        gt_memory = label["pose"].numpy()
        student_cad, gt_cad = student @ np.linalg.inv(fixed), gt_memory @ np.linalg.inv(fixed)
        for key, (pred, truth) in zip(result, ((student_cad, gt_cad), (fp, gt_cad),
                                              (student_cad, fp), (fp@teacher_fixed, gt_memory))):
            result[key].append(pose_errors(pred, truth, np.eye(4)[None]))
    summary = {}
    for key, errors in result.items():
        ok = validity if key.startswith("student") else np.ones(len(students), dtype=bool)
        summary[key] = summarize(errors, ok)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(summary=summary, per_frame=result, manifest_sha256=sha256(path),
                  student_report_sha256=sha256(Path(args.student)/"report.json"),
                  teacher_predictions_sha256=sha256(Path(args.teacher)/"predictions.npz"),
                  coordinate_protocol="One fixed GT reference CAD mapping for evaluation only; no trajectory alignment",
                  split=document["split"], reference_pose_alignment="fixed once; never per-query")
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export")
    e.add_argument("--packet-root", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--manifest", required=True)
    c.add_argument("--student", required=True)
    for command in (e, c):
        command.add_argument("--teacher", required=True)
        command.add_argument("--output", required=True)
    args = p.parse_args()
    (export if args.command == "export" else compare)(args)


if __name__ == "__main__":
    main()
