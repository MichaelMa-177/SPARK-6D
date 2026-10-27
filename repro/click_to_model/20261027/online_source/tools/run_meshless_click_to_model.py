#!/usr/bin/env python3
"""Single-frame click -> SAM2 -> SAM3D metric mesh -> causal 6D tracking.

No input mesh or input mask argument exists. Only the first RGB/depth pair and
K are exposed to initialization subprocesses. Largest admissible SAM2 component
is an explicitly ambiguous whole-object heuristic, not an identity guarantee.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from meshless_initialization import select_meshless_mask
from tools.generate_click_mesh import dependencies
from tools.run_click_to_model import (MEASURED_SOURCE_FILES, OUTPUT_POLICY_SOURCE_FILES, TENTATIVE_TRUST_FIELDS, add_tracking_policy_arguments,
                                     validate_tracking_policy_arguments)


def tracking_option_error(message):
    raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def first_observation(sequence, depth_scale, click):
    """Read only the first RGB/depth pair; filenames alone establish ordering."""
    sequence = Path(sequence)
    names = sorted(p.name for p in (sequence / "rgb").glob("*.png"))
    depth_names = sorted(p.name for p in (sequence / "depth").glob("*.png"))
    if not names or names != depth_names:
        raise ValueError("Require exactly matching RGB/depth PNG filenames")
    paths = {"rgb": sequence / "rgb" / names[0], "depth": sequence / "depth" / names[0],
             "intrinsics": sequence / "cam_K.txt"}
    color = cv2.imread(str(paths["rgb"]), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(paths["depth"]), cv2.IMREAD_UNCHANGED)
    if color is None or depth is None or depth.ndim != 2 or depth.dtype != np.uint16 or color.shape[:2] != depth.shape:
        raise ValueError("Require aligned RGB and uint16 single-channel depth PNGs")
    if not math.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale must be explicit positive metres per stored unit")
    xy = np.asarray(click, float)
    if (xy.shape != (2,) or not np.isfinite(xy).all() or
            not (0 <= xy[0] <= depth.shape[1] - 1 and 0 <= xy[1] <= depth.shape[0] - 1)):
        raise ValueError("Click must be finite and inside the RGB-D frame")
    K = np.loadtxt(paths["intrinsics"])
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1]) or not np.allclose([K[0, 1], K[1, 0]], 0)):
        raise ValueError("Invalid zero-skew pinhole camera matrix")
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB), depth.astype(np.float32) * depth_scale, paths, names


def segment_worker(args):
    """GPU is confined to this short-lived subprocess."""
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to(ROOT / "results"):
        raise ValueError("Output must be a new directory inside this repository's results/")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "meshless_segmentation/1", "status": "started", "click_xy": args.click,
              "supplied_mask_or_gt_read": False}
    started = time.perf_counter()
    try:
        rgb, depth, _, names = first_observation(args.sequence, args.depth_scale, args.click)
        if len(names) != 1:
            raise ValueError("Segmentation worker accepts exactly one isolated RGB-D frame")
        from tools.run_click_to_model import sam2_mask_candidates
        masks, scores = sam2_mask_candidates(rgb, args.click, args.sam2_checkpoint)
        selection, components = select_meshless_mask(masks, scores, depth, args.click,
                                                       ambiguity_policy=args.ambiguity_policy)
        for index, (raw, component) in enumerate(zip(masks, components)):
            for name, mask in (("raw", raw), ("component", component)):
                path = output / f"candidate_{index:02d}_{name}.png"
                if not cv2.imwrite(str(path), np.asarray(mask, np.uint8) * 255):
                    raise OSError(f"Cannot write {path}")
        save_json(output / "selection.json", selection)
        report["selection"] = selection
        selected = selection["selected_index"]
        if selected is None:
            raise ValueError(selection["status"])
        selected_path = output / "selected_mask.png"
        if not cv2.imwrite(str(selected_path), components[selected].astype(np.uint8) * 255):
            raise OSError("Cannot write selected mask")
        report.update(status="selected", selected_mask_sha256=digest(selected_path),
                      sam2_checkpoint_sha256=digest(args.sam2_checkpoint))
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        save_json(output / "report.json", report)
    return report


def launch_stage(command, log, env, timeout):
    """Own a process group so interrupted SAM3D grandchildren cannot leak GPU work."""
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        # The direct child can exit before a stubborn grandchild does.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise


def tracking_environment(python, output):
    """Use the explicitly selected existing environment without installing anything."""
    prefix = Path(python).expanduser().absolute().parent.parent
    env = os.environ.copy()
    env.update(CONDA_PREFIX=str(prefix), CUDA_HOME=str(prefix),
               PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENCV_FOR_THREADS_NUM="1",
               MPLCONFIGDIR=str(output / "cache/matplotlib"),
               TORCHINDUCTOR_CACHE_DIR=str(output / "cache/torchinductor"))
    env["PATH"] = str(prefix / "bin") + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(prefix / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "mycpp/build"), env.get("PYTHONPATH", "")])
    return env


def build_commands(args, output):
    validate_tracking_policy_arguments(args, tracking_option_error)
    python = str(Path(args.python).expanduser().absolute())
    first = output / "first_frame"
    selected_mask = output / "segmentation/selected_mask.png"
    commands = [
        ("segmentation", [python, str(Path(__file__).resolve()), "--stage", "segment",
            "--sequence", str(first), "--click", *map(str, args.click),
            "--depth-scale", str(args.depth_scale), "--sam2-checkpoint", str(Path(args.sam2_checkpoint).resolve()),
            "--ambiguity-policy", args.ambiguity_policy, "--output-dir", str(output / "segmentation")]),
        ("generation", [python, str(ROOT / "tools/generate_click_mesh.py"),
            "--rgb", str(first / "rgb/000000.png"), "--depth", str(first / "depth/000000.png"),
            "--mask", str(selected_mask), "--intrinsics", str(first / "cam_K.txt"),
            "--depth-scale", str(args.depth_scale), "--seed", str(args.seed),
            "--output", str(output / "generation"), "--python", python,
            "--click-to-model-root", str(Path(args.click_to_model_root).resolve()), "--timeout", str(args.timeout)]),
        ("tracking", [python, str(ROOT / "tools/run_click_to_model.py"),
            "--sequence", str(Path(args.sequence).resolve()), "--mesh", str(output / "generation/mesh/model.obj"),
            "--mask", str(selected_mask), "--depth-scale", str(args.depth_scale),
            "--seed", str(args.seed), "--mode", "adaptive", "--max-frames", str(args.max_frames),
            "--output-dir", str(output / "tracking")]),
    ]
    if args.tracking_policy == "tentative-preloss":
        commands[-1][1].extend([
            "--tracking-policy", args.tracking_policy,
            "--output-policy", getattr(args, "output_policy", "legacy"),
            "--working-mask-selection", args.working_mask_selection,
            "--initialization-admission", "all-hypotheses",
            "--tapnext-source-root", str(args.tapnext_source_root.resolve()),
            "--tapnext-checkpoint", str(args.tapnext_checkpoint.resolve()),
            "--tapnext-checkpoint-sha256", args.tapnext_checkpoint_sha256,
        ])
    return commands


def run(args):
    # Reject policy mistakes before SAM2/SAM3D dependency work or subprocesses.
    validate_tracking_policy_arguments(args, tracking_option_error)
    output = Path(args.output_dir).expanduser().resolve()
    if not output.is_relative_to(ROOT / "results"):
        raise ValueError("Output must be a new directory inside this repository's results/")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "meshless_click_to_model/1", "status": "started", "dry_run": args.dry_run,
              "created_utc": datetime.now(timezone.utc).isoformat(), "click_xy": args.click,
              "click_provenance": args.click_provenance,
              "stages": [], "input_mesh_used": False, "input_mask_used": False,
              "ground_truth_used_at_runtime": False, "identity_accuracy_claimed": False,
              "generated_mesh_is_ground_truth": False,
              "tracking_policy": args.tracking_policy,
              "output_policy": getattr(args, "output_policy", "legacy"),
              "limitations": ["Largest depth-admissible component is a whole-object heuristic.",
                              "Ambiguity remains explicit even if automatic selection is permitted.",
                              "Single-view geometric consistency does not establish hidden-surface accuracy.",
                              "Independent generated-mesh pose accuracy requires a separately justified coordinate alignment."]}
    started = time.perf_counter()
    source_hashes = {}
    try:
        if args.max_frames < 0 or args.max_frames == 1:
            raise ValueError("max_frames must be 0 or >=2")
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("timeout must be finite positive")
        rgb, depth, paths, names = first_observation(args.sequence, args.depth_scale, args.click)
        if len(names) < 2:
            raise ValueError("Tracking requires at least two frames")
        python = Path(args.python).expanduser().absolute()
        report["generator_dependencies"] = dependencies(Path(args.click_to_model_root).resolve(), python)
        if not Path(args.sam2_checkpoint).is_file():
            raise FileNotFoundError("SAM2 checkpoint is missing; downloads are not automatic")
        source_names = ["tools/run_meshless_click_to_model.py", "meshless_initialization.py",
                        "tools/generate_click_mesh.py", "tools/run_click_to_model.py",
                        "click_initialization.py", "click_to_model.py", "pose_validity.py",
                        "rgbd_seed.py", "trusted_view_seed.py", "quarantined_recovery.py", "adaptive_refinement.py",
                        "initialization_admission.py", "initialization_session.py", "tentative_tracking.py", "estimater.py", "Utils.py",
                        "datareader.py", "pose_kalman.py", "stage_profiler.py"]
        source_names += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "learning").rglob("*.py"))]
        if args.tracking_policy == "tentative-preloss":
            source_names += list(MEASURED_SOURCE_FILES)
        if getattr(args, "output_policy", "legacy") == "continuity":
            source_names += list(OUTPUT_POLICY_SOURCE_FILES)
        source_hashes = {name: digest(ROOT / name) for name in source_names}
        report["source_sha256"] = source_hashes
        for name in source_names:
            target = output / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        report["input_manifest"] = {k: {"path": str(v), "sha256": digest(v)} for k, v in paths.items()}
        report["sequence_frames"] = len(names)
        report["development_prefix"] = bool(args.max_frames and args.max_frames < len(names))
        report["depth_scale_m_per_unit"] = args.depth_scale
        first = output / "first_frame"
        for directory in ("rgb", "depth"):
            (first / directory).mkdir(parents=True)
            shutil.copyfile(paths[directory], first / directory / "000000.png")
        shutil.copyfile(paths["intrinsics"], first / "cam_K.txt")
        commands = build_commands(args, output)
        report["commands"] = {name: command for name, command in commands}
        save_json(output / "report.json", report)
        if args.dry_run:
            report["status"] = "dry_run_ready"
        else:
            env = tracking_environment(args.python, output)
            for name, command in commands:
                stage = {"stage": name, "status": "running", "command": command}
                report["stages"].append(stage)
                save_json(output / "report.json", report)
                stage_started = time.perf_counter()
                try:
                    if name != "segmentation" and digest(output / "segmentation/selected_mask.png") != report["selected_mask_sha256"]:
                        raise RuntimeError("Selected SAM2 mask changed between stages")
                    with (output / f"{name}.log").open("x") as log:
                        # Let the adapter write its own timeout receipt first.
                        returncode = launch_stage(command, log, env, args.timeout + (30 if name == "generation" else 0))
                    stage["returncode"] = returncode
                    if returncode:
                        raise RuntimeError(f"{name} exited {returncode}; see {name}.log and stage receipt")
                    receipt_name = "receipt.json" if name == "generation" else "report.json"
                    stage_report = json.loads((output / name / receipt_name).read_text())
                    required_status = {"segmentation": "selected", "generation": "accepted", "tracking": "completed"}[name]
                    if stage_report.get("status") != required_status:
                        raise RuntimeError(f"{name} did not satisfy {required_status}: {stage_report.get('status')}")
                    stage["status"] = "completed"
                    stage["receipt_sha256"] = digest(output / name / receipt_name)
                    if name == "segmentation":
                        report["initialization_ambiguous"] = stage_report["selection"]["ambiguous"]
                        report["mask_selection"] = stage_report["selection"]
                        report["selected_mask_sha256"] = stage_report["selected_mask_sha256"]
                    if name == "generation":
                        report["generated_mesh_sha256"] = stage_report["result"]["mesh"]["sha256"]
                        mask_inputs = stage_report["observation"]["originals"]["mask"]
                        if mask_inputs["sha256"] != report["selected_mask_sha256"]:
                            raise RuntimeError("Generator did not consume the selected SAM2 mask bytes")
                    if name == "tracking":
                        if stage_report["mesh_sha256"] != report["generated_mesh_sha256"]:
                            raise RuntimeError("Tracker did not consume this run's generated mesh")
                        selected = cv2.imread(str(output / "segmentation/selected_mask.png"), cv2.IMREAD_UNCHANGED)
                        consumed = cv2.imread(str(output / "tracking/initial_mask.png"), cv2.IMREAD_UNCHANGED)
                        if consumed is None or not np.array_equal(selected, consumed):
                            raise RuntimeError("Tracker did not consume the selected SAM2 mask")
                        report["tracking_summary"] = {key: stage_report[key] for key in
                                                      ("frames", "tracking_fps", "status_counts", "seed_sources")}
                        if args.tracking_policy == "tentative-preloss":
                            if stage_report.get("tracking_policy") != "tentative-preloss":
                                raise RuntimeError("Tracking child did not use the requested measured-motion policy")
                            requested_output = getattr(args, "output_policy", "legacy")
                            if stage_report.get("output_policy", "legacy") != requested_output:
                                raise RuntimeError("Tracking child did not use the requested output policy")
                            trust_fields = (*TENTATIVE_TRUST_FIELDS, "trusted_memory_initialized")
                            if any(stage_report.get(key) is not False for key in trust_fields):
                                raise RuntimeError("Measured-motion child must preserve all tentative trust flags")
                            report["tracking_summary"].update(tracking_policy="tentative-preloss",
                                output_policy=requested_output,
                                **{key: False for key in trust_fields})
                except BaseException as error:
                    stage.update(status="failed", error=f"{type(error).__name__}: {error}")
                    raise
                finally:
                    stage["wall_seconds"] = time.perf_counter() - stage_started
                    save_json(output / "report.json", report)
            report["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        report["source_unchanged"] = all((ROOT / name).is_file() and digest(ROOT / name) == value
                                         for name, value in source_hashes.items())
        if source_hashes and not report["source_unchanged"]:
            report["status"] = "invalid_source_changed"
        report["wall_seconds"] = time.perf_counter() - started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(output / "report.json", report)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", type=Path, required=True)
    p.add_argument("--click", nargs=2, type=float, required=True, metavar=("X", "Y"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--depth-scale", type=float, required=True)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--click-provenance", default="caller_supplied_unverified",
                   help="Audit text: disclose synthetic or mask-derived development clicks")
    p.add_argument("--ambiguity-policy", choices=("largest", "reject"), default="largest")
    p.add_argument("--sam2-checkpoint", type=Path, default=ROOT / "third_party/sam2/checkpoints/sam2.1_hiera_small.pt")
    p.add_argument("--click-to-model-root", type=Path, default=ROOT.parent / "Click-to-Model")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stage", choices=("orchestrate", "segment"), default="orchestrate", help=argparse.SUPPRESS)
    add_tracking_policy_arguments(p)
    return p


def main():
    args = parser().parse_args()
    if args.stage == "segment" and args.dry_run:
        raise ValueError("Use orchestrator --dry-run; segmentation worker requires real inference")
    result = segment_worker(args) if args.stage == "segment" else run(args)
    print(json.dumps({"status": result["status"], "output": str(args.output_dir)}), flush=True)
    return int(result["status"] not in ("completed", "selected", "dry_run_ready"))


if __name__ == "__main__":
    raise SystemExit(main())
