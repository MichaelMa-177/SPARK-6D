#!/usr/bin/env python3
"""Optional first-frame SAM3D adapter; generated geometry is never ground truth.

Uses an existing Click-to-Model checkout and its low-VRAM SAM3D fork. No
downloads, checkpoint deserialization, or GPU imports occur in --dry-run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def dependencies(checkout: Path, python: Path) -> dict:
    """Check the supported local adapter contract without importing its code."""
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(f"Python interpreter is unavailable: {python}")
    fork = checkout / "sam-3d-objects"
    required = [checkout / "run_sam3d_rgbd.py",
                checkout / "click_to_model/reconstruction/pipeline.py",
                fork / "notebook/inference.py", checkout / "dinov2/hubconf.py",
                fork / "checkpoints/hf/pipeline.yaml"]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required local dependency is missing or empty: {path}")
    if "low_vram" not in required[2].read_text():
        raise ValueError("SAM3D checkout lacks the required low_vram fork interface")
    config = required[-1]
    references = re.findall(r"^\w+_ckpt_path:\s*([^\s#]+)", config.read_text(), re.M)
    if not references:
        raise ValueError("No checkpoint paths found in the supported SAM3D config")
    weights = []
    for name in references:
        if Path(name).name != name:
            raise ValueError(f"Expected a checkpoint filename in pipeline.yaml: {name}")
        path = config.parent / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required checkpoint is missing or empty: {path}")
        weights.append({"path": str(path.resolve()), "bytes": path.stat().st_size,
                        "sha256": None, "hash_status": "not_computed_by_adapter"})
    return {"source_files": [artifact(p) for p in required], "weights": weights,
            "checkpoint_deserialized": False, "gpu_inference_verified": False}


def stage_observation(args: argparse.Namespace, output: Path) -> dict:
    import cv2

    paths = {name: Path(getattr(args, name)).expanduser().resolve()
             for name in ("rgb", "depth", "mask", "intrinsics")}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
        if name != "intrinsics" and path.suffix.lower() != ".png":
            raise ValueError(f"{name} must be a PNG input")
    if not math.isfinite(args.depth_scale) or args.depth_scale <= 0:
        raise ValueError("depth_scale must be finite positive metres per stored unit")
    rgb = cv2.imread(str(paths["rgb"]), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(paths["depth"]), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(paths["mask"]), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None or mask is None:
        raise ValueError("Failed to decode an RGB/depth/mask PNG")
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("Depth must be an aligned uint16 single-channel PNG")
    if mask.ndim != 2 or mask.shape != depth.shape or rgb.shape[:2] != depth.shape:
        raise ValueError("RGB, depth and single-channel mask dimensions must match")
    labels = np.unique(mask[mask > 0])
    if len(labels) != 1:
        raise ValueError("Mask must identify exactly one positive foreground label")
    if np.count_nonzero(mask) < 100 or np.count_nonzero((mask > 0) & (depth > 0)) < 100:
        raise ValueError("At least 100 foreground pixels with valid depth are required")
    K = np.loadtxt(paths["intrinsics"], dtype=float)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0
            or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1])
            or not np.allclose([K[0, 1], K[1, 0]], [0, 0])):
        raise ValueError("Intrinsics must be a finite zero-skew pinhole 3x3 matrix")
    for directory in ("rgb", "depth", "masks"):
        (output / directory).mkdir()
    # Exactly one observation is visible to the generator. No sequence directory
    # or evaluation/GT file is forwarded to the subprocess.
    shutil.copyfile(paths["rgb"], output / "rgb/000000.png")
    shutil.copyfile(paths["depth"], output / "depth/000000.png")
    if not cv2.imwrite(str(output / "masks/000000.png"), (mask > 0).astype(np.uint8) * 255):
        raise OSError("Could not write the binary mask")
    shutil.copyfile(paths["intrinsics"], output / "cam_K.txt")
    return {"originals": {name: artifact(path) for name, path in paths.items()},
            "staged": [artifact(output / name) for name in
                       ("rgb/000000.png", "depth/000000.png", "masks/000000.png", "cam_K.txt")],
            "mask_conversion": "one positive label -> 255; background -> 0",
            "frames": 1, "shape_hw": list(depth.shape), "depth_scale_m_per_unit": args.depth_scale,
            "alignment": "RGB/depth alignment supplied by caller; not inferred from matching dimensions",
            "ground_truth_used": False}


def child_environment(checkout: Path, python: Path, output: Path) -> dict:
    env = os.environ.copy()
    prefix = python.parent.parent
    fork = checkout / "sam-3d-objects"
    env.update({"CONDA_PREFIX": str(prefix), "CUDA_HOME": str(prefix),
                "CLICK_TO_MODEL_ROOT": str(checkout), "SPARK6D_REPO": str(REPO_ROOT),
                "SAM3D_REPO": str(fork), "SAM3D_CHECKPOINT_DIR": str(fork / "checkpoints/hf"),
                "SAM3D_DINO_REPO": str(checkout / "dinov2"), "SAM3D_DINO_SKIP_WEIGHTS": "1",
                "LIDRA_SKIP_INIT": "true", "ATTN_BACKEND": "sdpa", "SPARSE_ATTN_BACKEND": "sdpa",
                "SPARSE_BACKEND": "spconv", "XFORMERS_DISABLED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                "MPLCONFIGDIR": str(output / "cache/matplotlib"),
                "TORCHINDUCTOR_CACHE_DIR": str(output / "cache/torchinductor")})
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(prefix / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(checkout), str(fork), str(fork / "notebook"),
                                          str(REPO_ROOT), str(REPO_ROOT / "mycpp/build")])
    return env


def validate_mesh(output: Path, expected_depth_scale: float) -> dict:
    """Reject unsuccessful scale recovery even if the backend wrote an OBJ."""
    mesh = output / "mesh/model.obj"
    metadata_file = output / "mesh/model_metadata.json"
    metadata = json.loads(metadata_file.read_text())
    if metadata.get("scale_method") != "rgbd_bounded_scale_icp":
        raise ValueError("Mesh was not recovered with bounded RGB-D scale ICP")
    if metadata.get("scale_quality_issues") != []:
        raise ValueError(f"Rejected scale: {metadata.get('scale_quality_issues')}")
    icp = metadata.get("metric_scale_icp", {})
    if icp.get("accepted") is not True:
        raise ValueError("ICP did not explicitly accept the recovered scale")
    scale = float(metadata["applied_scale_factor"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid metric scale factor")
    if not math.isclose(float(metadata["depth_scale"]), expected_depth_scale, rel_tol=1e-9):
        raise ValueError("Generator used a different depth unit conversion")
    metrics = icp["metrics"]
    coverage = float(metrics["target_coverage"])
    rmse, threshold = float(metrics["target_rmse_m"]), float(icp["fine_threshold_m"])
    if not (0.45 <= coverage <= 1 and math.isfinite(rmse) and 0 <= rmse <= threshold
            and math.isfinite(threshold) and threshold > 0):
        raise ValueError("ICP coverage/RMSE metadata failed validation")
    depth_coverage = float(metadata["depth_filter"]["valid_depth_fraction"])
    if not 0.25 <= depth_coverage <= 1 or metadata["metric_mask_silhouette"]["touches_image_border"] is not False:
        raise ValueError("Depth coverage or truncated silhouette invalidates scale")
    for name, relative in (("image", "rgb/000000.png"), ("depth", "depth/000000.png"), ("mask", "masks/000000.png")):
        if Path(metadata[name]).resolve() != (output / relative).resolve():
            raise ValueError("Mesh metadata refers to a different observation")
    T = np.asarray(metadata["metric_mesh_to_camera"], dtype=float)
    if (T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1])
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-4)
            or not math.isclose(float(np.linalg.det(T[:3, :3])), 1, abs_tol=1e-4)):
        raise ValueError("Invalid rigid metric mesh transform")
    # Read OBJ geometry only: no material loading or external asset fetching.
    vertices, faces = [], []
    for line in mesh.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "v":
            vertices.append([float(x) for x in parts[1:4]])
        elif parts and parts[0] == "f":
            faces.append([int(x.split("/")[0]) for x in parts[1:]])
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3 or not np.isfinite(vertices).all() or not faces:
        raise ValueError("Mesh has no finite usable geometry")
    if any(len(face) < 3 or any(i < 1 or i > len(vertices) for i in face) for face in faces):
        raise ValueError("OBJ has invalid faces")
    extent = np.ptp(vertices, axis=0)
    claimed_extent = np.asarray(metadata["metric_mesh_extent_m"], dtype=float)
    if claimed_extent.shape != (3,) or not np.isfinite(claimed_extent).all() or extent.max() <= 0 or not np.allclose(extent, claimed_extent, atol=1e-6, rtol=1e-4):
        raise ValueError("OBJ geometry and metric extent metadata disagree")
    return {"mesh": artifact(mesh), "metadata": artifact(metadata_file),
            "units": "metres", "mesh_frame": "object-local",
            "metadata_pose_frame": "PyTorch3D camera; do not use directly as OpenCV camera pose",
            "generated_model_is_ground_truth": False, "hidden_surface_accuracy_verified": False,
            "metric_extent_m": extent.tolist(), "scale_factor": scale}


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output).expanduser().resolve()
    if not output.is_relative_to(REPO_ROOT / "results"):
        raise ValueError("Output must be a NEW directory inside this repository's results/")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "spark6d_click_mesh/1.0", "status": "incomplete", "dry_run": args.dry_run,
               "generated_model_is_ground_truth": False, "output": str(output),
               "started_unix_seconds": time.time()}
    try:
        checkout = Path(args.click_to_model_root).expanduser().resolve()
        # Preserve the interpreter path: resolving a conda symlink can select
        # the wrong prefix for extension/shared-library discovery.
        python = Path(args.python).expanduser().absolute()
        receipt["dependencies"] = dependencies(checkout, python)
        receipt["observation"] = stage_observation(args, output)
        command = [str(python), str(checkout / "run_sam3d_rgbd.py"), "--data-dir", str(output),
                   "--depth-scale", str(args.depth_scale), "--seed", str(args.seed),
                   "--distilled", "--scale-mode", "icp", "--require-icp-confidence"]
        receipt["command"] = command
        receipt["network_policy"] = "existing local sources and weights; Hugging Face/transformers offline"
        if args.dry_run:
            receipt["status"] = "dry_run_ready"
        else:
            with (output / "generation.log").open("w") as log:
                result = subprocess.run(command, cwd=checkout,
                                        env=child_environment(checkout, python, output),
                                        stdout=log, stderr=subprocess.STDOUT,
                                        timeout=args.timeout, check=False)
            receipt["returncode"] = result.returncode
            if result.returncode:
                raise RuntimeError(f"SAM3D exited {result.returncode}; see generation.log")
            receipt["result"] = validate_mesh(output, args.depth_scale)
            receipt["status"] = "accepted"
    except (Exception, KeyboardInterrupt) as error:
        receipt["status"] = "failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        receipt["finished_unix_seconds"] = time.time()
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("rgb", "depth", "mask", "intrinsics", "output"):
        result.add_argument(f"--{name}", required=True)
    result.add_argument("--depth-scale", type=float, required=True, help="metres per uint16 stored depth unit")
    result.add_argument("--click-to-model-root", default=str(REPO_ROOT.parent / "Click-to-Model"))
    result.add_argument("--python", default=sys.executable, help="existing FoundationPose/Click-to-Model environment interpreter")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--timeout", type=float, default=1800)
    result.add_argument("--dry-run", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    receipt = run(args)
    print(json.dumps({key: receipt[key] for key in ("status", "output", "dry_run")}))
    return int(receipt["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
