#!/usr/bin/env python3
"""Run FoundationPose registration and tracking on a live RealSense stream.

The RealSense depth image is aligned to the color image before inference.  RGB
is passed to FoundationPose as uint8 and depth is passed in metres as float32.
The returned 4x4 pose maps vertices in the original mesh frame into the camera
frame.

The first-frame segmentation is deliberately represented by ``MaskProvider``.
The file-backed implementation makes the FoundationPose path usable now; a SAM
implementation can later be injected without changing the camera or pose loop.
Likewise, a mesh produced by an upstream reconstruction stage is supplied with
``--mesh-file``.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys
import time
from typing import Protocol

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class MaskProvider(Protocol):
    """Interface for FoundationPose registration-mask sources.

    Implementations receive the exact aligned frame used for registration.
    This is the seam intended for the forthcoming live SAM integration.
    """

    def get_mask(
        self, rgb: np.ndarray, depth_m: np.ndarray, camera_matrix: np.ndarray
    ) -> np.ndarray:
        """Return an HxW foreground mask for the supplied RGB-D frame."""


class BoundedOutputExecutor:
    """Serialize pose/video writes off the tracking thread with bounded memory."""

    def __init__(
        self,
        max_pending: int,
        max_workers: int = 1,
        thread_name_prefix: str = "spark6d-output",
    ):
        self.max_pending = max_pending
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._pending = deque()

    def submit(self, function, *args) -> float:
        """Queue one write and return seconds spent applying backpressure."""
        wait_started = time.perf_counter()
        while len(self._pending) >= self.max_pending:
            self._pending.popleft().result()
        wait_seconds = time.perf_counter() - wait_started
        self._pending.append(self._executor.submit(function, *args))
        return wait_seconds

    def close(self) -> None:
        while self._pending:
            self._pending.popleft().result()
        self._executor.shutdown(wait=True)


@dataclass(frozen=True)
class FileMaskProvider:
    """Load a static binary mask, resizing it with nearest-neighbour sampling."""

    path: Path

    def get_mask(
        self, rgb: np.ndarray, depth_m: np.ndarray, camera_matrix: np.ndarray
    ) -> np.ndarray:
        del depth_m, camera_matrix
        raw = cv2.imread(str(self.path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise RuntimeError(f"Unable to read registration mask: {self.path}")
        if raw.ndim == 3:
            raw = np.any(raw[..., :3] > 0, axis=2)
        else:
            raw = raw > 0
        height, width = rgb.shape[:2]
        if raw.shape != (height, width):
            logging.warning(
                "Resizing registration mask from %sx%s to %sx%s",
                raw.shape[1],
                raw.shape[0],
                width,
                height,
            )
            raw = cv2.resize(
                raw.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        return np.ascontiguousarray(raw, dtype=bool)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Register and track one object with FoundationPose using aligned "
            "live RealSense RGB-D frames."
        )
    )
    parser.add_argument(
        "--mesh-file",
        type=Path,
        help="object mesh generated/provided upstream (OBJ, PLY, GLB, etc.)",
    )
    parser.add_argument(
        "--mask-file",
        type=Path,
        help="binary mask for the registration frame; foreground is non-zero",
    )
    parser.add_argument(
        "--mesh-scale",
        type=float,
        default=1.0,
        help="multiply mesh vertices by this value; FoundationPose expects metres",
    )
    parser.add_argument("--serial", help="RealSense serial; default uses the first device")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=15,
        help="discard this many frames while auto-exposure settles",
    )
    parser.add_argument(
        "--depth-min-m",
        type=float,
        default=0.10,
        help="depths nearer than this are made invalid",
    )
    parser.add_argument(
        "--depth-max-m",
        type=float,
        default=3.0,
        help="depths farther than this are made invalid",
    )
    parser.add_argument("--est-refine-iter", type=int, default=5)
    parser.add_argument(
        "--track-refine-iter",
        type=int,
        default=1,
        help="per-frame learned refinements (default: 1, validated FPS80 profile)",
    )
    parser.add_argument(
        "--compile-track-refiner",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "compile and CUDA-graph the fixed-shape per-frame RefineNet; "
            "registration stays eager (default: enabled)"
        ),
    )
    parser.add_argument(
        "--use-kalman",
        action="store_true",
        help="enable the current SPARK-6D adaptive Kalman tracking extension",
    )
    parser.add_argument("--kalman-refine-interval", type=int, default=2)
    parser.add_argument("--robust-tracking", action="store_true",
                        help="experimental depth-verified seeds; every frame still runs RefineNet")
    parser.add_argument("--robust-relocalization", action="store_true",
                        help="experimental ORB/PnP recovery (not validated); implies --robust-tracking")
    parser.add_argument(
        "--kalman-no-adaptive",
        action="store_true",
        help="disable motion/uncertainty fallback when --use-kalman is enabled",
    )
    parser.add_argument("--debug", type=int, default=0, choices=range(4))
    parser.add_argument(
        "--debug-dir", type=Path, default=PROJECT_ROOT / "debug" / "realsense_live"
    )
    parser.add_argument(
        "--pose-output-dir",
        type=Path,
        help="optionally save one original-mesh-to-camera 4x4 TXT pose per frame",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="0 runs until q/Esc/Ctrl-C"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="run without an OpenCV window"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="save the annotated live tracking stream as MP4",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        help="MP4 path (default: <debug-dir>/tracking.mp4)",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=30.0,
        help="saved-video frame rate (default: 30)",
    )
    parser.add_argument(
        "--output-queue-size",
        type=int,
        default=4,
        help="maximum pending asynchronous pose/video writes (default: 4)",
    )
    parser.add_argument(
        "--rgbd-output-dir",
        type=Path,
        help="save every aligned inference frame under <dir>/rgb and <dir>/depth",
    )
    parser.add_argument(
        "--rgbd-output-workers",
        type=int,
        default=1,
        help="background PNG encoding workers (default: 1; avoids CPU contention)",
    )
    parser.add_argument(
        "--rgbd-png-compression",
        type=int,
        default=1,
        help="OpenCV PNG compression from 0 (fastest) to 9 (smallest; default: 1)",
    )
    parser.add_argument(
        "--benchmark-report",
        type=Path,
        help="write online model-only and end-to-end timing statistics as JSON",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--list-devices", action="store_true", help="list RealSense devices and exit"
    )
    args = parser.parse_args(argv)
    args.robust_tracking = args.robust_tracking or args.robust_relocalization
    if args.robust_tracking and args.use_kalman:
        parser.error("--robust-tracking is incompatible with --use-kalman")

    positive_values = {
        "--width": args.width,
        "--height": args.height,
        "--fps": args.fps,
        "--timeout-ms": args.timeout_ms,
        "--mesh-scale": args.mesh_scale,
        "--est-refine-iter": args.est_refine_iter,
        "--track-refine-iter": args.track_refine_iter,
        "--kalman-refine-interval": args.kalman_refine_interval,
        "--video-fps": args.video_fps,
        "--output-queue-size": args.output_queue_size,
        "--rgbd-output-workers": args.rgbd_output_workers,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        parser.error(f"must be greater than zero: {', '.join(invalid)}")
    if args.warmup_frames < 0 or args.max_frames < 0:
        parser.error("--warmup-frames and --max-frames cannot be negative")
    if not 0 <= args.rgbd_png_compression <= 9:
        parser.error("--rgbd-png-compression must be between 0 and 9")
    if args.depth_min_m < 0 or args.depth_max_m <= args.depth_min_m:
        parser.error("--depth-max-m must be greater than --depth-min-m >= 0")
    if not args.list_devices:
        for name, path in (("--mesh-file", args.mesh_file), ("--mask-file", args.mask_file)):
            if path is None:
                parser.error(f"{name} is required unless --list-devices is used")
            if not path.expanduser().is_file():
                parser.error(f"{name} is not a file: {path}")
    return args


def _device_info(device: object, rs: object, field: object) -> str:
    try:
        return device.get_info(field) if device.supports(field) else "unknown"
    except RuntimeError:
        return "unknown"


def _select_device(rs: object, serial: str | None) -> tuple[object, object]:
    context = rs.context()
    devices = list(context.query_devices())
    if not devices:
        raise RuntimeError(
            "No RealSense device found; check the USB 3 connection and permissions"
        )
    if serial is None:
        return context, devices[0]
    for device in devices:
        if _device_info(device, rs, rs.camera_info.serial_number) == serial:
            return context, device
    detected = ", ".join(
        _device_info(device, rs, rs.camera_info.serial_number) for device in devices
    )
    raise RuntimeError(f"RealSense serial {serial!r} not found; detected: {detected}")


def _print_devices(rs: object) -> None:
    context = rs.context()
    devices = list(context.query_devices())
    if not devices:
        print("No RealSense devices detected.")
        return
    for index, device in enumerate(devices):
        name = _device_info(device, rs, rs.camera_info.name)
        serial = _device_info(device, rs, rs.camera_info.serial_number)
        firmware = _device_info(device, rs, rs.camera_info.firmware_version)
        usb = _device_info(device, rs, rs.camera_info.usb_type_descriptor)
        print(f"[{index}] {name} | serial={serial} | firmware={firmware} | USB={usb}")


def _load_mesh(path: Path, scale: float) -> object:
    import trimesh

    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise RuntimeError(f"Mesh scene has no geometry: {path}")
        # Bake scene-graph transforms before passing vertices to FoundationPose.
        if hasattr(loaded, "to_geometry"):
            loaded = loaded.to_geometry()
        else:  # Compatibility with the older trimesh used by upstream FP.
            loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise RuntimeError(f"Unsupported mesh content in {path}: {type(loaded).__name__}")
    if loaded.vertices.size == 0 or loaded.faces.size == 0:
        raise RuntimeError(f"Mesh is empty: {path}")
    mesh = loaded.copy()
    if scale != 1.0:
        mesh.vertices *= scale
    # Accessing vertex_normals makes trimesh calculate them when absent.
    _ = mesh.vertex_normals
    return mesh


def _camera_matrix(intrinsics: object) -> np.ndarray:
    return np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _prepare_depth(
    depth_raw: np.ndarray, depth_scale: float, minimum: float, maximum: float
) -> np.ndarray:
    depth_m = depth_raw.astype(np.float32) * np.float32(depth_scale)
    invalid = (~np.isfinite(depth_m)) | (depth_m < minimum) | (depth_m > maximum)
    depth_m[invalid] = 0.0
    return np.ascontiguousarray(depth_m)


def _put_text(image: np.ndarray, text: str, y: int) -> None:
    cv2.putText(
        image,
        text,
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_mask_preview(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    overlay = bgr.copy()
    overlay[mask] = (0, 180, 255)
    preview = cv2.addWeighted(bgr, 0.7, overlay, 0.3, 0.0)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(preview, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
    return preview


def _draw_pose_overlay(
    rgb: np.ndarray,
    pose: np.ndarray,
    camera_matrix: np.ndarray,
    to_origin: np.ndarray,
    bbox: np.ndarray,
) -> np.ndarray:
    center_pose = pose @ np.linalg.inv(to_origin)
    visualization = np.ascontiguousarray(rgb[..., ::-1])
    minimum, maximum = bbox.min(axis=0), bbox.max(axis=0)
    corners = np.asarray(
        [
            [minimum[0], minimum[1], minimum[2]],
            [maximum[0], minimum[1], minimum[2]],
            [minimum[0], maximum[1], minimum[2]],
            [maximum[0], maximum[1], minimum[2]],
            [minimum[0], minimum[1], maximum[2]],
            [maximum[0], minimum[1], maximum[2]],
            [minimum[0], maximum[1], maximum[2]],
            [maximum[0], maximum[1], maximum[2]],
        ],
        dtype=np.float64,
    )
    axis_scale = max(0.01, float(np.linalg.norm(maximum - minimum)) * 0.5)
    axis_points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [axis_scale, 0.0, 0.0],
            [0.0, axis_scale, 0.0],
            [0.0, 0.0, axis_scale],
        ],
        dtype=np.float64,
    )

    def project(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points_camera = (
            center_pose[:3, :3] @ points.T + center_pose[:3, 3:4]
        ).T
        projected = (camera_matrix @ points_camera.T).T
        valid = points_camera[:, 2] > 1e-6
        pixels = np.zeros((len(points), 2), dtype=np.int32)
        pixels[valid] = np.rint(
            projected[valid, :2] / projected[valid, 2:3]
        ).astype(np.int32)
        return pixels, valid

    corner_pixels, corner_valid = project(corners)
    edges = (
        (0, 1), (0, 2), (1, 3), (2, 3),
        (4, 5), (4, 6), (5, 7), (6, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    for start, end in edges:
        if corner_valid[start] and corner_valid[end]:
            cv2.line(
                visualization,
                tuple(corner_pixels[start]),
                tuple(corner_pixels[end]),
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    axis_pixels, axis_valid = project(axis_points)
    axis_colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    for endpoint, color in zip(range(1, 4), axis_colors):
        if axis_valid[0] and axis_valid[endpoint]:
            cv2.arrowedLine(
                visualization,
                tuple(axis_pixels[0]),
                tuple(axis_pixels[endpoint]),
                color,
                3,
                cv2.LINE_AA,
                tipLength=0.0,
            )
    return visualization


def _open_video_writer(path: Path, frame: np.ndarray, fps: float) -> object:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Unable to open tracking video writer: {path}")
    print(f"Saving tracking video: {path} ({fps:g} FPS)")
    return writer


def _write_frame_outputs(
    pose_path: Path | None,
    pose: np.ndarray | None,
    video_writer: object | None,
    video_frame: np.ndarray | None,
) -> None:
    if pose_path is not None and pose is not None:
        np.savetxt(pose_path, pose.reshape(4, 4), fmt="%.9f")
    if video_writer is not None and video_frame is not None:
        video_writer.write(video_frame)


def _next_rgbd_index(output_dir: Path) -> int:
    indices: list[int] = []
    for modality in ("rgb", "depth"):
        for path in (output_dir / modality).glob("*.png"):
            if path.stem.isdigit():
                indices.append(int(path.stem))
    return max(indices, default=-1) + 1


def _write_rgbd_outputs(
    rgb_path: Path,
    bgr: np.ndarray,
    depth_path: Path,
    depth_raw: np.ndarray,
    png_compression: int,
) -> None:
    parameters = [cv2.IMWRITE_PNG_COMPRESSION, png_compression]
    if not cv2.imwrite(str(rgb_path), bgr, parameters):
        raise RuntimeError(f"Failed to save inference RGB frame: {rgb_path}")
    if not cv2.imwrite(str(depth_path), depth_raw, parameters):
        raise RuntimeError(f"Failed to save inference depth frame: {depth_path}")


def _timing_percentiles(seconds: list[float]) -> dict[str, float]:
    if not seconds:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}
    milliseconds = np.asarray(seconds, dtype=np.float64) * 1000.0
    return {
        "mean_ms": float(milliseconds.mean()),
        "median_ms": float(np.median(milliseconds)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
    }


def _create_estimator(args: argparse.Namespace, mesh: object) -> object:
    import torch
    import nvdiffrast.torch as dr

    from estimater import FoundationPose
    from learning.training.predict_pose_refine import PoseRefinePredictor
    from learning.training.predict_score import ScorePredictor

    args.debug_dir.mkdir(parents=True, exist_ok=True)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    estimator = FoundationPose(
        model_pts=mesh.vertices.copy(),
        model_normals=mesh.vertex_normals.copy(),
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(args.debug_dir),
        debug=args.debug,
        glctx=glctx,
    )
    estimator._registration_refiner_model = refiner.model
    estimator._tracking_refiner_model = refiner.model
    estimator._tracking_compile_seconds = 0.0
    if args.compile_track_refiner:
        compile_started = time.perf_counter()
        compiled_model = torch.compile(
            refiner.model, mode="reduce-overhead", fullgraph=True
        )
        input_height, input_width = refiner.cfg.input_resize
        dummy = torch.zeros(
            (1, 6, input_height, input_width),
            device="cuda",
            dtype=torch.float32,
        )
        with torch.inference_mode(), torch.amp.autocast(
            "cuda", enabled=refiner.amp
        ):
            compiled_model(dummy, dummy)
        torch.cuda.synchronize()
        estimator._tracking_refiner_model = compiled_model
        estimator._tracking_compile_seconds = time.perf_counter() - compile_started
    if args.use_kalman:
        estimator.enable_kalman(
            refine_interval=args.kalman_refine_interval,
            adaptive=not args.kalman_no_adaptive,
        )
    if args.robust_tracking:
        from robust_pose import RobustPoseConfig
        logging.warning(
            "Experimental observation-gated tracking is enabled. Development "
            "runs show accuracy regressions and unsuccessful recovery; verified "
            "means sparse-depth support, not ground-truth correctness. See "
            "docs/ROBUST6D_RESEARCH.md."
        )
        estimator.enable_robust_tracking(RobustPoseConfig(
            enable_relocalization=args.robust_relocalization))
    return estimator


def run(
    args: argparse.Namespace,
    mask_provider: MaskProvider | None = None,
) -> None:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "pyrealsense2 is missing; activate scripts/activate_foundationpose.sh"
        ) from exc

    if args.list_devices:
        _print_devices(rs)
        return

    import torch
    import trimesh

    from Utils import set_logging_format, set_seed

    set_logging_format()
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    set_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("FoundationPose requires a CUDA-capable GPU")

    mesh_path = args.mesh_file.expanduser().resolve()
    mask_path = (
        args.mask_file.expanduser().resolve() if args.mask_file is not None else None
    )
    args.debug_dir = args.debug_dir.expanduser().resolve()
    if args.video_path is not None:
        args.video_path = args.video_path.expanduser().resolve()
        args.save_video = True
    elif args.save_video:
        args.video_path = args.debug_dir / "tracking.mp4"
    if args.pose_output_dir is not None:
        args.pose_output_dir = args.pose_output_dir.expanduser().resolve()
        args.pose_output_dir.mkdir(parents=True, exist_ok=True)
    if args.benchmark_report is not None:
        args.benchmark_report = args.benchmark_report.expanduser().resolve()
    rgbd_start_index = 0
    if args.rgbd_output_dir is not None:
        args.rgbd_output_dir = args.rgbd_output_dir.expanduser().resolve()
        (args.rgbd_output_dir / "rgb").mkdir(parents=True, exist_ok=True)
        (args.rgbd_output_dir / "depth").mkdir(parents=True, exist_ok=True)
        rgbd_start_index = _next_rgbd_index(args.rgbd_output_dir)
        print(
            f"Saving aligned inference RGB-D to {args.rgbd_output_dir} "
            f"starting at frame {rgbd_start_index:06d}"
        )
    if mask_provider is None:
        if mask_path is None:
            raise RuntimeError("A mask provider or --mask-file is required")
        mask_provider = FileMaskProvider(mask_path)

    print(f"Loading mesh: {mesh_path}")
    mesh = _load_mesh(mesh_path, args.mesh_scale)
    initial_mesh_vertices = int(len(mesh.vertices))
    initial_mesh_faces = int(len(mesh.faces))
    extent = mesh.extents
    print(
        "Mesh extent (m): "
        f"[{extent[0]:.6f}, {extent[1]:.6f}, {extent[2]:.6f}] "
        f"(scale={args.mesh_scale:g})"
    )
    to_origin, oriented_extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack((-oriented_extents / 2, oriented_extents / 2), axis=0)

    print("Loading FoundationPose scorer/refiner weights...")
    estimator = _create_estimator(args, mesh)
    print("FoundationPose initialized.")

    context, device = _select_device(rs, args.serial)
    serial = _device_info(device, rs, rs.camera_info.serial_number)
    name = _device_info(device, rs, rs.camera_info.name)
    firmware = _device_info(device, rs, rs.camera_info.firmware_version)
    print(f"Opening {name} | serial={serial} | firmware={firmware}")

    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    started = False
    registered = False
    pose: np.ndarray | None = None
    frame_count = 0
    successful_frames = 0
    registration_count = 0
    register_seconds = 0.0
    track_seconds: list[float] = []
    refined_track_seconds: list[float] = []
    predicted_track_seconds: list[float] = []
    input_seconds: list[float] = []
    render_seconds: list[float] = []
    output_queue_wait_seconds: list[float] = []
    rgbd_queue_wait_seconds: list[float] = []
    track_window: deque[float] = deque(maxlen=max(2, args.fps))
    completion_times: deque[float] = deque(maxlen=max(2, args.fps))
    track_completion_times: list[float] = []
    stream_fps = 0.0
    mask: np.ndarray | None = None
    video_writer = None
    video_frames = 0
    output_executor = BoundedOutputExecutor(args.output_queue_size)
    rgbd_executor = (
        BoundedOutputExecutor(
            args.output_queue_size,
            max_workers=args.rgbd_output_workers,
            thread_name_prefix="spark6d-rgbd",
        )
        if args.rgbd_output_dir is not None
        else None
    )
    rgbd_frames = 0
    run_started = 0.0
    loop_finished = 0.0
    output_flush_seconds = 0.0
    rgbd_flush_seconds = 0.0
    output_error: BaseException | None = None
    kalman_report: dict[str, object] = {}
    robust_diagnostics: list[dict[str, object]] = []
    interrupted = False
    window_name = "RealSense FoundationPose (q: quit, r: re-register)"

    try:
        profile = pipeline.start(config)
        started = True
        align_to_color = rs.align(rs.stream.color)
        depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()
        camera_matrix = _camera_matrix(intrinsics)
        print(
            "Aligned color intrinsics: "
            f"fx={intrinsics.fx:.3f}, fy={intrinsics.fy:.3f}, "
            f"cx={intrinsics.ppx:.3f}, cy={intrinsics.ppy:.3f}"
        )
        print(f"Depth scale: {depth_scale:.9f} m/unit")
        print(f"Discarding {args.warmup_frames} camera warmup frames...")
        for _ in range(args.warmup_frames):
            pipeline.wait_for_frames(args.timeout_ms)
        run_started = time.perf_counter()

        while args.max_frames == 0 or frame_count < args.max_frames:
            input_started = time.perf_counter()
            frames = pipeline.wait_for_frames(args.timeout_ms)
            aligned = align_to_color.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            bgr = np.asanyarray(color_frame.get_data())
            rgb = np.ascontiguousarray(bgr[..., ::-1])
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m = _prepare_depth(
                depth_raw, depth_scale, args.depth_min_m, args.depth_max_m
            )
            input_seconds.append(time.perf_counter() - input_started)
            frame_count += 1

            if not registered:
                mask = mask_provider.get_mask(rgb, depth_m, camera_matrix)
                if mask.shape != depth_m.shape:
                    raise RuntimeError(
                        f"Mask provider returned {mask.shape}; expected {depth_m.shape}"
                    )
                mask = np.ascontiguousarray(mask > 0)
                if np.count_nonzero(mask) < 4:
                    raise RuntimeError("Registration mask contains fewer than four pixels")
                valid_registration_depth = mask & (depth_m >= 0.001)
                if np.count_nonzero(valid_registration_depth) < 4:
                    logging.warning(
                        "Registration mask has no usable aligned depth; waiting for next frame"
                    )
                    if not args.no_display:
                        preview = _draw_mask_preview(rgb, mask)
                        _put_text(preview, "Waiting for valid depth inside mask", 28)
                        cv2.imshow(window_name, preview)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            break
                    continue
                torch.cuda.synchronize()
                inference_started = time.perf_counter()
                estimator.refiner.model = estimator._registration_refiner_model
                pose = estimator.register(
                    K=camera_matrix,
                    rgb=rgb,
                    depth=depth_m,
                    ob_mask=mask,
                    iteration=args.est_refine_iter,
                    timestamp=float(color_frame.get_timestamp()) / 1000.0 if args.robust_tracking else None,
                )
                estimator.refiner.model = estimator._tracking_refiner_model
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - inference_started
                register_seconds += elapsed
                registration_count += 1
                registered = True
                mode = "REGISTER"
                tracker_state = "register"
            else:
                predicted_before = int(
                    getattr(estimator, "kalman_stats", {}).get(
                        "predicted_frames", 0
                    )
                )
                torch.cuda.synchronize()
                inference_started = time.perf_counter()
                pose = estimator.track_one(
                    rgb=rgb,
                    depth=depth_m,
                    K=camera_matrix,
                    iteration=args.track_refine_iter,
                    timestamp=float(color_frame.get_timestamp()) / 1000.0 if args.robust_tracking else None,
                )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - inference_started
                track_seconds.append(elapsed)
                track_window.append(elapsed)
                mode = "TRACK"
                frame_modes = getattr(estimator, "kalman_frame_modes", None)
                tracker_state = frame_modes[-1] if frame_modes else "refined"
                if args.robust_tracking:
                    diagnostics = dict(estimator.tracking_diagnostics)
                    diagnostics["capture_frame"] = frame_count
                    robust_diagnostics.append(diagnostics)
                    tracker_state = diagnostics["status"]
                predicted_after = int(
                    getattr(estimator, "kalman_stats", {}).get(
                        "predicted_frames", 0
                    )
                )
                if predicted_after > predicted_before:
                    predicted_track_seconds.append(elapsed)
                else:
                    refined_track_seconds.append(elapsed)
            successful_frames += 1
            output_index = (
                rgbd_start_index + successful_frames - 1
                if rgbd_executor is not None
                else frame_count
            )
            if rgbd_executor is not None:
                filename = f"{output_index:06d}.png"
                rgbd_queue_wait_seconds.append(
                    rgbd_executor.submit(
                        _write_rgbd_outputs,
                        args.rgbd_output_dir / "rgb" / filename,
                        bgr.copy(),
                        args.rgbd_output_dir / "depth" / filename,
                        depth_raw.copy(),
                        args.rgbd_png_compression,
                    )
                )
                rgbd_frames += 1
            completed_at = time.perf_counter()
            completion_times.append(completed_at)
            if mode == "TRACK":
                track_completion_times.append(completed_at)

            tracking_fps = (
                len(track_window) / sum(track_window) if track_window else 0.0
            )
            stream_fps = 0.0
            if len(completion_times) >= 2:
                elapsed = completion_times[-1] - completion_times[0]
                if elapsed > 0:
                    stream_fps = (len(completion_times) - 1) / elapsed

            visualization = None
            video_frame = None
            quit_requested = False
            if (not args.no_display or args.save_video) and pose is not None:
                render_started = time.perf_counter()
                visualization = _draw_pose_overlay(
                    rgb, pose, camera_matrix, to_origin, bbox
                )
                display_mode = tracker_state.upper().replace("_", " ")
                _put_text(
                    visualization,
                    f"{display_mode} | model {tracking_fps:.1f} FPS | stream {stream_fps:.1f} FPS",
                    28,
                )
                _put_text(
                    visualization,
                    f"t = [{pose[0, 3]:+.3f}, {pose[1, 3]:+.3f}, {pose[2, 3]:+.3f}] m",
                    55,
                )
                if args.save_video:
                    if video_writer is None:
                        video_writer = _open_video_writer(
                            args.video_path, visualization, args.video_fps
                        )
                    video_frame = visualization.copy()
                    video_frames += 1
                if not args.no_display:
                    cv2.imshow(window_name, visualization)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        quit_requested = True
                    if key == ord("r"):
                        registered = False
                        print("Re-registration requested; reloading mask on next frame.")
                render_seconds.append(time.perf_counter() - render_started)

            pose_path = (
                args.pose_output_dir / f"{output_index:06d}.txt"
                if args.pose_output_dir is not None and pose is not None
                else None
            )
            if pose_path is not None or video_frame is not None:
                output_queue_wait_seconds.append(
                    output_executor.submit(
                        _write_frame_outputs,
                        pose_path,
                        None if pose is None else pose.copy(),
                        video_writer,
                        video_frame,
                    )
                )

            if args.no_display and (
                frame_count == 1
                or frame_count % args.fps == 0
                or (args.max_frames > 0 and frame_count == args.max_frames)
            ):
                assert pose is not None
                print(
                    f"frame={frame_count} mode={mode.lower()} state={tracker_state} "
                    f"tracking_fps={tracking_fps:.2f} stream_fps={stream_fps:.2f} "
                    f"t_m=[{pose[0, 3]:+.4f},{pose[1, 3]:+.4f},{pose[2, 3]:+.4f}]"
                )
            if quit_requested:
                break

        loop_finished = time.perf_counter()
        kalman_report = estimator.get_kalman_stats()
        print(
            f"Stopped after {successful_frames} inference frames; "
            f"model tracking FPS={len(track_seconds) / sum(track_seconds) if track_seconds else 0.0:.2f}; "
            f"final stream FPS={stream_fps:.2f}"
        )
    except KeyboardInterrupt:
        interrupted = True
        loop_finished = time.perf_counter()
        kalman_report = estimator.get_kalman_stats()
    finally:
        output_flush_started = time.perf_counter()
        try:
            output_executor.close()
        except BaseException as error:
            output_error = error
        output_flush_seconds = time.perf_counter() - output_flush_started
        if rgbd_executor is not None:
            rgbd_flush_started = time.perf_counter()
            try:
                rgbd_executor.close()
            except BaseException as error:
                if output_error is None:
                    output_error = error
            rgbd_flush_seconds = time.perf_counter() - rgbd_flush_started
        if video_writer is not None:
            video_writer.release()
            print(f"Tracking video saved: {args.video_path}")
        close_estimator = getattr(estimator, "close", None)
        if callable(close_estimator):
            close_estimator()
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()
        if output_error is not None:
            raise output_error

    run_seconds = max(0.0, loop_finished - run_started)
    tracking_fps = (
        len(track_seconds) / sum(track_seconds) if track_seconds else 0.0
    )
    tracking_end_to_end_fps = 0.0
    if len(track_completion_times) >= 2:
        tracking_elapsed = track_completion_times[-1] - track_completion_times[0]
        if tracking_elapsed > 0:
            tracking_end_to_end_fps = (
                len(track_completion_times) - 1
            ) / tracking_elapsed
    report = {
        "frames": successful_frames,
        "captured_frames": frame_count,
        "registration_count": registration_count,
        "register_seconds": float(register_seconds),
        "tracking_frames": len(track_seconds),
        "tracking_seconds": float(sum(track_seconds)),
        "tracking_fps": float(tracking_fps),
        "tracking_end_to_end_fps": float(tracking_end_to_end_fps),
        "end_to_end_fps": (
            float(successful_frames / run_seconds) if run_seconds else 0.0
        ),
        "final_stream_fps": float(stream_fps),
        "run_seconds": float(run_seconds),
        "input_timing": _timing_percentiles(input_seconds),
        "refined_timing": _timing_percentiles(refined_track_seconds),
        "predicted_timing": _timing_percentiles(predicted_track_seconds),
        "render_timing": _timing_percentiles(render_seconds),
        "kalman": kalman_report,
        "robust_tracking": {
            "enabled": bool(args.robust_tracking),
            "relocalization_enabled": bool(args.robust_relocalization),
            "diagnostics": robust_diagnostics,
            "warning": "verified denotes sparse depth support, not calibrated correctness; raw neural poses are saved even when unverified",
        },
        "deployment": {
            "compile_track_refiner": bool(args.compile_track_refiner),
            "compile_warmup_seconds": float(
                estimator._tracking_compile_seconds
            ),
            "track_refine_iter": int(args.track_refine_iter),
            "every_frame_learned_6d": bool(not args.use_kalman),
        },
        "camera": {
            "name": name,
            "serial": serial,
            "width": int(intrinsics.width),
            "height": int(intrinsics.height),
            "requested_fps": int(args.fps),
        },
        "mesh": {
            "path": str(mesh_path),
            "vertices": initial_mesh_vertices,
            "faces": initial_mesh_faces,
        },
        "output": {
            "queue_size": int(args.output_queue_size),
            "queue_wait_seconds": float(sum(output_queue_wait_seconds)),
            "flush_seconds": float(output_flush_seconds),
            "pose_directory": (
                str(args.pose_output_dir)
                if args.pose_output_dir is not None
                else None
            ),
        },
        "rgbd": (
            {
                "directory": str(args.rgbd_output_dir),
                "frames": int(rgbd_frames),
                "start_index": int(rgbd_start_index),
                "png_compression": int(args.rgbd_png_compression),
                "workers": int(args.rgbd_output_workers),
                "queue_size": int(args.output_queue_size),
                "queue_wait_seconds": float(sum(rgbd_queue_wait_seconds)),
                "flush_seconds": float(rgbd_flush_seconds),
            }
            if args.rgbd_output_dir is not None
            else None
        ),
        "video": (
            {
                "path": str(args.video_path),
                "fps": float(args.video_fps),
                "frames": int(video_frames),
            }
            if video_writer is not None
            else None
        ),
    }
    logging.info("benchmark: %s", json.dumps(report, sort_keys=True))
    if args.benchmark_report is not None:
        args.benchmark_report.parent.mkdir(parents=True, exist_ok=True)
        args.benchmark_report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Benchmark report saved: {args.benchmark_report}")
    if interrupted:
        raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Real-time FoundationPose error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
