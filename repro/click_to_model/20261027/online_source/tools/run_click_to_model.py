#!/usr/bin/env python3
"""Run mesh-conditioned Click-to-Model; no runtime GT or future-mask access.

RGB/depth must have exactly matching basenames; depth scale is explicit. A first
frame mask or SAM2 point prompt initializes the target. Results include all
frames, raw neural iterations, source hashes and synchronized wall latency.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np

MEASURED_SOURCE_FILES = (
    "measured_motion_session.py", "relative_motion_refinement.py", "tools/tapnext_stream.py",
    "tools/tentative_point_motion.py", "tools/preloss_measured_anchors.py", "tools/preloss_anchor_support.py",
)
OUTPUT_POLICY_SOURCE_FILES = ("identity_continuity.py", "identity_output.py")
TENTATIVE_TRUST_FIELDS = ("verified", "identity_verified", "initial_orientation_resolved",
    "absolute_orientation_resolved", "trusted_memory_updated", "memory_updated", "seed_admitted")


def add_tracking_policy_arguments(parser):
    """Shared CLI contract for the mesh-conditioned and meshless entrypoints."""
    parser.add_argument("--tracking-policy", choices=("legacy", "tentative-preloss"), default="legacy",
                        help="explicit measured-motion tentative session; legacy behavior remains the default")
    parser.add_argument("--working-mask-selection", choices=("legacy-geometry", "sam-score"),
                        default="legacy-geometry", help="tentative working pose only; admission and trust flags are unchanged")
    parser.add_argument("--output-policy", choices=("legacy", "continuity"), default="legacy",
                        help="continuity publishes a tentative pose or abstains; raw candidates remain evidence")
    parser.add_argument("--tapnext-source-root", type=Path, help="local pinned official TAPNext source")
    parser.add_argument("--tapnext-checkpoint", type=Path, help="local TAPNext++ checkpoint; no downloads")
    parser.add_argument("--tapnext-checkpoint-sha256", help="SHA256 of the explicitly supplied checkpoint")


def validate_tracking_policy_arguments(args, error):
    """Reject incompatible/unused options before imports, model loads or stages."""
    working_selection = getattr(args, "working_mask_selection", "legacy-geometry")
    output_policy = getattr(args, "output_policy", "legacy")
    if output_policy not in ("legacy", "continuity"):
        error("unknown output-policy")
    if output_policy == "continuity" and args.tracking_policy != "tentative-preloss":
        error("continuity output-policy requires --tracking-policy tentative-preloss")
    if working_selection not in ("legacy-geometry", "sam-score"):
        error("unknown working-mask-selection policy")
    options = (args.tapnext_source_root, args.tapnext_checkpoint, args.tapnext_checkpoint_sha256)
    if args.tracking_policy == "legacy":
        if working_selection != "legacy-geometry":
            error("sam-score working-mask-selection requires --tracking-policy tentative-preloss")
        if any(value is not None for value in options):
            error("TAPNext options require --tracking-policy tentative-preloss")
        return
    if any(value is None for value in options):
        error("tentative-preloss requires --tapnext-source-root, --tapnext-checkpoint and --tapnext-checkpoint-sha256")
    digest = args.tapnext_checkpoint_sha256
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        error("tapnext-checkpoint-sha256 must be 64 lowercase hexadecimal characters")
    if getattr(args, "mode", "adaptive") != "adaptive":
        error("tentative-preloss requires --mode adaptive")
    if getattr(args, "seed_backend", None) is not None:
        error("--seed-backend is a legacy correspondence policy and conflicts with tentative-preloss")
    if getattr(args, "initialization_admission", None) not in (None, "all-hypotheses"):
        error("tentative-preloss preserves all registration hypotheses; explicit legacy admission conflicts")
    if getattr(args, "motion_prior", "none") != "none":
        error("tentative-preloss cannot combine with a committed-memory motion prior")
    if getattr(args, "quarantined_recovery", False) or getattr(args, "disable_urgent_replenishment", False):
        error("tentative-preloss cannot combine with quarantine or legacy replenishment options")


def create_measured_session(args, checker, diameter, session_id, directory, refinement_config):
    """One model, fresh original/secondary recurrences, shared product session."""
    from measured_motion_session import DirectoryEvidenceSink, MeasuredMotionSession
    from tools.tapnext_stream import TAPNextConfig, TAPNextPointStream, TorchTAPNextPPBackend
    config = TAPNextConfig(source_root=str(args.tapnext_source_root.resolve()),
        checkpoint=str(args.tapnext_checkpoint.resolve()), checkpoint_sha256=args.tapnext_checkpoint_sha256,
        device="cuda", input_resolution=256, precision="float32")
    sink = DirectoryEvidenceSink(directory)
    policy = None
    if getattr(args, "output_policy", "legacy") == "continuity":
        from identity_output import OutputPolicyAdapter
        policy = OutputPolicyAdapter(session_id, evidence_sink=sink)
        sink = policy.observe
    try:
        backend = TorchTAPNextPPBackend(config)
        owner = MeasuredMotionSession(checker, diameter, session_id=session_id,
            stream_factory=lambda: TAPNextPointStream(config, backend),
            evidence_sink=sink, refinement_config=refinement_config)
    except BaseException:
        if policy is not None:
            policy.close()
        raise
    if policy is not None:
        owner.output_policy = policy
    return owner, backend


def publish_measured_output(owner, result, actual_original_pose, *, frame_id, slots, arrays):
    """Persist candidate bytes separately; only a successful decision publishes.

    The session remains the owner of its working candidate and recurrence. A
    policy/sink failure terminates the output epoch and leaves this slot failed.
    """
    policy = owner.output_policy
    slot = slots[frame_id]
    try:
        candidate = numeric_return(actual_original_pose)
        candidate_key = f"candidate_frame_{frame_id:06d}"
        arrays[candidate_key] = candidate.copy()
        slot.update(status="decision_started", candidate_array=candidate_key,
                    candidate_identity=array_identity(candidate))
        decision = policy.publish(result, candidate)
        if (decision.get("frame_id"), decision.get("source_frame_id")) != (frame_id, frame_id):
            raise ValueError("output publication clock differs from actual entry frame")
        if any(decision.get(key) is not False for key in TENTATIVE_TRUST_FIELDS):
            raise ValueError("output policy must preserve every tentative trust flag")
        if array_identity(decision["candidate_pose"]) != slot["candidate_identity"]:
            raise ValueError("output policy changed actual candidate bytes")
        available = decision["pose_available"]
        if type(available) is not bool:
            raise ValueError("output pose availability must be an explicit bool")
        published = decision["published_pose"]
        if available:
            if published is None or array_identity(published) != slot["candidate_identity"]:
                raise ValueError("published pose must retain actual candidate dtype and bytes")
        elif published is not None:
            raise ValueError("abstention must not expose a published pose")
        metadata = {k:v for k,v in decision.items() if k not in ("candidate_pose", "published_pose")}
        # Ensure the consumer record can be serialized before marking this
        # entry slot successful; a malformed result never leaks an output.
        json.dumps(metadata, allow_nan=False)
        published_key = f"published_frame_{frame_id:06d}" if available else None
        if available:
            arrays[published_key] = numeric_return(published)
        slot.update(status="decided", decision=metadata, pose_available=available,
                    published_array=published_key)
        return decision
    except BaseException as exc:
        slot.update(status="failed", pose_available=False, published_array=None,
                    error=f"{type(exc).__name__}: {exc}")
        arrays.pop(f"published_frame_{frame_id:06d}", None)
        for label, close in (("policy", policy.close), ("session", lambda: owner.close(reason="output_policy_failed"))):
            try:
                close()
            except BaseException as close_error:
                slot.setdefault("close_errors", []).append(f"{label}: {type(close_error).__name__}: {close_error}")
        raise


def save_output_policy(output, slots, arrays):
    """Full planned clock plus per-frame numeric arrays, never pickle/NaN fill."""
    payload = dict(arrays)
    payload["pose_available"] = np.asarray([s["pose_available"] is True for s in slots], dtype=bool)
    payload["decision_recorded"] = np.asarray([s["status"] == "decided" for s in slots], dtype=bool)
    np.savez_compressed(output / "published_poses.npz", **payload)
    write_json(output / "output_policy.json", dict(schema="click_to_model_output_policy/1", policy="continuity",
        planned_frames=len(slots), decided_frames=sum(s["status"] == "decided" for s in slots),
        pose_available_frames=sum(s["pose_available"] is True for s in slots),
        identity_verified=False, raw_candidate_file="poses.npy", numeric_file="published_poses.npz",
        array_contract="candidate_frame_* preserves each attempted decision's actual pose; published_frame_* exists only when available; decision_recorded separates abstention from unexecuted/failed slots",
        frames=slots))


def numeric_return(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "bfiuc":
        raise TypeError("actual callback evidence must be numeric; no object/pickle arrays")
    return array.copy()


def array_identity(value):
    value = np.ascontiguousarray(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def validate_measured_result(owner, result, *, stage):
    diagnostics = result.get("diagnostics") if isinstance(result, dict) else None
    if not isinstance(diagnostics, dict) or any(diagnostics.get(key) is not False for key in TENTATIVE_TRUST_FIELDS):
        owner.record_evidence("entry_rejected_result", dict(stage=stage, result=result,
            reason="measured session result must preserve every tentative trust flag"))
        raise ValueError("measured session result must preserve every tentative trust flag")


def measured_refiner(estimator, rgb, depth, K, *, centered_from_original, frame_id, records):
    """Actual FoundationPose seam; retain inputs/outputs even if a later save fails."""
    registered_centering = numeric_return(centered_from_original)
    if (registered_centering.shape != (4, 4) or registered_centering.dtype.kind not in "fi"
            or not np.isfinite(registered_centering).all()):
        raise ValueError("registered centering must be a finite real 4x4 matrix")
    def refine_one(seed):
        record = {"frame_id": frame_id, "iteration": len(records),
                  "supplied_centered_seed": numeric_return(seed), "returned": False,
                  "rgb_identity": array_identity(rgb), "depth_m_identity": array_identity(depth),
                  "K_identity": array_identity(K), "registered_centering": registered_centering.copy()}
        records.append(record)
        try:
            estimator._pose_last_from_numpy(seed)
            record["installed_centered_seed"] = numeric_return(estimator.pose_last)
            actual_original = estimator.track_one(rgb, depth, K, iteration=1)
            record["track_one_returned"] = True
            record["actual_original_return"] = numeric_return(actual_original)
            record["actual_centered_return"] = numeric_return(estimator.pose_last)
            actual_original = record["actual_original_return"]
            raw_centered = record["actual_centered_return"]
            if (actual_original.shape != (4, 4) or actual_original.dtype.kind not in "fi"
                    or not np.isfinite(actual_original).all()):
                raise ValueError("actual FP original return must be a finite real 4x4 matrix; raw value retained")
            if (raw_centered.shape not in ((4, 4), (1, 4, 4)) or raw_centered.dtype.kind not in "fi"
                    or not np.isfinite(raw_centered).all()):
                raise ValueError("actual FP centered return must be finite real 4x4 or singleton batch 1x4x4; raw value retained")
            # FoundationPose.track_one stores the refiner's explicit batch of
            # one in pose_last; its public original-coordinate return is 4x4.
            # Preserve that raw contract and select only this known singleton.
            if raw_centered.shape == (1, 4, 4):
                record["centered_return_layout"] = "singleton_batch_1x4x4"
                record["controller_centered_return"] = raw_centered[0].copy()
            else:
                record["centered_return_layout"] = "single_4x4"
                record["controller_centered_return"] = raw_centered.copy()
            derived = record["controller_centered_return"] @ registered_centering
            record["conversion_max_abs_difference"] = float(np.max(np.abs(record["actual_original_return"]-derived)))
            record["conversion_tolerance"] = 2e-7
            if not np.allclose(record["actual_original_return"], derived, rtol=0, atol=2e-7):
                raise ValueError("actual FP callback original return differs from centered return and registered centering")
            record["conversion_checked"] = True
            record["returned"] = True
            return record["controller_centered_return"].copy()
        except BaseException as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
    return refine_one


def save_measured_callback_prefix(output, frames):
    """Separate actual callback ledger, including a failing frame's prefix."""
    arrays, metadata = {}, []
    for frame_id, records in frames:
        for index, record in enumerate(records):
            row = {}
            for key, value in record.items():
                if isinstance(value, np.ndarray):
                    name = f"frame_{frame_id:06d}_call_{index:02d}_{key}"
                    arrays[name] = value
                    row[key] = name
                else:
                    row[key] = value
            metadata.append(row)
    np.savez_compressed(output / "actual_fp_callback_prefix.npz", **arrays)
    write_json(output / "actual_fp_callback_prefix.json", metadata)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def frame_names(sequence):
    rgb = sorted(p.name for p in (sequence / "rgb").glob("*.png"))
    depth = sorted(p.name for p in (sequence / "depth").glob("*.png"))
    if not rgb or rgb != depth:
        raise ValueError("require >=1 RGB/depth PNGs with exactly equal ordered basenames")
    return rgb


def read_frame(sequence, name, depth_scale):
    color = cv2.imread(str(sequence / "rgb" / name), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(sequence / "depth" / name), cv2.IMREAD_UNCHANGED)
    if color is None or depth is None:
        raise ValueError(f"unreadable frame {name}")
    if depth.ndim != 2 or depth.shape != color.shape[:2]:
        raise ValueError("depth and RGB must be registered at the same resolution")
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB), depth.astype(np.float32) * depth_scale


def sam2_mask_candidates(rgb, xy, checkpoint):
    import torch
    sys.path.insert(0, str(ROOT / "third_party/sam2"))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml",
                                             str(checkpoint), device="cuda"))
    with torch.inference_mode():
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(point_coords=np.asarray([xy]),
                                             point_labels=np.ones(1, dtype=int),
                                             multimask_output=True)
    return masks.astype(bool), scores


def capture_backend_registration(estimator, invoke, *, directory, context):
    """Retain actual backend arrays before checking them, including late failures.

    FoundationPose exposes score-sorted ``poses`` and ``scores``; its ``best_id``
    indexes the earlier unsorted batch and must never index these saved arrays.
    ``invoke`` includes device synchronization. No array is filled from an older
    registration when the backend fails before publishing a new result.
    """
    from initialization_admission import RegistrationHypothesisBundle, checker_fingerprint, _digest
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    def observation_binding():
        return _digest({"context": {key: value for key, value in context.items() if key != "checker"},
                        "checker": checker_fingerprint(context["checker"], context["cad_identity"])})
    before_binding = observation_binding()
    estimator.pose_last = estimator.poses = estimator.scores = None
    raw_pose, failure = None, None
    try:
        raw_pose = invoke()
    except BaseException as exc:
        failure = exc
    arrays, capture_errors = {}, []
    for key, value in (("returned_pose_original", raw_pose),
                       ("poses_centered", estimator.poses), ("scores", estimator.scores),
                       ("pose_last_centered", estimator.pose_last),
                       ("centered_from_original", context["centered_from_original"])):
        if value is None:
            continue
        try:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            value = np.asarray(value).copy()
            if value.dtype.kind not in "fiu":
                raise TypeError("backend artifact must be a real numeric array")
            arrays[key] = value
        except Exception as exc:
            capture_errors.append(f"{key}: {type(exc).__name__}: {exc}")
    np.savez_compressed(directory / "backend_arrays.npz", **arrays)
    observation_unchanged = False
    try:
        observation_unchanged = before_binding == observation_binding()
        if not observation_unchanged:
            raise ValueError("registration callback mutated observation/checker/CAD context")
    except Exception as exc:
        capture_errors.append(f"observation binding: {type(exc).__name__}: {exc}")
    error = None if failure is None else f"{type(failure).__name__}: {failure}"
    if capture_errors:
        error = "; ".join(([error] if error else []) + capture_errors)
    bundle = None
    try:
        if not observation_unchanged:
            raise ValueError("no registration bundle may bind a mutated observation")
        bundle = RegistrationHypothesisBundle.capture(
            **context, poses_centered=arrays.get("poses_centered"), scores=arrays.get("scores"),
            returned_pose_original=arrays.get("returned_pose_original"),
            selected_backend_index=0 if raw_pose is not None else None,
            order="est.poses and est.scores descending score order; best_id is not this index", error=error)
    except Exception as exc:
        capture_errors.append(f"bundle: {type(exc).__name__}: {exc}")
    write_json(directory / "capture.json", {
        "schema": "foundationpose_registration_capture/1", "mask_id": context["mask_id"],
        "frame_id": context["frame_id"], "cad_identity": context["cad_identity"],
        "pre_call_observation_binding": before_binding, "observation_unchanged": observation_unchanged,
        "bundle_digest": None if bundle is None else bundle.content_digest,
        "error": error, "capture_errors": capture_errors,
        "backend_arrays_sha256": sha256(directory / "backend_arrays.npz"),
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
        "complete_backend_output_claim": "Only actual exposed est.poses/scores; missing arrays remain missing",
        "selected_backend_index": 0 if raw_pose is not None else None})
    if failure is not None and not isinstance(failure, Exception):
        raise failure
    return {"bundle": bundle, "error": error or ("; ".join(capture_errors) or None),
            "raw_pose": arrays.get("returned_pose_original"),
            "centered_pose": arrays.get("pose_last_centered")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--click", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--sam2-checkpoint", type=Path,
                        default=ROOT / "third_party/sam2/checkpoints/sam2.1_hiera_small.pt")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--depth-scale", type=float, default=.001)
    parser.add_argument("--max-frames", type=int, default=0, help="0: all; prefixes are development only")
    parser.add_argument("--register-iterations", type=int, default=5)
    parser.add_argument("--max-refine-iterations", type=int, default=3)
    parser.add_argument("--mode", choices=("baseline", "validity", "rgbd", "adaptive"), default="adaptive")
    parser.add_argument("--symmetries", type=Path, help="BOP models_info object entry JSON; translations in metres")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-backend", choices=("lk", "trusted-view"), default=None,
                        help="RGB-D correspondence provider; trusted-view adds admitted CAD view recovery")
    parser.add_argument("--motion-prior", choices=("none", "constant_velocity"), default="none",
                        help="optional seed admission reference from two consecutive commits; assumes equal input spacing")
    parser.add_argument("--quarantined-recovery", action="store_true",
                        help="confirm a temporally rejected RGB-D candidate across three fresh neural observations")
    parser.add_argument("--disable-urgent-replenishment", action="store_true",
                        help="ablate early replenishment when committed point support falls below twice the minimum")
    parser.add_argument("--initialization-admission", choices=("legacy", "all-hypotheses"), default=None,
                        help="all-hypotheses checks every registration candidate before trusted memory; unresolved poses stay tentative")
    add_tracking_policy_arguments(parser)
    args = parser.parse_args(argv)
    validate_tracking_policy_arguments(args, parser.error)
    measured = args.tracking_policy == "tentative-preloss"
    continuity = args.output_policy == "continuity"
    if args.initialization_admission is None:
        args.initialization_admission = "all-hypotheses" if measured else "legacy"
    if args.seed_backend is None and not measured:
        args.seed_backend = "lk"
    if not np.isfinite(args.depth_scale) or args.depth_scale <= 0:
        parser.error("depth-scale must be finite positive")
    if args.mask is not None and args.click is not None:
        parser.error("select mask or click, not both")
    if args.max_frames < 0:
        parser.error("max-frames must be 0 or >=1")
    if args.register_iterations < 1 or not 1 <= args.max_refine_iterations <= 10:
        parser.error("require register-iterations>=1 and 1<=max-refine-iterations<=10")
    if args.quarantined_recovery and args.mode not in ("rgbd", "adaptive"):
        parser.error("quarantined-recovery requires rgbd or adaptive mode")
    if args.initialization_admission == "all-hypotheses" and args.mode == "baseline":
        parser.error("all-hypotheses requires validity, rgbd or adaptive mode")
    names = frame_names(args.sequence)
    total_frames = len(names)
    if args.max_frames:
        names = names[:args.max_frames]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_files = [ROOT / n for n in ("click_to_model.py", "click_initialization.py", "pose_validity.py", "rgbd_seed.py", "trusted_view_seed.py", "quarantined_recovery.py",
                    "adaptive_refinement.py", "initialization_admission.py", "initialization_session.py", "tentative_tracking.py",
                    "estimater.py", "Utils.py", "datareader.py", "pose_kalman.py", "stage_profiler.py")]
    source_files += sorted((ROOT / "learning").rglob("*.py"))
    if measured:
        source_files += [ROOT / name for name in MEASURED_SOURCE_FILES]
    if continuity:
        source_files += [ROOT / name for name in OUTPUT_POLICY_SOURCE_FILES]
    source_files += [Path(__file__).resolve()]
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in source_files}
    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report = {"schema": "click_to_model_run/1", "status": "started",
              "created_utc": datetime.now(timezone.utc).isoformat(), "arguments": args_dict,
              "source_sha256": source_hashes, "frame_names": names,
              "sequence_frames": total_frames, "development_prefix": len(names) != total_frames,
              "ground_truth_used_at_runtime": False,
              "timing_boundary": "decode + RGBD seed + neural + validity + CUDA sync; excludes output serialization",
              "identity_accuracy_claimed": False, "output_policy": args.output_policy}
    if measured:
        report.update(tracking_policy="tentative-preloss", verified=False, identity_verified=False,
            initial_orientation_resolved=False, absolute_orientation_resolved=False,
            trusted_memory_initialized=False, trusted_memory_updated=False, memory_updated=False, seed_admitted=False,
            source_frame_id_convention="zero-based ordered input sequence index; original basename retained; no hardware timestamp inferred",
            planned_frame_slots=[dict(frame_id=i, source_frame_id=i, name=name, status="not_executed")
                                 for i, name in enumerate(names)],
            timing_boundary="decode + shared measured-motion session + actual adaptive FP + CUDA sync; session evidence included, final output serialization excluded")
    if continuity:
        report["timing_boundary"] += "; current-frame continuity decision and publication evidence included"
    write_json(output / "plan.json", report)
    poses, details, timings, raw = [], [], [], []
    output_slots = [dict(frame_id=i, source_frame_id=i, name=name, status="not_executed",
                         pose_available=None, published_array=None) for i,name in enumerate(names)] if continuity else []
    output_arrays = {}
    initialization_start = None
    measured_owner = point_backend = None
    callback_frames = []
    try:
        import torch
        import trimesh
        from estimater import FoundationPose, set_seed, logging
        from click_to_model import ClickToModelTracker, ClickToModelConfig
        from adaptive_refinement import AdaptiveRefinementConfig
        from pose_validity import PoseValidityChecker, SymmetrySpec
        from click_initialization import initialize_candidates, select_sam_working_candidate
        logging.getLogger().setLevel(logging.WARNING)
        cv2.setNumThreads(1)
        set_seed(args.seed)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; run in the FoundationPose GPU environment")
        torch.cuda.synchronize()
        initialization_start = time.perf_counter()
        if measured:
            report["planned_frame_slots"][0]["status"] = "started"
        K = np.loadtxt(args.sequence / "cam_K.txt").reshape(3, 3)
        mesh = trimesh.load(str(args.mesh), force="mesh")
        rgb, depth = read_frame(args.sequence, names[0], args.depth_scale)
        if args.click is not None:
            masks, sam_scores = sam2_mask_candidates(rgb, args.click, args.sam2_checkpoint)
            mask_source = "SAM2.1_small_first_frame_click"
        else:
            mask_path = args.mask or args.sequence / "masks" / names[0]
            mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask_image is None or mask_image.ndim != 2 or mask_image.shape != depth.shape:
                raise ValueError("first mask must be a readable single-channel registered mask")
            masks, sam_scores = [mask_image > 0], [None]
            mask_source = "supplied_first_frame_mask"
        est = FoundationPose(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(),
                             mesh=mesh, debug=0, debug_dir=str(output / "debug"))
        model_points, face_ids = trimesh.sample.sample_surface(est.mesh, 12000, seed=args.seed)
        normals = est.mesh.face_normals[face_ids]
        symmetries = None
        if args.symmetries:
            symmetries = SymmetrySpec.from_bop(json.loads(args.symmetries.read_text()),
                                              model_center=est.model_center, unit_scale=1.0)
        checker = PoseValidityChecker(model_points, normals, est.diameter, symmetries=symmetries)
        seed_factory = None
        if args.seed_backend == "trusted-view":
            from trusted_view_seed import TrustedViewSeedTracker
            seed_factory = lambda: TrustedViewSeedTracker(checker=checker)
        tracking_config = ClickToModelConfig(enable_rgbd_seed=args.mode in ("rgbd", "adaptive"),
                                             motion_prior=args.motion_prior,
                                             enable_quarantined_recovery=args.quarantined_recovery,
                                             urgent_replenishment=not args.disable_urgent_replenishment)
        refinement_config = AdaptiveRefinementConfig(
                        max_iterations=args.max_refine_iterations if args.mode == "adaptive" else 1)
        tracker = driver = None
        if args.initialization_admission == "legacy" and not measured:
            tracker = ClickToModelTracker(checker, est.diameter,
                seed_tracker=None if seed_factory is None else seed_factory(),
                anchor_model_points=model_points, config=tracking_config, refinement_config=refinement_config)
            driver = tracker
            report["configuration"] = tracker.configuration()
        report["gpu"] = torch.cuda.get_device_name(0)
        report["versions"] = {"torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__}
        report["initialization"] = mask_source
        report["mesh_sha256"] = sha256(args.mesh)
        report["runtime_model_diameter_m"] = float(est.diameter)
        centering = est.get_tf_to_centered_mesh().detach().cpu().numpy().copy()
        report["centered_from_original"] = centering.tolist()
        report["K"] = K.tolist()
        cad_binding = {"mesh_file_sha256": report["mesh_sha256"],
            "centered_vertices": array_identity(est.mesh.vertices),
            "faces": array_identity(est.mesh.faces), "normals": array_identity(est.mesh.vertex_normals),
            "checker_points": array_identity(checker.points), "checker_normals": array_identity(checker.normals),
            "centered_from_original": array_identity(centering), "diameter_m": float(est.diameter)}
        if measured:
            cad_binding.update(checker_tolerance_m=float(checker.tolerance), checker_configuration=checker.configuration())
        report["runtime_cad_binding"] = cad_binding
        cad_identity = hashlib.sha256(json.dumps(cad_binding, sort_keys=True, allow_nan=False).encode()).hexdigest()
        session_id, bundles = str(output), []
        registration_returns = {}
        def register_candidate(candidate_mask):
            # Failed registration must not accidentally retain an earlier
            # candidate's pose. All candidates use the same frame and network.
            est.pose_last = None
            torch.cuda.synchronize()
            raw_pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=candidate_mask,
                                    iteration=args.register_iterations)
            torch.cuda.synchronize()
            candidate_pose = (None if est.pose_last is None else
                              est.pose_last.detach().cpu().numpy().reshape(4, 4).copy())
            return np.asarray(raw_pose).reshape(4, 4).copy(), candidate_pose
        def register_indexed_candidate(mask_id, candidate_mask):
            def invoke():
                torch.cuda.synchronize()
                value = est.register(K=K, rgb=rgb, depth=depth, ob_mask=candidate_mask,
                                     iteration=args.register_iterations)
                torch.cuda.synchronize()
                return value
            captured = capture_backend_registration(est, invoke,
                directory=output / "registration_hypotheses" / f"mask_{mask_id:02d}",
                context=dict(session_id=session_id, frame_id=0, mask_id=mask_id,
                    centered_from_original=centering, rgb=rgb, depth=depth, K=K,
                    mask=candidate_mask, checker=checker, cad_identity=cad_identity))
            registration_returns[mask_id] = captured
            if captured["bundle"] is not None:
                bundles.append(captured["bundle"])
            if captured["error"]:
                raise RuntimeError(captured["error"])
            return captured["raw_pose"], captured["centered_pose"]
        all_hypotheses = args.initialization_admission == "all-hypotheses"
        initialization = initialize_candidates(
            masks, sam_scores, register_indexed_candidate if all_hypotheses else register_candidate,
            lambda candidate_pose, candidate_mask: checker.evaluate(
                candidate_pose, rgb, depth, K, mask=candidate_mask),
            output / "initialization_candidates", indexed_register=all_hypotheses)
        report["initialization_candidates"] = initialization
        report["registration_seconds"] = initialization["registration_seconds_total"]
        selected = initialization["selected_index"]
        if all_hypotheses:
            from initialization_admission import assess_initial_hypotheses
            if not np.array_equal(centering, est.get_tf_to_centered_mesh().detach().cpu().numpy()):
                raise ValueError("FoundationPose CAD centering changed during registration")
            assessment_start = time.perf_counter()
            mask_mapping = {i: np.asarray(m, dtype=bool) for i, m in enumerate(masks)}
            admission = assess_initial_hypotheses(bundles, session_id=session_id, frame_id=0,
                rgb=rgb, depth=depth, K=K, masks=mask_mapping, checker=checker, cad_identity=cad_identity)
            report["admission_seconds"] = time.perf_counter() - assessment_start
            write_json(output / "initialization_admission.json", admission.as_record())
        if measured:
            # The assessment is evidence. It cannot install a trusted tracker in
            # this policy, including when the assessment says accepted.
            if args.working_mask_selection == "sam-score":
                working_selection = select_sam_working_candidate(initialization["candidates"],
                    registration_returns, centered_from_original=centering,
                    supplied_mask=mask_source == "supplied_first_frame_mask")
                chosen = next((b for b in bundles if b.mask_id == working_selection["selected_index"]), None)
            else:
                # Preserve the existing default, including its original fallback.
                candidates = sorted(bundles, key=lambda b: (b.mask_id != selected, b.mask_id))
                chosen = next((b for b in candidates if b.returned_pose_original is not None and b.error is None), None)
                working_selection = dict(schema="tentative_working_mask_selection/1",
                    strategy="legacy-geometry", selected_index=None if chosen is None else chosen.mask_id,
                    selection_rule="legacy observation-selected mask when available, otherwise first successful mask; actual FP selected index; tentative only",
                    fallback_used=chosen is not None and chosen.mask_id != selected,
                    admission_changed=False, identity_verified=False,
                    trusted_memory_initialized=False, trusted_memory_updated=False)
            report["working_mask_selection"] = working_selection
            write_json(output / "working_mask_selection.json", working_selection)
            if chosen is None:
                raise ValueError("no actual original registration return for tentative continuation; all raw banks retained")
            selected = chosen.mask_id
            selected_backend_index = chosen.selected_backend_index
            selected_arrays = output / "registration_hypotheses" / f"mask_{selected:02d}" / "backend_arrays.npz"
            with np.load(selected_arrays, allow_pickle=False) as saved:
                bank = saved["poses_centered"].copy() if "poses_centered" in saved else None
                actual_pose_last = saved["pose_last_centered"].copy() if "pose_last_centered" in saved else None
            if actual_pose_last is None:
                raise ValueError("actual selected registration working pose required")
            pose = numeric_return(chosen.returned_pose_original)
            centered = actual_pose_last
            if pose.shape != (4, 4) or centered.shape != (4, 4):
                raise ValueError("actual selected registration returns must be 4x4; no shape repair")
            if (bank is not None and bank.ndim == 3 and bank.shape[1:] == (4, 4)
                    and type(selected_backend_index) is int and 0 <= selected_backend_index < len(bank)
                    and not np.array_equal(bank[selected_backend_index], centered, equal_nan=True)):
                raise ValueError("actual pose_last differs from the captured selected backend hypothesis")
            mask = np.asarray(masks[selected], dtype=bool)
            metadata = dict(selected_mask_id=selected, selected_backend_index=selected_backend_index,
                original_mask_selection=initialization["selected_index"], mask_selection_rule=initialization["selection_rule"],
                all_registration_bundles=[dict(mask_id=b.mask_id, digest=b.content_digest,
                    selected_backend_index=b.selected_backend_index, error=b.error) for b in bundles],
                original_admission_assessment=admission.as_record(),
                runtime_cad_binding=cad_binding, source_name=names[0],
                working_mask_selection=working_selection,
                selection_rule=working_selection["selection_rule"])
            measured_owner, point_backend = create_measured_session(args, checker, est.diameter,
                session_id, output / "measured_motion", refinement_config)
            driver = tracker = measured_owner
            report["planned_frame_slots"][0]["status"] = "started"
            initial_result = driver.start(initial_hypotheses=bank, selected_backend_index=selected_backend_index,
                centered_from_original=centering, rgb=rgb, depth_m=depth, K=K, mask=mask,
                source_frame_id=0, timestamp=None, initial_pose_original=pose, registration_metadata=metadata)
            validate_measured_result(driver, initial_result, stage="start")
            initial = initial_result["diagnostics"]
            if not continuity:
                report["planned_frame_slots"][0]["status"] = "completed"
            report["initialization_admission"] = dict(accepted=admission.accepted, reason=admission.reason,
                digest=admission.content_digest, hypothesis_scope=admission.hypothesis_scope,
                selected_mask_id=selected, selected_backend_index=selected_backend_index,
                selected_source="actual_backend_selected_working_only", working_policy="tentative-preloss",
                working_mask_selection=args.working_mask_selection,
                assessment_installs_trusted_memory=False, identity_verified=False)
            registration_evidence = initial
        elif all_hypotheses:
            from initialization_session import InitializationTrackingSession, InitializationWorkingFallback
            fallback = None
            if not admission.accepted and admission.working_index is None:
                fallback_bundles = sorted(bundles, key=lambda b: (b.mask_id != selected, b.mask_id))
                for bundle in fallback_bundles:
                    if bundle.returned_pose_original is not None and bundle.error is None:
                        fallback = InitializationWorkingFallback(bundle.mask_id, bundle.returned_pose_original)
                        break
            driver = InitializationTrackingSession(session_id, checker, est.diameter, cad_identity,
                tracking_config=tracking_config, refinement_config=refinement_config,
                seed_factory=seed_factory, anchor_model_points=model_points)
            initial = driver.start(admission, frame_id=0, rgb=rgb, depth=depth, K=K,
                                   masks=mask_mapping, working_fallback=fallback)
            tracker = driver.tracker
            selected = driver.selected_mask_id
            pose, centered = driver.selected_pose_original, driver.selected_pose_centered
            report["initialization_admission"] = {
                "accepted": admission.accepted, "reason": admission.reason, "digest": admission.content_digest,
                "selected_index": driver.selected_index, "selected_mask_id": selected,
                "selected_backend_index": driver.selected_backend_index, "source": driver.selected_source,
                "identity_verified": False, "hypothesis_scope": admission.hypothesis_scope,
                "legacy_mask_selection_is_diagnostic_only": True,
                "fallback_rule": "original top-1 of legacy-selected mask if available, otherwise first successful mask; tentative only"}
            registration_evidence = initial
        elif selected is None:
            raise ValueError("all initialization mask candidates lack valid model/depth support; raw registrations and evidence retained")
        else:
            selected_dir = output / "initialization_candidates" / f"candidate_{selected:02d}"
            pose = np.load(selected_dir / "registration_pose.npy", allow_pickle=False).reshape(4, 4)
            centered = np.load(selected_dir / "centered_pose.npy", allow_pickle=False).reshape(4, 4)
            registration_evidence = initialization["candidates"][selected]["evidence"]
        mask = np.asarray(masks[selected], dtype=bool)
        est._pose_last_from_numpy(centered)
        est.ob_mask = mask
        np.savetxt(output / "registration_pose.txt", pose)
        write_json(output / "registration_evidence.json", registration_evidence)
        cv2.imwrite(str(output / "initial_mask.png"), mask.astype(np.uint8) * 255)
        if all_hypotheses:
            pass  # The session already installed trusted or tentative state atomically.
        elif args.mode == "baseline":
            initial = {"status": "baseline_registered", "evidence": checker.evaluate(centered, rgb, depth, K, mask=mask)}
        else:
            initial = tracker.reset(centered, rgb, depth, K, mask)
        report["configuration"] = tracker.configuration()
        torch.cuda.synchronize()
        report["initialization_seconds"] = time.perf_counter() - initialization_start
        report["initialization_timing_boundary"] = "mesh/frame loading + SAM2 candidate inference when used + FoundationPose setup + every candidate registration and observation check + selected memory initialization + artifact serialization"
        if measured:
            report["initialization_timing_boundary"] = "mesh/frame loading + first-frame mask/click + FP setup + complete candidate registration/assessment + current-input GFTT and tentative measured session startup + evidence serialization; no trusted-memory initialization"
        details.append(initial)
        poses.append(np.asarray(pose).reshape(4, 4))
        if continuity:
            publish_measured_output(driver, initial_result, pose, frame_id=0,
                                    slots=output_slots, arrays=output_arrays)
            report["planned_frame_slots"][0]["status"] = "completed"
            report["initialization_seconds"] = time.perf_counter() - initialization_start
            report["initialization_timing_boundary"] += " + continuity decision and publication evidence"
        for index, name in enumerate(names[1:], start=1):
            if measured:
                report["planned_frame_slots"][index]["status"] = "started"
            torch.cuda.synchronize()
            start = time.perf_counter()
            rgb, depth = read_frame(args.sequence, name, args.depth_scale)
            if args.mode == "baseline":
                pose = est.track_one(rgb, depth, K, iteration=1)
                centered = est.pose_last.detach().cpu().numpy().reshape(4, 4)
                evidence = checker.evaluate(centered, rgb, depth, K)
                result = {"pose": centered, "raw_iterations": centered[None],
                          "diagnostics": {"status": "baseline", "source": "previous_neural_pose",
                                          "neural_iterations": 1, "evidence": evidence}}
            else:
                def refine_one(seed):
                    est._pose_last_from_numpy(seed)
                    est.track_one(rgb, depth, K, iteration=1)
                    return est.pose_last.detach().cpu().numpy().reshape(4, 4)
                try:
                    if measured:
                        callbacks = []
                        callback_frames.append((index, callbacks))
                        callback = measured_refiner(est, rgb, depth, K, centered_from_original=centering,
                            frame_id=index, records=callbacks)
                        error = None
                        try:
                            driver.record_evidence("input_name", dict(frame_id=index, source_frame_id=index, name=name))
                            result = driver.step(rgb, depth, K, callback, frame_id=index, source_frame_id=index, timestamp=None)
                            validate_measured_result(driver, result, stage="step")
                        except BaseException as exc:
                            error = exc
                        for record in callbacks:
                            try:
                                driver.record_evidence("fp_callback", record)
                            except Exception as exc:
                                report.setdefault("callback_evidence_errors", []).append(f"{type(exc).__name__}: {exc}")
                                if error is None:
                                    error = exc
                        if error is not None:
                            raise error
                    else:
                        result = driver.step(rgb, depth, K, refine_one)
                except Exception as exc:
                    from tentative_tracking import TentativeStepError
                    if isinstance(exc, TentativeStepError):
                        # A failed frame is not appended as a valid output. Preserve
                        # every actual numeric return, including malformed shapes.
                        np.savez_compressed(output / f"failed_frame_{index:06d}_raw.npz",
                            **{f"return_{i:02d}": value for i, value in enumerate(exc.raw_returns)})
                        write_json(output / f"failed_frame_{index:06d}.json", exc.diagnostics)
                    raise
                if measured:
                    if not callbacks or not callbacks[-1].get("returned"):
                        raise ValueError("actual final FP original return missing")
                    pose = callbacks[-1]["actual_original_return"]
                    if pose.shape != (4, 4):
                        raise ValueError("actual FP original return must be 4x4; raw shape retained")
                    derived = np.asarray(result["pose"]) @ centering
                    if not np.allclose(pose, derived, rtol=0, atol=2e-7, equal_nan=True):
                        raise ValueError("actual FP original return differs from centered return and registration centering")
                else:
                    pose = result["pose"] @ est.get_tf_to_centered_mesh().detach().cpu().numpy()
            if continuity:
                # Preserve the real candidate even if publication/evidence
                # fails. It is not appended to the available consumer output.
                poses.append(np.asarray(pose).reshape(4, 4))
                raw.append(result["raw_iterations"])
                details.append(result["diagnostics"])
                publish_measured_output(driver, result, pose, frame_id=index,
                                        slots=output_slots, arrays=output_arrays)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
            if not continuity:
                poses.append(np.asarray(pose).reshape(4, 4))
                raw.append(result["raw_iterations"])
                details.append(result["diagnostics"])
            if not np.isfinite(poses[-1]).all():
                raise RuntimeError("nonfinite neural output retained; run failed, no frame dropping")
            if measured:
                report["planned_frame_slots"][index]["status"] = "completed"
            if index % 25 == 0:
                print(json.dumps({"frame": index, "total": len(names), "status": details[-1]["status"]}), flush=True)
        report.update(status="completed", frames=len(poses), tracking_seconds=sum(timings),
                      tracking_fps=len(timings) / sum(timings) if timings else None,
                      latency_p50_ms=float(np.percentile(timings, 50) * 1e3) if timings else None,
                      latency_p95_ms=float(np.percentile(timings, 95) * 1e3) if timings else None,
                      neural_iterations=dict(Counter(d["neural_iterations"] for d in details[1:])),
                      status_counts=dict(Counter(d["status"] for d in details[1:])),
                      seed_sources=dict(Counter(d["source"] for d in details[1:])))
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", completed_frames=len(poses))
        raise
    finally:
        if measured:
            if report["status"] == "failed":
                for slot in report["planned_frame_slots"]:
                    if slot["status"] == "started":
                        slot["status"] = "failed"
                if continuity:
                    for slot, runtime_slot in zip(output_slots, report["planned_frame_slots"]):
                        if slot["status"] == "not_executed" and runtime_slot["status"] == "failed":
                            slot.update(status="failed_before_publication", pose_available=False,
                                        error=report.get("error", "runtime failed before publication"))
            report["completed_frames"] = sum(s["status"] == "completed" for s in report["planned_frame_slots"])
            report["saved_pose_frames"] = len(poses)
            if measured_owner is not None:
                try:
                    measured_owner.close(reason=report["status"])
                except Exception as exc:
                    report["session_close_error"] = f"{type(exc).__name__}: {exc}"
                    report["status"] = "failed"
                if continuity:
                    try:
                        measured_owner.output_policy.close()
                    except Exception as exc:
                        report["output_policy_close_error"] = f"{type(exc).__name__}: {exc}"
                        report["status"] = "failed"
            if point_backend is not None:
                report["point_backend_provenance"] = point_backend.provenance
            save_measured_callback_prefix(output, callback_frames)
        if initialization_start is not None:
            report.setdefault("initialization_seconds", time.perf_counter() - initialization_start)
        report["source_unchanged"] = all(sha256(ROOT / name) == digest for name, digest in source_hashes.items())
        if not report["source_unchanged"]:
            report["status"] = "invalid_source_changed"
        np.save(output / "poses.npy", np.asarray(poses), allow_pickle=False)
        np.save(output / "latency_seconds.npy", np.asarray(timings), allow_pickle=False)
        np.savez_compressed(output / "raw_neural_iterations.npz", **{f"frame_{i+1:06d}": x for i, x in enumerate(raw)})
        write_json(output / "diagnostics.json", details)
        if continuity:
            save_output_policy(output, output_slots, output_arrays)
            report["output_policy_summary"] = dict(name="continuity", candidate_role="raw evidence; unavailable candidates are not consumer poses",
                decided_frames=sum(s["status"] == "decided" for s in output_slots),
                pose_available_frames=sum(s["pose_available"] is True for s in output_slots),
                publication_file="published_poses.npz", decision_file="output_policy.json", identity_verified=False)
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "report.json", report)
    print(json.dumps({k: report[k] for k in ("status", "frames", "tracking_fps", "status_counts", "seed_sources")}), flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
