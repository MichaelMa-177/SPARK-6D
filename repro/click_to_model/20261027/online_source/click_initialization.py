"""Observation-gated mask selection for a single RGB-D initialization frame.

SAM scores are a tie-break only. All masks, raw registrations and observation
diagnostics are retained before any tracker identity memory is initialized.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import cv2
import numpy as np


SELECTION_RULE = (
    "Only observation_valid candidates are eligible. Lexicographically maximize "
    "depth_support_fraction, minimize contradiction_fraction, minimize "
    "depth_residual_median_m, then maximize SAM score; exact ties use the "
    "earliest candidate index. No supplied comparison mask or GT is consulted."
)

SAM_WORKING_SELECTION_RULE = (
    "For tentative working state only, rank nonempty masks by finite SAM score "
    "descending, with original mask index breaking ties. Choose the first with "
    "successful actual original and pose_last returns bound by the registered C; "
    "record rejected higher-ranked candidates. A single supplied mask with no "
    "SAM score is eligible explicitly. No geometry or cross-mask FP score ranking, "
    "no pose repair, and no identity or trusted-memory admission."
)


def select_sam_working_candidate(records, returns, *, centered_from_original,
                                 supplied_mask=False):
    """Choose a working mask without modifying any raw outputs or admission.

    ``returns[index]`` is the actual registration capture: error, raw_pose and
    centered_pose. Observation-checker rejection is deliberately not a backend
    failure. Missing/invalid returns skip that mask in the original SAM order.
    The caller persists this decision even when selected_index is None.
    """
    from pose_validity import _pose
    if type(supplied_mask) is not bool:
        raise ValueError("supplied_mask must be explicit bool")
    indices = [record["index"] for record in records]
    if (not records or any(type(i) is not int or i < 0 for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("Unique original initialization candidate indices required")
    C = np.asarray(centered_from_original)
    if C.shape != (4, 4) or C.dtype.kind not in "fi" or not np.isfinite(C).all():
        raise ValueError("Actual registered C must be a finite real 4x4 matrix")
    _pose(C)  # Existing raw rigid-pose guard; never repair the matrix.
    single_supplied = (supplied_mask and len(records) == 1
                       and records[0].get("sam_score") is None)
    decisions, ranked = [], []
    for record in records:
        index, score = record["index"], record.get("sam_score")
        item = dict(index=index, sam_score=None, eligible=False, reason=None,
                    initialization_error=record.get("error"))
        pixels = record.get("mask_pixels")
        if type(pixels) is not int or pixels <= 0:
            item["reason"] = "empty_or_invalid_mask"
        elif single_supplied:
            ranked.append((0., index, item))
        else:
            try:
                value = float(score) if score is not None else float("nan")
            except (TypeError, ValueError, OverflowError):
                value = float("nan")
            if not math.isfinite(value):
                item["reason"] = "missing_or_nonfinite_SAM_score"
            else:
                item["sam_score"] = value
                ranked.append((-value, index, item))
        decisions.append(item)
    ranked.sort(key=lambda value: (value[0], value[1]))
    selected = None
    for _, index, item in ranked:
        captured = returns.get(index)
        if not isinstance(captured, dict):
            item["reason"] = "registration_capture_missing"
            continue
        if captured.get("error"):
            item.update(reason="registration_failed", backend_error=str(captured["error"]))
            continue
        original, centered = captured.get("raw_pose"), captured.get("centered_pose")
        if original is None or centered is None:
            item["reason"] = "actual_original_or_pose_last_missing"
            continue
        original, centered = np.asarray(original), np.asarray(centered)
        if any(value.shape != (4, 4) or value.dtype.kind not in "fi"
               or not np.isfinite(value).all() for value in (original, centered)):
            item["reason"] = "actual_returns_not_finite_real_4x4"
            continue
        try:
            _pose(original)
            _pose(centered, positive_depth=True)
        except (TypeError, ValueError):
            item["reason"] = "actual_returns_fail_existing_rigid_working_pose_guard"
            continue
        if not np.allclose(original, centered @ C,
                           rtol=0, atol=2e-7):
            item["reason"] = "actual_original_pose_last_C_mismatch"
            continue
        item.update(eligible=True, reason="actual_returns_valid")
        if selected is None:
            selected = index
    order = [index for _, index, _ in ranked]
    skipped = [item for _, index, item in ranked
               if selected is None or order.index(index) < order.index(selected)]
    return dict(schema="tentative_working_mask_selection/1", strategy="sam-score",
                selection_rule=SAM_WORKING_SELECTION_RULE, selected_index=selected,
                score_order=order, candidates=decisions,
                supplied_mask_without_SAM_score=single_supplied,
                fallback_used=selected is not None and bool(skipped),
                skipped_higher_ranked=skipped,
                failure_reason=None if selected is not None else "no_valid_actual_return_in_SAM_order",
                geometry_fallback_used=False, admission_changed=False,
                identity_verified=False, trusted_memory_initialized=False,
                trusted_memory_updated=False)


def _geometry_key(record):
    evidence = record.get("evidence", {})
    if record.get("error") or not isinstance(evidence, dict) or evidence.get("observation_valid") is not True:
        return None
    try:
        support = float(evidence["depth_support_fraction"])
        contradiction = float(evidence["contradiction_fraction"])
        residual = float(evidence["depth_residual_median_m"])
        score = record.get("sam_score")
        score = float(score) if score is not None else -math.inf
    except (KeyError, ValueError, TypeError):
        return None
    if (not all(math.isfinite(v) for v in (support, contradiction, residual))
            or not 0 <= support <= 1 or not 0 <= contradiction <= 1 or residual < 0
            or (record.get("sam_score") is not None and not math.isfinite(score))):
        return None
    return support, -contradiction, -residual, score, -int(record["index"])


def select_candidate(records):
    """Return an eligible candidate index or None, never the best invalid one."""
    eligible = [(key, int(record["index"])) for record in records
                if (key := _geometry_key(record)) is not None]
    return max(eligible)[1] if eligible else None


def _write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def initialize_candidates(masks, scores, register, evaluate, output_dir, *, indexed_register=False):
    """Run all current-frame proposals and persist failures as well as successes.

    ``register(mask)`` returns (raw original-mesh pose, centered pose) after
    device synchronization. ``evaluate(centered_pose, mask)`` is the unchanged
    observation checker. Neither callback receives GT or any later frame.
    With ``indexed_register=True``, call ``register(index, mask)`` so complete
    backend hypotheses remain bound to their original proposal, including when
    preceding proposals are empty or registration fails.
    """
    if len(masks) == 0 or len(masks) != len(scores):
        raise ValueError("Expected one score per initialization mask candidate")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    records = []
    for index, (mask, score) in enumerate(zip(masks, scores)):
        candidate_dir = directory / f"candidate_{index:02d}"
        candidate_dir.mkdir()
        record = {"index": index, "sam_score": None, "mask_pixels": 0,
                  "directory": candidate_dir.name, "evidence": None}
        started = None
        try:
            if score is not None:
                if not math.isfinite(float(score)):
                    raise ValueError("Nonfinite SAM candidate score")
                record["sam_score"] = float(score)
            mask = np.asarray(mask)
            if (mask.ndim != 2 or mask.dtype.kind not in "buif"
                    or not np.isfinite(mask).all() or not np.isin(mask, [0, 1, 255]).all()):
                raise ValueError("Initialization candidate must be a finite binary HxW mask")
            mask = mask != 0
            record["mask_pixels"] = int(mask.sum())
            if not cv2.imwrite(str(candidate_dir / "mask.png"), mask.astype(np.uint8) * 255):
                raise OSError("Failed to persist initialization candidate mask")
            if not mask.any():
                raise ValueError("Empty initialization mask candidate")
            started = time.perf_counter()
            raw_pose, centered_pose = (register(index, mask.copy()) if indexed_register
                                       else register(mask.copy()))
            record["registration_seconds"] = time.perf_counter() - started
            raw_pose = np.asarray(raw_pose)
            np.save(candidate_dir / "registration_pose.npy", raw_pose, allow_pickle=False)
            if raw_pose.shape == (4, 4):
                np.savetxt(candidate_dir / "registration_pose.txt", raw_pose)
            record["raw_registration_pose_file"] = "registration_pose.npy"
            # Preserve the returned raw pose even when registration did not
            # initialize pose_last, or observation evaluation itself fails.
            if centered_pose is None:
                raise ValueError("Registration did not produce a centered pose")
            centered_pose = np.asarray(centered_pose)
            np.save(candidate_dir / "centered_pose.npy", centered_pose, allow_pickle=False)
            record["centered_pose_file"] = "centered_pose.npy"
            record["evidence"] = evaluate(centered_pose.copy(), mask)
            if not isinstance(record["evidence"], dict):
                raise TypeError("Observation evaluator must return diagnostics")
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            if started is not None:
                record.setdefault("registration_seconds", time.perf_counter() - started)
        record["eligible"] = _geometry_key(record) is not None
        _write_json(candidate_dir / "record.json", record)
        records.append(record)
    selected = select_candidate(records)
    summary = {"schema": "click_initialization/1", "selection_rule": SELECTION_RULE,
               "selected_index": selected, "all_candidates_invalid": selected is None,
               "candidates": records, "gt_or_comparison_mask_used": False,
               "registration_seconds_total": sum(r.get("registration_seconds", 0) for r in records)}
    _write_json(directory / "selection.json", summary)
    return summary
