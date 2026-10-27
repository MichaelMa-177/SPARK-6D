"""Online output policy for a measured-motion session.

The tracker keeps its candidate and recurrence. This adapter publishes that
actual original-coordinate return or None, using only the current synchronous
session events. It does not read evidence files, load models, or certify identity.
An exception terminates this output epoch; a new click requires a new adapter.
"""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import numpy as np

from identity_continuity import ContinuityGate, FLAGS, rigid, signature


def numeric_snapshot(value):
    """Preserve typed event arrays without JSON conversion or shared scratch."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {f.name: numeric_snapshot(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: numeric_snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [numeric_snapshot(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("unsupported session evidence type: " + type(value).__name__)


class OutputPolicyAdapter:
    """Use ``observe`` as the session sink, then publish its completed result.

    Call ``publish(result, actual_original_pose)`` after the entrypoint checks
    the actual backend return. Its returned ``published_pose`` is the consumer
    output; ``candidate_pose`` remains evidence and must not replace abstention.
    The optional evidence sink receives the original envelope before a captured
    packet becomes available. Publication also succeeds only after its evidence
    write succeeds. Instances cannot be reset or reused after an exception.
    """
    _CAPTURED = frozenset(("registration", "initial_input", "frame_input",
                          "original_motion", "preloss_event"))

    def __init__(self, session_id, evidence_sink=None):
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("explicit nonempty session ID required")
        if evidence_sink is not None and not callable(evidence_sink):
            raise TypeError("callable evidence sink required")
        self.session_id = session_id
        self._sink = evidence_sink
        self._gate = None
        self._bank_signature = None
        self._pending = {}
        self._clock = None
        self._last_frame = -1
        self._closed = False
        self._last_publication = None
        self.failure = None

    @property
    def closed(self):
        return self._closed

    @property
    def last_publication(self):
        return numeric_snapshot(self._last_publication)

    def _fail(self, exc):
        self._closed = True
        self._last_publication = None
        self.failure = dict(error=type(exc).__name__ + ": " + str(exc),
                            frame_id=None if self._clock is None else self._clock[0],
                            source_frame_id=None if self._clock is None else self._clock[1],
                            pose_available=False, published_pose=None)

    def observe(self, kind, envelope):
        """Forward evidence and retain only this frame's typed gate inputs."""
        try:
            forwarded = None if self._sink is None else self._sink(kind, envelope)
            if kind not in self._CAPTURED:
                return forwarded
            if self._closed:
                raise RuntimeError("output epoch is closed; a new click is required")
            if not isinstance(envelope, Mapping) or envelope.get("session_id") != self.session_id:
                raise ValueError("session evidence ID differs from output epoch")
            payload = numeric_snapshot(envelope["payload"])
            if kind == "registration":
                if self._gate is not None or self._last_frame != -1 or self._pending:
                    raise ValueError("registration must be captured once before the first input")
                self._gate = ContinuityGate(
                    self.session_id, selected_backend_index=payload["selected_backend_index"],
                    centered_from_original=payload["centered_from_original"])
                self._bank_signature = signature(payload["initial_hypotheses"])
                return forwarded
            clock = (envelope["frame_id"], envelope["source_frame_id"])
            if kind in ("initial_input", "frame_input"):
                if self._gate is None or self._pending or clock[0] != self._last_frame + 1:
                    raise ValueError("exactly one current input after each published decision required")
                if (kind == "initial_input") != (clock[0] == 0):
                    raise ValueError("initial input kind and frame differ")
                self._clock = clock
                self._pending["input"] = payload
            else:
                if "input" not in self._pending or clock != self._clock:
                    raise ValueError("motion event does not belong to the current input")
                if kind in self._pending:
                    raise ValueError("duplicate current-frame motion event")
                self._pending[kind] = payload
            return forwarded
        except BaseException as exc:
            self._fail(exc)
            raise

    def publish(self, result, actual_original_pose):
        """Decide after refinement; never install a seed or change a candidate."""
        try:
            if self._closed:
                raise RuntimeError("output epoch is closed; a new click is required")
            if self._gate is None or "input" not in self._pending:
                raise ValueError("current registration/input evidence required")
            d = result["diagnostics"]
            if (d.get("session_id") != self.session_id
                    or (d.get("frame_id"), d.get("source_frame_id")) != self._clock):
                raise ValueError("completed result differs from the current input clock")
            if any(d.get(k) is not False for k in FLAGS):
                raise ValueError("all tentative trust flags must remain false")
            actual = np.asarray(actual_original_pose)
            rigid(actual)
            inp = self._pending["input"]
            motion = self._pending.get("original_motion", {}).get("result")
            event = self._pending.get("preloss_event", {}).get("event")
            if motion is not None and signature(motion["raw_initial_hypotheses"]) != self._bank_signature:
                raise ValueError("point motion bank differs from actual registration")
            decision = self._gate.update(
                frame=self._clock[0], source=self._clock[1], rgb=inp["rgb"],
                depth=inp["depth_m"], K=inp["K"], motion=motion, event=event,
                candidate_centered=result["pose"], candidate_original=actual,
                observation_valid=d["evidence"]["observation_valid"])
            # Preserve exact backend dtype and values on both public/evidence paths.
            if (decision["candidate_pose"].dtype != actual.dtype
                    or decision["candidate_pose"].tobytes() != actual.tobytes()):
                raise ValueError("output policy changed the actual candidate")
            if self._sink is not None:
                self._sink("output_publication", dict(
                    session_id=self.session_id, frame_id=self._clock[0],
                    source_frame_id=self._clock[1], stage="output_publication",
                    payload=numeric_snapshot(decision)))
            self._last_publication = numeric_snapshot(decision)
            self._last_frame = self._clock[0]
            self._pending = {}
            return decision
        except BaseException as exc:
            self._fail(exc)
            raise

    def close(self):
        self._closed = True
        self._pending = {}
        self._last_publication = None
