"""Observable-evidence budget for current-frame neural pose refinement.

This schedules an existing trained refiner; it does not train an object model.
Every input frame receives at least one neural call. All thresholds are explicit
research defaults, not calibrated confidence or a guaranteed latency budget.
"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class AdaptiveRefinementConfig:
    min_iterations: int = 1
    max_iterations: int = 3
    stable_support: float = 0.8
    stable_contradiction: float = 0.05
    min_improvement: float = 0.01
    reference_support_slack: float = 0.05
    reference_contradiction_slack: float = 0.02

    def __post_init__(self):
        if (type(self.min_iterations) is not int or type(self.max_iterations) is not int
                or not 1 <= self.min_iterations <= self.max_iterations <= 10):
            raise ValueError("require 1 <= min_iterations <= max_iterations <= 10")
        for name in ("stable_support", "stable_contradiction", "min_improvement",
                     "reference_support_slack", "reference_contradiction_slack"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite in [0, 1]")


class AdaptiveRefinementPolicy:
    def __init__(self, config=None):
        self.config = config or AdaptiveRefinementConfig()
        self._reference = None

    def configuration(self):
        return {**asdict(self.config), "object_reference": self._reference}

    def calibrate(self, evidence):
        """Set an object-specific compute target from supported registration.

        This changes only the neural iteration budget, never the validity or
        identity gates. It uses the first observation, not GT or future frames.
        The fixed reference cannot drift toward deteriorating observations.
        """
        self._reference = None
        if evidence.get("observation_valid", False):
            values = [evidence.get(k) for k in ("depth_support_fraction", "contradiction_fraction")]
            if all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in values):
                self._reference = dict(depth_support_fraction=float(values[0]),
                                       contradiction_fraction=float(values[1]),
                                       source="supported_registration_only")

    @staticmethod
    def quality(evidence):
        support = evidence.get("depth_support_fraction", 0.0)
        contradiction = evidence.get("contradiction_fraction", 1.0)
        if not all(isinstance(v, (int, float)) and math.isfinite(v)
                   for v in (support, contradiction)):
            return -1.0
        return float(support - contradiction)

    def decide(self, history):
        """Return (run_another_iteration, reason) for this frame's evidence."""
        n, cfg = len(history), self.config
        if n < cfg.min_iterations:
            return True, "minimum_current_frame_neural_budget"
        if n >= cfg.max_iterations:
            return False, "maximum_budget"
        current = history[-1]
        if current.get("status") in ("invalid_input", "invalid_pose"):
            return False, "invalid_output"
        if not current.get("observation_valid", False):
            # Extra local iterations cannot establish identity or recover unseen
            # geometry. Leave global re-detection to the independent seed stage.
            return False, "insufficient_observable_evidence"
        if n >= 2 and self.quality(current) - self.quality(history[-2]) < cfg.min_improvement:
            return False, "observed_plateau_or_regression"
        target_support, target_contradiction = cfg.stable_support, cfg.stable_contradiction
        if self._reference is not None:
            target_support = min(target_support, max(.2, self._reference["depth_support_fraction"] - cfg.reference_support_slack))
            target_contradiction = max(target_contradiction, self._reference["contradiction_fraction"] + cfg.reference_contradiction_slack)
        stable = (current.get("depth_support_fraction", 0) >= target_support
                  and current.get("contradiction_fraction", 1) <= target_contradiction)
        # A flat appearance patch is weak evidence for orientation; allow one
        # further geometric neural correction, then still require improvement.
        uncertain = (current.get("texture_level") in ("low", "textureless", "untextured", "directional", "unknown")
                     or current.get("rotation_ambiguous", False))
        if stable and (not uncertain or n >= 2):
            return False, "stable_observation"
        return True, "weak_texture_or_residual_geometry"
