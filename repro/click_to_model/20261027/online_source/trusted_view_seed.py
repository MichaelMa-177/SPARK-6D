"""Bounded, explicitly admitted CAD-backed views for RGB-D seed recovery.

This optional adapter preserves the registration ORB dictionary and the parent
tracker's transactional LK/commit behavior. Online views require an immediately
preceding successful external pose-verification commit; no lost-frame image can
become a view. Recovery returns a proposal only when all supported rigid
hypotheses agree and no sufficient unexplained correspondence set remains.
It is correspondence evidence, never instance identity proof.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from rgbd_seed import RGBDSeedTracker, _geometry, _pose, robust_rigid_transform


@dataclass(frozen=True)
class TrustedViewConfig:
    max_online_views: int = 7
    keyframe_commit_interval: int = 10
    merge_translation_m: float = .015
    merge_rotation_rad: float = .15
    max_hypotheses_per_view: int = 3

    def __post_init__(self):
        for name in ("max_online_views", "keyframe_commit_interval", "max_hypotheses_per_view"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
        for name in ("merge_translation_m", "merge_rotation_rad"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite positive")


@dataclass(frozen=True)
class _View:
    commit_index: int
    descriptors: np.ndarray
    anchors: np.ndarray


class TrustedViewSeedTracker(RGBDSeedTracker):
    """Use accept_view only after independent neural/CAD/continuity admission.

    Optional checker is the same PoseValidityChecker used by the controller.
    Without it, every rigid hypothesis counts towards conservative ambiguity
    rejection. The controller must still validate/refine any returned proposal.
    """

    def __init__(self, config=None, backend=None, view_config=None, checker=None):
        super().__init__(config, backend)
        self.view_config = view_config or TrustedViewConfig()
        if not isinstance(self.view_config, TrustedViewConfig):
            raise TypeError("view_config must be TrustedViewConfig")
        self.checker = checker
        self._sampled_views = []
        self._latest_view = None
        self._commit_index = 0
        self._view_available = False
        self._committed_K = None
        self._pending_depth = self._pending_depth_proposal = self._committed_depth = None
        self._recovery_rgb = None
        self._recovery_details = None

    def _views(self):
        views = self._sampled_views.copy()
        if self._latest_view is not None and (not views or views[-1] is not self._latest_view):
            views.append(self._latest_view)
        return views[-self.view_config.max_online_views:]

    @property
    def trusted_view_inventory(self):
        return [dict(commit_index=view.commit_index, anchors=len(view.anchors)) for view in self._views()]

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        self._sampled_views, self._latest_view = [], None
        self._commit_index, self._view_available = 0, False
        self._committed_K = None
        self._pending_depth = self._pending_depth_proposal = self._committed_depth = None
        return result

    def propose(self, rgb, depth, K, mask=None):
        if self._pending is not None:
            raise RuntimeError("commit or reject the pending proposal first")
        # The exact depth observation used to fit/commit the original points
        # must also be used when admitting additional CAD-backed descriptors.
        # Copy before inference so caller-side mutation cannot change the
        # observation associated with this proposal after it was emitted.
        depth_snapshot = np.asarray(depth).copy()
        depth_snapshot.setflags(write=False)
        self._view_available = False
        self._committed_depth = None
        self._recovery_details = None
        self._recovery_rgb = np.asarray(rgb).copy()
        try:
            proposal = super().propose(rgb, depth_snapshot, K, mask)
            self._pending_depth, self._pending_depth_proposal = depth_snapshot, proposal
            if self._recovery_details is not None:
                proposal.diagnostics["trusted_view_recovery"] = self._recovery_details
            return proposal
        finally:
            self._recovery_rgb = None

    def commit(self, proposal, verified_pose=None):
        if self._pending is None or self._pending.proposal is not proposal:
            # Preserve the valid pending depth on a stale/mismatched commit;
            # the parent's transaction check raises the canonical exception.
            return super().commit(proposal, verified_pose)
        if self._pending_depth_proposal is not proposal or self._pending_depth is None:
            raise ValueError("proposal has no bound depth observation")
        pending_depth = self._pending_depth
        self._view_available = False
        pending_K = None if self._pending is None else self._pending.K.copy()
        result = super().commit(proposal, verified_pose)
        self._pending_depth = self._pending_depth_proposal = None
        self._committed_depth = None
        if result:
            self._commit_index += 1
            self._view_available = True
            self._committed_K = pending_K
            self._committed_depth = pending_depth
        return result

    def reject(self, proposal):
        result = super().reject(proposal)
        if result:
            self._pending_depth = self._pending_depth_proposal = None
        return result

    def promote_confirmed(self, snapshot, *, frame_id, rgb, depth, K):
        """Preserve all trusted views and bind a new one-shot view opportunity."""
        staged = self._prepare_confirmed_promotion(snapshot, frame_id=frame_id, rgb=rgb, depth=depth, K=K)
        if staged is None:
            return False
        # All copies/validation precede the single local-memory replacement.
        try:
            committed_depth, committed_K = np.asarray(depth).copy(), np.asarray(K).copy()
            committed_depth.setflags(write=False)
            committed_K.setflags(write=False)
            commit_index = self._commit_index + 1
        except Exception:
            return False
        self._install_confirmed_promotion(staged)
        self._pending_depth = self._pending_depth_proposal = None
        self._committed_depth, self._committed_K = committed_depth, committed_K
        self._commit_index, self._view_available = commit_index, True
        self._recovery_rgb = self._recovery_details = None
        return True

    def accept_view(self, rgb, depth, K, mask, verified_pose, model_points):
        """CAD-snap one just-committed view; never overwrite registration memory.

        gray/depth/K/pose must match the current successful commit. Admission consumes
        that commit's one view opportunity, even if this image lacks features.
        No implicit call is made from propose, reject, or an unsuccessful commit.
        """
        if not self.initialized or self._pending is not None or not self._view_available:
            raise RuntimeError("accept_view requires an unconsumed successful commit")
        gray, depth, K, mask = self._frame(rgb, depth, K, mask)
        pose = _pose(verified_pose)
        if (not np.array_equal(gray, self._gray) or not np.array_equal(K, self._committed_K)
                or not np.array_equal(pose, self._pose)
                or self._committed_depth is None
                or not np.array_equal(depth, self._committed_depth, equal_nan=True)):
            raise ValueError("view image, depth, intrinsics and pose must equal the committed observation")
        if mask is None:
            raise ValueError("accept_view requires an externally verified CAD-supported mask")
        model = np.asarray(model_points, dtype=float)
        if model.ndim != 2 or model.shape[1:] != (3,) or len(model) < 3 or not np.isfinite(model).all():
            raise ValueError("finite Nx3 CAD surface points required")
        self._view_available = False
        self._committed_depth = None
        result = dict(updated=False, reason="insufficient_cad_supported_descriptors",
            commit_index=self._commit_index, descriptor_count=0, identity_verified=False,
            registration_dictionary_updated=False)
        selection = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        pixels, descriptors = self._orb(gray, selection)
        if descriptors is None:
            return result
        camera, valid = self._lift(pixels, depth, K)
        pixels, camera, descriptors = pixels[valid], camera[valid], descriptors[valid]
        local = (camera - pose[:3, 3]) @ pose[:3, :3]
        distances, surface_ids = cKDTree(model).query(local)
        anchors = model[surface_ids]
        expected = anchors @ pose[:3, :3].T + pose[:3, 3]
        projection = expected @ K.T
        errors = np.linalg.norm(projection[:, :2] / np.maximum(projection[:, 2:], 1e-12) - pixels, axis=1)
        keep = ((distances <= self.config.residual_threshold_m)
                & (expected[:, 2] > self.config.min_depth_m)
                & (np.abs(expected[:, 2] - camera[:, 2]) <= self.config.residual_threshold_m)
                & (errors <= self.config.commit_reprojection_px))
        selected, used = [], set()
        ids = np.flatnonzero(keep)
        for index in ids[np.argsort(distances[ids], kind="stable")]:
            anchor = tuple(anchors[index])
            if anchor not in used:
                used.add(anchor)
                selected.append(int(index))
        selected = np.asarray(selected, dtype=int)
        result["descriptor_count"] = len(selected)
        if len(selected) < self.config.min_points or not _geometry(anchors[selected], self.config)[0]:
            return result
        saved_descriptors, saved_anchors = descriptors[selected].copy(), anchors[selected].copy()
        saved_descriptors.setflags(write=False)
        saved_anchors.setflags(write=False)
        view = _View(self._commit_index, saved_descriptors, saved_anchors)
        if (not self._sampled_views or self._commit_index - self._sampled_views[-1].commit_index
                >= self.view_config.keyframe_commit_interval):
            self._sampled_views.append(view)
        # Reserve one slot for the latest admitted view. Registration lives in
        # the parent tracker and is never removed by this bounded pruning.
        sampled_limit = max(0, self.view_config.max_online_views - 1)
        self._sampled_views = self._sampled_views[-sampled_limit:] if sampled_limit else []
        self._latest_view = view
        result.update(updated=True, reason="verified_cad_view", inventory=self.trusted_view_inventory)
        return result

    def _match(self, saved, current):
        if saved is None or current is None or len(saved) < 2 or len(current) < 2:
            return np.empty((0, 2), int)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        forward = matcher.knnMatch(saved, current, k=2)
        backward = matcher.knnMatch(current, saved, k=2)
        def good(pair):
            return (len(pair) == 2 and pair[0].distance <= self.config.orb_max_hamming
                    and pair[0].distance < self.config.orb_ratio * pair[1].distance)
        reverse = {pair[0].queryIdx: pair[0].trainIdx for pair in backward if good(pair)}
        pairs = [(pair[0].queryIdx, pair[0].trainIdx) for pair in forward
                 if good(pair) and reverse.get(pair[0].trainIdx) == pair[0].queryIdx]
        return np.asarray(pairs, int).reshape(-1, 2)

    def _view_hypotheses(self, view, pixels, descriptors, depth, K, mask):
        pairs = self._match(view.descriptors, descriptors)
        details = dict(commit_index=view.commit_index, reciprocal_matches=len(pairs),
                       depth_correspondences=0, mask_support=None, hypotheses=[],
                       hypothesis_budget_exhausted=False, unresolved_correspondences=0)
        if len(pairs) < self.config.min_points:
            return [], details
        targets = pixels[pairs[:, 1]]
        observed, valid = self._lift(targets, depth, K)
        if mask is not None:
            indices, inside = self._inside(targets, depth.shape)
            ids = np.flatnonzero(valid & inside)
            support = mask[indices[ids, 1], indices[ids, 0]]
            details["mask_support"] = float(support.mean()) if len(ids) else 0.
            if len(ids) and support.mean() < self.config.min_mask_support:
                return [], details
            valid[ids[~support]] = False
        anchors = view.anchors[pairs[:, 0]][valid]
        targets, observed = targets[valid], observed[valid]
        details["depth_correspondences"] = len(observed)
        hypotheses = []
        for hypothesis_id in range(self.view_config.max_hypotheses_per_view):
            pose, inliers, rigid = robust_rigid_transform(anchors, observed, self.config)
            if pose is None:
                details["terminal_reason"] = rigid["reason"]
                # Failure of the unchanged majority-consensus test does not
                # prove that no smaller observable object cluster remains.
                # Until those correspondences are explained, withhold a
                # unique-hypothesis claim even when another view has a strong
                # primary fit. This intentionally trades recovery coverage
                # for conservative ambiguity rejection; no primary threshold
                # or minimum point count is relaxed.
                if len(anchors) >= self.config.min_points:
                    details["unresolved_correspondences"] = len(anchors)
                break
            diagnostic = dict(rigid, seed_source="trusted_view_orb_rgbd", identity_verified=False,
                commit_index=view.commit_index, hypothesis_index=hypothesis_id,
                tracked_points=len(targets), depth_correspondences=len(observed), mask_support=details["mask_support"])
            supported = True
            if self.checker is not None:
                evidence = self.checker.evaluate(pose, self._recovery_rgb, depth, K, mask=mask)
                supported = evidence.get("observation_valid", False) is True
                diagnostic["candidate_observation_valid"] = supported
            details["hypotheses"].append(dict(pose=pose.tolist(), inlier_count=int(inliers.sum()),
                                                candidate_observation_valid=supported))
            if supported:
                hypotheses.append((pose, targets[inliers].copy(), anchors[inliers].copy(),
                                   observed[inliers].copy(), diagnostic))
            # Inspect every remaining rigid cluster; a second plausible object
            # must prevent selecting the strongest cluster as instance identity.
            anchors, targets, observed = anchors[~inliers], targets[~inliers], observed[~inliers]
            if len(anchors) < self.config.min_points:
                break
            if hypothesis_id + 1 == self.view_config.max_hypotheses_per_view:
                details["hypothesis_budget_exhausted"] = True
        return hypotheses, details

    def _equivalent_hypotheses(self, first, second):
        candidate = second
        if self.checker is not None:
            candidate = self.checker.equivalent_pose(second, first)
        translation = np.linalg.norm(candidate[:3, 3] - first[:3, 3])
        rotation = np.arccos(np.clip((np.trace(candidate[:3, :3] @ first[:3, :3].T) - 1) / 2, -1, 1))
        return (translation <= self.view_config.merge_translation_m
                and rotation <= self.view_config.merge_rotation_rad)

    def _recover(self, gray, depth, K, mask):
        pixels, descriptors = self._orb(gray, mask=None)
        registration = _View(0, self._orb_descriptors, self._orb_anchors)
        hypotheses, views = [], []
        for view in [registration] + self._views():
            candidates, detail = self._view_hypotheses(view, pixels, descriptors, depth, K, mask)
            hypotheses.extend(candidates)
            views.append(detail)
        ambiguous = (any(view["hypothesis_budget_exhausted"] for view in views)
            or any(not self._equivalent_hypotheses(hypotheses[i][0], hypotheses[j][0])
                   for i in range(len(hypotheses)) for j in range(i)))
        unresolved = any(view["unresolved_correspondences"] >= self.config.min_points for view in views)
        reason = ("ambiguous_recovery_hypotheses" if ambiguous else
                  "unresolved_recovery_correspondences" if unresolved else
                  "insufficient_recovery_matches" if not hypotheses else "unique_recovery_hypothesis")
        self._recovery_details = dict(reason=reason, supported_hypotheses=len(hypotheses),
            ambiguous=ambiguous, views=views, identity_verified=False,
            unresolved_correspondences=unresolved,
            registration_dictionary_preserved=True)
        if not hypotheses or ambiguous or unresolved:
            return (None, pixels[:0], self._anchors[:0], self._anchors[:0],
                    dict(seed_source="trusted_view_orb_rgbd", identity_verified=False,
                         reason=reason, inlier_count=0, inlier_fraction=0., residual_m=None,
                         conditioning=0., observable=False))
        # Deterministic preference among equivalent hypotheses; never average
        # rotations or anchor gauges belonging to different view representatives.
        selected = min(hypotheses, key=lambda item: (-item[4]["inlier_count"],
                       item[4]["residual_m"], -item[4]["commit_index"]))
        selected[4]["supported_hypotheses"] = len(hypotheses)
        return selected
