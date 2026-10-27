"""Tentative neural refinement with an explicit current-frame motion seed.

This owner never creates trusted correspondence memory and never owns a 2D
stream. The caller supplies D(t,0) @ the fixed initial centered working pose.
All alternatives and the proof of that multiplication belong to the driver.
"""
from copy import deepcopy
import json

import numpy as np

from adaptive_refinement import AdaptiveRefinementConfig
from tentative_tracking import TentativePoseTracker, TentativeStepError


class RelativeMotionRefinementTracker(TentativePoseTracker):
    """Execute the existing tentative step on isolated staged working state.

    A supplied seed changes the actual first neural input and the innovation
    reference, but never establishes pose/identity trust. Without a seed, the
    base controller's numeric outputs and diagnostics are unchanged.
    """
    def __init__(self, checker, diameter, *, admission_id, refinement_config=None,
                 max_translation_diameter=.35, max_rotation_rad=.8):
        if refinement_config is not None and type(refinement_config) is not AdaptiveRefinementConfig:
            raise TypeError('typed refinement configuration required')
        super().__init__(checker,diameter,admission_id=admission_id,
            refinement_config=refinement_config,max_translation_diameter=max_translation_diameter,
            max_rotation_rad=max_rotation_rad)
        self._source_frame_id = self._anchor_source_frame_id = None
        self._working_source_frame_id = None
        self._last_seed_record = None

    @property
    def frame_id(self): return self._frame

    @property
    def source_frame_id(self): return self._source_frame_id

    @property
    def working_source_frame_id(self): return self._working_source_frame_id

    @property
    def last_seed_record(self): return deepcopy(self._last_seed_record)

    def reset(self, pose, rgb, depth, K, mask, *, source_frame_id=0):
        if type(source_frame_id) is not int or source_frame_id < 0:
            raise ValueError('initial source frame must be a nonnegative builtin integer')
        result = super().reset(pose,rgb,depth,K,mask)
        self._source_frame_id = self._anchor_source_frame_id = source_frame_id
        self._working_source_frame_id = source_frame_id
        return result

    def step(self, rgb, depth, K, refine_one, mask=None, *, frame_seed=None,
             frame_id=None, source_frame_id=None, anchor_source_frame_id=None):
        if self._working_pose is None:
            raise RuntimeError('reset must precede step')
        expected = self._frame+1
        frame_id = expected if frame_id is None else frame_id
        source_frame_id = self._source_frame_id+1 if source_frame_id is None else source_frame_id
        if type(frame_id) is not int or frame_id != expected:
            raise ValueError('stream frames must advance exactly once and consecutively')
        if type(source_frame_id) is not int or source_frame_id <= self._source_frame_id:
            raise ValueError('original source frame IDs must strictly increase')
        if anchor_source_frame_id is not None and (type(anchor_source_frame_id) is not int
                or anchor_source_frame_id != self._anchor_source_frame_id):
            raise ValueError('motion anchor must be the original source observation')
        self._inputs(rgb,depth,K,mask)
        raw_seed = None if frame_seed is None else np.asarray(frame_seed).copy()
        if raw_seed is not None and (raw_seed.shape != (4,4) or raw_seed.dtype.kind not in 'fi'
                or raw_seed.dtype.itemsize > 8 or not self._valid_pose(raw_seed)):
            raise ValueError('finite positive-depth raw centered SE3 frame seed required; no repair')
        actual_seed = self._working_pose.copy() if raw_seed is None else raw_seed.astype(float,copy=True)
        source = 'tentative_last_neural_working' if raw_seed is None else 'tentative_fixed_camera_motion'
        seed_record = dict(frame_id=frame_id,source_frame_id=source_frame_id,
            anchor_source_frame_id=self._anchor_source_frame_id,
            working_seed_source_frame_id=self._working_source_frame_id,seed_source=source,
            raw_supplied_seed=raw_seed,actual_first_seed=actual_seed.copy(),frame_consumed=False,
            working_seed_updated=False,identity_verified=False,trusted_memory_updated=False,
            point_stream_owned=False,point_stream_replayed=False)
        # Only this fresh local controller receives the external seed. Its
        # calibrated adaptive policy is copied; reset is not called again.
        staged = TentativePoseTracker(self.checker,self.diameter,admission_id=self.admission_id,
            refinement_config=self.policy.config,max_translation_diameter=self.max_translation_diameter,
            max_rotation_rad=self.max_rotation_rad)
        staged.policy = deepcopy(self.policy)
        staged._working_pose = actual_seed.copy()
        staged._frame = self._frame
        result = None
        try:
            result = staged.step(rgb,depth,K,refine_one,mask)
            updated = result['diagnostics']['working_seed_updated'] is True
            metadata = {key:seed_record[key] for key in ('frame_id','source_frame_id',
                'anchor_source_frame_id','working_seed_source_frame_id','seed_source')}
            if raw_seed is not None:
                result['diagnostics'] = dict(result['diagnostics'],source=source,
                    iteration_sources=[source]+['tentative_current_frame_neural']*(len(result['raw_iterations'])-1),
                    relative_motion_seed=metadata)
            json.dumps(result['diagnostics'],allow_nan=False)
            # A rejected neural update does NOT publish the external point
            # seed as a new working pose. Preserve the old neural fallback.
            next_working = staged.working_pose if updated else self._working_pose.copy()
            next_working_source = source_frame_id if updated else self._working_source_frame_id
            seed_record.update(frame_consumed=True,working_seed_updated=updated,
                working_source_frame_id_after=next_working_source,outcome='returned')
            captured_record = deepcopy(seed_record)
            captured_raw = staged.last_raw_returns
        except Exception as error:
            consumed = staged._frame == frame_id
            seed_record.update(frame_consumed=consumed,working_seed_updated=False,
                working_source_frame_id_after=self._working_source_frame_id,
                outcome='failed',error=type(error).__name__+': '+str(error))
            # Frame/return evidence advances even though working state does not.
            self._frame = staged._frame
            if consumed: self._source_frame_id = source_frame_id
            self._last_raw = staged.last_raw_returns
            self._last_seed_record = deepcopy(seed_record)
            if isinstance(error,TentativeStepError):
                if raw_seed is not None:
                    error.diagnostics.update(source=source,relative_motion_seed={
                        key:seed_record[key] for key in ('frame_id','source_frame_id',
                            'anchor_source_frame_id','working_seed_source_frame_id','seed_source')})
                raise
            diagnostics=dict(self._flags(),status='tentative_step_failed',
                neural_iterations=len(self._last_raw),working_seed_updated=False)
            raise TentativeStepError(error,self._last_raw,diagnostics) from error
        # All fallible output/record construction has completed before this
        # publication. Only a successful neural working update can change pose.
        self._frame = staged._frame
        self._source_frame_id = source_frame_id
        self._working_pose = next_working
        self._working_source_frame_id = next_working_source
        self._last_raw = captured_raw
        self._last_seed_record = captured_record
        return result
