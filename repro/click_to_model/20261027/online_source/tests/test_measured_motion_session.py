"""CPU-only product-session contracts with analytic current-frame observations."""
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from adaptive_refinement import AdaptiveRefinementConfig
from measured_motion_session import (MeasuredMotionSession, MeasuredMotionSessionError,
    DirectoryEvidenceSink, select_initial_queries)
from pose_validity import PoseValidityChecker


def rotation(frame):
    a=np.deg2rad(5*frame);D=np.eye(4);D[:2,:2]=[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]
    return D


@dataclass(frozen=True)
class Observation:
    frame_id:int
    query_ids:np.ndarray
    pixels_xy:np.ndarray
    visible:np.ndarray
    raw_return:object


class Stream:
    def __init__(self,h):
        self.h=h;self.frame_id=None;self.poisoned=False;self.last_raw_return=None;self.calls=[];self.modes={}
    def start(self,rgb,xy,*,frame_id):
        assert frame_id==0 and self.frame_id is None
        self.birth=int(rgb[0,0,0]);self.points=np.c_[xy,np.ones(len(xy))]@np.linalg.inv(self.h.K).T
        return self._forward(rgb,0)
    def advance(self,rgb,*,frame_id):
        assert frame_id==self.frame_id+1
        return self._forward(rgb,frame_id)
    def _forward(self,rgb,local):
        t=int(rgb[0,0,0]);D=rotation(t)@np.linalg.inv(rotation(self.birth))
        xyz=self.points@D[:3,:3].T+D[:3,3];mode=self.modes.get(t)
        if mode=='drift':xyz[:,0]+=.04
        uv=xyz@self.h.K.T;uv=uv[:,:2]/uv[:,2:3]
        visible=np.full(len(uv),mode!='invisible',bool)
        self.frame_id=local;self.calls.append((local,t));self.last_raw_return=dict(native_pixels=uv.copy(),visible=visible.copy())
        if mode=='raise':self.poisoned=True;raise RuntimeError('after_actual_point_return')
        return Observation(local+1 if mode=='clock'else local,np.arange(len(uv)),uv,visible,self.last_raw_return)


class Fixture:
    def __init__(self,*,iterations=1,sink=None,preloss=True,blank=False):
        self.K=np.array([[150.,0,80.],[0,150.,64.],[0,0,1.]])
        self.depth=np.ones((128,160),np.float64);self.mask=np.ones(self.depth.shape,bool)
        yy,xx=np.indices(self.depth.shape);image=(((xx//8+yy//8)%2)*255).astype(np.uint8)
        self.rgb=np.repeat(image[...,None],3,axis=2)
        if blank:self.rgb[:]=0
        xy=np.array([(x,y)for y in np.linspace(-.23,.23,20)for x in np.linspace(-.28,.28,20)])
        p=np.c_[xy,np.zeros(len(xy))];normals=np.tile([0.,0.,-1.],(len(p),1))
        self.checker=PoseValidityChecker(p,normals,.7)
        self.bank=np.repeat(np.eye(4)[None],252,axis=0);self.bank[:,2,3]=1
        self.bank[:,0,3]=np.arange(252)*1e-5
        self.C=np.eye(4);self.C[:3,3]=[-.02,.03,0]
        self.streams=[];self.events=[];self.sink=sink
        def collect(kind,payload):
            self.events.append((kind,payload))
            if self.sink:self.sink(kind,payload)
        self.session=MeasuredMotionSession(self.checker,.7,session_id='cpu_epoch',stream_factory=self.factory,
            evidence_sink=collect,enable_preloss=preloss,
            refinement_config=AdaptiveRefinementConfig(min_iterations=iterations,max_iterations=iterations))
    def factory(self):
        item=Stream(self);self.streams.append(item);return item
    def image(self,t):
        image=self.rgb.copy();image[0,0,0]=t;return image
    def start(self):
        return self.session.start(initial_hypotheses=self.bank,selected_backend_index=17,centered_from_original=self.C,
            rgb=self.image(0),depth_m=self.depth,K=self.K,mask=self.mask,source_frame_id=40,timestamp=1.)
    def step(self,t,callback=None,**kwargs):
        return self.session.step(self.image(t),self.depth,self.K,callback or(lambda p:p.copy()),
            frame_id=t,source_frame_id=40+4*t,timestamp=1+t*.1,**kwargs)
    def ready(self):
        self.start();rows=[self.step(i)for i in range(1,7)]
        assert rows[-1]['diagnostics']['manager_state']=='ready'
        return rows


class SessionTests(unittest.TestCase):
    def assert_tentative(self,d):
        for k in('verified','identity_verified','initial_orientation_resolved','absolute_orientation_resolved',
                 'trusted_memory_updated','memory_updated','seed_admitted'):self.assertIs(d[k],False,k)

    def test_current_GFTT_matches_declared_order_and_depth_exclusions(self):
        h=Fixture();rgb=h.image(0);q=select_initial_queries(rgb,h.depth,h.mask)
        raw=cv2.goodFeaturesToTrack(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),64,.01,5.,
            mask=cv2.erode(h.mask.astype(np.uint8),np.ones((5,5),np.uint8)))[:,0].astype(float)
        np.testing.assert_array_equal(q['pixels_xy'],raw)
        x,y=np.rint(raw[0]).astype(int);h.depth[y,x]=0
        q=select_initial_queries(rgb,h.depth,h.mask)
        np.testing.assert_array_equal(q['pixels_xy'],raw[1:]);self.assertFalse(q['depth_valid_mask'][0])

    def test_textureless_fallback_does_not_construct_or_replace_queries(self):
        h=Fixture(blank=True);initial=h.start();self.assertFalse(initial['diagnostics']['point_initialization_available'])
        self.assertEqual(len(h.streams),0);seen=[]
        row=h.step(1,lambda p:seen.append(p.copy())or p)
        np.testing.assert_array_equal(seen[0],h.bank[17]);self.assertEqual(row['diagnostics']['selected_motion_source'],'fallback')
        self.assert_tentative(row['diagnostics']);self.assertFalse(row['diagnostics']['seed_admitted'])

    def test_current_measured_seed_and_original_coordinate_conversion(self):
        h=Fixture();h.start();seen=[];row=h.step(1,lambda p:seen.append(p.copy())or p)
        np.testing.assert_allclose(seen[0],rotation(1)@h.bank[17],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(row['pose_original'],row['pose']@h.C)
        self.assertEqual(h.streams[0].calls,[(0,0),(1,1)]);self.assert_tentative(row['diagnostics'])

    def test_measured_birth_ready_then_original_loss_uses_independent_secondary(self):
        h=Fixture();h.ready();self.assertEqual(len(h.streams),2)
        self.assertEqual(h.streams[1].calls,[(0,3),(1,4),(2,5),(3,6)])
        h.streams[0].modes[7]='invisible';seen=[];row=h.step(7,lambda p:seen.append(p.copy())or p)
        self.assertEqual(row['diagnostics']['selected_motion_source'],'secondary')
        np.testing.assert_allclose(seen[0],rotation(7)@h.bank[17],atol=1e-12,rtol=0)
        self.assert_tentative(row['diagnostics'])

    def test_conflict_withholds_point_seed_and_uses_neural_working_state(self):
        h=Fixture();h.ready();previous=h.session.working_pose;h.streams[1].modes[7]='drift';seen=[]
        row=h.step(7,lambda p:seen.append(p.copy())or p)
        self.assertEqual(row['diagnostics']['selected_motion_source'],'conflict')
        np.testing.assert_array_equal(seen[0],previous);self.assert_tentative(row['diagnostics'])

    def test_invalid_alternative_preserved_and_manager_disabled_without_filter(self):
        h=Fixture();h.bank[200,0,0]=np.nan;row=h.start()
        self.assertEqual(len(h.streams),0);self.assertIn('full_initial_bank_unsupported',row['diagnostics']['point_unavailable_reason'])
        saved=next(payload['payload']['initial_hypotheses']for kind,payload in h.events if kind=='registration')
        self.assertEqual(saved.shape,(252,4,4));self.assertTrue(np.isnan(saved[200,0,0]));self.assert_tentative(h.step(1)['diagnostics'])

    def test_clock_intrinsics_and_backwards_time_close_without_backend_consumption(self):
        for kind in ('input','source','time','K'):
            h=Fixture();h.start();kw=dict(frame_id=1,source_frame_id=44,timestamp=1.1);K=h.K.copy()
            if kind=='input':kw['frame_id']=2
            if kind=='source':kw['source_frame_id']=40
            if kind=='time':kw['timestamp']=.9
            if kind=='K':K[0,2]+=1
            with self.assertRaises(MeasuredMotionSessionError):h.session.step(h.image(1),h.depth,K,lambda p:p,**kw)
            self.assertEqual(h.streams[0].calls,[(0,0)]);self.assertTrue(h.session.closed);self.assertEqual(h.session.frame_id,0)

    def test_late_neural_exception_keeps_actual_prefix_and_old_working(self):
        h=Fixture(iterations=2);h.start();previous=h.session.working_pose;calls=[]
        def refine(p):
            calls.append(p.copy())
            if len(calls)==2:raise RuntimeError('second actual invocation failed')
            return p
        with self.assertRaises(MeasuredMotionSessionError)as caught:h.step(1,refine)
        self.assertEqual(len(caught.exception.raw_returns),1)
        np.testing.assert_array_equal(h.session.working_pose,previous);self.assertTrue(h.session.closed)

    def test_final_evidence_failure_does_not_publish_working_state(self):
        def sink(kind,payload):
            if kind=='frame_result':raise OSError('disk full after actual return')
        h=Fixture(sink=sink);h.start();previous=h.session.working_pose
        with self.assertRaises(MeasuredMotionSessionError)as caught:h.step(1)
        self.assertEqual(len(caught.exception.raw_returns),1);np.testing.assert_array_equal(h.session.working_pose,previous)
        self.assertTrue(h.session.closed);self.assertIsNone(h.session.close('cleanup').get('evidence_error'))

    def test_return_evidence_failure_keeps_return_in_exception(self):
        def sink(kind,payload):
            if kind=='neural_return':raise OSError('return write failed')
        h=Fixture(sink=sink);h.start()
        with self.assertRaises(MeasuredMotionSessionError)as caught:h.step(1)
        self.assertEqual(len(caught.exception.raw_returns),1)
        failure=caught.exception.diagnostics;self.assertEqual(failure['stage'],'neural_refinement')

    def test_actual_point_return_survives_wrong_public_clock(self):
        h=Fixture();h.start();h.streams[0].modes[1]='clock'
        with self.assertRaises(MeasuredMotionSessionError):h.step(1)
        raw=[p['payload']['raw_return']for k,p in h.events if k=='original_point_return']
        self.assertEqual(len(raw),2);self.assertIn('native_pixels',raw[-1]);self.assertEqual(len(h.session.last_raw_returns),0)

    def test_sink_preserves_numeric_nonfinite_and_no_pickle(self):
        with tempfile.TemporaryDirectory()as d:
            sink=DirectoryEvidenceSink(Path(d)/'evidence');a=np.array([1.,np.nan,np.inf]);sink('raw',dict(value=a,scalar=float('nan')))
            a[0]=9
            record=json.loads(next((Path(d)/'evidence').glob('*.json')).read_text())
            with np.load(next((Path(d)/'evidence').glob('*.npz')),allow_pickle=False)as z:
                np.testing.assert_array_equal(z['array_000000'],[1.,np.nan,np.inf])
            self.assertEqual(record['payload']['scalar'],{'nonfinite_float':'nan'})
            with self.assertRaises(FileExistsError):DirectoryEvidenceSink(Path(d)/'evidence')

    def test_close_is_terminal_and_fresh_epoch_has_new_recurrence(self):
        h=Fixture();h.start();h.session.close('reclick')
        with self.assertRaises(RuntimeError):h.step(1)
        with self.assertRaises(RuntimeError):h.start()
        other=Fixture();other.start();self.assertIsNot(h.streams[0],other.streams[0])

    def test_checker_mutation_is_rejected_before_point_backend(self):
        h=Fixture();h.start();h.checker.points[0,0]+=.01
        with self.assertRaises(MeasuredMotionSessionError):h.step(1)
        self.assertEqual(h.streams[0].calls,[(0,0)])

    def test_actual_initial_original_binding_rejects_wrong_index_or_centering(self):
        for mode in('valid_float32','wrong_pose','malformed','nonfinite'):
            h=Fixture();actual=(h.bank[17]@h.C).astype(np.float32)
            if mode=='wrong_pose':actual[0,3]+=.01
            if mode=='malformed':actual=actual[None]
            if mode=='nonfinite':actual[0,0]=np.nan
            kwargs=dict(initial_hypotheses=h.bank,selected_backend_index=17,centered_from_original=h.C,
                rgb=h.image(0),depth_m=h.depth,K=h.K,mask=h.mask,initial_pose_original=actual)
            if mode=='valid_float32':
                value=h.session.start(**kwargs)['diagnostics']['initial_coordinate_check']
                self.assertTrue(value['checked']);self.assertEqual(value['actual_dtype'],'float32')
            else:
                with self.assertRaisesRegex(MeasuredMotionSessionError,'actual initial original'):h.session.start(**kwargs)
                self.assertEqual(len(h.streams),0)


if __name__=='__main__':unittest.main()
