"""Actual offline entry + measured session, CPU FP/TAP fixtures, no GPU/GT."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tools import run_click_to_model as entry
import test_preloss_offline_entry as legacy_fixture


class EntryOutputTests(unittest.TestCase):
    run_product = legacy_fixture.ProductWiringTests.run_product
    def arguments(self, root):
        args=legacy_fixture.OfflinePolicyTests.arguments(self,root)
        if getattr(self,'enable_continuity',True):args+=['--output-policy','continuity']
        return args

    def test_cli_rejects_non_measured_before_inputs(self):
        args=['--sequence','/missing','--mesh','/missing.ply','--output-dir','/missing-output',
              '--output-policy','continuity']
        with patch.object(entry,'frame_names',side_effect=AssertionError('input stage reached')):
            with self.assertRaises(SystemExit)as raised:entry.main(args)
        self.assertEqual(raised.exception.code,2)

    def test_default_legacy_has_no_publication_artifacts(self):
        self.enable_continuity=False
        root,report,details,stats,error,_=self.run_product(click=True)
        self.assertIsNone(error);self.assertEqual(report['output_policy'],'legacy')
        self.assertFalse((root/'output/published_poses.npz').exists())
        self.assertFalse((root/'output/output_policy.json').exists())
        self.assertFalse(hasattr(stats['sessions'][0],'output_policy'))

    def test_real_entry_online_publish_keeps_actual_candidate_bytes(self):
        root,report,details,stats,error,_=self.run_product(click=True)
        self.assertIsNone(error);self.assertEqual(report['output_policy'],'continuity')
        self.assertTrue(stats['sessions'][0].output_policy.closed)
        record=json.loads((root/'output/output_policy.json').read_text())
        self.assertEqual(record['planned_frames'],3);self.assertEqual(record['decided_frames'],3)
        candidate=np.load(root/'output/poses.npy',allow_pickle=False)
        with np.load(root/'output/published_poses.npz',allow_pickle=False)as arrays:
            np.testing.assert_array_equal(arrays['decision_recorded'],np.ones(3,bool))
            self.assertTrue(arrays['pose_available'].all())
            for i,row in enumerate(record['frames']):
                self.assertEqual(row['status'],'decided')
                self.assertEqual(arrays[row['candidate_array']].dtype,candidate[i].dtype)
                self.assertEqual(arrays[row['candidate_array']].tobytes(),candidate[i].tobytes())
                self.assertEqual(arrays[row['published_array']].tobytes(),candidate[i].tobytes())
                for key in entry.TENTATIVE_TRUST_FIELDS:self.assertIs(row['decision'][key],False)
        self.assertIn('identity_continuity.py',report['source_sha256'])
        self.assertIn('identity_output.py',report['source_sha256'])
        self.assertEqual(report['output_policy_summary']['pose_available_frames'],3)

    def test_unestablished_textureless_queries_abstain_not_fake_pose(self):
        root,report,_,_,error,_=self.run_product(texture=False,click=True)
        self.assertIsNone(error);self.assertEqual(report['status'],'completed')
        record=json.loads((root/'output/output_policy.json').read_text())
        self.assertEqual(record['decided_frames'],3);self.assertEqual(record['pose_available_frames'],0)
        self.assertEqual(len(np.load(root/'output/poses.npy',allow_pickle=False)),3)
        with np.load(root/'output/published_poses.npz',allow_pickle=False)as arrays:
            self.assertFalse(arrays['pose_available'].any());self.assertTrue(arrays['decision_recorded'].all())
            self.assertFalse(any(k.startswith('published_frame_')for k in arrays.files))
        for row in record['frames']:
            self.assertEqual(row['decision']['status'],'initial_continuity_unestablished')
            self.assertIsNone(row['published_array'])

    def test_publication_sink_failure_retains_candidate_and_stops_epoch(self):
        from measured_motion_session import DirectoryEvidenceSink
        original=DirectoryEvidenceSink.__call__
        def fail(sink,kind,envelope):
            if kind=='output_publication'and envelope['frame_id']==1:raise OSError('publication sink failed')
            return original(sink,kind,envelope)
        with patch.object(DirectoryEvidenceSink,'__call__',fail):
            root,report,_,stats,error,_=self.run_product(click=True)
        self.assertIsNotNone(error);self.assertIn('publication sink failed',str(error))
        self.assertEqual(report['status'],'failed');self.assertTrue(stats['sessions'][0].closed)
        self.assertTrue(stats['sessions'][0].output_policy.closed)
        record=json.loads((root/'output/output_policy.json').read_text())
        self.assertEqual([r['status']for r in record['frames']],['decided','failed','not_executed'])
        self.assertEqual(len(np.load(root/'output/poses.npy',allow_pickle=False)),2)
        with np.load(root/'output/published_poses.npz',allow_pickle=False)as arrays:
            self.assertIn('candidate_frame_000001',arrays.files)
            self.assertNotIn('published_frame_000001',arrays.files)
            np.testing.assert_array_equal(arrays['decision_recorded'],[True,False,False])

    def test_actual_FP_failure_before_decision_has_no_publication(self):
        root,report,_,stats,error,_=self.run_product(click=True,fail_track=True)
        self.assertIsNotNone(error);self.assertTrue(stats['sessions'][0].output_policy.closed)
        record=json.loads((root/'output/output_policy.json').read_text())
        failed=next(s['frame_id']for s in report['planned_frame_slots']if s['status']=='failed')
        self.assertEqual(record['frames'][failed]['status'],'failed_before_publication')
        self.assertFalse(record['frames'][failed]['pose_available'])
        with np.load(root/'output/published_poses.npz',allow_pickle=False)as arrays:
            self.assertNotIn(f'published_frame_{failed:06d}',arrays.files)

    def test_bad_policy_return_or_cleanup_exception_never_publishes(self):
        actual=np.eye(4,dtype=np.float32)
        for fault in ('abstain_with_pose','wrong_clock','mutate_candidate','trust'):
            with self.subTest(fault=fault):
                closed=[]
                def publish(result,pose):
                    row=dict(frame_id=0,source_frame_id=0,candidate_pose=pose.copy(),published_pose=pose.copy(),
                        pose_available=True,status='tentative_continuous',reason='fixture',**{k:False for k in entry.TENTATIVE_TRUST_FIELDS})
                    if fault=='abstain_with_pose':row['pose_available']=False
                    elif fault=='wrong_clock':row['source_frame_id']=2
                    elif fault=='trust':row['identity_verified']=True
                    else:pose[0,3]=.2;row['candidate_pose']=pose;row['published_pose']=pose
                    return row
                def bad_close():closed.append(True);raise RuntimeError('cleanup failure')
                owner=SimpleNamespace(output_policy=SimpleNamespace(publish=publish,close=bad_close),close=lambda **kw:closed.append(True))
                slots=[dict(frame_id=0,source_frame_id=0,status='not_executed',pose_available=None,published_array=None)];arrays={}
                with self.assertRaises(ValueError):entry.publish_measured_output(owner,{},actual,frame_id=0,slots=slots,arrays=arrays)
                self.assertEqual(closed,[True,True]);self.assertEqual(slots[0]['status'],'failed')
                self.assertNotIn('published_frame_000000',arrays)
                self.assertEqual(arrays['candidate_frame_000000'].tobytes(),actual.tobytes())


if __name__=='__main__':unittest.main()
