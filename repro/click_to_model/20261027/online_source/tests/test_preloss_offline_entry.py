"""Actual product entrypoint wiring with CPU FP/TAP returns, no GPU or GT."""
from contextlib import ExitStack
import hashlib
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tools import run_click_to_model as entry
from tools import run_meshless_click_to_model as meshless


class OfflinePolicyTests(unittest.TestCase):
    def arguments(self, root):
        return ['--sequence', str(root/'seq'), '--mesh', str(root/'mesh.obj'),
                '--output-dir', str(root/'output'), '--tracking-policy', 'tentative-preloss',
                '--tapnext-source-root', str(root/'tapnet'), '--tapnext-checkpoint', str(root/'tap.pt'),
                '--tapnext-checkpoint-sha256', 'a'*64]

    def test_conflicts_rejected_before_input_scan_or_model_import(self):
        conflicts = [('--seed-backend', 'lk'), ('--seed-backend', 'trusted-view'),
                     ('--initialization-admission', 'legacy'), ('--mode', 'baseline'),
                     ('--mode', 'validity'), ('--mode', 'rgbd'), ('--motion-prior', 'constant_velocity'),
                     ('--quarantined-recovery',), ('--disable-urgent-replenishment',)]
        with tempfile.TemporaryDirectory() as tmp:
            for flags in conflicts:
                with self.subTest(flags=flags), patch.object(entry, 'frame_names', side_effect=AssertionError('too late')):
                    with self.assertRaises(SystemExit) as exc:
                        entry.main(self.arguments(Path(tmp))+list(flags))
                    self.assertEqual(exc.exception.code, 2)
            self.assertFalse((Path(tmp)/'output').exists())

    def test_checkpoint_contract_missing_or_unused_rejected_before_inputs(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(entry, 'frame_names', side_effect=AssertionError('too late')):
            args = self.arguments(Path(tmp))
            with self.assertRaises(SystemExit):
                entry.main(args[:-2])
            legacy = args.copy()
            legacy[legacy.index('tentative-preloss')] = 'legacy'
            with self.assertRaises(SystemExit):
                entry.main(legacy)

    def test_meshless_forwards_policy_and_exact_pinned_inputs_only_to_tracking(self):
        p = meshless.parser()
        args = p.parse_args(['--sequence', '/tmp/seq', '--click', '10', '20', '--depth-scale', '.001',
            '--output-dir', '/tmp/out', '--tracking-policy', 'tentative-preloss',
            '--tapnext-source-root', '/tmp/tapnet', '--tapnext-checkpoint', '/tmp/local.pt',
            '--tapnext-checkpoint-sha256', 'b'*64])
        commands = dict(meshless.build_commands(args, Path('/tmp/out')))
        command = commands['tracking']
        for name, value in [('--tracking-policy', 'tentative-preloss'),
                ('--tapnext-source-root', '/tmp/tapnet'), ('--tapnext-checkpoint', '/tmp/local.pt'),
                ('--tapnext-checkpoint-sha256', 'b'*64), ('--initialization-admission', 'all-hypotheses')]:
            self.assertEqual(command[command.index(name)+1], value)
        for stage in ('segmentation', 'generation'):
            self.assertNotIn('--tracking-policy', commands[stage])
        self.assertEqual(command[command.index('--mask')+1], '/tmp/out/segmentation/selected_mask.png')
        self.assertEqual(command[command.index('--mesh')+1], '/tmp/out/generation/mesh/model.obj')

    def test_meshless_policy_rejected_before_generation_work(self):
        args = meshless.parser().parse_args(['--sequence', '/tmp/seq', '--click', '10', '20',
                '--depth-scale', '.001', '--output-dir', '/tmp/out', '--tracking-policy', 'tentative-preloss'])
        with patch.object(meshless, 'dependencies', side_effect=AssertionError('model stage reached')), \
                patch.object(meshless, 'first_observation', side_effect=AssertionError('input stage reached')):
            with self.assertRaisesRegex(ValueError, 'requires'):
                meshless.run(args)


class ProductWiringTests(unittest.TestCase):
    arguments = OfflinePolicyTests.arguments
    def run_product(self, *, texture=True, ambiguous=False, bad_alternative=False, callback_save_error=False,
                    click=False, fail_track=False, multiple_masks=False, malformed_return=None,
                    callback_fault=None, trust_fault=None, centered_layout='singleton_batch',
                    input_frames=3, max_frames=None):
        import torch
        import trimesh
        import measured_motion_session as session_module
        from tools.tapnext_stream import NativeStep, TAPNextRawReturn
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        sequence = root/'seq'
        for name in ('rgb', 'depth', 'masks'):
            (sequence/name).mkdir(parents=True)
        yy, xx = np.mgrid[:64, :64]
        gray = (((xx//6+yy//6) % 2)*255).astype(np.uint8) if texture else np.zeros((64, 64), np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        depth = np.ones((64, 64), np.float32)
        K = np.array([[100., 0, 32], [0, 100., 32], [0, 0, 1.]])
        mask = np.ones((64, 64), bool)
        for i in range(input_frames):
            cv2.imwrite(str(sequence/'rgb'/f'{i:06d}.png'), rgb)
            cv2.imwrite(str(sequence/'depth'/f'{i:06d}.png'), (depth*1000).astype(np.uint16))
        cv2.imwrite(str(sequence/'masks/000000.png'), mask.astype(np.uint8)*255)
        np.savetxt(sequence/'cam_K.txt', K)
        center = np.array([.04, -.02, 0.])
        mesh = trimesh.Trimesh(vertices=np.array([[-.1, -.1, 0], [-.1, .1, 0], [.1, .1, 0], [.1, -.1, 0]])+center,
                               faces=[[0, 1, 2], [0, 2, 3]], process=False)
        mesh.export(root/'mesh.obj')
        C = np.eye(4, dtype=np.float32)
        C[:3, 3] = -center
        Q = np.eye(4, dtype=np.float32)
        Q[2, 3] = 1
        stats = {'backend_instances': 0, 'point_calls': 0, 'FP_calls': [], 'sessions': [], 'starts': []}

        class FP:
            def __init__(self, model_pts, model_normals, mesh, **kwargs):
                self.mesh = mesh.copy()
                self.mesh.apply_translation(-center)
                self.model_center, self.diameter = center.copy(), .3
                self.pose_last = None
            def get_tf_to_centered_mesh(self):
                return torch.tensor(C.copy())
            def register(self, **kwargs):
                bank = [Q.copy()]
                if ambiguous or bad_alternative:
                    alternative = Q.copy()
                    alternative[:2, :2] *= -1
                    if bad_alternative:
                        alternative[0, 0] = np.nan
                    bank.append(alternative)
                self.poses = torch.tensor(np.stack(bank))
                self.scores = torch.tensor(np.arange(len(bank), 0, -1), dtype=torch.float32)
                self.best_id = 99  # Exposed bank order, never this unsorted index.
                self.pose_last = self.poses[0].clone()
                return self.pose_last.numpy()@C
            def _pose_last_from_numpy(self, seed):
                self.pose_last = torch.tensor(np.asarray(seed).copy(), dtype=torch.float32)
            def track_one(self, image, current_depth, current_K, *, iteration):
                stats['FP_calls'].append(dict(seed=self.pose_last.numpy().copy(), rgb=image.copy(),
                    depth=current_depth.copy(), K=current_K.copy(), iteration=iteration))
                if fail_track and len(stats['FP_calls']) == 2:
                    raise RuntimeError('synthetic actual FP failure')
                returned = self.pose_last.numpy().copy()@C
                if len(stats['FP_calls']) == 1:
                    if callback_fault == 'inconsistent_original':
                        returned[0, 3] += .05
                    elif callback_fault == 'nonfinite_original':
                        returned[0, 3] = np.nan
                    elif callback_fault == 'nonfinite_centered':
                        self.pose_last[0, 3] = float('nan')
                if centered_layout == 'singleton_batch':
                    self.pose_last = self.pose_last[None]
                if malformed_return == 'centered':
                    self.pose_last = self.pose_last[None]
                return returned[None] if malformed_return == 'original' else returned

        class TAP:
            def __init__(self, config):
                stats['backend_instances'] += 1
                self.config, self.last_raw_return = config, None
                self.provenance = {'backend': 'synthetic CPU fixture; no pretrained inference'}
            def forward(self, image, *, query_tyx, state):
                stats['point_calls'] += 1
                q = query_tyx.copy() if state is None else state.query_points.copy()
                step = 1 if state is None else state.step+1
                count = q.shape[1]
                raw = TAPNextRawReturn(q[:, None, :, 1:].copy(), np.zeros((1, 1, count, 512), np.float32),
                    np.full((1, 1, count, 1), 10, np.float32), q.copy(), step, 4, ())
                self.last_raw_return = raw
                return NativeStep(raw, SimpleNamespace(step=step, query_points=q))

        real_start = session_module.MeasuredMotionSession.start
        def observed_start(owner, **kwargs):
            stats['sessions'].append(owner)
            stats['starts'].append(kwargs)
            result = real_start(owner, **kwargs)
            if trust_fault is not None and trust_fault[0] == 'start':
                result['diagnostics'][trust_fault[1]] = True
            return result
        real_step = session_module.MeasuredMotionSession.step
        def observed_step(owner, *args, **kwargs):
            result = real_step(owner, *args, **kwargs)
            if trust_fault is not None and trust_fault[0] == 'step':
                result['diagnostics'][trust_fault[1]] = True
            return result
        real_save = session_module.DirectoryEvidenceSink.__call__
        def saved(sink, kind, payload):
            if callback_save_error and kind == 'fp_callback':
                raise OSError('synthetic callback evidence save failure')
            return real_save(sink, kind, payload)
        args = self.arguments(root)
        if max_frames is not None:
            args += ['--max-frames', str(max_frames)]
        if click:
            args += ['--click', '32', '32']
        error = None
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {'estimater': SimpleNamespace(
                FoundationPose=FP, set_seed=lambda seed: np.random.seed(seed), logging=logging)}))
            for name, value in [('torch.cuda.is_available', True), ('torch.cuda.get_device_name', 'CPU fixture')]:
                stack.enter_context(patch(name, return_value=value))
            stack.enter_context(patch('torch.cuda.synchronize'))
            stack.enter_context(patch('tools.tapnext_stream.TorchTAPNextPPBackend', TAP))
            stack.enter_context(patch.object(session_module.MeasuredMotionSession, 'start', observed_start))
            stack.enter_context(patch.object(session_module.MeasuredMotionSession, 'step', observed_step))
            stack.enter_context(patch.object(session_module.DirectoryEvidenceSink, '__call__', saved))
            stack.enter_context(patch('initialization_session.InitializationTrackingSession.start',
                                      side_effect=AssertionError('trusted startup forbidden')))
            stack.enter_context(patch('click_to_model.ClickToModelTracker.__init__',
                                      side_effect=AssertionError('trusted tracker construction forbidden')))
            stack.enter_context(patch('rgbd_seed.RGBDSeedTracker.__init__',
                                      side_effect=AssertionError('trusted point memory forbidden')))
            if click:
                proposed_masks, scores = np.array([mask]), [1.]
                if multiple_masks:
                    smaller = np.zeros_like(mask)
                    smaller[16:48, 16:48] = True
                    proposed_masks, scores = np.array([mask, smaller]), [.1, .9]
                stack.enter_context(patch.object(entry, 'sam2_mask_candidates', return_value=(proposed_masks, scores)))
            try:
                code = entry.main(args)
                self.assertEqual(code, 0)
            except Exception as exc:
                error = exc
        report = json.loads((root/'output/report.json').read_text())
        details = json.loads((root/'output/diagnostics.json').read_text())
        return root, report, details, stats, error, (rgb, depth, K, C)

    def assert_tentative(self, report, details):
        fields = entry.TENTATIVE_TRUST_FIELDS
        for item in [report, *details]:
            for key in fields:
                self.assertIs(item[key], False, key)
        self.assertFalse(report['trusted_memory_initialized'])

    def test_actual_entry_uses_shared_session_current_click_and_FP_callback(self):
        root, report, details, stats, error, inputs = self.run_product(click=True)
        self.assertIsNone(error)
        self.assertEqual(report['configuration']['class_name'], 'MeasuredMotionSession')
        self.assertEqual(len(stats['sessions']), 1)
        self.assertEqual(stats['backend_instances'], 1)
        self.assertEqual(stats['point_calls'], 3)
        self.assertTrue(report['initialization_admission']['accepted'])
        self.assertFalse(report['initialization_admission']['assessment_installs_trusted_memory'])
        self.assert_tentative(report, details)
        self.assertTrue(all(s['status'] == 'completed' for s in report['planned_frame_slots']))
        self.assertEqual(stats['starts'][0]['selected_backend_index'], 0)
        self.assertEqual(stats['starts'][0]['initial_hypotheses'].shape, (1, 4, 4))
        for call in stats['FP_calls']:
            for key, expected in zip(('rgb', 'depth', 'K'), inputs[:3]):
                np.testing.assert_array_equal(call[key], expected)
            self.assertEqual(call['iteration'], 1)
            self.assertEqual(call['seed'].dtype, np.float32)
        with np.load(root/'output/actual_fp_callback_prefix.npz', allow_pickle=False) as z:
            installed = [z[k] for k in z.files if k.endswith('installed_centered_seed')]
            original = [z[k] for k in z.files if k.endswith('actual_original_return')]
            raw_centered = [z[k] for k in z.files if k.endswith('actual_centered_return')]
            controller_centered = [z[k] for k in z.files if k.endswith('controller_centered_return')]
        self.assertEqual(len(installed), len(stats['FP_calls']))
        self.assertEqual(len(raw_centered), len(stats['FP_calls']))
        for raw_value, single_value in zip(raw_centered, controller_centered):
            self.assertEqual(raw_value.shape, (1, 4, 4))
            self.assertEqual(single_value.shape, (4, 4))
            self.assertEqual(single_value.dtype, raw_value.dtype)
            np.testing.assert_array_equal(raw_value[0], single_value)
        np.testing.assert_array_equal(np.load(root/'output/poses.npy')[-1], original[-1])
        self.assertFalse(np.array_equal(original[-1], stats['FP_calls'][-1]['seed']))

    def test_single_physical_frame_default_and_explicit_limit(self):
        for limit in (None, 1):
            with self.subTest(max_frames=limit):
                root, report, details, stats, error, _ = self.run_product(
                    click=True, input_frames=1, max_frames=limit)
                self.assertIsNone(error)
                self.assertEqual(report['status'], 'completed')
                self.assertEqual(report['frames'], 1)
                self.assertEqual(report['completed_frames'], 1)
                self.assertEqual(report['sequence_frames'], 1)
                self.assertEqual(report['tracking_seconds'], 0)
                for key in ('tracking_fps', 'latency_p50_ms', 'latency_p95_ms'):
                    self.assertIsNone(report[key])
                for key in ('neural_iterations', 'status_counts', 'seed_sources'):
                    self.assertEqual(report[key], {})
                self.assertEqual(len(stats['starts']), 1)
                self.assertEqual(stats['FP_calls'], [])
                self.assertEqual(len(details), 1)
                self.assertEqual(np.load(root/'output/poses.npy').shape, (1, 4, 4))
                self.assertEqual(np.load(root/'output/latency_seconds.npy').size, 0)
                with np.load(root/'output/raw_neural_iterations.npz', allow_pickle=False) as z:
                    self.assertEqual(z.files, [])
                self.assertEqual([s['status'] for s in report['planned_frame_slots']], ['completed'])
                self.assert_tentative(report, details)

    def test_one_frame_limit_keeps_actual_sequence_count(self):
        _, report, details, stats, error, _ = self.run_product(max_frames=1)
        self.assertIsNone(error)
        self.assertEqual(report['sequence_frames'], 3)
        self.assertEqual(report['frames'], 1)
        self.assertTrue(report['development_prefix'])
        self.assertEqual(stats['FP_calls'], [])
        self.assert_tentative(report, details)

    def test_unresolved_assessment_keeps_full_actual_bank(self):
        _, report, details, stats, error, _ = self.run_product(ambiguous=True)
        self.assertIsNone(error)
        self.assertFalse(report['initialization_admission']['accepted'])
        self.assertEqual(stats['starts'][0]['initial_hypotheses'].shape, (2, 4, 4))
        self.assert_tentative(report, details)

    def test_low_texture_current_frame_falls_back_without_point_model_call(self):
        _, report, details, stats, error, _ = self.run_product(texture=False)
        self.assertIsNone(error)
        self.assertEqual(stats['point_calls'], 0)
        self.assertGreater(len(stats['FP_calls']), 0)
        self.assertFalse(details[0]['point_initialization_available'])
        self.assertIn('fewer_than_eight', details[0]['point_unavailable_reason'])
        self.assert_tentative(report, details)

    def test_bad_alternative_is_retained_and_neural_only_is_explicit(self):
        root, report, details, stats, error, _ = self.run_product(bad_alternative=True)
        self.assertIsNone(error)
        bank = stats['starts'][0]['initial_hypotheses']
        self.assertEqual(bank.shape, (2, 4, 4))
        self.assertTrue(np.isnan(bank[1, 0, 0]))
        self.assertEqual(stats['point_calls'], 0)
        self.assertIn('full_initial_bank_unsupported', details[0]['point_unavailable_reason'])
        self.assert_tentative(report, details)

    def test_actual_return_survives_callback_evidence_write_failure(self):
        root, report, details, stats, error, _ = self.run_product(callback_save_error=True)
        self.assertIsInstance(error, OSError)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual([s['status'] for s in report['planned_frame_slots']], ['completed', 'failed', 'not_executed'])
        records = json.loads((root/'output/actual_fp_callback_prefix.json').read_text())
        self.assertTrue(records[-1]['returned'])
        with np.load(root/'output/actual_fp_callback_prefix.npz', allow_pickle=False) as z:
            self.assertTrue(any(k.endswith('actual_original_return') for k in z.files))
            self.assertTrue(any(k.endswith('actual_centered_return') for k in z.files))
        self.assert_tentative(report, details)

    def test_multiple_click_masks_retain_every_bank_and_use_selected_mask(self):
        root, report, details, stats, error, _ = self.run_product(click=True, multiple_masks=True)
        self.assertIsNone(error)
        self.assertEqual(report['initialization_candidates']['selected_index'], 1)
        self.assertEqual(report['initialization_admission']['selected_mask_id'], 1)
        start = stats['starts'][0]
        self.assertEqual(start['registration_metadata']['selected_mask_id'], 1)
        self.assertEqual(len(start['registration_metadata']['all_registration_bundles']), 2)
        expected_mask = np.zeros((64, 64), bool)
        expected_mask[16:48, 16:48] = True
        np.testing.assert_array_equal(start['mask'], expected_mask)
        for index in (0, 1):
            path = root/f'output/registration_hypotheses/mask_{index:02d}/backend_arrays.npz'
            with np.load(path, allow_pickle=False) as z:
                self.assertEqual(z['poses_centered'].shape, (1, 4, 4))
        self.assert_tentative(report, details)

    def test_actual_FP_exception_preserves_prefix_without_fabricating_return(self):
        root, report, details, stats, error, _ = self.run_product(fail_track=True)
        self.assertIsNotNone(error)
        self.assertEqual(report['status'], 'failed')
        records = json.loads((root/'output/actual_fp_callback_prefix.json').read_text())
        self.assertEqual(len(stats['FP_calls']), 2)
        self.assertTrue(records[0]['returned'])
        self.assertFalse(records[-1]['returned'])
        self.assertNotIn('actual_original_return', records[-1])
        self.assertIn('synthetic actual FP failure', records[-1]['error'])
        failed_frame = records[-1]['frame_id']
        self.assertEqual(report['planned_frame_slots'][failed_frame]['status'], 'failed')
        self.assertTrue(all(s['status'] == 'not_executed' for s in report['planned_frame_slots'][failed_frame+1:]))
        with np.load(root/f'output/failed_frame_{failed_frame:06d}_raw.npz', allow_pickle=False) as z:
            self.assertEqual(len(z.files), 1)
        self.assert_tentative(report, details)

    def test_malformed_actual_return_shape_is_preserved_and_never_repaired(self):
        for malformed in ('centered', 'original'):
            with self.subTest(malformed=malformed):
                root, report, details, _, error, _ = self.run_product(malformed_return=malformed)
                self.assertIsNotNone(error)
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(report['completed_frames'], 1)
                self.assertEqual(report['planned_frame_slots'][1]['status'], 'failed')
                with np.load(root/'output/actual_fp_callback_prefix.npz', allow_pickle=False) as z:
                    values = [z[k] for k in z.files if k.endswith(f'actual_{malformed}_return')]
                    self.assertEqual(values[0].shape, (1, 1, 4, 4) if malformed == 'centered' else (1, 4, 4))
                self.assert_tentative(report, details)

    def test_inconsistent_first_actual_original_stops_before_later_callback(self):
        root, report, details, stats, error, inputs = self.run_product(callback_fault='inconsistent_original')
        self.assertIsNotNone(error)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(len(stats['FP_calls']), 1)
        self.assertEqual([s['status'] for s in report['planned_frame_slots']], ['completed', 'failed', 'not_executed'])
        records = json.loads((root/'output/actual_fp_callback_prefix.json').read_text())
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]['track_one_returned'])
        self.assertFalse(records[0]['returned'])
        self.assertGreater(records[0]['conversion_max_abs_difference'], .049)
        with np.load(root/'output/actual_fp_callback_prefix.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(z[records[0]['registered_centering']], inputs[3])
            self.assertEqual(z[records[0]['actual_original_return']].shape, (4, 4))
            self.assertEqual(z[records[0]['actual_centered_return']].shape, (1, 4, 4))
        self.assertEqual(np.load(root/'output/poses.npy').shape, (1, 4, 4))
        self.assert_tentative(report, details)

    def test_nonfinite_actual_returns_retained_before_rejection(self):
        for coordinate in ('original', 'centered'):
            with self.subTest(coordinate=coordinate):
                root, report, details, stats, error, _ = self.run_product(callback_fault=f'nonfinite_{coordinate}')
                self.assertIsNotNone(error)
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(len(stats['FP_calls']), 1)
                records = json.loads((root/'output/actual_fp_callback_prefix.json').read_text())
                self.assertFalse(records[0]['returned'])
                with np.load(root/'output/actual_fp_callback_prefix.npz', allow_pickle=False) as z:
                    self.assertTrue(np.isnan(z[records[0][f'actual_{coordinate}_return']]).any())
                self.assertEqual(len(details), 1)
                self.assert_tentative(report, details)

    def test_unexpected_trust_flag_prevents_frame_publication(self):
        for stage in ('start', 'step'):
            for key in ('memory_updated', 'verified'):
                with self.subTest(stage=stage, key=key):
                    root, report, details, stats, error, _ = self.run_product(trust_fault=(stage, key))
                    self.assertIsNotNone(error)
                    self.assertEqual(report['status'], 'failed')
                    expected = 0 if stage == 'start' else 1
                    self.assertEqual(len(details), expected)
                    self.assertEqual(report['saved_pose_frames'], expected)
                    self.assertEqual(report['completed_frames'], expected)
                    self.assertEqual(report['planned_frame_slots'][expected]['status'], 'failed')
                    self.assertTrue(all(s['status'] == 'not_executed' for s in report['planned_frame_slots'][expected+1:]))
                    self.assert_tentative(report, details)

    def test_explicit_single_matrix_centered_layout_remains_supported(self):
        root, report, details, stats, error, _ = self.run_product(centered_layout='single_matrix')
        self.assertIsNone(error)
        records = json.loads((root/'output/actual_fp_callback_prefix.json').read_text())
        self.assertTrue(all(row['centered_return_layout']=='single_4x4' for row in records))
        with np.load(root/'output/actual_fp_callback_prefix.npz', allow_pickle=False) as z:
            for row in records:
                np.testing.assert_array_equal(z[row['actual_centered_return']], z[row['controller_centered_return']])
        self.assert_tentative(report, details)

    def test_other_centered_shapes_and_batched_original_still_rejected(self):
        for shape in ((2,4,4), (16,), (4,4,1), (1,16), (1,1,4,4)):
            with self.subTest(centered_shape=shape):
                class FP:
                    def _pose_last_from_numpy(self, value):self.pose_last=np.asarray(value).copy()
                    def track_one(self, *args, **kwargs):
                        self.pose_last=np.zeros(shape,dtype=np.float32)
                        return np.eye(4,dtype=np.float32)
                records=[]
                callback=entry.measured_refiner(FP(), np.zeros((2,2,3),np.uint8), np.ones((2,2),np.float32),np.eye(3),
                    centered_from_original=np.eye(4),frame_id=1,records=records)
                with self.assertRaisesRegex(ValueError,'centered return'):callback(np.eye(4))
                self.assertEqual(records[0]['actual_centered_return'].shape, shape)
                self.assertNotIn('controller_centered_return',records[0])
                self.assertFalse(records[0]['returned'])


if __name__ == '__main__':
    unittest.main()
