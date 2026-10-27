"""Online publication contracts, using the actual measured session on CPU.

Point/network returns are analytic fixtures, not GPU accuracy evidence.
"""
import unittest

import numpy as np

from identity_output import OutputPolicyAdapter, numeric_snapshot
from identity_continuity import FLAGS
from test_measured_motion_session import Fixture


class OutputTests(unittest.TestCase):
    def make(self, *, sink=None, blank=False):
        adapter = OutputPolicyAdapter("cpu_epoch", sink)
        fixture = Fixture(sink=adapter.observe, blank=blank, preloss=False)
        return fixture, adapter

    def publish(self, adapter, result):
        return adapter.publish(result, result["pose_original"].astype(np.float32))

    def test_online_actual_typed_events_publish_exact_original_candidate(self):
        h, a = self.make()
        r = h.start()
        actual = r["pose_original"].astype(np.float32)
        saved = actual.copy()
        decision = a.publish(r, actual)
        self.assertTrue(decision["pose_available"])
        self.assertEqual(decision["published_pose"].dtype, np.float32)
        self.assertEqual(decision["published_pose"].tobytes(), actual.tobytes())
        for name in FLAGS:
            self.assertIs(decision[name], False)
        actual[:] = 999
        np.testing.assert_array_equal(a.last_publication["candidate_pose"], saved)
        # Nonconsecutive source clocks remain valid when input clocks are consecutive.
        for i in range(1, 4):
            row = h.step(i)
            out = self.publish(a, row)
            self.assertEqual(out["frame_id"], i)
            self.assertEqual(out["source_frame_id"], 40 + 4*i)
            np.testing.assert_array_equal(out["candidate_pose"], row["pose_original"].astype(np.float32))

    def test_no_queries_never_publishes_fallback_pose(self):
        h, a = self.make(blank=True)
        for r in (h.start(),):
            out = self.publish(a, r)
            self.assertEqual(out["status"], "initial_continuity_unestablished")
            self.assertIsNone(out["published_pose"])
        out = self.publish(a, h.step(1))
        self.assertFalse(out["pose_available"])
        self.assertIsNone(out["published_pose"])
        self.assertTrue(np.isfinite(out["candidate_pose"]).all())

    def test_missing_current_motion_does_not_reuse_previous_packet(self):
        h, a = self.make()
        self.publish(a, h.start())
        def omit(kind, packet):
            if kind != "original_motion":
                return a.observe(kind, packet)
        h.sink = omit
        out = self.publish(a, h.step(1))
        self.assertEqual(out["status"], "identity_continuity_lost")
        self.assertIsNone(out["published_pose"])
        h.sink = a.observe
        later = self.publish(a, h.step(2))
        self.assertTrue(later["identity_loss_latched"])
        self.assertIsNone(later["published_pose"])

    def test_temporary_support_loss_can_resume_without_reinitialization(self):
        h, a = self.make()
        r = h.start()
        r["diagnostics"]["evidence"]["observation_valid"] = False
        out = self.publish(a, r)
        self.assertEqual(out["status"], "temporarily_unobservable")
        self.assertFalse(out["identity_loss_latched"])
        # Identical current RGB/depth and observed points, with a new input clock.
        image = h.image(0)
        r = h.session.step(image, h.depth, h.K, lambda p: p,
                           frame_id=1, source_frame_id=44, timestamp=1.1)
        out = self.publish(a, r)
        self.assertTrue(out["pose_available"])

    def test_wrong_actual_coordinate_return_terminates_without_output(self):
        h, a = self.make()
        r = h.start()
        bad = r["pose_original"].copy(); bad[0, 3] += .02
        with self.assertRaisesRegex(ValueError, "actual original pose"):
            a.publish(r, bad)
        self.assertTrue(a.closed)
        self.assertIsNone(a.last_publication)
        with self.assertRaises(RuntimeError):
            self.publish(a, r)

    def test_duplicate_publish_is_terminal(self):
        h, a = self.make()
        r = h.start(); self.publish(a, r)
        with self.assertRaises(ValueError):
            self.publish(a, r)
        self.assertTrue(a.closed)
        self.assertIsNone(a.last_publication)

    def test_duplicate_and_stale_motion_events_are_not_accepted(self):
        for corrupt in ("duplicate", "stale"):
            h, a = self.make()
            self.publish(a, h.start())
            h.step(1)
            envelope = next(p for k, p in reversed(h.events) if k == "original_motion")
            envelope = numeric_snapshot(envelope)
            if corrupt == "stale":
                envelope["source_frame_id"] = 40
            with self.assertRaises(ValueError):
                a.observe("original_motion", envelope)
            self.assertTrue(a.closed)

    def test_nested_clock_and_registration_bank_mismatch_are_terminal(self):
        for corrupt in ("nested_clock", "unselected_bank_pose"):
            h, a = self.make()
            def change(kind, envelope):
                if kind == "original_motion":
                    envelope = numeric_snapshot(envelope)
                    motion = envelope["payload"]["result"]
                    if corrupt == "nested_clock":
                        motion["raw_observation"]["source_frame_id"] += 1
                    else:
                        motion["raw_initial_hypotheses"][0, 0, 3] += .02
                return a.observe(kind, envelope)
            h.sink = change
            r = h.start()
            with self.assertRaises(ValueError):
                self.publish(a, r)
            self.assertTrue(a.closed)

    def test_sink_failure_after_gate_advance_does_not_publish_or_allow_retry(self):
        def sink(kind, envelope):
            if kind == "output_publication":
                raise OSError("output disk full")
        h, a = self.make(sink=sink)
        r = h.start()
        with self.assertRaisesRegex(OSError, "disk full"):
            self.publish(a, r)
        self.assertTrue(a.closed)
        self.assertIsNone(a.last_publication)
        self.assertIsNone(a.failure["published_pose"])
        with self.assertRaises(RuntimeError):
            self.publish(a, r)

    def test_upstream_sink_failure_cannot_be_followed_by_publication(self):
        def sink(kind, envelope):
            if kind == "original_motion":
                raise OSError("motion evidence disk full")
        h, a = self.make(sink=sink)
        with self.assertRaises(Exception):
            h.start()
        self.assertTrue(a.closed)
        self.assertTrue(h.session.closed)
        self.assertIsNone(a.last_publication)

    def test_trust_escalation_rejected_and_close_is_terminal(self):
        h, a = self.make()
        r = h.start(); r["diagnostics"]["identity_verified"] = True
        with self.assertRaises(ValueError):
            self.publish(a, r)
        self.assertTrue(a.closed)
        h, a = self.make(); r = h.start(); a.close()
        with self.assertRaises(RuntimeError):
            self.publish(a, r)


if __name__ == "__main__":
    unittest.main()
