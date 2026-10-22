import unittest

import numpy as np

from latent_pose.video_evaluate import trajectory_metrics


class VideoMetricsTests(unittest.TestCase):
    def test_exact_motion_and_frozen_pose(self):
        truth = np.tile(np.eye(4), (8, 1, 1))
        truth[:, 0, 3] = np.arange(8)*.02
        valid, ids = np.ones(8, dtype=bool), np.arange(8)
        exact = trajectory_metrics(truth, valid, truth, ids)
        self.assertEqual(exact["pass_10mm_5deg_all_frames"], 1)
        self.assertEqual(exact["origin_displacement_error_mm"]["median"], 0)
        held = trajectory_metrics(np.tile(truth[0], (8, 1, 1)), valid, truth, ids)
        self.assertLess(held["pass_10mm_5deg_all_frames"], .2)
        self.assertAlmostEqual(held["origin_displacement_error_mm"]["median"], 20)

    def test_invalid_and_nonconsecutive_pairs_are_counted(self):
        truth = np.tile(np.eye(4), (5, 1, 1))
        report = trajectory_metrics(truth, np.array([True, False, True, True, True]),
                                    truth, np.array([0, 1, 2, 4, 5]))
        self.assertEqual(report["valid_fraction"], .8)
        self.assertEqual(report["pass_10mm_5deg_all_frames"], .8)
        self.assertEqual(report["consecutive_valid_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
