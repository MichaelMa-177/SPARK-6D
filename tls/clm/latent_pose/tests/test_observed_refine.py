import unittest

import cv2
import numpy as np

from latent_pose.observed_refine import normal_map, point_plane_step, projective_refine


class ObservedRefineTests(unittest.TestCase):
    def test_normals_face_camera_and_reject_depth_edges(self):
        yy, xx = np.mgrid[:20, :20]
        xyz = np.stack((xx*.001, yy*.001, np.ones_like(xx)), -1)
        xyz[:, 10:, 2] += .1
        normals, valid = normal_map(xyz, np.ones((20, 20), bool))
        self.assertFalse(valid[:, 9:11].any())
        np.testing.assert_allclose(normals[valid][:, 2], -1)

    def test_metric_step_recovers_rigid_perturbation(self):
        rng = np.random.default_rng(10)
        source = rng.normal(0, .07, (2000, 3))
        normals = rng.normal(size=source.shape)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        pivot = np.array([.1, -.03, 1.])
        rotation = np.array([.003, -.004, .005])
        shift = np.array([.0004, -.0005, .0002])
        points = source+pivot
        targets = source @ cv2.Rodrigues(rotation)[0].T + pivot+shift
        update, rank = point_plane_step(points, targets, normals, pivot)
        self.assertEqual(rank, 6)
        np.testing.assert_allclose(update[:3], rotation, atol=2e-5)
        np.testing.assert_allclose(update[3:], shift, atol=2e-6)

    def test_absent_overlap_keeps_initial_pose(self):
        initial = np.eye(4)
        initial[2, 3] = 1
        pose, report = projective_refine(
            np.zeros((40, 3)), np.tile([0., 0., -1.], (40, 1)),
            np.zeros((20, 20, 3)), np.zeros((20, 20, 3)), np.zeros((20, 20), bool),
            np.eye(3), initial)
        self.assertFalse(report["accepted"])
        np.testing.assert_array_equal(pose, initial)


if __name__ == "__main__":
    unittest.main()
