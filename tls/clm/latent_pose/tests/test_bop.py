import unittest

import numpy as np
import torch

from latent_pose.bop import annotation_pose, cad_from_memory
from latent_pose.experiment import errors
from latent_pose.geometry import robust_rigid, transform_points, weighted_rigid


class BOPGaugeTests(unittest.TestCase):
    def test_fixed_gauge_matches_two_camera_projection(self):
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        annotation = dict(cam_R_m2c=rotation.ravel().tolist(), cam_t_m2c=[100., 50., 800.])
        ref = annotation_pose(annotation)
        origin = np.array([.13, .08, .82])
        cad_from_mem = cad_from_memory(ref, origin)
        point_cad = np.array([.02, -.01, .03, 1.])
        point_mem = np.linalg.inv(cad_from_mem) @ point_cad
        np.testing.assert_allclose(point_mem[:3], (ref @ point_cad)[:3] - origin, atol=1e-8)
        query = annotation_pose(dict(cam_R_m2c=np.eye(3).ravel(), cam_t_m2c=[-20, 10, 900]))
        np.testing.assert_allclose(query @ cad_from_mem @ point_mem, query @ point_cad, atol=1e-8)
        reference_from_mem = ref @ cad_from_mem
        np.testing.assert_allclose(reference_from_mem[:3, :3], np.eye(3), atol=1e-8)
        np.testing.assert_allclose(reference_from_mem[:3, 3], origin, atol=1e-8)

    def test_reflection_rejected_and_millimetres_converted(self):
        with self.assertRaisesRegex(ValueError, "reflection"):
            annotation_pose(dict(cam_R_m2c=np.diag([-1., 1, 1]).ravel(), cam_t_m2c=[0, 0, 500]))
        pose = annotation_pose(dict(cam_R_m2c=np.eye(3).ravel(), cam_t_m2c=[10, 20, 500]))
        np.testing.assert_allclose(pose[:3, 3], [.01, .02, .5])

    def test_metrics_handle_symmetry_and_invalid_predictions(self):
        truth = torch.eye(4)[None].repeat(2, 1, 1)
        symmetry = torch.diag(torch.tensor([-1., -1, 1, 1]))
        prediction = symmetry[None].repeat(2, 1, 1)
        sym = torch.stack((torch.eye(4), symmetry))[None].repeat(2, 1, 1, 1)
        degrees, mm, joint = errors(prediction, torch.tensor([True, False]), truth, sym)
        torch.testing.assert_close(degrees, torch.zeros(2))
        torch.testing.assert_close(mm, torch.zeros(2))
        self.assertEqual(joint.tolist(), [True, False])

    def test_robust_fit_rejects_positive_weight_outliers(self):
        torch.manual_seed(11)
        source = torch.randn(1, 120, 3) * .06
        pose = torch.eye(4)[None]
        pose[:, :3, 3] = torch.tensor([.04, -.02, .8])
        target = transform_points(source, pose)
        target[:, :25] += torch.randn(1, 25, 3) * .12
        weights = torch.ones(1, 120)
        baseline = weighted_rigid(source, target, weights)
        robust = robust_rigid(source, target, weights)
        self.assertTrue(robust["pose_valid"].all())
        self.assertEqual(int(robust["inlier_count"][0]), 95)
        torch.testing.assert_close(robust["pose"], pose, atol=2e-5, rtol=1e-5)
        self.assertGreater(float((baseline["pose"] - pose).abs().max()), .01)
        rejected = robust_rigid(source, target, torch.zeros_like(weights))
        self.assertFalse(rejected["pose_valid"].any())


if __name__ == "__main__":
    unittest.main()
