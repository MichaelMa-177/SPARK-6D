import unittest
import torch

from latent_pose.losses import coordinate_loss


class RigidityLossTests(unittest.TestCase):
    def test_rigidity_accepts_rigid_change_and_penalizes_shape_distortion(self):
        torch.manual_seed(4)
        xyz = torch.randn(1, 16, 3)
        rotation = torch.linalg.qr(torch.randn(3, 3)).Q
        rotation[:, -1] *= torch.det(rotation)
        coords = (xyz @ rotation + .3).requires_grad_()
        output = {"coordinates": coords, "query_xyz_m": xyz,
                  "query_valid": torch.ones(1, 16, dtype=torch.bool),
                  "log_variance": torch.zeros(1, 16), "membership_logits": torch.zeros(1, 16)}
        args = (torch.eye(4)[None], torch.ones(1, 1), torch.ones(1, 1, 4, 4))
        result = coordinate_loss(output, *args, rigidity_weight=.1)
        self.assertLess(float(result["rigidity"]), 1e-10)
        distorted = coords.detach().clone()
        distorted[..., 0] *= 1.5
        distorted.requires_grad_()
        result = coordinate_loss({**output, "coordinates": distorted}, *args, rigidity_weight=.1)
        self.assertGreater(float(result["rigidity"]), .01)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(distorted.grad).all())
        self.assertGreater(float(distorted.grad.abs().sum()), 0)

    def test_no_foreground_has_finite_zero_rigidity(self):
        xyz = torch.randn(1, 16, 3, requires_grad=True)
        output = {"coordinates": xyz, "query_xyz_m": xyz.detach(),
                  "query_valid": torch.ones(1, 16, dtype=torch.bool),
                  "log_variance": torch.zeros(1, 16),
                  "membership_logits": torch.zeros(1, 16, requires_grad=True)}
        result = coordinate_loss(output, torch.eye(4)[None], torch.ones(1, 1),
                                 torch.zeros(1, 1, 4, 4), rigidity_weight=.1)
        self.assertEqual(float(result["rigidity"]), 0)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(xyz.grad).all())


if __name__ == "__main__":
    unittest.main()
