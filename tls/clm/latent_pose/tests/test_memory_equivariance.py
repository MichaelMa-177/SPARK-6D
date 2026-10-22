from dataclasses import replace
import unittest

import torch

from latent_pose.model import LatentPoseModel
from latent_pose.tests.test_framework import fixture


class MemoryEquivarianceTests(unittest.TestCase):
    def test_exact_memory_gauge_rotation_and_finite_empty_generated_support(self):
        config, reference, query, _ = fixture()
        config = replace(config, foundation_feature_dim=16, coordinate_mode="equivariant_memory",
                         membership_mode="appearance_prior", shape_geometry_channels=2,
                         contextual_features=True)
        reference["foundation_features"] = torch.randn(1, 64, 16)
        query["foundation_features"] = torch.randn(1, 64, 16)
        reference["shape_geometry"] = torch.ones(1, 64, 2)
        model = LatentPoseModel(config).eval()
        rotation = torch.linalg.qr(torch.randn(1, 3, 3)).Q
        rotation[:, :, -1] *= torch.det(rotation)[:, None]
        with torch.no_grad():
            ordinary = model(reference, query)
            rotated = model({**reference, "memory_basis": rotation}, query)
            torch.testing.assert_close(rotated["coordinates"],
                                       ordinary["coordinates"] @ rotation.transpose(1, 2),
                                       atol=1e-6, rtol=1e-5)
            for key in ("membership_logits", "log_variance", "generated_coordinate_fraction"):
                torch.testing.assert_close(ordinary[key], rotated[key], rtol=0, atol=0)
            empty = model({**reference, "shape_geometry": torch.zeros_like(reference["shape_geometry"])}, query)
            self.assertTrue(torch.isfinite(empty["coordinates"]).all())
            self.assertEqual(float(empty["generated_coordinate_fraction"].abs().max()), 0)
            memory = model.build_memory(reference, cache=True)
            cached = model.decode(query, memory, solve=False)
            torch.testing.assert_close(cached["coordinates"], ordinary["coordinates"])


if __name__ == "__main__":
    unittest.main()
