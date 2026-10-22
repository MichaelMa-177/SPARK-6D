from dataclasses import replace
import unittest

import torch

from latent_pose.model import LatentPoseModel
from latent_pose.shape_prior import occupancy_fields
from latent_pose.tests.test_framework import fixture


class ShapeGeometryTests(unittest.TestCase):
    def test_decoded_fraction_order_and_boundary(self):
        logits = torch.full((1, 1, 8, 8, 8), -1.)
        logits[:, :, :4, :4, :4] = 1
        field = occupancy_fields(logits, 2)
        self.assertEqual(tuple(field.shape), (1, 8, 2))
        torch.testing.assert_close(field[0, :, 0], torch.tensor([1., 0, 0, 0, 0, 0, 0, 0]))
        # 4^3 cube has 2^3 interior voxels under 26-neighbour erosion.
        self.assertAlmostEqual(float(field[0, 0, 1]), 56/64)
        self.assertTrue((field[..., 1] <= field[..., 0]).all())

    def test_surface_support_mask_and_gradient(self):
        config, ref, query, _ = fixture()
        config = replace(config, shape_geometry_channels=2, memory_mode="geometry_only")
        model = LatentPoseModel(config)
        ref["shape_geometry"] = torch.zeros(1, 64, 2)
        ref["shape_geometry"][:, 0] = torch.tensor([.8, .4])
        memory = model.build_memory(ref)
        self.assertEqual(int(memory.token_valid[:, config.observed_tokens:].sum()), 1)
        result = model.decode(query, memory, solve=False)
        result["coordinates"].square().sum().backward()
        gradients = [p.grad for p in model.shape_geometry_adapter.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)

    def test_masked_tokens_cannot_affect_output_and_cache(self):
        config, ref, query, _ = fixture()
        model = LatentPoseModel(replace(config, shape_geometry_channels=2)).eval()
        ref["shape_geometry"] = torch.zeros(1, 64, 2)
        ref["shape_geometry"][:, 0] = 1
        with torch.no_grad():
            memory = model.build_memory(ref)
            first = model.decode(query, memory, solve=False)
            memory.tokens = memory.tokens.clone()
            memory.tokens[~memory.token_valid] = torch.randn_like(memory.tokens[~memory.token_valid])*1000
            changed = model.decode(query, memory, solve=False)
            model.cache_memory(memory)
            cached = model.decode(query, memory, solve=False)
        for key in ("coordinates", "membership_logits", "log_variance"):
            torch.testing.assert_close(first[key], changed[key])
            torch.testing.assert_close(changed[key], cached[key])

    def test_optional_adapter_preserves_common_initialization(self):
        config, _, _, _ = fixture()
        torch.manual_seed(12)
        plain = LatentPoseModel(config)
        torch.manual_seed(12)
        augmented = LatentPoseModel(replace(config, shape_geometry_channels=2))
        for key, value in plain.state_dict().items():
            torch.testing.assert_close(value, augmented.state_dict()[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
