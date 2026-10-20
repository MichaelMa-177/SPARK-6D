import dataclasses
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from latent_pose.data import read_packet
from latent_pose.features import geometry_sample_grid
from latent_pose.model import LatentPoseModel
from latent_pose.bop_episodes import symmetry_matrices
from latent_pose.tests.test_framework import fixture


class FeatureTests(unittest.TestCase):
    def test_grid_aligns_with_two_nearest_resizes(self):
        for w, h in ((283, 321), (640, 480), (37, 45)):
            y, x = np.indices((h, w), dtype=np.float32)
            expected = cv2.resize(np.stack((x, y), -1), (128, 128),
                                  interpolation=cv2.INTER_NEAREST)[::4, ::4]
            grid = geometry_sample_grid(w, h, 128, "cpu")[0].numpy()
            actual = (grid + 1) * np.array([w, h]) / 2 - .5
            np.testing.assert_allclose(actual, expected, atol=4e-5)

    def test_feature_adapter_gradients_and_required_input(self):
        config, ref, query, _ = fixture()
        config = dataclasses.replace(config, foundation_feature_dim=12)
        model = LatentPoseModel(config)
        with self.assertRaisesRegex(ValueError, "foundation_features"):
            model(ref, query)
        ref["foundation_features"] = torch.randn(1, 64, 12)
        query["foundation_features"] = torch.randn(1, 64, 12)
        model(ref, query)["coordinates"].square().mean().backward()
        grad = model.encoder.foundation[1].weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0)

    def test_observed_reference_file_and_inference_label_isolation(self):
        _, ref, query, _ = fixture()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"ref.npz"
            values = {k: v[0].numpy() for k, v in ref.items()
                      if k not in ("shape_latent", "latent_xyz_m")}
            # Deliberately unreadable object-typed labels must never be opened.
            np.savez(path, schema_version=1, units="metres",
                     pose=np.array({"forbidden": True}, dtype=object), **values)
            loaded = read_packet(path, reference=True)
            self.assertNotIn("pose", loaded)
            self.assertNotIn("shape_latent", loaded)
            with self.assertRaises(ValueError):
                read_packet(path, labels=True)

    def test_coordinate_readout_follows_memory_gauge(self):
        config, ref, query, _ = fixture()
        config = dataclasses.replace(config, foundation_feature_dim=12,
                                     coordinate_mode="memory_residual")
        ref["foundation_features"] = torch.randn(1, 64, 12)
        query["foundation_features"] = torch.randn(1, 64, 12)
        model = LatentPoseModel(config).eval()
        with torch.no_grad():
            memory = model.build_memory(ref, cache=True)
            before = model.decode(query, memory, solve=False)["coordinates"]
            delta = torch.tensor([.2, -.1, .05])
            memory.positions += delta
            after = model.decode(query, memory, solve=False)["coordinates"]
        torch.testing.assert_close(after, before + delta)

    def test_symmetries_convert_units_and_origin(self):
        fixed = np.eye(4)
        fixed[:3, 3] = [.1, .02, 0]
        transform = np.diag([-1., -1., 1., 1.])
        transform[0, 3] = 20  # BOP millimetres
        matrices = symmetry_matrices({"symmetries_discrete": [transform.ravel().tolist()]}, fixed)
        transform[0, 3] = .02
        point = np.array([.03, .01, 0, 1])
        np.testing.assert_allclose(fixed @ matrices[1] @ point, transform @ fixed @ point, atol=1e-7)
        continuous = symmetry_matrices({"symmetries_continuous": [
            {"axis": [0, 1, 0], "offset": [10, 0, 0]}]}, fixed)
        self.assertEqual(len(continuous), 72)
        np.testing.assert_allclose(continuous[0], np.eye(4), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
