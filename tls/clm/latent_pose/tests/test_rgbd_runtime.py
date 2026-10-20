import unittest
from dataclasses import replace
from unittest.mock import patch

import cv2
import numpy as np
import torch

from latent_pose.config import ModelConfig
from latent_pose.model import LatentPoseModel
from latent_pose.rgbd import measured_geometry, prepare_query, prepare_reference
from latent_pose.runtime import PoseRuntime


def raw_frame():
    rng = np.random.default_rng(3)
    # Piecewise constant pixels keep area-resize rounding identical at both scales.
    colors = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    rgb = np.repeat(np.repeat(colors, 2, axis=0), 2, axis=1)
    depth = np.full((64, 64), 800, np.uint16)
    mask = np.zeros((64, 64), np.uint8)
    mask[8:40, 16:48] = 255
    k = np.array([[80., 0, 32], [0, 80., 24], [0, 0, 1]])
    config = ModelConfig(image_size=32, width=32, observed_tokens=16,
                         latent_side=4, memory_side=2, memory_mode="observed_only")
    return rgb, depth, mask, k, config


class RGBDTests(unittest.TestCase):
    def test_metric_calibration_and_sensor_unit_equivalence(self):
        rgb, depth, _, k, config = raw_frame()
        _, xyz, valid = measured_geometry(rgb, depth, k, .001)
        np.testing.assert_allclose(xyz[24, 32], [0, 0, .8], atol=1e-7)
        np.testing.assert_allclose(xyz[24, 42], [.1, 0, .8], atol=1e-7)
        self.assertTrue(valid.all())
        a = prepare_query(rgb, depth, k, .001, config)
        b = prepare_query(rgb, depth.astype(np.float32) * .001, k, 1., config)
        np.testing.assert_allclose(a["xyz"], b["xyz"], atol=1e-7)

    def test_crop_preserves_original_camera_coordinates(self):
        rgb, depth, _, k, config = raw_frame()
        packet = prepare_query(rgb, depth, k, .001, config, box=(16, 8, 48, 40))
        # This crop is already 32x32: no resizing changes the sampled point.
        np.testing.assert_allclose(packet["xyz"][:, 16, 16], [0, 0, .8], atol=1e-7)
        np.testing.assert_allclose(packet["xyz"][:, 0, 0], [-.16, -.16, .8], atol=1e-7)

    def test_invalid_depth_reference_and_calibration(self):
        rgb, depth, mask, k, config = raw_frame()
        bad = depth.astype(np.float32)
        bad[0, :4] = [0, -1, np.nan, np.inf]
        _, xyz, valid = measured_geometry(rgb, bad, k, .001)
        self.assertFalse(valid[0, :4].any())
        self.assertTrue(np.isfinite(xyz).all())
        with self.assertRaisesRegex(ValueError, "valid masked"):
            prepare_reference(rgb, np.zeros_like(depth), mask, k, .001, config)
        with self.assertRaisesRegex(ValueError, "binary"):
            prepare_reference(rgb, depth, mask.astype(float) / 256, k, .001, config)
        skew = k.copy()
        skew[0, 1] = .1
        with self.assertRaisesRegex(ValueError, "zero-skew"):
            measured_geometry(rgb, depth, skew, .001)
        with self.assertRaisesRegex(ValueError, "registered"):
            measured_geometry(rgb, depth[:-1], k, .001)
        with self.assertRaisesRegex(ValueError, "positive"):
            measured_geometry(rgb, depth, k, 0)
        with self.assertRaisesRegex(ValueError, "inside"):
            prepare_query(rgb, depth, k, .001, config, (0, 0, 65, 48))

    def test_reference_origin_is_masked_metric_centroid(self):
        rgb, depth, mask, k, config = raw_frame()
        packet = prepare_reference(rgb, depth, mask, k, .001, config)
        _, xyz, _ = measured_geometry(rgb, depth, k, .001)
        np.testing.assert_allclose(packet["origin_m"], xyz[mask > 0].mean(0))
        self.assertAlmostEqual(float(packet["scale_m"][0]), .31, places=6)
        self.assertNotIn("shape_latent", packet)

    def test_spatial_mode_requires_aligned_latent(self):
        rgb, depth, mask, k, config = raw_frame()
        config = replace(config, memory_mode="spatial_latent")
        with self.assertRaisesRegex(ValueError, "aligned"):
            prepare_reference(rgb, depth, mask, k, .001, config)
        z = np.ones((64, 8), np.float32)
        xyz = np.zeros((64, 3), np.float32)
        packet = prepare_reference(rgb, depth, mask, k, .001, config, z, xyz)
        np.testing.assert_array_equal(packet["shape_latent"], z)
        np.testing.assert_array_equal(packet["latent_xyz_m"], xyz)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.rgb, self.depth, self.mask, self.k, self.config = raw_frame()
        self.runtime = PoseRuntime(LatentPoseModel(self.config).eval())
        self.registered = self.runtime.register_object(
            "synthetic:one", self.rgb, self.depth, self.mask, self.k, .001,
            mask_source="synthetic test rectangle",
        )

    def test_raw_runtime_computes_coarse_and_crop_features(self):
        config = replace(self.config, foundation_feature_dim=12, coordinate_mode="memory_residual",
                         membership_mode="appearance_prior")
        calls = []
        def extractor(rgb, size):
            calls.append(rgb.shape)
            pixels = cv2.resize(rgb, (size//4, size//4)).reshape(-1, 3).astype(np.float32)/255
            return np.tile(pixels, (1, 4))
        runtime = PoseRuntime(LatentPoseModel(config).eval(), extractor, retrieve_region=True)
        registered = runtime.register_object("roi", self.rgb, self.depth, self.mask,
                                              self.k, .001, mask_source="synthetic test rectangle")
        tokens = registered.memory.tokens.clone()
        video = runtime.start_video("new-camera", "roi")
        first = video.estimate(self.rgb, self.depth, self.k.tolist(), .001, 0.)
        second = runtime.start_video("another-camera", "roi").estimate(
            self.rgb, self.depth, self.k, .001, 0.)
        self.assertEqual(first["mode"], "retrieval_then_localization")
        self.assertEqual(first["query_box_xyxy"], second["query_box_xyxy"])
        self.assertIsNotNone(first["region_proposal"])
        self.assertGreaterEqual(len(calls), 5)
        torch.testing.assert_close(registered.memory.tokens, tokens)
        with patch("latent_pose.runtime.load_model", return_value=(runtime.model, {
            "foundation_features": {"weights_sha256": "required-weight-hash"}})):
            with self.assertRaisesRegex(ValueError, "weights differ"):
                PoseRuntime.from_checkpoint("unused.pt", feature_extractor=extractor)

    def test_cross_video_memory_and_localization_independence(self):
        original = self.registered.memory.tokens.clone()
        a = self.runtime.start_video("camera-a", "synthetic:one")
        b = self.runtime.start_video("camera-b", "synthetic:one")
        self.assertIs(a.registered.memory, b.registered.memory)
        # Same camera rays represented by a different resolution/calibration.
        large_rgb = np.repeat(np.repeat(self.rgb, 2, axis=0), 2, axis=1)
        large_depth = np.repeat(np.repeat(self.depth, 2, axis=0), 2, axis=1)
        large_k = self.k.copy()
        large_k[:2] *= 2
        # Integer downsampling makes the final sampled rays exactly equal.
        for key in ("rgb", "xyz", "valid"):
            np.testing.assert_allclose(
                prepare_query(self.rgb, self.depth, self.k, .001, self.config)[key],
                prepare_query(large_rgb, large_depth, large_k, .001, self.config)[key],
            )
        first = a.estimate(self.rgb, self.depth, self.k, .001, 0.)
        a.estimate(255 - self.rgb, self.depth, self.k, .001, 1.)
        other = b.estimate(large_rgb, large_depth, large_k, .001, 0.)
        again = a.estimate(self.rgb, self.depth, self.k, .001, 2.)
        self.assertTrue(first["pose_valid"])  # solver wiring only, random network
        np.testing.assert_allclose(first["pose"], again["pose"], atol=1e-6)
        np.testing.assert_allclose(first["pose"], other["pose"], atol=1e-6)
        self.assertEqual(other["frame_index"], 0)
        self.assertEqual(again["frame_index"], 2)
        torch.testing.assert_close(original, self.registered.memory.tokens, rtol=0, atol=0)

    def test_missing_depth_is_explicit_failure_and_next_frame_is_independent(self):
        video = self.runtime.start_video("test", "synthetic:one")
        missing = video.estimate(self.rgb, np.zeros_like(self.depth), self.k, .001, 10.)
        self.assertFalse(missing["pose_valid"])
        self.assertIsNone(missing["pose"])
        self.assertIsNone(missing["residual_m"])
        recovered = video.estimate(self.rgb, self.depth, self.k, .001, 11.)
        fresh = self.runtime.start_video("fresh", "synthetic:one")
        expected = fresh.estimate(self.rgb, self.depth, self.k, .001, 0.)
        self.assertTrue(recovered["pose_valid"])
        np.testing.assert_allclose(recovered["pose"], expected["pose"], atol=1e-6)

    def test_stream_validation_does_not_advance_state_on_bad_input(self):
        video = self.runtime.start_video("test", "synthetic:one")
        video.estimate(self.rgb, self.depth, self.k, .001, 1.)
        for timestamp in (0., 1., float("nan")):
            with self.assertRaisesRegex(ValueError, "timestamps"):
                video.estimate(self.rgb, self.depth, self.k, .001, timestamp)
        with self.assertRaisesRegex(ValueError, "registered"):
            video.estimate(self.rgb, self.depth[:-1], self.k, .001, 2.)
        self.assertEqual(video.frame_index, 1)
        self.assertEqual(video.last_timestamp_s, 1.)
        with self.assertRaisesRegex(ValueError, "already registered"):
            self.runtime.register_object("synthetic:one", self.rgb, self.depth, self.mask,
                                         self.k, .001, mask_source="test")
        with self.assertRaisesRegex(ValueError, "register"):
            self.runtime.start_video("test", "unknown")
        self.runtime.model.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            video.estimate(self.rgb, self.depth, self.k, .001, 2.)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
