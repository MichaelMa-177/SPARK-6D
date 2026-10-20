import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from latent_pose.cli import cache, infer, load_model, train
from latent_pose.config import ModelConfig
from latent_pose.data import PairDataset, read_packet
from latent_pose.evaluate import pose_errors
from latent_pose.evaluate import main as evaluate_main
from latent_pose.export_reference import sam3d_to_camera
from latent_pose.geometry import camera_to_memory, transform_points, weighted_rigid
from latent_pose.losses import coordinate_loss
from latent_pose.model import LatentPoseModel


def fixture(batch=1):
    torch.manual_seed(42)
    config = ModelConfig(image_size=32, latent_side=4, memory_side=2,
                         width=32, heads=4, observed_tokens=16, decoder_layers=2)
    y, x = torch.meshgrid(torch.linspace(-.08, .08, 32), torch.linspace(-.08, .08, 32), indexing="ij")
    xyz = torch.stack((x, y, .8 + x*y)).unsqueeze(0).repeat(batch, 1, 1, 1)
    rgb = torch.rand(batch, 3, 32, 32)
    mask = ((x*x+y*y) < .005)[None, None].repeat(batch, 1, 1, 1).float()
    origin = torch.tensor([[0., 0., .8]]).repeat(batch, 1)
    ref = dict(rgb=rgb, xyz=xyz, valid=torch.ones_like(mask), mask=mask,
               origin_m=origin, scale_m=torch.full((batch, 1), .16),
               shape_latent=torch.randn(batch, 64, 8),
               latent_xyz_m=torch.rand(batch, 64, 3) * .16 - .08 + origin[:, None])
    angle = torch.tensor(.4)
    rotation = torch.tensor([[angle.cos(), -angle.sin(), 0],
                             [angle.sin(), angle.cos(), 0], [0, 0, 1]])
    pose = torch.eye(4)[None].repeat(batch, 1, 1)
    pose[:, :3, :3] = rotation
    pose[:, :3, 3] = torch.tensor([.03, -.02, .9])
    points = xyz.flatten(2).transpose(1, 2) - origin[:, None]
    query_xyz = transform_points(points, pose).transpose(1, 2).reshape_as(xyz)
    query = dict(rgb=rgb.clone(), xyz=query_xyz, valid=torch.ones_like(mask))
    return config, ref, query, pose


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.source = torch.randn(2, 40, 3) * .1
        self.pose = fixture(2)[3]
        self.target = transform_points(self.source, self.pose)

    def test_exact_rigid_and_inverse(self):
        solved = weighted_rigid(self.source, self.target, torch.ones(2, 40))
        self.assertTrue(solved["pose_valid"].all())
        torch.testing.assert_close(solved["pose"], self.pose, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(camera_to_memory(self.target, self.pose, torch.ones(2, 1)),
                                   self.source, atol=1e-6, rtol=1e-5)

    def test_zero_weight_outliers_and_nan(self):
        target = self.target.clone()
        target[:, :4] = float("nan")
        target[:, 4:9] += 10
        weights = torch.ones(2, 40)
        weights[:, :9] = 0
        solved = weighted_rigid(self.source, target, weights)
        torch.testing.assert_close(solved["pose"], self.pose, atol=2e-6, rtol=1e-5)

    def test_empty_and_collinear_rejected(self):
        result = weighted_rigid(self.source, self.target, torch.zeros(2, 40))
        self.assertFalse(result["pose_valid"].any())
        self.assertTrue(torch.isinf(result["residual_m"]).all())
        line = torch.zeros_like(self.source)
        line[:, :, 0] = torch.linspace(-1, 1, 40)
        result = weighted_rigid(line, transform_points(line, self.pose), torch.ones(2, 40))
        self.assertFalse(result["pose_valid"].any())

    def test_planar_surface_accepted(self):
        points = self.source.clone()
        points[:, :, 2] = 0
        result = weighted_rigid(points, transform_points(points, self.pose), torch.ones(2, 40))
        self.assertTrue(result["pose_valid"].all())
        torch.testing.assert_close(result["pose"], self.pose, atol=2e-6, rtol=1e-5)

    def test_permutation_and_weight_scale_invariant(self):
        perm = torch.randperm(40)
        a = weighted_rigid(self.source, self.target, torch.ones(2, 40))
        b = weighted_rigid(self.source[:, perm], self.target[:, perm], torch.full((2, 40), 7.))
        torch.testing.assert_close(a["pose"], b["pose"], atol=2e-6, rtol=1e-5)
        self.assertTrue((torch.det(b["pose"][:, :3, :3]) > .999).all())

    def test_layout_row_vector_convention(self):
        q = np.array([np.sqrt(.5), 0, 0, np.sqrt(.5)])
        actual = sam3d_to_camera(np.array([[1., 0., 0.]]), np.ones(3), q, np.zeros(3))
        np.testing.assert_allclose(actual, [[0., 1., 0.]], atol=1e-6)

    def test_symmetry_scoring(self):
        truth = np.eye(4)
        symmetry = np.diag([-1., -1., 1., 1.])
        result = pose_errors(symmetry, truth, np.stack((truth, symmetry)))
        self.assertTrue(result["pass_10mm_5deg"])


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.config, self.ref, self.query, self.pose = fixture()
        self.model = LatentPoseModel(self.config)

    def test_forward_shapes_and_memory_provenance(self):
        memory = self.model.build_memory(self.ref)
        self.assertEqual(memory.tokens.shape, (1, 24, 32))
        self.assertEqual(int((memory.kinds == 0).sum()), 16)
        self.assertEqual(int((memory.kinds == 1).sum()), 8)
        result = self.model.decode(self.query, memory)
        self.assertEqual(result["coordinates"].shape, (1, 64, 3))
        self.assertTrue(torch.isfinite(result["coordinates"]).all())
        self.assertEqual(result["pose"].shape, (1, 4, 4))

    def test_backward_reaches_latent_and_image_adapters(self):
        self.model.train()
        result = self.model(self.ref, self.query)
        losses = coordinate_loss(result, self.pose, self.ref["scale_m"], self.ref["mask"])
        losses["loss"].backward()
        for module in (self.model.shape_adapter[0], self.model.encoder.rgb[0][0],
                       self.model.coordinate_head[-1], self.model.uncertainty_head):
            self.assertTrue(torch.isfinite(module.weight.grad).all())
            self.assertGreater(float(module.weight.grad.abs().sum()), 0)

    def test_cached_equals_uncached_and_query_order_independent(self):
        self.model.eval()
        with torch.no_grad():
            memory = self.model.build_memory(self.ref)
            a = self.model.decode(self.query, memory)["coordinates"]
            self.model.cache_memory(memory)
            b = self.model.decode(self.query, memory)["coordinates"]
            other = {**self.query, "rgb": 1 - self.query["rgb"]}
            self.model.decode(other, memory)
            c = self.model.decode(self.query, memory)["coordinates"]
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(b, c)

    def test_memory_changes_with_latent(self):
        self.model.eval()
        with torch.no_grad():
            a = self.model.build_memory(self.ref)
            changed = {**self.ref, "shape_latent": -self.ref["shape_latent"]}
            b = self.model.build_memory(changed)
        self.assertGreater(float((a.tokens - b.tokens).abs().max()), .01)
        torch.testing.assert_close(a.tokens[:, :16], b.tokens[:, :16])

    def test_geometry_only_ignores_latent_values(self):
        config = ModelConfig(**{**self.config.to_dict(), "memory_mode": "geometry_only"})
        model = LatentPoseModel(config).eval()
        with torch.no_grad():
            a = model.build_memory(self.ref)
            b = model.build_memory({**self.ref, "shape_latent": -self.ref["shape_latent"]})
        torch.testing.assert_close(a.tokens, b.tokens)

    def test_observed_only_does_not_require_shape_keys(self):
        config = ModelConfig(**{**self.config.to_dict(), "memory_mode": "observed_only"})
        ref = {k: v for k, v in self.ref.items() if k not in ("shape_latent", "latent_xyz_m")}
        memory = LatentPoseModel(config).build_memory(ref)
        self.assertEqual(memory.tokens.shape[1], 16)

    def test_invalid_query_depth_produces_explicit_failure(self):
        query = {**self.query, "xyz": torch.full_like(self.query["xyz"], float("nan"))}
        result = self.model(self.ref, query, solve=True)
        self.assertTrue(torch.isfinite(result["coordinates"]).all())
        self.assertFalse(result["pose_valid"].any())
        self.assertEqual(float(result["weights"].detach().sum()), 0)

    def test_empty_reference_and_invalid_scale_rejected(self):
        with self.assertRaises(ValueError):
            self.model.build_memory({**self.ref, "mask": torch.zeros_like(self.ref["mask"])})
        with self.assertRaises(ValueError):
            self.model.build_memory({**self.ref, "scale_m": torch.zeros(1, 1)})

    def test_cache_guard_and_bad_latent_shape(self):
        with self.assertRaises(ValueError):
            self.model.build_memory(self.ref, cache=True)
        with self.assertRaises(ValueError):
            self.model.build_memory({**self.ref, "shape_latent": self.ref["shape_latent"][:, :3]})

    def test_symmetry_loss_uses_object_level_candidate(self):
        result = self.model(self.ref, self.query)
        truth = camera_to_memory(result["query_xyz_m"], self.pose, self.ref["scale_m"])
        symmetry = torch.eye(4)
        symmetry[0, 0] = symmetry[1, 1] = -1
        result["coordinates"] = truth @ symmetry[:3, :3].T
        syms = torch.stack((torch.eye(4), symmetry))[None]
        losses = coordinate_loss(result, self.pose, self.ref["scale_m"], self.ref["mask"], syms)
        self.assertLess(float(losses["coordinate"].detach()), 1e-7)


class PipelineTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_default_cuda_forward_backward(self):
        _, ref, query, pose = fixture()
        for item in (ref, query):
            for key in ("rgb", "xyz", "valid", "mask"):
                if key in item:
                    item[key] = torch.nn.functional.interpolate(item[key], (128, 128), mode="nearest")
        ref["shape_latent"] = torch.randn(1, 4096, 8)
        axis = (torch.arange(16) + .5) / 16 - .5
        grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(1, -1, 3)
        ref["latent_xyz_m"] = grid * ref["scale_m"][:, None] + ref["origin_m"][:, None]
        ref, query = [{k: v.cuda() for k, v in item.items()} for item in (ref, query)]
        model = LatentPoseModel().cuda().train()
        result = model(ref, query)
        loss = coordinate_loss(result, pose.cuda(), ref["scale_m"], ref["mask"])["loss"]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        gradient = model.shape_adapter[0].weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0)

    def test_train_cache_infer_and_label_isolation(self):
        config, ref, query, pose = fixture()
        with tempfile.TemporaryDirectory(prefix="latent-pose-test-") as name:
            root = Path(name)
            meta = {"schema_version": np.int64(1), "units": np.str_("metres")}
            np.savez(root / "reference.npz", **meta, **{k: v[0].numpy() for k, v in ref.items()})
            np.savez(root / "query.npz", **meta, **{k: v[0].numpy() for k, v in query.items()},
                     pose=pose[0].numpy(), target_mask=ref["mask"][0].numpy())
            row = {"reference": "reference.npz", "query": "query.npz", "object_id": "synthetic-test"}
            manifest = {"schema_version": 1, "split": "train", "samples": [row, row]}
            (root / "train.json").write_text(json.dumps(manifest))
            (root / "config.json").write_text(json.dumps(config.to_dict()))
            args = argparse.Namespace(seed=0, config=str(root/"config.json"), manifest=str(root/"train.json"),
                                      validation=None, output=str(root/"training"), device="cpu",
                                      lr=1e-3, batch_size=2, epochs=2)
            train(args)
            checkpoint = root / "training/last.pt"
            model, saved = load_model(checkpoint, "cpu")
            self.assertEqual(saved["training_steps"], 2)
            self.assertIn("synthetic-test", saved["training_object_ids"])
            cache(argparse.Namespace(checkpoint=checkpoint, device="cpu", allow_untrained=False,
                                     reference=root/"reference.npz", output=str(root/"memory.pt")))
            infer(argparse.Namespace(checkpoint=checkpoint, memory=root/"memory.pt", device="cpu",
                                     allow_untrained=False, query=[root/"query.npz"], output=str(root/"predictions")))
            report = json.loads((root/"predictions/report.json").read_text())
            self.assertFalse(report["GT_read"])
            self.assertFalse(report["previous_pose_used"])
            evaluation = root / "evaluation"
            (root/"development.json").write_text(json.dumps({**manifest, "split": "development"}))
            def guarded_read(path, reference=False, labels=False):
                if labels:
                    self.assertTrue((evaluation/"predictions.npz").is_file())
                return read_packet(path, reference=reference, labels=labels)
            with patch("sys.argv", ["evaluate", "--manifest", str(root/"development.json"),
                                   "--checkpoint", str(checkpoint), "--output", str(evaluation)]):
                with patch("latent_pose.evaluate.read_packet", side_effect=guarded_read):
                    evaluate_main()
            scoring = json.loads((evaluation/"report.json").read_text())
            self.assertTrue(scoring["GT_loaded_after_predictions"])
            self.assertEqual(scoring["samples"], 2)
            (root/"validation.json").write_text(json.dumps({**manifest, "split": "validation"}))
            with self.assertRaisesRegex(ValueError, "overlap"):
                train(argparse.Namespace(**{**vars(args), "validation": str(root/"validation.json"),
                                            "output": str(root/"overlap")}))
            with self.assertRaises(FileExistsError):
                train(args)
            # GT arrays are inaccessible/invalid, while unlabelled reads still work.
            np.savez(root/"bad_labels.npz", **meta, **{k: v[0].numpy() for k, v in query.items()},
                     pose=np.full((4, 4), np.nan), target_mask=ref["mask"][0].numpy())
            read_packet(root/"bad_labels.npz")
            with self.assertRaises(ValueError):
                read_packet(root/"bad_labels.npz", labels=True)
            with self.assertRaises(ValueError):
                PairDataset(root/"train.json", expected_split="validation")
            saved["training_steps"] = 0
            torch.save(saved, root/"untrained.pt")
            with self.assertRaises(ValueError):
                load_model(root/"untrained.pt", "cpu")
            with self.assertRaises(ValueError):
                infer(argparse.Namespace(checkpoint=root/"untrained.pt", memory=root/"memory.pt",
                                         device="cpu", allow_untrained=True, query=[root/"query.npz"],
                                         output=str(root/"bad_predictions")))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
