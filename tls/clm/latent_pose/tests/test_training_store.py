import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from latent_pose.data import PairDataset
from latent_pose.experiment import IndexedPairStore, preload, subset
from latent_pose.ov9d_episodes import sanitize_depth
from latent_pose.tests.test_framework import fixture


class TrainingStoreTests(unittest.TestCase):
    def test_unique_cache_matches_dense_batches_without_sharing_labels(self):
        _, reference, query, pose = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {"schema_version": 1, "units": "metres"}
            np.savez(root/"ref.npz", **metadata, **{k: v[0].numpy() for k, v in reference.items()})
            for i in range(2):
                p = pose[0].numpy().copy()
                p[0, 3] += i*.1
                np.savez(root/f"q{i}.npz", **metadata, **{k: v[0].numpy() for k, v in query.items()},
                         pose=p, target_mask=reference["mask"][0].numpy())
            rows = [dict(reference="ref.npz", query=f"q{i}.npz", object_id="fixture", rgb_sha256="same",
                         box_xyxy=[0, 0, 32, 32]) for i in range(2)]
            (root/"train.json").write_text(json.dumps(dict(schema_version=1, split="train", samples=rows)))
            dataset = PairDataset(root/"train.json")
            compact, dense = IndexedPairStore(dataset, "cpu"), preload(dataset, "cpu")
            self.assertEqual(len(compact.references), 1)
            self.assertEqual(len(compact.queries), 1)
            for a, b in zip(subset(compact, torch.tensor([1, 0])), subset(dense, torch.tensor([1, 0]))):
                for key in a:
                    torch.testing.assert_close(a[key], b[key])
            # A mislabeled cache key must not silently replace inference inputs.
            changed = query["rgb"][0].numpy().copy()
            changed[0, 0, 0] = .999
            np.savez(root/"q1.npz", **metadata, rgb=changed, xyz=query["xyz"][0].numpy(),
                     valid=query["valid"][0].numpy(), pose=pose[0].numpy(),
                     target_mask=reference["mask"][0].numpy())
            with self.assertRaisesRegex(ValueError, "different inference tensors"):
                IndexedPairStore(dataset, "cpu")

    def test_integer_saturation_is_not_valid_depth(self):
        depth = np.array([[0, 65535], [1000, 2000]], dtype=np.uint16)
        clean, fraction = sanitize_depth(depth, np.ones((2, 2), dtype=bool))
        self.assertEqual(fraction, .5)
        np.testing.assert_array_equal(clean, [[0, 0], [1000, 2000]])


if __name__ == "__main__":
    unittest.main()
