import unittest

import numpy as np

from latent_pose.one2any_evaluate import crop_resize, reference_condition
from latent_pose.rgbd import measured_geometry, reference_crop_box


class One2AnyAdapterTests(unittest.TestCase):
    def test_reference_nocs_inverts_to_metric_camera_points(self):
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        depth = np.full((32, 32), 1000, dtype=np.uint16)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 6:26] = 255
        k = np.array([[100., 0, 16], [0, 100., 16], [0, 0, 1]])
        condition, center, scale = reference_condition(rgb, depth, mask, k, .001)
        xyz = measured_geometry(rgb, depth, k, .001)[1]
        box = reference_crop_box(mask > 0)
        expected = crop_resize(xyz, box)
        selected = condition[6] > .5
        recovered = (condition[3:6].transpose(1, 2, 0)-.5)*scale+center
        np.testing.assert_allclose(recovered[selected], expected[selected], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
