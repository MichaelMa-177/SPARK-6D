import unittest
import numpy as np

from latent_pose.generated_refine import centered_seed


class GeneratedRefineTests(unittest.TestCase):
    def test_centering_preserves_original_mesh_projection(self):
        camera_from_memory = np.array([[0., -1., 0., .2], [1., 0., 0., -.1],
                                       [0., 0., 1., 1.2], [0., 0., 0., 1.]])
        centering = np.eye(4)
        centering[:3, 3] = [-.03, .02, -.01]
        original_points = np.array([[.1, -.1, .04, 1.], [-.02, .05, .1, 1.]])
        seed = centered_seed(camera_from_memory, centering)
        np.testing.assert_allclose((seed @ centering @ original_points.T).T,
                                   (camera_from_memory @ original_points.T).T, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
