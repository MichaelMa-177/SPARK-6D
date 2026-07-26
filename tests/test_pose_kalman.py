import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from pose_kalman import SE3KalmanFilter, se3_exp, se3_log


class SE3KalmanFilterTest(unittest.TestCase):

  def test_exp_log_round_trip(self):
    tangent = np.array([0.03, -0.01, 0.02, 0.2, -0.1, 0.05])
    np.testing.assert_allclose(
        se3_log(se3_exp(tangent)), tangent, atol=1e-10)

  def test_rotation_wrap_is_continuous(self):
    first = np.eye(4)
    first[:3, :3] = Rotation.from_euler('z', 179, degrees=True).as_matrix()
    second = np.eye(4)
    second[:3, :3] = Rotation.from_euler('z', -179, degrees=True).as_matrix()
    innovation = se3_log(np.linalg.inv(first) @ second)
    self.assertAlmostEqual(np.degrees(np.linalg.norm(innovation[3:])), 2.0, places=6)

  def test_constant_velocity_prediction(self):
    velocity = np.array([0.002, -0.001, 0.0, 0.0, 0.0, np.deg2rad(0.5)])
    filter_ = SE3KalmanFilter(object_diameter=0.2)
    truth = np.eye(4)
    filter_.initialize(truth)
    for frame in range(1, 21):
      truth = truth @ se3_exp(velocity)
      prediction = filter_.predict()
      if frame % 2:
        filter_.update(truth)
      else:
        translation_error = np.linalg.norm(prediction[:3, 3] - truth[:3, 3])
        rotation_error = np.linalg.norm(
            se3_log(np.linalg.inv(prediction) @ truth)[3:])
        self.assertLess(translation_error, 0.002)
        self.assertLess(rotation_error, np.deg2rad(0.5))

  def test_covariance_remains_symmetric(self):
    filter_ = SE3KalmanFilter(object_diameter=0.1)
    filter_.initialize(np.eye(4))
    for _ in range(10):
      filter_.predict()
      filter_.update(np.eye(4))
    np.testing.assert_allclose(
        filter_.covariance, filter_.covariance.T, atol=1e-14)
    self.assertGreaterEqual(np.linalg.eigvalsh(filter_.covariance).min(), -1e-14)


if __name__ == '__main__':
  unittest.main()
