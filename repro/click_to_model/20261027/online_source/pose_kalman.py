"""Lightweight error-state Kalman filtering for FoundationPose tracking.

The filter keeps the pose on SE(3) instead of filtering Euler angles.  Its
12-dimensional error state is ``[pose_error, body_velocity]`` where both
terms use translation followed by a rotation vector.  This avoids angle
wrapping and gimbal-lock discontinuities.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


def _skew(vector):
  x, y, z = np.asarray(vector, dtype=np.float64)
  return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def se3_exp(tangent):
  """Map a 6D tangent ``[translation, rotation]`` to an SE(3) matrix."""
  tangent = np.asarray(tangent, dtype=np.float64).reshape(6)
  rho, phi = tangent[:3], tangent[3:]
  theta = np.linalg.norm(phi)
  phi_x = _skew(phi)
  if theta < 1e-7:
    V = np.eye(3) + 0.5 * phi_x + phi_x @ phi_x / 6.0
  else:
    theta2 = theta * theta
    V = (np.eye(3) + (1.0 - np.cos(theta)) / theta2 * phi_x
         + (theta - np.sin(theta)) / (theta2 * theta) * (phi_x @ phi_x))

  transform = np.eye(4, dtype=np.float64)
  transform[:3, :3] = Rotation.from_rotvec(phi).as_matrix()
  transform[:3, 3] = V @ rho
  return transform


def se3_log(transform):
  """Map an SE(3) matrix to a 6D tangent ``[translation, rotation]``."""
  transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
  phi = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
  theta = np.linalg.norm(phi)
  phi_x = _skew(phi)
  if theta < 1e-7:
    V_inv = np.eye(3) - 0.5 * phi_x + phi_x @ phi_x / 12.0
  else:
    theta2 = theta * theta
    coefficient = (1.0 / theta2
                   - (1.0 + np.cos(theta))
                   / (2.0 * theta * np.sin(theta)))
    V_inv = np.eye(3) - 0.5 * phi_x + coefficient * (phi_x @ phi_x)
  return np.r_[V_inv @ transform[:3, 3], phi]


@dataclass
class KalmanUpdate:
  translation_innovation: float
  rotation_innovation: float
  mahalanobis: float


class SE3KalmanFilter:
  """Constant-body-velocity error-state Kalman filter.

  All translation values are in metres and rotation values are in radians.
  FoundationPose refinement remains the measurement source; this class only
  predicts between measurements and fuses their corrections.
  """

  def __init__(
      self,
      object_diameter,
      process_translation=None,
      process_rotation=np.deg2rad(0.35),
      measurement_translation=None,
      measurement_rotation=np.deg2rad(0.20),
  ):
    diameter = max(float(object_diameter), 1e-3)
    self.process_translation = (
        float(process_translation)
        if process_translation is not None else diameter * 0.004)
    self.process_rotation = float(process_rotation)
    self.measurement_translation = (
        float(measurement_translation)
        if measurement_translation is not None else diameter * 0.0015)
    self.measurement_rotation = float(measurement_rotation)
    self.pose = None
    self.velocity = np.zeros(6, dtype=np.float64)
    self.covariance = None

  @property
  def initialized(self):
    return self.pose is not None

  def initialize(self, pose):
    self.pose = np.asarray(pose, dtype=np.float64).reshape(4, 4).copy()
    self.velocity.fill(0.0)
    pose_std = np.r_[
        np.full(3, self.measurement_translation),
        np.full(3, self.measurement_rotation)]
    velocity_std = np.r_[
        np.full(3, self.process_translation * 4.0),
        np.full(3, self.process_rotation * 4.0)]
    self.covariance = np.diag(np.square(np.r_[pose_std, velocity_std]))
    return self.pose.copy()

  def predict(self, dt=1.0):
    if not self.initialized:
      raise RuntimeError("SE3KalmanFilter must be initialized before predict")
    dt = float(dt)
    self.pose = self.pose @ se3_exp(self.velocity * dt)

    transition = np.eye(12)
    transition[:6, 6:] = np.eye(6) * dt
    pose_noise = np.r_[
        np.full(3, self.process_translation * dt),
        np.full(3, self.process_rotation * dt)]
    velocity_noise = np.r_[
        np.full(3, self.process_translation * 0.5 * np.sqrt(dt)),
        np.full(3, self.process_rotation * 0.5 * np.sqrt(dt))]
    process_covariance = np.diag(
        np.square(np.r_[pose_noise, velocity_noise]))
    self.covariance = (
        transition @ self.covariance @ transition.T + process_covariance)
    self.covariance = 0.5 * (self.covariance + self.covariance.T)
    return self.pose.copy()

  def update(self, measurement):
    if not self.initialized:
      self.initialize(measurement)
      return KalmanUpdate(0.0, 0.0, 0.0)

    measurement = np.asarray(measurement, dtype=np.float64).reshape(4, 4)
    innovation = se3_log(np.linalg.inv(self.pose) @ measurement)
    observation = np.zeros((6, 12), dtype=np.float64)
    observation[:, :6] = np.eye(6)
    measurement_std = np.r_[
        np.full(3, self.measurement_translation),
        np.full(3, self.measurement_rotation)]
    measurement_covariance = np.diag(np.square(measurement_std))
    innovation_covariance = (
        observation @ self.covariance @ observation.T
        + measurement_covariance)
    gain = np.linalg.solve(
        innovation_covariance,
        observation @ self.covariance).T
    correction = gain @ innovation

    self.pose = self.pose @ se3_exp(correction[:6])
    self.velocity += correction[6:]
    identity = np.eye(12)
    residual = identity - gain @ observation
    # Joseph form preserves positive semi-definiteness under round-off.
    self.covariance = (
        residual @ self.covariance @ residual.T
        + gain @ measurement_covariance @ gain.T)
    self.covariance = 0.5 * (self.covariance + self.covariance.T)
    mahalanobis = float(
        innovation @ np.linalg.solve(innovation_covariance, innovation))
    return KalmanUpdate(
        translation_innovation=float(np.linalg.norm(innovation[:3])),
        rotation_innovation=float(np.linalg.norm(innovation[3:])),
        mahalanobis=mahalanobis,
    )

  def pose_uncertainty(self):
    if not self.initialized:
      return np.inf, np.inf
    diagonal = np.maximum(np.diag(self.covariance)[:6], 0.0)
    return (
        float(np.sqrt(diagonal[:3].sum())),
        float(np.sqrt(diagonal[3:].sum())),
    )

  def motion(self):
    return (
        float(np.linalg.norm(self.velocity[:3])),
        float(np.linalg.norm(self.velocity[3:])),
    )
