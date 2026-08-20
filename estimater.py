# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
from datareader import *
import itertools
from learning.training.predict_score import *
from learning.training.predict_pose_refine import *
from pose_kalman import SE3KalmanFilter
import yaml
from concurrent.futures import ThreadPoolExecutor
from stage_profiler import profile_stage


class FoundationPose:
  def __init__(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer:ScorePredictor=None, refiner:PoseRefinePredictor=None, glctx=None, debug=0, debug_dir='/home/bowen/debug/novel_pose_debug/'):
    self.gt_pose = None
    self.ignore_normal_flip = True
    self.debug = debug
    self.debug_dir = debug_dir
    os.makedirs(debug_dir, exist_ok=True)

    self.reset_object(model_pts, model_normals, symmetry_tfs=symmetry_tfs, mesh=mesh)
    self.make_rotation_grid(min_n_views=40, inplane_step=60)

    self.glctx = glctx

    if scorer is not None:
      self.scorer = scorer
    else:
      self.scorer = ScorePredictor()

    if refiner is not None:
      self.refiner = refiner
    else:
      self.refiner = PoseRefinePredictor()

    self.pose_last = None   # Used for tracking; per the centered mesh
    self.kalman_filter = None
    self.kalman_refine_interval = 2
    self.kalman_adaptive = True
    self.kalman_track_count = 0
    self.kalman_frames_since_refine = 0
    self.kalman_force_refine = False
    self.kalman_stats = {
        'refined_frames': 0,
        'predicted_frames': 0,
        'flow_guard_frames': 0,
    }
    self.kalman_frame_modes = []
    self.kalman_prev_gray = None
    self.kalman_gray_executor = None
    self.kalman_bbox_corners = None
    self.kalman_flow_dilate_kernel = np.ones((7, 7), np.uint8)
    self.track_K_key = None
    self.track_K_tensor = None


  def reset_object(self, model_pts, model_normals, symmetry_tfs=None, mesh=None):
    max_xyz = mesh.vertices.max(axis=0)
    min_xyz = mesh.vertices.min(axis=0)
    self.model_center = (min_xyz+max_xyz)/2
    if mesh is not None:
      self.mesh_ori = mesh.copy()
      mesh = mesh.copy()
      mesh.vertices = mesh.vertices - self.model_center.reshape(1,3)

    model_pts = mesh.vertices
    self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
    self.vox_size = max(self.diameter/20.0, 0.003)
    logging.info(f'self.diameter:{self.diameter}, vox_size:{self.vox_size}')
    self.dist_bin = self.vox_size/2
    self.angle_bin = 20  # Deg
    pcd = toOpen3dCloud(model_pts, normals=model_normals)
    pcd = pcd.voxel_down_sample(self.vox_size)
    self.max_xyz = np.asarray(pcd.points).max(axis=0)
    self.min_xyz = np.asarray(pcd.points).min(axis=0)
    self.pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
    self.normals = F.normalize(torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device='cuda'), dim=-1)
    logging.info(f'self.pts:{self.pts.shape}')
    self.mesh_path = None
    self.mesh = mesh
    if self.mesh is not None:
      self.mesh_path = f'/tmp/{uuid.uuid4()}.obj'
      self.mesh.export(self.mesh_path)
    self.mesh_tensors = make_mesh_tensors(self.mesh)

    self.tf_to_centered_mesh = torch.eye(
        4, dtype=torch.float, device='cuda')
    self.tf_to_centered_mesh[:3, 3] = -torch.as_tensor(
        self.model_center, device='cuda', dtype=torch.float)

    if symmetry_tfs is None:
      self.symmetry_tfs = torch.eye(4).float().cuda()[None]
    else:
      self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device='cuda', dtype=torch.float)

    if getattr(self, 'kalman_filter', None) is not None:
      logging.warning(
          'reset_object invalidates the active Kalman state; re-enable it '
          'after registering the new object')
      self.disable_kalman()

    logging.info("reset done")



  def get_tf_to_centered_mesh(self):
    return self.tf_to_centered_mesh


  def enable_kalman(
      self,
      refine_interval=2,
      adaptive=True,
      process_translation=None,
      process_rotation=np.deg2rad(0.35),
      measurement_translation=None,
      measurement_rotation=np.deg2rad(0.20),
  ):
    """Enable SE(3) Kalman prediction and adaptive FoundationPose refinement.

    ``refine_interval=2`` skips at most one neural refinement in smooth
    motion.  High motion, high uncertainty, or a large previous correction
    automatically restores refinement on every frame.
    """
    if refine_interval < 1:
      raise ValueError('refine_interval must be at least 1')
    self.kalman_filter = SE3KalmanFilter(
        object_diameter=self.diameter,
        process_translation=process_translation,
        process_rotation=process_rotation,
        measurement_translation=measurement_translation,
        measurement_rotation=measurement_rotation,
    )
    self.kalman_refine_interval = int(refine_interval)
    self.kalman_adaptive = bool(adaptive)
    self.kalman_track_count = 0
    self.kalman_frames_since_refine = 0
    self.kalman_force_refine = False
    self.kalman_stats = {
        'refined_frames': 0,
        'predicted_frames': 0,
        'flow_guard_frames': 0,
    }
    self.kalman_frame_modes = []
    self.kalman_prev_gray = None
    if self.kalman_gray_executor is None:
      self.kalman_gray_executor = ThreadPoolExecutor(
          max_workers=1, thread_name_prefix='foundationpose-gray')
    self.kalman_bbox_corners = np.asarray(list(itertools.product(
        [self.min_xyz[0], self.max_xyz[0]],
        [self.min_xyz[1], self.max_xyz[1]],
        [self.min_xyz[2], self.max_xyz[2]])), dtype=np.float64)
    if self.pose_last is not None:
      pose_np = self.pose_last.reshape(4, 4).detach().cpu().numpy()
      self.kalman_filter.initialize(pose_np)


  def disable_kalman(self):
    self.kalman_filter = None
    self.close()


  def close(self):
    if self.kalman_gray_executor is not None:
      self.kalman_gray_executor.shutdown(wait=True)
      self.kalman_gray_executor = None


  def _kalman_should_refine(self):
    # The first two tracking frames establish a reliable velocity estimate.
    if self.kalman_track_count <= 2 or self.kalman_force_refine:
      return True
    if self.kalman_frames_since_refine + 1 >= self.kalman_refine_interval:
      return True
    if not self.kalman_adaptive:
      return False

    translation_motion, rotation_motion = self.kalman_filter.motion()
    translation_uncertainty, rotation_uncertainty = (
        self.kalman_filter.pose_uncertainty())
    high_motion = (
        translation_motion > self.diameter * 0.020
        or rotation_motion > np.deg2rad(1.5))
    high_uncertainty = (
        translation_uncertainty > self.diameter * 0.025
        or rotation_uncertainty > np.deg2rad(1.5))
    return high_motion or high_uncertainty


  def _pose_last_from_numpy(self, pose):
    device = self.pts.device
    self.pose_last = torch.as_tensor(
        pose, device=device, dtype=torch.float32).reshape(4, 4)


  def _kalman_gray(self, rgb):
    if rgb.ndim == 2:
      return np.ascontiguousarray(rgb.astype(np.uint8, copy=False))
    return cv2.cvtColor(
        np.ascontiguousarray(rgb.astype(np.uint8, copy=False)),
        cv2.COLOR_RGB2GRAY)


  def _kalman_project_bbox(self, pose, K, height, width):
    corners = self.kalman_bbox_corners
    if corners is None:
      corners = np.asarray(list(itertools.product(
          [self.min_xyz[0], self.max_xyz[0]],
          [self.min_xyz[1], self.max_xyz[1]],
          [self.min_xyz[2], self.max_xyz[2]])), dtype=np.float64)
    points_cam = (
        pose[:3, :3] @ corners.T + pose[:3, 3:4]).T
    valid = points_cam[:, 2] > 1e-4
    if valid.sum() < 3:
      return None
    projected = (K @ points_cam[valid].T).T
    projected = projected[:, :2] / projected[:, 2:3]
    projected[:, 0] = np.clip(projected[:, 0], 0, width - 1)
    projected[:, 1] = np.clip(projected[:, 1], 0, height - 1)
    hull = cv2.convexHull(
        np.round(projected).astype(np.int32).reshape(-1, 1, 2))
    if abs(cv2.contourArea(hull)) < 25:
      return None
    return hull


  def _kalman_project_center(self, pose, K):
    center_cam = pose[:3, 3]
    if center_cam[2] <= 1e-4:
      return None
    projected = K @ center_cam
    return projected[:2] / projected[2]


  def _kalman_flow_requires_refine(
      self, rgb, K, previous_pose, predicted_pose, current_gray=None):
    """Detect unmodelled image motion before skipping neural refinement."""
    if current_gray is None:
      current_gray = self._kalman_gray(rgb)
    if self.kalman_prev_gray is None:
      return True, current_gray
    if current_gray.shape != self.kalman_prev_gray.shape:
      return True, current_gray

    height, width = current_gray.shape
    hull = self._kalman_project_bbox(
        previous_pose, K, height, width)
    previous_center = self._kalman_project_center(previous_pose, K)
    predicted_center = self._kalman_project_center(predicted_pose, K)
    if hull is None or previous_center is None or predicted_center is None:
      return True, current_gray

    mask = np.zeros_like(current_gray)
    cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.dilate(
        mask, self.kalman_flow_dilate_kernel, iterations=1)
    # goodFeaturesToTrack only evaluates pixels selected by ``mask``.  Give
    # it the same mask plus the full detector halo in a tight ROI, then map
    # the resulting integer-pixel corners back to full-image coordinates.
    # LK remains full-frame, so its pyramid and border behaviour are intact.
    x, y, roi_width, roi_height = cv2.boundingRect(mask)
    detector_halo = 4
    x0 = max(0, x - detector_halo)
    y0 = max(0, y - detector_halo)
    x1 = min(width, x + roi_width + detector_halo)
    y1 = min(height, y + roi_height + detector_halo)
    feature_image = self.kalman_prev_gray[y0:y1, x0:x1]
    feature_mask = mask[y0:y1, x0:x1]
    points = cv2.goodFeaturesToTrack(
        feature_image,
        maxCorners=80,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=5,
        mask=feature_mask,
    )
    if points is None or len(points) < 6:
      return True, current_gray
    points += np.asarray([x0, y0], dtype=points.dtype)

    tracked, status, error = cv2.calcOpticalFlowPyrLK(
        self.kalman_prev_gray,
        current_gray,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )
    if tracked is None or status is None:
      return True, current_gray
    valid = status.reshape(-1).astype(bool)
    if error is not None:
      valid &= error.reshape(-1) < 30.0
    if valid.sum() < 6:
      return True, current_gray

    observed_flow = np.median(
        tracked[valid, 0] - points[valid, 0], axis=0)
    predicted_flow = predicted_center - previous_center
    mismatch = np.linalg.norm(observed_flow - predicted_flow)
    observed_motion = np.linalg.norm(observed_flow)
    # A small amount of optical noise is harmless.  Larger disagreement
    # indicates acceleration/occlusion that the constant-velocity state
    # cannot safely extrapolate through.
    requires_refine = mismatch > 1.5 and observed_motion > 1.0
    return bool(requires_refine), current_gray


  def get_kalman_stats(self):
    stats = dict(self.kalman_stats)
    stats['enabled'] = self.kalman_filter is not None
    stats['frame_modes'] = list(self.kalman_frame_modes)
    return stats


  def to_device(self, s='cuda:0'):
    if str(s).startswith('cuda'):
      cuda_device = torch.device(s)
      if cuda_device.index is not None:
        torch.cuda.set_device(cuda_device)
    for k in self.__dict__:
      self.__dict__[k] = self.__dict__[k]
      if torch.is_tensor(self.__dict__[k]) or isinstance(self.__dict__[k], nn.Module):
        logging.info(f"Moving {k} to device {s}")
        self.__dict__[k] = self.__dict__[k].to(s)
    for k in self.mesh_tensors:
      logging.info(f"Moving {k} to device {s}")
      self.mesh_tensors[k] = self.mesh_tensors[k].to(s)
    if self.refiner is not None:
      if hasattr(self.refiner, 'to_device'):
        self.refiner.to_device(s)
      else:
        self.refiner.model.to(s)
    if self.scorer is not None:
      self.scorer.model.to(s)
    if self.glctx is not None:
      self.glctx = dr.RasterizeCudaContext(s)



  def make_rotation_grid(self, min_n_views=40, inplane_step=60):
    cam_in_obs = sample_views_icosphere(n_views=min_n_views)
    logging.info(f'cam_in_obs:{cam_in_obs.shape}')
    rot_grid = []
    for i in range(len(cam_in_obs)):
      for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
        cam_in_ob = cam_in_obs[i]
        R_inplane = euler_matrix(0,0,inplane_rot)
        cam_in_ob = cam_in_ob@R_inplane
        ob_in_cam = np.linalg.inv(cam_in_ob)
        rot_grid.append(ob_in_cam)

    rot_grid = np.asarray(rot_grid)
    logging.info(f"rot_grid:{rot_grid.shape}")
    rot_grid = mycpp.cluster_poses(30, 99999, rot_grid, self.symmetry_tfs.data.cpu().numpy())
    rot_grid = np.asarray(rot_grid)
    logging.info(f"after cluster, rot_grid:{rot_grid.shape}")
    self.rot_grid = torch.as_tensor(rot_grid, device='cuda', dtype=torch.float)
    logging.info(f"self.rot_grid: {self.rot_grid.shape}")


  def generate_random_pose_hypo(self, K, rgb, depth, mask, scene_pts=None):
    '''
    @scene_pts: torch tensor (N,3)
    '''
    ob_in_cams = self.rot_grid.clone()
    center = self.guess_translation(depth=depth, mask=mask, K=K)
    ob_in_cams[:,:3,3] = torch.tensor(center, device='cuda', dtype=torch.float).reshape(1,3)
    return ob_in_cams


  def guess_translation(self, depth, mask, K):
    vs,us = np.where(mask>0)
    if len(us)==0:
      logging.info(f'mask is all zero')
      return np.zeros((3))
    uc = (us.min()+us.max())/2.0
    vc = (vs.min()+vs.max())/2.0
    valid = mask.astype(bool) & (depth>=0.001)
    if not valid.any():
      logging.info(f"valid is empty")
      return np.zeros((3))

    zc = np.median(depth[valid])
    center = (np.linalg.inv(K)@np.asarray([uc,vc,1]).reshape(3,1))*zc

    if self.debug>=2:
      pcd = toOpen3dCloud(center.reshape(1,3))
      o3d.io.write_point_cloud(f'{self.debug_dir}/init_center.ply', pcd)

    return center.reshape(3)


  def register(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None,
               iteration=5, gray=None):
    '''Copmute pose from given pts to self.pcd
    @pts: (N,3) np array, downsampled scene points
    '''
    set_seed(0)
    logging.info('Welcome')

    if self.glctx is None:
      if glctx is None:
        self.glctx = dr.RasterizeCudaContext()
        # self.glctx = dr.RasterizeGLContext()
      else:
        self.glctx = glctx

    with profile_stage('register.depth_erode', cuda=True):
      depth = erode_depth(depth, radius=2, device='cuda')
    with profile_stage('register.depth_bilateral', cuda=True):
      depth = bilateral_filter_depth(depth, radius=2, device='cuda')

    if self.debug>=2:
      xyz_map = depth2xyzmap(depth, K)
      valid = xyz_map[...,2]>=0.001
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_raw.ply',pcd)
      cv2.imwrite(f'{self.debug_dir}/ob_mask.png', (ob_mask*255.0).clip(0,255))

    normal_map = None
    valid = (depth>=0.001) & (ob_mask>0)
    if valid.sum()<4:
      logging.info(f'valid too small, return')
      pose = np.eye(4)
      pose[:3,3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
      return pose

    if self.debug>=2:
      imageio.imwrite(f'{self.debug_dir}/color.png', rgb)
      cv2.imwrite(f'{self.debug_dir}/depth.png', (depth*1000).astype(np.uint16))
      valid = xyz_map[...,2]>=0.001
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_complete.ply',pcd)

    self.H, self.W = depth.shape[:2]
    self.K = K
    K_array = np.asarray(K)
    self.track_K_key = (
        K_array.shape, K_array.dtype.str, K_array.tobytes())
    self.track_K_tensor = torch.as_tensor(
        K_array, dtype=torch.float, device='cuda')
    self.ob_id = ob_id
    self.ob_mask = ob_mask

    with profile_stage('register.hypothesis_generation', cuda=True):
      poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None)
      poses = poses.data.cpu().numpy()
      logging.info(f'poses:{poses.shape}')
      center = self.guess_translation(depth=depth, mask=ob_mask, K=K)

      poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
      poses[:,:3,3] = torch.as_tensor(center.reshape(1,3), device='cuda')

      add_errs = self.compute_add_err_to_gt_pose(poses)
      logging.info(f"after viewpoint, add_errs min:{add_errs.min()}")

    with profile_stage('register.xyz_map', cuda=True):
      xyz_map = depth2xyzmap(depth, K)
    with profile_stage('register.refiner', cuda=True):
      poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map, glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration, get_vis=self.debug>=2)
    if vis is not None:
      imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)

    with profile_stage('register.scorer', cuda=True):
      scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter, get_vis=self.debug>=2)
    if vis is not None:
      imageio.imwrite(f'{self.debug_dir}/vis_score.png', vis)

    with profile_stage('register.rank_and_select', cuda=True):
      add_errs = self.compute_add_err_to_gt_pose(poses)
      logging.info(f"final, add_errs min:{add_errs.min()}")

      ids = torch.as_tensor(scores).argsort(descending=True)
      logging.info(f'sort ids:{ids}')
      scores = scores[ids]
      poses = poses[ids]

      logging.info(f'sorted scores:{scores}')

      best_pose = poses[0]@self.get_tf_to_centered_mesh()
      self.pose_last = poses[0]
      self.best_id = ids[0]

      self.poses = poses
      self.scores = scores

    if self.kalman_filter is not None:
      with profile_stage('register.kalman_initialize'):
        self.kalman_filter.initialize(
            self.pose_last.reshape(4, 4).detach().cpu().numpy())
        self.kalman_track_count = 0
        self.kalman_frames_since_refine = 0
        self.kalman_force_refine = False
        self.kalman_stats = {
            'refined_frames': 0,
            'predicted_frames': 0,
            'flow_guard_frames': 0,
        }
        self.kalman_frame_modes = []
        self.kalman_prev_gray = (
            gray if gray is not None else self._kalman_gray(rgb))

    return best_pose.data.cpu().numpy()


  def compute_add_err_to_gt_pose(self, poses):
    '''
    @poses: wrt. the centered mesh
    '''
    return -torch.ones(len(poses), device='cuda', dtype=torch.float)


  def track_one(self, rgb, depth, K, iteration, extra={}, gray=None):
    if self.pose_last is None:
      logging.info("Please init pose by register first")
      raise RuntimeError
    logging.info("Welcome")

    if self.kalman_filter is not None:
      self.kalman_track_count += 1
      with profile_stage('track.kalman_predict'):
        previous_pose = self.kalman_filter.pose.copy()
        predicted_pose = self.kalman_filter.predict()
      with profile_stage('track.kalman_decision'):
        should_refine = self._kalman_should_refine()
      current_gray = gray
      flow_guard_triggered = False
      if not should_refine and self.kalman_adaptive:
        with profile_stage('track.kalman_flow_guard'):
          should_refine, current_gray = self._kalman_flow_requires_refine(
              rgb, K, previous_pose, predicted_pose,
              current_gray=current_gray)
        if should_refine:
          flow_guard_triggered = True
          self.kalman_stats['flow_guard_frames'] += 1
      if not should_refine:
        with profile_stage('track.kalman_prediction_output', cuda=True):
          self._pose_last_from_numpy(predicted_pose)
          self.kalman_frames_since_refine += 1
          self.kalman_stats['predicted_frames'] += 1
          self.kalman_frame_modes.append('predicted')
          self.kalman_prev_gray = (
              current_gray if current_gray is not None
              else self._kalman_gray(rgb))
          if self.debug>=2:
            extra['vis'] = None
          logging.info("pose predicted by SE(3) Kalman filter")
          return (
              self.pose_last @ self.get_tf_to_centered_mesh()
          ).detach().cpu().numpy().reshape(4, 4)
      # Keep the original FoundationPose convergence behaviour on measured
      # frames.  A constant-velocity prediction is useful for a skipped
      # frame, but can overshoot when an object stops abruptly; seeding the
      # Refiner from the last confirmed pose preserves its original basin.
      self._pose_last_from_numpy(previous_pose)

    gray_future = None
    if (self.kalman_filter is not None and current_gray is None
        and self.kalman_gray_executor is not None):
      # cvtColor releases the GIL, so it can overlap the CUDA refinement.
      # Its byte-identical result is consumed only after refinement finishes.
      gray_future = self.kalman_gray_executor.submit(
          self._kalman_gray, rgb)

    with profile_stage('track.host_to_device', cuda=True):
      depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    with profile_stage('track.depth_erode', cuda=True):
      depth = erode_depth(depth, radius=2, device='cuda')
    with profile_stage('track.depth_bilateral', cuda=True):
      depth = bilateral_filter_depth(depth, radius=2, device='cuda')
    logging.info("depth processing done")

    with profile_stage('track.xyz_map', cuda=True):
      K_array = np.asarray(K)
      K_key = (K_array.shape, K_array.dtype.str, K_array.tobytes())
      if K_key != self.track_K_key:
        self.track_K_tensor = torch.as_tensor(
            K_array, dtype=torch.float, device='cuda')
        self.track_K_key = K_key
      xyz_map = depth2xyzmap_batch(
          depth[None], self.track_K_tensor[None], zfar=np.inf)[0]

    # The Refiner already accepts tensors.  Keeping the seed pose on CUDA
    # avoids a synchronous CUDA -> CPU -> CUDA round trip on every measured
    # frame without changing a single pose value.
    with profile_stage('track.refiner', cuda=True):
      pose, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=self.pose_last.reshape(1,4,4), normal_map=None, xyz_map=xyz_map, mesh_diameter=self.diameter, glctx=self.glctx, iteration=iteration, get_vis=self.debug>=2)
    logging.info("pose done")
    if self.debug>=2:
      extra['vis'] = vis
    if self.kalman_filter is not None:
      with profile_stage('track.kalman_update', cuda=True):
        measured_pose = pose.reshape(4, 4).detach().cpu().numpy()
        update = self.kalman_filter.update(measured_pose)
        self._pose_last_from_numpy(self.kalman_filter.pose)
        self.kalman_frames_since_refine = 0
        self.kalman_stats['refined_frames'] += 1
        self.kalman_frame_modes.append(
            'flow_refined' if flow_guard_triggered else 'refined')
        # A surprising correction indicates unmodelled acceleration.  Refine
        # the next frame rather than trusting a prediction through the change.
        self.kalman_force_refine = (
            update.translation_innovation > self.diameter * 0.02
            or update.rotation_innovation > np.deg2rad(1.0))
        if current_gray is not None:
          self.kalman_prev_gray = current_gray
        elif gray_future is not None:
          self.kalman_prev_gray = gray_future.result()
        else:
          self.kalman_prev_gray = self._kalman_gray(rgb)
        return (
            self.pose_last @ self.get_tf_to_centered_mesh()
        ).detach().cpu().numpy().reshape(4, 4)

    with profile_stage('track.pose_output', cuda=True):
      self.pose_last = pose
      return (pose@self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4,4)
