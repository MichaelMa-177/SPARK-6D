# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  parser.add_argument('--no_display', action='store_true',
                      help='save debug renders without opening an OpenCV window')
  parser.add_argument('--save_render', action='store_true',
                      help='save every final pose render to debug_dir/track_vis')
  parser.add_argument('--use_kalman', action='store_true',
                      help='enable SE(3) Kalman prediction and adaptive refinement')
  parser.add_argument('--kalman_refine_interval', type=int, default=2,
                      help='maximum frames between neural refinements')
  parser.add_argument('--kalman_no_adaptive', action='store_true',
                      help='disable high-motion/uncertainty fallback')
  parser.add_argument('--max_frames', type=int, default=0,
                      help='process at most this many frames (0 means all)')
  parser.add_argument('--benchmark_report', type=str, default=None,
                      help='optional JSON path for timing and Kalman statistics')
  parser.add_argument('--quiet', action='store_true',
                      help='suppress per-stage INFO logs for maximum throughput')
  parser.add_argument('--cudnn_benchmark', action='store_true',
                      help='deprecated compatibility flag; deterministic mode keeps it disabled')
  parser.add_argument('--fast_io', action='store_true',
                      help='prefetch input frames and write outputs asynchronously')
  parser.add_argument('--png_compression', type=int, default=1,
                      choices=range(10), metavar='0..9',
                      help='PNG compression used for saved debug renders')
  args = parser.parse_args()

  set_logging_format()
  if args.quiet:
    logging.getLogger().setLevel(logging.WARNING)
  set_seed(0)
  if args.cudnn_benchmark:
    logging.warning(
        '--cudnn_benchmark is ignored because exact deterministic tracking '
        'is required')

  mesh = trimesh.load(args.mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  if args.use_kalman:
    est.enable_kalman(
        refine_interval=args.kalman_refine_interval,
        adaptive=not args.kalman_no_adaptive)
  logging.info("estimator initialization done")

  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

  frame_count = len(reader.color_files)
  if args.max_frames > 0:
    frame_count = min(frame_count, args.max_frames)

  def load_frame(index):
    color = reader.get_color(index)
    depth = reader.get_depth(index)
    gray = (
        est._kalman_gray(color)
        if args.use_kalman and args.fast_io else None)
    return color, depth, gray

  def write_outputs(pose_path, pose_to_write, render_path, render_to_write):
    np.savetxt(pose_path, pose_to_write.reshape(4,4))
    if render_path is not None:
      imageio.imwrite(
          render_path, render_to_write,
          compress_level=args.png_compression)

  frame_executor = None
  next_frame = None
  write_executor = None
  pending_writes = deque()
  register_seconds = 0.0
  track_seconds = []
  refined_track_seconds = []
  predicted_track_seconds = []
  run_started = time.perf_counter()
  if args.fast_io:
    frame_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix='foundationpose-input')
    write_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix='foundationpose-output')
    next_frame = frame_executor.submit(load_frame, 0)

  for i in range(frame_count):
    logging.info(f'i:{i}')
    if frame_executor is None:
      color, depth, gray = load_frame(i)
    else:
      color, depth, gray = next_frame.result()
      next_frame = (
          frame_executor.submit(load_frame, i + 1)
          if i + 1 < frame_count else None)
    if i==0:
      mask = reader.get_mask(0).astype(bool)
      torch.cuda.synchronize()
      started = time.perf_counter()
      pose = est.register(
          K=reader.K, rgb=color, depth=depth, ob_mask=mask,
          iteration=args.est_refine_iter, gray=gray)
      torch.cuda.synchronize()
      register_seconds = time.perf_counter() - started

      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      predicted_before = est.kalman_stats['predicted_frames']
      torch.cuda.synchronize()
      started = time.perf_counter()
      pose = est.track_one(
          rgb=color, depth=depth, K=reader.K,
          iteration=args.track_refine_iter, gray=gray)
      torch.cuda.synchronize()
      elapsed = time.perf_counter() - started
      track_seconds.append(elapsed)
      if est.kalman_stats['predicted_frames'] > predicted_before:
        predicted_track_seconds.append(elapsed)
      else:
        refined_track_seconds.append(elapsed)

    render_path = None
    render_to_write = None
    if debug>=1 or args.save_render:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      if not args.no_display:
        cv2.imshow('1', vis[...,::-1])
        cv2.waitKey(1)


    if debug>=2 or args.save_render:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      render_path = f'{debug_dir}/track_vis/{reader.id_strs[i]}.png'
      render_to_write = vis

    pose_path = f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt'
    if write_executor is None:
      write_outputs(pose_path, pose, render_path, render_to_write)
    else:
      while len(pending_writes) >= 2:
        pending_writes.popleft().result()
      pending_writes.append(write_executor.submit(
          write_outputs, pose_path, pose.copy(), render_path,
          None if render_to_write is None else render_to_write.copy()))

  while pending_writes:
    pending_writes.popleft().result()
  if write_executor is not None:
    write_executor.shutdown(wait=True)
  if frame_executor is not None:
    frame_executor.shutdown(wait=True)
  est.close()

  run_seconds = time.perf_counter() - run_started
  tracking_fps = len(track_seconds) / sum(track_seconds) if track_seconds else 0.0
  def timing_percentiles(seconds):
    if not seconds:
      return {'mean_ms': 0.0, 'median_ms': 0.0, 'p95_ms': 0.0}
    milliseconds = np.asarray(seconds, dtype=np.float64) * 1000.0
    return {
        'mean_ms': float(milliseconds.mean()),
        'median_ms': float(np.median(milliseconds)),
        'p95_ms': float(np.percentile(milliseconds, 95)),
    }

  report = {
      'frames': frame_count,
      'register_seconds': register_seconds,
      'tracking_seconds': float(sum(track_seconds)),
      'tracking_fps': float(tracking_fps),
      'end_to_end_fps': (
          float(frame_count / run_seconds) if run_seconds else 0.0),
      'refined_frame_ms': (
          1000.0 * sum(refined_track_seconds) / len(refined_track_seconds)
          if refined_track_seconds else 0.0),
      'predicted_frame_ms': (
          1000.0 * sum(predicted_track_seconds) / len(predicted_track_seconds)
          if predicted_track_seconds else 0.0),
      'run_seconds': run_seconds,
      'refined_timing': timing_percentiles(refined_track_seconds),
      'predicted_timing': timing_percentiles(predicted_track_seconds),
      'kalman': est.get_kalman_stats(),
      'fast_io': bool(args.fast_io),
  }
  logging.info(f'benchmark: {json.dumps(report, sort_keys=True)}')
  if args.benchmark_report:
    report_dir = os.path.dirname(os.path.abspath(args.benchmark_report))
    os.makedirs(report_dir, exist_ok=True)
    with open(args.benchmark_report, 'w', encoding='utf-8') as report_file:
      json.dump(report, report_file, indent=2, sort_keys=True)
