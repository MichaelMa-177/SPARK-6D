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
from stage_profiler import (
    configure_stage_profiler,
    profile_stage,
    set_profile_frame,
    set_profile_frame_mode,
    write_stage_profile,
)


class TimelineVideoWriter:
  """Write an annotated video whose timeline follows wall-clock runtime."""

  def __init__(self, path, fps):
    self.path = os.path.abspath(path)
    self.fps = float(fps)
    self.writer = None
    self.frame_count = 0
    self.last_frame = None

  def write_until(self, frame_rgb, elapsed_seconds):
    self.last_frame = frame_rgb
    if self.writer is None:
      output_dir = os.path.dirname(self.path)
      if output_dir:
        os.makedirs(output_dir, exist_ok=True)
      height, width = frame_rgb.shape[:2]
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      self.writer = cv2.VideoWriter(
          self.path, fourcc, self.fps, (width, height))
      if not self.writer.isOpened():
        raise RuntimeError(f'Unable to open video writer: {self.path}')

    target_frames = max(
        1, int(round(max(0.0, elapsed_seconds) * self.fps)))
    frame_bgr = np.ascontiguousarray(frame_rgb[..., ::-1])
    while self.frame_count < target_frames:
      self.writer.write(frame_bgr)
      self.frame_count += 1

  def close(self, elapsed_seconds):
    if self.writer is None:
      return
    self.write_until(self.last_frame, elapsed_seconds)
    self.writer.release()
    self.writer = None

  @property
  def duration_seconds(self):
    return self.frame_count / self.fps if self.fps else 0.0


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
  parser.add_argument('--save_video', action='store_true',
                      help='save the annotated runtime display as an MP4 video')
  parser.add_argument('--video_path', type=str, default=None,
                      help='MP4 output path (default: <debug_dir>/run.mp4)')
  parser.add_argument('--video_fps', type=float, default=30.0,
                      help='saved-video frame rate; frames are sampled/repeated to match runtime')
  parser.add_argument('--fps_window', type=int, default=30,
                      help='number of recent completed frames used for the FPS overlay')
  parser.add_argument('--stage_profile', action='store_true',
                      help='record hierarchical CPU/CUDA timings as JSON')
  parser.add_argument('--stage_profile_output', type=str, default=None,
                      help='timing JSON path (default: <debug_dir>/stage_profile.json)')
  parser.add_argument('--stage_profile_warmup', type=int, default=50,
                      help='frame index where steady-state timing begins')
  parser.add_argument('--stage_profile_sync_cuda', action='store_true',
                      help='synchronize every CUDA stage (accurate but intrusive)')
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
  if args.video_fps <= 0:
    parser.error('--video_fps must be greater than zero')
  if args.fps_window < 2:
    parser.error('--fps_window must be at least 2')
  if args.stage_profile_warmup < 0:
    parser.error('--stage_profile_warmup must not be negative')

  set_logging_format()
  if args.quiet:
    logging.getLogger().setLevel(logging.WARNING)
  set_seed(0)
  if args.cudnn_benchmark:
    logging.warning(
        '--cudnn_benchmark is ignored because exact deterministic tracking '
        'is required')

  debug = args.debug
  debug_dir = args.debug_dir
  stage_profile_output = (
      args.stage_profile_output
      or os.path.join(debug_dir, 'stage_profile.json'))
  configure_stage_profiler(
      args.stage_profile, stage_profile_output,
      warmup_frames=args.stage_profile_warmup,
      sync_cuda=args.stage_profile_sync_cuda)

  with profile_stage('startup.mesh_load'):
    mesh = trimesh.load(args.mesh_file)

  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  with profile_stage('startup.mesh_bounds'):
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  with profile_stage('startup.scorer_init', cuda=True):
    scorer = ScorePredictor()
  with profile_stage('startup.refiner_init', cuda=True):
    refiner = PoseRefinePredictor()
  with profile_stage('startup.raster_context_init', cuda=True):
    glctx = dr.RasterizeCudaContext()
  with profile_stage('startup.estimator_init', cuda=True):
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  if args.use_kalman:
    est.enable_kalman(
        refine_interval=args.kalman_refine_interval,
        adaptive=not args.kalman_no_adaptive)
  logging.info("estimator initialization done")

  with profile_stage('startup.reader_init'):
    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

  frame_count = len(reader.color_files)
  if args.max_frames > 0:
    frame_count = min(frame_count, args.max_frames)

  def load_frame(index):
    set_profile_frame(index)
    with profile_stage('pipeline.input.rgb_decode', frame=index):
      color = reader.get_color(index)
    with profile_stage('pipeline.input.depth_decode', frame=index):
      depth = reader.get_depth(index)
    with profile_stage('pipeline.input.grayscale', frame=index):
      gray = (
          est._kalman_gray(color)
          if args.use_kalman and args.fast_io else None)
    return color, depth, gray

  def write_outputs(frame_index, pose_path, pose_to_write, render_path,
                    render_to_write):
    set_profile_frame(frame_index)
    with profile_stage('pipeline.output.pose_write', frame=frame_index):
      np.savetxt(pose_path, pose_to_write.reshape(4,4))
    if render_path is not None:
      with profile_stage('pipeline.output.render_write', frame=frame_index):
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
  completion_times = deque(maxlen=args.fps_window + 1)
  video = None
  if args.save_video:
    video_path = args.video_path or os.path.join(debug_dir, 'run.mp4')
    video = TimelineVideoWriter(video_path, args.video_fps)
  run_started = time.perf_counter()
  if args.fast_io:
    frame_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix='foundationpose-input')
    write_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix='foundationpose-output')
    next_frame = frame_executor.submit(load_frame, 0)

  for i in range(frame_count):
    set_profile_frame(i)
    logging.info(f'i:{i}')
    with profile_stage('pipeline.input.wait', frame=i):
      if frame_executor is None:
        color, depth, gray = load_frame(i)
      else:
        color, depth, gray = next_frame.result()
        next_frame = (
            frame_executor.submit(load_frame, i + 1)
            if i + 1 < frame_count else None)
    if i==0:
      set_profile_frame_mode(i, 'register')
      with profile_stage('pipeline.input.mask_decode', frame=i):
        mask = reader.get_mask(0).astype(bool)
      torch.cuda.synchronize()
      started = time.perf_counter()
      with profile_stage('pipeline.register', cuda=True, frame=i):
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
      with profile_stage('pipeline.track', cuda=True, frame=i):
        pose = est.track_one(
            rgb=color, depth=depth, K=reader.K,
            iteration=args.track_refine_iter, gray=gray)
      torch.cuda.synchronize()
      elapsed = time.perf_counter() - started
      track_seconds.append(elapsed)
      if est.kalman_stats['predicted_frames'] > predicted_before:
        set_profile_frame_mode(i, 'kalman_predict')
        predicted_track_seconds.append(elapsed)
      else:
        set_profile_frame_mode(i, 'neural_refine')
        refined_track_seconds.append(elapsed)

    render_path = None
    render_to_write = None
    if debug>=1 or args.save_render or args.save_video:
      with profile_stage('pipeline.render_overlay', frame=i):
        completed_at = time.perf_counter()
        completion_times.append(completed_at)
        realtime_fps = 0.0
        if len(completion_times) >= 2:
          window_seconds = completion_times[-1] - completion_times[0]
          if window_seconds > 0:
            realtime_fps = (len(completion_times) - 1) / window_seconds
        center_pose = pose@np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
        fps_text = f'FPS: {realtime_fps:5.1f}'
        cv2.putText(
            vis, fps_text, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(
            vis, fps_text, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (255, 255, 255), 2, cv2.LINE_AA)
      if video is not None:
        with profile_stage('pipeline.video_encode', frame=i):
          video.write_until(vis, completed_at - run_started)
      if not args.no_display:
        with profile_stage('pipeline.display', frame=i):
          cv2.imshow('1', vis[...,::-1])
          cv2.waitKey(1)


    if debug>=2 or args.save_render:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      render_path = f'{debug_dir}/track_vis/{reader.id_strs[i]}.png'
      render_to_write = vis

    pose_path = f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt'
    if write_executor is None:
      write_outputs(i, pose_path, pose, render_path, render_to_write)
    else:
      with profile_stage('pipeline.output.queue_wait', frame=i):
        while len(pending_writes) >= 2:
          pending_writes.popleft().result()
      pending_writes.append(write_executor.submit(
          write_outputs, i, pose_path, pose.copy(), render_path,
          None if render_to_write is None else render_to_write.copy()))

  set_profile_frame(None)
  with profile_stage('pipeline.output.flush_wait'):
    while pending_writes:
      pending_writes.popleft().result()
  if write_executor is not None:
    write_executor.shutdown(wait=True)
  if frame_executor is not None:
    frame_executor.shutdown(wait=True)
  with profile_stage('pipeline.estimator_close', cuda=True):
    est.close()

  run_seconds = time.perf_counter() - run_started
  if video is not None:
    with profile_stage('pipeline.video_flush'):
      video.close(run_seconds)
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
      'video': (
          {
              'path': video.path,
              'fps': video.fps,
              'frames': video.frame_count,
              'duration_seconds': video.duration_seconds,
              'timeline_error_seconds': video.duration_seconds - run_seconds,
          }
          if video is not None else None),
  }
  logging.info(f'benchmark: {json.dumps(report, sort_keys=True)}')
  if args.benchmark_report:
    report_dir = os.path.dirname(os.path.abspath(args.benchmark_report))
    os.makedirs(report_dir, exist_ok=True)
    with profile_stage('pipeline.output.benchmark_report_write'):
      with open(args.benchmark_report, 'w', encoding='utf-8') as report_file:
        json.dump(report, report_file, indent=2, sort_keys=True)
  written_stage_profile = write_stage_profile({
      'variant': 'SPARK-6D',
      'frames': frame_count,
      'mesh_file': os.path.abspath(args.mesh_file),
      'test_scene_dir': os.path.abspath(args.test_scene_dir),
      'est_refine_iter': args.est_refine_iter,
      'track_refine_iter': args.track_refine_iter,
      'use_kalman': bool(args.use_kalman),
      'fast_io': bool(args.fast_io),
  })
  if written_stage_profile:
    logging.info(f'stage profile: {written_stage_profile}')
