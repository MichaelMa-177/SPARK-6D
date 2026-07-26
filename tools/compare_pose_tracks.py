#!/usr/bin/env python3
"""Compare timing and temporal stability of two FoundationPose pose tracks."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_track(directory):
  files = sorted(Path(directory).glob('*.txt'))
  if not files:
    raise ValueError(f'no pose files found in {directory}')
  poses = np.stack([np.loadtxt(path).reshape(4, 4) for path in files])
  return [path.name for path in files], poses


def relative_rotation(first, second):
  return np.einsum(
      'nij,njk->nik',
      np.transpose(first[..., :3, :3], (0, 2, 1)),
      second[..., :3, :3],
  )


def distribution(values):
  return {
      'rms': float(np.sqrt(np.mean(np.square(values)))),
      'p95': float(np.percentile(values, 95)),
      'max': float(np.max(values)),
  }


def compare_tracks(baseline, candidate):
  translation_deviation = np.linalg.norm(
      candidate[:, :3, 3] - baseline[:, :3, 3], axis=1) * 1000.0
  rotation_deviation = np.degrees(
      Rotation.from_matrix(
          relative_rotation(baseline, candidate)).magnitude())

  translation_acceleration_baseline = np.linalg.norm(
      np.diff(baseline[:, :3, 3], n=2, axis=0), axis=1) * 1000.0
  translation_acceleration_candidate = np.linalg.norm(
      np.diff(candidate[:, :3, 3], n=2, axis=0), axis=1) * 1000.0

  baseline_steps = relative_rotation(baseline[:-1], baseline[1:])
  candidate_steps = relative_rotation(candidate[:-1], candidate[1:])
  rotation_acceleration_baseline = np.degrees(
      Rotation.from_matrix(
          relative_rotation(
              baseline_steps[:-1], baseline_steps[1:])).magnitude())
  rotation_acceleration_candidate = np.degrees(
      Rotation.from_matrix(
          relative_rotation(
              candidate_steps[:-1], candidate_steps[1:])).magnitude())

  valid = (
      np.isfinite(candidate).all()
      and np.max(np.abs(
          np.linalg.det(candidate[:, :3, :3]) - 1.0)) < 1e-4)
  return {
      'frames': len(candidate),
      'valid_se3': bool(valid),
      'deviation_mm': {
          'median': float(np.median(translation_deviation)),
          'p95': float(np.percentile(translation_deviation, 95)),
          'max': float(np.max(translation_deviation)),
          'last': float(translation_deviation[-1]),
      },
      'deviation_deg': {
          'median': float(np.median(rotation_deviation)),
          'p95': float(np.percentile(rotation_deviation, 95)),
          'max': float(np.max(rotation_deviation)),
          'last': float(rotation_deviation[-1]),
      },
      'translation_acceleration_mm': {
          'baseline': distribution(translation_acceleration_baseline),
          'candidate': distribution(translation_acceleration_candidate),
      },
      'rotation_acceleration_deg': {
          'baseline': distribution(rotation_acceleration_baseline),
          'candidate': distribution(rotation_acceleration_candidate),
      },
  }


def load_json(path):
  with open(path, encoding='utf-8') as file:
    return json.load(file)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--baseline', required=True)
  parser.add_argument('--candidate', required=True)
  parser.add_argument('--baseline-timing')
  parser.add_argument('--candidate-timing')
  parser.add_argument('--output')
  parser.add_argument(
      '--require-exact', action='store_true',
      help='exit non-zero unless pose files and Kalman decisions match exactly')
  args = parser.parse_args()

  baseline_names, baseline = load_track(args.baseline)
  candidate_names, candidate = load_track(args.candidate)
  if baseline_names != candidate_names:
    raise ValueError('baseline and candidate frame names do not match')

  report = compare_tracks(baseline, candidate)
  report['exact_pose_files'] = all(
      (Path(args.baseline) / name).read_bytes()
      == (Path(args.candidate) / name).read_bytes()
      for name in baseline_names)
  kalman_stats_match = True
  if args.baseline_timing and args.candidate_timing:
    baseline_timing = load_json(args.baseline_timing)
    candidate_timing = load_json(args.candidate_timing)
    baseline_fps = float(baseline_timing['tracking_fps'])
    candidate_fps = float(candidate_timing['tracking_fps'])
    report['timing'] = {
        'baseline_fps': baseline_fps,
        'candidate_fps': candidate_fps,
        'fps_gain_percent': (candidate_fps / baseline_fps - 1.0) * 100.0,
        'baseline_tracking_seconds': float(
            baseline_timing['tracking_seconds']),
        'candidate_tracking_seconds': float(
            candidate_timing['tracking_seconds']),
    }
    baseline_kalman = baseline_timing.get('kalman', {})
    candidate_kalman = candidate_timing.get('kalman', {})
    summary_keys = (
        'enabled', 'refined_frames', 'predicted_frames',
        'flow_guard_frames')
    kalman_stats_match = all(
        baseline_kalman.get(key) == candidate_kalman.get(key)
        for key in summary_keys)
    baseline_modes = baseline_kalman.get('frame_modes')
    candidate_modes = candidate_kalman.get('frame_modes')
    if baseline_modes is not None and candidate_modes is not None:
      report['kalman_frame_modes_match'] = (
          baseline_modes == candidate_modes)
      kalman_stats_match &= report['kalman_frame_modes_match']
    report['kalman_stats_match'] = kalman_stats_match

  serialized = json.dumps(report, indent=2, sort_keys=True)
  print(serialized)
  if args.output:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized + '\n', encoding='utf-8')
  if args.require_exact and (
      not report['exact_pose_files'] or not kalman_stats_match):
    raise SystemExit(1)


if __name__ == '__main__':
  main()
