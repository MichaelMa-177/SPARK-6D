"""Low-overhead hierarchical CPU/CUDA stage timing for FoundationPose."""

from contextlib import contextmanager, nullcontext
import json
import os
import statistics
import threading
import time


_ACTIVE_PROFILER = None
_THREAD_CONTEXT = threading.local()


def _percentile(values, fraction):
  if not values:
    return 0.0
  ordered = sorted(values)
  position = (len(ordered) - 1) * fraction
  lower = int(position)
  upper = min(lower + 1, len(ordered) - 1)
  weight = position - lower
  return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(records, key):
  values = [record[key] for record in records if record.get(key) is not None]
  if not values:
    return None
  return {
      'count': len(values),
      'total_ms': float(sum(values)),
      'mean_ms': float(statistics.fmean(values)),
      'median_ms': float(statistics.median(values)),
      'p95_ms': float(_percentile(values, 0.95)),
      'p99_ms': float(_percentile(values, 0.99)),
      'min_ms': float(min(values)),
      'max_ms': float(max(values)),
  }


class StageProfiler:
  def __init__(self, output_path, warmup_frames=50, sync_cuda=False):
    self.output_path = os.path.abspath(output_path)
    self.warmup_frames = int(warmup_frames)
    self.sync_cuda = bool(sync_cuda)
    self.started_ns = time.perf_counter_ns()
    self.records = []
    self.frame_modes = {}
    self._lock = threading.Lock()
    self._torch = None

  def _get_torch(self):
    if self._torch is None:
      try:
        import torch
        self._torch = torch
      except ImportError:
        self._torch = False
    return self._torch or None

  def _record_function(self, name):
    torch = self._get_torch()
    if torch is None:
      return nullcontext()
    return torch.profiler.record_function(name)

  @contextmanager
  def stage(self, name, cuda=False, frame=None, metadata=None):
    if frame is None:
      frame = getattr(_THREAD_CONTEXT, 'frame', None)
    torch = self._get_torch() if cuda else None
    use_cuda = bool(
        cuda and torch is not None and torch.cuda.is_available())
    start_event = None
    end_event = None
    if use_cuda:
      if self.sync_cuda:
        torch.cuda.synchronize()
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      start_event.record()
    start_ns = time.perf_counter_ns()
    ok = False
    try:
      with self._record_function(name):
        yield
      ok = True
    finally:
      if use_cuda:
        end_event.record()
        if self.sync_cuda:
          end_event.synchronize()
      end_ns = time.perf_counter_ns()
      record = {
          'stage': str(name),
          'frame': None if frame is None else int(frame),
          'thread': threading.current_thread().name,
          'start_ms': (start_ns - self.started_ns) / 1e6,
          'cpu_ms': (end_ns - start_ns) / 1e6,
          'gpu_ms': None,
          'ok': ok,
          'metadata': dict(metadata or {}),
          '_start_event': start_event,
          '_end_event': end_event,
      }
      if self.sync_cuda and use_cuda:
        record['gpu_ms'] = float(start_event.elapsed_time(end_event))
        record['_start_event'] = None
        record['_end_event'] = None
      with self._lock:
        self.records.append(record)

  def set_frame_mode(self, frame, mode):
    with self._lock:
      self.frame_modes[int(frame)] = str(mode)

  def _resolve_cuda_events(self):
    pending = [
        record for record in self.records
        if record['_end_event'] is not None]
    if pending:
      torch = self._get_torch()
      if torch is not None:
        torch.cuda.synchronize()
      for record in pending:
        try:
          record['gpu_ms'] = float(
              record['_start_event'].elapsed_time(record['_end_event']))
        except RuntimeError:
          record['gpu_ms'] = None
        record['_start_event'] = None
        record['_end_event'] = None

  @staticmethod
  def _summarize(records):
    grouped = {}
    for record in records:
      grouped.setdefault(record['stage'], []).append(record)
    return {
        stage: {
            'cpu': _stats(stage_records, 'cpu_ms'),
            'gpu': _stats(stage_records, 'gpu_ms'),
        }
        for stage, stage_records in sorted(grouped.items())
    }

  def write(self, metadata=None):
    with self._lock:
      self._resolve_cuda_events()
      records = []
      for source in self.records:
        record = {
            key: value for key, value in source.items()
            if not key.startswith('_')
        }
        frame = record['frame']
        record['mode'] = self.frame_modes.get(frame) if frame is not None else None
        records.append(record)

    steady_records = [
        record for record in records
        if record['frame'] is not None
        and record['frame'] >= self.warmup_frames]
    mode_summary = {}
    modes = sorted({record['mode'] for record in records if record['mode']})
    for mode in modes:
      mode_summary[mode] = self._summarize(
          [record for record in records if record['mode'] == mode])
    payload = {
        'schema_version': 1,
        'clock': 'time.perf_counter_ns',
        'cuda_timing': 'synchronized_per_stage' if self.sync_cuda else 'cuda_events',
        'warmup_frames': self.warmup_frames,
        'metadata': dict(metadata or {}),
        'summary': self._summarize(records),
        'steady_state_summary': self._summarize(steady_records),
        'mode_summary': mode_summary,
        'records': records,
    }
    output_dir = os.path.dirname(self.output_path)
    if output_dir:
      os.makedirs(output_dir, exist_ok=True)
    with open(self.output_path, 'w', encoding='utf-8') as output_file:
      json.dump(payload, output_file, indent=2, sort_keys=True)
    return self.output_path


def configure_stage_profiler(enabled, output_path, warmup_frames=50,
                             sync_cuda=False):
  global _ACTIVE_PROFILER
  _ACTIVE_PROFILER = (
      StageProfiler(output_path, warmup_frames, sync_cuda)
      if enabled else None)
  return _ACTIVE_PROFILER


def set_profile_frame(frame):
  _THREAD_CONTEXT.frame = None if frame is None else int(frame)


def set_profile_frame_mode(frame, mode):
  if _ACTIVE_PROFILER is not None:
    _ACTIVE_PROFILER.set_frame_mode(frame, mode)


@contextmanager
def profile_stage(name, cuda=False, frame=None, metadata=None):
  if _ACTIVE_PROFILER is None:
    yield
  else:
    with _ACTIVE_PROFILER.stage(
        name, cuda=cuda, frame=frame, metadata=metadata):
      yield


def write_stage_profile(metadata=None):
  if _ACTIVE_PROFILER is None:
    return None
  return _ACTIVE_PROFILER.write(metadata)
