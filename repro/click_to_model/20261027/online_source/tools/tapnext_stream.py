"""Strict causal TAPNext++ observations, independent of trusted 3D memory.

The public pixel convention is OpenCV's integer pixel centers. Native TAPNext
uses raster corners, (t,y,x) queries and (y,x) outputs in a 256-coordinate
domain, including when input images are resized to 512. No network access,
support queries, query chunking, compilation, or expected-distance head.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


OFFICIAL_COMMIT = '730cda1c730877cfedbe01bf87fb1cadb78a565d'
COORDINATE_DOMAIN = 256


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _numeric_snapshot(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.dtype.kind not in 'bfiuc':
        raise TypeError('require numeric array evidence')
    return _readonly(value)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _is_digest(value, length=64):
    return isinstance(value, str) and len(value) == length and all(c in '0123456789abcdef' for c in value)


@dataclass(frozen=True)
class TAPNextConfig:
    source_root: str
    checkpoint: str
    checkpoint_sha256: str
    source_commit: str = OFFICIAL_COMMIT
    device: str = 'cpu'
    input_resolution: int = 256
    precision: str = 'float32'
    max_queries: int = 256
    compile_model: bool = False
    matmul_allow_tf32: bool = False
    cudnn_allow_tf32: bool = True

    def __post_init__(self):
        if any(not isinstance(x, str) or not x for x in (self.source_root, self.checkpoint)):
            raise ValueError('explicit local source and checkpoint paths required')
        if not _is_digest(self.checkpoint_sha256) or not _is_digest(self.source_commit, 40):
            raise ValueError('checkpoint SHA256 and source commit must be pinned lowercase hex')
        if self.device not in ('cpu', 'cuda'):
            raise ValueError('device must be cpu or cuda')
        if type(self.input_resolution) is not int or self.input_resolution not in (256, 512):
            raise ValueError('input resolution must be declared as 256 or 512')
        if self.precision not in ('float32', 'cuda_autocast_float16'):
            raise ValueError('precision must be explicit float32 or cuda_autocast_float16')
        if self.precision == 'cuda_autocast_float16' and self.device != 'cuda':
            raise ValueError('float16 autocast requires declared CUDA device')
        if type(self.max_queries) is not int or self.max_queries < 1:
            raise ValueError('max_queries must be a positive integer')
        if self.compile_model is not False:
            raise ValueError('compilation is disabled in this protocol')
        if any(type(x) is not bool for x in (self.matmul_allow_tf32, self.cudnn_allow_tf32)):
            raise ValueError('TF32 flags must be explicit booleans')


def integer_xy_to_native_tyx(points, height, width):
    """Integer-center xy to native raster-center tyx, all query times zero."""
    points = np.asarray(points)
    if (points.ndim != 2 or points.shape[1] != 2 or points.dtype.kind not in 'fiu'
            or not np.isfinite(points).all() or height < 1 or width < 1):
        raise ValueError('finite Nx2 real coordinates and positive image dimensions required')
    out = np.zeros((1, len(points), 3), np.float32)
    out[0, :, 1] = (points[:, 1].astype(np.float64) + .5) * COORDINATE_DOMAIN / height
    out[0, :, 2] = (points[:, 0].astype(np.float64) + .5) * COORDINATE_DOMAIN / width
    return out


def native_yx_to_integer_xy(points, height, width):
    points = np.asarray(points)
    if (points.ndim != 2 or points.shape[1] != 2 or points.dtype.kind not in 'fiu'
            or not np.isfinite(points).all() or height < 1 or width < 1):
        raise ValueError('finite Nx2 native yx coordinates required')
    return points[:, ::-1].astype(np.float64) * np.array([width, height]) / COORDINATE_DOMAIN - .5


def preprocess_rgb(rgb, resolution, *, torch, device):
    """RGB, not BGR; same bilinear/align_corners=False normalization as upstream."""
    image = torch.from_numpy(np.array(rgb, copy=True, order='C')).to(device)
    image = image.permute(2, 0, 1)[None].float()
    image = torch.nn.functional.interpolate(image, size=(resolution, resolution),
                                            mode='bilinear', align_corners=False)
    return image.div(127.5).sub(1.).permute(0, 2, 3, 1)[:, None]


@dataclass(frozen=True)
class TAPNextRawReturn:
    """Detached numeric evidence, captured before any output/state validation.

    Wrong shapes/nonfinite arrays are retained. A failed copy is recorded per
    field, without preventing capture of other outputs. Recurrent tensors stay
    private to the stream; their returned query reference and step are recorded.
    """
    tracks_yx: np.ndarray | None
    track_logits: np.ndarray | None
    visibility_logits: np.ndarray | None
    state_queries_tyx: np.ndarray | None
    state_step: int | None
    return_items: int | None
    capture_errors: tuple[str, ...]


def capture_return(returned):
    errors = []
    count = len(returned) if isinstance(returned, (tuple, list)) else None
    state = returned[3] if count is not None and count > 3 else None

    def array(value, name):
        try:
            return _numeric_snapshot(value)
        except Exception as exc:
            errors.append(name + ':' + type(exc).__name__)
            return None

    arrays = []
    for index, name in enumerate(('tracks', 'track_logits', 'visibility_logits')):
        if count is not None and count > index:
            arrays.append(array(returned[index], name))
        else:
            errors.append(name + ':missing')
            arrays.append(None)
    try:
        queries = array(state.query_points, 'state_queries')
    except Exception as exc:
        errors.append('state_queries:' + type(exc).__name__)
        queries = None
    try:
        step = state.step
        if type(step) is not int:
            errors.append('state_step:not_builtin_int')
            step = None
    except Exception as exc:
        errors.append('state_step:' + type(exc).__name__)
        step = None
    return TAPNextRawReturn(*arrays, queries, step, count, tuple(errors)), state


@dataclass(frozen=True)
class NativeStep:
    raw: TAPNextRawReturn
    state: object


class TAPNextStepError(RuntimeError):
    def __init__(self, message, *, frame_id, raw_return=None):
        super().__init__(message)
        self.frame_id = frame_id
        self.raw_return = raw_return


class TorchTAPNextPPBackend:
    """Lazy local checkpoint; captures actual return before state/output checks.

    Constructing this object does not import torch or the model. Loading is
    isolated from CPU RNG, requires a clean pinned clone and verifies source
    and checkpoint bytes before and after load. There is no downloader.
    """
    def __init__(self, config):
        if not isinstance(config, TAPNextConfig):
            raise TypeError('TAPNextConfig required')
        self.config = config
        self._model = self._torch = None
        self.provenance = None
        self.last_raw_return = None

    def _source_inventory(self):
        root = Path(self.config.source_root).resolve()
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=root, text=True)
        if head != self.config.source_commit or dirty:
            raise ValueError('official source must match the pinned clean commit')
        required = ('tapnet/__init__.py', 'tapnet/tapnext/tapnext_torch.py',
                    'tapnet/tapnext/tapnext_lru_modules.py', 'tapnet/tapnext/pscan.py')
        if not all((root / name).is_file() for name in required):
            raise FileNotFoundError('incomplete official TAPNext source')
        return {str(p.relative_to(root)): _sha(p) for p in sorted((root/'tapnet').rglob('*.py'))}

    def _check_imports(self):
        root = (Path(self.config.source_root)/'tapnet').resolve()
        for name, module in tuple(sys.modules.items()):
            if name == 'tapnet' or name.startswith('tapnet.'):
                origin = getattr(module, '__file__', None)
                locations = [origin] if origin is not None else list(getattr(module, '__path__', ()))
                if not locations or any(not Path(p).resolve().is_relative_to(root) for p in locations):
                    raise ValueError('imported tapnet module differs from pinned source: ' + name)

    def _precision(self):
        torch, cfg = self._torch, self.config
        actual = dict(matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                      cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32))
        if actual != dict(matmul_allow_tf32=cfg.matmul_allow_tf32, cudnn_allow_tf32=cfg.cudnn_allow_tf32):
            raise ValueError('runtime TF32 policy differs from explicit configuration')
        return actual

    def _load(self):
        start = time.perf_counter()
        cfg = self.config
        checkpoint = Path(cfg.checkpoint).resolve()
        before_source = self._source_inventory()
        if not checkpoint.is_file() or _sha(checkpoint) != cfg.checkpoint_sha256:
            raise ValueError('local checkpoint SHA256 differs from pinned input')
        self._check_imports()
        before_path = sys.path.copy()
        try:
            sys.path.insert(0, str(Path(cfg.source_root).resolve()))
            module = importlib.import_module('tapnet.tapnext.tapnext_torch')
        finally:
            sys.path[:] = before_path
        self._check_imports()
        import torch
        self._torch = torch
        precision = self._precision()
        with torch.device('cpu'), torch.random.fork_rng(devices=[]):
            model = module.TAPNext(image_size=(COORDINATE_DOMAIN, COORDINATE_DOMAIN))
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            state = state.get('state_dict', state)
            state = {k.removeprefix('tapnext.'): v for k, v in state.items()}
            model.load_state_dict(state, strict=True)
        model = model.to(device=cfg.device, dtype=torch.float32).eval()
        if self._source_inventory() != before_source or _sha(checkpoint) != cfg.checkpoint_sha256:
            raise ValueError('source or checkpoint changed during loading')
        self.provenance = dict(backend='official_tapnextpp', config=asdict(cfg),
            checkpoint=str(checkpoint), checkpoint_sha256=cfg.checkpoint_sha256,
            source_sha256=before_source, packages={name: importlib.metadata.version(name)
                for name in ('torch', 'torchvision', 'numpy', 'einops')},
            precision=precision, model_parameter_dtype='float32', coordinate_domain=256,
            public_pixel_convention='integer_centers', native_pixel_convention='raster_corners',
            query_order='t,y,x', output_order='y,x', support_queries=0, query_chunking=False,
            expected_distance_head=False, downloads=False, compile_model=False,
            load_seconds=time.perf_counter()-start)
        self._model = model

    def forward(self, rgb, *, query_tyx, state):
        self.last_raw_return = None
        if self._model is None:
            self._load()
        torch, cfg = self._torch, self.config
        self._precision()
        video = preprocess_rgb(rgb, cfg.input_resolution, torch=torch, device=cfg.device)
        query = None if query_tyx is None else torch.from_numpy(np.array(query_tyx, copy=True)).to(cfg.device)
        with torch.inference_mode(), torch.autocast(device_type=cfg.device,
                enabled=cfg.precision == 'cuda_autocast_float16', dtype=torch.float16):
            returned = self._model(video=video, query_points=query, state=state)
            raw, new_state = capture_return(returned)
            self.last_raw_return = raw
        return NativeStep(raw, new_state)


@dataclass(frozen=True)
class TAPNextObservation:
    frame_id: int
    query_ids: np.ndarray
    pixels_xy: np.ndarray
    visibility_logits: np.ndarray
    visibility_probability: np.ndarray
    visible: np.ndarray
    raw_return: TAPNextRawReturn
    elapsed_seconds: float
    identity_verified: bool = False


class TAPNextPointStream:
    """Current-frame-only stream with direct visibility and no trusted writes.

    An injected backend has forward(rgb, *, query_tyx, state) -> NativeStep.
    The first call includes queries; subsequent calls pass only opaque state.
    Raw numeric evidence is exposed even when output/state validation fails.
    """
    def __init__(self, config, backend=None):
        if not isinstance(config, TAPNextConfig):
            raise TypeError('TAPNextConfig required')
        self.config = config
        self._backend = backend if backend is not None else TorchTAPNextPPBackend(config)
        self._frame = self._shape = self._queries = self._native_queries = self._state = None
        self._poisoned = self._busy = False
        self._last_raw = self._failure = None

    @property
    def frame_id(self): return self._frame

    @property
    def poisoned(self): return self._poisoned

    @property
    def initial_queries(self): return None if self._queries is None else self._queries.copy()

    @property
    def last_raw_return(self): return self._last_raw

    @property
    def failure(self): return self._failure

    @property
    def state_step(self): return None if self._state is None else self._state.step

    def _available(self):
        if self._poisoned or self._busy:
            raise RuntimeError('stream poisoned or busy; never replay a partially consumed frame')

    def _image(self, rgb):
        image = np.asarray(rgb)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2:
            raise ValueError('one HxWx3 uint8 RGB frame required')
        if self._shape is not None and image.shape != self._shape:
            raise ValueError('image shape changed; no implicit reset')
        return image

    def start(self, rgb, query_xy, *, frame_id=0):
        self._available()
        if self._frame is not None:
            raise RuntimeError('initial query identities cannot be replaced')
        if type(frame_id) is not int or frame_id != 0:
            raise ValueError('only initial frame zero queries are allowed')
        image, query = self._image(rgb), np.asarray(query_xy)
        if (query.ndim != 2 or query.shape[1] != 2 or query.dtype.kind not in 'fiu'
                or not 1 <= len(query) <= self.config.max_queries or not np.isfinite(query).all()
                or np.any(query < 0) or np.any(query[:, 0] > image.shape[1]-1)
                or np.any(query[:, 1] > image.shape[0]-1)):
            raise ValueError('finite in-image queries within max_queries required')
        self._shape, self._queries = image.shape, _readonly(query.astype(np.float64))
        self._native_queries = _readonly(integer_xy_to_native_tyx(query, *image.shape[:2]))
        return self._step(image, 0)

    def advance(self, rgb, *, frame_id):
        self._available()
        if self._frame is None:
            raise RuntimeError('start must precede advance')
        if type(frame_id) is not int or frame_id != self._frame+1:
            raise ValueError('frame IDs must be consecutive and consumed exactly once')
        return self._step(self._image(rgb), frame_id)

    def _step(self, image, frame_id):
        self._busy = True
        self._last_raw = None
        start = time.perf_counter()
        previous_backend_raw = getattr(self._backend, 'last_raw_return', None)
        try:
            query = self._native_queries.copy() if frame_id == 0 else None
            result = self._backend.forward(image.copy(), query_tyx=query, state=self._state)
            if not isinstance(result, NativeStep) or not isinstance(result.raw, TAPNextRawReturn):
                raise TypeError('backend must return captured NativeStep')
            self._last_raw = result.raw
            raw, count = result.raw, len(self._queries)
            shapes = ((1, 1, count, 2), (1, 1, count, 512), (1, 1, count, 1))
            arrays = (raw.tracks_yx, raw.track_logits, raw.visibility_logits)
            if (raw.return_items != 4 or raw.capture_errors
                    or any(x is None or x.shape != shape or x.dtype.kind != 'f' or not np.isfinite(x).all()
                           for x, shape in zip(arrays, shapes))):
                raise ValueError('invalid native output shapes, types or finite values')
            if (raw.state_step != frame_id+1 or raw.state_queries_tyx is None
                    or raw.state_queries_tyx.dtype != np.float32
                    or not np.array_equal(raw.state_queries_tyx, self._native_queries)
                    or type(getattr(result.state, 'step', None)) is not int
                    or result.state.step != raw.state_step):
                raise ValueError('state clock or immutable query identity changed')
            # The captured query array is evidence, while this mutable state is
            # passed to the next forward. Bind both before publishing a frame.
            actual_queries = _numeric_snapshot(result.state.query_points)
            if (actual_queries.dtype != np.float32
                    or actual_queries.shape != self._native_queries.shape
                    or not np.isfinite(actual_queries).all()
                    or not np.array_equal(actual_queries, raw.state_queries_tyx)
                    or not np.array_equal(actual_queries, self._native_queries)):
                raise ValueError('actual recurrent query state differs from captured initial queries')
            pixels = native_yx_to_integer_xy(raw.tracks_yx[0, 0], *self._shape[:2])
            logits = raw.visibility_logits[0, 0, :, 0].astype(np.float64)
            probability = np.empty_like(logits)
            positive = logits >= 0
            probability[positive] = 1. / (1. + np.exp(-logits[positive]))
            exp_negative = np.exp(logits[~positive])
            probability[~positive] = exp_negative / (1. + exp_negative)
            inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] <= self._shape[1]-1)
                      & (pixels[:, 1] >= 0) & (pixels[:, 1] <= self._shape[0]-1))
            observation = TAPNextObservation(frame_id, _readonly(np.arange(count, dtype=np.int64)),
                _readonly(pixels), _readonly(logits), _readonly(probability),
                _readonly(inside & (logits > 0)), raw, time.perf_counter()-start)
            self._state, self._frame = result.state, frame_id
            return observation
        except Exception as exc:
            self._poisoned = True
            if self._last_raw is None:
                candidate = getattr(self._backend, 'last_raw_return', None)
                if isinstance(candidate, TAPNextRawReturn) and candidate is not previous_backend_raw:
                    self._last_raw = candidate
            self._failure = TAPNextStepError(type(exc).__name__ + ': ' + str(exc),
                                            frame_id=frame_id, raw_return=self._last_raw)
            raise self._failure from exc
        finally:
            self._busy = False
