"""Five warmups then thirty fresh SS1/DINO518 first-frame latent calls."""
import fcntl
import json
import time
import numpy as np
import torch
import cv2
import prepare_fresh_prior as base

RUN = base.WORK / 'runs/warm-latent-benchmark'


def stats(values):
    return {name: float(fn(values)) for name, fn in (
        ('p50', np.median), ('p95', lambda x: np.percentile(x, 95)),
        ('p99', lambda x: np.percentile(x, 99)), ('max', np.max),
        ('min', np.min), ('mean', np.mean))}


@torch.no_grad()
def main():
    RUN.mkdir(exist_ok=True)
    lock = (RUN / 'benchmark.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (RUN / 'DONE').exists(), 'Do not overwrite completed benchmark'
    (RUN / 'STARTED').write_text('Cold call, five warmups, thirty fresh calls\n')
    record = base.scene(59, supervision=False)[0]
    reference = json.loads((base.CODE / 'reference/clm3_report.json').read_text())['first_prior']
    assert base.sha(record['rgb_path']) == reference['rgb_sha256']
    assert base.sha(base.ROOT / 'inputs/mask.png') == reference['human_mask_sha256']
    assert np.array_equal(record['depth'], cv2.imread(str(base.ROOT / 'inputs/first_depth.png'), -1))
    assert 'gt' not in record and 'mask' not in record
    mask = cv2.imread(str(base.ROOT / 'inputs/mask.png'), 0) > 0
    torch.set_num_threads(4)
    torch.manual_seed(0)
    cfg = base.OmegaConf.load(base.WEIGHTS / 'pipeline.yaml')
    cfg.workspace_dir = str(base.WEIGHTS)
    cfg.compile_model = False
    cfg.low_vram = True
    cfg.decode_formats = ['mesh']
    cfg.depth_model = {'_target_': 'torch.nn.Identity'}
    cfg.rendering_engine = 'pytorch3d'
    with torch.device('cuda'):
        start = time.perf_counter()
        pipeline = base.instantiate(cfg)
        torch.cuda.synchronize()
        load_ms = (time.perf_counter() - start) * 1000
        print(json.dumps(dict(event='model_loaded', ms=load_ms)), flush=True)
        generator = pipeline.models['ss_generator']
        generator.inference_steps = 1
        generator.no_shortcut = False
        generator.reverse_fn.strength = generator.reverse_fn.strength_pm = 0
        first_latent = None

        def once(phase, index):
            nonlocal first_latent
            torch.manual_seed(0)  # Reset random noise outside the measured region.
            torch.cuda.synchronize()
            start = time.perf_counter()
            xyz, _ = base.lift(record)
            xyz[record['depth'] == 0] = np.nan
            xyz[:, :, :2] *= -1
            rgba = np.concatenate((record['rgb'], (mask.astype(np.uint8) * 255)[:, :, None]), -1)
            xyz_ms = (time.perf_counter() - start) * 1000
            torch.cuda.synchronize()
            start = time.perf_counter()
            pm = pipeline._clip_pointmap(torch.from_numpy(xyz).cuda().permute(2, 0, 1),
                                         torch.from_numpy(mask.astype(np.float32)).cuda())
            item = pipeline.preprocess_image(rgba, pipeline.ss_preprocessor, pointmap=pm)
            with torch.autocast('cuda', dtype=torch.float16):
                args, kwargs = pipeline.get_condition_input(pipeline.condition_embedders['ss_condition_embedder'],
                                                            item, pipeline.ss_condition_input_mapping)
                shapes = {key: (1, value.pos_emb.shape[0], value.input_layer.in_features)
                          for key, value in generator.reverse_fn.backbone.latent_mapping.items()}
                raw = generator(shapes, 'cuda', *args, **kwargs)
            layout = pipeline.pose_decoder(raw, scene_scale=item.get('pointmap_scale'), scene_shift=item.get('pointmap_shift'))
            torch.cuda.synchronize()
            latent_ms = (time.perf_counter() - start) * 1000
            assert raw['shape'].shape == (1, 4096, 8)
            assert all(torch.isfinite(x).all().item() for x in raw.values())
            assert all(torch.isfinite(layout[k]).all().item() for k in ('rotation', 'translation', 'scale'))
            if first_latent is None:
                first_latent = raw['shape'].detach().clone()
            row = dict(phase=phase, index=index, xyz_rgba_ms=xyz_ms, latent_ms=latent_ms,
                       xyz_plus_latent_ms=xyz_ms + latent_ms,
                       latent_max_abs_difference_from_cold=float((raw['shape'] - first_latent).abs().max().item()))
            print(json.dumps(row), flush=True)
            return row

        cold = once('cold', 0)
        warmup = [once('warmup', i + 1) for i in range(5)]
        measured = [once('measured', i + 1) for i in range(30)]
        report = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
                      warmup_calls=5, measured_calls=30, seed=0, threads=4,
                      model_load_ms=load_ms, cold_call=cold, warmup=warmup, measured=measured,
                      warmed={key: stats([r[key] for r in measured]) for key in
                              ('latent_ms', 'xyz_rgba_ms', 'xyz_plus_latent_ms')},
                      protocol='CPU XYZ/RGBA ready -> H2D, preprocess, DINO518, SS1, layout, CUDA synchronize',
                      excluded=['model_load', 'PNG_read', 'SAM', 'occupancy_decode', 'file_save', 'video_export'],
                      latent_shape=[1, 4096, 8], all_outputs_finite=True,
                      fresh_forward_every_call=True, previous_latent_or_pose_loaded=False, GT_loaded=False,
                      rgb_sha256=reference['rgb_sha256'], human_mask_sha256=reference['human_mask_sha256'],
                      historical_clm3_single_call_ms=reference['rgbd_to_latent_ms'],
                      code_sha256=base.sha(__file__), cudnn_benchmark=torch.backends.cudnn.benchmark)
        (RUN / 'report.json').write_text(json.dumps(report, indent=2))
        (RUN / 'DONE').write_text('All 36 fresh forward calls finite; 30 measured calls reported.\n')
        print(json.dumps(dict(event='benchmark_complete', warmed=report['warmed'])), flush=True)


if __name__ == '__main__':
    main()
