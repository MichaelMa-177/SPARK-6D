"""Regenerate scene59 SS1 memory from the exact saved human-selected RGB-D frame."""
import json
import os
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
from scipy.ndimage import binary_erosion

CODE = Path('/home/mmw/tls/clm')
WORK = Path('/data/mmw/clm_disk/clm-mmw-20261020')
ROOT = CODE / 'experiments/first-memory-fresh-sam3d'
SOURCE = CODE / 'vendor/sam3d-runtime/sam-3d-objects'
WEIGHTS = WORK / 'weights/sam3d'
os.environ.update(SAM3D_DINO_REPO=str(CODE / 'vendor/sam3d-runtime/dinov2'),
                  SAM3D_DINO_WEIGHTS=str(WEIGHTS / 'dinov2_vitl14_reg4_pretrain.pth'),
                  ATTN_BACKEND='sdpa', SPARSE_ATTN_BACKEND='sdpa', LIDRA_SKIP_INIT='1')
sys.path[:0] = [str(ROOT), str(SOURCE / 'pytorch3d'), str(SOURCE)]
from data_fast import scene, lift, sha
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch3d.transforms import quaternion_to_matrix
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform


@torch.no_grad()
def main():
    out = ROOT / 'priors'
    out.mkdir(exist_ok=True)
    assert not (out / 'scene59.npz').exists(), 'Do not overwrite fresh memory'
    reference = json.loads((CODE / 'reference/clm3_report.json').read_text())['first_prior']
    record = scene(59, supervision=False)[0]
    assert sha(record['rgb_path']) == sha(ROOT / 'inputs/first_frame.png') == reference['rgb_sha256']
    assert sha(ROOT / 'inputs/mask.png') == reference['human_mask_sha256']
    assert np.array_equal(record['depth'], cv2.imread(str(ROOT / 'inputs/first_depth.png'), -1))
    mask = cv2.imread(str(ROOT / 'inputs/mask.png'), 0) > 0
    torch.set_num_threads(4)
    torch.manual_seed(0)
    cfg = OmegaConf.load(WEIGHTS / 'pipeline.yaml')
    cfg.workspace_dir = str(WEIGHTS)
    cfg.compile_model = False
    cfg.low_vram = True
    cfg.decode_formats = ['mesh']
    cfg.depth_model = {'_target_': 'torch.nn.Identity'}
    cfg.rendering_engine = 'pytorch3d'
    with torch.device('cuda'):
        start = time.perf_counter()
        pipeline = instantiate(cfg)
        torch.cuda.synchronize()
        load_ms = (time.perf_counter() - start) * 1000
        print(json.dumps(dict(event='SS_models_loaded', model_load_ms=load_ms)), flush=True)
        torch.manual_seed(0)
        xyz, _ = lift(record)
        xyz[record['depth'] == 0] = np.nan
        xyz[:, :, :2] *= -1
        rgba = np.concatenate((record['rgb'], (mask.astype(np.uint8) * 255)[:, :, None]), -1)
        torch.cuda.synchronize()
        start = time.perf_counter()
        pm = pipeline._clip_pointmap(torch.from_numpy(xyz).cuda().permute(2, 0, 1),
                                    torch.from_numpy(mask.astype(np.float32)).cuda())
        item = pipeline.preprocess_image(rgba, pipeline.ss_preprocessor, pointmap=pm)
        generator = pipeline.models['ss_generator']
        generator.inference_steps = 1
        generator.no_shortcut = False
        generator.reverse_fn.strength = generator.reverse_fn.strength_pm = 0
        with torch.autocast('cuda', dtype=torch.float16):
            args, kwargs = pipeline.get_condition_input(pipeline.condition_embedders['ss_condition_embedder'],
                                                        item, pipeline.ss_condition_input_mapping)
            shapes = {key: (1, value.pos_emb.shape[0], value.input_layer.in_features)
                      for key, value in generator.reverse_fn.backbone.latent_mapping.items()}
            raw = generator(shapes, 'cuda', *args, **kwargs)
        layout = pipeline.pose_decoder(raw, scene_scale=item.get('pointmap_scale'), scene_shift=item.get('pointmap_shift'))
        torch.cuda.synchronize()
        latent_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.float16):
            logits = pipeline.models['ss_decoder'](raw['shape'].permute(0, 2, 1).contiguous().reshape(1, 8, 16, 16, 16))
        occupied = logits[0, 0].float().cpu().numpy() > 0
        coords = np.argwhere(occupied & ~binary_erosion(occupied, border_value=0))
        assert len(coords) >= 96, 'No usable generated boundary'
        coords = coords[np.linspace(0, len(coords) - 1, 96).round().astype(int)]
        canonical = torch.from_numpy(coords.astype(np.float32) / 64 - .5).cuda()
        transform = compose_transform(layout['scale'], quaternion_to_matrix(layout['rotation']), layout['translation'])
        points = transform.transform_points(canonical[None])[0].cpu().numpy()
        points[:, :2] *= -1
        coarse = raw['shape'].float().cpu().numpy()
        linear = (coords[:, 0] // 4) * 256 + (coords[:, 1] // 4) * 16 + coords[:, 2] // 4
        assert coarse.shape == (1, 4096, 8) and np.isfinite(coarse).all() and np.isfinite(points).all()
        decode_ms = (time.perf_counter() - start) * 1000
        np.savez_compressed(out / 'scene59.npz', points=points, features=coarse[0, linear], coarse_shape=coarse,
                            boundary_coords=coords, **{f'layout_{k}': layout[k].float().cpu().numpy()
                                                     for k in ('rotation', 'translation', 'scale')})
        report = dict(scene=59, frame=record['frame'], mask_source='saved_human_mask_7f37ebe6a09819a6',
                      rgb_sha256=sha(record['rgb_path']), human_mask_sha256=sha(ROOT / 'inputs/mask.png'),
                      raw_latent_shape=list(coarse.shape), prior_points=96, source='fresh_SS1_DINO518_new_server',
                      geometry='predicted SS occupancy boundary, NOT measured backside/mesh/SLAT',
                      model_load_ms=load_ms, rgbd_to_latent_ms=latent_ms, latent_decode_ms=decode_ms,
                      fresh_inference=True, previous_prior_loaded=False, GT_loaded=False,
                      gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                      weight_sha256={n: sha(WEIGHTS / n) for n in ('ss_generator.ckpt', 'ss_decoder.ckpt')},
                      file_sha256=sha(out / 'scene59.npz'))
        (out / 'scene59.json').write_text(json.dumps(report, indent=2))
        (out / 'DONE').write_text('First-frame memory regenerated from RGB-D on new server.\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
