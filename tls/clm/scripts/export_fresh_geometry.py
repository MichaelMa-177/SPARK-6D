"""Decode the saved new-server SS latent; export measured and inferred clouds separately."""
import json
import time
import numpy as np
import torch
import cv2
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation
import prepare_fresh_prior as base
from sam3d_objects.pipeline.inference_pipeline import InferencePipeline
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
from pytorch3d.transforms import quaternion_to_matrix


@torch.no_grad()
def main():
    out = base.WORK / 'runs/fresh-geometry-preview'
    out.mkdir(exist_ok=True)
    assert not (out / 'DONE').exists(), 'Preserve completed geometry export'
    (out / 'STARTED').write_text('Exact saved latent decode, no SS resampling\n')
    source = base.ROOT / 'priors/scene59.npz'
    source_report = json.loads((base.ROOT / 'priors/scene59.json').read_text())
    assert base.sha(source) == source_report['file_sha256']
    record = base.scene(59, supervision=False)[0]
    assert base.sha(record['rgb_path']) == source_report['rgb_sha256']
    assert base.sha(base.ROOT / 'inputs/mask.png') == source_report['human_mask_sha256']
    assert np.array_equal(record['depth'], cv2.imread(str(base.ROOT / 'inputs/first_depth.png'), -1))
    mask = cv2.imread(str(base.ROOT / 'inputs/mask.png'), 0) > 0
    xyz, valid = base.lift(record)
    observed, colors = xyz[mask & valid], record['rgb'][mask & valid]
    np.savez_compressed(out / 'measured.npz', xyz_m=observed, rgb=colors)
    decoder_path = base.WEIGHTS / 'ss_decoder.ckpt'
    assert base.sha(decoder_path) == source_report['weight_sha256']['ss_decoder.ckpt']
    torch.set_num_threads(4)
    decoder = InferencePipeline.instantiate_and_load_from_pretrained(
        None, base.OmegaConf.load(base.WEIGHTS / 'ss_decoder.yaml'), str(decoder_path),
        state_dict_key=None, device='cuda')
    with np.load(source, allow_pickle=False) as packet:
        coarse = torch.from_numpy(packet['coarse_shape'].copy()).cuda()
        layout = {k: torch.from_numpy(packet['layout_' + k].copy()).cuda()
                  for k in ('scale', 'rotation', 'translation')}
        saved_anchors = packet['points'].copy()
    samples = []
    with torch.autocast('cuda', dtype=torch.float16):
        for _ in range(2):
            decoder(coarse.permute(0, 2, 1).contiguous().reshape(1, 8, 16, 16, 16))
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.perf_counter()
            logits = decoder(coarse.permute(0, 2, 1).contiguous().reshape(1, 8, 16, 16, 16))
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
    grid = logits[0, 0].float().cpu().numpy()
    assert grid.shape == (64, 64, 64) and np.isfinite(grid).all()
    occupied = grid > 0
    coords = np.argwhere(occupied & ~binary_erosion(occupied, border_value=0))
    canonical = torch.from_numpy(coords.astype(np.float32) / 64 - .5).cuda()
    transform = compose_transform(layout['scale'], quaternion_to_matrix(layout['rotation']), layout['translation'])
    generated = transform.transform_points(canonical[None])[0].cpu().numpy()
    generated[:, :2] *= -1
    q = layout['rotation'][0].cpu().numpy()
    expected = (coords / 64 - .5) * layout['scale'].cpu().numpy()
    expected = expected @ Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix() + layout['translation'].cpu().numpy()
    expected[:, :2] *= -1
    assert np.allclose(expected, generated, atol=1e-6)
    assert np.isfinite(generated).all() and len(generated) > 96
    blue = np.tile(np.array([103, 188, 216], np.uint8), (len(generated), 1))
    np.savez_compressed(out / 'generated.npz', xyz_m=generated, rgb=blue)
    np.savez_compressed(out / 'overlay.npz', xyz_m=np.concatenate((observed, generated)),
                        rgb=np.concatenate((colors, blue)))
    np.savez_compressed(out / 'occupancy.npz', logits=grid, occupied=occupied, boundary_coords=coords)
    np.savez_compressed(out / 'memory_anchors.npz', xyz_m=saved_anchors,
                        rgb=np.tile(np.array([103, 188, 216], np.uint8), (len(saved_anchors), 1)))
    report = dict(source_latent_sha256=base.sha(source), source_latent_shape=list(coarse.shape),
                  first_frame_report=source_report, exact_saved_latent_decoded=True,
                  new_ss_sampling=False, slat_used=False, ICP_or_alignment_used=False,
                  coordinate_frame='RGB camera: X right, Y down, Z forward; meters',
                  measured_points=len(observed), generated_boundary_points=len(generated),
                  generated_memory_anchors=len(saved_anchors), mask_pixels=int(mask.sum()),
                  invalid_depth_inside_mask=int((mask & ~valid).sum()),
                  occupied_voxels=int(occupied.sum()), occupancy_threshold_logit=0,
                  independent_transform_max_error_m=float(np.abs(expected - generated).max()),
                  decoder_ms=dict(p50=float(np.median(samples)), max=max(samples), samples=samples,
                                  warmups=2, repeats=5),
                  display='Measured RGB from registered first frame; inferred boundary has uniform display color, no inferred texture.',
                  output_sha256={name: base.sha(out / name) for name in
                                 ('measured.npz', 'generated.npz', 'overlay.npz', 'occupancy.npz', 'memory_anchors.npz')})
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    (out / 'DONE').write_text('Saved latent decoded and independently transformed; no SS resampling.\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
