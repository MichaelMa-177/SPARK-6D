"""Three fresh trained-weight frames; explicitly not full accuracy acceptance."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
CODE=Path('/home/mmw/tls/clm')
WORK=Path('/data/mmw/clm_disk/clm-mmw-20261020')
SOURCE=CODE/'experiments/first-memory-fast-refine'
sys.path.insert(0,str(SOURCE))
from data_fast import scene,sha
from fast_model import FastRefiner
from evaluate_fast import run

torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cudnn.benchmark=True
checkpoint=SOURCE/'training/last.pt'
reference=json.loads((CODE/'reference/clm3_report.json').read_text())
assert sha(checkpoint)==reference['checkpoint_sha256']
assert sha(SOURCE/'inputs/mask.png')==reference['first_prior']['human_mask_sha256']
records=scene(59,False)[:3]
assert len(records)==3 and not any('gt' in r or 'mask' in r for r in records)
assert sha(Path(records[0]['rgb_path']))==sha(SOURCE/'inputs/first_frame.png')
model=FastRefiner().cuda().eval()
model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False)['model'])
with torch.inference_mode():
    run(model,records,'full')  # Warm kernels, discard every pose/state.
    poses,timing,diagnostics,init,points=run(model,records,'full')
assert np.isfinite(poses).all() and np.isfinite(points).all()
assert np.allclose(np.linalg.det(poses[:,:3,:3]),1,atol=1e-4)
out=WORK/'runs/migration';out.mkdir(exist_ok=True)
np.savez_compressed(out/'smoke_fresh_predictions.npz',poses=poses,frames=[r['frame'] for r in records])
report=dict(result='PASS',role='trained-model execution smoke only, NOT full accuracy acceptance',
            gpu=torch.cuda.get_device_name(0),torch=torch.__version__,frames=[r['frame'] for r in records],
            pose_shape=list(poses.shape),checkpoint_sha256=sha(checkpoint),saved_human_prompt_unchanged=True,
            fresh_per_frame_inference=True,GT_annotations_loaded=False,previous_saved_poses_loaded=False,
            rigid_transforms_finite=True,first_frame_prior='original migrated SS1 output',
            model_reference_init_ms=init,post_warmup_timing=timing,diagnostics=diagnostics,
            full_75_frame_accuracy_pending=True,new_self_data_first_frame_prompt_pending=True)
(out/'smoke_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
