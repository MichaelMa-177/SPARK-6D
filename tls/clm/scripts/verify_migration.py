"""Verify copied artifacts before a single validation frame is inferred."""
import hashlib
import json
from pathlib import Path
import os
ROOT=Path('/home/mmw/tls/clm')
WORK=Path('/data/mmw/clm_disk/clm-mmw-20261020')
SOURCE=ROOT/'experiments/first-memory-fast-refine'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
manifest=json.loads((ROOT/'reference/data_manifest.json').read_text())
data=Path(os.environ['CLM_DATA_ROOT'])
assert data==WORK/'data/bop/ycbv/test'
for item in manifest['files']:
    path=data/item['path']
    assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],str(path)
reference=json.loads((ROOT/'reference/clm3_report.json').read_text())
assert sha(SOURCE/'training/last.pt')==reference['checkpoint_sha256']
assert sha(SOURCE/'inputs/first_frame.png')==reference['first_prior']['rgb_sha256']
assert sha(SOURCE/'inputs/mask.png')==reference['first_prior']['human_mask_sha256']
for number in (50,54,59):
    record=json.loads((SOURCE/f'priors/scene{number}.json').read_text())
    assert sha(SOURCE/f'priors/scene{number}.npz')==record['file_sha256']
assert sha(data/'000059/rgb/000001.png')==sha(SOURCE/'inputs/first_frame.png')
assert sha(data/'000059/depth/000001.png')==sha(SOURCE/'inputs/first_depth.png')
row=dict(data_files=len(manifest['files']),rgb_frames=225,all_source_hashes_match=True,
         checkpoint_sha256=reference['checkpoint_sha256'],original_human_prompt_unchanged=True,
         first_frame_prior='migrated original SS1 output; no cached per-frame poses')
out=WORK/'runs/migration';out.mkdir(exist_ok=True)
(out/'input_audit.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
