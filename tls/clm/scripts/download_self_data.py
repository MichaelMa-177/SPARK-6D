"""Pinned self_data-only download, publisher SHA256 + safe ZIP/PNG checks."""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import urllib.request
from urllib.parse import quote
import zipfile

BASE=Path('/home/mmw/tls')
WORK=Path('/data/mmw/clm_disk/clm-mmw-20261020')
OUT=WORK/'data/click-to-model/self_data'
REPO='Mingwei000/click-to-model-data-online'
REV='cbae3dc787dd7c4a8d2149f725356b06a4c356ad'
assert subprocess.check_output(['findmnt','-rn','-T',str(WORK),'-o','SOURCE'],text=True).strip()=='/dev/sda1'
assert WORK.stat().st_uid==os.getuid() and shutil.disk_usage(WORK).free>10*1024**3
os.umask(0o077)
OUT.mkdir(parents=True,exist_ok=True);(OUT/'control').mkdir(exist_ok=True)
lock=open(OUT/'control/download.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (OUT/'DONE').exists():print('Already verified complete');sys.exit(0)
(OUT/'STARTED').write_text(REV+'\n')
env=os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(WORK/'tmp'),XDG_CACHE_HOME=str(WORK/'cache'),
           MPLCONFIGDIR=str(WORK/'cache/matplotlib'))
credentials=json.loads((BASE/'config/local_auth.json').read_text())
proxy='http://'+quote(credentials['username'],safe='')+':'+quote(credentials['password'],safe='')+'@127.0.0.1:18089'
for key in ('http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):env[key]=proxy
env['NO_PROXY']=env['no_proxy']=''
opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':proxy,'https':proxy}))
def url(path):return f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/'+quote(path,safe='/')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

body=opener.open(url('self_data/SHA256SUMS'),timeout=60).read()
assert hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()=='2a3386897fd624aa9e9ba9c7563f4bfdc58f6591'
checks={}
for line in body.decode().splitlines():
    digest,name=line.split(None,1);name=name.strip().lstrip('*')
    relative=PurePosixPath(name)
    assert relative.parts[0]=='self_data' and '..' not in relative.parts and not relative.is_absolute()
    checks[name]=digest
assert len(checks)==17
(OUT/'hf/self_data').mkdir(parents=True,exist_ok=True)
(OUT/'hf/self_data/SHA256SUMS').write_bytes(body)
lines=[]
for name,digest in checks.items():
    target=OUT/'hf'/name;target.parent.mkdir(parents=True,exist_ok=True)
    lines.extend([url(name),f' dir={target.parent}',f' out={target.name}',f' checksum=sha-256={digest}'])
inputfile=OUT/'control/aria2.input';inputfile.write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(event='download_start',revision=REV,files=18,scope='self_data only; both captures, all original archives + viewer derivatives')),flush=True)
subprocess.run(['aria2c','--continue=true','--max-connection-per-server=16','--split=16','--max-concurrent-downloads=2',
                '--min-split-size=1M','--file-allocation=none','--auto-file-renaming=false','--allow-overwrite=false',
                '--check-integrity=true','--summary-interval=20','--enable-color=false','--console-log-level=notice',
                '--max-tries=8','--retry-wait=5','--connect-timeout=15','--timeout=60','--input-file='+str(inputfile)],env=env,check=True)
for name,digest in checks.items():assert sha(OUT/'hf'/name)==digest,name
manifest=json.loads((OUT/'hf/self_data/manifest.json').read_text())
import cv2
report=dict(repo_id=REPO,revision=REV,downloaded_files=18,publisher_sha256_verified=17,
            checksum_list_git_blob_verified=True,sequences=[],no_segmentation_or_pose_inference=True)
for item in manifest['sequences']:
    name=item['sequence'];archive=OUT/'hf'/item['archive']
    assert sha(archive)==item['archive_sha256'] and archive.stat().st_size==item['archive_bytes']
    final=OUT/'raw'/name;stage=OUT/'raw'/(name+'.extracting')
    if not final.exists():
        stage.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            entries=z.infolist();files=[x for x in entries if not x.is_dir()]
            assert len(files)==item['original_files']
            strip_root=all(PurePosixPath(x.filename).parts[0]==name for x in files)
            seen=set()
            for info in entries:
                p=PurePosixPath(info.filename)
                assert not p.is_absolute() and '..' not in p.parts and not stat.S_ISLNK(info.external_attr>>16)
                parts=p.parts[1:] if strip_root else p.parts
                if not parts:continue
                target=stage.joinpath(*parts)
                assert str(target.resolve()).startswith(str(stage.resolve())+'/')
                if info.is_dir():target.mkdir(parents=True,exist_ok=True);continue
                assert parts not in seen;seen.add(parts);target.parent.mkdir(parents=True,exist_ok=True)
                # Reading ZipExtFile to EOF verifies each member's CRC.
                if target.exists():
                    with z.open(info) as src:assert src.read()==target.read_bytes(),str(target)
                else:
                    part=target.with_name(target.name+'.part')
                    with z.open(info) as src,part.open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
                    assert part.stat().st_size==info.file_size
                    part.replace(target)
        stage.rename(final)
    rgb=sorted((final/'rgb').glob('*.png'));depth=sorted((final/'depth').glob('*.png'))
    assert len(rgb)==len(depth)==item['frame_pairs']
    assert [p.name for p in rgb]==[p.name for p in depth]
    for rp,dp in zip(rgb,depth):
        a=cv2.imread(str(rp),cv2.IMREAD_UNCHANGED);b=cv2.imread(str(dp),cv2.IMREAD_UNCHANGED)
        assert a is not None and a.shape==(480,640,3) and a.dtype.name=='uint8'
        assert b is not None and b.shape==(480,640) and b.dtype.name=='uint16'
    actual=[p for p in final.rglob('*') if p.is_file()]
    assert len(actual)==item['original_files'] and sum(p.stat().st_size for p in actual)==item['original_bytes']
    camera=json.loads((final/'camera_info.json').read_text())
    assert camera['stream']['depth_file_unit']=='millimeter'
    report['sequences'].append(dict(sequence=name,raw_directory=str(final),frames=len(rgb),original_files=len(actual),
                                   archive_sha256=item['archive_sha256'],all_RGB_depth_images_decoded=True,
                                   depth_dtype='uint16',depth_to_meters=.001,shape=[480,640],
                                   camera_duration_s=item['camera_duration_s'],zip_crc_verified=True))
    print(json.dumps(report['sequences'][-1]),flush=True)
report['frame_pairs']=sum(x['frames'] for x in report['sequences'])
report['original_files']=sum(x['original_files'] for x in report['sequences'])
report['original_bytes']=manifest['original_bytes']
(OUT/'download_report.json').write_text(json.dumps(report,indent=2))
(OUT/'DONE').write_text('Both captures downloaded, SHA256/ZIP CRC/2054 RGB-D pairs verified\n')
print(json.dumps(report),flush=True)
