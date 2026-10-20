"""Download the first-frame SS/DINO weights; require exact CLM3 SHA256 matches."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import quote

WORK = Path('/data/mmw/clm_disk/clm-mmw-20261020')
OUT = WORK / 'weights/sam3d'
RUN = WORK / 'runs/sam3d-migration'
assert subprocess.check_output(['findmnt', '-rn', '-T', str(WORK), '-o', 'SOURCE'], text=True).strip() == '/dev/sda1'
os.umask(0o077)
OUT.mkdir(parents=True, exist_ok=True)
RUN.mkdir(parents=True, exist_ok=True)
request = RUN / 'download_request.json'
if sys.argv[1:] == ['--prepare']:
    spec = json.load(sys.stdin)
    assert len(spec['files']) == 3
    assert not (RUN / 'DONE').exists()
    for item in spec['files']:
        assert item['name'] in ('ss_generator.ckpt', 'ss_decoder.ckpt', 'dinov2_vitl14_reg4_pretrain.pth')
        assert re.fullmatch('[a-f0-9]{64}', item['sha256'])
        assert item['url'].startswith('https://')
    request.write_text(json.dumps(spec))
    print('Prepared three downloads; no account token stored.')
    sys.exit(0)

lock = (RUN / 'download.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not (RUN / 'DONE').exists()
(RUN / 'STARTED').write_text('First-frame SS generator/decoder and DINO; not all SLAT weights.\n')
spec = json.loads(request.read_text())
auth = json.loads(Path('/home/mmw/tls/config/local_auth.json').read_text())
proxy = 'http://' + quote(auth['username'], safe='') + ':' + quote(auth['password'], safe='') + '@127.0.0.1:18089'
env = os.environ.copy()
for name in ('http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'):
    env[name] = proxy
env['NO_PROXY'] = env['no_proxy'] = ''
lines = []
for item in spec['files']:
    lines.extend([item['url'], f' dir={OUT}', f' out={item["name"]}', f' checksum=sha-256={item["sha256"]}'])
input_file = RUN / 'aria2.input'
input_file.write_text('\n'.join(lines) + '\n')
args = ['aria2c', '-c', '-x16', '-s16', '-j2', '--min-split-size=1M', '--file-allocation=none',
        '--auto-file-renaming=false', '--allow-overwrite=false', '--check-integrity=true',
        '--summary-interval=20', '--enable-color=false', '--max-tries=10', '--retry-wait=5',
        '--connect-timeout=15', '--timeout=60', '--input-file=' + str(input_file)]
print(json.dumps(dict(event='download_started', total_bytes=sum(x['bytes'] for x in spec['files']))), flush=True)
with subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as child:
    for line in child.stdout:
        print(re.sub(r'https?://\S+', '<download-url>', line.rstrip()), flush=True)
    assert child.wait() == 0, 'aria2 failed; inspect redacted download.log'
verified = []
for item in spec['files']:
    path = OUT / item['name']
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    assert path.stat().st_size == item['bytes'] and h.hexdigest() == item['sha256'], item['name']
    verified.append({key: item[key] for key in ('name', 'bytes', 'sha256')})
(RUN / 'weights_verified.json').write_text(json.dumps(dict(files=verified, all_match_clm3=True), indent=2))
(RUN / 'DONE').write_text('All three weight files match CLM3 bytes and SHA256. Source/environment separately verified.\n')
request.unlink()
input_file.unlink()
print('DONE: all three weights verified against CLM3.', flush=True)
