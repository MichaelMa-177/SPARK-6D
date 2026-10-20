"""Add SS inference dependencies inside the private CLM venv only."""
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import quote

WORK = Path('/data/mmw/clm_disk/clm-mmw-20261020')
PYTHON = WORK / 'envs/clm/bin/python'
RUN = WORK / 'runs/sam3d-setup'
assert subprocess.check_output(['findmnt', '-rn', '-T', str(WORK), '-o', 'SOURCE'], text=True).strip() == '/dev/sda1'
RUN.mkdir(exist_ok=True)
lock = (RUN / 'setup.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
(RUN / 'STARTED').write_text('SS inference dependency setup\n')
env = os.environ.copy()
for name in ('PYTHONPATH', 'PYTHONHOME', 'PIP_TARGET', 'PIP_DEST'):
    env.pop(name, None)
env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PIP_CONFIG_FILE='/dev/null',
           PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_CACHE_DIR=str(WORK / 'cache/pip'),
           TMPDIR=str(WORK / 'tmp'), XDG_CACHE_HOME=str(WORK / 'cache'))
auth = json.loads(Path('/home/mmw/tls/config/local_auth.json').read_text())
proxy = 'http://' + quote(auth['username'], safe='') + ':' + quote(auth['password'], safe='') + '@127.0.0.1:18089'
for name in ('http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'):
    env[name] = proxy
env['NO_PROXY'] = env['no_proxy'] = ''
constraints = RUN / 'constraints.txt'
constraints.write_text('torch==2.8.0+cu128\nnumpy==1.26.4\n')
def pip(*args):
    command = [str(PYTHON), '-B', '-m', 'pip', *args]
    with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as child:
        for line in child.stdout:
            print(re.sub(r'https?://\S+', '<package-url>', line.rstrip()), flush=True)
        assert child.wait() == 0, 'Dependency install failed; see setup.log'

pip('install', '-c', str(constraints), '--index-url', 'https://download.pytorch.org/whl/cu128', 'torchvision==0.23.0')
pip('install', '-c', str(constraints), '--index-url', 'https://pypi.org/simple',
    'hydra-core==1.3.2', 'loguru==0.7.2', 'einops==0.8.1', 'safetensors',
    'timm==1.0.28', 'lightning==2.6.5', 'torchmetrics==1.9.0', 'addict', 'easydict',
    'fvcore', 'iopath', 'trimesh', 'h5py', 'scikit-image==0.23.1', 'roma', 'joblib',
    'optree', 'astor')
pip('check')
(RUN / 'BASE_DEPS_DONE').write_text('Base dependencies installed; actual SS import/inference still required.\n')
