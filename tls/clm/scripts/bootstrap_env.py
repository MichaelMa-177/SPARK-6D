"""Install only into our new workspace; never change system Python or old files."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from urllib.parse import quote

BASE=Path('/home/mmw/tls')
WORK=Path('/data/mmw/clm_disk/clm-mmw-20261020')
ENV=WORK/'envs/clm'
assert WORK.is_dir() and WORK.stat().st_uid==os.getuid()
assert subprocess.check_output(['findmnt','-rn','-T',str(WORK),'-o','SOURCE'],text=True).strip()=='/dev/sda1'
assert shutil.disk_usage(WORK).free>30*1024**3,'Need at least 30GiB free before environment setup'
for name in ('cache/pip','cache/matplotlib','cache/torch','cache/cuda','cache/triton','tmp','downloads/bootstrap','runs/bootstrap'):
    (WORK/name).mkdir(parents=True,exist_ok=True)
lock=open(WORK/'runs/bootstrap/setup.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (WORK/'runs/bootstrap/DONE').exists(),'Completed environment must not be overwritten'
(WORK/'runs/bootstrap/STARTED').write_text('Installing private clm environment\n')
env=os.environ.copy()
for name in ('PIP_TARGET','PIP_DEST','PYTHONPATH','PYTHONHOME'):env.pop(name,None)
env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
           PIP_CONFIG_FILE='/dev/null',PIP_DISABLE_PIP_VERSION_CHECK='1',
           PIP_CACHE_DIR=str(WORK/'cache/pip'),TMPDIR=str(WORK/'tmp'),
           XDG_CACHE_HOME=str(WORK/'cache'),MPLCONFIGDIR=str(WORK/'cache/matplotlib'),
           TORCH_HOME=str(WORK/'cache/torch'),CUDA_CACHE_PATH=str(WORK/'cache/cuda'),
           TRITON_CACHE_DIR=str(WORK/'cache/triton'),CUDA_VISIBLE_DEVICES='0')
subprocess.run([str(BASE/'tls'),'start-proxy'],check=True)
credentials=json.loads((BASE/'config/local_auth.json').read_text())
proxy='http://'+quote(credentials['username'],safe='')+':'+quote(credentials['password'],safe='')+'@127.0.0.1:18089'
for name in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy','ALL_PROXY','all_proxy'):env[name]=proxy
env['NO_PROXY']=env['no_proxy']=''
with socket.create_connection(('127.0.0.1',18089),timeout=5):pass
def run(args,**kwargs):subprocess.run(args,env=env,check=True,**kwargs)
if not (ENV/'pyvenv.cfg').exists():run([sys.executable,'-m','venv','--without-pip',str(ENV)])
python=str(ENV/'bin/python')
prefix=subprocess.check_output([python,'-c','import sys;print(sys.prefix)'],env=env,text=True).strip()
assert prefix==str(ENV),'Refusing to install into a different Python prefix'
run([sys.executable,'-m','pip','download','--no-deps','--no-cache-dir','--index-url','https://pypi.org/simple',
     '--dest',str(WORK/'downloads/bootstrap'),'pip==24.3.1'])
wheel=next((WORK/'downloads/bootstrap').glob('pip-24.3.1-*.whl'))
seed_env={**env,'PYTHONPATH':str(wheel)}
subprocess.run([python,'-m','pip','install','--no-index','--no-deps','--no-cache-dir',str(wheel)],env=seed_env,check=True)
run([python,'-m','pip','install','--no-cache-dir','torch==2.8.0','--index-url','https://download.pytorch.org/whl/cu128'])
run([python,'-m','pip','install','--no-cache-dir','--index-url','https://pypi.org/simple',
     'numpy==1.26.4','scipy==1.14.1','opencv-python-headless==4.10.0.84','matplotlib==3.9.4'])
run([python,'-m','pip','check'])
probe='''import json,torch,numpy,cv2,scipy,matplotlib
assert torch.cuda.is_available()
x=torch.randn(2,3,32,32,device="cuda",requires_grad=True)
m=torch.nn.Conv2d(3,8,3).cuda();m(x).square().mean().backward()
a=torch.randn(1,3,3,device="cuda");torch.linalg.svd(a);torch.cuda.synchronize()
print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),numpy=numpy.__version__,cv2=cv2.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,cuda_conv_backward_svd=True)))'''
result=subprocess.check_output([python,'-c',probe],env=env,text=True)
(WORK/'runs/bootstrap/environment.json').write_text(result)
freeze=subprocess.check_output([python,'-m','pip','freeze'],env=env,text=True)
(WORK/'runs/bootstrap/requirements.lock.txt').write_text(freeze)
(WORK/'runs/bootstrap/DONE').write_text('Private CUDA environment verified\n')
print(result,flush=True)
