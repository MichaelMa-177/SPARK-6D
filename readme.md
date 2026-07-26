# fpBoost: Exact-Stable, High-FPS FoundationPose Tracking

fpBoost is an engineering extension of
[NVIDIA FoundationPose](https://github.com/NVlabs/FoundationPose) for faster
model-based 6D object tracking. It combines adaptive SE(3) Kalman prediction
with exact-equivalent GPU and pipeline optimizations. The expensive neural
refiner is skipped only on low-risk frames; motion, uncertainty, correction,
and optical-flow guards immediately restore refinement when needed.

> 中文简介：fpBoost 在 FoundationPose 原版上加入 SE(3) 卡尔曼预测、自适应神经
> 精修、光流保护和端到端异步 I/O，并对渲染、裁剪和张量传输热路径做等价优化。
> 在仓库自带的 737 帧 YCB mustard demo 上，最终版本的跟踪结果与稳定版
> Kalman 基准逐文件完全一致，同时 tracking FPS 从 31.35 提升到
> 44.33–44.71。

This is an independent fork, not an official NVIDIA release. The original
FoundationPose authors, paper, license, setup instructions, and citations are
retained below.

## What changed from upstream FoundationPose

### 1. Adaptive SE(3) Kalman tracking

- Added a constant-body-velocity error-state filter in `pose_kalman.py`.
- Translation and SO(3) rotation-vector residuals are filtered directly on
  SE(3), avoiding Euler-angle wrapping and gimbal-lock artifacts.
- A Joseph-form covariance update keeps the covariance numerically stable.
- Refinement decisions use motion, covariance, correction, periodic refresh,
  and image-observation gates instead of blindly skipping fixed frames.
- A lightweight ROI optical-flow guard forces neural refinement whenever
  observed image motion disagrees with the Kalman prediction.
- Any high-motion or high-uncertainty frame falls back to the original neural
  refiner; the default interval skips at most one refinement frame.

### 2. Exact-equivalent tracking hot paths

- Cached invariant camera, crop, mesh, depth-grid, homogeneous-coordinate,
  and OpenGL transform data.
- Removed redundant allocations, zero initialization, auxiliary network
  outputs, normal/depth work, and host/device pose copies from the tracking
  loop.
- Added direct CUDA tensor paths between rendering, preprocessing, and pose
  refinement.
- Preserved cache invalidation across device changes and multi-GPU model
  placement.
- Determinism takes priority over a marginal benchmark gain:
  `--cudnn_benchmark` is intentionally ignored in this path because it can
  change the numerical result.

### 3. End-to-end pipeline improvements

- `--fast_io` prefetches image/depth input and grayscale preparation while the
  GPU processes the current frame.
- Pose text files and optional PNG debug renders use a bounded asynchronous
  writer, so disk I/O does not stall tracking.
- Same-size image resize work is avoided while array contiguity is preserved.
- Debug rendering remains lossless; `--png_compression 1` reduces encoder
  latency without changing decoded pixels.
- Benchmark reports include decision counts, latency percentiles,
  `tracking_fps`, and `end_to_end_fps`.

### 4. Reproducibility and regression checks

- Added unit tests for the Kalman model, SE(3) behavior, and covariance safety.
- Added `tools/compare_pose_tracks.py` for frame-by-frame pose, temporal
  stability, mode-decision, and timing comparisons.
- `--require-exact` fails validation unless every output pose file is
  byte-for-byte identical to the selected golden run.

## Measured result

Local benchmark on an NVIDIA GeForce RTX 3060 Laptop GPU with
PyTorch 2.6.0+cu124, using the complete 737-frame bundled YCB mustard demo:

| Configuration | Tracking FPS | End-to-end FPS | Result |
| --- | ---: | ---: | --- |
| Local always-refine baseline (Kalman off) | 23.11 | 14.58 | Neural refinement every tracking frame |
| Stable adaptive-Kalman baseline | 31.35 | 18.12 | Golden stability/accuracy run |
| **fpBoost final, run 1** | **44.33** | **29.94** | Exact match to golden poses |
| **fpBoost final, run 2** | **44.71** | **30.39** | Exact match to golden poses |
| fpBoost + all debug renders | 32.03 | 15.69 | 737 lossless PNGs and pose files saved |

Compared with the stable adaptive-Kalman baseline, the final tracking loop is
41.4–42.6% faster. Compared with the local always-refine baseline, it is
91.8–93.4% faster on this machine.

Regression-equivalence and temporal-stability checks on all 737 demo frames:

- every saved pose text file is byte-for-byte identical to the stable
  adaptive-Kalman golden run;
- two final reruns have the same 736 frame decisions: 279 predicted, 454
  refined, and 3 optical-flow-guard refinements;
- translation deviation is 0 mm;
- temporal translation/rotation acceleration metrics are unchanged;
- all 737 decoded debug-render images are pixel-identical to the prior
  rendering path.

`tracking_fps` measures the `track_one` segment. `end_to_end_fps` additionally
includes first-frame registration, input waits, and pose/render output flush,
but excludes one-time model loading. Results above describe this specific
hardware and demo, not a universal FPS or accuracy guarantee. On other
cameras, objects, resolutions, or motion regimes, validate against an
always-refine run before changing the stability-first defaults.

## Run fpBoost

Activate the requested Conda environment after completing the setup below:

```bash
conda activate foundationpose
```

Fast headless tracking:

```bash
python run_demo.py --use_kalman --fast_io --debug 0 --quiet --no_display \
  --debug_dir debug/fpboost \
  --benchmark_report debug/fpboost_report.json
```

Tracking while retaining every debug-render image:

```bash
python run_demo.py --use_kalman --fast_io --debug 1 --quiet --no_display \
  --save_render --png_compression 1 \
  --debug_dir debug/fpboost_rendered \
  --benchmark_report debug/fpboost_rendered_report.json
```

Rendered PNGs are saved in `<debug_dir>/track_vis`, and poses are saved in
`<debug_dir>/ob_in_cam`.

Strictly compare a candidate run with a golden pose track:

```bash
python tools/compare_pose_tracks.py \
  --baseline debug/kalman_golden/ob_in_cam \
  --candidate debug/fpboost/ob_in_cam \
  --baseline-timing debug/kalman_golden_report.json \
  --candidate-timing debug/fpboost_report.json \
  --require-exact
```

Run the regression tests:

```bash
python -m unittest discover -s tests -v
```

The optimization was inspired by the adaptive prediction/refinement direction
of [FoundationPose++](https://github.com/teal024/FoundationPose-plus-plus),
while the implementation here was derived from and validated against
FoundationPose's actual tracking, rendering, and I/O hot paths.

---

# FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects
[[Paper]](https://arxiv.org/abs/2312.08344) [[Website]](https://nvlabs.github.io/FoundationPose/)

This is the official implementation of our paper to be appeared in CVPR 2024 (Highlight)

Contributors: Bowen Wen, Wei Yang, Jan Kautz, Stan Birchfield

We present FoundationPose, a unified foundation model for 6D object pose estimation and tracking, supporting both model-based and model-free setups. Our approach can be instantly applied at test-time to a novel object without fine-tuning, as long as its CAD model is given, or a small number of reference images are captured. We bridge the gap between these two setups with a neural implicit representation that allows for effective novel view synthesis, keeping the downstream pose estimation modules invariant under the same unified framework. Strong generalizability is achieved via large-scale synthetic training, aided by a large language model (LLM), a novel transformer-based architecture, and contrastive learning formulation. Extensive evaluation on multiple public datasets involving challenging scenarios and objects indicate our unified approach outperforms existing methods specialized for each task by a large margin. In addition, it even achieves comparable results to instance-level methods despite the reduced assumptions.


<img src="assets/intro.jpg" width="70%">

**🤖 For ROS version, please check [Isaac ROS Pose Estimation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation), which enjoys TRT fast inference and C++ speed up.**

\
**🥇 No. 1 on the world-wide [BOP leaderboard](https://bop.felk.cvut.cz/leaderboards/pose-estimation-unseen-bop23/core-datasets/) (as of 2024/03) for model-based novel object pose estimation.**
<img src="assets/bop.jpg" width="80%">

## Demos

Robotic Applications:

https://github.com/NVlabs/FoundationPose/assets/23078192/aa341004-5a15-4293-b3da-000471fd74ed


AR Applications:

https://github.com/NVlabs/FoundationPose/assets/23078192/80e96855-a73c-4bee-bcef-7cba92df55ca


Results on YCB-Video dataset:

https://github.com/NVlabs/FoundationPose/assets/23078192/9b5bedde-755b-44ed-a973-45ec85a10bbe



# Bibtex
```bibtex
@InProceedings{foundationposewen2024,
author        = {Bowen Wen, Wei Yang, Jan Kautz, Stan Birchfield},
title         = {{FoundationPose}: Unified 6D Pose Estimation and Tracking of Novel Objects},
booktitle     = {CVPR},
year          = {2024},
}
```

If you find the model-free setup useful, please also consider cite:

```bibtex
@InProceedings{bundlesdfwen2023,
author        = {Bowen Wen and Jonathan Tremblay and Valts Blukis and Stephen Tyree and Thomas M\"{u}ller and Alex Evans and Dieter Fox and Jan Kautz and Stan Birchfield},
title         = {{BundleSDF}: {N}eural 6-{DoF} Tracking and {3D} Reconstruction of Unknown Objects},
booktitle     = {CVPR},
year          = {2023},
}
```

# Data prepare


1) Download all network weights from [here](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing) and put them under the folder `weights/`. For the refiner, you will need `2023-10-28-18-33-37`. For scorer, you will need `2024-01-11-20-02-45`.

1) [Download demo data](https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP?usp=sharing) and extract them under the folder `demo_data/`

1) [Optional] Download our large-scale training data: ["FoundationPose Dataset"](https://drive.google.com/drive/folders/1s4pB6p4ApfWMiMjmTXOFco8dHbNXikp-?usp=sharing)

1) [Optional] Download our preprocessed reference views [here](https://drive.google.com/drive/folders/1PXXCOJqHXwQTbwPwPbGDN9_vLVe0XpFS?usp=sharing) in order to run model-free few-shot version.

# Env setup option 1: docker (recommended)
  ```
  cd docker/
  docker pull wenbowen123/foundationpose && docker tag wenbowen123/foundationpose foundationpose  # Or to build from scratch: docker build --network host -t foundationpose .
  bash docker/run_container.sh
  ```


If it's the first time you launch the container, you need to build extensions. Run this command *inside* the Docker container.
```
bash build_all.sh
```

Later you can execute into the container without re-build.
```
docker exec -it foundationpose bash
```

For more recent GPU such as 4090, refer to [this](https://github.com/NVlabs/FoundationPose/issues/27).
In short, do the following:
```
docker pull shingarey/foundationpose_custom_cuda121:latest
```
Then modify the bash script to use this image instead of `foundationpose:latest`.


# Env setup option 2: conda (local)

1) **Create the environment** (C++ build deps + Python; all on `conda-forge`):

```bash
conda env create -f environment.yml
conda activate foundationpose
```

2) **Install PyTorch** with a CUDA build that matches your machine. The [PyTorch “Get Started”](https://pytorch.org/get-started/locally/) page lists the right `--index-url` (for example `cu124` works on most current NVIDIA drivers). Example:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

3) **Install PyTorch3D and NVDiffRast** (compile from source; needs the CUDA toolkit for `nvcc`. Point `CUDA_HOME` at your install, e.g. `/usr/local/cuda-12.8` or `/usr/local/cuda`):

```bash
export CUDA_HOME=/usr/local/cuda   # or e.g. /usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
python -m pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"
```

`--no-build-isolation` is required so the build can see the `torch` you already installed.

4) **Install remaining Python dependencies** and **build the `mycpp` extension** (BundleSDF’s `mycuda` step is optional; skip it unless you need the model-free / NeRF path):

```bash
python -m pip install -r requirements.txt
bash build_all_conda.sh
```

5) **Optional — Kaolin** (only for the model-free setup; version must match your PyTorch/CUDA—see the [Kaolin install docs](https://kaolin.readthedocs.io/en/latest/notes/installation.html)).


# Run model-based demo
The paths have been set in argparse by default. If you need to change the scene, you can pass the args accordingly. By running on the demo data, you should be able to see the robot manipulating the mustard bottle. Pose estimation is conducted on the first frame, then it automatically switches to tracking mode for the rest of the video. The resulting visualizations will be saved to the `debug_dir` specified in the argparse. (Note the first time running could be slower due to online compilation)
```
python run_demo.py
```

<img src="assets/demo.jpg" width="50%">


Feel free to try on other objects (**no need to retrain**) such as driller, by changing the paths in argparse.

<img src="assets/demo_driller.jpg" width="50%">


# Run on public datasets (LINEMOD, YCB-Video)

For this you first need to download LINEMOD dataset and YCB-Video dataset.

To run model-based version on these two datasets respectively, set the paths based on where you download. The results will be saved to `debug` folder
```
python run_linemod.py --linemod_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/LINEMOD --use_reconstructed_mesh 0

python run_ycb_video.py --ycbv_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video --use_reconstructed_mesh 0
```

To run model-free few-shot version. You first need to train Neural Object Field. `ref_view_dir` is based on where you download in the above "Data prepare" section. Set the `dataset` flag to your interested dataset.
```
python bundlesdf/run_nerf.py --ref_view_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16 --dataset ycbv
```

Then run the similar command as the model-based version with some small modifications. Here we are using YCB-Video as example:
```
python run_ycb_video.py --ycbv_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video --use_reconstructed_mesh 1 --ref_view_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16
```

# Troubleshooting


- For more recent GPU such as 4090, refer to [this](https://github.com/NVlabs/FoundationPose/issues/27).

- For setting up on Windows, refer to [this](https://github.com/NVlabs/FoundationPose/issues/148).

- If you are getting unreasonable results, check [this](https://github.com/NVlabs/FoundationPose/issues/44#issuecomment-2048141043) and [this](https://github.com/030422Lee/FoundationPose_manual)

# Training data download
Our training data include scenes using 3D assets from GSO and Objaverse, rendered with high quality photo-realism and large domain randomization. Each data point includes **RGB, depth, object pose, camera pose, instance segmentation, 2D bounding box**. [[Google Drive]](https://drive.google.com/drive/folders/1s4pB6p4ApfWMiMjmTXOFco8dHbNXikp-?usp=sharing).

<img src="assets/train_data_vis.png" width="80%">

- To parse the camera params including extrinsics and intrinsics
  ```
  glcam_in_cvcam = np.array([[1,0,0,0],
                          [0,-1,0,0],
                          [0,0,-1,0],
                          [0,0,0,1]]).astype(float)
  W, H = camera_params["renderProductResolution"]
  with open(f'{base_dir}/camera_params/camera_params_000000.json','r') as ff:
    camera_params = json.load(ff)
  world_in_glcam = np.array(camera_params['cameraViewTransform']).reshape(4,4).T
  cam_in_world = np.linalg.inv(world_in_glcam)@glcam_in_cvcam
  world_in_cam = np.linalg.inv(cam_in_world)
  focal_length = camera_params["cameraFocalLength"]
  horiz_aperture = camera_params["cameraAperture"][0]
  vert_aperture = H / W * horiz_aperture
  focal_y = H * focal_length / vert_aperture
  focal_x = W * focal_length / horiz_aperture
  center_y = H * 0.5
  center_x = W * 0.5

  fx, fy, cx, cy = focal_x, focal_y, center_x, center_y
  K = np.eye(3)
  K[0,0] = fx
  K[1,1] = fy
  K[0,2] = cx
  K[1,2] = cy
  ```



# Notes
Due to the legal restrictions of Stable-Diffusion that is trained on LAION dataset, we are not able to release the diffusion-based texture augmented data, nor the pretrained weights using it. We thus release the version without training on diffusion-augmented data. Slight performance degradation is expected.

# Acknowledgement

We would like to thank Jeff Smith for helping with the code release; NVIDIA Isaac Sim and Omniverse team for the support on synthetic data generation; Tianshi Cao for the valuable discussions. Finally, we are also grateful for the positive feebacks and constructive suggestions brought up by reviewers and AC at CVPR.

<img src="assets/cvpr_review.png" width="100%">


# License
The code and data are released under the NVIDIA Source Code License. Copyright © 2024, NVIDIA Corporation. All rights reserved.


# Contact
For questions, please contact [Bowen Wen](https://wenbowen123.github.io/).
