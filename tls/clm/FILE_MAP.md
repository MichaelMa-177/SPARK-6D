# 新服务器项目文件位置

核对时间：2026-10-20 14:17（北京时间）。以下为实际存在的目录，不是待创建的规划。

服务器：`ssh mmw@8.210.184.25`，使用本地现有 SSH 配置连接。该机器不是 CLM2/CLM3。

## 1. 先记住两个根目录

| 根目录 | 用途 |
|---|---|
| `/home/mmw/tls/clm` | 项目代码、启动脚本、原始参考报告、迁来的训练权重和人工首帧输入 |
| `/data/mmw/clm_disk/clm-mmw-20261020` | 本项目的大盘工作区：数据、SAM3D/DINO权重、Python环境、缓存和运行结果 |

本文用 `CODE` 表示第一个目录，`WORK` 表示第二个目录。下文相对路径均以明确标出的根目录为准。

操作范围只包括本项目上述目录。`/data/mmw/clm_disk` 是共享磁盘的挂载入口，不是整盘都归我们；不要清理其他目录或修改他人文件。

## 2. 代码与入口：CODE

```text
/home/mmw/tls/clm/
├── FILE_MAP.md                         本文的服务器副本
├── AGENTS.md                           人工首帧点击、mask和点云展示规则
├── python                              私有Python环境启动器，使用 bash python 调用
├── scripts/                            环境、下载、生成、评测和测速入口
├── experiments/
│   ├── first-memory-fast-refine/        从CLM3迁来的原始轻量跟踪实验
│   │   ├── fast_model.py               RGB-XYZ编码、记忆、对应和两轮位姿精修
│   │   ├── geometry.py                 刚体变换、SVD等几何计算
│   │   ├── data_fast.py                RGB-D读取、XYZ、ROI和相机参数处理
│   │   ├── train_fast.py               原轻量网络训练代码；本轮没有重新训练
│   │   ├── evaluate_fast.py            先推理，再加载GT评分及导出视频
│   │   ├── training/                  原训练检查点
│   │   ├── inputs/                    原人工确认首帧、mask、点击和相机信息
│   │   └── priors/                    旧实验迁来的scene50/54/59首帧记忆
│   └── first-memory-fresh-sam3d/       本次新机重新生成首帧记忆的实验
│       ├── *.py                       原轻量网络代码的独立实验副本
│       ├── inputs -> ../first-memory-fast-refine/inputs
│       ├── training -> ../first-memory-fast-refine/training
│       ├── priors -> WORK/runs/fresh-sam3d-pipeline/priors
│       └── evaluation -> WORK/runs/fresh-sam3d-pipeline/evaluation
├── vendor/
│   ├── clm3-20261020/                  迁来的源码基线，保留用于比对
│   │   ├── sam-3d-objects/
│   │   └── dinov2/
│   └── sam3d-runtime/                  当前首帧推理实际使用的源码
│       ├── sam-3d-objects/
│       └── dinov2/
└── reference/
    ├── clm3_report.json                原CLM3实验结果，只作参考
    └── data_manifest.json              684个迁移数据文件的大小及SHA256
```

上树中的 `WORK` 是路径缩写，不是服务器上字面名为WORK的目录。符号链接的实际目标如下，修改链接内文件会影响对应原文件：

| 链接 | 实际目标 |
|---|---|
| `CODE/experiments/first-memory-fresh-sam3d/inputs` | `/home/mmw/tls/clm/experiments/first-memory-fast-refine/inputs` |
| `CODE/experiments/first-memory-fresh-sam3d/training` | `/home/mmw/tls/clm/experiments/first-memory-fast-refine/training` |
| `CODE/experiments/first-memory-fresh-sam3d/priors` | `/data/mmw/clm_disk/clm-mmw-20261020/runs/fresh-sam3d-pipeline/priors` |
| `CODE/experiments/first-memory-fresh-sam3d/evaluation` | `/data/mmw/clm_disk/clm-mmw-20261020/runs/fresh-sam3d-pipeline/evaluation` |

### 当前用哪份代码

- 后续位姿推理：`CODE/experiments/first-memory-fresh-sam3d/`。
- 首帧SAM3D/DINO：`CODE/vendor/sam3d-runtime/`。
- `vendor/clm3-20261020/` 保留原样；runtime只调整了4个Python文件的可选模块导入方式，没有替换SS模型计算。
- 当前是SS1 Shortcut一步、DINO518、粗形状latent流程，没有运行SLAT纹理/完整mesh流程。初始化日志的默认25步不代表实际采样步数；生成前已覆盖为1步。

## 3. 权重和人工输入

### 轻量6D位姿网络

```text
/home/mmw/tls/clm/experiments/first-memory-fast-refine/training/last.pt
```

当前新机使用原训练的 `last.pt`，没有重训或按新机结果选权重。SHA256以 `0a9415f5d03f0b92` 开头。其他原检查点也保留在同一 `training/` 目录。

### SAM3D和DINO

```text
/data/mmw/clm_disk/clm-mmw-20261020/weights/sam3d/
├── ss_generator.ckpt                  SS生成器，约6.23GiB
├── ss_decoder.ckpt                    粗形状occupancy解码器，约141MiB
├── dinov2_vitl14_reg4_pretrain.pth     DINOv2 ViT-L/14寄存器版本，约1.13GiB
├── pipeline.yaml
├── ss_generator.yaml
├── ss_decoder.yaml
└── slat_*.yaml                        迁来的配置；并不表示SLAT权重已经迁入
```

三个权重与CLM3的SHA256均已核验。当前没有迁入全部SLAT网格/纹理权重。

### 首帧人工确认输入

目录：`/home/mmw/tls/clm/experiments/first-memory-fast-refine/inputs/`。

包含 `first_frame.png`、`first_depth.png`、`mask.png`、`user_prompt.json`、`camera.json`。对应原钻头scene59首帧及确认记录 `7f37ebe6a09819a6`。

复现同一首帧可复用这些文件；换图像、目标或序列必须重新让用户交互选点，不能用GT或自动点击替代。

## 4. 数据：WORK/data

### BOP / YCB-V钻头实验数据

```text
/data/mmw/clm_disk/clm-mmw-20261020/data/bop/ycbv/test/
├── 000050/                            75对RGB-D，原训练场景
├── 000054/                            75对RGB-D，原训练场景
└── 000059/                            75对RGB-D，开发验证场景
    ├── rgb/
    ├── depth/
    ├── mask_visib/
    ├── scene_camera.json
    ├── scene_gt.json
    └── scene_gt_info.json
```

共225对RGB-D、684个迁移文件，已经逐项校验。数据为稀疏BOP关键帧，不是连续30Hz视频。路径中的 `test` 是原数据组织名称；本项目的50/54用于训练、59用于开发验证，不因此成为独立测试集。

深度转米按 `scene_camera.json` 中的 `depth_scale / 1000` 处理。后续推理不使用GT mask/pose；GT仅用于推理完成后的评分。

### 新下载的self_data

```text
/data/mmw/clm_disk/clm-mmw-20261020/data/click-to-model/self_data/
├── raw/                               实际使用的已解压原始RGB-D
│   ├── d415_907222062024_20261019_221738/   1283对
│   └── d415_907222062024_20261019_221847/    771对
├── hf/self_data/                       HF原归档、预览及元数据
├── control/                           下载控制文件
├── download_report.json                下载与完整性核验报告
└── DONE
```

每个 `raw/` 序列包含 `rgb/`、`depth/`、相机信息、内参和时间戳等原文件。共2054对640×480 RGB-D，4118个原始文件，已核验SHA256、ZIP CRC及全部图像解码。

深度为 `uint16` 毫米，转米乘 `0.001`，不要直接套用旧BOP的深度尺度。`masks/`、`mesh/` 内为说明占位，不代表已经有目标分割、CAD或位姿GT。当前尚未对新self_data选择目标或执行完整跟踪。

来源：`Mingwei000/click-to-model-data-online`，固定提交 `cbae3dc787dd7c4a8d2149f725356b06a4c356ad`。只下载了self_data部分，不是整个仓库。

## 5. Python环境及启动方式

环境：`/data/mmw/clm_disk/clm-mmw-20261020/envs/clm/`。

- Python `3.10.12`。
- PyTorch `2.8.0+cu128`，CUDA运行时 `12.8`。
- 已在RTX3090验证SS首帧生成、occupancy解码和后续位姿网络。
- 是本项目私有venv，不是系统Python，也不是旧CLM3的conda环境。

统一通过启动器调用，避免进入错误环境：

```bash
cd /home/mmw/tls/clm
bash python -c 'import sys, torch; print(sys.executable); print(torch.__version__)'
```

`CODE/python` 会设置BOP数据根目录及缓存路径；pip/HF/Torch/CUDA/Triton/Matplotlib缓存、临时文件都放在本项目大盘工作区，不写系统盘或Mac的 `/private/tmp`。

依赖记录：`WORK/runs/sam3d-setup/requirements.lock.txt`；初始环境记录在 `WORK/runs/bootstrap/`。

## 6. 当前结果：WORK/runs

| 相对WORK的目录 | 内容及用途 |
|---|---|
| `runs/fresh-sam3d-pipeline/` | 当前主要结果：新机重新生成首帧记忆，再重新推理75帧 |
| `runs/warm-latent-benchmark/` | 首帧速度复测：首次调用、5次预热、30次新推理的原始耗时 |
| `runs/fresh-geometry-preview/` | 对保存的首帧latent解码，供建模网页展示 |
| `runs/migration/` | 数据/权重哈希审计和早期三帧烟测；不是最新完整视频结果 |
| `runs/bootstrap/` | 初始环境安装、CUDA检查及日志 |
| `runs/sam3d-setup/` | SS推理依赖安装日志和当前依赖锁定记录 |
| `runs/sam3d-migration/` | SAM3D/DINO权重下载、源哈希核验及完成标记 |
| `runs/self-data-download/` | self_data下载日志；最终数据核验报告在数据目录下 |

### 主要视频、位姿和首帧记忆

以下均位于 `/data/mmw/clm_disk/clm-mmw-20261020/runs/fresh-sam3d-pipeline/`：

```text
priors/scene59.npz                     新机实际生成的首帧latent、layout及96个生成点
priors/scene59.json                    输入、权重、时延及SHA256记录
evaluation/full.mp4                    完整版本输出视频
evaluation/full_predictions.npz        每帧新推理的位姿及记忆点
evaluation/full_timing.json            每帧时延
evaluation/full_errors.json            每帧平移/旋转误差
evaluation/full_diagnostics.json       对应、残差、目标性等诊断
evaluation/report.json                 汇总结果与协议
evaluation/accuracy_latency.png        误差及延迟曲线
evaluation/no_generated.mp4            不使用生成部分的对照
evaluation/shuffled_observed.mp4       打乱实测外观特征的对照
pipeline.log                          整段运行日志
DONE / exit_code                      已完成，退出码0
```

该版已跑通，但精度未达标：平移中位9.94mm、旋转中位21.71°，10mm/5°联合通过率16.2%。不要把运行完成解释为准确跟踪已解决。

### 建模网页对应的几何数据

目录：`/data/mmw/clm_disk/clm-mmw-20261020/runs/fresh-geometry-preview/`。

| 文件 | 含义 |
|---|---|
| `measured.npz` | 13,773个实测可见表面点及真实RGB |
| `generated.npz` | 11,996个推断occupancy边界点，统一显示色，不是纹理 |
| `overlay.npz` | 两者在原相机坐标中的叠加，无ICP或额外对齐 |
| `memory_anchors.npz` | 实际跟踪使用的96个生成锚点 |
| `occupancy.npz` | 64³解码logits、占据体素及边界坐标 |
| `report.json` | latent来源、解码时延、点数和几何文件SHA256 |

### 常用脚本位置

均在 `/home/mmw/tls/clm/scripts/`：

| 文件 | 用途 |
|---|---|
| `prepare_fresh_prior.py` | 从原RGB-D和人工mask重新生成SS1/DINO518首帧记忆 |
| `run_fresh_pipeline.sh` | 首帧生成、75帧推理、评分和视频导出的完整运行入口 |
| `benchmark_warm_latent.py` / `run_warm_latent.sh` | 预热后的首帧latent测速 |
| `export_fresh_geometry.py` / `run_geometry_preview.sh` | 解码保存的latent并导出实测/生成/叠加点云 |
| `verify_migration.py` | 校验迁移数据、训练权重及人工首帧输入 |
| `smoke_trained_refiner.py` | 原轻量网络三帧执行烟测，使用迁来的首帧prior |
| `bootstrap_env.py` / `run_setup.sh` | 初始私有环境配置 |
| `install_sam3d_deps.py` / `run_sam3d_setup.sh` | SS推理依赖配置 |
| `download_self_data.py` / `run_download_self_data.sh` | self_data下载及完整性验证 |
| `download_sam3d_weights.py` / `run_download_sam3d.sh` | 核心SS/DINO权重下载及源SHA验证 |

这些任务已有结果且带防覆盖/锁检查，不要把运行入口当成可以无条件重复执行的命令。新实验应另设输出目录，不覆盖已有prior、报告或视频；单独首帧生成也会拒绝覆盖现有 `scene59.npz`。

## 7. 容量与挂载注意事项

核对时项目占用约为：

| 位置 | 占用 |
|---|---:|
| CODE整个目录 | 139MiB |
| WORK/data | 2.5GiB |
| WORK/weights | 7.6GiB |
| WORK/envs | 7.0GiB |
| WORK/cache | 55MiB |
| WORK/runs | 20MiB |
| WORK/downloads | 1.8MiB |

系统文件系统 `/dev/sdb3` 当时仅余约4.5GiB；大数据和大环境继续放WORK。大盘 `/dev/sda1` 挂在 `/data/mmw/clm_disk`，当时普通用户共享可用约352GiB；这不是我们独占的配额，容量随他人使用而变化。

当前挂载只对本次运行有效，未修改 `/etc/fstab`。服务器重启后，先检查：

```bash
findmnt -rn -T /data/mmw/clm_disk/clm-mmw-20261020 -o SOURCE,TARGET,FSTYPE
```

期望源设备为 `/dev/sda1`、挂载点为 `/data/mmw/clm_disk`。如果没有挂载或输出变成系统盘，不要继续往该路径下载、安装或训练，避免误写系统盘；先核实并恢复正确挂载。

历史挂载脚本另存于 `/home/mmw/tls/clm-bootstrap/mount_clm_storage.sh`，不是日常实验入口，不要在未核对当前挂载状态时重复执行。

## 8. 与本地Mac的对应位置

本地迁移记录和备份目录：

```text
/Users/nelson/Projects/Research/Language-Guided-Video-Understanding/experiments/2026-10-20-server-migration/
```

| 本地相对路径 | 内容 |
|---|---|
| `NEW_SERVER_FILE_MAP.md` | 本文的本地版本 |
| `payload/clm/` | 迁移脚本及代码的本地工作副本，不是服务器环境或整个大数据盘的完整镜像 |
| `verification/fresh-sam3d-pipeline/` | 已下载的视频、报告、曲线及预览图 |
| `warm-latent-benchmark/report.json` | 30次预热后测速原始记录 |
| `model-preview/dist/index.html` | 已在Chrome打开的离线交互建模网页 |
| `model-preview/dist/assets/` | 网页所用实测/生成点云、occupancy、mask和报告 |
| `EVIDENCE_INDEX.md` | 迁移、运行与精度结论的证据索引 |

注意：交互建模HTML目前只在本地Mac；服务器保存的是对应NPZ几何产物，不要在服务器代码根目录寻找已部署的网站。网页没有发布到公网，也不依赖常驻推理服务。
