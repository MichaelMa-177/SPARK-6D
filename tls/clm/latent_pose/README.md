# SPARK-6D V1：空间 latent 物体记忆

本目录实现**可训练的模型框架**：从首帧 RGB-D、已提供 mask 和 SAM3D
粗形状 latent 建立固定物体记忆，独立解码当前帧的物体坐标，再求解 6D 位姿。
当前版本提供训练、记忆序列化、独立查询、评测和测试入口；模型精度需要经过训练和独立评测验证。
旧实验的 `last.pt` 与这个新架构不兼容，也不代表新模型的已训练权重。

目标和验收协议见 [GOAL.md](GOAL.md)：同一个网络为未见物体从单帧建立记忆，
在不同 RGB-D 视频中复用；精度、时序稳定和泛化分别验证。
已执行的真实训练、教师校准、错误定位和未见物体开发结果见
[EXPERIMENTS.md](EXPERIMENTS.md)。目标仍在进行中，尚未通过完整泛化与时序验收。

## 架构

```text
首帧 RGB-D + 确认 mask ── RGB/XYZ 编码 ── 256 个实测表面 token ─┐
已有 SS latent [4096,8] ── 3D 卷积适配 + 空间池化 ── 512 token ─┤
                                                             ↓
                                位置编码 + 实测/生成来源编码 → 固定物体记忆
                                                             ↓
当前 RGB-D → RGB/XYZ 编码 → 1024 query token → 两层 cross-attention
                                                             ↓
                          连续物体坐标 + 目标归属 + 坐标不确定性
                                                             ↓
                                           加权 Kabsch → R, t, pose_valid
```

- 默认保留完整 `16³×8` SS 输入，经三维适配器和空间池化形成 `8³` 生成记忆。
  `memory_side=16` 可以保留 4096 个生成 token，比较压缩与定位的取舍。
- 生成网格中心用作空间位置编码。它们不是表面点，不直接送入刚体求解。
- 位姿解码器预测当前可见表面在物体记忆坐标系中的**连续坐标**。
- 实测与生成 token 有不同来源编码；查询端预测目标概率和坐标不确定性。
  不确定性需要监督学习，几何求解成功不等价于位姿正确。
- `build_memory()` 只需对一个参考帧执行一次；推理模式可以缓存每层 K/V。
  `decode()` 无上一帧位姿参数，查询之间不积累状态。
- 基础配置使用轻量 RGB/XYZ CNN；`dino-*.json` 增加冻结 DINOv2 特征，
  这属于显式使用预训练 backbone，没有将其称为蒸馏。
- `memory_residual` 从实测记忆读出坐标，再预测连续残差；`appearance_prior`
  用参考前景/背景描述子对比约束目标归属。可选择 IRLS 稳健刚体求解。
- `shape_prior.py` 可从一帧已提供 RGB-D/mask 生成并对齐 SS latent，
  不输入 CAD 或物体位姿；不执行新的 SAM 分割，也不需要 SLAT/纹理网格。

## 坐标约定

所有物理坐标使用**米**。物体原点为首帧 mask 内有效实测点的质心，
物体轴与首帧 RGB 相机轴一致：X 右、Y 下、Z 前。

`scale_m` 是固定参考尺寸，只用于网络坐标归一化；输出变换不包含缩放：

```text
X_query_camera = R @ (scale_m * X_memory_normalized) + t
```

`pose[:3,3]` 是该固定物体原点在当前相机中的位置。初始 `origin_m`
只用于建立记忆，不作为当前帧的平移先验。首帧参考质心不是 CAD 的规范原点。
需要与 CAD GT 比较时，应在首次建立参考系时确定一次固定转换。

`export_reference.py` 复现原 SAM3D/PyTorch3D 的行向量布局变换，并用保存的
96 个生成锚点验证旋转、缩放和平移方向。SS voxel token 使用 XYZ 顺序的网格中心。

## 文件与运行

| 文件 | 用途 |
|---|---|
| `model.py` | RGB-D 编码、空间 latent 适配、记忆构建、缓存与查询 |
| `geometry.py` | 米制刚体求解、反射修正、退化检测和坐标变换 |
| `losses.py` | 连续坐标、目标归属、不确定性及对象级对称监督 |
| `data.py` | NPZ/manifest 接口及训练标签隔离 |
| `cli.py` | `train`、`cache`、`infer` |
| `evaluate.py` | 先保存所有预测，再加载标签做固定坐标系评分 |
| `export_reference.py` | 从已有人工首帧与 fresh prior 导出真实输入 |
| `rgbd.py` | 已注册 RGB-D、mask、标定与深度单位到通用参考/查询输入 |
| `runtime.py` | 一次登记物体、跨视频复用记忆、独立流时间戳与失败状态 |
| `smoke.py` | 随机初始化模型的真实输入连通性检查 |
| `tests/` | 几何、梯度、缓存、查询独立性、训练和序列化测试 |
| `experiment.py` | 固定步数训练、开发集选择、坐标/归属诊断与源代码快照 |
| `features.py` / `retrieval.py` | 冻结视觉特征、无查询 GT 的目标区域定位 |
| `shape_prior.py` | 通用单帧 SS latent 与米制布局生成 |
| `foundationpose_teacher.py` | 连续帧教师推理后再加载 GT 校准 |
| `teacher_validation.py` | 伪标签导出、学生/教师/GT 三方对照 |
| `surface_evaluate.py` | 官方 BOP MSSD/MSPD 实现在开发子集上的评分 |

服务器工作副本位于 `/home/mmw/mmw/tls/clm`。统一使用新启动器：

```bash
cd /home/mmw/mmw/tls/clm
bash python_latent -m unittest discover -s latent_pose/tests -v
```

`python_latent` 根据自身位置定位代码，默认复用：

```text
/data/mmw/clm_disk/clm-mmw-20261020/envs/clm/bin/python
```

已有环境：Python 3.10、PyTorch 2.8、NumPy 1.26、OpenCV 4.10。新模块无需
安装额外依赖。其他机器可通过 `CLM_WORK_ROOT`、`CLM_PYTHON` 指定环境和缓存目录。
输出目录必须显式提供；训练、缓存及推理拒绝覆盖已有产物。
旧 `python` 和 `scripts/` 保留历史行为，新框架通过 `python_latent` 启动。

## 原始 RGB-D 与不同视频

`PoseRuntime` 接收 HWC uint8 **RGB**、同分辨率注册深度、3×3 内参，以及
显式的 `depth_scale_m`（如毫米整数深度填 `0.001`，米制浮点深度填 `1.0`）。
输入需已去畸变，使用零 skew 针孔内参；每段视频使用自己的标定。

```python
from latent_pose.runtime import PoseRuntime

runtime = PoseRuntime.from_checkpoint("/path/to/trained/last.pt", device="cuda")
runtime.register_object(
    "object_001", reference_rgb, reference_depth, supplied_mask,
    reference_K, reference_depth_scale_m,
    mask_source="record the actual mask source and saved prompt path",
    shape_latent=shape_latent,       # [4096,8]，对应该参考帧的 SAM3D prior
    latent_xyz_m=latent_xyz_m,       # [4096,3]，在参考 RGB 相机下的米制网格中心
)

video_a = runtime.start_video("video_a", "object_001")
result_a = video_a.estimate(rgb_a, depth_a, K_a, scale_a, timestamp_s=0.0)
video_b = runtime.start_video("video_b", "object_001")
result_b = video_b.estimate(rgb_b, depth_b, K_b, scale_b, timestamp_s=0.0)
```

同一物体只登记一次；不同视频共享固定记忆、分别记录时间戳。
`observed_only` 模型无需传入两个 shape 参数，可直接从原始 RGB-D＋mask 建忆。
空间记忆模式可传入已有生成 prior，或向 runtime 提供 `ShapePriorEncoder`，
在登记物体时生成一次 prior。运行时不生成 SAM 点击或执行交互分割。
`mask_source` 保存调用方声明的来源；它不替代 SAM 交互流程的完整点击记录。

查询默认整帧；`box=(x0,y0,x1,y1)` 可传入外部检测器提供的半开 ROI。
先用原始内参反投影再裁剪，XYZ 始终保留在当前相机坐标。
带冻结特征的模型可设置 `retrieve_region=True`，先在整帧检索目标区域，再编码裁剪区域。
检索只使用首帧记忆、当前 RGB-D 与内参；失败样本不从评测分母移除。

```python
from latent_pose.features import FrozenDINO
from latent_pose.shape_prior import ShapePriorEncoder

features = FrozenDINO("/path/to/local/dinov2", "/path/to/dino_weights.pth", "cuda")
shape = ShapePriorEncoder("/path/to/sam-3d-objects", "/path/to/local/dinov2",
                          "/path/to/sam3d_weights", "cuda", steps=1)
runtime = PoseRuntime.from_checkpoint(
    "/path/to/trained/spatial_checkpoint.pt", device="cuda",
    feature_extractor=features, retrieve_region=True, shape_encoder=shape,
)
# register_object(...) 接收同一帧 RGB-D 和调用方已提供的 mask，
# 此时可省略 shape_latent / latent_xyz_m。
```

`observed_only` 对照不需要实例化 `ShapePriorEncoder`。前处理与 DINO 计算计入
runtime 的 `raw_frame_to_output_ms`；离线 NPZ 评测的网络时间不能当作完整系统延迟。

返回值包含 `pose`、`pose_valid`、`residual_m`、流 ID 和时间戳。
几何无效时 `pose=None`，不将单位矩阵或上一帧位姿当作有效结果。
`pose_valid` 当前只表示求解器通过几何检查，尚不是经校准的正确性置信度。
现阶段所有帧独立定位；本接口未增加学习式时序细化、在线记忆写入或平滑。
模型权重变化后需重建 runtime 和记忆，持久化的检查点绑定缓存使用下文 `cache` 命令。

## 复用 scene59 的已确认输入

以下示例输出目录应尚不存在：

```bash
bash python_latent -m latent_pose.export_reference \
  --prior /data/mmw/clm_disk/clm-mmw-20261020/runs/fresh-sam3d-pipeline/priors/scene59.npz \
  --scene /data/mmw/clm_disk/clm-mmw-20261020/data/bop/ycbv/test/000059 \
  --output /data/mmw/clm_disk/clm-mmw-20261020/runs/v1-reference
```

导出器核验 RGB、mask、深度、内参和 prior，保留原始人工点击记录。
它只复用这个完全相同的首帧；新目标遵循上层 `AGENTS.md` 的人工交互要求。
导出的查询使用完整图像，既不读取 GT mask/pose，也不从上一帧获取 ROI。
这个导出包是开发输入检查材料，没有训练标签，不能直接作为训练集。

```bash
CUDA_VISIBLE_DEVICES=0 bash python_latent -m latent_pose.smoke \
  --reference /data/mmw/clm_disk/clm-mmw-20261020/runs/v1-reference/reference.npz \
  --query /data/mmw/clm_disk/clm-mmw-20261020/runs/v1-reference/query_000029.npz \
  --device cuda \
  --output /data/mmw/clm_disk/clm-mmw-20261020/runs/v1-smoke
```

`smoke` 保存 `untrained.pt` 和报告，明确标记随机初始化和未评测精度。
默认推理拒绝 `training_steps=0` 的检查点。需要检查接口时显式添加
`--allow-untrained`，该标志不表示权重具备估计精度。

## 训练数据协议

每个 NPZ 包含 `schema_version=np.int64(1)`、`units=np.str_("metres")`。
数值数组推荐 float32，不使用 pickle。

| 键 | 无 batch 维的形状 | 含义 |
|---|---|---|
| `rgb` | `3,H,W` | `[0,1]` RGB |
| `xyz` | `3,H,W` | 相机坐标中的米制 XYZ，与 RGB 注册 |
| `valid` | `1,H,W` | 有效深度 |
| `foundation_features` | `(H/4)²,C` | 可选：与 XYZ 网格位置对齐的冻结 RGB 特征 |
| `mask` | `1,H,W` | 仅参考包：已确认的目标 mask |
| `origin_m` | `3` | 仅参考包：固定首帧质心 |
| `scale_m` | `1` | 仅参考包：正的固定归一化尺寸 |
| `shape_latent` | `L³,8` | 仅参考包：按 XYZ 网格展开的 SS latent |
| `latent_xyz_m` | `L³,3` | 仅参考包：网格中心在首帧 RGB 相机中的坐标 |
| `pose` | `4,4` | 仅标注查询包：固定物体记忆坐标到查询相机的 GT |
| `target_mask` | `1,H,W` | 仅标注查询包：用于监督，不作为模型查询输入 |
| `symmetry_transforms` | `S,4,4` | 可选：同一物体记忆坐标系中的米制对称变换，须含单位矩阵 |

H/W 与模型配置 `image_size` 一致。裁剪后必须保证 RGB/XYZ/valid 逐像素注册。
归一化尺寸需要在不同查询间保持不变。

训练 manifest 示例（路径相对于 manifest 文件）：

```json
{
  "schema_version": 1,
  "split": "train",
  "provenance": {"dataset": "your_dataset", "reference_mask_source": "documented_source"},
  "samples": [
    {"object_id": "dataset:object_001", "reference": "references/001.npz", "query": "queries/001_020.npz"}
  ]
}
```

```bash
bash python_latent -m latent_pose.cli train \
  --manifest /path/to/train.json \
  --validation /path/to/validation.json \
  --config latent_pose/configs/default.json \
  --device cuda --epochs 20 --batch-size 4 \
  --output /data/mmw/clm_disk/clm-mmw-20261020/runs/v1-train
```

验证 manifest 的 split 为 `validation`，object_id 必须与训练集分离。
应使用带数据集前缀、可追溯的物体标识；代码无法识别人为改名造成的泄漏。
目前提供单设备训练；训练只更新轻量网络，SAM3D 特征离线缓存。
损失不通过退化初值的 SVD 反传；模型从坐标和置信度监督学习。
离散对称候选按整个对象选择一致的解释；连续对称物体需要在标注端按评测协议处理。

## 记忆缓存、独立推理与评分

```bash
bash python_latent -m latent_pose.cli cache \
  --checkpoint /path/to/trained/last.pt --reference /path/to/reference.npz \
  --device cuda --output /path/to/object-memory.pt

bash python_latent -m latent_pose.cli infer \
  --checkpoint /path/to/trained/last.pt --memory /path/to/object-memory.pt \
  --query /path/to/query-a.npz /path/to/query-b.npz \
  --device cuda --output /path/to/new-predictions

bash python_latent -m latent_pose.evaluate \
  --checkpoint /path/to/trained/last.pt --manifest /path/to/test.json \
  --device cuda --output /path/to/new-evaluation
```

序列化记忆绑定检查点 SHA256，权重改变后必须重建。无效深度、共线对应或
权重不足时返回 `pose_valid=False`；占位单位矩阵不能当作有效输出。
推理接口不读取查询 NPZ 中的标签数组。

评分器先持久化全部预测，再读取标签；无效位姿计入联合通过率的失败分母。
该评分器提供固定坐标系下的平移/旋转和离散对称诊断，尚不是官方 BOP AR。
`surface_evaluate.py` 已接入官方 BOP MSSD/MSPD 实现及对称离散化；
当前已计算自定义开发子集的 VSD/MSSD/MSPD；它不等同于完整官方 BOP AR，正式独立协议仍需补齐。
正式论文实验还需统一检测/分割前端，并计入前处理和目标定位耗时。
计时报告明确区分冷调用、网络部分和完整系统，不能将 smoke 时间当作稳态速度。

## 建议的研究验证

1. 在相同训练数据与解码器下比较 `observed_only`、`geometry_only`、
   `spatial_latent`。`geometry_only` 保留生成网格位置，移除 latent 特征。
   它不是旧实验“96 个 occupancy 边界点”的替代，旧网络作为独立基线保留。
2. 比较 `memory_side=4/8/16` 的定位能力与计算成本。
3. 按首帧/查询表面重叠程度分组；训练与测试按物体分离。
4. 替换物体 latent、打乱 latent 与空间位置的对应，检验网络是否使用三维记忆。
5. 随机查询顺序、跳帧和遮挡后重现，验证独立定位。
6. 在输入、训练数据和计时口径一致的条件下，与 One2Any、PANY、Any6D 比较。

本实现是一套检验这些假设的基础。尚未完成跨物体训练、RGB-only/PnP、
SAM3D layout-only 条件生成、在线记忆更新或官方 BOP 基准评测。
