# SAM-score-first：已保存初始化返回的离线反事实

固定规则：从原 SAM 三个非空 mask 中取最高原 SAM 分数；同分取原序最前者。直接复用该 mask 的实际 FP top1 原坐标返回，不做深度/几何过滤，不比较不同 mask 的 FP 分数，也不重新精修。

该 mask 选择遵循 [SAM 2 官方 predictor 文档](https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py) 已有的用法：单点击多 mask 输出可按预测的 mask 质量分数选择；这不是新方法，分数也不是位姿置信度或物理身份概率。

13 个计划槽位包含 TLESS 的完整 12 槽和 mustard 的一个初始化帧。11 个槽有实际返回：8 个改变 mask，3 个保持，另 2 个 unavailable。TLESS 另有 2 个原点击没有命中声明物体的可见 GT mask，始终保留 unknown，没有移动点击或重选 GT。每槽前后数值见 `per_trial.csv`。

TLESS 在全部 20 个固定阈值下，新结果均为 8 正确、0 错误、4 未知，以 12 为分母。原 0.1D MSSD / normalized 5px MSPD 均为 6 正确、2 错误、4 未知。两个 object 18 失败槽的 MSSD 从 166.187 / 162.641 mm 降至均约 3.289 mm。两类 pilot 平均阈值 recall 分别从 0.491667 / 0.500000 变为 0.666667 / 0.666667；这不是 BOP leaderboard AR 或 mAP。`threshold_counts.csv` 保留全部阈值前后计数。

回退同样保留：TLESS trial 10/11 的误差略增但仍正确；轴对称 trial 01 的 MSSD 改善而普通 ADD 增大。Mustard 初始化 ADD 从 1.934132 mm 增至 2.185928 mm，两者仍低于原 19.646326 mm 阈值。Mustard 只测初始化，不与 TLESS 指标合并，也没有新的追踪轨迹或追踪精度结论。

TLESS 使用 models_eval 全部原 vertex rows、米制、声明对称的 GT @ S；连续对称使用固定 315 角近似，MSPD 按 raw pixels × 640/720 归一化，阈值采用严格 `<`。Mustard 使用冻结的 10,983 个 loader vertex rows（保留 UV 缝重复权重）和 raw original GT frame zero；ADD 成功条件仍为 `<= 0.1D`，没有 GT0 对齐。

原 admission 接纳数为 0，新选择也未升级任何可信状态；零接纳风险记为 null，不是 0。SAM 分数不表示物理实例身份可信度。该基线是在已知失败和已查看的开发数据上分析，同图 A/B 点击不是独立样本；这是离线反事实，不是 held-out 验证、新模型推理或已部署改动。

此前固定 8 mm 硬点击表面约束没有增加主阈值成功数，并在一个 trial 排除了全部 26 个正确候选，未部署。候选上限诊断表明两个失败 trial 的原候选库各有 26 个正确候选；GT 仅用于诊断，不参与本包的运行选择规则。其收据哈希列于 `summary.json`，不重复原始候选数据。

本包仅含汇总、每槽误差、unknown、来源哈希和说明；不含位姿矩阵、原始 GT、图像、mask、模型、日志、源码或绝对本地路径。来源和文件校验见 `summary.json` / `manifest.json`。
