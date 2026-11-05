# 单帧深度缺失与 native 身份断链：CPU 反例

确认当前 native 门会把某些“RGB cue 持续、仅缺当前深度”情况当作永久断链。本包是诊断证据，尚未实现或合并修复。

11 个固定场景各计划 3 个输入，共 33 槽。32 个输入实际进入 CPU session，其中 1 个 ID 协议错误；31 个完成 gate 决策，错误后的 1 槽未执行。所有槽、失败和负例均在 evidence.json 中。使用真实 NumPy FixedCameraPointMotion、PoseValidityChecker、MeasuredMotionSession 与 native TwoPhaseContinuity；点流和神经返回是明确的解析 CPU 桩，没有真实 TAP/FP/GPU。

- **纯深度孔**：RGB 字节、64 个 query ID/像素/可见性完全不变，一帧深度全 0 或 NaN。原始 RGB 梯度支持仍有 64 点，但 current RGB cue 被深度 valid 清成 0，branch 终结并 latch lost。下一帧恢复 64/64 三维内点、几何支持与原 anchors fit 后，仍不发布、不允许 private。
- **对照**：只缺掉 32 点深度时其余 32 点可保持追踪；仅候选暂时偏离表面时原门会 temporarily_unobservable，并能恢复。已有这种临时分支没有解决全深度孔引发的提前 latch。
- **必须保留的拒绝**：RGB blank（该平面无深度结构）、RGB 遮挡且可见性为 false、ID 重排均拒绝；ID 重排使 point owner 和 session 失败，单独送该真实失败 packet 到 native gate 也被拒绝。
- **周期纹理边界**：人为把查询整体跳 16 px，RGB/ID 不变。若候选不跟随错误对应，原 anchors fit 拒绝；若候选与这种错误对应一致，56 个刚性内点和表面支持足以让当前门暂定发布并允许 private，identity_verified 仍为 false。缺深度期间也可产生同样别名。这是合成的可观察性反例，不是真实 TAP 错 ID 率，也不能把 RGB-D 可解释的刚性移动与错误实例区分开。

最小下一步是在独立候选中分离“2D cue 连接”与“当前 3D fit mask”：RGB 连续性不依赖当前深度；3D 发布与学习仍要求当前 measured、真实 depth lift、原 anchors、原点数/残差/几何阈值与候选表面支持。缺整个 packet、RGB/visibility 丢失、ID 重排和冲突保持拒绝。深度回来须重新做当前几何验证，不能清除已有身份歧义或假称 trusted。只拆这两个 mask 不能解决不可观察的周期/同外观身份交换。

未读取 GT，未发布 RGB、深度图、位姿、轨迹、权重、源码、原始路径或日志。没有修改冻结 runtime、没有运行新的 GPU 实验；下一轮才考虑实现与真实神经验证。evidence.json 的 audit_only_current_fixed_anchor_fit 是纯诊断复核，不能作为新增授权或恢复身份的证据。
