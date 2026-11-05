# Native 缺深度连续性候选：源码审查与 CPU 对照

已在独立实际 native 包中实现最小修复，冻结原 root 未改。当前仅通过 CPU，尚未完成真实 GPU 接入或精度验证。

21 组原版/候选对照各 3 个计划输入，共 126 槽。每臂 61 个完成决策、1 个预期 ID 协议失败、1 个后续未执行槽。frame_slots.json 采用显式 columns 数组保存全部槽；null 表示字段未产生，不代表 false。真实 package.create_output_policy 接收 MeasuredMotionSession typed 事件并在 selected_seed 边界完成当前观察，随后执行实际 evaluate_centered/run_private/publish。点流、冻结神经返回和 private 回调均是解析 CPU 桩。

纯 zero/NaN 深度缺失时，候选保留充分的 RGB/visible/固定 ID 连接，但三维 fit count 为 0，不发布、不执行 private。深度恢复后，必须重新通过当前真实深度与原始 anchors 的拟合。恢复后候选仍错时继续弃权。RGB blank、visibility0、ID 重排、7 个可见点、重复像素、有限深度越界/断层、缺测与 Inf/有限矛盾混合、负深度和已 lost 支路仍拒绝。没有降低几何/残差/点数阈值。

新增 min_points=8 只用于缺深度 RGB 支路；原全局 overlap>0 保留。8 个相邻帧 RGB/visible 对并不意味着 8 个点均属于上一帧原三维连接集。实际 3D current mask 显式保留 source/current cue.valid，再与原 motion.valid_mask 相交；缺测 RGB 点不进入拟合。独立审查通过 80 个源码/API/AST/边界检查，该数字不是 80 个场景。

周期纹理别名没有解决：候选随 16 px 周期错误对应运动时，两版无 gap 都可暂定发布并允许 private；候选在 depth-only gap 后也可能恢复这种暂定许可。所有 identity/trusted 标志仍为 false。这不是实例身份已验证，也不是真实 TAP 错 ID 率。

首次 cpu_v1 的完整失败与当时源码快照保留：CPU fixture 漏传 actual initial_pose_original，使两臂均在初始帧报 OBJECT array 错误，最终断言显示 fixed source anchors。它不是已经观察到锚点漂移，也没有计入最终 PASS 分母。修正测试接入、补齐原 3D cue-valid 语义后重新执行 cpu_v2。source_hashes.json 绑定最终证据、早期失败和源码快照；本包不包含源码。

这是单帧深度干预的 CPU 结果，不保证长 gap、自然遮挡、同外观身份或真实重定位。包内没有图像、深度图、位姿、轨迹、GT、模型、权重、原始路径或日志；真实神经验证由后续独立协议进行。
