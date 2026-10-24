# Click-to-Model 逐帧实验数值

当前是研发结果，尚未达到 CVPR 水平。包含 round1–4、两段冻结验证序列与生成模型 smoke 的全部已保存位姿、延迟、静修中间位姿和逐帧评估数值；失败运行、未通过验证的帧、NaN/Infinity 都保留。没有按成功与否筛帧。inventory 的 completed=1 表示程序完整结束，0 表示失败；完整结束不代表精度合格，0 帧失败保留空表。

poses 每帧一行，m00–m33 是按行排列的 4×4 模型到相机齐次变换；平移单位米，旋转无单位。frame_index 从 0 开始，0 为注册帧。latency 单位秒，只记录后续追踪帧，不含注册与模型生成。raw_iterations 每帧一行，iteration_count 为静修次数，iteration_0_m00 等列保存全部中间变换；空单元格表示该次迭代未执行，首帧没有追踪迭代。

per_frame_metrics 中 add_errors_m 单位为米；gt_defined、tracking_evaluated、prediction_present、prediction_valid 是 0/1 标记；verified_flag 的 1/0 表示控制器接纳/未接纳，-1 表示无该标记，不等于真实身份或 GT 精度。source_frame_indices 是数值帧序号，保留原评估配对。evaluations_v2 优先于 v1，两版数值均保留。未提供独立 GT 的生成模型 smoke 没有 ADD 表，不得与原 CAD 位姿直接比较。

CSV 使用 17 位有效数字，已回读核对所有原始数值、NaN 位置及有符号零。每片小于 100000 字节，均带表头，part0000、part0001 按顺序连接。相同实验跨轮次及评估版本分开保留。

该载荷只含可直接阅读的实验数值及本说明，不含源码、日志、内部文件路径、文件哈希、图像、深度/掩码、模型或权重；逐帧数组中的图像文件名字符串也已移除。全部原始二进制与完整诊断仍保留在本地。相同几何与纹理不能保证实例身份；生成模型目前只有开发 smoke，尚无独立完整流程泛化验收。
