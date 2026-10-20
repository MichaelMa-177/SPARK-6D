import numpy as np
import cv2
import json
import open3d as o3d


def compute_scale_factor(
    depth_path: str,
    mask_path: str,
    intrinsics_json: str,
    unit_mesh_path: str,
    depth_scale_default=0.001,
    percentile_clip=2
):
    """
    通过深度图 + mask + 相机内参 计算单位化模型的真实尺度缩放系数。
    返回: scale_factor, real_extent_vec, unit_extent_vec
    """

    # -------------------------
    # 1. 读取深度图 & mask
    # -------------------------
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 128).astype(np.float32)

    h, w = depth_raw.shape

    # -------------------------
    # 2. 读取相机内参
    # -------------------------
    with open(intrinsics_json, "r") as f:
        intr = json.load(f)

    fx = intr["fx"]
    fy = intr["fy"]
    cx = intr["cx"]
    cy = intr["cy"]
    depth_scale = intr.get("depth_scale", depth_scale_default)

    # -------------------------
    # 3. 深度图转为米并裁剪物体区域
    # -------------------------
    depth_m = depth_raw * depth_scale
    depth_m = depth_m * mask  # 非物体区域置零

    # 取有效深度
    valid = depth_m > 0
    if valid.sum() < 20:
        raise ValueError("物体有效点太少，mask 或深度图可能错误")

    # -------------------------
    # 4. 深度 + 内参 → 真实尺度点云
    # -------------------------
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    us = us[valid]
    vs = vs[valid]
    Z = depth_m[valid]

    X = (us - cx) * Z / fx
    Y = (vs - cy) * Z / fy

    pts_real = np.stack([X, Y, Z], axis=-1)

    # -------------------------
    # 5. 去噪：用百分位裁剪点云范围
    # -------------------------
    lower = percentile_clip
    upper = 100 - percentile_clip
    min_vals = np.percentile(pts_real, lower, axis=0)
    max_vals = np.percentile(pts_real, upper, axis=0)

    real_extent_vec = max_vals - min_vals
    real_size = real_extent_vec.max()

    print(f"[真实物体尺寸] (米): {real_extent_vec}")
    return real_extent_vec
