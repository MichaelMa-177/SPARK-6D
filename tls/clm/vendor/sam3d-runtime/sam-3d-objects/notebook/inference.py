# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import open3d as o3d
# not ideal to put that here
os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]
os.environ["LIDRA_SKIP_INIT"] = "true"

import sys
from typing import Union, Optional, List, Callable
import numpy as np
from PIL import Image
from omegaconf import OmegaConf, DictConfig, ListConfig
from hydra.utils import instantiate, get_method
import torch
import cv2
import json
import math
import utils3d
import shutil
import subprocess
import seaborn as sns
from PIL import Image
import numpy as np
# import gradio as gr
import matplotlib.pyplot as plt
from copy import deepcopy
from kaolin.visualize import IpyTurntableVisualizer
from kaolin.render.camera import Camera, CameraExtrinsics, PinholeIntrinsics
import builtins
from pytorch3d.transforms import quaternion_multiply, quaternion_invert
from torchvision.transforms import ToTensor

import sam3d_objects  # REMARK(Pierre) : do not remove this import
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils
from sam3d_objects.utils.visualization import SceneVisualizer


__all__ = ["Inference"]

WHITELIST_FILTERS = [
    lambda target: target.split(".", 1)[0] in {"sam3d_objects", "torch", "torchvision", "moge"},
]

BLACKLIST_FILTERS = [
    lambda target: get_method(target)
    in {
        builtins.exec,
        builtins.eval,
        builtins.__import__,
        os.kill,
        os.system,
        os.putenv,
        os.remove,
        os.removedirs,
        os.rmdir,
        os.fchdir,
        os.setuid,
        os.fork,
        os.forkpty,
        os.killpg,
        os.rename,
        os.renames,
        os.truncate,
        os.replace,
        os.unlink,
        os.fchmod,
        os.fchown,
        os.chmod,
        os.chown,
        os.chroot,
        os.fchdir,
        os.lchown,
        os.getcwd,
        os.chdir,
        shutil.rmtree,
        shutil.move,
        shutil.chown,
        subprocess.Popen,
        builtins.help,
    },
]


class Inference:
    # public facing inference API
    # only put publicly exposed arguments here
    def __init__(
        self,
        config_file: str,
        compile: bool = False,
        decoder_device: str = None,
        low_vram: bool = False,
        mesh_only: bool = False,
        use_depth_model: bool = True,
    ):
        # load inference pipeline
        config = OmegaConf.load(config_file)
        config.rendering_engine = "pytorch3d"  # overwrite to disable nvdiffrast
        config.compile_model = compile
        config.workspace_dir = os.path.dirname(config_file)
        config.low_vram = low_vram
        if mesh_only:
            config.decode_formats = ["mesh"]
            config.slat_decoder_gs_config_path = None
            config.slat_decoder_gs_ckpt_path = None
            config.slat_decoder_gs_4_config_path = None
            config.slat_decoder_gs_4_ckpt_path = None
        if not use_depth_model:
            config.depth_model = None
        # Set decoder_device for dual-GPU support (decoders on separate GPU to avoid OOM)
        if decoder_device is not None:
            config.decoder_device = decoder_device
        check_hydra_safety(config, WHITELIST_FILTERS, BLACKLIST_FILTERS)
        self._pipeline: InferencePipelinePointMap = instantiate(config)

    def merge_mask_to_rgba(self, image, mask):
        mask = mask.astype(np.uint8) * 255
        mask = mask[..., None]
        # embed mask in alpha channel
        rgba_image = np.concatenate([image[..., :3], mask], axis=-1)
        return rgba_image

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: Optional[Union[None, Image.Image, np.ndarray]],
        seed: Optional[int] = None,
        pointmap=None,
        stage1_inference_steps: Optional[int] = None,
        stage2_inference_steps: Optional[int] = None,
        use_stage1_distillation: bool = False,
        use_stage2_distillation: bool = False,
    ) -> dict:
        image = self.merge_mask_to_rgba(image, mask)
        # 显式清理一下显存，防止上一次运行的残留
        torch.cuda.empty_cache()

        mesh_only = self._pipeline.decode_formats == ["mesh"]
        return self._pipeline.run(
            image,
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=not mesh_only,
            use_vertex_color=True,
            stage1_inference_steps=stage1_inference_steps,
            stage2_inference_steps=stage2_inference_steps,
            use_stage1_distillation=use_stage1_distillation,
            use_stage2_distillation=use_stage2_distillation,
            pointmap=pointmap,
            decode_formats=self._pipeline.decode_formats,
        )

def _yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()
        orig = (
            torch.tensor(
                [
                    torch.sin(yaw) * torch.cos(pitch),
                    torch.sin(pitch),
                    torch.cos(yaw) * torch.cos(pitch),
                ]
            ).cuda()
            * r
        )
        extr = utils3d.torch.extrinsics_look_at(
            orig,
            torch.tensor([0, 0, 0]).float().cuda(),
            torch.tensor([0, 1, 0]).float().cuda(),
        )
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


def render_video(
    sample,
    resolution=512,
    bg_color=(0, 0, 0),
    num_frames=300,
    r=2.0,
    fov=40,
    pitch_deg=0,
    yaw_start_deg=-90,
    **kwargs,
):

    yaws = (
        torch.linspace(0, 2 * torch.pi, num_frames) + math.radians(yaw_start_deg)
    ).tolist()
    pitch = [math.radians(pitch_deg)] * num_frames

    extr, intr = _yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)

    return render_utils.render_frames(
        sample,
        extr,
        intr,
        {"resolution": resolution, "bg_color": bg_color, "backend": "gsplat"},
        **kwargs,
    )


def ready_gaussian_for_video_rendering(scene_gs, in_place=False, fix_alignment=False):
    if fix_alignment:
        scene_gs = _fix_gaussian_alignment(scene_gs, in_place=in_place)
    scene_gs = normalized_gaussian(scene_gs, in_place=fix_alignment)
    return scene_gs


def _fix_gaussian_alignment(scene_gs, in_place=False):
    if not in_place:
        scene_gs = deepcopy(scene_gs)

    device = scene_gs._xyz.device
    dtype = scene_gs._xyz.dtype
    scene_gs._xyz = (
        scene_gs._xyz
        @ torch.tensor(
            [
                [-1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
            ],
            device=device,
            dtype=dtype,
        ).T
    )
    return scene_gs


def normalized_gaussian(scene_gs, in_place=False, outlier_percentile=None):
    if not in_place:
        scene_gs = deepcopy(scene_gs)

    orig_xyz = scene_gs.get_xyz
    orig_scale = scene_gs.get_scaling

    active_mask = (scene_gs.get_opacity > 0.9).squeeze()
    inv_scale = (
        orig_xyz[active_mask].max(dim=0)[0] - orig_xyz[active_mask].min(dim=0)[0]
    ).max()
    norm_scale = orig_scale / inv_scale
    norm_xyz = orig_xyz / inv_scale

    if outlier_percentile is None:
        lower_bound_xyz = torch.min(norm_xyz[active_mask], dim=0)[0]
        upper_bound_xyz = torch.max(norm_xyz[active_mask], dim=0)[0]
    else:
        lower_bound_xyz = torch.quantile(
            norm_xyz[active_mask],
            outlier_percentile,
            dim=0,
        )
        upper_bound_xyz = torch.quantile(
            norm_xyz[active_mask],
            1.0 - outlier_percentile,
            dim=0,
        )
    # 原始中心移到原点
    center = (lower_bound_xyz + upper_bound_xyz) / 2
    norm_xyz = norm_xyz - center
    scene_gs.from_xyz(norm_xyz)
    
    scene_gs.mininum_kernel_size /= inv_scale.item()
    scene_gs.from_scaling(norm_scale)
    return scene_gs


def make_scene(*outputs, in_place=False):
    if not in_place:
        outputs = [deepcopy(output) for output in outputs]

    all_outs = []
    minimum_kernel_size = float("inf")
    for output in outputs:
        # move gaussians to scene frame of reference
        PC = SceneVisualizer.object_pointcloud(
            points_local=output["gaussian"][0].get_xyz.unsqueeze(0),
            quat_l2c=output["rotation"],
            trans_l2c=output["translation"],
            scale_l2c=output["scale"],
        )
        output["gaussian"][0].from_xyz(PC.points_list()[0])
        # must ... ROTATE
        output["gaussian"][0].from_rotation(
            quaternion_multiply(
                quaternion_invert(output["rotation"]),
                output["gaussian"][0].get_rotation,
            )
        )
        scale = output["gaussian"][0].get_scaling
        adjusted_scale = scale * output["scale"]
        assert (
            output["scale"][0, 0].item()
            == output["scale"][0, 1].item()
            == output["scale"][0, 2].item()
        )
        output["gaussian"][0].mininum_kernel_size *= output["scale"][0, 0].item()
        adjusted_scale = torch.maximum(
            adjusted_scale,
            torch.tensor(
                output["gaussian"][0].mininum_kernel_size * 1.1,
                device=adjusted_scale.device,
            ),
        )
        output["gaussian"][0].from_scaling(adjusted_scale)
        minimum_kernel_size = min(
            minimum_kernel_size,
            output["gaussian"][0].mininum_kernel_size,
        )
        all_outs.append(output)

    # merge gaussians
    scene_gs = all_outs[0]["gaussian"][0]
    scene_gs.mininum_kernel_size = minimum_kernel_size
    for out in all_outs[1:]:
        out_gs = out["gaussian"][0]
        scene_gs._xyz = torch.cat([scene_gs._xyz, out_gs._xyz], dim=0)
        scene_gs._features_dc = torch.cat(
            [scene_gs._features_dc, out_gs._features_dc], dim=0
        )
        scene_gs._scaling = torch.cat([scene_gs._scaling, out_gs._scaling], dim=0)
        scene_gs._rotation = torch.cat([scene_gs._rotation, out_gs._rotation], dim=0)
        scene_gs._opacity = torch.cat([scene_gs._opacity, out_gs._opacity], dim=0)

    return scene_gs


def check_target(
    target: str,
    whitelist_filters: List[Callable],
    blacklist_filters: List[Callable],
):
    if any(filt(target) for filt in whitelist_filters):
        if not any(filt(target) for filt in blacklist_filters):
            return
    raise RuntimeError(
        f"target '{target}' is not allowed to be hydra instantiated, if this is a mistake, please do modify the whitelist_filters / blacklist_filters"
    )


def check_hydra_safety(
    config: DictConfig,
    whitelist_filters: List[Callable],
    blacklist_filters: List[Callable],
):
    to_check = [config]
    while len(to_check) > 0:
        node = to_check.pop()
        if isinstance(node, DictConfig):
            to_check.extend(list(node.values()))
            if "_target_" in node:
                check_target(node["_target_"], whitelist_filters, blacklist_filters)
        elif isinstance(node, ListConfig):
            to_check.extend(list(node))


def load_image(path):
    image = Image.open(path)
    image = np.array(image)
    image = image.astype(np.uint8)
    return image


def load_mask(path):
    mask = load_image(path)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask[..., -1]
    return mask


def load_single_mask(folder_path, index=0, extension=".png"):
    masks = load_masks(folder_path, [index], extension)
    return masks[0]


def load_masks(folder_path, indices_list=None, extension=".png"):
    masks = []
    indices_list = [] if indices_list is None else list(indices_list)
    if not len(indices_list) > 0:  # get all all masks if not provided
        idx = 0
        while os.path.exists(os.path.join(folder_path, f"{idx}{extension}")):
            indices_list.append(idx)
            idx += 1

    for idx in indices_list:
        mask_path = os.path.join(folder_path, f"{idx}{extension}")
        assert os.path.exists(mask_path), f"Mask path {mask_path} does not exist"
        mask = load_mask(mask_path)
        masks.append(mask)
    return masks


def display_image(image, masks=None):
    def imshow(image, ax):
        ax.axis("off")
        ax.imshow(image)

    grid = (1, 1) if masks is None else (2, 2)
    fig, axes = plt.subplots(*grid)
    if masks is not None:
        mask_colors = sns.color_palette("husl", len(masks))
        black_image = np.zeros_like(image[..., :3], dtype=float)  # background
        mask_display = np.copy(black_image)
        mask_union = np.zeros_like(image[..., :3])
        for i, mask in enumerate(masks):
            mask_display[mask] = mask_colors[i]
            mask_union |= mask[..., None] if mask.ndim == 2 else mask
        imshow(black_image, axes[0, 1])
        imshow(mask_display, axes[1, 0])
        imshow(image * mask_union, axes[1, 1])

    image_axe = axes if masks is None else axes[0, 0]
    imshow(image, image_axe)

    fig.tight_layout(pad=0)
    fig.show()


def interactive_visualizer(ply_path):
    with gr.Blocks() as demo:
        gr.Markdown("# 3D Gaussian Splatting (black-screen loading might take a while)")
        gr.Model3D(
            value=ply_path,  # splat file
            label="3D Scene",
        )
    demo.launch(share=True)

# 计算ply模型的尺寸
def get_ply_scale(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)

    min_vals = pts.min(axis=0)
    max_vals = pts.max(axis=0)

    scale = max_vals - min_vals

    print("模型最小坐标:", min_vals)
    print("模型最大坐标:", max_vals)
    print("模型尺寸 (X, Y, Z):", scale)
    print("长宽高分别是：")
    print(f"X方向: {scale[0]}")
    print(f"Y方向: {scale[1]}")
    print(f"Z方向: {scale[2]}")

# # 通过深度图 + mask + 相机内参 计算模型的真实尺寸
# def compute_realobject_size(
#     depth_path: str,
#     mask_path: str,
#     intrinsics_json: str,
#     depth_scale_default=0.001,
#     percentile_clip=2
# ):
#     """
#     通过深度图 + mask + 相机内参 计算单位化模型的真实尺度缩放系数。
#     返回: scale_factor, real_extent_vec, unit_extent_vec
#     """

#     # -------------------------
#     # 1. 读取深度图 & mask
#     # -------------------------
#     depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     mask = (mask > 128).astype(np.float32)

#     h, w = depth_raw.shape

#     # -------------------------
#     # 2. 读取相机内参
#     # -------------------------
#     with open(intrinsics_json, "r") as f:
#         intr = json.load(f)

#     fx = intr["fx"]
#     fy = intr["fy"]
#     cx = intr["cx"]
#     cy = intr["cy"]
#     depth_scale = intr.get("depth_scale", depth_scale_default)

#     # -------------------------
#     # 3. 深度图转为米并裁剪物体区域
#     # -------------------------
#     depth_m = depth_raw * depth_scale
#     depth_m = depth_m * mask  # 非物体区域置零

#     # 取有效深度
#     valid = depth_m > 0
#     if valid.sum() < 20:
#         raise ValueError("物体有效点太少,mask 或深度图可能错误")

#     # -------------------------
#     # 4. 深度 + 内参 → 真实尺度点云
#     # -------------------------
#     us, vs = np.meshgrid(np.arange(w), np.arange(h))
#     us = us[valid]
#     vs = vs[valid]
#     Z = depth_m[valid]

#     X = (us - cx) * Z / fx
#     Y = (vs - cy) * Z / fy

#     pts_real = np.stack([X, Y, Z], axis=-1)

#     # -------------------------
#     # 5. 去噪：用百分位裁剪点云范围
#     # -------------------------
#     lower = percentile_clip
#     upper = 100 - percentile_clip
#     min_vals = np.percentile(pts_real, lower, axis=0)
#     max_vals = np.percentile(pts_real, upper, axis=0)

#     real_extent_vec = max_vals - min_vals
#     real_size = real_extent_vec.max()

#     print(f"[真实物体尺寸] (米): {real_extent_vec}")
#     return real_extent_vec
