import os
import uuid
import argparse
import imageio
import numpy as np
from IPython.display import Image as ImageDisplay

from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.distributed as dist
import torch
###自定义数据读取类导入


from inference import Inference, ready_gaussian_for_video_rendering, load_image, load_masks, display_image, make_scene, render_video, interactive_visualizer
PATH = os.getcwd()
TAG = "hf"
config_path = f"{PATH}/checkpoints/{TAG}/pipeline.yaml"
inference = Inference(config_path, compile=False)
IMAGE_PATH = f"{PATH}/notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png"
IMAGE_NAME = os.path.basename(os.path.dirname(IMAGE_PATH))

image = load_image(IMAGE_PATH)
masks = load_masks(os.path.dirname(IMAGE_PATH), extension=".png")
display_image(image, masks)
outputs = [inference(image, mask, seed=42) for mask in masks]
scene_gs = make_scene(*outputs)
scene_gs = ready_gaussian_for_video_rendering(scene_gs)

# export gaussian splatting (as point cloud)
scene_gs.save_ply(f"{PATH}/notebook/gaussians/multi/{IMAGE_NAME}.ply")

video = render_video(
    scene_gs,
    r=1,
    fov=60,
    resolution=512,
)["color"]

# save video as gif
imageio.mimsave(
    os.path.join(f"{PATH}/notebook/gaussians/multi/{IMAGE_NAME}.gif"),
    video,
    format="GIF",
    duration=1000 / 30,  # default assuming 30fps from the input MP4
    loop=0,  # 0 means loop indefinitely
)

# notebook display
ImageDisplay(url=f"gaussians/multi/{IMAGE_NAME}.gif?cache_invalidator={uuid.uuid4()}",)
# might take a while to load (black screen)
interactive_visualizer(f"{PATH}/gaussians/multi/{IMAGE_NAME}.ply")