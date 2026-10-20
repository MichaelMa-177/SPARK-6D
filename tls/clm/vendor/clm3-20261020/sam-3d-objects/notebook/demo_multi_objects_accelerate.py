import os
import uuid
import argparse
import imageio
import numpy as np

import torch
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.distributed as dist
import torch

from inference import (
    Inference, ready_gaussian_for_video_rendering, 
    load_image, load_masks, display_image,
    make_scene, render_video, interactive_visualizer
)

PATH = os.getcwd()
TAG = "hf"
config_path = f"{PATH}/checkpoints/{TAG}/pipeline.yaml"

def run_inference_with_accelerate():
    # -------------------------------------------------------
    # Initialize accelerator
    # -------------------------------------------------------
    accelerator = Accelerator()   # auto detect and use all GPUs
    device = accelerator.device

    # ---------------------------
    # Load data (CPU side)
    # ---------------------------
    IMAGE_PATH = f"{PATH}/notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png"
    IMAGE_NAME = os.path.basename(os.path.dirname(IMAGE_PATH))

    image = load_image(IMAGE_PATH)  # numpy / PIL → CPU
    masks = load_masks(os.path.dirname(IMAGE_PATH), extension=".png")

    # only rank 0 display once
    if accelerator.is_main_process:
        display_image(image, masks)

    # -------------------------------------------------------
    # Scatter masks across GPUs
    # -------------------------------------------------------
    # Split masks across devices: each GPU handles a sub-list
    masks = list(masks)
    chunks = np.array_split(masks, accelerator.num_processes)
    local_masks = chunks[accelerator.process_index]

    # -------------------------------------------------------
    # Build model only once per process
    # -------------------------------------------------------
    inference = Inference(config_path, compile=False)
    inference = accelerator.prepare(inference)  # move to correct GPU

    # -------------------------------------------------------
    # Run inference for each mask on this GPU
    # -------------------------------------------------------
    local_outputs = []
    for mask in local_masks:
        out = inference(image, mask, seed=42)
        local_outputs.append(out)

    # -------------------------------------------------------
    # Gather all results to main process
    # -------------------------------------------------------
    outputs = accelerator.gather(local_outputs)

    # Rank-0 will process results
    if accelerator.is_main_process:
        print(f"Total outputs collected = {len(outputs)}")

        # -------------------------
        # Build scene
        # -------------------------
        scene_gs = make_scene(*outputs)
        scene_gs = ready_gaussian_for_video_rendering(scene_gs)

        # Save PLY
        out_dir = f"{PATH}/notebook/gaussians/multi"
        os.makedirs(out_dir, exist_ok=True)
        ply_path = f"{out_dir}/{IMAGE_NAME}.ply"
        scene_gs.save_ply(ply_path)

        # Render video
        video = render_video(
            scene_gs,
            r=1,
            fov=60,
            resolution=512,
        )["color"]

        # Save GIF
        gif_path = f"{out_dir}/{IMAGE_NAME}.gif"
        imageio.mimsave(
            gif_path,
            video,
            format="GIF",
            duration=1000 / 30,
            loop=0,
        )

        from IPython.display import Image as ImageDisplay
        ImageDisplay(url=f"gaussians/multi/{IMAGE_NAME}.gif?cache_invalidator={uuid.uuid4()}")

        interactive_visualizer(ply_path)


if __name__ == "__main__":
    run_inference_with_accelerate()
