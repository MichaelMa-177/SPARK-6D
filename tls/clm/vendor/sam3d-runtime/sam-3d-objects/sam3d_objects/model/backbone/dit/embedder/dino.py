# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
from typing import Optional, Dict, Any
import warnings
import os
import weakref
from pathlib import Path
from torchvision.transforms import Normalize
import torch.nn.functional as F
from loguru import logger


_SHARED_BACKBONES = weakref.WeakValueDictionary()


class Dino(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 224,
        repo_or_dir: str = "facebookresearch/dinov2",
        dino_model: str = "dinov2_vitb14",
        source: str = "github",
        path_to_local_weights: Optional[str] = None,
        backbone_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        # for backward compatible #/data/mmw/test/sam-3d-objects/checkpoints/dino_checkpoints
        prenorm_features: bool = False,
        freeze_backbone: bool = True,
        prune_network: bool = False,  # False for backward compatible
    ):
        super().__init__()
        if backbone_kwargs is None:
            backbone_kwargs = {}
            
        sam3d_root = Path(__file__).resolve().parents[5]
        local_weights_path = Path(
            path_to_local_weights
            or os.environ.get("SAM3D_DINO_WEIGHTS", "")
            or sam3d_root / "checkpoints" / "dino_checkpoints" / "dinov2_vitl14_reg4.pth"
        )
        local_repo_dir = Path(
            os.environ.get("SAM3D_DINO_REPO", "")
            or sam3d_root.parent / "dinov2"
        )
        skip_external_weights = os.environ.get("SAM3D_DINO_SKIP_WEIGHTS", "0") == "1"
        if not local_repo_dir.is_dir():
            raise FileNotFoundError(
                f"DINOv2 source not found: {local_repo_dir}. Set SAM3D_DINO_REPO."
            )
        if not skip_external_weights and not local_weights_path.is_file():
            raise FileNotFoundError(
                f"DINOv2 weights not found: {local_weights_path}. Set SAM3D_DINO_WEIGHTS."
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # logger.info(f"Loading DINO model: {dino_model} from {repo_or_dir} (source: {source})")
            logger.info(f"Building local DINO model: {dino_model}")
            if backbone_kwargs:
                logger.info(f"DINO backbone kwargs: {backbone_kwargs}")
            
            cache_key = (str(local_repo_dir), dino_model, repr(sorted(backbone_kwargs.items())))
            self.backbone = _SHARED_BACKBONES.get(cache_key)
            if self.backbone is None:
                self.backbone = torch.hub.load(
                    repo_or_dir=str(local_repo_dir),
                    model=dino_model,
                    source='local',
                    pretrained=False,
                    verbose=False,
                    **backbone_kwargs,
                )
                if skip_external_weights:
                    _SHARED_BACKBONES[cache_key] = self.backbone
            else:
                logger.info("Reusing identical frozen DINO backbone")
            try:
                if skip_external_weights:
                    logger.info(
                        "Skipping external DINO weights; the enclosing SAM3D checkpoint will load them"
                    )
                    state_dict = None
                else:
                    logger.info(f"Attempting OFFLINE weight load from: {local_weights_path}")
                    state_dict = torch.load(local_weights_path, map_location='cpu', weights_only=False)
                # DINOv2 权重文件可能包含 'state_dict' 键，需要判断
                if state_dict is not None and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                if state_dict is not None:
                    self.backbone.load_state_dict(state_dict)
                    logger.info("Successfully loaded DINO weights from local file.")
                
            except Exception as e:
                logger.error(f"Failed to load DINO weights from local path {local_weights_path}: {e}")
                # 如果本地加载失败，可以根据需求选择是否继续（但通常会重新尝试在线下载）
                raise RuntimeError(f"Failed to load DINO weights locally.")
            
            
            # Log model properties after loading
            logger.info(f"Loaded DINO model - type: {type(self.backbone)}, "
                        f"embed_dim: {self.backbone.embed_dim}, "
                        f"patch_size: {getattr(self.backbone.patch_embed, 'patch_size', 'N/A')}")


        self.resize_input_size = (input_size, input_size)
        self.embed_dim = self.backbone.embed_dim
        self.input_size = input_size
        self.input_channels = 3
        self.normalize_images = normalize_images
        self.prenorm_features = prenorm_features
        self.register_buffer('mean', torch.as_tensor([[0.485, 0.456, 0.406]]).view(-1, 1, 1), persistent=False)
        self.register_buffer('std', torch.as_tensor([[0.229, 0.224, 0.225]]).view(-1, 1, 1), persistent=False)

        # freeze
        if freeze_backbone:
            self.requires_grad_(False)
            self.eval()
        elif not prune_network:
            logger.warning(
                "Unfreeze encoder w/o prune parameter may lead to error in ddp/fp16 training"
            )

        if prune_network:
            self._prune_network()

    def _preprocess_input(self, x):
        _resized_images = torch.nn.functional.interpolate(
            x,
            size=self.resize_input_size,
            mode="bilinear",
            align_corners=False,
        )

        if x.shape[1] == 1:
            _resized_images = _resized_images.repeat(1, 3, 1, 1)

        if self.normalize_images:
            _resized_images = _resized_images.sub_(self.mean).div_(self.std)

        return _resized_images

    def _forward_intermediate_layers(
        self, input_img, intermediate_layers, cls_token=True
    ):
        return self.backbone.get_intermediate_layers(
            input_img,
            intermediate_layers,
            return_class_token=cls_token,
        )

    def _forward_last_layer(self, input_img):
        output = self.backbone.forward_features(input_img)
        if self.prenorm_features:
            features = output["x_prenorm"]
            tokens = F.layer_norm(features, features.shape[-1:])
        else:
            tokens = torch.cat(
                [
                    output["x_norm_clstoken"].unsqueeze(1),
                    output["x_norm_patchtokens"],
                ],
                dim=1,
            )
        return tokens

    def forward(self, x, **kwargs):
        _resized_images = self._preprocess_input(x)
        tokens = self._forward_last_layer(_resized_images)
        return tokens.to(x.dtype)

    def _prune_network(self):
        """
        Ran this script:
        out = model(input)
        loss = out.sum()
        loss.backward()

        for name, p in dino_model.named_parameters():
            if p.grad is None:
                print(name)
        model.zero_grad()
        """
        self.backbone.mask_token = None
        if self.prenorm_features:
            self.backbone.norm = torch.nn.Identity()


class DinoForMasks(torch.nn.Module):
    def __init__(
        self,
        backbone: Dino,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = self.backbone.embed_dim

    def forward(self, image, mask):
        return self.backbone.forward(mask)
