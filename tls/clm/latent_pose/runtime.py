"""One object memory shared across independent RGB-D video sessions."""

from dataclasses import dataclass
import math
import time

import torch

from .cli import load_model
from .model import ObjectMemory
from .rgbd import checked_box, prepare_query, prepare_reference, reference_crop_box
from .retrieval import propose_region


def tensor_packet(packet, device):
    return {k: torch.as_tensor(v, device=device).unsqueeze(0)
            for k, v in packet.items() if k not in {"schema_version", "units"}}


@dataclass(frozen=True)
class RegisteredObject:
    object_id: str
    memory: ObjectMemory
    mask_source: str
    retrieval_reference: dict | None = None
    shape_provenance: dict | None = None


class PoseRuntime:
    """Inference API; use a new runtime after changing model weights.

    The raw model constructor is intended for research and tests. Production
    callers should use from_checkpoint(), which retains the untrained-weight
    guard. Training steps alone do not establish accuracy or generalization.
    """

    def __init__(self, model, feature_extractor=None, retrieve_region=False, shape_encoder=None):
        if model.training:
            raise ValueError("runtime requires model.eval()")
        self.model = model
        if model.config.foundation_feature_dim and feature_extractor is None:
            raise ValueError("this checkpoint requires its frozen RGB feature extractor")
        self.feature_extractor = feature_extractor
        if retrieve_region and not model.config.foundation_feature_dim:
            raise ValueError("region retrieval requires a frozen feature extractor")
        self.retrieve_region = retrieve_region
        self.shape_encoder = shape_encoder
        self.objects = {}

    @classmethod
    def from_checkpoint(cls, checkpoint, device="cpu", allow_untrained=False,
                        feature_extractor=None, retrieve_region=False, shape_encoder=None, solver=None):
        model, metadata = load_model(checkpoint, device, allow_untrained, solver_mode=solver)
        expected = metadata.get("foundation_features", {}).get("weights_sha256")
        if model.config.foundation_feature_dim and expected and expected != getattr(feature_extractor, "weights_sha256", None):
            raise ValueError("feature extractor weights differ from the training features")
        expected_shape = (metadata.get("shape_prior") or {}).get("weight_sha256")
        if shape_encoder is not None and expected_shape and expected_shape != shape_encoder.provenance.get("weight_sha256"):
            raise ValueError("shape encoder weights differ from training priors")
        return cls(model, feature_extractor, retrieve_region, shape_encoder)

    def add_features(self, packet, rgb, box):
        if self.model.config.foundation_feature_dim:
            x0, y0, x1, y1 = box
            features = self.feature_extractor(rgb[y0:y1, x0:x1], self.model.config.image_size)
            packet["foundation_features"] = torch.as_tensor(features).float()

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def register_object(self, object_id, rgb, depth, mask, intrinsics, depth_scale_m,
                        *, mask_source, shape_latent=None, latent_xyz_m=None, shape_geometry=None):
        if not isinstance(object_id, str) or not object_id.strip():
            raise ValueError("object_id must be a nonempty string")
        if object_id in self.objects:
            raise ValueError("object already registered; preserve its fixed reference frame")
        if not isinstance(mask_source, str) or not mask_source.strip():
            raise ValueError("record the actual supplied mask source")
        shape_provenance = None
        if self.model.config.memory_mode != "observed_only" and shape_latent is None and latent_xyz_m is None and self.shape_encoder is not None:
            prior, shape_provenance = self.shape_encoder(rgb, depth, mask, intrinsics, depth_scale_m)
            shape_latent, latent_xyz_m = prior["shape_latent"], prior["latent_xyz_m"]
            shape_geometry = prior.get("shape_geometry")
        packet = prepare_reference(rgb, depth, mask, intrinsics, depth_scale_m,
                                   self.model.config, shape_latent, latent_xyz_m,
                                   shape_geometry=shape_geometry)
        self.add_features(packet, rgb, reference_crop_box(mask))
        tensors = tensor_packet(packet, self.device)
        memory = self.model.build_memory(tensors, cache=True)
        retrieval = {k: tensors[k] for k in ("valid", "mask", "foundation_features", "scale_m")} if self.retrieve_region else None
        self.objects[object_id] = RegisteredObject(object_id, memory, mask_source, retrieval, shape_provenance)
        return self.objects[object_id]

    def start_video(self, video_id, object_id):
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError("video_id must be a nonempty string")
        if object_id not in self.objects:
            raise ValueError("register the object before starting a video")
        return VideoSession(self, video_id, self.objects[object_id])


class VideoSession:
    """Independent localizations using a persistent object memory.

    Only stream bookkeeping is stateful today. No learned temporal refinement,
    pose smoothing, or online memory writes are implemented by this baseline.
    Each stream supplies its own calibration and timestamps.
    """

    def __init__(self, runtime, video_id, registered):
        self.runtime, self.video_id, self.registered = runtime, video_id, registered
        self.frame_index = 0
        self.last_timestamp_s = None

    @torch.inference_mode()
    def estimate(self, rgb, depth, intrinsics, depth_scale_m, timestamp_s, box=None):
        timestamp_s = float(timestamp_s)
        if not math.isfinite(timestamp_s) or (
            self.last_timestamp_s is not None and timestamp_s <= self.last_timestamp_s
        ):
            raise ValueError("timestamps must be finite and strictly increasing within each video")
        model = self.runtime.model
        if model.training:
            raise ValueError("runtime requires model.eval(); rebuild memory after changing weights")
        started = time.perf_counter()
        proposal = None
        if self.runtime.retrieve_region and box is None:
            coarse = prepare_query(rgb, depth, intrinsics, depth_scale_m, model.config)
            self.runtime.add_features(coarse, rgb, checked_box(None, rgb.shape))
            box, proposal = propose_region(self.registered.retrieval_reference,
                                           tensor_packet(coarse, self.runtime.device),
                                           rgb.shape[:2], intrinsics)
        query = prepare_query(rgb, depth, intrinsics, depth_scale_m, model.config, box)
        self.runtime.add_features(query, rgb, checked_box(box, rgb.shape))
        result = model.decode(tensor_packet(query, self.runtime.device), self.registered.memory)
        valid = bool(result["pose_valid"][0])
        output = {
            "object_id": self.registered.object_id, "video_id": self.video_id,
            "frame_index": self.frame_index, "timestamp_s": timestamp_s,
            "mode": "retrieval_then_localization" if proposal is not None else "independent_localization",
            "pose_valid": valid,
            "pose": result["pose"][0].cpu().numpy() if valid else None,
            "residual_m": float(result["residual_m"][0]) if valid else None,
            "effective_correspondences": float(result["effective_correspondences"][0]),
            "query_box_xyxy": checked_box(box, rgb.shape),
            "region_proposal": proposal,
            "raw_frame_to_output_ms": (time.perf_counter() - started) * 1000,
        }
        self.frame_index += 1
        self.last_timestamp_s = timestamp_s
        return output
