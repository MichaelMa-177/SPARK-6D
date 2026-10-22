from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 128
    latent_side: int = 16
    latent_channels: int = 8
    memory_side: int = 8
    observed_tokens: int = 256
    width: int = 96
    heads: int = 4
    decoder_layers: int = 2
    memory_mode: str = "spatial_latent"
    min_correspondences: int = 6
    solver_mode: str = "weighted"
    membership_threshold: float = .5
    inlier_radius_m: float = .01
    robust_iterations: int = 4
    foundation_feature_dim: int = 0
    coordinate_mode: str = "direct"
    membership_mode: str = "learned"
    consensus_trials: int = 256
    shape_geometry_channels: int = 0
    contextual_features: bool = False
    rigidity_weight: float = 0.

    def __post_init__(self):
        if self.heads < 1:
            raise ValueError("heads must be positive")
        if self.image_size < 16 or self.image_size % 4:
            raise ValueError("image_size must be a multiple of 4 and >= 16")
        if self.width < 16 or self.width % 8 or self.width % self.heads:
            raise ValueError("width must be divisible by 8 and heads")
        if not 1 <= self.memory_side <= self.latent_side:
            raise ValueError("memory_side must be between 1 and latent_side")
        if min(self.observed_tokens, self.decoder_layers, self.latent_channels) < 1:
            raise ValueError("token count, channels and layer count must be positive")
        if self.memory_mode not in {"spatial_latent", "geometry_only", "observed_only"}:
            raise ValueError("unknown memory_mode")
        if self.min_correspondences < 3:
            raise ValueError("at least three correspondences are required")
        if self.solver_mode not in {"weighted", "robust", "consensus"}:
            raise ValueError("unknown solver_mode")
        if not 0 <= self.membership_threshold <= 1 or self.inlier_radius_m <= 0 or self.robust_iterations < 1:
            raise ValueError("invalid robust solver settings")
        if self.foundation_feature_dim < 0:
            raise ValueError("foundation_feature_dim cannot be negative")
        if self.coordinate_mode not in {"direct", "memory_residual", "equivariant_memory"}:
            raise ValueError("unknown coordinate_mode")
        if self.coordinate_mode in {"memory_residual", "equivariant_memory"} and not self.foundation_feature_dim:
            raise ValueError("memory coordinate readout requires frozen foundation descriptors")
        if self.membership_mode not in {"learned", "appearance_prior"}:
            raise ValueError("unknown membership_mode")
        if self.membership_mode == "appearance_prior" and self.coordinate_mode not in {"memory_residual", "equivariant_memory"}:
            raise ValueError("appearance_prior requires memory coordinate readout")
        if self.consensus_trials < 1:
            raise ValueError("consensus_trials must be positive")
        if self.shape_geometry_channels not in {0, 2}:
            raise ValueError("shape_geometry_channels must be 0 or 2")
        if self.shape_geometry_channels and self.memory_mode == "observed_only":
            raise ValueError("shape geometry requires generated memory")
        if not isinstance(self.contextual_features, bool):
            raise ValueError("contextual_features must be boolean")
        if not 0 <= self.rigidity_weight <= 1:
            raise ValueError("rigidity_weight must lie in [0,1]")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)
