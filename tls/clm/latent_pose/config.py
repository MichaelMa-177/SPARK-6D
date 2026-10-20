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

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)
