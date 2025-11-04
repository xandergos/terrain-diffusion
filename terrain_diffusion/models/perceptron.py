import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


def _activation(name: str) -> nn.Module:
    name = (name or "silu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "identity":
        return nn.Identity()
    return nn.SiLU()


class Perceptron(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims=(128, 128),
        activation: str = "silu",
        final_activation: str | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        dims = [in_dim, *(list(hidden_dims) if isinstance(hidden_dims, (list, tuple)) else [hidden_dims]), out_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
            is_last = i == len(dims) - 2
            if not is_last:
                layers.append(_activation(activation))
                if dropout and dropout > 0:
                    layers.append(nn.Dropout(dropout))

        if final_activation is not None:
            layers.append(_activation(final_activation))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.ndim > 2:
            x = x.reshape(-1, orig_shape[-1])
            y = self.net(x)
            return y.reshape(*orig_shape[:-1], self.config.out_dim)
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


