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
        random_ensembles: int = 1,
    ):
        super().__init__()

        self.random_ensembles = random_ensembles
        self.nets = nn.ModuleList([])
        for i in range(random_ensembles):
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

            net = nn.Sequential(*layers)
            self.nets.append(net)

    def forward(self, x: torch.Tensor, ensemble_idx: torch.Tensor | None = None) -> torch.Tensor:
        """
        Optionally route each sample through a different ensemble member.
        - If ensemble_idx is None: choose a random ensemble index per sample.
        - If provided: expects shape (B,) with values in [0, random_ensembles).
        """
        if self.random_ensembles == 1:
            return self.nets[0](x)
        
        batch_size = x.shape[0]
        if ensemble_idx is None:
            ensemble_idx = torch.randint(0, self.random_ensembles, (batch_size,), device=x.device)
        else:
            if ensemble_idx.dim() != 1 or ensemble_idx.shape[0] != batch_size:
                raise ValueError(f"ensemble_idx must be shape (B,), got {tuple(ensemble_idx.shape)}")
        
        out_dtype = self.nets[0](torch.zeros((1, self.config.in_dim), device=x.device)).dtype
        out = torch.zeros((batch_size, self.config.out_dim), device=x.device, dtype=out_dtype)
        for k in range(self.random_ensembles):
            sel = (ensemble_idx == k).nonzero(as_tuple=False).flatten()
            if sel.numel() == 0:
                continue
            out[sel] = self.nets[k](x[sel])
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


