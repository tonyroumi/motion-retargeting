from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from moveitmoveit.src.models.networks.utils import get_activation


class BaseMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_layers: List[int] = [256, 256],
        activation: str = "relu",
        weight_init: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_layers = hidden_layers
        self.activation = activation

        trunk_layers = []
        dims = [in_channels] + hidden_layers
        for i in range(len(dims) - 1):
            trunk_layers.append(nn.Linear(dims[i], dims[i + 1]))
            trunk_layers.append(get_activation(activation))
        self.trunk = nn.Sequential(*trunk_layers)
        self.output_head = nn.Linear(dims[-1], out_channels)

        if weight_init is not None:
            self.apply_weight_init(weight_init)

    def apply_weight_init(self, config: Dict[str, Any]) -> None:
        """Initialize selected linear layers according to config."""
        init_type = config.get("type")
        init_type = init_type.lower() if isinstance(init_type, str) else init_type

        if init_type in ("ortho", "orthogonal"):
            init_fn = self._init_linear_orthogonal
        elif init_type == "uniform":
            init_fn = self._init_linear_uniform
        else:
            raise ValueError(
                f"Unknown weight initialization: {init_type}. "
                "Supported: 'orthogonal', 'uniform'."
            )

        trunk_config: Dict[str, Any] = config.get("trunk") or {}
        output_config: Dict[str, Any] = config.get("output") or {}

        for module in self.trunk.modules():
            if isinstance(module, nn.Linear):
                if trunk_config.get("enabled"):
                    init_fn(module, trunk_config.get("gain", 1.0))
                if trunk_config.get("zero_bias", False):
                    self._init_linear_bias_zeros(module)

        if output_config.get("enabled"):
            init_fn(self.output_head, output_config.get("gain", 1.0))
        if output_config.get("zero_bias", False):
            self._init_linear_bias_zeros(self.output_head)
    
    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    @staticmethod
    def _init_linear_orthogonal(
        module: nn.Linear,
        gain: float,
    ) -> None:
        nn.init.orthogonal_(module.weight, gain=gain)

    @staticmethod
    def _init_linear_uniform(
        module: nn.Linear,
        gain: float,
    ) -> None:
        nn.init.uniform_(module.weight, -gain, gain)

    @staticmethod
    def _init_linear_bias_zeros(module: nn.Linear) -> None:
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_head(self.trunk(x))

    def get_logit_weights(self) -> torch.Tensor:
        """Return the weight matrix of the output / logit head."""
        return torch.flatten(self.output_head.weight)
