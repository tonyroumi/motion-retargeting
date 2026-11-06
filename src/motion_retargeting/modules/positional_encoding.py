import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding over the joint dimension.

    Input shape:
        [B, W, J, D]

    PE(pos, 2i)   = sin(pos / basis ** (2i / d))
    PE(pos, 2i+1) = cos(pos / basis ** (2i / d))
    """

    def __init__(
        self,
        d: int,
        basis: float = 10000.0,
        max_len: int = 5000,
    ):
        super().__init__()

        if d % 2 != 0:
            raise ValueError(f"d must be even, got {d}")

        self.d = d
        self.basis = basis
        self.max_len = max_len

        self.register_buffer(
            "pe",
            self.build_pe(max_len),
            persistent=False,
        )

    def build_pe(self, max_len: int) -> torch.Tensor:
        position = torch.arange(
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)  # [J, 1]

        div_term = torch.exp(
            torch.arange(
                0,
                self.d,
                2,
                dtype=torch.float32,
            )
            * (-math.log(self.basis) / self.d)
        )  # [D/2]

        pe = torch.zeros(max_len, self.d)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe  # [max_len, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, W, J, D]

        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape [B, W, J, D], got {x.shape}"
            )

        J = x.shape[2]

        if J > self.max_len:
            self.max_len = J
            self.pe = self.build_pe(J).to(x.device)

        pe = self.pe[:J]              # [J, D]
        pe = pe[None, None, :, :]     # [1, 1, J, D]

        return x + pe