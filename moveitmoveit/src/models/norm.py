import numpy as np
import torch
import torch.nn as nn

class EmpiricalNorm(nn.Module):
    def __init__(
        self,
        shape,
        init_mean=None,
        init_std=None,
        min_std: float = 1e-4,
        clip: float = np.inf,
        max_samples: int = 100_000_000,
        dtype=torch.float32,
    ):
        super().__init__()

        self._min_var = min_std * min_std
        self._clip = clip
        self._max_samples = max_samples
        self.dtype = dtype

        mean = (
            torch.zeros(shape, dtype=dtype)
            if init_mean is None
            else torch.as_tensor(init_mean, dtype=dtype)
        )

        std = (
            torch.ones(shape, dtype=dtype)
            if init_std is None
            else torch.as_tensor(init_std, dtype=dtype)
        )

        self.register_buffer("_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_mean", mean.clone())
        self.register_buffer("_std", std.clone())
        self.register_buffer("_mean_sq", self._calc_mean_sq(mean, std))

        self.register_buffer("_new_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_new_sum", torch.zeros_like(self._mean))
        self.register_buffer("_new_sum_sq", torch.zeros_like(self._mean))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = (x - self._mean) / self._std

        if np.isfinite(self._clip):
            norm_x = torch.clamp(norm_x, -self._clip, self._clip)

        return norm_x

    def unnormalize(self, norm_x: torch.Tensor) -> torch.Tensor:
        norm_x = norm_x.to(device=self._mean.device, dtype=self.dtype)
        return norm_x * self._std + self._mean

    @torch.no_grad()
    def record(self, x: torch.Tensor) -> None:
        if self._count.item() >= self._max_samples:
            return

        x = x.to(device=self._mean.device, dtype=self.dtype)

        shape = self.get_shape()
        x = x.flatten(start_dim=0, end_dim=x.ndim - len(shape) - 1)

        remaining = self._max_samples - self._count.item()
        if x.shape[0] > remaining:
            x = x[:remaining]

        self._new_count.add_(x.shape[0])
        self._new_sum.add_(x.sum(dim=0))
        self._new_sum_sq.add_(x.square().sum(dim=0))

    @torch.no_grad()
    def update(self) -> None:
        new_count = self._new_count

        if new_count.item() == 0:
            return

        new_mean = self._new_sum / new_count
        new_mean_sq = self._new_sum_sq / new_count

        total_count = self._count + new_count

        w_old = self._count.to(self.dtype) / total_count.to(self.dtype)
        w_new = new_count.to(self.dtype) / total_count.to(self.dtype)

        self._mean.copy_(w_old * self._mean + w_new * new_mean)
        self._mean_sq.copy_(w_old * self._mean_sq + w_new * new_mean_sq)
        self._std.copy_(self._calc_std(self._mean, self._mean_sq))

        self._count.copy_(total_count)

        self._new_count.zero_()
        self._new_sum.zero_()
        self._new_sum_sq.zero_()

    def get_shape(self) -> torch.Size:
        return self._mean.shape

    @staticmethod
    def _calc_mean_sq(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return std.square() + mean.square()

    def _calc_std(self, mean: torch.Tensor, mean_sq: torch.Tensor) -> torch.Tensor:
        var = torch.clamp(mean_sq - mean.square(), min=self._min_var)
        return torch.sqrt(var)