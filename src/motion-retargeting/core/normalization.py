from dataclasses import dataclass
import torch

@dataclass
class NormalizationStats:
    mean: torch.Tensor
    var: torch.Tensor

    def norm(self, x: torch.Tensor):
        idx = self.var < 1e-5 # Don't clip tiny variances
        self.var[idx] = 1
        return (x - self.mean) / (self.var)

    def denorm(self, x):
        return x * self.var + self.mean