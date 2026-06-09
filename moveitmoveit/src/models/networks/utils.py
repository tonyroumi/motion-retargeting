import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
}

def get_activation(name: str) -> nn.Module:
    """Resolve an activation function by name."""
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation: '{name}'. Choose from {list(ACTIVATIONS.keys())}"
        )
    return ACTIVATIONS[name]()
