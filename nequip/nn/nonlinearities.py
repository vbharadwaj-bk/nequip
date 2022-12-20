from typing import Callable

import math

import torch

from e3nn.util.jit import compile_mode


@torch.jit.script
def shifted_soft_plus(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x) - math.log(2.0)


@compile_mode("script")
class ShiftedSoftPlus(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return shifted_soft_plus(x)


def get_nonlinearity(name: str) -> Callable:
    if name == "abs":
        return torch.abs
    elif name == "tanh":
        return torch.nn.Tanh()
    elif name == "ssp":
        return ShiftedSoftPlus()
    elif name == "silu":
        return torch.nn.SiLU()
    else:
        raise KeyError(f"No such nonlinearity `{name}`")
