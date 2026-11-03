"""Empirical ETAS kernels and nonnegative neural kernels.

The neural stacks match the TensorFlow layers: ``size_layer`` hidden widths,
then a scalar output. Hidden and output kernels are initialized with
``abs(Glorot uniform)`` and projected to be nonnegative after every optimizer
step (Keras ``NonNeg`` is a post-update constraint, not a gradient through
``abs``). Temporal and spatial networks have no bias and use ``tanh``. The
productivity network uses a bias and a ``softplus`` output.

Each network owns a generator seeded with the same integer, so the three
stacks start from the same Glorot stream, as they do when each Keras
initializer is constructed with that seed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _generator(seed: int | None) -> torch.Generator:
    generator = torch.Generator()
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(int(seed))
    return generator


def _init_abs_glorot(linear: nn.Linear, generator: torch.Generator) -> None:
    fan_in = linear.weight.shape[1]
    fan_out = linear.weight.shape[0]
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        linear.weight.uniform_(-limit, limit, generator=generator)
        linear.weight.abs_()
        if linear.bias is not None:
            linear.bias.zero_()


class BackgroundFunction(nn.Module):
    """Constant background density ``mu`` (events per day per km^2).

    ``mu`` is a parameter so it is saved with the checkpoint, but the training
    loop does not put it in Adam. It is overwritten at the end of each epoch
    by the declustering update in ``KernelPointProcess.update_mu``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mu = nn.Parameter(torch.tensor([[[1.0]]], dtype=torch.float32))

    def project(self) -> None:
        self.mu.data.clamp_(min=1e-9)


class ProductionFunction(nn.Module):
    """ETAS productivity ``A * exp(alpha * (m - m0))``."""

    def __init__(self, input_m0: float) -> None:
        super().__init__()
        self.mag_A = nn.Parameter(torch.tensor(3.5, dtype=torch.float32))
        self.mag_alpha = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
        self.register_buffer(
            "mag_m0", torch.tensor(float(input_m0), dtype=torch.float32)
        )

    def forward(self, ht_mag: torch.Tensor) -> torch.Tensor:
        return self.mag_A * torch.exp(self.mag_alpha * (ht_mag - self.mag_m0))

    def project(self) -> None:
        self.mag_A.data.clamp_(min=1e-5)
        self.mag_alpha.data.clamp_(min=1e-5)


class OmoriFunction(nn.Module):
    """Modified Omori CDF ``1 - (1 + t / c) ** (1 - p)``."""

    def __init__(self) -> None:
        super().__init__()
        self.omori_c = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.omori_p = nn.Parameter(torch.tensor(1.02, dtype=torch.float32))

    def forward(self, interval_time: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.pow(1.0 + interval_time / self.omori_c, -self.omori_p + 1.0)

    def project(self) -> None:
        self.omori_c.data.clamp_(min=1e-5)
        self.omori_p.data.clamp_(min=1.0 + 1e-5)


class SpatialFunction(nn.Module):
    """ETAS spatial kernel in the plane.

    ``dist_2`` is squared epicentral distance in km^2. Returns ``(pdf, cdf)``
    of the isotropic kernel used by the TensorFlow ``SpatialFunction``.
    """

    def __init__(self, input_m0: float) -> None:
        super().__init__()
        self.spatial_q = nn.Parameter(torch.tensor(1.2, dtype=torch.float32))
        self.spatial_D2 = nn.Parameter(torch.tensor(0.001, dtype=torch.float32))
        self.spatial_gamma = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.register_buffer(
            "spatial_m0", torch.tensor(float(input_m0), dtype=torch.float32)
        )

    def forward(
        self, dist_2: torch.Tensor, hist_m: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.spatial_D2 * torch.exp(
            self.spatial_gamma * (hist_m - self.spatial_m0)
        )
        base = 1.0 + dist_2 / scale
        pdf = (self.spatial_q - 1.0) / (math.pi * scale) * torch.pow(
            base, -self.spatial_q
        )
        cdf = 1.0 - torch.pow(base, 1.0 - self.spatial_q)
        return pdf, cdf

    def project(self) -> None:
        self.spatial_q.data.clamp_(min=1.0 + 1e-5)
        self.spatial_D2.data.clamp_(min=1e-5)
        self.spatial_gamma.data.clamp_(min=1e-5)


class _NonnegativeMLP(nn.Module):
    def __init__(
        self,
        size_nn: int,
        size_layer: int,
        seed: int | None,
        use_bias: bool,
        output_activation: str,
    ) -> None:
        super().__init__()
        if size_layer < 1:
            raise ValueError("size_layer must be >= 1")
        if size_nn < 1:
            raise ValueError("size_nn must be >= 1")
        generator = _generator(seed)
        hidden: list[nn.Linear] = []
        in_features = 1
        for _ in range(size_layer):
            layer = nn.Linear(in_features, size_nn, bias=use_bias)
            _init_abs_glorot(layer, generator)
            hidden.append(layer)
            in_features = size_nn
        self.hidden = nn.ModuleList(hidden)
        self.output = nn.Linear(in_features, 1, bias=use_bias)
        _init_abs_glorot(self.output, generator)
        if output_activation not in ("tanh", "softplus"):
            raise ValueError(output_activation)
        self.output_activation = output_activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        for layer in self.hidden:
            x = torch.tanh(layer(x))
        y = self.output(x)
        if self.output_activation == "tanh":
            return torch.tanh(y)
        return F.softplus(y)

    def project(self) -> None:
        for layer in list(self.hidden) + [self.output]:
            layer.weight.data.clamp_(min=0.0)


class TemporalKernelNetwork(_NonnegativeMLP):
    def __init__(self, size_nn: int, size_layer: int, seed: int | None = None) -> None:
        super().__init__(size_nn, size_layer, seed, use_bias=False, output_activation="tanh")


class SpatialKernelNetwork(_NonnegativeMLP):
    def __init__(self, size_nn: int, size_layer: int, seed: int | None = None) -> None:
        super().__init__(size_nn, size_layer, seed, use_bias=False, output_activation="tanh")


class KappaNetwork(_NonnegativeMLP):
    def __init__(self, size_nn: int, size_layer: int, seed: int | None = None) -> None:
        super().__init__(size_nn, size_layer, seed, use_bias=True, output_activation="softplus")
