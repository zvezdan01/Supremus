"""DeepStack HUNL counterfactual-value network architecture contract.

Primary-source architecture (DeepStack arXiv v3 paper + supplement):
  * postflop input: two 1,000-bucket player ranges + one pot feature = 2001
  * seven fully-connected hidden layers, width 500, PReLU activations
  * postflop output: two 1,000-bucket CFV vectors = 2000
  * differentiable outer zero-sum correction
  * output CFVs are fractions of the pot size

The original HUNL weights, exact bucket centroids/mapping, and exact weight
initialization were not released.  This module is therefore an architecture
and algebra contract, not an original-weight reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .value_bucketing import POSTFLOP_BUCKET_COUNT


@dataclass(frozen=True)
class HUNLValueNetworkSpec:
    bucket_count: int = POSTFLOP_BUCKET_COUNT
    hidden_layers: int = 7
    hidden_width: int = 500
    batch_size: int = 1000
    learning_rate: float = 1e-3
    learning_rate_after_200: float = 1e-4
    learning_rate_drop_epoch: int = 200
    approx_epochs: int = 350

    @property
    def input_size(self) -> int:
        return 2 * self.bucket_count + 1

    @property
    def output_size(self) -> int:
        return 2 * self.bucket_count


class DeepStackHUNLValueNet(nn.Module):
    """Structural reconstruction of the HUNL flop/turn value net.

    PReLU uses one learned slope per hidden layer, matching the released
    DeepStack-Leduc ``nn.PReLU()`` convention.  This scalar-vs-per-unit detail
    is code-anchored by the public author implementation, not explicitly stated
    in the HUNL article.
    """

    def __init__(self, spec: HUNLValueNetworkSpec = HUNLValueNetworkSpec()) -> None:
        super().__init__()
        self.spec = spec
        layers: list[nn.Module] = []
        in_features = spec.input_size
        for _ in range(spec.hidden_layers):
            layers.append(nn.Linear(in_features, spec.hidden_width))
            layers.append(nn.PReLU(num_parameters=1))
            in_features = spec.hidden_width
        layers.append(nn.Linear(in_features, spec.output_size))
        self.feedforward = nn.Sequential(*layers)

    def raw_values(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.spec.input_size:
            raise ValueError(
                f"inputs must have shape [N,{self.spec.input_size}], got {tuple(inputs.shape)}"
            )
        return self.feedforward(inputs)

    def zero_sum_correct(self, raw: Tensor, inputs: Tensor) -> Tensor:
        if raw.shape != (inputs.shape[0], self.spec.output_size):
            raise ValueError("raw output shape mismatch")
        ranges = inputs[:, : self.spec.output_size]
        # Released author net_builder.lua semantics: one weighted sum over the
        # concatenated two-player values/ranges, subtract half from every CFV.
        total_error = torch.sum(raw * ranges, dim=1, keepdim=True)
        return raw - 0.5 * total_error

    def forward(self, inputs: Tensor) -> Tensor:
        return self.zero_sum_correct(self.raw_values(inputs), inputs)

    @property
    def architecture_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
