import math

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """Feed-forward MLP with ``num_layers`` hidden layers."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for layer_index in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout / (layer_index + 1)))

        layers += [nn.Linear(hidden_dim, output_dim), nn.Sigmoid()]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class NonNegativeLinear(nn.Module):
    """Linear map whose effective weights remain nonnegative during training."""

    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self.raw_weight = nn.Parameter(torch.empty(output_dim, input_dim))
        self.bias = nn.Parameter(torch.empty(output_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        # Positive weights have a nonzero mean, so a fan-in-sized layer needs a
        # smaller initialization than an ordinary zero-mean linear layer.
        # Squared raw magnitudes in this range produce effective weights in
        # [0.005, 0.015] while keeping the raw scale comparable to an MLP.
        with torch.no_grad():
            self.raw_weight.uniform_(math.sqrt(0.005), math.sqrt(0.015))
            signs = torch.where(
                torch.rand_like(self.raw_weight) < 0.5,
                -torch.ones_like(self.raw_weight),
                torch.ones_like(self.raw_weight),
            )
            self.raw_weight.mul_(signs)
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.raw_weight.shape[1])
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def weight(self):
        return self.raw_weight.square()

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class ICNN(nn.Module):
    """Vector-output input-convex backbone for bounded FSNet predictions.

    Every hidden state is convex in the input: input-to-hidden maps are affine,
    hidden-to-hidden weights are nonnegative, and Softplus is convex and
    nondecreasing. The final sigmoid maps the backbone outputs to ``[0, 1]`` so
    the existing FSNet variable scaling remains valid.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")

        self.num_layers = num_layers
        self.input_layers = nn.ModuleList(
            nn.Linear(input_dim, hidden_dim) for _ in range(num_layers)
        )
        self.hidden_layers = nn.ModuleList(
            NonNegativeLinear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_layers - 1)
        )
        self.hidden_biases = nn.ParameterList(
            nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_layers - 1)
        )
        self.output_input = nn.Linear(input_dim, output_dim)
        self.output_hidden = NonNegativeLinear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        z = F.softplus(self.input_layers[0](x))
        for layer_index in range(1, self.num_layers):
            z = F.softplus(
                self.input_layers[layer_index](x)
                + self.hidden_layers[layer_index - 1](self.dropout(z))
                + self.hidden_biases[layer_index - 1]
            )
        return torch.sigmoid(self.output_input(x) + self.output_hidden(z))


class ResidualMLP(nn.Module):
    """Post-activation MLP with skips across width-preserving hidden layers."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.residual_layers = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.residual_scale = 1.0 / math.sqrt(max(1, num_layers - 1))

    def forward_features(self, x):
        hidden = F.silu(self.input_layer(x))
        for layer in self.residual_layers:
            update = self.dropout(F.silu(layer(hidden)))
            hidden = F.silu(hidden + self.residual_scale * update)
        return hidden

    def forward(self, x):
        return torch.sigmoid(self.output_layer(self.forward_features(x)))


# Short CLI-friendly alias used by the landscape ablation.
ResMLP = ResidualMLP

NETWORK_CLASSES = {
    "MLP": MLP,
    "ICNN": ICNN,
    "ResMLP": ResidualMLP,
    "ResidualMLP": ResidualMLP,
}

NETWORK_IMPLEMENTATION_VERSIONS = {
    "MLP": 1,
    "ICNN": 1,
    "ResMLP": 2,
    "ResidualMLP": 2,
}


def build_network(name, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
    """Construct one of the supported FSNet backbones by its CLI name."""
    try:
        network_class = NETWORK_CLASSES[name]
    except KeyError as error:
        choices = ", ".join(sorted(NETWORK_CLASSES))
        raise ValueError(f"Unknown network type {name!r}; choose one of: {choices}") from error
    return network_class(
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
