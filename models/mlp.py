"""
models/mlp.py
-------------
Shared MLP backbone used by all continual learning methods in this series.

Design decisions:
- Hidden layers configurable so the same architecture can be used for
  lightweight benchmarks and full Split-MNIST experiments.
- get_parameter_names() / num_parameters() support PackNet's per-task
  mask bookkeeping without leaking implementation details into the model.
- Weights initialised with He (Kaiming) initialisation — important for
  ReLU networks; the default PyTorch initialisation is Kaiming uniform,
  which is fine, but we make it explicit for reproducibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Optional


class MLP(nn.Module):
    """
    Fully-connected network with configurable depth and width.

    Parameters
    ----------
    input_dim : int
        Number of input features (784 for flattened 28×28 MNIST).
    hidden_dims : list of int
        Number of units in each hidden layer. Default: [256, 256].
    output_dim : int
        Number of output classes.
    dropout_p : float
        Dropout probability applied after each hidden layer. 0.0 = disabled.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 10,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_p > 0.0:
                layers.append(nn.Dropout(p=dropout_p))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming (He) initialisation for all Linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept both flat (B, D) and image (B, C, H, W) inputs
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_names(self) -> List[str]:
        """Ordered list of named parameter keys — used by PackNet masks."""
        return [name for name, _ in self.named_parameters()]


class MultiHeadMLP(nn.Module):
    """
    MLP with a shared feature extractor and separate output heads per task.

    This architecture is used in task-incremental learning where the task
    identity is known at inference time. Each task head is a separate
    Linear layer; the shared trunk is frozen or regularised by EWC / replay.

    Parameters
    ----------
    input_dim : int
    hidden_dims : list of int
    head_output_dim : int
        Number of classes *per task* head (e.g., 2 for binary tasks).
    dropout_p : float
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: Optional[List[int]] = None,
        head_output_dim: int = 2,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.head_output_dim = head_output_dim

        # Shared trunk
        trunk_layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            trunk_layers.append(nn.Linear(prev_dim, h_dim))
            trunk_layers.append(nn.ReLU())
            if dropout_p > 0.0:
                trunk_layers.append(nn.Dropout(p=dropout_p))
            prev_dim = h_dim

        self.trunk = nn.Sequential(*trunk_layers)
        self.heads = nn.ModuleList()
        self._feature_dim = prev_dim
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def add_task_head(self) -> int:
        """
        Append a new output head for a new task.
        Returns the task index (0-based).
        """
        head = nn.Linear(self._feature_dim, self.head_output_dim)
        nn.init.kaiming_uniform_(head.weight, nonlinearity="relu")
        nn.init.zeros_(head.bias)
        self.heads.append(head)
        return len(self.heads) - 1

    def forward(self, x: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        features = self.trunk(x)
        if task_id >= len(self.heads):
            raise IndexError(
                f"Task {task_id} has no head. "
                f"Call add_task_head() before forward()."
            )
        return self.heads[task_id](features)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_trunk_parameter_names(self) -> List[str]:
        return [f"trunk.{n}" for n, _ in self.trunk.named_parameters()]
