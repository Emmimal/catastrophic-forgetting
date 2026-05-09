"""
data/task_data.py
-----------------
Generates the Split-MNIST continual learning benchmark.

Split-MNIST divides MNIST's 10 classes into 5 sequential tasks, each
containing 2 classes. The model sees Task 0 first, then Task 1, ..., Task 4.
After training on Task 4, we measure how much it still remembers about Task 0.

This is the standard benchmark for catastrophic forgetting research
(Kirkpatrick et al., 2017; Zenke et al., 2017).

Usage
-----
    from data.task_data import SplitMNIST

    benchmark = SplitMNIST(data_dir="~/.pytorch/mnist", num_tasks=5)
    train_loader_task0 = benchmark.get_train_loader(task_id=0, batch_size=64)
    test_loader_task0  = benchmark.get_test_loader(task_id=0)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

logger = logging.getLogger(__name__)


# ── Default class groupings for Split-MNIST ───────────────────────────────
# Task 0: digits 0, 1
# Task 1: digits 2, 3
# Task 2: digits 4, 5
# Task 3: digits 6, 7
# Task 4: digits 8, 9
DEFAULT_TASK_CLASSES = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
]


def _remap_labels(labels: torch.Tensor, classes: List[int]) -> torch.Tensor:
    """
    Remap original class labels to task-local indices (0, 1, ...).

    Example: classes=[4, 5] → label 4 becomes 0, label 5 becomes 1.
    This is required for multi-head architectures where each task head
    has head_output_dim outputs, not 10.
    """
    remapped = torch.zeros_like(labels)
    for local_idx, global_cls in enumerate(classes):
        remapped[labels == global_cls] = local_idx
    return remapped


class SplitMNIST:
    """
    Split-MNIST continual learning benchmark.

    Downloads MNIST once and creates task-specific DataLoaders on demand.
    Labels within each task are remapped to [0, len(classes)-1] so that
    multi-head models can use a small per-task output layer.

    Parameters
    ----------
    data_dir : str or Path
        Directory where MNIST will be downloaded and cached.
    num_tasks : int
        Number of tasks (max 5 for standard Split-MNIST).
    task_classes : list of list of int, optional
        Override the default class groupings.
    seed : int
        RNG seed for reproducible shuffling.
    flatten : bool
        If True, return (B, 784) tensors. If False, return (B, 1, 28, 28).
    """

    def __init__(
        self,
        data_dir: str = "~/.pytorch/mnist",
        num_tasks: int = 5,
        task_classes: Optional[List[List[int]]] = None,
        seed: int = 42,
        flatten: bool = True,
    ) -> None:
        self.num_tasks = num_tasks
        self.seed = seed
        self.flatten = flatten

        if task_classes is not None:
            self.task_classes = task_classes[:num_tasks]
        else:
            self.task_classes = DEFAULT_TASK_CLASSES[:num_tasks]

        if len(self.task_classes) < num_tasks:
            raise ValueError(
                f"Only {len(self.task_classes)} task class groups provided "
                f"for {num_tasks} tasks."
            )

        self._train_data, self._train_targets = self._load_mnist(
            data_dir, train=True
        )
        self._test_data, self._test_targets = self._load_mnist(
            data_dir, train=False
        )

        logger.info(
            "SplitMNIST initialised | %d tasks | train=%d test=%d",
            num_tasks,
            len(self._train_targets),
            len(self._test_targets),
        )

    # ── Public API ────────────────────────────────────────────────────────

    def get_train_loader(
        self,
        task_id: int,
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> DataLoader:
        """DataLoader for training data of a specific task."""
        dataset = self._make_task_dataset(
            self._train_data, self._train_targets, task_id
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_test_loader(
        self,
        task_id: int,
        batch_size: int = 256,
    ) -> DataLoader:
        """DataLoader for test data of a specific task."""
        dataset = self._make_task_dataset(
            self._test_data, self._test_targets, task_id
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def get_all_test_loaders(self, batch_size: int = 256) -> List[DataLoader]:
        """Returns test loaders for all tasks (used in evaluation)."""
        return [
            self.get_test_loader(task_id, batch_size)
            for task_id in range(self.num_tasks)
        ]

    def task_class_count(self, task_id: int) -> int:
        """Number of classes in a given task."""
        return len(self.task_classes[task_id])

    def task_info(self) -> List[Dict]:
        """Human-readable summary of all tasks."""
        return [
            {"task_id": i, "classes": self.task_classes[i]}
            for i in range(self.num_tasks)
        ]

    # ── Internal helpers ──────────────────────────────────────────────────

    def _make_task_dataset(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        task_id: int,
    ) -> TensorDataset:
        """Filter and remap data for a single task."""
        classes = self.task_classes[task_id]
        mask = torch.zeros(len(targets), dtype=torch.bool)
        for cls in classes:
            mask |= targets == cls

        task_data = data[mask]
        task_targets = _remap_labels(targets[mask], classes)

        if self.flatten:
            task_data = task_data.view(task_data.size(0), -1).float() / 255.0
        else:
            task_data = task_data.unsqueeze(1).float() / 255.0

        return TensorDataset(task_data, task_targets)

    @staticmethod
    def _load_mnist(
        data_dir: str,
        train: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load MNIST from torchvision. Returns raw uint8 tensors.
        """
        try:
            from torchvision import datasets
        except ImportError as exc:
            raise ImportError(
                "torchvision is required. Install with: pip install torchvision"
            ) from exc

        path = Path(data_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)

        dataset = datasets.MNIST(
            root=str(path),
            train=train,
            download=True,
        )
        return dataset.data, dataset.targets


# ── Permuted MNIST (alternative benchmark) ───────────────────────────────


class PermutedMNIST:
    """
    Permuted-MNIST benchmark: each task applies a fixed random pixel
    permutation to all MNIST images.

    Unlike Split-MNIST, all tasks share the same 10-class output space.
    Suited for domain-incremental learning scenarios where task identity
    is NOT assumed to be available at test time.

    Parameters
    ----------
    num_tasks : int
        Number of tasks (permutations).
    data_dir : str or Path
    seed : int
        RNG seed used to generate permutation matrices.
    """

    def __init__(
        self,
        num_tasks: int = 5,
        data_dir: str = "~/.pytorch/mnist",
        seed: int = 42,
    ) -> None:
        self.num_tasks = num_tasks
        self.seed = seed

        rng = np.random.RandomState(seed)
        self._permutations = [
            torch.from_numpy(rng.permutation(784)).long()
            for _ in range(num_tasks)
        ]

        self._train_data, self._train_targets = SplitMNIST._load_mnist(
            data_dir, train=True
        )
        self._test_data, self._test_targets = SplitMNIST._load_mnist(
            data_dir, train=False
        )

        logger.info(
            "PermutedMNIST initialised | %d tasks", num_tasks
        )

    def get_train_loader(
        self,
        task_id: int,
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> DataLoader:
        dataset = self._make_permuted_dataset(
            self._train_data, self._train_targets, task_id
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_test_loader(
        self,
        task_id: int,
        batch_size: int = 256,
    ) -> DataLoader:
        dataset = self._make_permuted_dataset(
            self._test_data, self._test_targets, task_id
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def get_all_test_loaders(self, batch_size: int = 256) -> List[DataLoader]:
        return [
            self.get_test_loader(task_id, batch_size)
            for task_id in range(self.num_tasks)
        ]

    def _make_permuted_dataset(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        task_id: int,
    ) -> TensorDataset:
        perm = self._permutations[task_id]
        flat = data.view(data.size(0), -1).float() / 255.0
        permuted = flat[:, perm]
        return TensorDataset(permuted, targets)
