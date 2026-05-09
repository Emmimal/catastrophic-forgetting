"""
methods/baseline.py
-------------------
Naive sequential fine-tuning — the "forgetful" baseline.

No regularisation. No replay. The model simply trains on each task in
sequence, overwriting previous weights completely. This is what
catastrophic forgetting looks like in practice.

We include this so every benchmark has a "forgetting floor" — the worst-
case performance that regularisation and replay methods are competing against.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class NaiveFineTuning:
    """
    Sequential fine-tuning with no forgetting prevention.

    Trains the model on each task until convergence, then moves to the
    next task without any mechanism to preserve prior knowledge.

    Parameters
    ----------
    model : nn.Module
        The neural network to train. Must accept (x, task_id) or just (x).
    device : torch.device
        CPU or CUDA.
    lr : float
        Learning rate for SGD with momentum.
    momentum : float
    weight_decay : float
    multi_head : bool
        If True, calls model(x, task_id=t) during training.
        If False, calls model(x) — single-head architecture.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        multi_head: bool = False,
    ) -> None:
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.multi_head = multi_head
        self._current_task = 0

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        epochs: int = 5,
    ) -> List[float]:
        """
        Train on one task for a fixed number of epochs.

        Returns
        -------
        list of float
            Per-epoch training loss.
        """
        self._current_task = task_id
        self.model.train()

        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        epoch_losses: List[float] = []

        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            n_batches = 0

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                if self.multi_head:
                    logits = self.model(x, task_id=task_id)
                else:
                    logits = self.model(x)

                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                n_batches += 1

            epoch_loss = running_loss / max(n_batches, 1)
            epoch_losses.append(epoch_loss)
            logger.debug(
                "[Baseline | Task %d | Epoch %d/%d] loss=%.4f",
                task_id, epoch, epochs, epoch_loss,
            )

        logger.info(
            "[Baseline] Task %d trained | final loss=%.4f",
            task_id, epoch_losses[-1],
        )
        return epoch_losses

    @torch.no_grad()
    def evaluate(
        self,
        task_id: int,
        test_loader: DataLoader,
    ) -> float:
        """
        Compute accuracy on a task's test set.

        Returns
        -------
        float
            Top-1 accuracy in [0, 1].
        """
        self.model.eval()
        correct = total = 0

        for x, y in test_loader:
            x, y = x.to(self.device), y.to(self.device)

            if self.multi_head:
                logits = self.model(x, task_id=task_id)
            else:
                logits = self.model(x)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        return correct / max(total, 1)
