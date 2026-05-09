"""
methods/experience_replay.py
-----------------------------
Experience Replay for continual learning.

Core idea
---------
After training on Task A, store a small random subset of Task A's training
data in a fixed-size "replay buffer". When training on Task B, interleave
old examples from the buffer with new Task B examples in each mini-batch.
This prevents the model from forgetting Task A by continuing to see it.

Design choices
--------------
1. Reservoir sampling (Vitter, 1985) maintains a uniform random sample of
   all data seen so far — no task label is required at sampling time.
   Each incoming example replaces a buffer slot with probability
   buffer_size / (n_seen_so_far + 1). This is equivalent to drawing a
   uniform sample from the stream without storing all of it.

2. The mix ratio (replay_ratio) controls the fraction of each mini-batch
   that comes from the replay buffer. A ratio of 0.5 means 50% old, 50%
   new. For Split-MNIST with 5 tasks, 0.3–0.5 works well.

3. The replay buffer stores task IDs alongside examples, so multi-head
   models can route old examples to the correct task head.

Trade-offs vs EWC
-----------------
- Simpler to implement and tune (one hyperparameter: buffer_size).
- Requires storing raw data — a privacy concern in some domains.
- More effective at preventing forgetting on diverse task sequences.
- Memory cost scales linearly with buffer_size.
- EWC stores parameter statistics (no raw data), but requires tuning λ.

References
----------
Robins (1995). Catastrophic forgetting, rehearsal and pseudorehearsal.
  Connection Science, 7(2), 123–146.

Vitter (1985). Random sampling with a reservoir.
  ACM Transactions on Mathematical Software, 11(1), 37–57.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Fixed-size experience replay buffer using reservoir sampling.

    Maintains a uniform random sample of all (x, y, task_id) tuples seen
    since initialisation, regardless of task boundaries.

    Parameters
    ----------
    capacity : int
        Maximum number of examples to store. 200–500 per task is typical.
    seed : int
        RNG seed for reproducible reservoir sampling.
    """

    def __init__(self, capacity: int = 1000, seed: int = 42) -> None:
        self.capacity = capacity
        self._rng = random.Random(seed)

        self._buffer_x: List[torch.Tensor] = []
        self._buffer_y: List[torch.Tensor] = []
        self._buffer_task_ids: List[int] = []
        self._n_seen: int = 0

    def add_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
    ) -> None:
        """
        Add a batch of (x, y) pairs from task_id to the buffer using
        reservoir sampling.

        x and y should be CPU tensors (detached from graph).
        """
        x = x.detach().cpu()
        y = y.detach().cpu()

        for i in range(x.size(0)):
            self._n_seen += 1
            if len(self._buffer_x) < self.capacity:
                # Buffer not full — always add
                self._buffer_x.append(x[i])
                self._buffer_y.append(y[i])
                self._buffer_task_ids.append(task_id)
            else:
                # Reservoir sampling: replace with probability capacity / n_seen
                j = self._rng.randint(0, self._n_seen - 1)
                if j < self.capacity:
                    self._buffer_x[j] = x[i]
                    self._buffer_y[j] = y[i]
                    self._buffer_task_ids[j] = task_id

    def sample(
        self,
        n: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, List[int]]]:
        """
        Sample n examples from the buffer uniformly at random.

        Returns
        -------
        (x, y, task_ids) or None if the buffer is empty.
        """
        if len(self._buffer_x) == 0:
            return None

        n = min(n, len(self._buffer_x))
        indices = self._rng.sample(range(len(self._buffer_x)), n)

        x = torch.stack([self._buffer_x[i] for i in indices])
        y = torch.stack([self._buffer_y[i] for i in indices])
        task_ids = [self._buffer_task_ids[i] for i in indices]

        return x, y, task_ids

    def __len__(self) -> int:
        return len(self._buffer_x)

    def task_distribution(self) -> Dict[int, int]:
        """Count how many examples from each task are in the buffer."""
        counts: Dict[int, int] = {}
        for tid in self._buffer_task_ids:
            counts[tid] = counts.get(tid, 0) + 1
        return counts


class ExperienceReplay:
    """
    Continual learning trainer using experience replay.

    Parameters
    ----------
    model : nn.Module
    device : torch.device
    buffer_size : int
        Total replay buffer capacity across all tasks.
        Rule of thumb: ~200 per task for Split-MNIST.
    replay_ratio : float
        Fraction of each mini-batch that comes from the replay buffer.
        0.0 = no replay (degenerates to baseline), 1.0 = pure replay.
        Typical: 0.3 – 0.5.
    lr : float
    momentum : float
    weight_decay : float
    multi_head : bool
        If True, calls model(x, task_id=t).
    seed : int
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        buffer_size: int = 1000,
        replay_ratio: float = 0.5,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        multi_head: bool = False,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.multi_head = multi_head

        self.buffer = ReplayBuffer(capacity=buffer_size, seed=seed)

    # ── Public API ─────────────────────────────────────────────────────────

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        epochs: int = 5,
    ) -> List[float]:
        """
        Train on one task, mixing in replay examples at each step.

        The buffer is populated with examples from the current task during
        training (so they become available for future tasks).

        Returns
        -------
        list of float
            Per-epoch loss.
        """
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
            running_loss = running_new_loss = running_replay_loss = 0.0
            n_batches = 0

            for x_new, y_new in train_loader:
                x_new = x_new.to(self.device)
                y_new = y_new.to(self.device)

                # Populate buffer with current task examples
                self.buffer.add_batch(x_new, y_new, task_id)

                optimizer.zero_grad()

                # Forward on new-task examples
                if self.multi_head:
                    logits_new = self.model(x_new, task_id=task_id)
                else:
                    logits_new = self.model(x_new)
                new_loss = criterion(logits_new, y_new)

                # Forward on replay examples
                replay_loss = self._compute_replay_loss(
                    task_id=task_id,
                    batch_size=x_new.size(0),
                    criterion=criterion,
                )

                loss = new_loss + self.replay_ratio * replay_loss
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_new_loss += new_loss.item()
                running_replay_loss += replay_loss.item()
                n_batches += 1

            nb = max(n_batches, 1)
            epoch_losses.append(running_loss / nb)

            logger.debug(
                "[Replay | Task %d | Epoch %d/%d] total=%.4f new=%.4f replay=%.4f buffer=%d",
                task_id, epoch, epochs,
                running_loss / nb,
                running_new_loss / nb,
                running_replay_loss / nb,
                len(self.buffer),
            )

        logger.info(
            "[Replay] Task %d trained | final_loss=%.4f | buffer_size=%d",
            task_id, epoch_losses[-1], len(self.buffer),
        )
        return epoch_losses

    @torch.no_grad()
    def evaluate(
        self,
        task_id: int,
        test_loader: DataLoader,
    ) -> float:
        """Top-1 accuracy on a task's test set."""
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

    def buffer_stats(self) -> Dict:
        """Return buffer occupancy and per-task distribution."""
        return {
            "size": len(self.buffer),
            "capacity": self.buffer_size,
            "task_distribution": self.buffer.task_distribution(),
        }

    # ── Internal helpers ───────────────────────────────────────────────────

    def _compute_replay_loss(
        self,
        task_id: int,
        batch_size: int,
        criterion: nn.Module,
    ) -> torch.Tensor:
        """
        Sample from the replay buffer and compute loss.

        If the buffer is empty (first task, no prior examples), returns zero.

        For multi-head models, we need to route each replayed example to
        the correct task head. This is handled by processing examples by
        task ID in groups.
        """
        n_replay = max(1, int(batch_size * self.replay_ratio))
        sample = self.buffer.sample(n_replay)

        if sample is None:
            return torch.tensor(0.0, device=self.device, requires_grad=False)

        x_replay, y_replay, replay_task_ids = sample
        x_replay = x_replay.to(self.device)
        y_replay = y_replay.to(self.device)

        if not self.multi_head:
            logits = self.model(x_replay)
            return criterion(logits, y_replay)

        # Multi-head: group by task and process each group separately
        total_loss = torch.tensor(0.0, device=self.device)
        unique_tids = set(replay_task_ids)

        for tid in unique_tids:
            # Only replay from prior tasks (not current task_id)
            # to avoid double-counting
            if tid == task_id:
                continue
            mask = torch.tensor(
                [i == tid for i in replay_task_ids], dtype=torch.bool
            )
            if mask.sum() == 0:
                continue
            logits = self.model(x_replay[mask], task_id=tid)
            total_loss = total_loss + criterion(logits, y_replay[mask])

        return total_loss
