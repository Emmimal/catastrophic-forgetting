"""
methods/packnet.py
------------------
PackNet — Mallya & Lazebnik, 2018.

Paper: "PackNet: Adding Multiple Tasks to a Single Network by Iterative
       Pruning"
       Proceedings of the IEEE/CVF CVPR Workshops, 2018.
       https://arxiv.org/abs/1711.05769

Core idea
---------
PackNet guarantees zero forgetting by partitioning the network's weights
into non-overlapping subsets — one per task. After training Task A:

  1. Prune the lowest-|weight| fraction of weights (free capacity for future).
  2. Freeze the remaining weights (assigned to Task A).
  3. Re-initialise pruned weights for Task B training.
  4. At inference, apply the task-specific binary mask.

Because Task A's weights are frozen during all future training, they cannot
change. Forgetting is structurally impossible rather than approximately
prevented.

Key properties
--------------
- Forgetting: Zero (by construction). No hyperparameter controls forgetting.
- New-task capacity: Limited by pruning_rate. If you prune 50% after Task A,
  Task B has 50% of the weights. Task C has 25%. etc.
- Memory footprint: Grows with n_tasks (one binary mask per task per layer).
- Inference: Task ID must be known — strictly task-incremental.

When PackNet breaks down
------------------------
- Class-incremental or domain-incremental scenarios (task ID unknown at test).
- More tasks than pruning_rate allows (network runs out of capacity).
- Very small networks with few parameters.

Pruning rate guidance
---------------------
If you plan for T tasks, each task needs at least 1/T of the network.
So pruning_rate should be no more than (T-1)/T per task.
For 5 tasks: prune at most 80% after each task (keeping ≥20%).
A conservative default is 0.5 (50% pruned per task) for up to 5 tasks.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class PackNet:
    """
    PackNet continual learning trainer.

    Parameters
    ----------
    model : nn.Module
        Multi-head MLP (MultiHeadMLP) or single-head MLP.
    device : torch.device
    pruning_rate : float
        Fraction of weights to prune (free for future tasks) after each
        task consolidation. Range: (0, 1). Default: 0.5.
    lr : float
    momentum : float
    weight_decay : float
    post_prune_retrain_epochs : int
        Number of epochs to fine-tune the surviving (non-pruned) weights
        after pruning, before freezing them. Helps recover accuracy lost
        to pruning. Default: 3.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        pruning_rate: float = 0.5,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        post_prune_retrain_epochs: int = 3,
    ) -> None:
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.pruning_rate = pruning_rate
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.post_prune_retrain_epochs = post_prune_retrain_epochs

        # Binary masks per task per parameter.
        # _task_masks[task_id][param_name] = BoolTensor (1 = owned by this task)
        self._task_masks: List[Dict[str, torch.Tensor]] = []

        # Accumulated frozen mask across all prior tasks.
        # frozen_mask[param_name] = BoolTensor (1 = frozen, must not change)
        self._frozen_mask: Dict[str, torch.Tensor] = {}

        # Free mask: weights available for the current task.
        # Initialised to all-True; narrows with each consolidation.
        self._free_mask: Dict[str, torch.Tensor] = {}
        self._initialise_free_mask()

        self._n_tasks = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        epochs: int = 5,
    ) -> List[float]:
        """
        Train using only the free (unassigned) weights.

        Frozen weights from prior tasks are not updated — their gradients
        are zeroed before each optimiser step.

        Returns
        -------
        list of float
            Per-epoch loss.
        """
        self.model.train()

        # Only train parameters that are in the free mask
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

                logits = self.model(x, task_id=task_id)
                loss = criterion(logits, y)
                loss.backward()

                # Zero out gradients for frozen weights before the step
                self._mask_frozen_gradients()

                optimizer.step()

                # Re-zero pruned weights (they might have picked up tiny
                # floating-point errors from the optimiser state)
                self._enforce_free_mask()

                running_loss += loss.item()
                n_batches += 1

            epoch_loss = running_loss / max(n_batches, 1)
            epoch_losses.append(epoch_loss)

            logger.debug(
                "[PackNet | Task %d | Epoch %d/%d] loss=%.4f | free_params=%d",
                task_id, epoch, epochs, epoch_loss, self._count_free_params(),
            )

        logger.info(
            "[PackNet] Task %d trained | loss=%.4f | free_params=%d",
            task_id, epoch_losses[-1], self._count_free_params(),
        )
        return epoch_losses

    def consolidate(
        self,
        task_id: int,
        train_loader: DataLoader,
    ) -> None:
        """
        Prune, freeze, and retrain after completing a task.

        Steps:
        1. Identify the lowest-|weight| fraction of FREE weights → prune them.
        2. Store the surviving free weights as task_id's mask.
        3. Extend the frozen mask to include task_id's weights.
        4. Re-initialise the newly pruned weights for the next task.
        5. Fine-tune only the non-pruned, non-frozen weights.

        Call this AFTER train_task(), BEFORE the next task.
        """
        logger.info(
            "[PackNet] Consolidating task %d | pruning_rate=%.2f",
            task_id, self.pruning_rate,
        )

        # Step 1–2: prune free weights and get this task's mask
        task_mask = self._prune_and_get_task_mask()
        self._task_masks.append(task_mask)

        # Step 3: extend frozen mask
        for name in self._frozen_mask:
            self._frozen_mask[name] = (
                self._frozen_mask[name] | task_mask[name]
            )

        # Step 4: update free mask (subtract frozen from all-ones)
        for name, frozen in self._frozen_mask.items():
            self._free_mask[name] = ~frozen

        # Re-initialise freed weights for next task
        self._reinitialise_free_weights()

        # Step 5: retrain with frozen prior-task weights
        self._post_prune_retrain(task_id, train_loader)

        self._n_tasks += 1

        free_pct = self._count_free_params() / self._total_params() * 100
        logger.info(
            "[PackNet] Consolidated task %d | free_capacity=%.1f%%",
            task_id, free_pct,
        )

    @torch.no_grad()
    def evaluate(
        self,
        task_id: int,
        test_loader: DataLoader,
    ) -> float:
        """
        Evaluate accuracy for a specific task.

        At inference, applies the task-specific binary mask so only the
        weights assigned to this task are active. Prior-task weights in
        non-relevant positions are masked to zero.

        This is the strict PackNet evaluation: task ID is required.
        """
        self.model.eval()
        correct = total = 0

        for x, y in test_loader:
            x, y = x.to(self.device), y.to(self.device)

            # Apply task mask for inference
            with self._apply_task_mask_context(task_id):
                logits = self.model(x, task_id=task_id)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        return correct / max(total, 1)

    def capacity_report(self) -> Dict:
        """
        Report on network capacity usage.

        Returns
        -------
        dict with:
            total_params, frozen_params, free_params, free_pct,
            per_task_params
        """
        total = self._total_params()
        frozen = self._count_frozen_params()
        free = self._count_free_params()
        per_task = {
            f"task_{i}": sum(m.sum().item() for m in mask.values())
            for i, mask in enumerate(self._task_masks)
        }
        return {
            "total_params": total,
            "frozen_params": frozen,
            "free_params": free,
            "free_pct": free / max(total, 1) * 100,
            "per_task_params": per_task,
        }

    # ── Internal helpers ───────────────────────────────────────────────────

    def _initialise_free_mask(self) -> None:
        """
        Initialise masks to all-free and all-unfrozen.
        Called once at construction time.
        """
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            self._free_mask[name] = torch.ones_like(
                param, dtype=torch.bool, device=self.device
            )
            self._frozen_mask[name] = torch.zeros_like(
                param, dtype=torch.bool, device=self.device
            )

    def _prune_and_get_task_mask(self) -> Dict[str, torch.Tensor]:
        """
        Prune the lowest-|weight| fraction of free weights.

        Among the currently free weights, find the pruning_rate fraction
        with the smallest absolute values and zero them out.

        Returns a binary mask indicating which weights belong to this task
        (i.e., the surviving non-pruned free weights).
        """
        # Collect absolute values of all free weights
        all_free_values: List[float] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad or name not in self._free_mask:
                continue
            free_vals = param.data[self._free_mask[name]].abs()
            all_free_values.extend(free_vals.tolist())

        if not all_free_values:
            raise RuntimeError(
                "No free weights remaining. Network has run out of capacity. "
                "Reduce pruning_rate or use fewer tasks."
            )

        # Compute pruning threshold
        all_free_values_sorted = sorted(all_free_values)
        n_prune = int(len(all_free_values_sorted) * self.pruning_rate)
        n_prune = min(n_prune, len(all_free_values_sorted) - 1)
        threshold = all_free_values_sorted[n_prune]

        # Build task mask: free weights above threshold are kept for this task
        task_mask: Dict[str, torch.Tensor] = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad or name not in self._free_mask:
                continue

            free = self._free_mask[name]
            # A weight is in this task's mask if:
            #   - It was free (not assigned to a prior task)
            #   - Its absolute value is above the pruning threshold (survived)
            survived = (param.data.abs() >= threshold) & free
            task_mask[name] = survived

            # Zero out pruned (below threshold) free weights
            pruned_mask = (param.data.abs() < threshold) & free
            param.data[pruned_mask] = 0.0

        return task_mask

    def _reinitialise_free_weights(self) -> None:
        """
        Re-initialise free (unpruned, unassigned) weights with small random
        values so they are ready for the next task.

        Uses Kaiming uniform for 2-D weight matrices and uniform(-0.1, 0.1)
        for 1-D bias vectors (kaiming_uniform_ requires at least 2 dims).
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad or name not in self._free_mask:
                    continue
                free = self._free_mask[name]
                if free.sum() == 0:
                    continue
                new_vals = torch.empty_like(param)
                if param.dim() >= 2:
                    nn.init.kaiming_uniform_(new_vals, nonlinearity="relu")
                else:
                    # Bias vectors: small uniform initialisation
                    nn.init.uniform_(new_vals, -0.1, 0.1)
                param.data[free] = new_vals[free]

    def _post_prune_retrain(
        self,
        task_id: int,
        train_loader: DataLoader,
    ) -> None:
        """
        Fine-tune after pruning to recover accuracy lost due to weight removal.
        Only the current task's surviving weights are trained (frozen weights
        from prior tasks are not touched).
        """
        if self.post_prune_retrain_epochs == 0:
            return

        self.model.train()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr * 0.1,  # Lower LR for fine-tuning
            momentum=self.momentum,
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, self.post_prune_retrain_epochs + 1):
            running_loss = 0.0
            n_batches = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x, task_id=task_id)
                loss = criterion(logits, y)
                loss.backward()
                self._mask_frozen_gradients()
                optimizer.step()
                running_loss += loss.item()
                n_batches += 1

            logger.debug(
                "[PackNet] Post-prune retrain epoch %d/%d | loss=%.4f",
                epoch, self.post_prune_retrain_epochs,
                running_loss / max(n_batches, 1),
            )

    def _mask_frozen_gradients(self) -> None:
        """
        Zero out gradients for all frozen weights before the optimiser step.
        This prevents the optimiser from modifying weights owned by prior tasks.
        """
        for name, param in self.model.named_parameters():
            if param.grad is None or name not in self._frozen_mask:
                continue
            param.grad.data[self._frozen_mask[name]] = 0.0

    def _enforce_free_mask(self) -> None:
        """
        After an optimiser step, ensure pruned positions remain exactly zero.
        Floating-point arithmetic in the optimiser can introduce small non-zero
        values in positions that should be zero.
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name not in self._frozen_mask:
                    continue
                # Zero out positions that are neither frozen nor free
                # (i.e., permanently pruned)
                both_false = ~(self._frozen_mask[name] | self._free_mask[name])
                param.data[both_false] = 0.0

    def _apply_task_mask_context(self, task_id: int):
        """
        Context manager that applies the task-specific mask during inference
        and restores original weights afterward.

        For tasks beyond the first, weights not belonging to this task are
        temporarily zeroed to prevent cross-task interference.
        """
        return _TaskMaskContext(self.model, self._task_masks, task_id)

    def _count_free_params(self) -> int:
        return sum(m.sum().item() for m in self._free_mask.values())

    def _count_frozen_params(self) -> int:
        return sum(m.sum().item() for m in self._frozen_mask.values())

    def _total_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class _TaskMaskContext:
    """
    Context manager for masked inference.

    Saves the original weight values, zeroes all non-task weights, runs
    inference, then restores the originals.
    """

    def __init__(
        self,
        model: nn.Module,
        task_masks: List[Dict[str, torch.Tensor]],
        task_id: int,
    ) -> None:
        self.model = model
        self.task_masks = task_masks
        self.task_id = task_id
        self._saved: Dict[str, torch.Tensor] = {}

    def __enter__(self):
        if self.task_id >= len(self.task_masks):
            return self  # Current (not-yet-consolidated) task — no mask needed

        task_mask = self.task_masks[self.task_id]
        for name, param in self.model.named_parameters():
            if name not in task_mask:
                continue
            self._saved[name] = param.data.clone()
            # Zero weights NOT belonging to this task
            param.data[~task_mask[name]] = 0.0
        return self

    def __exit__(self, *args):
        for name, saved in self._saved.items():
            for n, param in self.model.named_parameters():
                if n == name:
                    param.data.copy_(saved)
                    break
