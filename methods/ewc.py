"""
methods/ewc.py
--------------
Elastic Weight Consolidation (EWC) — Kirkpatrick et al., 2017.

Paper: "Overcoming catastrophic forgetting in neural networks"
       Proceedings of the National Academy of Sciences, 114(13), 3521–3526.
       https://doi.org/10.1073/pnas.1611835114

Core idea
---------
After training on Task A, compute the Fisher Information Matrix (FIM) diagonal
for each weight parameter. The Fisher diagonal approximates how important each
weight is for Task A. When training on Task B, add a regularisation penalty
that resists large changes to high-Fisher weights.

The penalty term is:

    L_ewc = (λ / 2) * Σ_i F_i * (θ_i - θ*_i)²

where:
    θ*_i   = weight value after training Task A (the "anchor")
    F_i    = Fisher Information diagonal for weight i (the "importance")
    λ      = regularisation strength (lambda_ewc)

Tuning lambda_ewc
-----------------
    - Too high: the model resists learning Task B (underfits new task)
    - Too low:  the model overwrites Task A knowledge (catastrophic forgetting)
    - Right:    backward transfer is close to zero while forward transfer is
                close to the single-task baseline

Tune on a held-out validation set that contains examples from BOTH the old
task and the new task. Typical range: 0.1 – 10.0. Start at 0.4.

Online EWC
----------
When more than one prior task exists, we accumulate penalties rather than
storing separate FIM + anchor sets per task. This is the "online EWC" variant
(Schwarz et al., 2018) — it is computationally cheaper and performs similarly
to the multi-task version for moderate numbers of tasks.
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


class EWC:
    """
    Elastic Weight Consolidation trainer.

    Parameters
    ----------
    model : nn.Module
    device : torch.device
    lambda_ewc : float
        Regularisation strength. Controls the forgetting/plasticity trade-off.
        Typical range: 0.1 – 10.0. Default: 0.4.
    lr : float
        SGD learning rate.
    momentum : float
    weight_decay : float
    n_fisher_samples : int
        Number of training samples used to estimate the Fisher diagonal.
        More samples = more accurate estimate but slower. 200–1000 is usually
        sufficient.
    multi_head : bool
        If True, calls model(x, task_id=t).
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        lambda_ewc: float = 0.4,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        n_fisher_samples: int = 200,
        multi_head: bool = False,
    ) -> None:
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.lambda_ewc = lambda_ewc
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.n_fisher_samples = n_fisher_samples
        self.multi_head = multi_head

        # Accumulated Fisher diagonal (sum over all prior tasks)
        # Keys: parameter names; values: Tensors on self.device
        self._fisher_diag: Dict[str, torch.Tensor] = {}

        # Anchor parameters (θ*) — snapshot after the most recent task
        self._anchor_params: Dict[str, torch.Tensor] = {}

        # Number of tasks completed (used for online EWC accumulation)
        self._n_tasks_completed = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        epochs: int = 5,
    ) -> List[float]:
        """
        Train on one task with EWC regularisation applied for all prior tasks.

        Returns
        -------
        list of float
            Per-epoch total loss (task loss + EWC penalty).
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
            running_loss = running_task_loss = running_ewc_loss = 0.0
            n_batches = 0

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                if self.multi_head:
                    logits = self.model(x, task_id=task_id)
                else:
                    logits = self.model(x)

                task_loss = criterion(logits, y)
                ewc_penalty = self._compute_ewc_penalty()
                loss = task_loss + ewc_penalty

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_task_loss += task_loss.item()
                running_ewc_loss += ewc_penalty.item()
                n_batches += 1

            nb = max(n_batches, 1)
            epoch_losses.append(running_loss / nb)

            logger.debug(
                "[EWC | Task %d | Epoch %d/%d] total=%.4f task=%.4f ewc=%.4f",
                task_id, epoch, epochs,
                running_loss / nb,
                running_task_loss / nb,
                running_ewc_loss / nb,
            )

        logger.info(
            "[EWC] Task %d trained | final_loss=%.4f | lambda=%.3f",
            task_id, epoch_losses[-1], self.lambda_ewc,
        )
        return epoch_losses

    def consolidate(
        self,
        task_id: int,
        train_loader: DataLoader,
    ) -> None:
        """
        Compute and accumulate Fisher diagonal + snapshot anchor parameters.

        Call this AFTER training on a task, BEFORE training on the next task.

        The Fisher diagonal is estimated using the squared gradient of the
        log-likelihood with respect to each parameter, averaged over
        n_fisher_samples training samples.

        Parameters
        ----------
        task_id : int
            The task just trained (used for multi-head forward pass).
        train_loader : DataLoader
            Training data for the completed task (used for Fisher estimation).
        """
        logger.info("[EWC] Computing Fisher diagonal for task %d...", task_id)

        new_fisher = self._estimate_fisher(task_id, train_loader)

        if self._n_tasks_completed == 0:
            # First task: initialise accumulators
            self._fisher_diag = {
                name: fisher.clone()
                for name, fisher in new_fisher.items()
            }
        else:
            # Online EWC: accumulate (sum) Fisher diagonals
            for name, fisher in new_fisher.items():
                self._fisher_diag[name] = self._fisher_diag[name] + fisher

        # Snapshot current weights as anchor
        self._anchor_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }

        self._n_tasks_completed += 1
        logger.info(
            "[EWC] Consolidated task %d | total_tasks=%d",
            task_id, self._n_tasks_completed,
        )

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

    # ── Internal helpers ───────────────────────────────────────────────────

    def _compute_ewc_penalty(self) -> torch.Tensor:
        """
        Compute the EWC regularisation term:

            (λ / 2) * Σ_i F_i * (θ_i - θ*_i)²

        Returns zero if no tasks have been consolidated yet (first task).
        """
        if not self._fisher_diag:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)

        for name, param in self.model.named_parameters():
            if name not in self._fisher_diag:
                continue
            fisher = self._fisher_diag[name]
            anchor = self._anchor_params[name]
            penalty = penalty + (fisher * (param - anchor) ** 2).sum()

        return (self.lambda_ewc / 2.0) * penalty

    def _estimate_fisher(
        self,
        task_id: int,
        train_loader: DataLoader,
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate the diagonal of the Fisher Information Matrix.

        For each training sample, we compute the gradient of the log-
        probability of the model's own prediction (the "online" estimator).
        The Fisher diagonal is the average of the squared gradients.

        This is the empirical Fisher approximation — it is not identical to
        the true Fisher (which uses the ground-truth label distribution) but
        is standard in the EWC literature and scales to large networks.

        Returns
        -------
        dict
            Parameter name → Fisher diagonal tensor (same shape as the param).
        """
        self.model.train()

        fisher: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        n_samples = 0
        for x, y in train_loader:
            if n_samples >= self.n_fisher_samples:
                break

            x, y = x.to(self.device), y.to(self.device)
            batch_size = x.size(0)

            # Per-sample gradient accumulation
            for i in range(batch_size):
                if n_samples >= self.n_fisher_samples:
                    break

                xi = x[i : i + 1]
                self.model.zero_grad()

                if self.multi_head:
                    logits = self.model(xi, task_id=task_id)
                else:
                    logits = self.model(xi)

                # Use model's own predicted label (empirical Fisher)
                log_probs = torch.log_softmax(logits, dim=1)
                predicted_label = logits.argmax(dim=1)
                loss = -log_probs[0, predicted_label[0]]
                loss.backward()

                for name, param in self.model.named_parameters():
                    if param.grad is not None and name in fisher:
                        fisher[name] += param.grad.data ** 2

                n_samples += 1

        # Normalise by number of samples
        for name in fisher:
            fisher[name] /= max(n_samples, 1)

        logger.debug(
            "[EWC] Fisher estimated over %d samples", n_samples
        )
        return fisher

    # ── Diagnostics ────────────────────────────────────────────────────────

    def fisher_summary(self) -> Dict[str, float]:
        """
        Returns the mean Fisher value per layer — useful for debugging.
        High values indicate weight importance for prior tasks.
        """
        return {
            name: fisher.mean().item()
            for name, fisher in self._fisher_diag.items()
        }

    def ewc_penalty_value(self) -> float:
        """Return the current EWC penalty as a Python float (for logging)."""
        with torch.no_grad():
            return self._compute_ewc_penalty().item()
