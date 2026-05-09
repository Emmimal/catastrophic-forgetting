"""
benchmarks/metrics.py
---------------------
Continual learning evaluation metrics.

Standard metrics used in the continual learning literature
(Lopez-Paz & Ranzato, 2017; Chaudhry et al., 2018).

Metrics
-------
Average Accuracy (ACC)
    Mean test accuracy across all tasks after training on the final task.
    Higher is better. Primary headline metric.

Backward Transfer (BWT)
    How much learning Task N changed performance on previously learned tasks.
    BWT = (1/T-1) Σ_{i<T} [acc(task_i, after_task_T) - acc(task_i, after_task_i)]
    - Negative BWT = catastrophic forgetting (performance dropped).
    - Zero BWT = no forgetting.
    - Positive BWT = backward transfer (new tasks helped old ones — rare).

Forward Transfer (FWT)
    How much learning prior tasks helped on future tasks at their very first
    exposure (before training on them at all).
    FWT = (1/T-1) Σ_{i>0} [acc(task_i, before_training) - acc(task_i, random_init)]
    Positive FWT = transfer. Negative = interference.

Forgetting (F)
    Maximum accuracy drop across all tasks.
    F = (1/T-1) Σ_{i<T} max_{j≤T-1} [acc(task_i, after_task_j) - acc(task_i, after_task_T)]
    Always ≥ 0. Difference from BWT: uses max rather than final snapshot.

References
----------
Lopez-Paz & Ranzato (2017). Gradient episodic memory for continual learning.
  NeurIPS, 30.

Chaudhry et al. (2018). Efficient lifelong learning with A-GEM.
  ICLR 2019. https://arxiv.org/abs/1812.00420
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np


class CLMetrics:
    """
    Tracks and computes continual learning metrics from an accuracy matrix.

    The accuracy matrix R has shape (n_tasks, n_tasks) where:
        R[i][j] = accuracy on task j after training on tasks 0..i

    Parameters
    ----------
    n_tasks : int
    random_init_accs : list of float, optional
        Accuracy of a randomly initialised model on each task.
        Used for Forward Transfer computation. If None, FWT is not computed.
    """

    def __init__(
        self,
        n_tasks: int,
        random_init_accs: Optional[List[float]] = None,
    ) -> None:
        self.n_tasks = n_tasks
        self.random_init_accs = random_init_accs

        # R[i][j] = accuracy on task j after seeing tasks 0..i
        self._R: List[List[Optional[float]]] = [
            [None] * n_tasks for _ in range(n_tasks)
        ]

    def record(self, after_task: int, task_id: int, accuracy: float) -> None:
        """
        Record accuracy on task_id after training on after_task.

        Parameters
        ----------
        after_task : int
            The last task trained on (row index).
        task_id : int
            The task being evaluated (column index).
        accuracy : float
            Top-1 accuracy in [0, 1].
        """
        self._R[after_task][task_id] = accuracy

    def average_accuracy(self) -> float:
        """
        ACC: Mean accuracy across all tasks after training on the last task.
        Uses row n_tasks-1 of the accuracy matrix.
        """
        final_row = [
            self._R[self.n_tasks - 1][j]
            for j in range(self.n_tasks)
            if self._R[self.n_tasks - 1][j] is not None
        ]
        return float(np.mean(final_row)) if final_row else 0.0

    def backward_transfer(self) -> float:
        """
        BWT: Mean change in accuracy on all prior tasks after full training.

        Negative → catastrophic forgetting.
        Zero → perfect retention.
        Positive → backward transfer (learning new tasks helped old ones).
        """
        deltas = []
        for i in range(self.n_tasks - 1):
            after_final = self._R[self.n_tasks - 1][i]
            after_own   = self._R[i][i]
            if after_final is not None and after_own is not None:
                deltas.append(after_final - after_own)

        return float(np.mean(deltas)) if deltas else 0.0

    def forgetting(self) -> float:
        """
        F: Average maximum performance drop across tasks.
        Always ≥ 0. Larger = more forgetting.
        """
        task_forgetting = []
        for i in range(self.n_tasks - 1):
            peak = max(
                self._R[j][i]
                for j in range(i, self.n_tasks)
                if self._R[j][i] is not None
            )
            final = self._R[self.n_tasks - 1][i]
            if final is not None:
                task_forgetting.append(peak - final)

        return float(np.mean(task_forgetting)) if task_forgetting else 0.0

    def forward_transfer(self) -> Optional[float]:
        """
        FWT: How much prior learning helped on future tasks.
        Requires random_init_accs to be provided.
        Returns None if not available.
        """
        if self.random_init_accs is None:
            return None

        fwts = []
        for i in range(1, self.n_tasks):
            # Accuracy on task i before any training on it (after tasks 0..i-1)
            before = self._R[i - 1][i]
            rand   = self.random_init_accs[i]
            if before is not None:
                fwts.append(before - rand)

        return float(np.mean(fwts)) if fwts else 0.0

    def task_accuracy_at_final(self, task_id: int) -> Optional[float]:
        """Accuracy on a specific task after training on all tasks."""
        return self._R[self.n_tasks - 1][task_id]

    def task_accuracy_after_training(self, task_id: int) -> Optional[float]:
        """Accuracy on a specific task immediately after its own training."""
        return self._R[task_id][task_id]

    def accuracy_matrix(self) -> np.ndarray:
        """Return the full accuracy matrix as a numpy array."""
        matrix = np.full((self.n_tasks, self.n_tasks), np.nan)
        for i in range(self.n_tasks):
            for j in range(self.n_tasks):
                val = self._R[i][j]
                if val is not None:
                    matrix[i][j] = val
        return matrix

    def summary(self) -> dict:
        """Return all metrics as a dictionary."""
        fwt = self.forward_transfer()
        result = {
            "ACC": round(self.average_accuracy(), 4),
            "BWT": round(self.backward_transfer(), 4),
            "Forgetting": round(self.forgetting(), 4),
        }
        if fwt is not None:
            result["FWT"] = round(fwt, 4)
        return result

    def print_matrix(self) -> None:
        """Pretty-print the accuracy matrix to stdout."""
        matrix = self.accuracy_matrix()
        header = "After \\ Task  " + "  ".join(f"T{j}" for j in range(self.n_tasks))
        print(header)
        print("-" * len(header))
        for i in range(self.n_tasks):
            row_vals = []
            for j in range(self.n_tasks):
                v = matrix[i, j]
                if np.isnan(v):
                    row_vals.append("  ---")
                else:
                    row_vals.append(f"{v:.3f}")
            print(f"After Task {i}:   " + "  ".join(row_vals))
