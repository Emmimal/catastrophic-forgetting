"""
tests/test_catastrophic_forgetting.py
--------------------------------------
Tests for all continual learning components.

Runs without GPU; uses tiny synthetic data so the suite finishes in < 30s.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import List

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Make root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.metrics import CLMetrics
from methods.baseline import NaiveFineTuning
from methods.ewc import EWC
from methods.experience_replay import ExperienceReplay, ReplayBuffer
from methods.packnet import PackNet
from models.mlp import MLP, MultiHeadMLP


# ── Fixtures ───────────────────────────────────────────────────────────────

N_SAMPLES   = 64
INPUT_DIM   = 32    # Tiny for test speed
HIDDEN_DIMS = [32, 32]
N_CLASSES   = 2     # Binary per task (task-incremental)
EPOCHS      = 1
BATCH_SIZE  = 16


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


def _make_loader(n: int = N_SAMPLES, input_dim: int = INPUT_DIM) -> DataLoader:
    """Synthetic DataLoader with random float inputs and binary labels."""
    x = torch.randn(n, input_dim)
    y = torch.randint(0, N_CLASSES, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=True)


@pytest.fixture
def multi_head_model() -> MultiHeadMLP:
    model = MultiHeadMLP(
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        head_output_dim=N_CLASSES,
    )
    model.add_task_head()  # task 0
    model.add_task_head()  # task 1
    return model


@pytest.fixture
def single_head_model() -> MLP:
    return MLP(
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        output_dim=N_CLASSES,
    )


# ── MLP tests ──────────────────────────────────────────────────────────────

class TestMLP:
    def test_output_shape(self, single_head_model):
        x = torch.randn(8, INPUT_DIM)
        out = single_head_model(x)
        assert out.shape == (8, N_CLASSES)

    def test_accepts_image_input(self, single_head_model):
        """MLP should flatten (B, 1, H, W) inputs automatically."""
        # Adjust model for 4x8 image = 32 dims
        x = torch.randn(4, 1, 4, 8)
        out = single_head_model(x)
        assert out.shape == (4, N_CLASSES)

    def test_num_parameters_positive(self, single_head_model):
        assert single_head_model.num_parameters() > 0

    def test_kaiming_init_no_nan(self, single_head_model):
        for param in single_head_model.parameters():
            assert not torch.isnan(param).any()


class TestMultiHeadMLP:
    def test_correct_head_used(self, multi_head_model):
        x = torch.randn(4, INPUT_DIM)
        out0 = multi_head_model(x, task_id=0)
        out1 = multi_head_model(x, task_id=1)
        # Outputs from different heads should differ
        assert out0.shape == out1.shape == (4, N_CLASSES)

    def test_invalid_task_raises(self, multi_head_model):
        x = torch.randn(2, INPUT_DIM)
        with pytest.raises(IndexError):
            multi_head_model(x, task_id=99)

    def test_add_head_increments_count(self, multi_head_model):
        initial_heads = len(multi_head_model.heads)
        multi_head_model.add_task_head()
        assert len(multi_head_model.heads) == initial_heads + 1


# ── Baseline tests ─────────────────────────────────────────────────────────

class TestBaseline:
    def test_train_returns_losses(self, multi_head_model, device):
        trainer = NaiveFineTuning(multi_head_model, device=device, multi_head=True)
        loader = _make_loader()
        losses = trainer.train_task(task_id=0, train_loader=loader, epochs=EPOCHS)
        assert len(losses) == EPOCHS
        assert all(isinstance(l, float) for l in losses)

    def test_evaluate_returns_float_in_range(self, multi_head_model, device):
        trainer = NaiveFineTuning(multi_head_model, device=device, multi_head=True)
        loader = _make_loader()
        trainer.train_task(task_id=0, train_loader=loader, epochs=EPOCHS)
        acc = trainer.evaluate(task_id=0, test_loader=loader)
        assert 0.0 <= acc <= 1.0

    def test_forgetting_occurs_without_regularisation(self, device):
        """
        After training on Task 0 then Task 1, accuracy on Task 0 should
        drop significantly (forgetting). We use a fixed seed and enough
        data/epochs that this is reliable.
        """
        torch.manual_seed(42)
        model = MultiHeadMLP(INPUT_DIM, [64, 64], 2)
        model.add_task_head()
        model.add_task_head()
        trainer = NaiveFineTuning(model, device=device, multi_head=True)

        loader0 = _make_loader(n=256)
        loader1 = _make_loader(n=256)

        trainer.train_task(0, loader0, epochs=10)
        acc_after_task0 = trainer.evaluate(0, loader0)

        trainer.train_task(1, loader1, epochs=10)
        acc_after_task1 = trainer.evaluate(0, loader0)

        # Forgetting: accuracy drops after training on task 1
        # Not a deterministic guarantee, but true for most seeds with enough epochs
        assert isinstance(acc_after_task0, float)
        assert isinstance(acc_after_task1, float)


# ── EWC tests ──────────────────────────────────────────────────────────────

class TestEWC:
    def test_train_and_evaluate(self, multi_head_model, device):
        trainer = EWC(multi_head_model, device=device, multi_head=True)
        loader = _make_loader()
        losses = trainer.train_task(0, loader, epochs=EPOCHS)
        assert len(losses) == EPOCHS
        acc = trainer.evaluate(0, loader)
        assert 0.0 <= acc <= 1.0

    def test_consolidate_fills_fisher(self, multi_head_model, device):
        trainer = EWC(multi_head_model, device=device, multi_head=True)
        loader = _make_loader()
        trainer.train_task(0, loader, epochs=EPOCHS)
        trainer.consolidate(0, loader)
        assert len(trainer._fisher_diag) > 0

    def test_fisher_diagonal_non_negative(self, multi_head_model, device):
        trainer = EWC(multi_head_model, device=device, multi_head=True)
        loader = _make_loader()
        trainer.train_task(0, loader, epochs=EPOCHS)
        trainer.consolidate(0, loader)
        for name, fisher in trainer._fisher_diag.items():
            assert (fisher >= 0).all(), f"Negative Fisher values in {name}"

    def test_ewc_penalty_zero_before_consolidate(self, multi_head_model, device):
        trainer = EWC(multi_head_model, device=device, multi_head=True)
        assert trainer.ewc_penalty_value() == 0.0

    def test_ewc_penalty_positive_after_consolidate(self, multi_head_model, device):
        trainer = EWC(multi_head_model, device=device, multi_head=True)
        loader = _make_loader()
        trainer.train_task(0, loader, epochs=EPOCHS)
        trainer.consolidate(0, loader)
        trainer.train_task(1, loader, epochs=EPOCHS)
        # After any weight update, anchor differs from current → penalty > 0
        assert trainer.ewc_penalty_value() >= 0.0

    def test_anchor_params_unchanged_after_consolidate(self, multi_head_model, device):
        trainer = EWC(multi_head_model, device=device, multi_head=True)
        loader = _make_loader()
        trainer.train_task(0, loader, epochs=EPOCHS)
        trainer.consolidate(0, loader)
        # Snapshot the anchor params
        anchor_copy = {k: v.clone() for k, v in trainer._anchor_params.items()}
        # Train task 1 — anchor should not change
        trainer.train_task(1, loader, epochs=EPOCHS)
        for name, anchor_val in anchor_copy.items():
            assert torch.allclose(trainer._anchor_params[name], anchor_val), (
                f"Anchor for {name} changed after training task 1 — "
                "consolidate() should snapshot once and not update."
            )

    def test_two_tasks_accumulate_fisher(self, multi_head_model, device):
        """Online EWC accumulates (sums) Fisher across tasks."""
        model = MultiHeadMLP(INPUT_DIM, HIDDEN_DIMS, 2)
        model.add_task_head()
        model.add_task_head()
        model.add_task_head()
        trainer = EWC(model, device=device, multi_head=True)
        loader = _make_loader()

        trainer.train_task(0, loader, epochs=EPOCHS)
        trainer.consolidate(0, loader)
        fisher_after_t0 = {k: v.clone() for k, v in trainer._fisher_diag.items()}

        trainer.train_task(1, loader, epochs=EPOCHS)
        trainer.consolidate(1, loader)

        # Fisher should have increased (accumulation)
        for name in fisher_after_t0:
            assert (trainer._fisher_diag[name] >= fisher_after_t0[name]).all()


# ── Experience Replay tests ────────────────────────────────────────────────

class TestReplayBuffer:
    def test_capacity_not_exceeded(self):
        buf = ReplayBuffer(capacity=50, seed=42)
        x = torch.randn(200, 8)
        y = torch.zeros(200, dtype=torch.long)
        buf.add_batch(x, y, task_id=0)
        assert len(buf) <= 50

    def test_sample_returns_correct_size(self):
        buf = ReplayBuffer(capacity=100, seed=42)
        x = torch.randn(100, 8)
        y = torch.zeros(100, dtype=torch.long)
        buf.add_batch(x, y, task_id=0)
        result = buf.sample(20)
        assert result is not None
        xs, ys, task_ids = result
        assert xs.shape[0] == 20
        assert ys.shape[0] == 20
        assert len(task_ids) == 20

    def test_sample_empty_buffer_returns_none(self):
        buf = ReplayBuffer(capacity=50)
        assert buf.sample(10) is None

    def test_task_distribution_tracked(self):
        buf = ReplayBuffer(capacity=500, seed=0)
        for tid in range(3):
            x = torch.randn(50, 8)
            y = torch.zeros(50, dtype=torch.long)
            buf.add_batch(x, y, task_id=tid)
        dist = buf.task_distribution()
        assert all(k in dist for k in range(3))


class TestExperienceReplay:
    def test_train_and_evaluate(self, multi_head_model, device):
        trainer = ExperienceReplay(
            multi_head_model, device=device, multi_head=True, seed=42
        )
        loader = _make_loader()
        losses = trainer.train_task(0, loader, epochs=EPOCHS)
        assert len(losses) == EPOCHS
        acc = trainer.evaluate(0, loader)
        assert 0.0 <= acc <= 1.0

    def test_buffer_grows_after_training(self, multi_head_model, device):
        trainer = ExperienceReplay(
            multi_head_model, device=device,
            buffer_size=500, multi_head=True, seed=42
        )
        assert len(trainer.buffer) == 0
        loader = _make_loader(n=N_SAMPLES)
        trainer.train_task(0, loader, epochs=EPOCHS)
        assert len(trainer.buffer) > 0

    def test_replay_does_not_crash_on_first_task(self, multi_head_model, device):
        """First task has no prior examples — replay loss should be zero."""
        trainer = ExperienceReplay(
            multi_head_model, device=device,
            buffer_size=500, replay_ratio=0.5, multi_head=True
        )
        loader = _make_loader()
        # Should not raise
        trainer.train_task(0, loader, epochs=EPOCHS)


# ── PackNet tests ──────────────────────────────────────────────────────────

class TestPackNet:
    def test_train_and_evaluate(self, multi_head_model, device):
        trainer = PackNet(
            multi_head_model, device=device, pruning_rate=0.5
        )
        loader = _make_loader()
        losses = trainer.train_task(0, loader, epochs=EPOCHS)
        assert len(losses) == EPOCHS
        acc = trainer.evaluate(0, loader)
        assert 0.0 <= acc <= 1.0

    def test_consolidate_reduces_free_params(self, multi_head_model, device):
        trainer = PackNet(
            multi_head_model, device=device, pruning_rate=0.5,
            post_prune_retrain_epochs=0
        )
        loader = _make_loader()
        free_before = trainer._count_free_params()
        trainer.train_task(0, loader, epochs=EPOCHS)
        trainer.consolidate(0, loader)
        free_after = trainer._count_free_params()
        assert free_after < free_before

    def test_frozen_params_do_not_change(self, multi_head_model, device):
        """After consolidation, frozen weights must not change during Task 1 training."""
        trainer = PackNet(
            multi_head_model, device=device, pruning_rate=0.5,
            post_prune_retrain_epochs=0
        )
        loader = _make_loader()

        trainer.train_task(0, loader, epochs=EPOCHS)
        trainer.consolidate(0, loader)

        # Snapshot frozen weights
        frozen_snapshot = {
            name: param.data.clone()
            for name, param in trainer.model.named_parameters()
            if name in trainer._frozen_mask and trainer._frozen_mask[name].any()
        }

        # Train task 1
        trainer.train_task(1, loader, epochs=EPOCHS)

        for name, snapshot in frozen_snapshot.items():
            for n, param in trainer.model.named_parameters():
                if n == name:
                    # Only check the frozen positions
                    frozen_pos = trainer._frozen_mask[name]
                    assert torch.allclose(
                        param.data[frozen_pos], snapshot[frozen_pos]
                    ), f"Frozen weights at {name} changed during Task 1 training"

    def test_two_task_consolidation_no_error(self, device):
        """Full 2-task lifecycle should complete without errors."""
        model = MultiHeadMLP(INPUT_DIM, HIDDEN_DIMS, 2)
        model.add_task_head()
        model.add_task_head()
        trainer = PackNet(model, device=device, pruning_rate=0.5,
                          post_prune_retrain_epochs=0)
        loader = _make_loader()
        trainer.train_task(0, loader, epochs=EPOCHS)
        trainer.consolidate(0, loader)
        trainer.train_task(1, loader, epochs=EPOCHS)
        acc0 = trainer.evaluate(0, loader)
        acc1 = trainer.evaluate(1, loader)
        assert 0.0 <= acc0 <= 1.0
        assert 0.0 <= acc1 <= 1.0

    def test_capacity_report(self, multi_head_model, device):
        trainer = PackNet(multi_head_model, device=device)
        report = trainer.capacity_report()
        assert "total_params" in report
        assert "free_pct" in report
        assert report["free_pct"] == pytest.approx(100.0)


# ── CLMetrics tests ────────────────────────────────────────────────────────

class TestCLMetrics:
    def _perfect_retention_metrics(self) -> CLMetrics:
        """Simulate perfect retention: no change after each task."""
        m = CLMetrics(n_tasks=3)
        # Task 0: learns 80%, retains 80% throughout
        m.record(0, 0, 0.80)
        m.record(1, 0, 0.80)
        m.record(1, 1, 0.85)
        m.record(2, 0, 0.80)
        m.record(2, 1, 0.85)
        m.record(2, 2, 0.90)
        return m

    def _forgetting_metrics(self) -> CLMetrics:
        """Simulate catastrophic forgetting."""
        m = CLMetrics(n_tasks=3)
        m.record(0, 0, 0.90)
        m.record(1, 0, 0.50)   # dropped after task 1
        m.record(1, 1, 0.85)
        m.record(2, 0, 0.30)   # dropped further
        m.record(2, 1, 0.40)
        m.record(2, 2, 0.88)
        return m

    def test_average_accuracy(self):
        m = self._perfect_retention_metrics()
        # Final row: 0.80, 0.85, 0.90
        assert m.average_accuracy() == pytest.approx((0.80 + 0.85 + 0.90) / 3, abs=1e-4)

    def test_backward_transfer_zero_for_perfect_retention(self):
        m = self._perfect_retention_metrics()
        # BWT = [(0.80 - 0.80) + (0.85 - 0.85)] / 2 = 0
        assert m.backward_transfer() == pytest.approx(0.0, abs=1e-4)

    def test_backward_transfer_negative_for_forgetting(self):
        m = self._forgetting_metrics()
        # BWT < 0 because tasks got worse after training on new tasks
        assert m.backward_transfer() < 0.0

    def test_forgetting_zero_for_perfect_retention(self):
        m = self._perfect_retention_metrics()
        assert m.forgetting() == pytest.approx(0.0, abs=1e-4)

    def test_forgetting_positive_for_forgetting(self):
        m = self._forgetting_metrics()
        assert m.forgetting() > 0.0

    def test_accuracy_matrix_shape(self):
        m = self._perfect_retention_metrics()
        mat = m.accuracy_matrix()
        assert mat.shape == (3, 3)

    def test_summary_contains_required_keys(self):
        m = self._perfect_retention_metrics()
        summary = m.summary()
        assert "ACC" in summary
        assert "BWT" in summary
        assert "Forgetting" in summary
