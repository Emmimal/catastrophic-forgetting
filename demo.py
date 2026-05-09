"""
demo.py
-------
Self-contained demo: catastrophic forgetting and how to prevent it.

This script demonstrates all three methods on a simple 2-task scenario
using synthetic data (no MNIST download needed), so it runs in seconds.

For the full 5-task Split-MNIST benchmark, run:
    python benchmarks/benchmark.py

Usage
-----
    python demo.py
"""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.metrics import CLMetrics
from methods.baseline import NaiveFineTuning
from methods.ewc import EWC
from methods.experience_replay import ExperienceReplay
from methods.packnet import PackNet
from models.mlp import MLP, MultiHeadMLP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Shared configuration ───────────────────────────────────────────────────

SEED       = 42
INPUT_DIM  = 64
N_SAMPLES  = 200
BATCH_SIZE = 32
EPOCHS     = 8

torch.manual_seed(SEED)
device = torch.device("cpu")


def make_task(n: int = N_SAMPLES, seed: int = 0) -> DataLoader:
    """Synthetic 2-class classification task."""
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(n, INPUT_DIM, generator=rng)
    y = torch.randint(0, 2, (n,), generator=rng)
    return DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=True)


# ── Section 1: What is catastrophic forgetting? (Baseline demo) ───────────

def demo_catastrophic_forgetting():
    """
    Shows what happens with naive fine-tuning:
    - Train on Task A → high accuracy on A
    - Train on Task B → accuracy on A collapses
    """
    print("\n" + "=" * 60)
    print("  DEMO 1: Catastrophic Forgetting (Baseline)")
    print("=" * 60)

    model = MultiHeadMLP(INPUT_DIM, [128, 128], head_output_dim=2)
    model.add_task_head()  # Task 0
    model.add_task_head()  # Task 1

    trainer = NaiveFineTuning(model, device=device, multi_head=True)

    loader_a = make_task(seed=0)
    loader_b = make_task(seed=1)

    # Train on Task A
    trainer.train_task(0, loader_a, epochs=EPOCHS)
    acc_a_after_a = trainer.evaluate(0, loader_a)

    # Now train on Task B — this overwrites what was learned for Task A
    trainer.train_task(1, loader_b, epochs=EPOCHS)
    acc_a_after_b = trainer.evaluate(0, loader_a)
    acc_b_after_b = trainer.evaluate(1, loader_b)

    print(f"\n  Accuracy on Task A after training on A only:  {acc_a_after_a:.1%}")
    print(f"  Accuracy on Task A after training on B also:  {acc_a_after_b:.1%}")
    print(f"  Accuracy on Task B after training on B:       {acc_b_after_b:.1%}")
    print(f"\n  Forgetting on Task A: {(acc_a_after_a - acc_a_after_b):.1%}")
    print("\n  → Catastrophic forgetting: Task A knowledge was overwritten.\n")


# ── Section 2: Method 1 — EWC ──────────────────────────────────────────────

def demo_ewc():
    """
    Demonstrates Elastic Weight Consolidation.
    Shows how lambda_ewc controls the forgetting/plasticity trade-off.
    """
    print("\n" + "=" * 60)
    print("  DEMO 2: Method 1 — Elastic Weight Consolidation (EWC)")
    print("=" * 60)

    # --- Core EWC code (Article H3: EWC Implementation in PyTorch) ---

    def ewc_loss(model, original_params, fisher_diag, criterion,
                 outputs, labels, lambda_ewc=0.4):
        """
        Combines task loss with EWC regularisation.

        lambda_ewc controls a genuine tension:
          - Too high → model resists learning new patterns (underfits new distribution)
          - Too low  → model overwrites old knowledge (catastrophic forgetting)
        Tune this on a held-out evaluation set that includes both old and new examples.
        """
        task_loss = criterion(outputs, labels)

        ewc_penalty = 0
        for name, param in model.named_parameters():
            fisher = fisher_diag[name]
            old_param = original_params[name]
            ewc_penalty += (fisher * (param - old_param) ** 2).sum()

        return task_loss + (lambda_ewc / 2) * ewc_penalty

    # --- Using the EWC trainer class ---
    for lambda_val in [0.0, 0.4, 4.0]:
        torch.manual_seed(SEED)
        model = MultiHeadMLP(INPUT_DIM, [128, 128], head_output_dim=2)
        model.add_task_head()
        model.add_task_head()

        trainer = EWC(
            model=model,
            device=device,
            lambda_ewc=lambda_val,
            n_fisher_samples=100,
            multi_head=True,
        )

        loader_a = make_task(seed=0)
        loader_b = make_task(seed=1)

        trainer.train_task(0, loader_a, epochs=EPOCHS)
        acc_a_after_a = trainer.evaluate(0, loader_a)

        trainer.consolidate(0, loader_a)          # ← compute Fisher + anchor

        trainer.train_task(1, loader_b, epochs=EPOCHS)
        acc_a_after_b = trainer.evaluate(0, loader_a)
        acc_b_after_b = trainer.evaluate(1, loader_b)

        print(
            f"\n  λ={lambda_val:.1f}  "
            f"| Task A after B: {acc_a_after_b:.1%}  "
            f"| Task B: {acc_b_after_b:.1%}  "
            f"| Forgetting: {(acc_a_after_a - acc_a_after_b):+.1%}"
        )

    print()
    print("  → λ=0.0 degenerates to baseline (full forgetting).")
    print("  → λ=0.4 balances retention and plasticity.")
    print("  → λ=4.0 protects Task A but inhibits Task B learning.")


# ── Section 3: Method 2 — Experience Replay ────────────────────────────────

def demo_experience_replay():
    """
    Demonstrates Experience Replay with reservoir sampling.
    """
    print("\n" + "=" * 60)
    print("  DEMO 3: Method 2 — Experience Replay")
    print("=" * 60)

    torch.manual_seed(SEED)
    model = MultiHeadMLP(INPUT_DIM, [128, 128], head_output_dim=2)
    model.add_task_head()
    model.add_task_head()

    trainer = ExperienceReplay(
        model=model,
        device=device,
        buffer_size=200,      # 200 examples total across all tasks
        replay_ratio=0.5,     # 50% of each mini-batch from the buffer
        multi_head=True,
        seed=SEED,
    )

    loader_a = make_task(seed=0)
    loader_b = make_task(seed=1)

    trainer.train_task(0, loader_a, epochs=EPOCHS)
    acc_a_after_a = trainer.evaluate(0, loader_a)

    trainer.train_task(1, loader_b, epochs=EPOCHS)   # buffer fills with Task A
    acc_a_after_b = trainer.evaluate(0, loader_a)
    acc_b_after_b = trainer.evaluate(1, loader_b)

    stats = trainer.buffer_stats()

    print(f"\n  Accuracy on Task A after training on A only:  {acc_a_after_a:.1%}")
    print(f"  Accuracy on Task A after training on B also:  {acc_a_after_b:.1%}")
    print(f"  Accuracy on Task B:                           {acc_b_after_b:.1%}")
    print(f"  Forgetting on Task A: {(acc_a_after_a - acc_a_after_b):+.1%}")
    print(f"\n  Buffer: {stats['size']}/{stats['capacity']} used")
    print(f"  Task distribution: {stats['task_distribution']}")
    print()
    print("  → Replay significantly reduces forgetting by keeping old examples alive.")


# ── Section 4: Method 3 — PackNet ──────────────────────────────────────────

def demo_packnet():
    """
    Demonstrates PackNet's zero-forgetting guarantee via weight masking.
    """
    print("\n" + "=" * 60)
    print("  DEMO 4: Method 3 — PackNet")
    print("=" * 60)

    torch.manual_seed(SEED)
    model = MultiHeadMLP(INPUT_DIM, [256, 256], head_output_dim=2)
    model.add_task_head()
    model.add_task_head()

    trainer = PackNet(
        model=model,
        device=device,
        pruning_rate=0.5,             # Prune 50% of free weights per task
        post_prune_retrain_epochs=3,  # Fine-tune survivors after pruning
    )

    loader_a = make_task(seed=0)
    loader_b = make_task(seed=1)

    trainer.train_task(0, loader_a, epochs=EPOCHS)
    acc_a_after_a = trainer.evaluate(0, loader_a)

    trainer.consolidate(0, loader_a)   # prune + freeze Task A weights

    cap_before = trainer.capacity_report()
    trainer.train_task(1, loader_b, epochs=EPOCHS)
    cap_after = trainer.capacity_report()

    acc_a_after_b = trainer.evaluate(0, loader_a)  # with Task 0 mask
    acc_b_after_b = trainer.evaluate(1, loader_b)  # with Task 1 mask

    print(f"\n  Accuracy on Task A after training on A only:  {acc_a_after_a:.1%}")
    print(f"  Accuracy on Task A after training on B also:  {acc_a_after_b:.1%}")
    print(f"  Accuracy on Task B:                           {acc_b_after_b:.1%}")
    print(f"  Forgetting on Task A: {(acc_a_after_a - acc_a_after_b):+.1%}")
    print(f"\n  Network capacity after consolidation:")
    print(f"    Total params:   {cap_before['total_params']:,}")
    print(f"    Frozen (Task 0): {cap_before['total_params'] - cap_before['free_params']:,}")
    print(f"    Free for future: {cap_after['free_params']:,} ({cap_after['free_pct']:.0f}%)")
    print()
    print("  → PackNet guarantees zero forgetting by never touching Task A weights.")
    print("  → Trade-off: network capacity shrinks with each new task.")


# ── Section 5: Summary comparison ──────────────────────────────────────────

def demo_summary():
    """
    Side-by-side comparison: same tasks, same model, all four approaches.
    """
    print("\n" + "=" * 60)
    print("  DEMO 5: EWC vs Experience Replay vs PackNet — Summary")
    print("=" * 60)

    results = []

    configs = [
        ("Baseline",    dict(method="baseline")),
        ("EWC (λ=0.4)", dict(method="ewc", lambda_ewc=0.4)),
        ("Replay",      dict(method="replay", buffer_size=200)),
        ("PackNet",     dict(method="packnet", pruning_rate=0.5)),
    ]

    for label, cfg in configs:
        torch.manual_seed(SEED)
        model = MultiHeadMLP(INPUT_DIM, [128, 128], head_output_dim=2)
        model.add_task_head()
        model.add_task_head()

        method = cfg["method"]
        loader_a = make_task(seed=0)
        loader_b = make_task(seed=1)

        if method == "baseline":
            trainer = NaiveFineTuning(model, device=device, multi_head=True)
            trainer.train_task(0, loader_a, epochs=EPOCHS)
            acc_a_after_a = trainer.evaluate(0, loader_a)
            trainer.train_task(1, loader_b, epochs=EPOCHS)

        elif method == "ewc":
            trainer = EWC(model, device=device,
                          lambda_ewc=cfg["lambda_ewc"],
                          n_fisher_samples=100, multi_head=True)
            trainer.train_task(0, loader_a, epochs=EPOCHS)
            acc_a_after_a = trainer.evaluate(0, loader_a)
            trainer.consolidate(0, loader_a)
            trainer.train_task(1, loader_b, epochs=EPOCHS)

        elif method == "replay":
            trainer = ExperienceReplay(model, device=device,
                                       buffer_size=cfg["buffer_size"],
                                       replay_ratio=0.5, multi_head=True)
            trainer.train_task(0, loader_a, epochs=EPOCHS)
            acc_a_after_a = trainer.evaluate(0, loader_a)
            trainer.train_task(1, loader_b, epochs=EPOCHS)

        elif method == "packnet":
            trainer = PackNet(model, device=device,
                              pruning_rate=cfg["pruning_rate"],
                              post_prune_retrain_epochs=2)
            trainer.train_task(0, loader_a, epochs=EPOCHS)
            acc_a_after_a = trainer.evaluate(0, loader_a)
            trainer.consolidate(0, loader_a)
            trainer.train_task(1, loader_b, epochs=EPOCHS)

        acc_a = trainer.evaluate(0, loader_a)
        acc_b = trainer.evaluate(1, loader_b)
        forgetting = acc_a_after_a - acc_a

        results.append({
            "label": label,
            "task_a": acc_a,
            "task_b": acc_b,
            "forgetting": forgetting,
        })

    print(f"\n  {'Method':<22} {'Task A':>8} {'Task B':>8} {'Forgetting':>12}")
    print("  " + "-" * 54)
    for r in results:
        print(
            f"  {r['label']:<22} "
            f"{r['task_a']:>8.1%} "
            f"{r['task_b']:>8.1%} "
            f"{r['forgetting']:>+12.1%}"
        )
    print()
    print("  Note: Results vary by random seed; run benchmarks/benchmark.py")
    print("  for statistically robust Split-MNIST results.\n")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Catastrophic Forgetting in PyTorch — Demo")
    print("  Series: Production ML Engineering, Article 05")
    print("=" * 60)

    demo_catastrophic_forgetting()
    demo_ewc()
    demo_experience_replay()
    demo_packnet()
    demo_summary()

    print("=" * 60)
    print("  Demo complete.")
    print("  For the full Split-MNIST benchmark, run:")
    print("    python benchmarks/benchmark.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
