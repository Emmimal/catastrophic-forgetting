"""
benchmarks/benchmark.py
-----------------------
Head-to-head benchmark: Baseline vs EWC vs Experience Replay vs PackNet.

Runs all four methods on Split-MNIST with identical hyperparameters
and model architectures, then computes ACC, BWT, and Forgetting metrics.

Usage
-----
    python benchmarks/benchmark.py

This will:
1. Download MNIST (once, ~11MB).
2. Train all four methods sequentially on 5 tasks.
3. Print a comparison table.
4. Save results to benchmark_results.json.

Expected runtime: ~2–5 minutes on CPU, ~30 seconds on GPU.
"""

from __future__ import annotations

import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.metrics import CLMetrics
from data.task_data import SplitMNIST
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


# ── Benchmark configuration ────────────────────────────────────────────────

NUM_TASKS      = 5
EPOCHS_PER_TASK = 5
BATCH_SIZE     = 64
SEED           = 42

# Architecture shared by all single-head methods
MLP_INPUT_DIM    = 784
MLP_HIDDEN_DIMS  = [256, 256]

# EWC
EWC_LAMBDA          = 0.4
EWC_FISHER_SAMPLES  = 200

# Experience Replay
REPLAY_BUFFER_SIZE  = 500   # 100 examples per task
REPLAY_RATIO        = 0.5

# PackNet
PACKNET_PRUNE_RATE  = 0.5   # 50% pruned per task — supports up to 5 tasks
PACKNET_RETRAIN_EPS = 3


# ── Helper: evaluate one method on all tasks ───────────────────────────────

def run_method(
    method_name: str,
    benchmark: SplitMNIST,
    device: torch.device,
) -> Dict:
    """
    Train and evaluate one continual learning method on Split-MNIST.

    Returns a dict with:
        method, metrics (CLMetrics summary), per_task_accs, runtime_s
    """
    logger.info("=" * 60)
    logger.info("  Running: %s", method_name)
    logger.info("=" * 60)

    torch.manual_seed(SEED)

    # ── Instantiate method ────────────────────────────────────────────────
    if method_name == "Baseline":
        model = MultiHeadMLP(
            input_dim=MLP_INPUT_DIM,
            hidden_dims=MLP_HIDDEN_DIMS,
            head_output_dim=2,
        )
        for _ in range(NUM_TASKS):
            model.add_task_head()
        trainer = NaiveFineTuning(
            model=model, device=device, multi_head=True
        )
        cl_metrics = CLMetrics(n_tasks=NUM_TASKS)
        needs_consolidate = False
        is_packnet = False

    elif method_name == "EWC":
        model = MultiHeadMLP(
            input_dim=MLP_INPUT_DIM,
            hidden_dims=MLP_HIDDEN_DIMS,
            head_output_dim=2,
        )
        for _ in range(NUM_TASKS):
            model.add_task_head()
        trainer = EWC(
            model=model,
            device=device,
            lambda_ewc=EWC_LAMBDA,
            n_fisher_samples=EWC_FISHER_SAMPLES,
            multi_head=True,
        )
        cl_metrics = CLMetrics(n_tasks=NUM_TASKS)
        needs_consolidate = True
        is_packnet = False

    elif method_name == "ExperienceReplay":
        model = MultiHeadMLP(
            input_dim=MLP_INPUT_DIM,
            hidden_dims=MLP_HIDDEN_DIMS,
            head_output_dim=2,
        )
        for _ in range(NUM_TASKS):
            model.add_task_head()
        trainer = ExperienceReplay(
            model=model,
            device=device,
            buffer_size=REPLAY_BUFFER_SIZE,
            replay_ratio=REPLAY_RATIO,
            multi_head=True,
            seed=SEED,
        )
        cl_metrics = CLMetrics(n_tasks=NUM_TASKS)
        needs_consolidate = False
        is_packnet = False

    elif method_name == "PackNet":
        model = MultiHeadMLP(
            input_dim=MLP_INPUT_DIM,
            hidden_dims=MLP_HIDDEN_DIMS,
            head_output_dim=2,
        )
        for _ in range(NUM_TASKS):
            model.add_task_head()
        trainer = PackNet(
            model=model,
            device=device,
            pruning_rate=PACKNET_PRUNE_RATE,
            post_prune_retrain_epochs=PACKNET_RETRAIN_EPS,
        )
        cl_metrics = CLMetrics(n_tasks=NUM_TASKS)
        needs_consolidate = True
        is_packnet = True

    else:
        raise ValueError(f"Unknown method: {method_name}")

    # ── Training loop ─────────────────────────────────────────────────────
    t0 = time.perf_counter()

    for task_id in range(NUM_TASKS):
        train_loader = benchmark.get_train_loader(task_id, BATCH_SIZE)
        trainer.train_task(task_id, train_loader, epochs=EPOCHS_PER_TASK)

        # Evaluate all tasks seen so far
        for eval_task_id in range(task_id + 1):
            test_loader = benchmark.get_test_loader(eval_task_id)
            acc = trainer.evaluate(eval_task_id, test_loader)
            cl_metrics.record(
                after_task=task_id,
                task_id=eval_task_id,
                accuracy=acc,
            )
            logger.info(
                "  [%s] After Task %d | Acc(Task %d) = %.3f",
                method_name, task_id, eval_task_id, acc,
            )

        # Consolidate before next task
        if needs_consolidate and task_id < NUM_TASKS - 1:
            if is_packnet:
                trainer.consolidate(task_id, train_loader)
            else:
                trainer.consolidate(task_id, train_loader)

    runtime = time.perf_counter() - t0

    # ── Extra: evaluate ALL tasks after full training ─────────────────────
    for task_id in range(NUM_TASKS):
        test_loader = benchmark.get_test_loader(task_id)
        acc = trainer.evaluate(task_id, test_loader)
        cl_metrics.record(
            after_task=NUM_TASKS - 1,
            task_id=task_id,
            accuracy=acc,
        )

    summary = cl_metrics.summary()

    logger.info("")
    logger.info("  [%s] DONE in %.1fs", method_name, runtime)
    logger.info("  ACC=%.3f  BWT=%.3f  Forgetting=%.3f",
                summary["ACC"], summary["BWT"], summary["Forgetting"])
    logger.info("")

    return {
        "method": method_name,
        "metrics": summary,
        "accuracy_matrix": cl_metrics.accuracy_matrix().tolist(),
        "runtime_s": round(runtime, 1),
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Tasks: %d | Epochs/task: %d | Seed: %d",
                NUM_TASKS, EPOCHS_PER_TASK, SEED)

    benchmark = SplitMNIST(num_tasks=NUM_TASKS, seed=SEED)

    methods = ["Baseline", "EWC", "ExperienceReplay", "PackNet"]
    all_results: List[Dict] = []

    for method in methods:
        result = run_method(method, benchmark, device)
        all_results.append(result)

    # ── Print comparison table ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  HEAD-TO-HEAD BENCHMARK: Split-MNIST (5 tasks)")
    print("=" * 70)
    header = f"{'Method':<22} {'ACC':>8} {'BWT':>8} {'Forgetting':>12} {'Runtime':>9}"
    print(header)
    print("-" * 70)

    for r in all_results:
        m = r["metrics"]
        print(
            f"{r['method']:<22} "
            f"{m['ACC']:>8.3f} "
            f"{m['BWT']:>8.3f} "
            f"{m['Forgetting']:>12.3f} "
            f"{r['runtime_s']:>8.1f}s"
        )

    print("=" * 70)
    print()
    print("ACC        = Average accuracy across all tasks (higher is better)")
    print("BWT        = Backward transfer (0 = no forgetting; negative = forgot)")
    print("Forgetting = Max accuracy drop across tasks (lower is better)")
    print()

    # ── Save results ───────────────────────────────────────────────────────
    output_path = Path(__file__).parent / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
