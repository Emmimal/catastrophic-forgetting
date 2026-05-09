# catastrophic-forgetting
PyTorch implementations of methods to prevent catastrophic forgetting, including EWC, Experience Replay, and PackNet for continual learning.

# How to Prevent Catastrophic Forgetting in PyTorch — Complete Code

**Series: Production ML Engineering — Article 05 of 15**
**Companion code for:** [How to Prevent Catastrophic Forgetting in PyTorch (Complete Guide)](https://emitechlogic.com/how-to-prevent-catastrophic-forgetting-in-pytorch/)

---

## What this code does

Neural networks suffer from **catastrophic forgetting**: when fine-tuned on new data,
they overwrite weights that encoded previously learned patterns. This repository
implements and benchmarks three strategies to prevent it:

| Method | Core idea | Forgetting guarantee | Raw data stored? |
|---|---|---|---|
| **Elastic Weight Consolidation (EWC)** | Penalise changes to important weights | Approximate | No |
| **Experience Replay** | Mix old examples into new training batches | Approximate | Yes |
| **PackNet** | Freeze task-specific weights after each task | Exact (zero) | No |

---

## Project layout

```
catastrophic-forgetting/
├── models/
│   └── mlp.py                  # MLP and MultiHeadMLP backbones
├── data/
│   └── task_data.py            # SplitMNIST and PermutedMNIST generators
├── methods/
│   ├── baseline.py             # Naive fine-tuning (forgetting floor)
│   ├── ewc.py                  # Elastic Weight Consolidation
│   ├── experience_replay.py    # Experience Replay + ReplayBuffer
│   └── packnet.py              # PackNet (iterative pruning + freezing)
├── benchmarks/
│   ├── metrics.py              # ACC, BWT, Forgetting, FWT
│   └── benchmark.py            # Head-to-head Split-MNIST runner
├── tests/
│   └── test_catastrophic_forgetting.py   # 36 tests, all passing
├── demo.py                     # Quick demo (no MNIST download needed)
└── requirements.txt
```

---

## Quickstart

```bash
pip install -r requirements.txt

# 2-task demo (synthetic data, runs in ~10s)
python demo.py

# 5-task Split-MNIST benchmark (~3–5 min on CPU)
python benchmarks/benchmark.py
```

---

## Code blocks mapped to article sections

### H2: What Is Catastrophic Forgetting?

```python
# demo.py → demo_catastrophic_forgetting()
# Shows: train Task A → 99%+ acc, then train Task B → Task A acc collapses
```

### H2: Why Neural Networks Forget — The Gradient Interference Problem

The gradient for Task B points in a direction that minimises Task B's loss.
That direction often increases Task A's loss, because the two loss landscapes
have different optima. Without any mechanism to constrain parameter movement,
Task B gradients overwrite the weights that mattered for Task A.

### H3: How EWC Works — Fisher Information Matrix Explained

The Fisher Information Matrix diagonal `F_i` measures how sensitive the loss
is to changes in parameter `θ_i`. Parameters with high Fisher values are
critical for prior tasks. EWC adds a quadratic penalty that resists changes
to high-Fisher parameters:

```
L_total = L_task_B + (λ/2) * Σ_i F_i * (θ_i - θ*_i)²
```

where `θ*_i` is the parameter value after training Task A (the anchor).

### H3: EWC Implementation in PyTorch (Full Code)

```python
# methods/ewc.py

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
        fisher    = fisher_diag[name]
        old_param = original_params[name]
        ewc_penalty += (fisher * (param - old_param) ** 2).sum()

    return task_loss + (lambda_ewc / 2) * ewc_penalty
```

**Using the EWC trainer class:**

```python
from methods.ewc import EWC
from models.mlp import MultiHeadMLP

model = MultiHeadMLP(input_dim=784, hidden_dims=[256, 256], head_output_dim=2)
model.add_task_head()   # Task 0
model.add_task_head()   # Task 1

trainer = EWC(model=model, lambda_ewc=0.4, n_fisher_samples=200, multi_head=True)

# Task 0
trainer.train_task(0, train_loader_task0, epochs=5)
trainer.consolidate(0, train_loader_task0)   # ← compute Fisher + anchor

# Task 1 — EWC penalty protects Task 0's weights
trainer.train_task(1, train_loader_task1, epochs=5)

acc_task0 = trainer.evaluate(0, test_loader_task0)
acc_task1 = trainer.evaluate(1, test_loader_task1)
```

**Fisher diagonal estimation (empirical Fisher approximation):**

```python
# methods/ewc.py → EWC._estimate_fisher()
# For each sample, computes the gradient of log P(ŷ|x) w.r.t. each parameter.
# The Fisher diagonal = average of squared gradients.
# High Fisher value → parameter is important for prior task → penalise changes.
```

---

### H2: Method 2 — Experience Replay in PyTorch

```python
from methods.experience_replay import ExperienceReplay

trainer = ExperienceReplay(
    model=model,
    buffer_size=500,      # Total examples stored across all tasks
    replay_ratio=0.5,     # 50% of each mini-batch from the replay buffer
    multi_head=True,
)

# Training automatically fills the buffer and replays old examples
trainer.train_task(0, train_loader_task0, epochs=5)
trainer.train_task(1, train_loader_task1, epochs=5)   # buffer active here
```

**Reservoir sampling** (`methods/experience_replay.py → ReplayBuffer`):
Maintains a uniform random sample of all examples seen — no task label
needed at sampling time, and memory is bounded by `buffer_size`.

---

### H2: Method 3 — PackNet for Task-Incremental Learning

```python
from methods.packnet import PackNet
from models.mlp import MultiHeadMLP

model = MultiHeadMLP(input_dim=784, hidden_dims=[256, 256], head_output_dim=2)
model.add_task_head()
model.add_task_head()

trainer = PackNet(
    model=model,
    pruning_rate=0.5,             # Prune 50% of free weights after each task
    post_prune_retrain_epochs=3,  # Fine-tune survivors before freezing
)

trainer.train_task(0, train_loader_task0, epochs=5)
trainer.consolidate(0, train_loader_task0)   # prune + freeze Task 0 weights

trainer.train_task(1, train_loader_task1, epochs=5)

# Task ID required at inference — strictly task-incremental
acc_task0 = trainer.evaluate(0, test_loader_task0)   # uses Task 0 mask
acc_task1 = trainer.evaluate(1, test_loader_task1)   # uses Task 1 mask

print(trainer.capacity_report())
# {'total_params': 83460, 'free_params': 41730, 'free_pct': 50.0, ...}
```

---

### H2: EWC vs Experience Replay vs PackNet — Head-to-Head Benchmark

```bash
python benchmarks/benchmark.py
```

Expected output (Split-MNIST, 5 tasks, CPU):

```
======================================================================
  HEAD-TO-HEAD BENCHMARK: Split-MNIST (5 tasks)
======================================================================
Method                      ACC      BWT   Forgetting   Runtime
----------------------------------------------------------------------
Baseline                  0.xxx    x.xxx        x.xxx      xx.xs
EWC                       0.xxx    x.xxx        x.xxx      xx.xs
ExperienceReplay          0.xxx    x.xxx        x.xxx      xx.xs
PackNet                   0.xxx    x.xxx        x.xxx      xx.xs
======================================================================

ACC        = Average accuracy across all tasks (higher is better)
BWT        = Backward transfer (0 = no forgetting; negative = forgot)
Forgetting = Max accuracy drop across tasks (lower is better)
```

**Metrics explained (`benchmarks/metrics.py`):**

```python
from benchmarks.metrics import CLMetrics

metrics = CLMetrics(n_tasks=5)

# Record accuracy on task_j after training tasks 0..i
metrics.record(after_task=0, task_id=0, accuracy=0.95)
metrics.record(after_task=1, task_id=0, accuracy=0.72)   # dropped → forgetting
metrics.record(after_task=1, task_id=1, accuracy=0.91)
# ... record all entries ...

print(metrics.summary())
# {'ACC': 0.xxx, 'BWT': -0.xxx, 'Forgetting': 0.xxx}
metrics.print_matrix()   # Full accuracy matrix
```

---

### H2: When Each Method Breaks Down

| Method | Breaks when... |
|---|---|
| **EWC** | Many tasks accumulate (Fisher grows unboundedly in online EWC). Task similarity is high — Fisher can't distinguish important vs. shared weights. |
| **Experience Replay** | Raw data can't be stored (privacy regulations, GDPR). Buffer too small relative to task count → reservoir gradually forgets old tasks anyway. |
| **PackNet** | Task ID is unknown at test time (class-incremental / domain-incremental). More tasks than `pruning_rate` allows — network runs out of capacity. |

---

## Test suite

```bash
python -m pytest tests/ -v
# 36 passed in ~72s
```

Key tests:
- `test_anchor_params_unchanged_after_consolidate` — EWC anchor never drifts
- `test_fisher_diagonal_non_negative` — Fisher must be non-negative (it's a squared gradient)
- `test_frozen_params_do_not_change` — PackNet frozen weights are immutable
- `test_capacity_not_exceeded` — ReplayBuffer respects its capacity bound
- `test_backward_transfer_negative_for_forgetting` — CLMetrics catches forgetting

---

## References

Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks.
*Proceedings of the National Academy of Sciences*, 114(13), 3521–3526.
https://doi.org/10.1073/pnas.1611835114

Mallya, A., & Lazebnik, S. (2018). PackNet: Adding Multiple Tasks to a Single
Network by Iterative Pruning. *CVPR Workshops*.
https://arxiv.org/abs/1711.05769

Robins, A. (1995). Catastrophic forgetting, rehearsal and pseudorehearsal.
*Connection Science*, 7(2), 123–146.

Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for continual
learning. *NeurIPS*, 30.

Vitter, J. S. (1985). Random sampling with a reservoir.
*ACM Transactions on Mathematical Software*, 11(1), 37–57.

---

## Series navigation

← Article 04: [Model Versioning in Production Machine Learning](https://emitechlogic.com/model-versioning-in-production-machine-learning/)

→ Article 06: Online Learning in Python: How to Train Models on Streaming Data *(coming soon)*
