# SPEC-TUNE-2026-001: Automatic Hyperparameter Tuning for Classification Fine-Tuning

**Version**: 1.0.0
**Status**: DRAFT
**Author**: paiml engineering
**Date**: 2026-02-28
**Requires**: entrenar >= 1.1, aprender >= 0.27.0, trueno >= 0.15.0
**References**: SSC-028 (Shell Safety Classifier auto-tuning)

---

## Abstract

This specification defines `apr tune`, a production-grade automatic hyperparameter
optimization (HPO) system for classification fine-tuning in the entrenar training engine.
The system combines Bayesian optimization (TPE), successive halving (ASHA), and
population-based training (PBT) to find optimal LoRA + classifier configurations without
manual intervention.

The design draws on lessons from six industry-leading AutoML systems:

| System | Key Insight Adopted |
|--------|-------------------|
| [Optuna](https://optuna.org/) | Define-by-run API, TPE with Parzen estimators, pruning integration |
| [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) | Scheduler/searcher separation, ASHA for early stopping, PBT for schedule tuning |
| [H2O AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) | Leaderboard ranking, stacked ensembles, time-based budgets, cross-validation |
| [AutoGluon](https://auto.gluon.ai/) | Multi-fidelity optimization, foundation model adaptation, 3-line API |
| [SageMaker Autopilot](https://aws.amazon.com/sagemaker/ai/autopilot/) | Bayesian vs multi-fidelity selection by dataset size, white-box explainability |
| [Ludwig](https://github.com/ludwig-ai/ludwig) | Minimal-config auto_train with time budgets |

The goal is not to clone these systems but to provide **sovereign, Rust-native HPO** that
equals their tuning quality while running entirely within the paiml stack (trueno + aprender +
entrenar + realizar) — no Python, no external services.

---

## 1. Motivation

### 1.1 The Problem

The first SSC (Shell Safety Classifier) training run achieved:

| Metric | Value | Baseline |
|--------|-------|----------|
| Val accuracy | 74.9% | 58.9% (majority class) |
| Val loss | 1.53 | 1.61 (random, ln(5)) |

This is only 16 points above majority-class guessing and barely below random loss. The
hyperparameters were manually selected (LR=1e-4, rank=16, uniform class weights) without
systematic search. The class-imbalanced dataset (58.9% safe) dominates the unweighted loss.

### 1.2 Why AutoML

Manual tuning is:
- **Slow**: Each training run takes 7-14 hours on an RTX 4090
- **Biased**: Human intuition favors familiar configurations
- **Incomplete**: 7+ continuous parameters create a vast search space
- **Non-reproducible**: Ad-hoc choices are hard to document

Automatic tuning is:
- **Systematic**: Explores the space according to principled algorithms
- **Budget-aware**: Allocates resources to promising configurations via early stopping
- **Reproducible**: Seeded search with full trial history
- **Self-documenting**: Every configuration and result is logged

### 1.3 Why Sovereign

Existing HPO frameworks (Optuna, Ray Tune) are Python-only. Using them would:
1. Break the sovereign stack principle (no Python dependency)
2. Require IPC between Rust training and Python tuning
3. Add ~500MB of Python runtime to the deployment
4. Create version-compatibility issues across Python/Rust boundaries

entrenar already has TPE, Grid Search, and Hyperband schedulers in Rust
(`entrenar/src/optim/hpo/`). This spec wires them to the classification pipeline.

### 1.4 Citations

| # | Citation | Relevance |
|---|----------|-----------|
| C1 | Bergstra et al. (2011). *Algorithms for Hyper-Parameter Optimization*. NeurIPS. | TPE algorithm foundation |
| C2 | Li et al. (2018). *Hyperband: A Novel Bandit-Based Approach*. JMLR 18. | Hyperband/successive halving |
| C3 | Li et al. (2020). *A System for Massively Parallel Hyperparameter Tuning*. MLSys. | ASHA algorithm |
| C4 | Jaderberg et al. (2017). *Population Based Training of Neural Networks*. arXiv. | PBT for schedule optimization |
| C5 | Akiba et al. (2019). *Optuna: A Next-generation HPO Framework*. KDD. | Define-by-run API, pruning |
| C6 | Erickson et al. (2020). *AutoGluon-Tabular: Robust and Accurate AutoML*. ICML Workshop. | Multi-layer stacking, time budgets |
| C7 | Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR. | LoRA fine-tuning methodology |
| C8 | Raschka (2025). *Practical Tips for Finetuning LLMs Using LoRA*. | LoRA HP best practices |
| C9 | Unsloth (2025). *LoRA Hyperparameters Guide*. | LR 2e-4 baseline, alpha=rank heuristic |

---

## 2. Design Principles

### 2.1 Three-Line API (inspired by AutoGluon/Ludwig)

The simplest usage should require minimal configuration:

```bash
apr tune --task classify --data corpus.jsonl --budget 10
```

This auto-selects: TPE strategy, scout mode (1 epoch/trial), default search space,
auto-detected model size, and outputs a leaderboard + best config.

### 2.2 Searcher/Scheduler Separation (inspired by Ray Tune)

Like Ray Tune, separate the **searcher** (what config to try next) from the **scheduler**
(when to stop a trial early):

| Component | Role | Implementations |
|-----------|------|----------------|
| **Searcher** | Suggests next hyperparameter config | TPE, Grid, Random |
| **Scheduler** | Decides whether to continue/stop a trial | ASHA, Median, None |
| **Executor** | Runs a trial (ClassifyTrainer) | Sequential, (future: Parallel) |

This allows mixing: TPE searcher + ASHA scheduler = Bayesian-guided search with
aggressive early stopping (the BOHB combination from Ray Tune).

### 2.3 Leaderboard (inspired by H2O AutoML)

Every trial is ranked on a live leaderboard sorted by objective (val_loss or val_accuracy):

```
╭──────────────────────────────────────────────────────────────────╮
│  apr tune — Leaderboard (10/10 trials complete)                  │
├────┬──────────┬──────────┬────────┬──────┬────────┬─────────────┤
│ #  │ Val Loss │ Val Acc  │ LR     │ Rank │ Wt     │ Time        │
├────┼──────────┼──────────┼────────┼──────┼────────┼─────────────┤
│  1 │ 0.682    │ 86.3%    │ 3.2e-5 │ 32   │ sqrt   │ 2h 31m      │
│  2 │ 0.741    │ 84.7%    │ 5.1e-5 │ 16   │ inv    │ 2h 28m      │
│  3 │ 0.823    │ 82.1%    │ 1.8e-5 │ 32   │ inv    │ 2h 35m      │
│ ...│          │          │        │      │        │             │
│ 10 │ 1.412    │ 71.2%    │ 8.3e-4 │ 4    │ unif   │ 1h 12m      │
╰────┴──────────┴──────────┴────────┴──────┴────────┴─────────────╯
```

### 2.4 Multi-Fidelity (inspired by SageMaker Autopilot)

Adapt strategy based on dataset size (SageMaker's approach):

| Dataset Size | Strategy | Rationale |
|-------------|----------|-----------|
| < 1,000 | Grid + full epochs | Small enough for exhaustive search |
| 1,000 - 50,000 | TPE + ASHA scout | Bayesian with early stopping |
| > 50,000 | Random + Hyperband | Massively parallel-friendly |

For SSC (29,307 samples), the default is TPE + ASHA.

### 2.5 LoRA-Specific Search Space (informed by C7, C8, C9)

Based on LoRA fine-tuning best practices from the literature:

| Parameter | Research Finding | Our Range |
|-----------|-----------------|-----------|
| Learning rate | 2e-4 starting point, range 5e-6 to 2e-4 (C9) | 5e-6 .. 5e-4 (log) |
| Rank | No good heuristic, must explore per dataset (C8) | 4 .. 64 (discrete) |
| Alpha | alpha=rank is reliable baseline, alpha=2*rank more aggressive (C9) | 0.5x .. 2x rank |
| Target modules | All major linear layers best, Q+V is the minimum (C8) | Q+V, Q+K+V, all |
| Batch size | Performance declines above 128 for LoRA (C8) | 8 .. 128 (discrete) |
| LR invariant to rank | Optimal LR mostly invariant across ranks early on (C8) | Single LR search |

### 2.6 Explainability (inspired by H2O, SageMaker)

Every tuning run produces a **white-box report**:

1. **Parameter importance**: Which HPs had the most impact on val_loss (via fANOVA-style analysis)
2. **Convergence plot**: Val loss over trials (shows whether more budget would help)
3. **Parallel coordinates**: Visualize HP interactions
4. **Best config export**: Ready-to-use ClassifyConfig for production training

---

## 3. Architecture

```
                    apr tune --task classify \
                      --data corpus.jsonl \
                      --budget 10 --strategy tpe
                           |
                           v
                    +--------------+
                    |  apr-cli     |
                    |  tune.rs     |  (orchestration)
                    +------+-------+
                           |
                           v
                    +--------------+
                    |  entrenar    |
                    |              |
                    | ClassifyTuner|  (NEW: HPO orchestrator)
                    |   |          |
                    |   +-> Searcher (TPE/Grid/Random)
                    |   +-> Scheduler (ASHA/Median/None)
                    |   +-> Executor (ClassifyTrainer per trial)
                    |   +-> Leaderboard (ranked trial results)
                    |              |
                    +------+-------+
                           |
                    (per trial)
                           |
                           v
                    +--------------+
                    |ClassifyTrainer|
                    |  + Pipeline   |  (Transformer + LoRA + ClassHead)
                    |  + Validator  |  (val split, early stopping)
                    |  + Checkpoint |  (dual APR + SafeTensors)
                    +--------------+
```

### 3.1 Component Responsibilities

| Component | Crate | Responsibility |
|-----------|-------|---------------|
| `ClassifyTuner` | entrenar | HPO orchestration: search space, trial lifecycle, leaderboard |
| `TuneSearcher` | entrenar | Trait: `suggest() -> Trial`, `record(trial, score)`, `best()` |
| `TuneScheduler` | entrenar | Trait: `should_stop(trial, epoch, metric) -> bool` |
| `ClassifyTrainer` | entrenar | Single trial execution: train loop, validation, checkpoint |
| `ClassifyPipeline` | entrenar | Model forward/backward: Transformer + LoRA + loss |
| `TPEOptimizer` | entrenar (existing) | Bayesian optimization via Tree-structured Parzen Estimators |
| `HyperbandScheduler` | entrenar (existing) | Successive halving for early stopping |
| `apr tune` CLI | apr-cli | User interface, config parsing, result display |

### 3.2 Data Flow

```
corpus.jsonl
    │
    ├─── Fixed val split (20%, frozen across all trials)
    │
    └─── For each trial:
           │
           ├── Searcher.suggest() → {lr, rank, alpha, batch, warmup, clip, weights}
           │
           ├── Build ClassifyConfig + TrainingConfig from trial params
           │
           ├── Create fresh ClassifyPipeline (new LoRA adapters per rank)
           │
           ├── ClassifyTrainer.train()
           │     │
           │     ├── Per epoch: train → validate → checkpoint
           │     │
           │     └── Scheduler.should_stop(epoch_metrics)?
           │           ├── Continue → next epoch
           │           └── Stop → prune trial
           │
           ├── Record: Searcher.record(trial, best_val_loss)
           │
           └── Update leaderboard
```

---

## 4. Search Space Specification

### 4.1 Default Search Space (Classification + LoRA)

```yaml
search_space:
  learning_rate:
    type: continuous
    low: 5e-6
    high: 5e-4
    log_scale: true
    default: 1e-4
    rationale: "LoRA optimal range per Raschka (2025), Unsloth guide"

  lora_rank:
    type: discrete
    low: 4
    high: 64
    step: 4
    default: 16
    rationale: "Must explore per dataset (no universal heuristic)"

  lora_alpha_ratio:
    type: continuous
    low: 0.5
    high: 2.0
    log_scale: false
    default: 1.0
    rationale: "alpha = ratio * rank. 1.0 = reliable baseline (C9)"

  batch_size:
    type: discrete
    values: [8, 16, 32, 64, 128]
    default: 32
    rationale: "LoRA degrades above 128 (C8)"

  warmup_fraction:
    type: continuous
    low: 0.01
    high: 0.2
    log_scale: false
    default: 0.1

  gradient_clip_norm:
    type: continuous
    low: 0.5
    high: 5.0
    log_scale: false
    default: 1.0

  class_weights:
    type: categorical
    choices: ["uniform", "inverse_freq", "sqrt_inverse"]
    default: "uniform"
    rationale: "Critical for imbalanced datasets (SSC: 58.9% class 0)"

  target_modules:
    type: categorical
    choices: ["qv", "qkv", "all_linear"]
    default: "qv"
    rationale: "All linear best but most expensive; Q+V is baseline"

  lr_min_ratio:
    type: continuous
    low: 0.001
    high: 0.1
    log_scale: true
    default: 0.01
    rationale: "Cosine decay floor = lr * lr_min_ratio"
```

### 4.2 Derived Parameters

Some parameters are derived from others (not independently searched):

| Derived | Formula | Rationale |
|---------|---------|-----------|
| `lora_alpha` | `lora_rank * lora_alpha_ratio` | Ties alpha to rank per best practice |
| `lr_min` | `learning_rate * lr_min_ratio` | Cosine floor relative to peak |
| `accumulation_steps` | `max(1, 128 / batch_size)` | Effective batch ~128 |

### 4.3 Fixed Parameters (Not Searched)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_classes` | 5 | Task-specific, fixed |
| `max_seq_len` | 512 | Qwen2.5 context, memory-bounded |
| `val_split` | 0.2 | Standard, frozen across trials |
| `optimizer` | AdamW | Industry standard for LoRA |

---

## 5. Searcher Specifications

### 5.1 TPE Searcher (Default)

Based on Bergstra et al. (2011), implemented in `entrenar/src/optim/hpo/tpe/optimizer.rs`.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `gamma` | 0.25 | Top 25% as "good" trials (Optuna default) |
| `n_startup` | max(3, budget/3) | Random exploration before Bayesian guidance |
| `kde_bandwidth` | 1.0 | Kernel density estimation bandwidth |

**Algorithm**:
1. For first `n_startup` trials: sample uniformly from search space
2. After `n_startup`: split completed trials into "good" (top gamma quantile) and "bad"
3. Fit Parzen estimators `l(x)` and `g(x)` to good and bad trials respectively
4. Sample from `l(x)/g(x)` to maximize Expected Improvement

### 5.2 Grid Searcher

For small search spaces or when exhaustive coverage is needed.

```rust
GridSearcher::new(vec![
    ("learning_rate", vec![1e-5, 5e-5, 1e-4, 5e-4]),
    ("lora_rank", vec![8, 16, 32]),
    ("class_weights", vec!["uniform", "inverse_freq", "sqrt_inverse"]),
])
// Total: 4 * 3 * 3 = 36 trials
```

### 5.3 Random Searcher

Uniform random sampling. Simple baseline, often competitive with Bayesian methods
for < 20 trials (Bergstra & Bengio, 2012).

---

## 6. Scheduler Specifications

### 6.1 ASHA Scheduler (Default)

Asynchronous Successive Halving (Li et al., 2020). Prunes underperforming trials early.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `grace_period` | 1 epoch | Minimum epochs before pruning eligible |
| `reduction_factor` (eta) | 3 | Keep top 1/3 at each rung |
| `max_resource` | max_epochs | Maximum epochs any trial can run |

**Rung schedule** (for max_epochs=27, eta=3):
- Rung 0: All trials run 1 epoch
- Rung 1: Top 1/3 run 3 epochs
- Rung 2: Top 1/9 run 9 epochs
- Rung 3: Top 1/27 run 27 epochs

This aggressively prunes bad configs after just 1 epoch (~2.5h), focusing compute
on the most promising candidates.

### 6.2 Median Scheduler

Prune trials whose intermediate metric is worse than the median of all trials at the
same epoch (Optuna's MedianPruner). Simpler than ASHA but still effective.

| Parameter | Value |
|-----------|-------|
| `n_warmup_steps` | 1 epoch |
| `percentile` | 50.0 (median) |

### 6.3 No Scheduler

Run every trial for the full budget. Used with scout mode (1 epoch per trial) where
scheduling is unnecessary.

---

## 7. Executor Specification

### 7.1 Sequential Executor (v1)

Runs one trial at a time. Sufficient for single-GPU training.

```
Trial 1: [train 1 epoch] → [validate] → [report]
Trial 2: [train 1 epoch] → [validate] → [report]
...
Trial N: [train 1 epoch] → [validate] → [report]
```

### 7.2 Pipeline Executor (v2, future)

Overlap model initialization with previous trial's cleanup:

```
Trial 1: [=== train ===][cleanup]
Trial 2:         [init  ][=== train ===][cleanup]
Trial 3:                          [init ][=== train ===]
```

Saves ~30s per trial transition (model loading + LoRA setup).

### 7.3 Multi-GPU Executor (v3, future)

For multi-GPU systems, run N trials in parallel (one per GPU):

```
GPU 0: Trial 1 → Trial 5 → Trial 9
GPU 1: Trial 2 → Trial 6 → Trial 10
GPU 2: Trial 3 → Trial 7
GPU 3: Trial 4 → Trial 8
```

---

## 8. Tuning Modes

### 8.1 Scout Mode

Fast exploration: 1 epoch per trial. Answers "which region of the search space is promising?"

| Property | Value |
|----------|-------|
| Epochs per trial | 1 |
| Scheduler | None (all trials run to completion) |
| Time per trial | ~2.5h (SSC, 29K samples, RTX 4090) |
| Typical budget | 10-20 trials |
| Total time | 25-50 hours |
| Use case | Narrow the search space before full training |

### 8.2 Full Mode

Deep exploration: up to `max_epochs` per trial with early stopping + ASHA scheduling.

| Property | Value |
|----------|-------|
| Epochs per trial | up to 20 (with early stopping patience=5) |
| Scheduler | ASHA (default) or Median |
| Time per trial | 5-50h depending on pruning |
| Typical budget | 5-10 trials |
| Total time | 25-100 hours |
| Use case | Find the best possible configuration |

### 8.3 Two-Phase Mode (Recommended)

Combine scout + full for optimal efficiency (inspired by SageMaker's approach):

```bash
# Phase 1: Scout — 10 trials x 1 epoch (narrow the space)
apr tune --task classify --data corpus.jsonl --budget 10 --scout -o phase1/

# Phase 2: Full — top 3 configs x 20 epochs (deep training)
apr tune --task classify --data corpus.jsonl --budget 3 --from-scout phase1/ -o phase2/
```

Phase 2 loads the leaderboard from Phase 1 and warm-starts TPE with those results,
then runs full training only on the most promising configurations.

---

## 9. Class-Weighted Loss

### 9.1 Motivation

The SSC dataset is imbalanced:

| Class | Label | Count | % | Unweighted | Inverse Freq | Sqrt Inverse |
|-------|-------|-------|---|-----------|-------------|-------------|
| 0 | safe | 17,252 | 58.9% | 1.00 | 0.34 | 0.58 |
| 1 | needs-quoting | 2,402 | 8.2% | 1.00 | 2.44 | 1.56 |
| 2 | non-deterministic | 2,858 | 9.7% | 1.00 | 2.05 | 1.43 |
| 3 | non-idempotent | 2,875 | 9.8% | 1.00 | 2.04 | 1.43 |
| 4 | unsafe | 3,920 | 13.4% | 1.00 | 1.50 | 1.22 |

Without class weights, the model optimizes for class 0 accuracy (58.9% of the loss).

### 9.2 Weight Strategies

| Strategy | Formula | When to Use |
|----------|---------|-------------|
| `uniform` | w_c = 1.0 | Balanced datasets |
| `inverse_freq` | w_c = N / (K * n_c) | Strongly imbalanced (penalizes majority heavily) |
| `sqrt_inverse` | w_c = sqrt(N / (K * n_c)) | Moderate imbalance (gentler correction) |
| `effective_num` | w_c = (1 - beta) / (1 - beta^n_c) | Long-tailed distributions (beta=0.9999) |

Weights are normalized to sum to `num_classes` to preserve the loss scale.

### 9.3 Weighted Cross-Entropy

```
loss = -w[label] * log(softmax(logits)[label])

gradient[i] = w[label] * (softmax(logits)[i] - one_hot(label)[i])
```

The weight multiplier applies to both the loss scalar and the gradient, preserving
the correct gradient direction while scaling the contribution of each class.

---

## 10. Leaderboard and Reporting

### 10.1 Trial History (JSON)

Every tuning run saves a full history to `tune_history.json`:

```json
{
  "experiment_id": "tune-classify-1772300000",
  "strategy": "tpe",
  "mode": "scout",
  "budget": 10,
  "search_space": { ... },
  "trials": [
    {
      "id": 0,
      "status": "completed",
      "config": {
        "learning_rate": 3.2e-5,
        "lora_rank": 32,
        "lora_alpha_ratio": 1.0,
        "batch_size": 64,
        "warmup_fraction": 0.08,
        "gradient_clip_norm": 1.5,
        "class_weights": "sqrt_inverse",
        "target_modules": "qv"
      },
      "metrics": {
        "val_loss": 0.682,
        "val_accuracy": 0.863,
        "train_loss": 0.412,
        "train_accuracy": 0.921,
        "epochs_run": 1,
        "time_ms": 9060000,
        "samples_per_sec": 3.2
      }
    }
  ],
  "best_trial_id": 0,
  "total_time_ms": 90600000,
  "timestamp": "2026-02-28T12:00:00Z"
}
```

### 10.2 Terminal Output

Rich formatted output using apr-cli's `output.rs` primitives:

```
=== apr tune — Shell Safety Classifier ===

  Strategy: TPE (Bayesian)
  Mode: Scout (1 epoch/trial)
  Budget: 10 trials
  Dataset: 29,307 samples (5 classes)
  Device: NVIDIA RTX 4090

◉ Trial 1/10  lr=1.2e-4 rank=16 wt=uniform
  ✓ val_loss=1.31  val_acc=73.8%  (2h 31m)

◉ Trial 2/10  lr=3.2e-5 rank=32 wt=sqrt_inverse
  ✓ val_loss=0.68  val_acc=86.3%  (2h 28m)  ★ NEW BEST

...

╭─ Leaderboard ────────────────────────────────────────╮
│ #1  val_loss=0.682  acc=86.3%  lr=3.2e-5  r=32  √inv│
│ #2  val_loss=0.741  acc=84.7%  lr=5.1e-5  r=16  inv │
│ #3  val_loss=0.823  acc=82.1%  lr=1.8e-5  r=32  inv │
╰──────────────────────────────────────────────────────╯

Best config saved to: tune-results/best/config.json
Best model saved to: tune-results/best/model.safetensors
```

### 10.3 Parameter Importance (v2)

After all trials complete, compute parameter importance via fANOVA-style analysis:

```
Parameter Importance:
  class_weights    ████████████████████  0.35
  learning_rate    ████████████████      0.28
  lora_rank        ██████████            0.18
  batch_size       █████                 0.09
  warmup_fraction  ███                   0.05
  gradient_clip    ██                    0.03
  target_modules   █                     0.02
```

This tells users which parameters matter most, guiding further manual refinement.

---

## 11. CLI Specification

### 11.1 Command Syntax

```bash
apr tune [MODEL_PATH] [OPTIONS]
```

### 11.2 Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `MODEL_PATH` | path | — | Path to pretrained model (SafeTensors directory) |
| `--task` | string | — | Task type: `classify` |
| `--data` | path | — | Training data JSONL |
| `--budget` | int | 10 | Maximum number of trials |
| `--strategy` | string | `tpe` | Search strategy: `tpe`, `grid`, `random` |
| `--scheduler` | string | `asha` | Trial scheduler: `asha`, `median`, `none` |
| `--scout` | flag | false | Scout mode (1 epoch per trial) |
| `--max-epochs` | int | 20 | Max epochs per trial (full mode) |
| `--num-classes` | int | 5 | Number of output classes |
| `--model-size` | string | — | Model architecture: `0.5B`, `1.5B`, `7B` |
| `--from-scout` | path | — | Warm-start from scout phase results |
| `--seed` | int | 42 | Random seed for reproducibility |
| `-o, --output` | path | `tune-results/` | Output directory |
| `--json` | flag | false | JSON output (for CI/CD integration) |
| `--time-limit` | duration | — | Max wall-clock time (e.g., `8h`, `30m`) |
| `--vram` | float | auto | Available VRAM in GB |

### 11.3 Examples

```bash
# Minimal: auto-configure everything
apr tune --task classify --data corpus.jsonl

# Scout mode with 10 trials
apr tune --task classify --model-size 0.5B \
    /path/to/qwen2.5-coder-0.5b \
    --data corpus.jsonl \
    --budget 10 --scout \
    -o scout-results/

# Full mode with ASHA scheduler, time-limited
apr tune --task classify --model-size 0.5B \
    /path/to/qwen2.5-coder-0.5b \
    --data corpus.jsonl \
    --budget 5 --max-epochs 20 --scheduler asha \
    --time-limit 48h \
    -o full-results/

# Two-phase: scout then full
apr tune --task classify --data corpus.jsonl --budget 10 --scout -o phase1/
apr tune --task classify --data corpus.jsonl --budget 3 --from-scout phase1/ -o phase2/

# Grid search over specific values
apr tune --task classify --data corpus.jsonl \
    --strategy grid \
    --budget 36 \
    -o grid-results/

# CI/CD: JSON output with time limit
apr tune --task classify --data corpus.jsonl \
    --budget 5 --scout --time-limit 4h --json
```

---

## 12. Implementation Plan

### 12.1 Workstream 1: Class-Weighted Loss (entrenar)

**Files**:
- `entrenar/src/finetune/classify_pipeline.rs` — add `class_weights: Option<Vec<f32>>` to `ClassifyConfig`, apply in `train_step()`, `forward_backward_single()`, `forward_only()`
- `entrenar/src/finetune/classification.rs` — add `compute_class_weights(stats, strategy) -> Vec<f32>`

**Tests**: 4 unit tests (uniform, inverse_freq, sqrt_inverse, weighted loss correctness)

### 12.2 Workstream 2: Searcher/Scheduler Traits (entrenar)

**New file**: `entrenar/src/finetune/classify_tuner.rs`

```rust
pub trait TuneSearcher {
    fn suggest(&mut self) -> Result<Trial>;
    fn record(&mut self, trial: &Trial, score: f64, epochs: usize);
    fn best(&self) -> Option<&Trial>;
}

pub trait TuneScheduler {
    fn should_stop(&self, trial_id: usize, epoch: usize, val_loss: f64) -> bool;
}
```

Implement for existing `TPEOptimizer`, `GridSearch`, and `HyperbandScheduler`.

### 12.3 Workstream 3: ClassifyTuner (entrenar)

**Same file**: `entrenar/src/finetune/classify_tuner.rs`

Core struct with:
- `TuneConfig` — budget, strategy, mode, scheduler, seed, output_dir
- `ClassifyTuner::new()` — from model config + corpus + tune config
- `ClassifyTuner::run()` — main loop: suggest → build pipeline → train → record → leaderboard
- `trial_to_configs()` — convert ParameterValue map to ClassifyConfig + TrainingConfig
- `TuneResult` + `TrialSummary` — result types
- Leaderboard persistence to JSON

**Tests**: 5 unit tests + 1 integration test with tiny model

### 12.4 Workstream 4: CLI Integration (apr-cli)

**Files**:
- `aprender/crates/apr-cli/src/extended_commands.rs` — extend Tune variant with new args
- `aprender/crates/apr-cli/src/commands/tune.rs` — add `run_classify_tune()`
- `aprender/crates/apr-cli/src/dispatch_analysis.rs` — route `--task classify`

**Tests**: 3 CLI tests

### 12.5 Dependency Order

```
WS1 (class weights) → WS2 (traits) → WS3 (ClassifyTuner) → WS4 (CLI)
```

---

## 13. Contracts

### 13.1 Tuning Invariants

| ID | Invariant | Verification |
|----|-----------|-------------|
| F-TUNE-001 | Val split is identical across all trials | Hash of val indices matches |
| F-TUNE-002 | Best trial has lowest val_loss among all completed trials | Leaderboard sorted correctly |
| F-TUNE-003 | TPE startup trials are uniformly random | Chi-squared test on first N configs |
| F-TUNE-004 | ASHA pruned trials ran fewer epochs than non-pruned | `pruned.epochs < max_epochs` |
| F-TUNE-005 | Class weights sum to num_classes (± epsilon) | `abs(sum(w) - K) < 1e-5` |
| F-TUNE-006 | Trial configs are within search space bounds | `space.validate(config)` |
| F-TUNE-007 | Seeded runs are reproducible | Same seed → same trial sequence |
| F-TUNE-008 | Time limit is respected (± 1 trial) | `total_time <= time_limit + max_trial_time` |
| F-TUNE-009 | Best model checkpoint is loadable | `ClassifyPipeline::from_checkpoint()` succeeds |
| F-TUNE-010 | JSON output is valid and parseable | `serde_json::from_str()` succeeds |

### 13.2 Falsification Tests

| ID | Test | Expected |
|----|------|----------|
| FALSIFY-TUNE-001 | Budget=0 → error, not empty run | `Err(InvalidBudget)` |
| FALSIFY-TUNE-002 | Unknown strategy → error | `Err(UnknownStrategy)` |
| FALSIFY-TUNE-003 | Data file not found → error | `Err(DataNotFound)` |
| FALSIFY-TUNE-004 | num_classes=0 → error | `Err(InvalidClasses)` |
| FALSIFY-TUNE-005 | Negative learning rate in search space → error | `Err(InvalidDomain)` |

---

## 14. Verification Matrix

| Verification | Command | Expected |
|-------------|---------|----------|
| Class weights compile | `cargo check -p entrenar` | Clean |
| Weight computation tests | `cargo test -p entrenar -- class_weights` | 4/4 pass |
| Tuner traits compile | `cargo check -p entrenar` | Clean |
| ClassifyTuner unit tests | `cargo test -p entrenar -- classify_tuner` | 5/5 pass |
| Integration test (tiny) | `cargo test -p entrenar -- tune_integration` | 1/1 pass |
| CLI tests | `cargo test -p apr-cli -- tune` | 3/3 pass |
| Scout run (3 trials, tiny) | `apr tune --task classify --budget 3 --scout --data test.jsonl` | Leaderboard printed |
| JSON output valid | `apr tune ... --json \| python3 -m json.tool` | Valid JSON |
| Clippy clean | `cargo clippy -p entrenar -p apr-cli -- -D warnings` | 0 warnings |
| Falsification tests | `cargo test -p entrenar -- falsify_tune` | 5/5 pass |

---

## 15. Future Work

### 15.1 Population-Based Training (v2)

PBT (Jaderberg et al., 2017) mutates hyperparameters *during* training, not just between
trials. This is particularly valuable for learning rate schedules — instead of searching
for a fixed warmup/cosine config, PBT discovers the optimal LR trajectory.

### 15.2 Neural Architecture Search for Classification Head (v2)

Search over classifier head architectures:
- Linear(896 → 5)
- MLP(896 → 256 → 5)
- MLP(896 → 256 → 64 → 5) with dropout

### 15.3 Multi-Objective Optimization (v3)

Pareto-optimal search across multiple objectives:
- Minimize val_loss
- Minimize trainable parameters (smaller adapter)
- Minimize inference latency

### 15.4 Transfer Learning from Prior Tuning Runs (v3)

Warm-start TPE with results from previous tuning experiments on similar datasets,
transferring knowledge about which HP regions are generally good for LoRA classification.

---

## Appendix A: Comparison with Industry Systems

| Feature | apr tune | Optuna | Ray Tune | H2O AutoML | AutoGluon | SageMaker |
|---------|---------|--------|----------|------------|-----------|-----------|
| Language | Rust | Python | Python | Java/Python | Python | Python |
| TPE | Yes | Yes | Via Optuna | No | No | Yes |
| ASHA | Yes | Yes (pruner) | Yes | No | Yes | Yes |
| Grid | Yes | Yes | Yes | Yes | Yes | Yes |
| PBT | Planned | No | Yes | No | No | No |
| Class weights | Yes | Manual | Manual | Auto | Auto | Auto |
| Leaderboard | Yes | Dashboard | TensorBoard | Yes | Yes | Yes |
| Time budgets | Yes | Yes | Yes | Yes | Yes | Yes |
| Multi-GPU | Planned | Yes | Yes | Yes | Yes | Yes |
| Ensembling | Planned | Manual | Manual | Yes (stacking) | Yes | Yes |
| No Python | **Yes** | No | No | No | No | No |
| LoRA-aware | **Yes** | No | No | No | No | No |

Key differentiators of `apr tune`:
1. **Zero Python dependency** — runs entirely in the sovereign Rust stack
2. **LoRA-aware search space** — rank/alpha coupling, target module selection, batch size caps
3. **LoRA-specific research integration** — search ranges informed by Raschka (2025) and Unsloth
4. **Dual-format checkpoints** — APR + SafeTensors per trial
5. **Shell safety domain** — preconfigured search space for classification fine-tuning
