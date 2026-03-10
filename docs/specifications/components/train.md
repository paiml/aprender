# Train Component Specification

**Parent**: [apr-spec.md](../apr-spec.md) §9
**Status**: Active
**CLI**: `apr train`
**Implementation**: `crates/apr-cli/src/commands/train.rs`
**Library**: `entrenar`

---

## 1. Overview

Full training pipeline with plan/apply pattern. Supports classification
fine-tuning, continual pre-training, HPO with multiple search strategies,
distributed training, and crash recovery.

## 2. Subcommands

| Subcommand | Description |
|------------|-------------|
| `plan` | Validate data, check compatibility, estimate resources |
| `apply` | Execute plan: allocate GPU, run trials |
| `watch` | Monitor with auto-restart on crash/hang |
| `sweep` | Generate hyperparameter sweep configs |
| `archive` | Package checkpoint into release bundle |
| `submit` | Submit multi-adapter jobs to cluster |
| `cluster-status` | Show cluster health and capacity |

## 3. Plan/Apply Pattern

```bash
# Phase 1: Plan (no GPU)
apr train plan --data train.jsonl --model-size 0.5B --strategy tpe --budget 20

# Phase 2: Apply (commits GPU)
apr train apply --plan plan.yaml --data train.jsonl
```

## 4. HPO Strategies

| Strategy | Description |
|----------|-------------|
| `tpe` | Tree-structured Parzen Estimator (Bayesian) |
| `grid` | Exhaustive grid search |
| `random` | Random sampling |
| `manual` | User-specified hyperparameters |

## 5. Distributed Training

```bash
apr train apply --distributed --world-size 4 --coordinator-addr 0.0.0.0:9000
```

## 6. Crash Recovery (Watch)

```bash
apr train watch --config train.yaml --max-restarts 5 --heartbeat-timeout 300
```

Detects: SIGABRT, SIGSEGV, OOM, heartbeat staleness, CUDA async errors.

## 7. Sweep Generation

```bash
apr train sweep --config base.yaml --strategy random --num-configs 10
```

---

## Provable Contracts

### Contract: `training-pipeline-v1.yaml`

```yaml
metadata:
  description: "Training pipeline — plan/apply, HPO, distributed, crash recovery"
  depends_on:
    - "adamw-kernel-v1"
    - "cross-entropy-kernel-v1"
    - "lora-algebra-v1"

equations:
  plan_resource_estimate:
    formula: "vram_required = model_size + optimizer_state + activations + gradients"
    invariants:
      - "vram_required > 0"
      - "vram_required <= available_vram (validated before apply)"
      - "Estimate within 20% of actual usage"

  tpe_acquisition:
    formula: "EI(x) = ∫ max(f* - f, 0) * p(f|x) df"
    invariants:
      - "EI(x) >= 0 for all x"
      - "Next sample maximizes EI"

  checkpoint_determinism:
    formula: "load(save(state)) == state"
    invariants:
      - "Model weights roundtrip exactly"
      - "Optimizer state roundtrip exactly"
      - "Epoch counter preserved"

  heartbeat_staleness:
    formula: "stale = now() - last_heartbeat > timeout"
    invariants:
      - "Stale detection triggers restart"
      - "Restart count <= max_restarts"
      - "Backoff delay doubles per restart (capped)"

proof_obligations:
  - type: invariant
    property: "Checkpoint roundtrip"
    formal: "load(save(state)).weights == state.weights"
  - type: bound
    property: "VRAM estimate accuracy"
    formal: "|estimated - actual| / actual < 0.2"
  - type: invariant
    property: "Restart limit"
    formal: "restart_count <= max_restarts"
  - type: bound
    property: "EI non-negativity"
    formal: "expected_improvement(x) >= 0"
  - type: invariant
    property: "Backoff cap"
    formal: "backoff_delay <= backoff_max"

falsification_tests:
  - id: FALSIFY-TRAIN-001
    rule: "Checkpoint roundtrip"
    prediction: "save → load produces identical model weights"
    if_fails: "Serialization loses precision or drops tensors"
  - id: FALSIFY-TRAIN-002
    rule: "Restart limit"
    prediction: "watch gives up after max_restarts crashes"
    if_fails: "Counter not incremented or compared"
  - id: FALSIFY-TRAIN-003
    rule: "Backoff cap"
    prediction: "delay never exceeds backoff_max seconds"
    if_fails: "Exponential backoff uncapped"
  - id: FALSIFY-TRAIN-004
    rule: "Plan rejects OOM"
    prediction: "plan fails if vram_required > vram_available"
    if_fails: "VRAM validation missing in plan phase"
  - id: FALSIFY-TRAIN-005
    rule: "Sweep determinism"
    prediction: "same seed → same sweep configs"
    if_fails: "RNG not seeded correctly"

kani_harnesses:
  - id: KANI-TRAIN-001
    obligation: "Restart limit"
    property: "restart_count <= max_restarts for bounded execution"
    bound: 10
    strategy: bounded_int
```

### Binding Requirements

```yaml
  - contract: training-pipeline-v1.yaml
    equation: checkpoint_determinism
    module_path: "entrenar::checkpoint"
    function: "save_checkpoint / load_checkpoint"
    status: implemented

  - contract: training-pipeline-v1.yaml
    equation: heartbeat_staleness
    module_path: "entrenar::watch"
    function: detect_stale
    status: implemented
```
