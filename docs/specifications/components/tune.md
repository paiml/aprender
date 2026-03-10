# Tune (HPO) Component Specification

**Parent**: [apr-spec.md](../apr-spec.md) §10
**Status**: Active
**CLI**: `apr tune`
**Implementation**: `crates/apr-cli/src/commands/tune.rs`
**Library**: `entrenar::hpo`

---

## 1. Overview

Hyperparameter optimization with automatic search. Supports TPE, grid, and
random strategies with ASHA early stopping. Two-phase scout → full workflow
for efficient exploration.

## 2. CLI Interface

```
apr tune <FILE> \
  --method <auto|full|lora|qlora> \
  --strategy <tpe|grid|random> \
  --scheduler <asha|median|none> \
  --budget <N> \
  [--scout] \
  [--from-scout <DIR>] \
  [--data <JSONL>] \
  [--time-limit <DURATION>] \
  [--plan]
```

## 3. Search Strategies

| Strategy | Description |
|----------|-------------|
| `tpe` | Tree-structured Parzen Estimator (Bayesian, default) |
| `grid` | Exhaustive grid over discrete values |
| `random` | Uniform random sampling |

## 4. Schedulers

| Scheduler | Description |
|-----------|-------------|
| `asha` | Async Successive Halving (early stopping, default) |
| `median` | Stop if below median at same epoch |
| `none` | Run all trials to completion |

## 5. Two-Phase Workflow

```bash
# Phase 1: Scout (1 epoch per trial, fast)
apr tune model.apr --scout --budget 50 --data train.jsonl -o scout-results/

# Phase 2: Full (warm-start from scout, more epochs)
apr tune model.apr --from-scout scout-results/ --budget 10 --max-epochs 20
```

---

## Provable Contracts

### Contract: `hpo-v1.yaml`

```yaml
metadata:
  description: "Hyperparameter optimization — TPE, ASHA, scout/full"
  references:
    - "Bergstra et al. (2011) Algorithms for Hyper-Parameter Optimization"
    - "Li et al. (2020) ASHA: Asynchronous Successive Halving"

equations:
  tpe_split:
    formula: "l(x) = KDE(x | f(x) < f*), g(x) = KDE(x | f(x) >= f*)"
    invariants:
      - "l and g are valid probability densities"
      - "EI(x) ∝ l(x)/g(x)"
      - "f* = quantile(observed_values, gamma)"

  asha_promotion:
    formula: "promote(trial) iff perf(trial, rung_r) in top_1/η at rung r"
    invariants:
      - "Only top 1/η fraction promoted"
      - "η > 1 (halving factor)"
      - "Rungs are geometrically spaced: r, r*η, r*η², ..."

  budget_exhaustion:
    formula: "total_trials <= budget"
    invariants:
      - "Never exceeds budget"
      - "Each trial uses ≥ 1 epoch"

proof_obligations:
  - type: bound
    property: "Budget enforcement"
    formal: "total_trials <= budget"
  - type: invariant
    property: "ASHA promotion fraction"
    formal: "promoted_count <= ceil(candidates / η)"
  - type: invariant
    property: "Time limit enforcement"
    formal: "elapsed <= time_limit + grace_period"
  - type: monotonicity
    property: "Best-so-far tracking"
    formal: "best_score(t) >= best_score(t-1)"

falsification_tests:
  - id: FALSIFY-HPO-001
    rule: "Budget exhaustion"
    prediction: "total trials never exceeds budget"
    if_fails: "Trial counter or termination check broken"
  - id: FALSIFY-HPO-002
    rule: "ASHA halving"
    prediction: "at most ceil(N/η) trials promoted per rung"
    if_fails: "Promotion threshold computation incorrect"
  - id: FALSIFY-HPO-003
    rule: "Scout warm-start"
    prediction: "from-scout loads previous trial results"
    if_fails: "Result deserialization or path resolution broken"
  - id: FALSIFY-HPO-004
    rule: "Best monotonic"
    prediction: "best_score never decreases across trials"
    if_fails: "Best tracking uses wrong comparison direction"

kani_harnesses:
  - id: KANI-HPO-001
    obligation: "Budget enforcement"
    property: "trial_count <= budget for bounded iterations"
    bound: 20
    strategy: bounded_int
```

### Binding Requirements

```yaml
  - contract: hpo-v1.yaml
    equation: budget_exhaustion
    module_path: "entrenar::hpo"
    function: run_hpo
    status: implemented

  - contract: hpo-v1.yaml
    equation: asha_promotion
    module_path: "entrenar::hpo::asha"
    function: should_promote
    status: implemented
```
