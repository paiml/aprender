# Eval Component Specification

**Parent**: [apr-spec.md](../apr-spec.md) §12
**Status**: Active
**CLI**: `apr eval`, `apr bench`
**Implementation**: `crates/apr-cli/src/commands/eval.rs`, `bench.rs`

---

## 1. Overview

Model evaluation and benchmarking. Supports perplexity, classification metrics,
pass@k code evaluation, and throughput benchmarking.

## 2. CLI Interface

```
apr eval <FILE> \
  [--dataset <wikitext-2|lambada|custom>] \
  [--task <classify>] \
  [--data <JSONL>] \
  [--benchmark <hellaswag|mmlu|...>] \
  [--drift --baseline <FILE>] \
  [--samples <N>] \
  [--temperature <F>]
```

## 3. Evaluation Modes

| Mode | Metric | Spec Target |
|------|--------|-------------|
| Perplexity | PPL | ≤ 20 |
| Classification | Accuracy, F1 | Task-dependent |
| Pass@k | Code correctness | k=1,10,100 |
| Throughput | tok/s | ≥ 10 |

## 4. Planned Features

- **lm-eval harness integration** (GH-454): HellaSwag, MMLU, TruthfulQA, etc.
- **Model drift detection** (GH-454): Compare outputs over time

---

## Provable Contracts

### Contract: `eval-v1.yaml`

```yaml
metadata:
  description: "Model evaluation — perplexity, classification, pass@k, benchmarks"
  depends_on:
    - "cross-entropy-kernel-v1"
    - "softmax-kernel-v1"

equations:
  perplexity:
    formula: "PPL = exp(-1/N * Σ log P(x_i | x_{<i}))"
    invariants:
      - "PPL >= 1 (perfect model has PPL=1)"
      - "PPL is deterministic for temperature=0"
      - "N > 0 (at least one token)"

  pass_at_k:
    formula: "pass@k = 1 - C(n-c, k) / C(n, k)"
    domain: "n = total samples, c = correct samples, k = selection size"
    invariants:
      - "0 <= pass@k <= 1"
      - "pass@k is monotonically non-decreasing in k"
      - "c > 0 → pass@k > 0 for sufficiently large k"

  f1_score:
    formula: "F1 = 2 * precision * recall / (precision + recall)"
    invariants:
      - "0 <= F1 <= 1"
      - "F1 = 0 when precision = 0 or recall = 0"
      - "F1 = 1 when precision = 1 and recall = 1"

  throughput:
    formula: "tok_per_sec = generated_tokens / wall_clock_seconds"
    invariants:
      - "tok_per_sec > 0"
      - "Excludes warmup iterations"

proof_obligations:
  - type: bound
    property: "Perplexity lower bound"
    formal: "PPL >= 1.0"
  - type: bound
    property: "Pass@k bounds"
    formal: "0 <= pass_at_k <= 1"
  - type: monotonicity
    property: "Pass@k monotonic in k"
    formal: "pass@(k+1) >= pass@k"
  - type: bound
    property: "F1 bounds"
    formal: "0 <= F1 <= 1"
  - type: invariant
    property: "Deterministic eval"
    formal: "eval(model, data, temp=0) is deterministic"

falsification_tests:
  - id: FALSIFY-EVAL-001
    rule: "PPL lower bound"
    prediction: "PPL >= 1.0 for any model and data"
    if_fails: "Log probability or averaging computation bug"
  - id: FALSIFY-EVAL-002
    rule: "Pass@k monotonic"
    prediction: "pass@10 >= pass@1 for same model and data"
    if_fails: "Combinatorial formula incorrect"
  - id: FALSIFY-EVAL-003
    rule: "F1 bounds"
    prediction: "F1 in [0, 1] for any predictions and labels"
    if_fails: "Division by zero or precision/recall computation bug"
  - id: FALSIFY-EVAL-004
    rule: "Empty dataset"
    prediction: "eval on empty dataset returns error, not NaN/panic"
    if_fails: "Division by zero on empty input"
  - id: FALSIFY-EVAL-005
    rule: "Determinism"
    prediction: "same model + same data + temp=0 → same PPL"
    if_fails: "Non-deterministic path in inference"

kani_harnesses:
  - id: KANI-EVAL-001
    obligation: "Pass@k bounds"
    property: "pass_at_k in [0,1] for all valid n,c,k"
    bound: 10
    strategy: exhaustive
```

### Binding Requirements

```yaml
  - contract: eval-v1.yaml
    equation: perplexity
    module_path: "aprender::eval"
    function: compute_perplexity
    status: implemented

  - contract: eval-v1.yaml
    equation: pass_at_k
    module_path: "aprender::eval"
    function: pass_at_k
    status: implemented

  - contract: eval-v1.yaml
    equation: f1_score
    module_path: "aprender::metrics"
    function: f1_score
    status: implemented
```
