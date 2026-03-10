# QA Component Specification

**Parent**: [apr-spec.md](../apr-spec.md) §18
**Status**: Active
**CLI**: `apr qa`, `apr parity`, `apr qualify`
**Implementation**: `crates/apr-cli/src/commands/qa.rs`

---

## 1. Overview

Falsifiable quality assurance pipeline for model releases. 8+ gate checklist
with regression detection, GPU/CPU parity, cross-format parity, and CI
integration.

## 2. CLI Interface

```
apr qa <FILE> \
  [--assert-tps <N>] \
  [--assert-speedup <N>] \
  [--previous-report <FILE>] \
  [--regression-threshold <RATIO>] \
  [--skip-golden] [--skip-throughput] [--skip-ollama] \
  [--json] [--min-executed <N>]
```

## 3. QA Gates

| Gate | Description | Skip Flag |
|------|-------------|-----------|
| Golden output | Known-good response comparison | `--skip-golden` |
| Throughput | tok/s benchmark (≥10 spec) | `--skip-throughput` |
| Ollama parity | Match Ollama output | `--skip-ollama` |
| GPU speedup | GPU faster than CPU | `--skip-gpu-speedup` |
| Tensor contracts | Weight shape/value validation | `--skip-contract` |
| Format parity | APR vs SafeTensors comparison | `--skip-format-parity` |
| PTX parity | GPU kernel validation | `--skip-ptx-parity` |
| Metadata | Plausibility validation | `--skip-metadata` |

## 4. Regression Detection

```bash
apr qa model.apr --previous-report v1-qa.json --regression-threshold 0.10
```

Flags any metric that regressed >10% from baseline.

## 5. Qualify (Cross-Subcommand Smoke Test)

```bash
apr qualify model.apr --tier standard --timeout 120
```

Runs every tool against the model to verify no panics or errors.

---

## Provable Contracts

### Contract: `qa-pipeline-v1.yaml`

```yaml
metadata:
  description: "QA pipeline — falsifiable gates, regression detection"
  depends_on:
    - "eval-v1"
    - "inference-v1"

equations:
  gate_pass:
    formula: "pass(gate) = metric(gate) >= threshold(gate)"
    invariants:
      - "Each gate is independently pass/fail"
      - "Overall pass requires all non-skipped gates pass"
      - "Gate results are deterministic for temp=0"

  regression_detection:
    formula: "regressed(metric) = (baseline - current) / baseline > threshold"
    invariants:
      - "Only non-skipped metrics checked"
      - "threshold > 0 (positive regression margin)"
      - "Missing baseline → skip regression check"

  qualify_coverage:
    formula: "∀ subcommand s: run(s, model) completes without panic"
    invariants:
      - "Timeout enforced per gate"
      - "Exit code 0 → pass, non-zero → fail"
      - "Output captured for debugging"

proof_obligations:
  - type: invariant
    property: "Gate independence"
    formal: "pass(gate_i) does not depend on execution of gate_j"
  - type: invariant
    property: "Regression threshold positivity"
    formal: "regression_threshold > 0"
  - type: completeness
    property: "Qualify gate coverage"
    formal: "qualify runs ≥ min_executed gates"
  - type: invariant
    property: "Timeout enforcement"
    formal: "no gate runs longer than timeout seconds"

falsification_tests:
  - id: FALSIFY-QA-001
    rule: "Gate independence"
    prediction: "skipping gate A does not change result of gate B"
    if_fails: "Gates share mutable state"
  - id: FALSIFY-QA-002
    rule: "Regression detection"
    prediction: "10% throughput drop flagged with threshold=0.10"
    if_fails: "Regression comparison inverted or threshold ignored"
  - id: FALSIFY-QA-003
    rule: "Timeout enforcement"
    prediction: "infinite-loop model killed after timeout"
    if_fails: "Timeout not applied to gate subprocess"
  - id: FALSIFY-QA-004
    rule: "JSON output completeness"
    prediction: "JSON report contains all gate results"
    if_fails: "Some gates not serialized to JSON"
  - id: FALSIFY-QA-005
    rule: "Exit code"
    prediction: "any gate failure → exit code 1"
    if_fails: "Exit code always 0 regardless of failures"

kani_harnesses:
  - id: KANI-QA-001
    obligation: "Regression threshold"
    property: "regression detected iff delta > threshold"
    bound: 8
    strategy: bounded_int
```

### Binding Requirements

```yaml
  - contract: qa-pipeline-v1.yaml
    equation: gate_pass
    module_path: "aprender::qa"
    function: run_qa_gate
    status: implemented

  - contract: qa-pipeline-v1.yaml
    equation: regression_detection
    module_path: "aprender::qa"
    function: check_regression
    status: implemented
```
