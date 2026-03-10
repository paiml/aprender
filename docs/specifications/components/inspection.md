# Inspect / Debug / Validate Component Specification

**Parent**: [apr-spec.md](../apr-spec.md) §16
**Status**: Active
**CLI**: `apr inspect`, `apr debug`, `apr validate`, `apr diff`, `apr tensors`,
`apr trace`, `apr lint`, `apr explain`, `apr hex`, `apr tree`
**Implementation**: `crates/apr-cli/src/commands/inspect.rs`, `validate.rs`, `diff.rs`, etc.

---

## 1. Overview

Model introspection tools for debugging, validation, comparison, and
architecture visualization.

## 2. Commands

| Command | Purpose |
|---------|---------|
| `inspect` | Metadata, vocab, structure, weight stats |
| `debug` | Drama mode, hex dump, ASCII extraction |
| `validate` | Integrity check, 100-point quality score |
| `diff` | Two-model comparison (metadata, weights, values) |
| `tensors` | List tensor names, shapes, statistics |
| `trace` | Layer-by-layer analysis with reference comparison |
| `lint` | Best practices checking |
| `explain` | Error codes, tensors, kernel dispatch |
| `hex` | Format-aware binary forensics |
| `tree` | Architecture tree visualization |

## 3. Validate Quality Score

100-point assessment across:
- Format integrity (header, checksums)
- Tensor completeness (all expected tensors present)
- Value sanity (no NaN/Inf, reasonable ranges)
- Architecture consistency (shapes match config)
- Metadata completeness

## 4. Diff Modes

- **Metadata diff**: Architecture, hyperparameters
- **Weight diff**: Tensor name/shape comparison
- **Value diff**: Statistical comparison of tensor values
- **Transpose-aware**: Account for GGUF col-major vs APR row-major

---

## Provable Contracts

### Contract: `inspection-v1.yaml`

```yaml
metadata:
  description: "Model inspection — validate, diff, lint, explain"
  depends_on:
    - "apr-format-v2"
    - "tensor-shape-flow-v1"

equations:
  quality_score:
    formula: "score = Σ gate_i * weight_i, score ∈ [0, 100]"
    invariants:
      - "0 <= score <= 100"
      - "Σ weight_i = 100"
      - "Each gate_i ∈ {0, 1} (pass/fail)"

  diff_symmetry:
    formula: "diff(A, B) reports same divergences as diff(B, A)"
    invariants:
      - "Tensor count difference is symmetric"
      - "Value deltas are identical (not direction-dependent)"

  nan_detection:
    formula: "∀ tensor T: count(isnan(T)) reported"
    invariants:
      - "NaN detected regardless of tensor dtype"
      - "Inf also reported as anomaly"
      - "Zero tensors flagged as warning"

proof_obligations:
  - type: bound
    property: "Quality score bounds"
    formal: "0 <= quality_score <= 100"
  - type: symmetry
    property: "Diff symmetry"
    formal: "diff(A,B).divergences == diff(B,A).divergences"
  - type: completeness
    property: "NaN detection completeness"
    formal: "∀ tensor with NaN: flagged in validation report"
  - type: invariant
    property: "Lint idempotency"
    formal: "lint(model) produces same warnings on repeated runs"

falsification_tests:
  - id: FALSIFY-INSP-001
    rule: "Score bounds"
    prediction: "validate score always in [0, 100]"
    if_fails: "Weight sum != 100 or gate returns non-binary"
  - id: FALSIFY-INSP-002
    rule: "NaN detection"
    prediction: "model with NaN tensor fails validation"
    if_fails: "NaN check missing for some dtype"
  - id: FALSIFY-INSP-003
    rule: "Diff symmetry"
    prediction: "diff(A,B) and diff(B,A) report same tensor mismatches"
    if_fails: "Diff only checks in one direction"
  - id: FALSIFY-INSP-004
    rule: "Explain coverage"
    prediction: "every error code has an explanation"
    if_fails: "Error code added without corresponding explain entry"
  - id: FALSIFY-INSP-005
    rule: "Transpose-aware diff"
    prediction: "GGUF vs APR diff with --transpose-aware shows match"
    if_fails: "Transpose not applied before comparison"

kani_harnesses:
  - id: KANI-INSP-001
    obligation: "Score bounds"
    property: "weighted sum of binary gates in [0, 100]"
    bound: 10
    strategy: exhaustive
```

### Binding Requirements

```yaml
  - contract: inspection-v1.yaml
    equation: quality_score
    module_path: "aprender::scoring"
    function: compute_quality_score
    status: implemented

  - contract: inspection-v1.yaml
    equation: nan_detection
    module_path: "aprender::validate"
    function: check_tensor_sanity
    status: implemented
```
