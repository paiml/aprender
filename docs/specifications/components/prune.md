# Prune Component Specification

**Parent**: [apr-spec.md](../apr-spec.md) §7
**Status**: Active
**CLI**: `apr prune`
**Implementation**: `crates/apr-cli/src/commands/prune.rs`
**Library**: `aprender::pruning`

---

## 1. Overview

Model pruning removes redundant weights to reduce model size, memory footprint,
and inference latency while preserving quality.

## 2. CLI Interface

```
apr prune <FILE> \
  --method <magnitude|structured|depth|width|wanda|sparsegpt> \
  --target-ratio <0-1> \
  --output <PATH> \
  [--sparsity <0-1>] \
  [--remove-layers <RANGE>] \
  [--calibration <FILE>] \
  [--analyze] \
  [--plan]
```

## 3. Methods

| Method | Description | Granularity |
|--------|-------------|-------------|
| `magnitude` | Remove smallest-magnitude weights | Unstructured |
| `structured` | Remove entire neurons/attention heads | Structured |
| `depth` | Remove entire transformer layers | Layer-level |
| `width` | Reduce hidden dimensions | Structured |
| `wanda` | Weights And Activations (calibration-based) | Unstructured |
| `sparsegpt` | Second-order pruning via Hessian approximation | Unstructured |

### 3.1 Magnitude Pruning

Zeroes out the smallest-magnitude weights globally or per-layer.

### 3.2 Structured Pruning

Removes entire neurons or attention heads based on importance scores.

### 3.3 Depth Pruning

Removes entire transformer layers: `--remove-layers 20-24`.

### 3.4 Width Pruning

Reduces hidden dimensions uniformly across layers.

### 3.5 WANDA (Weights And Activations)

Uses calibration data to compute activation-weighted importance. Prunes
weights where `|w| * ||X||` is smallest.

### 3.6 SparseGPT

Second-order pruning using approximate Hessian inverse. Produces higher
quality sparse models than magnitude pruning at the same sparsity.

## 4. Features

- **Analyze mode**: Identify pruning opportunities without modifying model
- **Plan mode**: Estimate size reduction and quality impact
- **Calibration data**: Required for WANDA and SparseGPT

---

## Provable Contracts

### Contract: `pruning-v1.yaml`

```yaml
metadata:
  description: "Model pruning — magnitude, structured, depth, WANDA, SparseGPT"
  references:
    - "Sun et al. (2023) WANDA: A Simple and Effective Pruning Approach"
    - "Frantar & Alistarh (2023) SparseGPT"

equations:
  magnitude_prune:
    formula: "mask_i = I(|w_i| > threshold(target_ratio))"
    invariants:
      - "count(mask == 0) / count(mask) ≈ target_ratio"
      - "Pruned weights are exactly zero"
      - "Unpruned weights unchanged"

  wanda_score:
    formula: "score_i = |w_i| * ||X_col_i||₂"
    invariants:
      - "score >= 0 for all weights"
      - "Higher score → more important weight"
      - "Requires calibration data"

  depth_prune:
    formula: "layers_out = layers_in \\ removed_set"
    invariants:
      - "num_layers_out = num_layers_in - |removed_set|"
      - "Layer indices in removed_set are valid"
      - "Remaining layers re-indexed contiguously"

  sparsity_ratio:
    formula: "sparsity = count(w == 0) / count(w)"
    invariants:
      - "0 <= sparsity <= 1"
      - "sparsity ≈ target_ratio after pruning"

proof_obligations:
  - type: invariant
    property: "Unpruned weight preservation"
    formal: "∀ w where mask==1: w_pruned == w_original"
  - type: bound
    property: "Sparsity ratio accuracy"
    formal: "|actual_sparsity - target_ratio| < 0.01"
  - type: invariant
    property: "Depth prune layer count"
    formal: "output.num_layers == input.num_layers - removed_count"
  - type: invariant
    property: "WANDA score non-negativity"
    formal: "wanda_score(w, X) >= 0 for all w, X"
  - type: invariant
    property: "Shape preservation (unstructured)"
    formal: "pruned.shape == original.shape for magnitude/wanda/sparsegpt"

falsification_tests:
  - id: FALSIFY-PRUNE-001
    rule: "Weight preservation"
    prediction: "unpruned weights bitwise identical to original"
    if_fails: "Pruning modifies non-target weights"
  - id: FALSIFY-PRUNE-002
    rule: "Sparsity accuracy"
    prediction: "actual sparsity within 1% of target"
    if_fails: "Threshold computation incorrect"
  - id: FALSIFY-PRUNE-003
    rule: "Depth prune count"
    prediction: "output has exactly N - removed layers"
    if_fails: "Layer removal or re-indexing bug"
  - id: FALSIFY-PRUNE-004
    rule: "Zero enforcement"
    prediction: "all pruned weights are exactly 0.0, not near-zero"
    if_fails: "Mask application uses multiplication instead of zeroing"
  - id: FALSIFY-PRUNE-005
    rule: "Invalid layer rejection"
    prediction: "remove-layers outside range returns error"
    if_fails: "Missing bounds check on layer indices"

kani_harnesses:
  - id: KANI-PRUNE-001
    obligation: "Sparsity ratio"
    property: "sparsity within tolerance of target for small tensors"
    bound: 16
    strategy: exhaustive
```

### Binding Requirements

```yaml
  - contract: pruning-v1.yaml
    equation: magnitude_prune
    module_path: "aprender::pruning"
    function: prune_magnitude
    status: implemented

  - contract: pruning-v1.yaml
    equation: wanda_score
    module_path: "aprender::pruning"
    function: prune_wanda
    status: implemented

  - contract: pruning-v1.yaml
    equation: depth_prune
    module_path: "aprender::pruning"
    function: prune_depth
    status: implemented
```
