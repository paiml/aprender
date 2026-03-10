# Merge Component Specification

**Parent**: [apr-spec.md](../apr-spec.md) §4
**Status**: Active
**CLI**: `apr merge`
**Implementation**: `crates/apr-cli/src/commands/merge.rs`
**Library**: `aprender::format::apr_merge`

---

## 1. Overview

Model merging combines multiple trained models into a single model by
interpolating weights in parameter space. This is the fastest way to create a
new capable model without any training compute.

**Goal**: Full parity with [MergeKit](https://github.com/arcee-ai/mergekit) plus
sovereign-native features (APR format, SIMD acceleration, offline-only).

---

## 2. CLI Interface

```
apr merge <FILE1> <FILE2> [FILE3...] \
  --strategy <STRATEGY> \
  --output <PATH> \
  [--weights <W1,W2,...>] \
  [--base-model <PATH>] \
  [--drop-rate <0.0-1.0>] \
  [--density <0.0-1.0>] \
  [--seed <N>] \
  [--config <YAML>] \
  [--plan] \
  [--parallel] \
  [--layer-config <YAML>]
```

---

## 3. Implemented Strategies

### 3.1 Average

Simple arithmetic mean of all model weights. No base model required.

```bash
apr merge model-a.apr model-b.apr --strategy average -o merged.apr
```

### 3.2 Weighted

User-specified per-model weights. Must sum to ~1.0 (warning if not).

```bash
apr merge model-a.apr model-b.apr --strategy weighted --weights 0.7,0.3 -o merged.apr
```

### 3.3 SLERP (Spherical Linear Interpolation)

Smooth interpolation on the unit hypersphere. Only 2 models. First weight is
interpolation factor `t` (default 0.5).

```bash
apr merge model-a.apr model-b.apr --strategy slerp --weights 0.3 -o merged.apr
```

### 3.4 TIES (Trim, Elect Sign, Merge)

Computes task vectors (delta from base model), trims small values, resolves
sign conflicts by majority vote, then merges. Requires `--base-model`.

```bash
apr merge model-a.apr model-b.apr --strategy ties \
  --base-model base.apr --density 0.2 -o merged.apr
```

### 3.5 DARE (Drop And REscale)

Randomly drops elements of task vectors with probability `--drop-rate`, then
rescales remaining values. Combines with TIES sign election.

```bash
apr merge model-a.apr model-b.apr --strategy dare \
  --base-model base.apr --drop-rate 0.9 --seed 42 -o merged.apr
```

---

## 4. Planned Strategies

### 4.1 Task Arithmetic

Linear combination of task vectors: `θ_merged = θ_base + Σ λ_i * (θ_i - θ_base)`.
Simplest task vector method. Foundation for TIES, DARE, DELLA, Breadcrumbs.

**CLI**:
```bash
apr merge model-a.apr model-b.apr --strategy task-arithmetic \
  --base-model base.apr --weights 0.5,0.5 -o merged.apr
```

**Complexity**: Low — subtract base, scale, add.

### 4.2 NuSLERP

Enhanced SLERP with optimized numerical stability and support for batch
processing. Handles degenerate cases (near-parallel vectors) more gracefully.

**CLI**:
```bash
apr merge model-a.apr model-b.apr --strategy nuslerp --weights 0.3 -o merged.apr
```

### 4.3 Multi-SLERP

Barycentric SLERP for >2 models. Iteratively applies SLERP along barycentric
coordinates in the weight simplex.

**CLI**:
```bash
apr merge a.apr b.apr c.apr --strategy multi-slerp --weights 0.4,0.3,0.3 -o merged.apr
```

### 4.4 DELLA (Adaptive Magnitude Pruning)

Task arithmetic with adaptive per-parameter pruning based on magnitude.
Parameters with larger task-vector magnitude are retained with higher
probability. More principled than DARE's uniform random drop.

**CLI**:
```bash
apr merge a.apr b.apr --strategy della \
  --base-model base.apr --drop-rate 0.7 -o merged.apr
```

**Reference**: Extends DARE with magnitude-aware retention.

### 4.5 Model Breadcrumbs

Task arithmetic with outlier removal: trims both very small AND very large
parameter deltas (unlike TIES which only trims small). Retains the "moderate
signal" range.

**CLI**:
```bash
apr merge a.apr b.apr --strategy breadcrumbs \
  --base-model base.apr --density 0.2 --trim-top 0.01 -o merged.apr
```

### 4.6 SCE (Variance-Based Adaptive Weighting)

Adaptive matrix-level weighting based on parameter variance across models.
High-variance parameters (where models disagree) get different treatment than
low-variance (consensus) parameters.

**CLI**:
```bash
apr merge a.apr b.apr c.apr --strategy sce -o merged.apr
```

**Complexity**: Medium — requires variance computation per parameter matrix.

### 4.7 Passthrough / Frankenmerge

Direct tensor copy for layer stacking. Combines layers from different models
into a single model (e.g., layers 0-15 from model A, layers 16-31 from model B).
No interpolation — pure surgical assembly.

**CLI**:
```bash
apr merge a.apr b.apr --strategy passthrough \
  --layer-config franken.yaml -o merged.apr
```

**Config** (franken.yaml):
```yaml
slices:
  - model: model-a.apr
    layers: [0, 16]
  - model: model-b.apr
    layers: [16, 32]
```

### 4.8 Evolutionary Merge Optimization

CMA-ES optimization of merge config parameters (strategy, weights, density,
drop_rate) against evaluation benchmarks. Requires eval harness integration.

**CLI**:
```bash
apr merge evolve \
  --config evolve.yaml \
  --eval-tasks hellaswag,winogrande \
  --budget 50 \
  --seed 42 \
  -o best-merge.apr
```

**Config** (evolve.yaml):
```yaml
models:
  - path: model-a.apr
  - path: model-b.apr
base_model: base.apr
strategies: [ties, dare, della]
search_space:
  density: [0.1, 0.5]
  drop_rate: [0.5, 0.95]
  weights: [[0.3, 0.7], [0.7, 0.3]]
```

**Complexity**: High — requires eval harness + CMA-ES optimizer.

### 4.9 MoE Construction from Dense Models

Combine N dense models into a Mixture-of-Experts architecture. Each model
becomes an expert. Adds a router layer trained on calibration data.

**CLI**:
```bash
apr merge moe \
  --experts model-a.apr model-b.apr model-c.apr model-d.apr \
  --router-data calibration.jsonl \
  --num-experts-per-token 2 \
  --output-format mixtral \
  -o moe-model.apr
```

**Complexity**: High — requires router training + architecture transformation.

### 4.10 DAM (Differentiable Adaptive Merging)

Trainable per-column merge coefficients. Uses gradient descent on a small
calibration dataset to learn optimal per-parameter merge weights.

**CLI**:
```bash
apr merge dam \
  model-a.apr model-b.apr \
  --base-model base.apr \
  --data calibration.jsonl \
  --dam-epochs 5 \
  --dam-lr 0.01 \
  -o merged.apr
```

**Complexity**: Medium-High — requires forward pass through merged model.

---

## 5. Planned Features

### 5.1 Tokenizer Surgery

Transplant tokenizers between models. Required for speculative decoding where
the draft model must share vocabulary with the target model.

**CLI**:
```bash
apr merge tokenizer-transplant \
  --source source-model.apr \
  --target target-model.apr \
  --align-embeddings \
  -o target-with-new-tokenizer.apr
```

### 5.2 Per-Layer Granularity

Specify different merge strategies or weights per layer or layer group.

**CLI**:
```bash
apr merge a.apr b.apr --strategy slerp \
  --layer-config per-layer.yaml -o merged.apr
```

**Config**:
```yaml
default:
  strategy: slerp
  weight: 0.5
layers:
  "0-8": { weight: 0.3 }     # Early layers: favor model A
  "24-31": { weight: 0.8 }   # Late layers: favor model B
  "self_attn": { weight: 0.6 }
  "mlp": { weight: 0.4 }
```

### 5.3 Multi-GPU Acceleration

Parallelize merge operations across GPUs. Near-linear speedup for large models.

```bash
apr merge a.apr b.apr --strategy ties --parallel --gpus 0,1,2,3 -o merged.apr
```

### 5.4 YAML Config Mode

Full merge specification in YAML for reproducibility and CI integration.

```bash
apr merge --config merge-config.yaml -o merged.apr
```

---

## 6. Validation

All merge operations validate:
- Input model compatibility (architecture, tensor shapes, vocab size)
- Weight count/sum constraints
- Output model integrity (100-point quality score)
- Plan mode for dry-run estimation

---

## 7. Testing Requirements

- Unit tests for each strategy with known-good reference outputs
- Property tests: merge(A, B, t=0) ≈ A, merge(A, B, t=1) ≈ B
- Roundtrip: merge → eval → assert quality within tolerance
- Mutation testing: >80% mutation score on merge logic

---

## Provable Contracts

### Contract: `merge-core-v1.yaml`

Extends `lora-algebra-v1.yaml` which already covers task vectors and DARE.

```yaml
metadata:
  description: "Model merging — weight-space interpolation strategies"
  references:
    - "Wortsman et al. (2022) Model Soups"
    - "Yadav et al. (2023) TIES-Merging"
    - "Yu et al. (2023) DARE"
  depends_on:
    - "lora-algebra-v1"
    - "tensor-shape-flow-v1"

equations:
  weighted_average:
    formula: "θ_merged = Σ w_i * θ_i"
    invariants:
      - "Σ w_i = 1 (normalization)"
      - "w_i >= 0 (non-negative weights)"
      - "shape(θ_merged) == shape(θ_i)"

  slerp:
    formula: "SLERP(A, B, t) = sin((1-t)Ω)/sin(Ω) * A + sin(tΩ)/sin(Ω) * B"
    invariants:
      - "t=0 → output = A"
      - "t=1 → output = B"
      - "||output|| interpolates between ||A|| and ||B||"

  ties_merge:
    formula: "trim(δ) → elect_sign(δ) → disjoint_merge(δ)"
    invariants:
      - "Only top-density fraction of task vector retained"
      - "Sign conflicts resolved by majority vote"
      - "Requires base model for task vector computation"

proof_obligations:
  - type: invariant
    property: "SLERP boundary t=0"
    formal: "slerp(A, B, 0) == A"
  - type: invariant
    property: "SLERP boundary t=1"
    formal: "slerp(A, B, 1) == B"
  - type: invariant
    property: "Shape preservation"
    formal: "shape(merge(models)) == shape(models[0])"
  - type: invariant
    property: "Weight non-negativity"
    formal: "∀ w_i in weighted_merge: w_i >= 0"
  - type: invariant
    property: "TIES requires base"
    formal: "ties without base_model → validation error"

falsification_tests:
  - id: FALSIFY-MCORE-001
    rule: "SLERP boundary t=0"
    prediction: "slerp(A, B, 0.0) == A within ULP"
    if_fails: "SLERP formula or degenerate case bug"
  - id: FALSIFY-MCORE-002
    rule: "SLERP boundary t=1"
    prediction: "slerp(A, B, 1.0) == B within ULP"
    if_fails: "SLERP formula or boundary handling bug"
  - id: FALSIFY-MCORE-003
    rule: "Shape preservation"
    prediction: "merged model has same tensor shapes as inputs"
    if_fails: "Accumulation or reshape bug"
  - id: FALSIFY-MCORE-004
    rule: "Negative weight rejection"
    prediction: "negative weight returns validation error"
    if_fails: "Weight validation missing"
  - id: FALSIFY-MCORE-005
    rule: "TIES base model required"
    prediction: "ties without --base-model returns error"
    if_fails: "Base model check missing for TIES strategy"

kani_harnesses:
  - id: KANI-MCORE-001
    obligation: "Shape preservation"
    property: "output shape == input shape for bounded dims"
    bound: 8
    strategy: exhaustive
```

### Binding Requirements

```yaml
  - contract: merge-core-v1.yaml
    equation: weighted_average
    module_path: "aprender::format::apr_merge"
    function: merge_weighted
    status: implemented

  - contract: merge-core-v1.yaml
    equation: slerp
    module_path: "aprender::format::apr_merge"
    function: merge_slerp
    status: implemented

  - contract: merge-core-v1.yaml
    equation: ties_merge
    module_path: "aprender::format::apr_merge"
    function: merge_ties
    status: implemented
```
