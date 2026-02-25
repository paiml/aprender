# Fine-Tuning Provable Design by Contract

**Reference**: Meyer, B. (1992). "Applying 'Design by Contract'." *IEEE Computer*, 25(10), 40-51.

**Status**: Phase 1 COMPLETE (CPU training). Phase 2 COMPLETE (PTX forward kernels). Phase 3 PLANNED (GPU backward kernels).
**Date**: 2026-02-25
**Scope**: aprender, entrenar, trueno, provable-contracts

---

## 1. Overview

This specification defines the provable contract chain for the **fine-tuning path** (classification LoRA training), achieving parity with the inference contract chain already enforced down to PTX.

### Inference vs Fine-Tuning Contract Parity

| Layer | Inference Path | Fine-Tuning Path |
|-------|---------------|------------------|
| **YAML Contract** | `model-families/*.yaml`, `tensor-layout-v1.yaml` | `classification-finetune-v1.yaml`, `lora-algebra-v1.yaml` |
| **Compile-Time Proc Macro** | `#[contract]` via `build.rs` env vars | `#[contract]` via `build.rs` env vars |
| **Poka-Yoke Types** | `ValidatedTensorLayout`, `ValidatedShape` | `ValidatedClassLogits`, `ValidatedSafetyLabel`, `ValidatedClassifierWeight` |
| **Config Factory** | `GpuModelConfig::from_safetensors()` | `TransformerConfig::qwen3_5_9b()`, `TransformerConfig::qwen2_0_5b()` |
| **Runtime Dispatch** | `forward_block_incremental()` / `forward_linear_block_incremental()` | `ClassifyTrainer::train()` → `ClassifyPipeline::forward_backward_single()` |
| **Kernel Contracts** | `softmax-kernel-v1`, `rmsnorm-kernel-v1`, `rope-kernel-v1` | `cross-entropy-kernel-v1`, `adamw-kernel-v1`, `lora-algebra-v1` |
| **SIMD Dispatch** | scalar / AVX2 / PTX for all inference kernels | scalar / AVX2 / PTX for CE+AdamW (PARITY ACHIEVED) |
| **Falsification** | `FALSIFY-MF-QWEN35-001..008`, `FALSIFY-TL-*` | `FALSIFY-CLASS-001..006`, `FALSIFY-FT-QWEN35-001..007` |
| **QA Gate** | `apr qa` (8 gates) | `apr qa --assert-classifier-head` + `cargo test -- falsify` |

---

## 2. Meyer's Framework Applied to Fine-Tuning

### 2.1 Preconditions (Caller Obligations)

| Component | Precondition | Enforcement |
|-----------|-------------|-------------|
| Config Factory | `model_size` must map to valid config | CLI match arm; unknown → `TransformerConfig::tiny()` |
| Corpus Loader | JSONL must have `input` and `label` fields | `entrenar::Result<Vec<SafetySample>>` — no defaults |
| LoRA Init | `rank > 0`, `rank < min(in_features, out_features)` | `lora-algebra-v1.yaml` FALSIFY-LA-003 |
| Label Validation | `label_index < num_classes` | `ValidatedSafetyLabel::new()` returns `Err` on violation (F-CLASS-002) |
| Logit Shape | `logits.len() == num_classes` | `ValidatedClassLogits::new()` returns `Err` on violation (F-CLASS-001) |

Meyer (p.44): "The stronger the precondition, the heavier the burden on the client, and the easier for the supplier." All preconditions here are **strong** — no silent defaults, no `unwrap_or`.

### 2.2 Postconditions (Supplier Guarantees)

| Component | Postcondition | Enforcement |
|-----------|--------------|-------------|
| Cross-Entropy | `loss >= 0`, finite output | `cross-entropy-kernel-v1.yaml` FALSIFY-CE-001, CE-003 |
| AdamW | `v_t >= 0` (second moment), finite update | `adamw-kernel-v1.yaml` FALSIFY-AW-002, AW-004 |
| LoRA Forward | `output_shape == input_shape` | `lora-algebra-v1.yaml` FALSIFY-LA-005 |
| Softmax | `sum(probs) == 1.0` within `1e-5` | FALSIFY-CLASS-003 |
| Gradient Flow | LoRA A, B matrices receive non-zero gradients | autograd backward pass; verified by `train()` convergence |

Meyer (p.42): "Properties that are ensured in return by the execution of the call."

### 2.3 Class Invariants

| Type | Invariant | Mechanism |
|------|-----------|-----------|
| `ValidatedClassLogits` | `data.len() == num_classes`, `num_classes >= 2`, no NaN/Inf | Private inner field + validated `new()` |
| `ValidatedSafetyLabel` | `index < num_classes` | Private inner field + validated `new()` |
| `ValidatedClassifierWeight` | `data.len() == hidden_size * num_classes` | Private inner field + validated `new()` |
| `TransformerConfig` | Dimensions consistent with `qwen3_5.yaml` contract | Factory method with constants; falsification tests verify |

Meyer (p.45): "A property that applies to all instances of the class."

---

## 3. Contract Chain: Config → Types → Training → Loss → Optimizer

### 3.1 Stage 1: YAML Contract (Source of Truth)

```
contracts/classification-finetune-v1.yaml
    |
    |  F-CLASS-001..008: classification invariants
    |  Poka-Yoke type specs
    |  Shell safety domain (5 classes)
    |  Model architecture (Qwen2.5-0.5B, Qwen3.5-9B)
    |
    v
contracts/model-families/qwen3_5.yaml
    |
    |  hidden_dim=4096, num_heads=16, num_kv_heads=4
    |  has_bias=false, vocab_size=248320, head_dim=256
    |
    v
provable-contracts/contracts/lora-algebra-v1.yaml
provable-contracts/contracts/cross-entropy-kernel-v1.yaml
provable-contracts/contracts/adamw-kernel-v1.yaml
```

### 3.2 Stage 2: Compile-Time Type Enforcement

```rust
// Private inner fields — ONLY way to construct is validated new()
pub struct ValidatedClassLogits { data: Vec<f32>, num_classes: usize }
pub struct ValidatedSafetyLabel { inner: SafetyClass }
pub struct ValidatedClassifierWeight { data: Vec<f32>, hidden_size: usize, num_classes: usize }

// Poka-Yoke: invalid states are unrepresentable
ValidatedClassLogits::new(vec![0.1; 3], 5)  // → Err (3 != 5)
ValidatedSafetyLabel::new(5, 5)             // → Err (5 >= 5)
ValidatedClassifierWeight::new(vec![0.1; 100], 128, 5) // → Err (100 != 640)
```

### 3.3 Stage 3: Config Factory Enforcement

```rust
// TransformerConfig::qwen3_5_9b() — hardcoded from qwen3_5.yaml contract
const QWEN3_5_9B_HIDDEN_SIZE: usize = 4096;
const QWEN3_5_VOCAB_SIZE: usize = 248320;

pub fn qwen3_5_9b() -> Self {
    Self {
        hidden_size: 4096,
        num_attention_heads: 16,     // qwen3_5.yaml: num_heads
        num_kv_heads: 4,             // qwen3_5.yaml: num_kv_heads
        num_hidden_layers: 32,       // qwen3_5.yaml: num_layers
        vocab_size: 248320,          // qwen3_5.yaml: vocab_size
        use_bias: false,             // qwen3_5.yaml: has_bias=false
        rope_theta: 1_000_000.0,     // qwen3_5.yaml: rope_theta
        ..
    }
}
```

Falsification tests (FALSIFY-FT-QWEN35-001..007) verify the factory matches the YAML contract.

### 3.4 Stage 4: Training Loop Contracts

```
ClassifyTrainer::new()
    |
    |  Loads corpus → F-CLASS-002 validated labels at load time
    |  Builds ClassificationHead(hidden_size, num_classes)
    |  Initializes LoRA adapters on Q/V projections
    |
    v
ClassifyTrainer::train() → ClassifyPipeline::train_batch()
    |
    |  For each sample in batch:
    |    forward_backward_single():
    |      Precondition:  debug_assert!(label < num_classes)       [F-CLASS-002]
    |      1. Forward: embeddings → LoRA(Q,V) → attention → mean_pool → classifier
    |      Postcondition: debug_assert!(logits.len() == num_classes) [F-CLASS-001]
    |      Postcondition: debug_assert!(logits.all(is_finite))      [F-CLASS-001]
    |      2. Loss: cross_entropy(logits, label)
    |      Postcondition: debug_assert!(loss.is_finite() && loss >= 0) [F-CLASS-005]
    |      3. Backward: autograd computes gradients through LoRA A, B matrices
    |    Optimizer: AdamW step with decoupled weight decay (once per batch)
    |
    v
ClassifyTrainer::save_checkpoint()
    |
    |  classifier.weight, classifier.bias → model.safetensors
    |  lora.{layer}.{q,v}_proj.lora_{a,b} → model.safetensors
    |  metadata.json (epoch, loss, accuracy)
```

### 3.5 Stage 5: Kernel Contracts (Provable)

#### Cross-Entropy Loss

```yaml
# From cross-entropy-kernel-v1.yaml
equations:
  cross_entropy:
    formula: "CE(targets, logits) = -sum(targets_i * log_softmax(logits)_i)"
    invariants:
      - "CE >= 0 (non-negativity)"
      - "CE(one_hot(k), logits) = -log_softmax(logits)_k"

kernel_structure:
  phases: [find_max, log_sum_exp, log_softmax, nll]

simd_dispatch:
  cross_entropy:
    scalar: cross_entropy_scalar      # ✅ Implemented
    avx2: cross_entropy_avx2          # ✅ Implemented
    ptx: cross_entropy_ptx            # ✅ Implemented (3-phase shared memory reduction)
```

#### AdamW Optimizer

```yaml
# From adamw-kernel-v1.yaml
equations:
  weight_update:
    formula: "theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat)+eps) + lambda*theta)"
    invariants:
      - "Weight decay applied AFTER Adam update (decoupled)"
      - "Update finite when inputs finite and eps > 0"

simd_dispatch:
  adamw:
    scalar: adamw_step_scalar         # ✅ Implemented
    avx2: adamw_step_avx2             # ✅ Implemented
    ptx: adamw_ptx                    # ✅ Implemented (elementwise, 1 thread per param)
```

#### LoRA Algebra

```yaml
# From lora-algebra-v1.yaml
equations:
  task_vector:
    formula: "delta = W_fine - W_base"
    invariants:
      - "Additive: W_base + delta == W_fine (roundtrip)"
  lora_shape:
    formula: "A ∈ ℝ^{m×r}, B ∈ ℝ^{r×n}, A @ B ∈ ℝ^{m×n}"
    invariants:
      - "A @ B has same shape as original weight"
      - "Storage: r*(m+n) << m*n for small r"

simd_dispatch:
  # LoRA forward is matmul — dispatched through matmul-kernel-v1
  # No separate SIMD dispatch needed
```

---

## 4. Binding Registry Coverage

The `provable-contracts/contracts/aprender/binding.yaml` maps kernel equations to Rust implementations. Fine-tuning bindings:

| Contract | Equation | Module Path | Status |
|----------|----------|-------------|--------|
| `cross-entropy-kernel-v1.yaml` | `cross_entropy` | `aprender::nn::loss::CrossEntropyLoss::forward` | implemented |
| `cross-entropy-kernel-v1.yaml` | `log_softmax` | `aprender::nn::loss::CrossEntropyLoss::forward` | implemented |
| `adamw-kernel-v1.yaml` | `adam_moments` | `entrenar::optim::AdamW::step` | implemented |
| `adamw-kernel-v1.yaml` | `adam_variance` | `entrenar::optim::AdamW::step` | implemented |
| `adamw-kernel-v1.yaml` | `bias_correction` | `entrenar::optim::AdamW::step` | implemented |
| `adamw-kernel-v1.yaml` | `weight_update` | `entrenar::optim::AdamW::step` | implemented |
| `lora-algebra-v1.yaml` | `task_vector` | `entrenar::lora::LoRALayer` | implemented |
| `lora-algebra-v1.yaml` | `eckart_young` | `entrenar::lora::LoRALayer` | implemented |
| `lora-algebra-v1.yaml` | `lora_shape` | `entrenar::lora::LoRALayer` | implemented |
| `lora-algebra-v1.yaml` | `dare_unbiased` | `entrenar::lora::LoRALayer` | implemented |
| `lora-algebra-v1.yaml` | `shape_preservation` | `entrenar::lora::LoRALayer` | implemented |

### Provable-Contracts Kernel Implementations

| Contract | Kernel | scalar | AVX2 | PTX |
|----------|--------|--------|------|-----|
| `cross-entropy-kernel-v1` | `cross_entropy` | `cross_entropy_scalar` | `cross_entropy_avx2` | `cross_entropy_ptx` |
| `adamw-kernel-v1` | `adamw` | `adamw_step_scalar` | `adamw_step_avx2` | `adamw_step_ptx` |
| `activation-kernel-v1` | `silu` | `silu_scalar` | `silu_avx2` | `silu_ptx` |
| `softmax-kernel-v1` | `softmax` | `softmax_scalar` | `softmax_avx2` | `softmax_ptx` |

**Parity achieved**: All fine-tuning forward kernels (cross-entropy, AdamW) have full scalar/AVX2/PTX dispatch, matching inference kernels. The remaining gap is GPU backward kernels (autograd on GPU) — see Phase 3.

---

## 5. Falsification Test Registry

### Classification Invariants (aprender)

| Test ID | Rule | What it tries to break |
|---------|------|----------------------|
| FALSIFY-CLASS-001 | F-CLASS-001 | Construct `ValidatedClassLogits` with wrong element count |
| FALSIFY-CLASS-002 | F-CLASS-002 | Construct `ValidatedSafetyLabel` with out-of-range index |
| FALSIFY-CLASS-003 | F-CLASS-003 | Softmax sum deviates from 1.0 |
| FALSIFY-CLASS-004 | F-CLASS-004 | Construct `ValidatedClassifierWeight` with wrong shape |
| FALSIFY-CLASS-005 | F-CLASS-001 | Construct `ValidatedClassLogits` with NaN values |
| FALSIFY-CLASS-006 | Type enforcement | Construct `ValidatedClassLogits` with `num_classes < 2` |
| FALSIFY-CLASS-007 | F-CLASS-007 | Qwen3.5 config must have `use_bias=false` |
| FALSIFY-CLASS-008 | F-CLASS-008 | LoRA adapter count must equal `num_layers * 2` (Q/V per layer) |

### Fine-Tuning Config Factory (aprender)

| Test ID | Rule | What it tries to break |
|---------|------|----------------------|
| FALSIFY-FT-QWEN35-001 | vocab_size | `qwen3_5_9b().vocab_size` must be 248320 (not 152064) |
| FALSIFY-FT-QWEN35-002 | use_bias | `qwen3_5_9b().use_bias` must be false |
| FALSIFY-FT-QWEN35-003 | head_dim | `qwen3_5_9b().head_dim()` must be 256 |
| FALSIFY-FT-QWEN35-004 | num_layers | `qwen3_5_9b().num_hidden_layers` must be 32 |
| FALSIFY-FT-QWEN35-005 | num_kv_heads | `qwen3_5_9b().num_kv_heads` must be 4 |
| FALSIFY-FT-QWEN35-006 | contract sync | All YAML dimensions match config factory |
| FALSIFY-FT-QWEN35-007 | CLI dispatch | `--model-size 9B` resolves to `qwen3_5_9b()` (not Qwen2) |

### Cross-Entropy Kernel (provable-contracts)

| Test ID | Rule | What it tries to break |
|---------|------|----------------------|
| FALSIFY-CE-001 | Non-negativity | `CE(targets, logits) >= 0` for valid targets |
| FALSIFY-CE-002 | Log-softmax bound | `log_softmax(x)_i <= 0` for all i |
| FALSIFY-CE-003 | Numerical stability | No NaN/Inf for finite inputs |
| FALSIFY-CE-004 | Decomposition | Fused CE vs separate log_softmax + NLL |
| FALSIFY-CE-005 | SIMD equivalence | AVX2 matches scalar within 8 ULP |
| FALSIFY-CE-006 | Boundary | CE → 0 for perfect predictions |

### AdamW Kernel (provable-contracts)

| Test ID | Rule | What it tries to break |
|---------|------|----------------------|
| FALSIFY-AW-001 | Decoupled decay | AdamW != Adam + L2 for `lambda > 0` |
| FALSIFY-AW-002 | Moment non-negativity | `v_t >= 0` after 100 random gradient steps |
| FALSIFY-AW-003 | Bias correction | `1/(1-beta^t) > 1` for all valid t, beta |
| FALSIFY-AW-004 | Update finiteness | Finite output for extreme gradients |
| FALSIFY-AW-005 | SIMD equivalence | AVX2 matches scalar within 8 ULP |
| FALSIFY-AW-006 | Boundary (zero grad) | Only weight decay modifies theta when g=0 |

### LoRA Algebra (provable-contracts)

| Test ID | Rule | What it tries to break |
|---------|------|----------------------|
| FALSIFY-LA-001 | Task vector roundtrip | `base + delta == fine_tune` for random matrices |
| FALSIFY-LA-002 | Eckart-Young bound | Truncation error <= next singular value |
| FALSIFY-LA-003 | Shape compatibility | LoRA decomposition preserves output shape |
| FALSIFY-LA-004 | DARE unbiased | Mean of 1000 samples ≈ delta |
| FALSIFY-LA-005 | Shape preservation | `shape(W + BA) == shape(W)` |
| FALSIFY-LA-006 | SIMD equivalence | SIMD LoRA matches scalar |

### Cross-Crate Contract (aprender ↔ entrenar)

| Test ID | Rule | What it tries to break |
|---------|------|----------------------|
| FALSIFY-FT-XCRATE-001 | F-CLASS-006 | `ClassifyConfig::default().num_classes` must be >= 2 |
| FALSIFY-FT-XCRATE-002 | F-CLASS-004 | `ClassificationHead` weight shape must equal `hidden * classes` |
| FALSIFY-FT-XCRATE-003 | F-CLASS-005 | `cross_entropy_loss()` output must be finite and non-negative |
| FALSIFY-FT-XCRATE-004 | F-CLASS-001 | `ClassificationHead::forward()` logits accepted by `ValidatedClassLogits::new()` |

**Total fine-tuning falsification tests: 31** (8 CLASS + 7 FT-QWEN35 + 4 FT-XCRATE + 6 CE + 6 AW + 6 LA — but CE+AW+LA are shared with inference path)

---

## 6. Kani Bounded Model Checking

### Existing Harnesses (provable-contracts, Shared with Inference)

| Harness | Contract | Property | Location |
|---------|----------|----------|----------|
| KANI-CE-001 | `cross-entropy-kernel-v1` | Cross-entropy non-negative for small inputs | `kani_proofs_cd.rs` |
| KANI-CE-002 | `cross-entropy-kernel-v1` | Log-softmax bounded above by zero | `kani_proofs_cd.rs` |
| KANI-CE-003 | `cross-entropy-kernel-v1` | Output finite for finite inputs | `kani_proofs_cd.rs` |
| KANI-AW-001 | `adamw-kernel-v1` | Weight decay decoupled from Adam update | `kani_proofs_e1.rs` |
| KANI-AW-002 | `adamw-kernel-v1` | Second moment stays non-negative | `kani_proofs_e1.rs` |
| KANI-AW-003 | `adamw-kernel-v1` | Update finite with positive epsilon | `kani_proofs_e1.rs` |
| KANI-LA-001 | `lora-algebra-v1` | Shape compatibility for bounded dimensions | PLANNED (contract only) |

### Classification Harnesses (aprender, Fine-Tuning Specific)

| Harness | Contract | Property | Location |
|---------|----------|----------|----------|
| KANI-FT-001 | `classification-finetune-v1` | `ValidatedClassLogits::new` rejects wrong len | `proofs/kani_harnesses.rs` |
| KANI-FT-002 | `classification-finetune-v1` | `ValidatedSafetyLabel::new` rejects out-of-bounds | `proofs/kani_harnesses.rs` |
| KANI-FT-003 | `classification-finetune-v1` | Classifier weight shape matches `hidden * classes` | `proofs/kani_harnesses.rs` |

---

## 7. PTX Parity Gap Analysis

### Inference Path (COMPLETE)

All inference kernels have scalar → AVX2 → PTX dispatch:

```
softmax:  softmax_scalar  → softmax_avx2  → softmax_ptx    ✅
silu:     silu_scalar     → silu_avx2     → silu_ptx       ✅
rmsnorm:  rmsnorm_scalar  → rmsnorm_avx2  → rmsnorm_ptx    ✅
rope:     rope_scalar     → rope_avx2     → rope_ptx       ✅
matmul:   matmul_scalar   → matmul_avx2   → matmul_ptx     ✅
```

### Fine-Tuning Forward Path (COMPLETE)

All training forward kernels have full scalar → AVX2 → PTX dispatch:

```
cross_entropy:   cross_entropy_scalar  → cross_entropy_avx2  → cross_entropy_ptx    ✅
adamw:           adamw_step_scalar     → adamw_step_avx2     → adamw_step_ptx       ✅
lora_forward:    (uses matmul dispatch — ✅ has PTX via matmul-kernel-v1)
```

### Fine-Tuning Backward Path (GAP)

GPU backward kernels for autograd do not yet exist:

```
dL/d_logits:     ❌ MISSING (backward cross-entropy gradient)
dL/d_params:     ❌ MISSING (backward AdamW parameter gradient)
dL/d_LoRA_A:     ❌ MISSING (backward LoRA A matrix gradient)
dL/d_LoRA_B:     ❌ MISSING (backward LoRA B matrix gradient)
```

### Why the Gap Exists

1. **Autograd**: entrenar's training loop uses `aprender::autograd::Tensor` with CPU-only backward ops
2. **Backward kernels**: Gradient computation (dL/dW, dL/dQ, dL/dK, dL/dV) requires backward-pass GPU kernels that don't exist yet in trueno
3. **Priority**: Fine-tuning on CPU with quantized inference on GPU is the current workflow — QLoRA trains the adapter on CPU, deploys the merged model on GPU via realizar

### Parity Roadmap

| Phase | Scope | Deliverable |
|-------|-------|-------------|
| **Phase 1** (COMPLETE) | CPU training kernels | scalar + AVX2 for CE, AdamW; LoRA via matmul |
| **Phase 2** (COMPLETE) | PTX forward training kernels | `cross_entropy_ptx`, `adamw_step_ptx` in provable-contracts |
| **Phase 3** (PLANNED) | GPU backward kernels | `trueno::backward::*` ops for autograd on GPU |
| **Phase 4** (PLANNED) | Full GPU training | entrenar GPU training loop with PTX parity |

---

## 8. QA Gate Integration

### Current Validation Flow

```bash
# 1. Config factory matches contract
cargo test -- falsify_ft_qwen35           # 7 tests

# 2. Poka-Yoke types enforce invariants
cargo test -- falsify_class               # 6 tests

# 3. Cross-crate contract (aprender ↔ entrenar)
cargo test -- falsify_ft_xcrate           # 4 tests

# 4. Kernel contracts verified
cargo test -p provable-contracts -- falsify_ce   # 6 tests
cargo test -p provable-contracts -- falsify_aw   # 6 tests
cargo test -p provable-contracts -- falsify_la   # 6 tests

# 5. QA gate for fine-tuned models
apr qa finetuned-model.apr --assert-classifier-head

# 6. End-to-end dogfood
apr finetune --task classify --model-size 9B --plan --json
```

### Target Validation Flow (Phase 3+)

```bash
# All of the above, PLUS:

# 5. GPU backward kernel parity
cargo test -p trueno -- backward_cross_entropy  # GPU backward matches CPU
cargo test -p trueno -- backward_lora           # GPU LoRA gradient matches CPU

# 6. apr qa gate for fine-tuned models
apr qa finetuned-model.apr --assert-classifier-head
```

---

## 9. Toyota Way Principles

| Principle | Application |
|-----------|-------------|
| **Jidoka** (Automation with human touch) | Validation stops the training pipeline when a defect is detected. No mismatched logits propagate to loss computation. |
| **Poka-Yoke** (Mistake-proofing) | `ValidatedClassLogits` makes wrong-shaped logits a compile-time impossibility. `ValidatedSafetyLabel` makes out-of-range labels impossible. |
| **Genchi Genbutsu** (Go and see) | Falsification tests run known-bad inputs through constructors and verify rejection. 31 total fine-tuning falsification tests (8 CLASS + 7 FT-QWEN35 + 4 XCRATE + 12 kernel). |
| **Kaizen** (Continuous improvement) | As we add GPU training (Phase 2-4), each new kernel gets the same contract chain: YAML → proc macro → falsification → Kani → PTX parity. |
| **Heijunka** (Level production) | Config factories produce consistent dimensions regardless of CLI alias. `9B`, `qwen3.5-9b`, `qwen3_5`, `qwen3.5` all resolve to identical `qwen3_5_9b()` config. |

---

## 10. Verification Commands

```bash
# Full fine-tuning contract validation
cargo test -- falsify_ft_qwen35 falsify_class falsify_ft_xcrate  # 19 tests (aprender)
cargo test -p provable-contracts -- falsify_ce falsify_aw falsify_la  # 18 tests (kernels)

# QA gate for fine-tuned models
apr qa finetuned-model.apr --assert-classifier-head

# Config factory dogfood
apr finetune --task classify --model-size 9B --plan --json

# Book build (documents contracts)
cd book && mdbook build

# Design-by-contract example
cargo run --example design_by_contract
```
