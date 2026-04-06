# LoRA & QLoRA Enhancement Specification

**Version**: 2.0.0
**Date**: 2026-03-08
**Status**: RESEARCH COMPLETE — ready for implementation
**Scope**: entrenar LoRA/QLoRA training must work correctly for all paths (CUDA NF4, CUDA FP32/BF16, CPU), fully wired through `apr` CLI

---

## 1. Problem Statement

entrenar has LoRA infrastructure but only the **CUDA NF4 path** correctly freezes base weights. The standard (non-NF4) training path ignores LoRA config and performs full fine-tuning, destroying base model capabilities.

**Evidence**: FALSIFY-CHAT-003 in bashrs provable-contracts confirmed that despite `lora.enabled: true` in YAML, all 494M Qwen-0.5B weights were updated during chat model training (runs 1-6). This caused catastrophic forgetting after 1 epoch on 17K samples.

Meanwhile, the NF4 CUDA path (Qwen3-4B on Lambda) correctly trained only LoRA adapters and achieved 91.4% val accuracy in epoch 1.

---

## 2. Current State Audit

### 2.1 What Works (CUDA NF4 Path)

| Component | File | Status |
|-----------|------|--------|
| `CudaNf4TransformerBlock::forward()` | `cuda_block.rs:2474-2522` | Base weights frozen (NF4), LoRA add separate |
| `backward_nf4()` | `cuda_block.rs:3194-3323` | Only LoRA A/B gradients computed, NO base weight gradients |
| `lora_optimizer_step()` | `cuda_block.rs:3461-3524` | AdamW on LoRA params only |
| `CudaLoraGradWorkspace` | `cuda_block.rs:2795-2852` | Contains only `grad_lora_{a,b}_{q,v}` — no base weight grad buffers |
| `download_lora_weights()` | `cuda_block.rs:3559-3579` | Exports A/B matrices from GPU |
| B pre-scaling | `cuda_block.rs:2387` | B matrices pre-scaled by `lora_scale` at upload (avoids scale kernel in forward) |

**Confirmed correct**: Base weights stored as NF4, never receive gradients, never updated by optimizer. Only LoRA A/B matrices train.

### 2.2 What Exists But Doesn't Work (Standard Path)

| Component | File | Issue |
|-----------|------|-------|
| `LoRALayer` struct | `lora/layer/core.rs` | Correct design: `trainable_params()` returns only A/B. But not wired into standard training loop |
| `QLoRALayer` struct | `lora/qlora.rs` | Correct: base as `Quantized4Bit`, A/B trainable. CPU-only path |
| `LoRAConfig` | `lora/config.rs` | Config parsing works. `should_apply()` correct. But hardcoded to `target_qv_projections()` |
| CPU LoRA forward | `transformer/attention.rs:450-461` | KAIZEN-011 added `matmul_nt` + `forward_hidden_with_lora()`. Fallback only |
| YAML `lora:` section | training manifest | Parsed but **not connected** to standard `TransformerTrainer` |
| `TransformerTrainer` | `finetune/` | **Does NOT check LoRA config**. Trains all parameters |

### 2.3 What's Missing Entirely

| Component | Impact |
|-----------|--------|
| Standard (non-NF4) CUDA LoRA path | Cannot do LoRA on FP16/BF16 models without NF4 quantization |
| Parameter group separation in optimizer | Optimizer receives all params, not just LoRA params |
| `requires_grad = false` enforcement on base weights | Base weights participate in gradient computation |
| YAML config wiring to `TransformerTrainer` | LoRA YAML section is dead config |
| Adapter-only checkpoint saving | Always saves full model (1.98GB) instead of adapter (few MB) |
| LoRA merge for inference export | No `W_merged = W_0 + (alpha/r) * B @ A` export path |
| rsLoRA scaling option | Only `alpha/r`, not `alpha/sqrt(r)` |
| Target module flexibility from YAML | Hardcoded to Q+V only |
| All-linear targeting | Cannot target MLP projections (gate, up, down) |

### 2.4 Architecture Summary

```
                    CUDA NF4 Path (WORKS)          Standard Path (BROKEN)
                    =====================          ======================
Base weights:       NF4 quantized, frozen          FP32/BF16, NOT frozen
LoRA A/B:           GPU buffers, trained           Exist in LoRALayer, NOT used
Gradients:          LoRA-only workspace            All-weights autograd
Optimizer:          LoRA params only               All params
Checkpoint:         Full model.safetensors         Full model.safetensors
Config wiring:      ClassifyConfig -> build_lora   YAML lora: section IGNORED
```

---

## 3. Target Architecture

### 3.1 Design Principles

1. **Single LoRA abstraction** — one `LoRAAdapter` that works across all backends (CPU, CUDA FP16, CUDA NF4)
2. **Config-driven** — YAML `lora:` section fully wired to all training paths
3. **Optimizer param groups** — base weights excluded from optimizer; only LoRA params + optional norm weights receive updates
4. **Adapter-only checkpoints** — save/load adapters separately (few MB vs full model GB)
5. **HuggingFace PEFT compatible** — export `adapter_config.json` + `adapter_model.safetensors`
6. **rsLoRA default** — use `alpha/sqrt(r)` scaling for rank-stable training

### 3.2 YAML Config Schema (Full)

```yaml
lora:
  enabled: true
  rank: 16
  alpha: 32.0
  scaling: "rslora"           # "standard" (alpha/r) | "rslora" (alpha/sqrt(r))
  dropout: 0.0                # LoRA dropout (0.0 = disabled)
  target_modules:             # flexible targeting
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  # OR shorthand:
  # target_modules: "all_linear"   # targets all nn.Linear equivalents
  # target_modules: "attention"    # q/k/v/o only
  # target_modules: "qv"           # q/v only (original paper default)
  bias: "none"                # "none" | "all" | "lora_only"
  init_weights: "kaiming"     # "kaiming" | "gaussian" | "pissa" | "zeros"
  layers: null                # null = all layers, or [0, 1, 5, 23] for specific layers
  quantize_base: false        # true = QLoRA (NF4 quantize base weights)
  quantize_bits: 4            # 4 = NF4, 8 = INT8
  quant_type: "nf4"           # "nf4" | "fp4"
  double_quantize: true       # quantize the quantization constants (saves ~0.37 bits/param)
  merge_on_save: false        # true = save merged weights, false = save adapter only
```

### 3.3 Training Path Matrix

| Config | Base Storage | Base Frozen | LoRA Compute | Optimizer Params |
|--------|-------------|-------------|--------------|-----------------|
| `lora.enabled=true, quantize_base=false, device=cuda` | FP16/BF16 on GPU | `requires_grad=false` | FP16/BF16 GEMM | LoRA A/B + norms |
| `lora.enabled=true, quantize_base=true, device=cuda` | NF4 on GPU | Inherently frozen (quantized) | Dequant + BF16 GEMM | LoRA A/B + norms |
| `lora.enabled=true, quantize_base=false, device=cpu` | FP32 in RAM | `requires_grad=false` | FP32 matmul | LoRA A/B + norms |
| `lora.enabled=false` | FP16/BF16/FP32 | All trainable | Standard | All params |

### 3.4 Forward Pass (Pseudocode)

```
fn forward_with_lora(x, W_base, lora_a, lora_b, scale):
    # Base computation (no gradient for W_base when frozen)
    if quantized:
        h_base = dequantize(W_base) @ x    # NF4 -> BF16 on the fly
    else:
        h_base = W_base @ x                # FP16/BF16, requires_grad=false

    # LoRA computation (always has gradient)
    h_lora = lora_b @ (lora_a @ x)         # [d_out, r] @ ([r, d_in] @ [d_in, seq])
    h_lora = h_lora * scale                # alpha/sqrt(r) for rsLoRA

    return h_base + h_lora
```

### 3.5 Backward Pass

```
fn backward_with_lora(grad_output, x, W_base, lora_a, lora_b, scale):
    # Propagate gradient through base path (for upstream layers)
    if quantized:
        grad_x_base = dequantize(W_base).T @ grad_output
    else:
        grad_x_base = W_base.T @ grad_output   # W_base has no grad itself

    # LoRA gradients
    inter = lora_a @ x                          # cached from forward
    grad_lora_b = (grad_output * scale) @ inter.T
    grad_inter = lora_b.T @ (grad_output * scale)
    grad_lora_a = grad_inter @ x.T
    grad_x_lora = lora_a.T @ grad_inter

    return grad_x_base + grad_x_lora, grad_lora_a, grad_lora_b
    # NOTE: NO grad_W_base computed
```

### 3.6 Optimizer Integration

```
fn build_optimizer(model, lora_config, lr, weight_decay):
    trainable_params = []

    for layer in model.layers:
        for (name, module) in layer.named_modules():
            if lora_config.should_apply(name, layer.idx):
                # Freeze base weight
                module.weight.requires_grad = false
                # Collect LoRA params
                trainable_params.extend([module.lora_a, module.lora_b])

        # Always train norm weights (small, critical for adaptation)
        trainable_params.extend([layer.input_norm.weight, layer.post_attn_norm.weight])

    # Optional: LoRA+ different LR for A vs B
    # param_groups = [
    #     {"params": all_lora_a, "lr": lr},
    #     {"params": all_lora_b, "lr": lr * 16},  # LoRA+
    # ]

    return AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
```

---

## 4. Implementation Plan

### Phase 1: Fix Standard Path (P0 — unblocks all non-NF4 LoRA)

**ENT-LoRA-001**: Wire YAML `lora:` config to `TransformerTrainer`
- Parse `lora:` section in training manifest
- Create `LoRALayer` instances for each target module
- Replace base module forward with LoRA-wrapped forward
- **Test**: Train Qwen-0.5B with `lora.enabled=true, quantize_base=false` — verify base weight checksums unchanged after training

**ENT-LoRA-002**: Parameter group separation
- `TransformerTrainer::build_optimizer()` must filter params by `requires_grad`
- Base weights: `requires_grad = false` (set at init, never changed)
- LoRA A/B: `requires_grad = true`
- Norm weights: `requires_grad = true` (trainable, as in NF4 path)
- **Test**: Assert `optimizer.param_count() == lora_param_count + norm_param_count`

**ENT-LoRA-003**: Adapter-only checkpoint saving
- Save: `adapter_model.safetensors` (LoRA A/B tensors only) + `adapter_config.json`
- Load: Load base model, then overlay adapter weights
- **Test**: Save adapter, reload, verify `forward(x)` produces identical output

**Provable contracts**:
- C-LORA-FREEZE-001: `base_weight.requires_grad == false` for all adapted modules
- C-LORA-PARAM-001: `optimizer.params` contains only LoRA + norm tensors
- C-LORA-CKPT-001: Adapter checkpoint < 1% of full model size

### Phase 2: Enhance LoRA Quality (P1 — world-class features)

**ENT-LoRA-004**: rsLoRA scaling
- Add `scaling: "rslora"` config option
- Compute `scale = alpha / sqrt(r)` instead of `alpha / r`
- Default to rsLoRA for rank > 16
- **Test**: Verify scale factor correct for r=4,8,16,32,64,128

**ENT-LoRA-005**: Flexible target modules
- Support `"all_linear"` shorthand → expand to all projection names for model architecture
- Support per-architecture module name mapping (Qwen: q/k/v/o/gate/up/down, LLaMA: same, GPT-2: c_attn/c_proj/c_fc)
- Support layer filtering: `layers: [0, 1, 22, 23]` for first/last layer targeting
- **Test**: `should_apply("gate_proj", 5)` returns true with `target_modules: "all_linear"`

**ENT-LoRA-006**: LoRA+ optimizer (separate LR for A and B)
- Two param groups: `{A matrices, lr=base_lr}` and `{B matrices, lr=base_lr * ratio}`
- Default ratio: 16 (per Hayou et al. ICML 2024)
- Config: `lora.lora_plus_ratio: 16` (0 = disabled)
- **Test**: Verify A and B receive different effective learning rates

**ENT-LoRA-007**: HuggingFace PEFT export
- Export `adapter_config.json` with standard PEFT schema
- Tensor naming: `base_model.model.model.layers.{i}.self_attn.{proj}.lora_{A,B}.weight`
- Compatible with `peft.PeftModel.from_pretrained()` for loading in Python
- **Test**: Export adapter, load in HF PEFT, verify output matches

### Phase 3: QLoRA Enhancements (P1 — complete the NF4 path)

**ENT-LoRA-008**: Double quantization
- Quantize the FP32 absmax constants to FP8 with block_size=256
- Saves ~0.37 bits/param (~0.5 GB for 7B model)
- Config: `lora.double_quantize: true` (default true when `quantize_base: true`)
- **Test**: Verify dequantize(double_quant(x)) within 1% of dequantize(single_quant(x))

**ENT-LoRA-009**: Merge and export
- `merge_lora_weights()`: Compute `W_merged = dequant(W_base) + scale * B @ A`
- Export as standard safetensors (no adapter, no quantization — ready for inference)
- Support FP16 and BF16 output dtypes
- **Test**: `merged_forward(x) == lora_forward(x)` within FP16 tolerance

**ENT-LoRA-010**: Paged optimizer states (stretch)
- Page AdamW m/v states to CPU RAM when GPU VRAM pressure detected
- Use CUDA unified memory or manual page-in/page-out
- Enables training larger models on smaller GPUs
- Config: `optimizer.paged: true`
- **Test**: Train 7B QLoRA on 16GB GPU without OOM

### Phase 4: apr-cli Wiring (P0 — must ship with Phase 1)

Every LoRA/QLoRA feature MUST be accessible through the `apr` CLI. No dead config.

**ENT-LoRA-014**: `apr train` LoRA integration
- `apr train --config manifest.yaml` where `lora:` section is fully honored
- `train_transformer_from_spec()` in `config/train/loader.rs` must:
  1. Read `spec.lora` (already parsed as `Option<LoRASpec>`)
  2. Call `build_lora_layers()` to create adapters for target modules
  3. Set `requires_grad = false` on all base weight tensors
  4. Pass only LoRA + norm params to optimizer
  5. Use LoRA-wrapped forward in training loop
- Print LoRA summary at startup:
  ```
  LoRA: rank=16, alpha=32.0, scaling=rsLoRA, targets=[q_proj, v_proj]
    Trainable: 1,081,344 / 494,032,896 (0.22%)
    Base weights: FROZEN
  ```
- **Test**: `apr train --config lora_test.yaml --dry-run` shows correct LoRA summary

**ENT-LoRA-015**: `apr train` adapter checkpoint flow
- When `lora.enabled=true`:
  - `save_every` saves `adapter_model.safetensors` + `adapter_config.json` (NOT full model)
  - Final save writes both adapter and optionally merged model (`merge_on_save: true`)
  - `--resume` loads base model + adapter overlay (NOT full checkpoint)
- Print checkpoint info:
  ```
  Saving adapter checkpoint (step 500): adapter_model.safetensors (2.1 MB)
  ```
- **Test**: Checkpoint at step 500 is < 50MB; resume produces identical loss at step 501

**ENT-LoRA-016**: `apr finetune plan` LoRA HPO
- `finetune plan --lora-rank 16 --lora-alpha 32 --target-modules q_proj,v_proj` already accepts LoRA flags
- Must generate plan YAML with correct `lora:` section
- HPO grid must include LoRA rank as searchable hyperparameter:
  ```yaml
  search_space:
    lora_rank: [8, 16, 32]
    lora_alpha: [16, 32, 64]
    lr: [1e-4, 3e-4, 5e-4]
  ```
- `finetune apply --plan plan.yaml` must honor LoRA config per trial
- **Test**: `apr finetune plan` with LoRA flags generates valid plan; `apply` trains with frozen base

**ENT-LoRA-017**: `apr merge` for LoRA adapters
- New subcommand or extend existing `merge`:
  ```
  apr merge --base /path/to/model --adapter /path/to/adapter --output /path/to/merged
  ```
- Computes `W_merged = W_base + scale * B @ A` for each adapted module
- Outputs standard safetensors (no LoRA, no quantization) ready for inference
- For QLoRA: `apr merge --base /path/to/nf4_model --adapter /path/to/adapter --output /path/to/merged --dtype f16`
- **Test**: `merged_model.forward(x) == lora_model.forward(x)` within tolerance

**ENT-LoRA-018**: `apr inspect` LoRA diagnostics
- Extend existing `inspect` command:
  ```
  apr inspect --checkpoint /path/to/adapter/
  ```
- Output:
  ```
  Adapter: /path/to/adapter/adapter_model.safetensors
  Format: PEFT-compatible LoRA
  Base model: Qwen2.5-Coder-0.5B-Instruct
  Rank: 16, Alpha: 32.0, Scaling: rsLoRA
  Target modules: q_proj, v_proj (48 adapter tensors)
  Trainable params: 1,081,344 (0.22% of base)
  Adapter size: 2.1 MB
  ```
- **Test**: `apr inspect` on saved adapter outputs correct metadata

### Phase 5: Advanced Variants (P2 — future work)

**ENT-LoRA-011**: DoRA (Weight-Decomposed LoRA)
- Decompose W into magnitude `m` and direction `V/||V||`
- Apply LoRA to direction only: `W' = m * (V + scale * B @ A) / ||V + scale * B @ A||`
- +1-3% accuracy over standard LoRA (ICML 2024 Oral)

**ENT-LoRA-012**: PiSSA initialization
- SVD of base weight: `W = U * S * V^T`
- Initialize A, B from top-r singular components
- Faster convergence (+5% on some benchmarks, NeurIPS 2024 Spotlight)

**ENT-LoRA-013**: Multi-adapter training
- Already partially implemented (GPU-SHARE Phase 2.1)
- N adapters sharing one frozen base model
- Independent optimizer states per adapter

---

## 5. Equations (Provable Contracts)

### E-LORA-FWD: LoRA Forward

```
h = W_0 @ x + scale * (B @ (A @ x))

where:
  scale = alpha / sqrt(r)           # rsLoRA (default)
  scale = alpha / r                 # standard LoRA
  W_0 in R^{d_out x d_in}          # frozen base weight
  A in R^{r x d_in}                # trainable, Kaiming init
  B in R^{d_out x r}               # trainable, zero init
  r << min(d_in, d_out)            # low rank constraint
```

**Invariants**:
- `delta_W = scale * B @ A = 0` at initialization (B=0)
- `W_0.requires_grad = false` throughout training
- Trainable parameter count = `2 * r * (d_in + d_out)` per adapted module

### E-NF4-QUANT: NF4 Quantization

```
Quantize:
  absmax_b = max(|w_b|) for each block b of 64 elements
  w_norm = w_b / absmax_b           # normalize to [-1, 1]
  idx = argmin_i |w_norm - NF4[i]|  # nearest NF4 codebook entry
  store: (idx as u4, absmax_b as f32)

Dequantize:
  w_approx = NF4[idx] * absmax_b

NF4 codebook (16 values from N(0,1) equiprobable quantiles):
  [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
    0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]
```

### E-LORA-MERGE: Weight Merging

```
W_merged = W_0 + scale * B @ A

For QLoRA:
  W_merged = dequantize(W_0_nf4) + scale * B @ A

Invariant:
  ||forward(x, W_merged) - forward_lora(x, W_0, A, B, scale)|| < eps
  where eps = O(machine_epsilon * ||x||)
```

### E-LORA-PARAM: Parameter Count

```
Per adapted module:
  lora_params = r * d_in + r * d_out     # A + B (no bias)

Per model (targeting N modules across L layers):
  total_lora = L * N * (r * d_in + r * d_out)
  total_norm = L * 2 * d_model            # input_norm + post_attn_norm

Example: Qwen-0.5B, r=16, Q+V targeting:
  Per layer: 16*896 + 16*896 = 28,672 (Q) + 16*896 + 16*128 = 16,384 (V) = 45,056
  Total: 24 layers * 45,056 = 1,081,344 LoRA params
  Ratio: 1.08M / 494M = 0.22% of base model
```

---

## 6. Falsification Tests

Every prediction below MUST be tested. If any prediction is falsified, the corresponding
implementation ticket is blocked until the root cause is fixed. This is the primary
quality gate — passing unit tests is necessary but not sufficient.

### Layer 0: Mathematical Invariants

These test the LoRA equations directly. No training loop, no CLI, no I/O.

| ID | Prediction | Test | If Falsified |
|----|-----------|------|-------------|
| F-MATH-001 | `B @ A = 0` at initialization (B=zeros) | Create LoRALayer(r=16, d_in=896, d_out=896), compute `lora_b.data @ lora_a.data`, assert all zeros | Init bug: B not zero-initialized |
| F-MATH-002 | `lora_forward(x) == base_forward(x)` when B=0 | Forward with fresh LoRALayer, compare to `W_base @ x` | Forward path adds non-zero LoRA contribution at init |
| F-MATH-003 | Scale = `alpha / sqrt(r)` for rsLoRA | For r in [4,8,16,32,64,128,256], assert `scale == alpha / r.sqrt()` | Scale formula bug |
| F-MATH-004 | Scale = `alpha / r` for standard LoRA | Same sweep, assert `scale == alpha / r` | Scale formula bug |
| F-MATH-005 | Merge produces exact equivalence | `forward(x, W_0 + scale*B@A) == forward_lora(x, W_0, A, B, scale)` for 100 random x | Merge formula incorrect, precision loss |
| F-MATH-006 | Unmerge is lossless roundtrip | `unmerge(merge(layer)); assert layer.lora_a == original_a, layer.lora_b == original_b` | Floating point accumulation in merge/unmerge |
| F-MATH-007 | Trainable param count is exact | `2 * num_targets * num_layers * r * (d_in + d_out)` matches `layer.trainable_params().len()` summed | Param counting bug |
| F-MATH-008 | NF4 dequantize within tolerance | For 1000 random f32 vectors, `max(abs(dequant(quant(x)) - x)) < 0.15 * max(abs(x))` | NF4 codebook or block quantization bug |

### Layer 1: Weight Freezing (The Critical Bug)

These test the exact bug that caused FALSIFY-CHAT-003. Every training path must pass ALL of these.

| ID | Prediction | Test | If Falsified |
|----|-----------|------|-------------|
| F-FREEZE-001 | Base weights unchanged after 100 steps (CUDA FP16 LoRA) | SHA256 of all base weight tensors before and after training. Assert identical. | `requires_grad` not set on CUDA FP16 path |
| F-FREEZE-002 | Base weights unchanged after 100 steps (CUDA NF4 QLoRA) | Same SHA256 check on NF4-quantized base weights | NF4 path regression |
| F-FREEZE-003 | Base weights unchanged after 100 steps (CPU LoRA) | Same SHA256 check on CPU path | CPU path doesn't freeze |
| F-FREEZE-004 | LoRA A matrices CHANGE after 100 steps | Assert `lora_a` tensors differ from initialization | LoRA not receiving gradients |
| F-FREEZE-005 | LoRA B matrices CHANGE after 100 steps | Assert `lora_b` tensors differ from zero initialization | LoRA not receiving gradients |
| F-FREEZE-006 | Norm weights CHANGE after 100 steps | Assert `input_norm` and `post_attn_norm` weights differ from init | Norms accidentally frozen |
| F-FREEZE-007 | Optimizer state has NO entries for base weights | Inspect optimizer m/v buffers, assert no buffer exists for any base weight tensor | Optimizer tracking frozen params (wasting memory + potential update) |
| F-FREEZE-008 | `grad(base_weight) is None` after backward | After one forward+backward step, check `.grad` field of every base weight | Autograd computing unnecessary base gradients |

### Layer 2: apr-cli Integration

These test end-to-end CLI behavior. Must use actual `apr` binary via `assert_cmd`.

| ID | Prediction | Test | If Falsified |
|----|-----------|------|-------------|
| F-CLI-001 | `apr train --config lora.yaml` creates LoRA layers | Run with `--dry-run`, parse stdout for "LoRA: rank=16" and "Trainable:" | YAML `lora:` section not wired to TransformerTrainer |
| F-CLI-002 | `apr train --config lora.yaml` saves adapter checkpoint | Train 10 steps with `save_every: 5`, verify `adapter_model.safetensors` exists at step 5 and 10 | Checkpoint path still saves full model |
| F-CLI-003 | Adapter checkpoint < 1% of base model size | `size(adapter_model.safetensors) / size(base_model.safetensors) < 0.01` | Saving full weights instead of adapter |
| F-CLI-004 | `apr train --resume` loads adapter and continues | Train 50 steps, save. Resume, train 50 more. Assert loss at step 51 is close to loss at step 50. | Resume doesn't load adapter or resets optimizer |
| F-CLI-005 | `apr train --config qlora.yaml` uses NF4 quantization | Run with QLoRA config, parse stdout for "NF4" or "quantized". Assert VRAM < full FP16 model | QLoRA config ignored, falls back to FP16 |
| F-CLI-006 | `apr finetune plan --lora-rank 16` generates valid plan | Generate plan, parse YAML, assert `lora.rank == 16` in plan | Finetune plan ignores LoRA flags |
| F-CLI-007 | `apr finetune apply` with LoRA plan freezes base weights | Run apply with SHA256 base weight check | Finetune apply path doesn't freeze |
| F-CLI-008 | `apr merge --base X --adapter Y --output Z` produces valid model | Merge, then run inference on merged model. Assert output matches LoRA model. | Merge formula bug or tensor naming mismatch |
| F-CLI-009 | `apr inspect --checkpoint adapter/` shows correct metadata | Parse inspect output for rank, alpha, target modules, param count | Inspect doesn't read adapter format |
| F-CLI-010 | `apr train` with NO `lora:` section trains all params | Train without LoRA, verify all weights change. Ensures LoRA code doesn't break normal training. | LoRA changes broke standard training path |

### Layer 3: Training Convergence

These test that LoRA training actually learns. Run on small model (Qwen-0.5B or smaller test model).

| ID | Prediction | Test | If Falsified |
|----|-----------|------|-------------|
| F-CONV-001 | Loss decreases over 200 steps (FP16 LoRA) | Train 200 steps, assert `loss[200] < loss[1] * 0.5` | Forward/backward bug in FP16 LoRA path |
| F-CONV-002 | Loss decreases over 200 steps (NF4 QLoRA) | Same on NF4 path | NF4 dequantization or gradient flow bug |
| F-CONV-003 | Loss decreases over 200 steps (CPU LoRA) | Same on CPU path | CPU LoRA matmul or autograd bug |
| F-CONV-004 | FP16 LoRA and NF4 QLoRA converge to similar loss | After 500 steps on same data, assert `abs(loss_fp16 - loss_nf4) / loss_fp16 < 0.10` | Quantization error too large, or path divergence |
| F-CONV-005 | rsLoRA stable at r=128 | Train 200 steps at r=128 with rsLoRA, assert loss decreases and doesn't diverge | rsLoRA scaling not applied (alpha/r at r=128 ~= 0) |
| F-CONV-006 | Higher rank = lower loss (given sufficient data) | Train r=4 vs r=16 vs r=64 on same data 500 steps, assert `loss[r=64] <= loss[r=4]` | Rank increase not helping — possible scale bug |
| F-CONV-007 | LoRA loss < random baseline | After 200 steps LoRA loss < untrained model loss on same test data | Training has no effect |
| F-CONV-008 | No catastrophic forgetting with LoRA | Train 3 epochs on 17K samples, assert `loss[epoch3] < loss[epoch1]` (unlike full FT which diverges) | LoRA not actually freezing base weights, or norms destabilizing |

### Layer 4: Checkpoint & Serialization

| ID | Prediction | Test | If Falsified |
|----|-----------|------|-------------|
| F-CKPT-001 | Adapter save/load roundtrip preserves forward output | Save adapter, create fresh model, load adapter. Assert `forward(x)` identical to 6 decimal places | Tensor naming, shape, or scale mismatch in save/load |
| F-CKPT-002 | PEFT export loads in HuggingFace | Export `adapter_config.json` + `adapter_model.safetensors`. Validate JSON schema. Load with `peft.PeftModel.from_pretrained()`. Assert forward matches. | Tensor naming convention wrong, or config schema mismatch |
| F-CKPT-003 | Merged model has no LoRA tensors | After merge, inspect safetensors keys. Assert NO key contains "lora_a" or "lora_b" | Merge didn't fold adapters into base weights |
| F-CKPT-004 | Merged model is same size as base model | `abs(size(merged) - size(base)) < 1%` | Extra tensors leaked into merged checkpoint |
| F-CKPT-005 | Resume from adapter checkpoint continues training correctly | Save at step 100. Resume. Assert step counter = 101, optimizer state restored, loss trajectory continuous | Optimizer state not saved/loaded with adapter |

### Layer 5: Edge Cases & Adversarial

| ID | Prediction | Test | If Falsified |
|----|-----------|------|-------------|
| F-EDGE-001 | LoRA with rank=1 works | Train 100 steps at r=1, loss decreases | Off-by-one in dimension handling |
| F-EDGE-002 | LoRA with rank=d_model works (full rank) | Train 100 steps at r=896 (Qwen-0.5B), loss decreases | Dimension mismatch when r equals hidden size |
| F-EDGE-003 | LoRA on single layer works | `layers: [0]` — only first layer has adapters. Train, verify only layer 0 adapters change | Layer filtering bug |
| F-EDGE-004 | Empty target_modules fails with clear error | `target_modules: []` — must error at config validation, not crash during training | Validation gap |
| F-EDGE-005 | Invalid target name fails with clear error | `target_modules: ["nonexistent_proj"]` — must error at config validation | Silent ignore of bad config |
| F-EDGE-006 | `lora.enabled: false` means full fine-tuning | All weights must change, optimizer has all params | LoRA code path leaks into non-LoRA training |
| F-EDGE-007 | Gradient accumulation works with LoRA | `gradient_accumulation: 4` — accumulated gradients match 4x larger batch | Accumulation clears LoRA grads incorrectly |
| F-EDGE-008 | Mixed precision (BF16) LoRA works | Train with `dtype: bfloat16`, assert loss decreases | BF16 precision insufficient for LoRA scale factor |

---

## 7. Priority and Dependencies

```
Phase 1 (P0 — MUST FIX):
  ENT-LoRA-001 (wire YAML) ──────────┐
  ENT-LoRA-002 (param groups) ───────┼──> unblocks all non-NF4 LoRA training
  ENT-LoRA-003 (adapter checkpoint) ─┘
  Falsification gate: F-FREEZE-001..008, F-MATH-001..008

Phase 1b (P0 — apr-cli, ships WITH Phase 1):
  ENT-LoRA-014 (apr train LoRA) ─────── depends on ENT-LoRA-001+002
  ENT-LoRA-015 (adapter ckpt flow) ──── depends on ENT-LoRA-003
  ENT-LoRA-016 (finetune plan LoRA) ─── depends on ENT-LoRA-001
  ENT-LoRA-017 (apr merge) ──────────── depends on ENT-LoRA-003
  ENT-LoRA-018 (apr inspect) ────────── depends on ENT-LoRA-003
  Falsification gate: F-CLI-001..010, F-CONV-001..008, F-CKPT-001..005

Phase 2 (P1 — world class):
  ENT-LoRA-004 (rsLoRA) ─────── independent
  ENT-LoRA-005 (all_linear) ─── independent
  ENT-LoRA-006 (LoRA+) ──────── depends on ENT-LoRA-002
  ENT-LoRA-007 (PEFT export) ── depends on ENT-LoRA-003
  Falsification gate: F-CONV-005..006, F-CKPT-002

Phase 3 (P1 — QLoRA complete):
  ENT-LoRA-008 (double quant) ── independent (NF4 path)
  ENT-LoRA-009 (merge export) ── depends on ENT-LoRA-003
  ENT-LoRA-010 (paged optim) ─── stretch goal
  Falsification gate: F-CONV-002+004, F-MATH-008

Phase 5 (P2 — advanced):
  ENT-LoRA-011 (DoRA) ────── depends on Phase 1
  ENT-LoRA-012 (PiSSA) ───── depends on Phase 1
  ENT-LoRA-013 (multi-adapt)── partially exists
  Falsification gate: F-EDGE-001..008
```

### Ship Criteria

Phase 1 + Phase 1b are a **single atomic release**. You cannot ship LoRA without CLI wiring.
The following falsification tests are **mandatory pass** before any LoRA release:

**Hard gates** (block release):
- F-FREEZE-001..003: Base weights frozen on ALL three paths
- F-FREEZE-004..005: LoRA params actually train
- F-CLI-001..003: apr train creates LoRA layers, saves adapter, adapter is small
- F-CONV-001: Loss decreases (LoRA learns something)
- F-CKPT-001: Save/load roundtrip works

**Soft gates** (warn, don't block):
- F-CONV-004: FP16/NF4 parity within 10%
- F-EDGE-007: Gradient accumulation
- F-EDGE-008: BF16 mixed precision

---

## 8. References

### Original Papers
- Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685
- Dettmers et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." arXiv:2305.14314

### Recent Advances
- Liu et al. (2024). "DoRA: Weight-Decomposed Low-Rank Adaptation." ICML 2024 Oral. arXiv:2402.09353
- Hayou et al. (2024). "LoRA+: Efficient Low Rank Adaptation of Large Models." ICML 2024. arXiv:2402.12354
- Meng et al. (2024). "PiSSA: Principal Singular Values and Singular Vectors Adaptation." NeurIPS 2024 Spotlight. arXiv:2404.02948
- Kopiczko et al. (2024). "VeRA: Vector-based Random Matrix Adaptation." ICLR 2024. arXiv:2310.11454
- Kalajdzievski (2023). "Rank Stabilization Scaling Factor for Fine-Tuning (rsLoRA)." arXiv:2312.03732

### Implementation References
- HuggingFace PEFT: github.com/huggingface/peft (canonical Python LoRA)
- Microsoft LoRA: github.com/microsoft/LoRA (original reference)
- Meta torchtune: github.com/meta-pytorch/torchtune (PyTorch-native)
- candle-lora: github.com/EricLBuehler/candle-lora (Rust LoRA for Candle)

### Analysis
- Biderman et al. (2024). "LoRA Learns Less and Forgets Less." arXiv:2405.09673
- Shuttleworth et al. (2024). "LoRA vs Full Fine-tuning: Spectral Analysis." arXiv:2410.21228

### Target Module Selection
- All linear layers > MLP-only > attention-only (Biderman et al.)
- gate_proj has strongest individual effect in MLP
- V projection most influential in attention, followed by Q

### Best Practices (Current Consensus)
- Scaling: rsLoRA (`alpha/sqrt(r)`) for r > 16
- Targets: all linear layers for instruction tuning; Q+V sufficient for classification
- Rank: r=16 general purpose, r=32-64 for complex tasks
- Init: A=Kaiming, B=zeros (standard); PiSSA for faster convergence
- LR: 1e-4 to 3e-4 (higher than full FT); LoRA+ ratio 16 for A/B
- Epochs: 1-3 for static datasets; more causes overfitting
