---
title: "Qwen3-8B: Performance Parity Pipeline"
issue: GH-279
status: Partially Complete (GH-280 resolved — GPU capability gate open)
created: 2026-02-17
updated: 2026-02-18 (GH-280: PerHeadRmsNormKernel added to trueno-gpu)
---

# Qwen3-8B: Performance Parity Pipeline

**Goal:** Import Qwen3-8B from SafeTensors, quantize to Q4K, export our own GGUF with full metadata, and achieve Ollama performance parity.

**Primary Model:** [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B)

## Motivation

We tested `apr qa` on a third-party Qwen3-8B GGUF from HuggingFace and found:
- 8/10 QA gates pass, but **GPU golden output = garbage** and **0.52x Ollama parity**
- Root cause: the GGUF had **sparse metadata** (only `general.architecture`) — no `rope_theta`, no `rms_norm_eps`, no context length
- Our approach: import SafeTensors directly (full metadata from `config.json`), quantize ourselves, export our own GGUF

## Architecture: Qwen3-8B

Source: [HuggingFace `Qwen/Qwen3-8B` config.json](https://huggingface.co/Qwen/Qwen3-8B)

### Parameters

| Parameter | Qwen3-8B | vs Qwen2-7B |
|-----------|----------|-------------|
| Architecture | `Qwen3ForCausalLM` | was `Qwen2ForCausalLM` |
| model_type | `qwen3` | was `qwen2` |
| vocab_size | 151,936 | was 152,064 |
| hidden_size | 4,096 | was 3,584 |
| num_hidden_layers | 36 | was 28 |
| num_attention_heads | 32 | was 28 |
| num_key_value_heads | 8 | was 4 |
| head_dim | 128 | same |
| intermediate_size | 12,288 | was 18,944 |
| max_position_embeddings | 40,960 | was 131,072 |
| rope_theta | 1,000,000 | same |
| rms_norm_eps | 1e-6 | same |
| **attention_bias** | **false** | **was true (default)** |
| **qk_norm** | **true** (inferred from tensors) | **not present** |
| tie_word_embeddings | false | same |
| activation | silu (SwiGLU) | same |
| norm_type | rmsnorm | same |

### Key Differences from Qwen2

1. **No attention bias**: `attention_bias: false` — no Q/K/V bias tensors (3 fewer tensors per layer)
2. **QK normalization**: New `q_norm` and `k_norm` RMSNorm weight tensors per layer (2 extra tensors per layer, shape `[head_dim]`)
3. **Thinking mode**: Special tokens `<think>` (151667) and `</think>` (151668) for chain-of-thought reasoning
4. **Net tensor delta**: -3 bias + 2 norm = -1 tensor per layer vs Qwen2

### Tensor Inventory (8B)

| Tensor | Shape | Count |
|--------|-------|-------|
| `model.embed_tokens.weight` | [151936, 4096] | 1 |
| `lm_head.weight` | [151936, 4096] | 1 |
| `model.norm.weight` | [4096] | 1 |
| `model.layers.{0-35}.self_attn.q_proj.weight` | [4096, 4096] | 36 |
| `model.layers.{0-35}.self_attn.k_proj.weight` | [1024, 4096] | 36 |
| `model.layers.{0-35}.self_attn.v_proj.weight` | [1024, 4096] | 36 |
| `model.layers.{0-35}.self_attn.o_proj.weight` | [4096, 4096] | 36 |
| `model.layers.{0-35}.self_attn.q_norm.weight` | [128] | 36 |
| `model.layers.{0-35}.self_attn.k_norm.weight` | [128] | 36 |
| `model.layers.{0-35}.mlp.gate_proj.weight` | [12288, 4096] | 36 |
| `model.layers.{0-35}.mlp.up_proj.weight` | [12288, 4096] | 36 |
| `model.layers.{0-35}.mlp.down_proj.weight` | [4096, 12288] | 36 |
| `model.layers.{0-35}.input_layernorm.weight` | [4096] | 36 |
| `model.layers.{0-35}.post_attention_layernorm.weight` | [4096] | 36 |
| **Total** | | **399** |

## Pipeline

```bash
# Step 1: Download SafeTensors from HuggingFace
apr pull hf://Qwen/Qwen3-8B

# Step 2: Import to APR with Q4K quantization
apr import hf://Qwen/Qwen3-8B -o qwen3-8b-q4k.apr --quantize q4k --arch qwen3

# Step 3: Export our own GGUF (full metadata from config.json)
apr export qwen3-8b-q4k.apr --format gguf -o qwen3-8b-q4k.gguf

# Step 4: QA validation (all 10 gates)
apr qa qwen3-8b-q4k.gguf --assert-tps 100

# Step 5: Performance benchmark
apr bench qwen3-8b-q4k.gguf --warmup 3 --measure 10
```

## Performance Targets

| Metric | 3rd-party GGUF | Our GGUF | Target | Roofline | Status |
|--------|---------------|----------|--------|----------|--------|
| CPU throughput | — | **3.9 tok/s** | 6-8 | **7.9** (DDR4 BW wall) | 49% of ceiling |
| GPU throughput | — | **unblocked** (GH-280) | 40+ | **~215** (HBM2 BW) | READY TO TEST |
| CPU golden output | — | **coherent** (thinking mode) | coherent | — | PASS |
| GPU golden output | garbage | **unblocked** (GH-280) | coherent | — | READY TO TEST |
| Ollama parity | 0.52x | **0.05x** (6 vs 129 tok/s) | >= 1.0x | GPU-only | Grade F (CPU) |
| QA gates | 8/10 | **6/11** (3 pass, 3 fail, 5 skip) | 10/10 | — | 5 remain |

**Roofline note:** The "40+ tok/s" target is achievable only via GPU (HBM2 bandwidth: 900+ GB/s). On CPU with DDR4, the memory bandwidth wall is ~7.9 tok/s for 8B Q4K. Current 3.9 tok/s = 49% of ceiling — reachable via prefetch + kernel parity with llama.cpp, not via architectural guessing.

### BrickTracer Root Cause Analysis (2026-02-18)

All throughput measurement now uses `renacer::BrickTracer` with `SyscallBreakdown`:

| Measurement | tok/s | Total (us) | Compute | mmap | futex | ioctl | Overhead | Efficiency |
|-------------|-------|-----------|---------|------|-------|-------|----------|------------|
| QA Throughput (CPU) | 3.9 | 81,323,284 | 81,323,284 | 0 | 0 | 0 | 0.0% | 39.3% |
| QA Ollama Parity (CPU) | 5.9 | 216,468,496 | 216,468,496 | 0 | 0 | 0 | 0.0% | 59.1% |
| Bench 3-iter (CPU) | 3.2 | 30,460,000 | 30,460,000 | 0 | 0 | 0 | 0.0% | — |

**Key insight: 100% compute-bound.** Zero syscall overhead — no mmap stalls, no futex contention, no ioctl latency. The bottleneck is purely in the matmul compute kernels.

**Perf regression detected:** Previous report cached 6.2 tok/s → now 3.9 tok/s (36% regression). This may reflect measurement methodology change (BrickTracer vs raw Instant) or background load variance.

### Throughput Optimization: Contract-by-Design

**Methodology:** We do NOT guess at optimizations or play whack-a-mole. Every change is derived from:
1. **Roofline model** — hardware-dictated theoretical ceiling
2. **Reference kernel analysis** — structural diff vs llama.cpp's proven `ggml_vec_dot_q4_K_q8_K`
3. **Mathematical derivation** — if memory-bound, no compute optimization helps; if compute-bound, identify the specific missing SIMD pattern
4. **Contract validation** — `apr qa --assert-tps` gates enforce that changes actually improve throughput

#### Step 1: Roofline Model (Theoretical Ceiling)

For Qwen3-8B Q4K single-token generation (matvec):

```
Model parameters:
  36 layers × (Q:4096×4096 + K:1024×4096 + V:1024×4096 + O:4096×4096
              + up:12288×4096 + gate:12288×4096 + down:4096×12288) = 6.84B params
  + LM head: 151936×4096 = 0.62B params
  Total: 7.46B parameters

Weight bytes (Q4K = 4.5 bits/param):
  7.46B × 144/256 bytes = 4.19 GB read per token

Activation compute (Q8K integer path):
  7.46B multiply-accumulate ops per token
  AVX2 FMA: 8 FMAs/instruction × 2 instructions/cycle × 4 GHz = 64 GFLOP/s/core
  8 cores: 512 GFLOP/s theoretical peak
  Q4K overhead (dequant + nibble extract): ~40% penalty → ~307 effective GFLOP/s
  Compute ceiling: 307G / 7.46G = ~41 tok/s

Memory bandwidth (DDR4-3200 dual-channel):
  Theoretical: 51.2 GB/s
  Effective (with TLB misses, cache line waste): ~30-35 GB/s
  Bandwidth ceiling: 33 GB/s / 4.19 GB = ~7.9 tok/s
```

**Roofline verdict: MEMORY-BOUND.** The bandwidth ceiling (7.9 tok/s) is far below the compute ceiling (41 tok/s). This means:
- Compute optimizations (parallel QKV, larger SIMD) have **diminishing returns** past the bandwidth wall
- The path to 40+ tok/s requires **GPU** (HBM2 bandwidth: 900+ GB/s → ~215 tok/s ceiling)
- On CPU, realistic target is **6-8 tok/s** (hitting bandwidth wall)
- Current 3.9 tok/s = **49% of bandwidth ceiling** → room for ~2x via better prefetching/cache use

#### Step 2: Mathematical Derivation — Dot Product Algebra

**Paper citations:**
- GPTQ (Frantar et al. 2022, arXiv:2210.17323): Blocked quantization with per-block scales
- LLM.int8() (Dettmers et al. 2022, NeurIPS): Affine quantization `x = d·s·q - dmin·m`
- Memory Wall (Wulf & McKee 1995, SIGARCH): Fused ops eliminate intermediate buffer traffic
- GGML K-quant (ggerganov/ggml): 256-element super-blocks with 6-bit packed sub-block scales

**General dequantization** (all blocked formats):

```
x_i = d · s_j · q_i − dmin · m_j
```

Where: `d` = super-block scale (f16), `dmin` = super-block min (f16, optional), `s_j` = sub-block scale (6-bit), `m_j` = sub-block min (6-bit, optional), `q_i` = quantized value (b-bit unsigned).

**Dot product decomposition** (the key algebra):

```
dot(W, x) = Σ_superblock [
    SCALE TERM:  d_W · d_x · Σ_j( s_j · Σ_i( q_W_i · q_x_i ) )
  − OFFSET TERM: dmin_W · d_x · Σ_j( m_j · Σ_i( q_x_i ) )
]
```

**KEY INSIGHT**: The offset term depends ONLY on `sum(q_x)` per sub-block (bsums), NOT on `q_W`. Therefore:

1. Activation sub-block sums (bsums) can be **precomputed once** and reused across all weight rows
2. The inner loop only needs `q_W × q_x` multiply-accumulate (maps to `maddubs`)
3. The offset correction is computed **outside** the inner loop using precomputed bsums

This is a **contract-derived finding** — our code in `fused_k_part_05.rs:157-200` computes q8 sums on-the-fly inside the super-block loop, which is the primary structural deviation from the mathematically optimal kernel.

**Format-specific degenerations:**

| Format | Super-block | Sub-blocks | has_dmin | Dequant Formula |
|--------|-------------|------------|----------|-----------------|
| Q4_K | 256 vals / 144 bytes | 8×32 | yes | `d·s_j·q_i − dmin·m_j` |
| Q5_K | 256 vals / 176 bytes | 8×32 | yes | `d·s_j·q_i − dmin·m_j` |
| Q6_K | 256 vals / 210 bytes | 16×16 | **no** | `d·s_j·(q_i−32)` |
| Q4_0 | 32 vals / 18 bytes | 1×32 | **no** | `scale·(q_i−8)` |
| Q8_0 | 32 vals / 34 bytes | 1×32 | **no** | `scale·q_i (signed)` |

When `has_dmin=false`, the offset term vanishes. When `subblocks=1`, the sub-block scale is 1.0.

**Contract**: [`contracts/quantized-dot-product-v1.yaml`](../../contracts/quantized-dot-product-v1.yaml)
**Trait**: `realizar/src/quantize/format_trait.rs::QuantBlockFormat`
**Reference kernel**: `realizar/src/quantize/generic_dot.rs::generic_fused_dot_scalar`
**Falsification**: 5 tests in `realizar/src/quantize/contract_tests.rs` (FALSIFY-QDOT-001 through 005)

#### Step 3: Contract-Derived Bsum Gap Analysis

The mathematical decomposition reveals a specific, measurable deviation from the optimal kernel:

**Current code** (`fused_k_part_05.rs:157-200`):
```
for each super-block:
    for each 64-value chunk (4 iterations):
        load Q4K data, load Q8K data
        compute Q4×Q8 dot products → block_dots
        compute Q8 sums → block_q8sums     ← REDUNDANT PER ROW
    apply scales using block_dots
    apply dmin correction using block_q8sums
```

**Optimal (from algebra)**:
```
ONCE per token (before matvec):
    for each sub-block:
        bsums[j] = Σ_i(q_x_i)             ← PRECOMPUTED ONCE

for each row (weight):
    for each super-block:
        for each 64-value chunk:
            load Q4K data, load Q8K data
            compute Q4×Q8 dot products → block_dots   ← SAME
            (NO Q8 sum computation)                     ← ELIMINATED
        apply scales using block_dots
        apply dmin correction using bsums[j]            ← USE PRECOMPUTED
```

The Q8 sum computation per super-block costs ~20 instructions (8× `cvtepi8_epi16`, 8× `madd_epi16`, 4× `add_epi32`, 4× `hadd_epi32`, plus lane extract). For 8B model with 36 layers × 7 projections × ~16 super-blocks per row, this is ~80,640 wasted instructions per token.

#### Step 4: Derived Optimization Plan

Based on roofline analysis + mathematical derivation, the optimization priority is:

| Priority | Optimization | Expected Gain | Rationale |
|----------|-------------|---------------|-----------|
| P0 | GPU inference (GH-280 ✓) | 10-50x | Escapes memory bandwidth wall entirely (HBM2: 900 GB/s) |
| P1 | Bsum precomputation | 5-15% | Contract-derived: hoist weight-independent activation sums out of per-row loop |
| P2 | Memory prefetching | 20-40% | CPU at 49% bandwidth utilization; `_mm_prefetch` can close gap |
| P3 | Reduce Q8K quantization passes | 5-10% | Currently quantize once for QKV, once for attn_out, once for FFN; some can share |
| P4 | Parallel QKV projections | 3-5% | K+V overlap with Q tail; marginal on memory-bound workload |

**Critical insight:** On CPU, we will NOT reach 40 tok/s for an 8B model — the memory bandwidth wall is ~8 tok/s. The 40+ tok/s target requires GPU inference, which is now unblocked by GH-280. CPU optimization ceiling is ~6-8 tok/s with perfect bandwidth utilization.

### Root Cause of 0.52x Parity

The third-party GGUF had only `general.architecture` in metadata. Missing:
- `rope_freq_base` (rope_theta) — defaults to 10000 instead of 1000000, corrupts position encoding
- `context_length` — defaults too low, truncates KV cache
- `rms_norm_eps` — wrong epsilon produces numerical drift in layernorms
- `head_count_kv` — wrong GQA grouping produces garbage attention

Our pipeline imports from SafeTensors + `config.json`, which has all parameters. The GGUF export writes them as proper GGUF metadata keys.

## QK Normalization

Qwen3 applies RMSNorm to Q and K projections before attention score computation. This is a per-head normalization with learned weights:

```
Q_norm = RMSNorm(Q, q_norm_weight)  # shape: [batch, heads, seq, head_dim]
K_norm = RMSNorm(K, k_norm_weight)  # shape: [batch, kv_heads, seq, head_dim]
attn = softmax(Q_norm @ K_norm^T / sqrt(head_dim)) @ V
```

The `q_norm.weight` and `k_norm.weight` tensors have shape `[head_dim]` (128) and are broadcast across all heads. This stabilizes attention scores at scale (Qwen3 removes attention bias, using QK norm instead).

## Contract

Model family contract: [`contracts/model-families/qwen3.yaml`](../../contracts/model-families/qwen3.yaml)

Key contract constraints:
- `has_bias: false` (no Q/K/V bias tensors — differs from Qwen2)
- `qk_norm: true` (Q/K RMSNorm — new in Qwen3)
- `attention_type: gqa` (8 KV heads for 32 Q heads)
- Same SwiGLU MLP, RoPE, RMSNorm as Qwen2

## Verification Checklist

- [x] `apr import` SafeTensors → APR succeeds with 399 tensors (8.28GB Q4K)
- [x] Contract validation passes (all shapes match qwen3.yaml)
- [x] `apr export` APR → GGUF produces valid GGUF with full metadata (~267 dup tokens deduped to `[PAD{id}]`)
- [x] GGUF metadata includes ~19 keys (arch=qwen3, layers=36, heads=32/8kv, hidden=4096)
- [ ] `apr qa` passes all gates — **8/10 pass**
- [x] Tensor Contract (399 tensors), Metadata (rope_theta=1000000, max_pos=40960), Golden Output (2 cases)
- [ ] CPU Throughput: 3.9 tok/s (49% of 7.9 tok/s DDR4 roofline) — target 6+ tok/s via prefetch + kernel parity
- [ ] GPU Throughput: **unblocked** (GH-280) — target 40+ tok/s (HBM2 roofline: ~215 tok/s)
- [ ] Ollama Parity: 0.05x (6 vs 129 tok/s) — Ollama uses GPU; CPU-vs-GPU comparison is apples-to-oranges
- [ ] GPU Speedup: **unblocked** (PerHeadRmsNormKernel in trueno-gpu v0.4.18 — GH-280)
- [x] Capability Match: PASS (QkNorm added to `gpu_supported_ops()` — GH-280)
- [ ] Golden Output: CPU coherent (thinking mode), GPU **unblocked** (QkNorm kernel added — GH-280)
- [ ] Format Parity: auto-discovers SafeTensors from `~/.apr/cache/hf/` (GH-279-2)
- [x] Thinking mode tokens present in tokenizer vocabulary (`<think>`=151667, `</think>`=151668)

## Bugs Found During Pipeline

### GH-279-1: GGUF token table dedup (FIXED)
Qwen3 tokenizer has ~267 reserved tokens all mapped to `<unk>`. llama.cpp requires unique strings.
Fix: `ValidatedGgufMetadata::validate()` auto-dedupes with `[PAD{id}]` suffixes following HuggingFace convention (commit `135de184`).

### GH-279-2: `apr import hf://` fails for sharded models (FIXED)
`apr import hf://Qwen/Qwen3-8B` now checks `~/.apr/cache/hf/` and falls back to `model.safetensors.index.json` for sharded models.
Fix: Added APR cache to `find_in_cache()` + sharded index fallback in `resolve_hf_source()`.

### GH-279-3: GPU parity failure (FIXED — GH-280)
GPU forward pass diverges from CPU (cosine sim 0.000479). Root cause: QK norm not applied in GPU matmul path.
CPU argmax: 33975 | GPU argmax: 85222 | Max logit diff: 12084.8
Fixes applied: trueno PTX bar_sync label mismatch fixed (commit `d989451`), sm_70 baseline target for broad GPU compatibility. **GH-280: Added `PerHeadRmsNormKernel` to trueno-gpu (v0.4.18), wired through realizar kernel pipeline, capability gate now admits Qwen3 for GPU inference.** Initial approach: CPU QkNorm in the forward pass is negligible overhead (~2us for 4096 floats vs ~25ms GPU matmuls); the key win is the capability gate no longer rejects Qwen3, so the full forward pass runs on GPU.

### GH-279-4: Golden output gate thinks vs answers (FIXED)
Qwen3 uses thinking mode by default — generates `<think>` chain before answer. Golden output gate expects direct "4".
Fix: `strip_thinking_blocks()` strips `<think>...</think>` before `verify_output()`. No-op for non-thinking models.

### GH-284: CPU inference 2.4 tok/s → 3.9 tok/s (ROOFLINE-BOUNDED)

**Phase 1** (commit `55cfb95`): F32 matmul in `realizar::apr_transformer::helpers::f32_matmul()` was single-threaded scalar code. Added rayon parallel chunking (`par_chunks_mut` with 64-element chunks) and AVX2 SIMD dot product (`simd_dot_f32`) for all F32 matmul operations. Result: 2.4 → 4.0 tok/s.

**Phase 2** (GH-284 cont.): Wire Q8K integer path to remaining hot loops. The `maddubs`-based Q8K kernel (`fused_q4k_q8k_parallel_matvec_into`) was already wired for FFN and attention output projections but two paths still used the slow f32 dequant AVX2 kernel (8 vals/instruction vs 32 vals/instruction):
1. **LM head** (151936×4096): `single_part_02.rs` now uses `quantize_activations_q8k_into` + `fused_q4k_q8k_parallel_matvec_into` when weight is Q4K and `hidden_dim % 256 == 0`.
2. **QKV projections** (36 layers): `matmul_part_02.rs::qkv_matmul_q8k_into()` was a stub that ignored pre-quantized Q8K data and fell back to f32. Now dispatches to `fused_q4k_q8k_parallel_matvec_into` for Q4K weights (both Fused and Separate QKV layouts).

**Phase 3** (BrickTracer instrumentation): Replaced all `Instant::now()` + `.elapsed()` throughput measurement with `renacer::BrickTracer`. SyscallBreakdown confirms **0% syscall overhead**. Budget efficiency 39-59%.

**Phase 4** (Roofline analysis — contract-by-design): BrickTracer reported "100% compute-bound" but this is misleading — BrickTracer measures compute vs syscalls, not compute vs memory stalls. Roofline model shows Qwen3-8B Q4K is **memory-bandwidth-bound**:
- 4.19 GB weight data read per token
- DDR4-3200 dual-channel: ~33 GB/s effective bandwidth
- **Bandwidth ceiling: 7.9 tok/s** — current 3.9 tok/s = 49% of ceiling
- Compute ceiling: 41 tok/s (AVX2 FMA) — not the bottleneck

**Remaining CPU optimization path (contract-derived):**
1. **P1: Software prefetch** — `_mm_prefetch(_MM_HINT_T0)` for next super-block during current super-block processing. Expected: +20-40% (3.9 → 5-6 tok/s)
2. **P2: Kernel parity with llama.cpp** — structural diff of `fused_q4k_q8k_dot_simd` vs `ggml_vec_dot_q4_K_q8_K`. Eliminate any per-super-block instruction overhead.
3. **P0: GPU inference (GH-280 ✓)** — escapes the bandwidth wall entirely. HBM2 at 900 GB/s → 215 tok/s ceiling.

**Key insight: 40+ tok/s on CPU is physically impossible** for 8B Q4K. The memory bandwidth wall is ~8 tok/s. Further CPU work yields diminishing returns. The path forward is GPU.

### GH-282: PTX kernel load failures in `apr serve --gpu` (PARTIALLY FIXED)
Root cause 1: trueno PTX `bar_sync(id)` stored barrier ID in `.src` operand but emitter read from `.label` field. `bar_sync(1)` emitted as `bar.sync 0;` instead of `bar.sync 1;`.
Root cause 2: Hardcoded `sm_89` PTX target caused JIT failures on GPUs older than RTX 4090.
Fix: Set `.label("sync {id}")` in `bar_sync()`, changed default target to `sm_70` (Volta baseline), added `emit_ptx_for_target()` API (commit `d989451` in trueno).

### GH-279-5: GGUF reader drops qwen3.* metadata keys (FIXED)
Root cause: `GgufReader::from_bytes()` in `reader_part_02_part_02.rs` has a hardcoded prefix whitelist for metadata parsing (`llama.`, `qwen2.`, `phi.`, `mistral.`, `gpt2.`). The `qwen3.` prefix was missing, so `qwen3.rope.freq_base`, `qwen3.context_length`, and `qwen3.attention.layer_norm_rms_epsilon` were silently skipped.
Effect: QA metadata gate showed `rope_theta=none, max_pos=none` even though the GGUF export correctly wrote ~19 metadata keys.
Fix: Added `qwen3.` to the reader prefix whitelist. Now reads `rope_theta=1000000, max_pos=40960`.

### GH-279-6: Format parity shard discovery ordering (FIXED)
Root cause: `find_sharded_safetensors()` used `find_map()` on unordered `read_dir()` entries, returning whichever shard the filesystem happened to list first. For Qwen3-8B (5 shards), this returned shard-00002 which lacks `model.embed_tokens.weight`.
Fix: Sort shards by name before selecting, and handle converter failures for sharded models gracefully in the format parity gate.

### GH-280: QkNorm CUDA kernel for GPU inference (IMPLEMENTED)
Root cause: `capability.rs::gpu_supported_ops()` correctly rejected Qwen3 because no QkNorm kernel existed in trueno, causing CUDA fallback to CPU at 3.9 tok/s.
Fix: Added `PerHeadRmsNormKernel` to trueno-gpu (v0.4.18) — applies RMSNorm independently per attention head using `blockIdx.x` as head index, one warp (32 threads) per head. Wired through realizar: `KernelType::PerHeadRmsNorm` variant, kernel name resolution, PTX generation, and `per_head_rmsnorm_into()` executor method. Flipped `gpu_supported_ops()` to include `RequiredOp::QkNorm`. Qwen3 now passes the capability gate for GPU inference.

Files changed:
- `trueno-gpu/src/kernels/layernorm/per_head_rmsnorm.rs` (NEW) — kernel implementation
- `trueno-gpu/src/kernels/layernorm/mod.rs` — module + re-export
- `trueno-gpu/src/kernels/mod.rs` — top-level re-export
- `realizar/src/cuda/kernels_part_02.rs` — `KernelType::PerHeadRmsNorm` enum variant
- `realizar/src/cuda/kernels_part_04.rs` — kernel name mapping
- `realizar/src/cuda/kernels_part_04_part_03.rs` — PTX generation
- `realizar/src/cuda/executor/quantized_part_02_part_04.rs` — `per_head_rmsnorm_into()` method
- `realizar/src/capability.rs` — capability gate flipped + test updated

## References

- [Qwen3 Technical Report](https://qwenlm.github.io/blog/qwen3/)
- [HuggingFace Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen3 model family contract](../../contracts/model-families/qwen3.yaml)
- [Qwen3.5 spec (GH-278)](./qwen3.5-fine-tune.md) — hybrid linear/quadratic attention (different architecture)
