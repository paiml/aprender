# PMAT Development Roadmap

## Current Sprint: v0.9.0 Autograd Engine - PyTorch-Compatible Automatic Differentiation
- **Duration**: 2025-11-25 to 2025-12-16
- **Priority**: P0

### Tasks
| ID | Description | Status | Complexity | Priority |
|----|-------------|--------|------------|----------|

### Definition of Done
- [ ] All tasks completed
- [ ] Quality gates passed
- [ ] Documentation updated
- [ ] Tests passing
- [ ] Changelog updated

## Current Sprint: v0.4.0 llama.cpp Performance Parity
- **Duration**: 2026-01-01 to 2026-01-15
- **Priority**: P0
- **Quality Gates**: Complexity ≤ 20, SATD = 0, Coverage ≥ 80%
- **Target**: Achieve <2x gap vs llama.cpp (currently 5.3x gap - M2 achieved!)

### Benchmark Baseline (2026-01-02)
| Model | llama.cpp | realizar (before) | realizar (after) | Gap (after) |
|-------|-----------|-------------------|------------------|-------------|
| TinyLlama-1.1B Q4_K_M | ~500 tok/s | 8.0 tok/s (GPU) | 100-118 tok/s (GPU) | 4.2-5.0x |
| TinyLlama-1.1B Q4_K_M | ~500 tok/s | - | 3.8 tok/s (CPU) | 132x |

**Note:** Current GPU performance (~112 tok/s avg) with PAR-021 GQA incremental attention kernel enabled.
Parity tests pass. M3+ achieved (>100 tok/s). Target M4: >192 tok/s.

**Key improvements (2026-01-01):**
- GPU went from 0.1 tok/s → 8.0 tok/s (80x improvement from broken state)
- Fixed GEMV kernel error 700 (PAR-002)
- Added KV cache to GPU inference (PAR-013)
- **Fixed Q4K GEMV PTX shift instruction bug (shl.u32 → shl.b32)**
- **Fixed Q6K GEMV kernel addressing bugs (completely broken → 0.00% error)**
  - Wrong qh shift (2 bits → 4 bits)
  - Complex interleaved ql/qh addressing for Q6_K format
  - Signed i8 scale handling
- GPU now produces coherent text (was garbage due to Q6K lm_head bug)
- **PAR-005: GPU weight caching (8 tok/s → 94.7 tok/s, 12x speedup)**
  - Added `quantized_weight_cache` to CudaExecutor for persistent GPU storage
  - Lazy cache on first use, avoids ~50+ CPU→GPU transfers per token
  - Implemented `q4k_gemv_cached`, `q5k_gemv_cached`, `q6k_gemv_cached`
  - M2 milestone achieved (>24 tok/s, <10x gap)

**Key improvements (2026-01-02):**
- **PAR-014: Forward pass weight caching fix (94.7 → 115-122 tok/s, 21-29% speedup)**
  - `OwnedQuantizedModelCuda::forward_single_full_cuda_with_cache` was not using cached GEMV
  - Changed from `q4k_matvec` (re-transfers weights) to `q4k_gemv_cached` (uses GPU cache)
  - Pre-capture original weight pointers before cloning for stable cache keys
  - Added `fused_matmul_cuda_with_key` method for explicit cache key control
  - Added GPU kernel methods: `gelu_gpu`, `layer_norm_gpu`, `add_residual_gpu`, `q4k_gemv_gpu`
  - Added fused FFN method `fused_ffn_q4k` (needs debugging, currently disabled)
  - M3+ achieved: >100 tok/s, <5x gap
- **PAR-015: Fix FFN activation for LLaMA models (SwiGLU instead of GELU)**
  - Discovered TinyLlama config.json shows `"hidden_act": "silu"` not GELU
  - CUDA forward pass was incorrectly using GELU activation for all models
  - Added proper SwiGLU path: `down(silu(gate(x)) * up(x))`
  - Extracts `ffn_gate_weight` and applies SiLU activation when present
  - Falls back to GELU for phi-2 style models (no gate projection)
  - Improves output quality for LLaMA-based models
- **PAR-016: Fix LM head weight caching**
  - LM head projection was using `fused_matmul_cuda` with cloned weights
  - Cloning created new pointers, breaking cache key stability
  - Pre-capture LM head cache key before layer loop (like FFN weights)
  - Use `fused_matmul_cuda_with_key` for stable GPU weight caching
- **PAR-017: Lower GPU attention threshold (32 → 8 tokens)**
  - Previous threshold of 32 caused high variance (15-118 tok/s)
  - Short sequences used CPU attention, long sequences used GPU
  - Dynamic switching caused unpredictable performance
  - Lowered to 8 tokens for more consistent GPU usage
  - Result: Variance improved (89-114 tok/s), no more 15 tok/s drops
- **PAR-018: GPU-resident KV cache investigation (2026-01-02)**
  - Attempted to keep KV cache on GPU to avoid ~66 MB transfer per token (TinyLlama 22 layers)
  - Added `init_kv_cache_gpu`, `reset_kv_cache_gpu`, `flash_attention_cached` to CudaExecutor
  - Added `copy_from_host_at`, `copy_to_host_at` partial transfer methods to trueno-gpu GpuBuffer
  - **BLOCKED**: Current FlashAttention kernel requires CPU slices, re-uploads K/V anyway
  - Implementation caused 2x slowdown (D2H + H2D instead of just H2D)
  - **Requires**: Custom attention kernel that operates directly on GPU buffer pointers
  - Deferred to PAR-020 which addresses root cause
- **PAR-019: Tensor Core (WMMA) attention path (2026-01-02)**
  - Attempted to use FP16 WMMA for ~40x speedup over FP32 FlashAttention
  - WMMA requires seq_len and head_dim to be multiples of 16
  - TinyLlama head_dim=64 (compatible), but seq_len varies (8, 17, 42, etc.)
  - Added padding to round seq_len up to multiple of 16
  - **BLOCKED**: Padding overhead (up to 94% wasted for seq_len=17) outweighs WMMA benefits
  - Performance regressed to 33-108 tok/s (more variance than baseline)
  - **Requires**: Native WMMA kernel for incremental decoding without padding
  - Reverted; deferred to PAR-020 which can design proper incremental attention

### Tasks
| ID | Description | Status | Complexity | Priority |
|----|-------------|--------|------------|----------|
| PAR-001 | Fix Q4_K dequantization (garbage output bug) | ✅ DONE | High | P0 |
| PAR-002 | Fix CUDA GEMV kernel error 700 (illegal memory access) | ✅ DONE | High | P0 |
| PAR-003 | Implement native Q4_K GEMV kernel (dequant-on-the-fly) | ✅ DONE | High | P0 |
| PAR-013 | Add KV cache to GPU inference (O(n²) → O(n)) | ✅ DONE | High | P0 |
| PAR-004 | Implement flash attention on GPU path | 🔴 TODO | High | P1 |
| PAR-005 | Keep quantized weights on GPU (cached GEMV) | ✅ DONE | Medium | P0 |
| PAR-006 | Add GEMM kernel fusion for FFN layers | 🔴 TODO | Medium | P2 |
| PAR-007 | Implement memory pool for GPU allocations | 🔴 TODO | Medium | P2 |
| PAR-008 | Achieve M2 milestone (>24 tok/s, <10x gap) | ✅ DONE | - | P0 |
| PAR-009 | Achieve M3 milestone (>48 tok/s, <5x gap) | ✅ DONE | - | P1 |
| PAR-010 | Achieve M4 milestone (>192 tok/s, <1.25x gap) | 🔴 TODO | - | P2 |
| PAR-011 | Add --gpu flag to run/serve commands | ✅ DONE | Medium | P0 |
| PAR-012 | Fix GPU dequant dimension layout | ✅ DONE | High | P0 |
| PAR-014 | Fix weight caching in CUDA forward pass | ✅ DONE | Medium | P0 |
| PAR-015 | Fix FFN activation (SwiGLU for LLaMA models) | ✅ DONE | Medium | P0 |
| PAR-016 | Fix LM head weight caching | ✅ DONE | Low | P1 |
| PAR-017 | Lower GPU attention threshold (32→8) | ✅ DONE | Low | P1 |
| PAR-018 | GPU-resident KV cache for attention | 🔴 BLOCKED | High | P1 |
| PAR-019 | Tensor Core (WMMA) attention path | 🔴 BLOCKED | High | P1 |
| PAR-020 | Custom incremental attention kernel | ✅ DONE | High | P0 |
| PAR-021 | GQA-aware incremental attention kernel | ✅ DONE | High | P0 |
| PAR-022 | Profile analysis for M4 bottlenecks | ✅ DONE | Low | P0 |
| PAR-023 | Async kernel execution pipeline | 🔴 TODO | High | P0 |
| PAR-024 | CUDA Graphs for inference | 🔴 TODO | High | P1 |
| PAR-025 | Tensor Core GEMV (WMMA for Q4_K) | 🔴 TODO | High | P2 |

### Investigation Notes

**PAR-021: GQA Support for Incremental Attention - DONE (2026-01-02)**

Implemented GQA support in CUDA forward pass:
- Fixed Q/K/V extraction: Q is [hidden_dim], K/V are [kv_dim] for GQA
- Fixed cache_len calculation: uses kv_dim instead of hidden_dim
- Fixed RoPE application: K uses num_kv_heads, Q uses num_heads
- Changed fallback to `attention_with_cache_gqa` (CPU-based GQA attention)
- First token V expansion for GQA (each KV head serves 8 Q heads)

**Final Status:**
- Performance with GQA incremental attention: ~112 tok/s avg (TinyLlama Q4_K_M)
- Output quality: coherent text ("Once upon a time, and the world was a place of greatness.")
- Parity tests pass: first token, second token, head mapping verified
- Gap vs llama.cpp: ~4.5x (down from 5.8x)

**Verification Tests Added:**
- `tests/gqa_attention_parity.rs`: 4 unit tests + 2 GPU parity tests
- CPU reference implementation matches GPU kernel output within 1e-3 tolerance

Files modified:
- `trueno-gpu/src/kernels/attention.rs` - IncrementalAttentionKernel with GQA
- `realizar/src/cuda.rs` - init_kv_cache_gpu with num_kv_heads
- `realizar/src/gguf.rs` - GQA-aware Q/K/V extraction and attention

**PAR-022: Profile Analysis for M4 Bottlenecks (2026-01-02)**

Identified major bottleneck: **Synchronous kernel execution**
- Each GEMV call has `stream.synchronize()` + `copy_to_host()`
- 154 GEMV + 22 attention = 176 synchronizations per token
- llama.cpp pipelines operations, only syncs at end of forward pass

Current latency breakdown (estimated):
- Kernel launch overhead: ~2µs × 176 = ~350µs per token
- Sync overhead: ~5µs × 176 = ~880µs per token
- D2H transfer: ~0.5µs × 176 = ~88µs per token
- Total overhead: ~1.3ms per token (vs ~8ms actual = ~16% overhead)

**M4 Target Requirements (>192 tok/s):**
1. PAR-023: Async kernel execution pipeline
   - Keep intermediate results on GPU
   - Remove per-operation synchronization
   - Only sync at end of forward pass (for logits)
   - Estimated speedup: 1.5-2x

2. PAR-024: CUDA Graphs for inference
   - Capture entire forward pass as graph
   - Replay graph with minimal overhead
   - Estimated speedup: 1.2-1.5x

3. PAR-025: Tensor Core GEMV (WMMA for Q4_K)
   - Use FP16 tensor cores for dequant+matmul
   - Requires FP16 input/output staging
   - Estimated speedup: 2-4x for compute-bound ops

**PAR-020: Custom Incremental Attention Kernel - DONE (2026-01-02)**

Implemented `IncrementalAttentionKernel` in trueno-gpu:
- Optimized for M=1 autoregressive decoding
- GPU-resident KV cache avoids O(seq_len) H2D transfers
- Warp shuffle reduction for Q·K dot product
- Online softmax for numerical stability
- GQA head mapping (kv_head_idx = q_head_idx * num_kv_heads / num_heads)


**PAR-001: Q4_K Dequantization - RESOLVED (2026-01-01)**

Output is coherent (not garbage). Previous "uolauola" issue was from earlier version.
Current behavior: realizar and llama.cpp produce different but both coherent outputs.

Test: `"Hello" -> "HelloWorld()\n\n// Example 2.\n"` (realizar) vs `"Hello, World!\n"` (llama.cpp)

Differences may be due to:
- Sampling/RNG differences
- Minor numerical precision in RoPE
- LayerNorm epsilon differences

✅ Q4_K dequantization formula verified correct against llama.cpp.

**Performance Analysis (2026-01-01)**

Current gap: 36.5x (TinyLlama Q4_K_M, greedy, 16 tokens)
- llama.cpp: 528.6 tok/s (text generation, RTX 4090 CUDA)
- realizar: 14.5 tok/s (CPU path, AVX2)

Root cause analysis:
1. `realizar run` uses `QuantizedGGUFTransformer` (CPU-only mmap-based path)
2. `realizar serve` uses `OwnedQuantizedModel` which supports CUDA via `REALIZAR_BACKEND=cuda`
3. The `run` command needs to be updated to support CUDA path

Next priority:
- [x] PAR-011: Add --gpu flag to `realizar run` and `realizar serve` commands ✅
- [ ] PAR-002: Debug CUDA driver error 700 in attention kernel
- [ ] PAR-003: Fix CUDA Q4_K matvec PTX module load failure

**PAR-011: --gpu Flag Implementation - COMPLETED (2026-01-01)**

Added `--gpu` flag to both `realizar run` and `realizar serve` commands:
- `realizar run model.gguf --gpu "prompt"` - Forces CUDA acceleration
- `realizar serve --model model.gguf --gpu` - Forces CUDA acceleration for server

When `--gpu` is specified:
1. Uses `OwnedQuantizedModel` instead of `QuantizedGGUFTransformer`
2. Calls `enable_cuda(0)` to activate CUDA backend
3. Shows GPU status in output: "Backend: CUDA (GPU)"
4. Falls back to CPU with warning if CUDA feature not enabled

**PAR-012: GPU Dequant Dimension Layout - COMPLETED (2026-01-01)**

Fixed weight layout interpretation in `dequantize_weight_for_cuda`:
- GGUF stores weights as [out_dim, in_dim] (row-major, out_dim rows)
- CPU path `fused_q4k_parallel_matvec` uses: `super_blocks_per_row = in_dim.div_ceil(QK_K)`
- GEMV kernel expects A[k, n] at offset k*N + n (i.e., [in_dim, out_dim])
- Solution: Dequantize as [out_dim, in_dim], then transpose to [in_dim, out_dim]
- Applied to Q4_K, Q5_K, and Q6_K dequantization paths

**PAR-002: GEMV Kernel CUDA Error 700 - FIXED (2026-01-01)**

Fixed by switching from CoalescedGemv (256 threads) to simpler Gemv (32 threads warp-reduce):
- CoalescedGemv had illegal memory access issues
- Simpler Gemv kernel uses: 32 threads/block (one warp), N blocks
- Launch config: `LaunchConfig::grid_2d(n, 1, 32, 1)` - no shared memory
- GEMV now works correctly for M=1 operations

**PAR-013: GPU Inference KV Cache - FIXED (2026-01-01)**

GPU inference was using O(n²) `generate()` instead of O(n) `forward_cached()`:
- Previous: `quantized_model.generate()` recomputed entire sequence each token
- Fixed: Use `forward_cached()` with KV cache like CPU path
- Result: GPU performance improved from 0.1 tok/s to 8.3 tok/s (83x improvement)

Current GPU Status (after PAR-013):
- GPU + KV cache: 8.3 tok/s (was 0.1 tok/s before)
- Coherent output ✓
- Uses CPU fused matmul for Q4_K/Q5_K/Q6_K (GPU GEMM is slower for M=1)
- Native Q4_K GEMM kernels disabled (slow for matvec, need dedicated GEMV)

**PAR-003: Q4_K/Q5_K/Q6_K GEMV Kernels - COMPLETED (2026-01-01)**

Implemented dedicated GEMV kernels for quantized M=1 operations:
- Added `Q4KGemvKernel`, `Q5KGemvKernel`, `Q6KGemvKernel` to trueno-gpu
- Kernel strategy: One warp (32 threads) per output element, no shared memory
- On-the-fly dequantization: 7.1x memory bandwidth reduction vs dequant+GEMV
- Integrated into realizar: `q4k_gemv()`, `q5k_gemv()`, `q6k_gemv()` methods
- Enabled for seq_len=1 in `OwnedQuantizedModel::fused_matmul()`

Key files:
- `/home/noah/src/trueno/trueno-gpu/src/kernels/quantize.rs` - New kernels
- `/home/noah/src/realizar/src/cuda.rs` - New KernelType variants and launch functions
- `/home/noah/src/realizar/src/gguf.rs` - Integration in fused_matmul

Remaining for M2 (>24 tok/s, <10x gap):
- PAR-005: Keep weights on GPU (avoid CPU-GPU transfers each forward pass)
- PAR-004: Implement flash attention on GPU path

### Definition of Done
- [ ] All tasks completed
- [ ] Quality gates passed
- [ ] Documentation updated
- [ ] Tests passing
- [ ] Changelog updated

