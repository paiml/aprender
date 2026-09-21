# PMAT-3759: CUDA module-cache key audit

**Question (the cop, 2026-09-21):** for every PTX/JIT module cache in `aprender-serve`, is every
parameter baked into the PTX part of the key, or read at runtime? #3727 (the FP8 activation cache)
and #3759 (RMSNorm epsilon) were two cache-key-scope defects found on the same day.

**Invariant:** *same key ⇒ same PTX.* A key that omits a parameter the kernel bakes in as an
immediate hands every later request with a different value the first request's kernel.

## Method

1. **Mechanized, for the `KernelType` sites:** `CudaExecutor::ensure_kernel_module(key, &KernelType)`
   (`cuda/executor/module_key_guard.rs`) is now the compile-on-miss step. In debug builds and every
   `cargo test` build (`cfg(any(debug_assertions, test))`) it records the hash of the PTX compiled
   under each key. A later hit whose request differs from every request already proven equivalent
   is regenerated, and a different hash **panics** naming both. Grid-only differences produce the
   same PTX and pass, so the check has no false positives. There are **203** live module-cache
   sites (218 counted naively, including 3 dead files). A script migrated **154** of them: every
   compile-on-miss idiom site, including the preloads, which is where #3759's 1e-5 came from. The whole `aprender-serve --features cuda` lib suite then ran under the guard (see the
   fragment's MEASURED).
2. **By reading, for the 49 live sites the guard does not cover** (plus the guard's own lookup): the table below.

## Defects found and fixed in this change

| Keys | Baked parameter missing from the key | Effect | Fix |
|---|---|---|---|
| `rmsnorm_{h}`, `rmsnorm_simple_{h}`, `rmsnorm_vectorized_{h}`, `rmsnorm_precise_{h}`, `batched_rmsnorm_vectorized_{h}`, `per_head_rmsnorm_{hd}_{nh}`, `batched_per_head_rmsnorm_{hd}_{nh}`, `fused_residual_rmsnorm_{h}` (×2 sites), `batched_fused_residual_rmsnorm_{h}`, `layernorm_{h}_{b}`, `fused_rmsnorm_gate_up_swiglu_q4k_{h}_{i}` | epsilon (`mov_f32_imm`) | The model-load preload compiled the norm keys at a hardcoded 1e-5, so every model ran RMSNorm at 1e-5. Special tokens with near-zero embeddings (Qwen2.5/Qwen3.5 `<|im_start|>`, mean-square ≈1e-6) came out at 0.431× and the F2 gate rejected the GPU (#3759) | `f32_bits_tag(epsilon)` in every key; preload threads the model's epsilon |
| `fused_rmsnorm_q4k_gemv_{k}_{n}_{:.0e}` | epsilon, but **rounded** (`{:.0e}`: 1.5e-6 and 2e-6 collide) | latent | exact bits |
| `rope_{nh}_{hd}`, `rope_indirect_…`, `rope_precise_indirect_…`, `rope_neox_…`, `rope_neox_indirect_…`, `batched_rope_{nh}_{hd}`, `batched_rope_neox_{nh}_{hd}`, `gdn_partial_neox_rope_{nh}_{hd}_{n_rot}` | theta / theta_scale (`mov_f32_imm(theta.log2())`) | Masked today: one theta per process, and the model-load preload runs after `set_rope_theta`. A module compiled before that (at the 10000 default), or a second model, would rotate at the wrong frequency | theta bits in every key (preload and launch both read `self.rope_theta`, so no new compiles in normal operation) |
| `"fused_prefill_attn"` (a constant key) | `head_dim`, `heads_per_kv`, and the `1/√head_dim` scale | Masked by one model per process; a second model with another head_dim or GQA ratio would get the first model's kernel | `fused_prefill_attn_{head_dim}_{heads_per_kv}` |

## The 49 unguarded live sites, plus the guard's own lookup (14 + 1 + 5 + 9 + 8 + 11 + 1 + 1 = 50)

| Class | Sites | Verdict |
|---|---:|---|
| Constant PTX under a constant key: `cublas_prefill/mod.rs` (`f32_to_f16`, `f32_to_e4m3`, `f16_to_f32`, `bf16_to_f32_act_scaled`, `f32_to_e4m3_scaled`, `absmax_reduce` ×2, `f32_to_e4m3_device_scaled`), `cublas_prefill/attention.rs` (`causal_mask_softmax` ×3, `scatter_packed_kv` ×3) | 14 | Complete by construction: nothing is baked, since the PTX is a `const` |
| `fused_prefill_attn` | 1 | **Was incomplete; fixed above** |
| KV scatter: `kv_scatter_{nkv}_{hd}` (×3), `kv_scatter_indirect_…`, `batched_kv_scatter_{nkv}_{hd}_{max_len}` | 5 | Complete. `max_len`/`head_dim` are **runtime params** (`load_param_u32` in `kernels/elementwise/kv_cache.rs`); the batched one keys every value its hand PTX formats in |
| Incremental / multi-warp / batched attention: `incremental_attention_{max}_{hd}_{nh}_{nkv}` (×4, all `indirect: false`; the indirect variant has its own key and is guarded), `multi_warp_attention_…_{warps}`, `batched_incr_attn_…_{m}`, `tensor_core_attn_{seq}_{hd}_{nh}_{causal}`, `multi_head_attn_…`, `flash_attn_{seq}_{hd}_{causal}` | 9 | Complete: every constructor argument is in the key (`KernelType::Attention` has no head-count field) |
| Flash decoding: `flash_decode_chunk_{max}_{hd}_{nh}_{nkv}` (×4), `flash_decode_reduce_{hd}_{nh}` (×4) | 8 | Complete. The chunk kernel's fifth argument (`batch_size`) is a grid dimension: `let _batch_size = self.batch_size;` is unused in emission |
| `argmax_{vocab}`, `argmax_final_{blocks}` (×2 each), `gemv_{k}_{n}` / `gemm_{m}_{n}_{k}_32`, `gemv_simple_{k}_{n}`, `gemm_opt_{m}_{n}_{k}_{tile}`, `q8_dequant_{elements_per_head}`, `batched_bias_bcast_{dim}`, `batched_rope_neox_{nh}_{hd}_{theta}` (hand PTX, theta-keyed above), `rmsnorm_into`'s key (a GH-559 print sits inside the compile block; epsilon-keyed above) | 11 | Complete: each key names every field of its `KernelType`/format. `par-062.rs` and `gemm_tiled.rs` keep the old idiom (whole files) because a field borrow is held across the block, and `ensure_kernel_module` needs `&mut self` |
| `gdn_ops.rs` `gdn_prepare` helper (per-head L2 norm and gated RMSNorm keys print epsilon with `{:e}`, shortest round-trip, so they are exact; partial NeoX RoPE is now theta-keyed) | 1 | Complete |
| The guard itself | 1 | — |

**Not in scope, recorded:** `layers/loading.rs` and `layers/preload_utilities.rs` are an older copy of
the preload path that nothing `include!`s (`preload_modules.rs` → `modules_utilities.rs` is the live
one), and `executor/kernel.rs` likewise. They still hardcode 1e-5 and are dead code; deleting them
is its own change.

## Where the guard runs

`cfg(any(debug_assertions, test))`: every `cargo test -p aprender-serve --features cuda --lib` run,
release or debug. In CI that is `ci.yml` job `cuda-unit`, step "aprender-serve cuda unit tests",
which runs `--lib --release gguf::cuda::`. The `gguf::cuda::` filter does **not** select the
`cuda::executor::` rows added here (`rmsnorm_eps_tests_3759`), and they need a GPU. Widening that
filter, or running `cuda::executor::` in `cuda-nightly` on gx10, is a workflow change and is left to
the release owner. Until one of them runs it, the backstop on the release path is the ladder's F2
fallback cell. A shipped (release, non-test) library does only the lookup.
