# apr CUDA decode "stall" — root cause: FP8 JIT warmup OOB read (PMAT-082) — 2026-06-15

Host: noah-Lambda-Vector (RTX 4090, sm_89). Binary: apr v0.49.1 --features cuda.
Model: qwen2.5-coder-1.5b-instruct-q4_k_m.gguf (hidden=1536, heads=12, kv_heads=2, head_dim=128).

## Failure signature (clean vs stalled)
Clean run (~5s, 60-80 tok/s): startup logs PMAT-082 "cuBLASLt FP8 JIT warmed",
PMAT-053 "FP8 weight cache: 197 matrices cached", then decode on the CUDA graph path.

Stalled run (~13-24s, ~6-13 tok/s), exact cascade:
  [PMAT-053] FP8 cache warmup failed (non-fatal): CUDA stream synchronization failed:
             CUDA driver error: CUDA_ERROR_ILLEGAL_ADDRESS (code: 700)
  [GH-181]   Workspace reinit failed (non-fatal): ... CUDA_ERROR_ILLEGAL_ADDRESS (code: 700)
  [PAR-054]  Workspace not ready, using non-graphed path
  [CUDA-FAILFAST] Context poisoned during executor lifetime ...
  Backend: wgpu (Vulkan)
  [PMAT-333] Dequantizing 28 layers ... 6174.9 MB F32       <- wgpu fallback
  [apr-cpu-vs-gpu-output-parity-v1] wgpu path rejected, cosine vs CPU = 0.884301 (<0.99)  <- CPU fallback
NOTE: the illegal address happens at STARTUP (FP8 warmup), before any decode token —
NOT seq_len/graph-replay/EOS. The "stall" is the slow wgpu->CPU fallback path.

## Localized stage / kernel
warmup_fp8_cache() -> PMAT-082 cuBLASLt FP8 JIT warmup GEMM
file: crates/aprender-serve/src/cuda/executor/layers/cublas_prefill/attention.rs (~L1419-1464)

## Root cause (verified)
The warmup picked an ARBITRARY cached weight via fp8_weight_cache.values().next()
(non-deterministic HashMap iteration order) but HARDCODED the GEMM dims to
hidden_dim x hidden_dim (1536x1536). With GemmOp::Trans + lda=k the weight operand is
read as [k x n] = 1536*1536 = 2.36 MB of FP8. The GQA K/V projection weights are only
kv_dim*hidden = 256*1536 = 393 KB. When HashMap order happened to return a K/V weight,
the GEMM read ~1.97 MB past the buffer end -> CUDA_ERROR_ILLEGAL_ADDRESS -> context
poison -> wgpu/CPU fallback. HashMap order varies per process => the ~1-in-6 intermittency.

## Evidence (this box, RTX 4090)
- Baseline (v0.49.1 stock binary, varied prompts, 256 tok): 2/24 POISONED (~8%).
- APR_SKIP_FP8_WARMUP=1 (skips ONLY the warmup GEMM, keeps FP8 cache): 0/48 POISONED.
  => isolates poison to the PMAT-082 warmup GEMM, not FP8 caching or decode.
- Patched binary: 0/48 POISONED over two batches; decode 75-80 tok/s retained.
- DIRECT PROOF of HashMap-order trigger: patched warmup log shows the chosen weight's
  dim VARYING per run: 1536x1536 (attn q/o), 3696x3696 (FFN 8960x1536), and 624x624
  (the 256x1536 K/V weight — the exact one that OOB'd at 1536x1536 before the fix).

## Fix (low-risk)
Derive the warmup square dim from the CHOSEN weight's actual byte length
(1 byte/elem FP8): dim = floor(sqrt(len)) & !15, max 16. Read can never exceed the
buffer regardless of which weight HashMap returns. cuBLASLt JIT is per-handle not
per-shape, so any valid shape warms it equally.
Branch: fix/fp8-warmup-oob-poison-pmat082

## Relation to prior memory
- FUSION-003 1-in-6 (project_fusion_003_1_5b_falsified): SAME error code, but that was
  only when FUSION-003 was WIRED (now reverted). This is a SEPARATE, always-live source
  of the same poison via the PMAT-082 warmup GEMM.
- cuda-decode-graph-audit-2026-06-14: that audit's batched (m>1, seq>1024) graph-staleness
  findings are NOT this bug; apr run is c=1 and the poison is at startup, not in decode.
