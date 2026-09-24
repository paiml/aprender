# PMAT-4234 — implementation receipt

Ticket #4234: `apr serve` on Qwen3.5 served one request at a time
(`Mutex<Qwen35Session>`), so 4 and 16 concurrent streams ran serially.
Branch `feat/4234-qwen35-cb`, based on `rc/0.69.3-rc.2`, per the cop's direction.

## What changed

- **Slots.** `APR_QWEN35_SERVE_SLOTS=N` (1..=64) serves N sibling sessions over
  one device model (`Qwen35Slots`, `api/mod_app_state_qwen35.rs`). A request
  waits for a free slot. The default stays **1**, the 0.69.x behaviour.
- **Decode batcher.** Sibling sessions' decode steps meet in
  `qwen35_decode_batcher.rs` (leader/follower, 2 ms window). A lone stream takes
  `forward_single` unchanged.
- **`Qwen35CudaModel::forward_batch`** (`gguf/cuda/forward_qwen35_cuda_batch.rs`)
  runs one layer-major step for B sequences. Every projection weight is read
  once per step:
  - **FFN:** gate/up/down are batched GEMVs. SwiGLU and the residual are one
    launch over the packed rows.
  - **DeltaNet:** `attn_qkv`, `ssm_alpha`/`ssm_beta`, `attn_gate` and `ssm_out`
    are batched. The conv window, the delta rule and the gated norm run per
    sequence, on that sequence's state.
  - **Attention:** q, k and `attn_output` are batched. `attn_v` stays per
    sequence, because it writes straight into that sequence's KV cache row.
  - **lm_head:** one batched dispatch.
- **Batched GEMV kernels** (`aprender-gpu`, M ≤ 8, tiled above that): Q4_K Mwv,
  Q6_K Mwv, and Q5_K (`BatchedQ5KGemvKernel`, new). Each dequantizes a
  super-block once and runs the M=1 kernel's FMA order per vector: a fresh
  partial, `acc += partial`, then the same shuffle tree. Each output row is
  therefore **bitwise** the single-vector kernel's.
- The single-sequence body (`forward_qwen35_cuda.rs`) is unchanged from the
  pre-batching commit.
- **Observability.** With N > 1 the server prints
  `[qwen35] serving up to N requests at once`. On each slot release it prints
  the batcher's running `steps / tokens carried / mean / widest`.

## Acceptance

**1. Token identity.** Quiet RTX 4090: no foreign compute apps at start, at run
or at end (`nvidia-smi --query-compute-apps`), via `gpu-q`.

- `qwen35_cuda_forward_batch_is_bitwise_forward_single_per_sequence`: logits
  bitwise equal per sequence.
- `gpu_sibling_sessions_decoding_together_match_each_alone_token_for_token`:
  16 streams × 24 tokens, token for token.
- `batched_{mwv_q4k,mwv_q6k,q5k}_gemv_is_bitwise_the_single_vector_kernel_per_vector`:
  M = 1..=12.
- Serve-path filter (chat backend, slots, batcher, batched GEMVs,
  `forward_batch`): **53 passed, 0 failed**.

**2. Throughput at 1, 4 and 16 streams, through `apr serve run --gpu`.**

- Setup: `apr 0.69.3 (1b21a962b)` + the log-line commit `b701c9ff9`,
  Qwen3.5-0.8B-Q4_K_M, greedy, `max_tokens` 256, quiet GPU, `gpu-q`.
- The 1-slot server and the 16-slot server get the same prompts.

| concurrent streams | 1 slot | 16 slots | answers identical |
|---|---|---|---|
| 1  | 134.6 tok/s | 149.0 tok/s | 1/1 |
| 4  | 246.6 tok/s | 336.8 tok/s (+37%) | 4/4 |
| 16 | 249.3 tok/s | 405.5 tok/s (+63%) | 16/16 |

- The 1-stream row is each server's first request (warm-up included), so the
  gap there is noise, not a gain.
- **Mechanism engaged:** the 16-slot server logged `807 steps carried 4612
  tokens (mean 5.7, widest 16)`. The 1-slot server logged no batching line.

**Kernel-level (unit test, same GPU).**

- 16 streams × 24 tokens: 219 tok/s alone vs 290 together.
- 4 sequences × 8 steps: batched decode takes 95.3 ms, against 124.9 ms before
  the projections were batched.

## Checks

- `cargo fmt --all -- --check`: clean.
- `cargo clippy -p aprender-serve -p aprender-gpu --lib --features aprender-serve/cuda -- -D warnings`:
  clean.

## Not in this row / open

- **Default slot count.** The code keeps 1, so batching is opt-in. The per-slot
  decode state is allocated per conversation, but I have not measured VRAM
  under N long conversations on the 9B, so I have not raised the default. That
  is a decision for the cop.
- **Prefill** is per request (chunked, batched within a request). It is not
  interleaved with other requests' decodes as vLLM's scheduler does.
- Mean batch stays well under the slot count (5.7 of 16), because streams
  arrive and finish at different times. Widening the window was not tried.
