# Continuous-batching / batch scheduler — adversarial bug-hunt + triage (2026-06-14)

Adversarial bug-hunt over the CUDA continuous-batching scheduler (5 concurrency dimensions:
token routing, slot lifecycle, per-request config isolation, completion counters, queue
backpressure). **10 findings marked REAL.** Findings 1 and 2 were HAND-VERIFIED against the
source (the others are hunt-confirmed and listed with my assessment — each needs individual
verification before a fix).

**IMPORTANT — why no fix PR yet:** every confirmed finding lives in `#[cfg(feature="cuda")]`
GPU code (`generate_batched_streaming.rs`, `cuda_batch_scheduler.rs`, the batched executor
kernels). These cannot be unit-tested without a GPU run, several are perf-sensitive (the whole
point of on-GPU argmax is speed), and the highest-impact one is a sizeable feature, not a
one-liner. A blind, untested patch to the production GPU serving path would be reckless. These
are surfaced as priority defects for a FOCUSED effort with GPU access (lambda-vector RTX 4090),
not loop-tick patches. **Do not blind-patch GPU batching code.**

## HAND-VERIFIED — top priorities

### [2] CRITICAL — batched decode is greedy-argmax ONLY; per-request temperature/top_k/seed ignored
`generate_batched_streaming.rs:301 batched_decode_step` → `executor.forward_batched_to_token_ids`
(and `_graphed`) returns token IDs via **on-GPU argmax** (timer label "fwd+sync+argmax"). It is
passed only `embed_buf, pos_buf, dims, eps` — **`state.configs` (per-request temperature/top_k/
top_p/seed) is never consulted** for sampling (only checked for stop_tokens + max_tokens in
`distribute_tokens`). So when the CUDA batch scheduler is active, EVERY request is sampled
greedily regardless of its requested temperature/top_k. The single-request path
(`generate_gpu_resident_streaming`) DOES sample (`sample_topk`), so this is a batched-only
divergence. **Violates `continuous-batching-v1.yaml` parity (output_batched ≈ output_single).**
Reachability: only when continuous batching is enabled (`state.cuda_batch_tx()` is Some).
FIX (large): a `forward_batched_to_logits` variant returning per-slot logits, then CPU-sample
per `configs[slot]` (temperature/top_k/seed) in the decode loop — with a perf check vs the
current on-GPU argmax fast path. GPU testing required.

### [1] HIGH — setup/prefill error skips batched_cleanup → next batch KV corruption
`cuda_batch_scheduler.rs:331` — on `batched_setup_and_prefill` error, it notifies the channels
then `return;` WITHOUT `batched_cleanup()`. The high-water-mark strategy (PMAT-075) keeps KV
buffers allocated across batches, so a stale `batched_kv_stride` persists; the NEXT same-or-
smaller batch reuses the buffers and prefill/scatter writes K/V to the wrong slots → cross-
request token corruption (NOT a raw memory leak — buffers are freed on executor drop). FIX:
reset the batched logical state (`batched_kv_stride`) on the error path — ideally make
`batched_setup_and_prefill` reset its own state on internal error (RAII) rather than the caller
constructing a dummy `BatchedDecodeState`. GPU testing required.

## HUNT-CONFIRMED — pending individual verification (all GPU/concurrent, need a GPU run)

- **[4] HIGH** per-request `seed` not used in batched inference (corollary of [2]; seed only
  matters once stochastic sampling exists — fix alongside [2]).
- **[5] HIGH** `add_slot_to_batch` (generate_batched_streaming.rs:544-546) sets
  `max_tokens_max = max(max_tokens_max, configs.last().max_tokens)` ignoring the `gen_idx`
  offset, so a slot joining mid-batch at step N terminates ~N steps early.
- **[6] HIGH** inconsistency: `add_slot_to_batch` (545-546) uses `max_tokens` only while
  `recycle_slot` (638-640) uses `gen_idx + max_tokens` — same join scenario, different
  termination depending on path.
- **[7] LARGELY FALSE POSITIVE (hand-verified 2026-06-14)** `try_batch_completion` recv path
  (realize_handlers_embed_completion.rs:241-243) returns `Ok(None)` on `Err` -> graceful
  fallthrough to the next backend. A CRASHED processor drops the tx -> recv `Err` -> handled.
  Only a STALLED processor that holds the tx open and never sends hangs (narrow); a naive
  timeout there risks killing legitimately-slow long generations. Not a clean fix; deprioritized.
- **[8] HIGH** `cuda_batch_scheduler.rs:70-72,82` token `try_send` failure breaks the loop
  silently — verify whether this is intended (client disconnected) vs lost-token-on-full-channel.
- **[9] LARGELY FALSE POSITIVE (hand-verified 2026-06-14)** the send path
  (realize_handlers_embed_completion.rs:238) returns `Ok(None)` on send `Err` -> graceful
  fallthrough to the next backend, NOT a swallowed hang. Queue-capacity tuning remains a
  possible enhancement but there is no correctness bug here.
- **[10] HIGH** `gpu_handlers.rs:410-435` a stuck `process_batch` blocks all subsequent
  requests until the window timeout (liveness).
- **[3] MEDIUM** `max_tokens_max = configs.iter().map(max_tokens).max()` keeps the batch
  decoding until the LARGEST-limit request finishes — smaller-limit requests are marked done
  (correct output) but the GPU keeps stepping (efficiency, not correctness).

## Finding-2 implementation plan (ready to execute — focused GPU effort)

Scope confirmed (2026-06-14): the logits ALREADY exist on GPU before argmax
(`batched_forward.rs` `batched_output_norm_lm_head_argmax` fills `workspace.logits_buf` via the
LM-head GEMV, then calls `batched_gpu_argmax`). The reusable CPU sampler is
`OwnedQuantizedModel::sample_topk(logits: &[f32], temperature, top_k) -> u32`
(gguf/inference/fails.rs:92). Validation is feasible: RTX 4090 free + 0.5B models present
(`/mnt/nvme-raid0/models/qwen2.5-coder-0.5b-instruct-q4k.apr`) + `apr` binary built.

Steps:
1. Refactor `batched_forward.rs` (non-graphed path only — graphed is a launch-opt; the sampling
   path can use non-graphed):
   - extract `batched_forward_run_layers(...)` (embed + per-layer loop, leaves result in
     hidden_buf2) — currently lines ~38-164.
   - extract `batched_output_norm_lm_head_into_logits(m, hidden_dim, vocab_size, eps) ->
     (logits_ptr, logits_size)` (output_norm + LM-head GEMV, NO argmax) — lines ~178-276.
   - `forward_batched_to_token_ids` = run_layers + into_logits + `batched_gpu_argmax`
     (behaviorally identical — build-verify with --features cuda).
   - NEW `forward_batched_to_logits(...) -> Vec<f32>` = run_layers + into_logits +
     `copy_to_host` the m×vocab logits.
2. `generate_batched_streaming.rs` `batched_decode_step`: if ALL live slots are greedy
   (temperature == 0.0) → keep `forward_batched_to_token_ids` (fast GPU argmax). If ANY slot
   has temperature > 0 → call `forward_batched_to_logits`, then per slot:
   greedy → argmax the slot's logits slice; else `sample_topk(slot_logits, configs[slot].
   temperature, configs[slot].top_k)`. (Seed = finding [4] resolved here too once stochastic.)
3. Unit-test (CPU, no GPU): a pure `select_batched_token(slot_logits, &config) -> u32` helper
   (argmax vs sample_topk dispatch) — mutation-verifiable.
4. GPU VALIDATION (mandatory before merge — do NOT auto-merge): `apr serve` the 0.5B model with
   continuous batching enabled; fire 2+ concurrent requests with temperature 0.0 vs 1.5 and the
   same prompt+seed; assert temp=0 is deterministic/greedy and temp=1.5 diverges across runs and
   from the greedy output. Plus a perf check: ITL with all-greedy batch must match the current
   fast path (the download cost is paid only when sampling is requested).

Risk note: minimal-blast option is to DUPLICATE the body into `forward_batched_to_logits`
(leaving the proven greedy path byte-untouched) rather than refactor — but both require the GPU
validation in step 4, so the cleaner refactor is acceptable if step-1 build-verifies + step-4
passes. Open the PR WITHOUT auto-merge until step 4 is green.

## Method / lesson
Hunt + skeptic verification, then HAND-VERIFY each REAL verdict before acting — this audit's
findings 7 & 9 were verified as FALSE POSITIVES (graceful `Ok(None)` fallthrough already handles
the send/recv-error paths), and a naive timeout "fix" would have killed slow-but-valid
generations. The headline ([2]) is real and high-impact, but the right response is a careful
GPU-tested effort (plan above), not a blind loop-tick patch.
