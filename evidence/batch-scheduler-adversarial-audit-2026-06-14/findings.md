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
- **[7] CRITICAL-ish** `realize_handlers_embed_completion.rs:241` awaits the oneshot response
  with NO timeout — if the batch processor task crashes/stalls, the HTTP request hangs forever.
  (Fix needs a generous timeout that won't kill legitimately-slow long generations.)
- **[8] HIGH** `cuda_batch_scheduler.rs:70-72,82` token `try_send` failure breaks the loop
  silently — verify whether this is intended (client disconnected) vs lost-token-on-full-channel.
- **[9] HIGH** `gpu_handlers.rs:378` queue capacity hardcoded; overflow → `send` Err is
  swallowed at the handler (realize_handlers_embed_completion.rs:238) → client hang / lost req.
- **[10] HIGH** `gpu_handlers.rs:410-435` a stuck `process_batch` blocks all subsequent
  requests until the window timeout (liveness).
- **[3] MEDIUM** `max_tokens_max = configs.iter().map(max_tokens).max()` keeps the batch
  decoding until the LARGEST-limit request finishes — smaller-limit requests are marked done
  (correct output) but the GPU keeps stepping (efficiency, not correctness).

## Method / lesson
Hunt + skeptic verification, then hand-verify. The recurring lesson holds: the headline finding
([2]) is real and high-impact, but the right response is a careful GPU-tested effort, not a
blind loop-tick patch. Findings recorded so the focused effort starts from a vetted list.
