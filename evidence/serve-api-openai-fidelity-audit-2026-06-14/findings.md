# Serve HTTP / OpenAI-compat API — adversarial fidelity audit (2026-06-14)

Adversarial bug-hunt over the serve API layer (param plumbing, stop sequences, SSE
streaming, response/usage): parallel finders → skeptic verification → confirmed-only.
**12 confirmed real bugs of 16 findings** — the most bug-dense subsystem audited (the
user-facing OpenAI-compat surface was under-tested for spec fidelity). Tracked below,
prioritized; fixed incrementally (one coherent cluster per PR).

## SSE streaming (CRITICAL/HIGH)
1. **[FIXED PMAT-753]** Double `data:` prefix — `chat_completions_stream.rs` did
   `Event::default().data(format!("data: {}\n", json))`, but axum's `Sse` adds the
   `data: ` field itself → wire was `data: data: {json}`, so spec clients `JSON.parse`
   the literal "data: {json}" and fail. **Streaming broken for every client.** Fixed to
   `Event::default().data(json)` (matches the correct `openai_handlers.rs` form).
2. **[TODO]** Multi-byte UTF-8 split across tokens (`chat_completions_stream.rs:83`):
   `decode(&[token_id])` per token → emoji/CJK spanning two tokens emit replacement
   chars. Needs a cross-token byte buffer (decode longest valid UTF-8 prefix, carry rest).

## Stop sequences ignored (HIGH) — model doesn't stop / leaks stop text
3. **[FIXED PMAT-754]** `try_cached_completions` (realize_handlers_embed_completion.rs) — now applies stop via shared truncate_at_stop().
4. **[FIXED PMAT-754]** `try_quantized_completions` (realize_handlers_embed_completion.rs) — now applies stop via shared truncate_at_stop().
5. **[FIXED PMAT-755]** `try_gpu_completions` (gpu_completions_handler.rs) — now applies stop via truncate_at_stop().
   Also **[FIXED PMAT-755]** `try_apr_q4k_completions` (the audit mis-stated it was already correct — it wasn't; now uses the helper).
6. **[FIXED PMAT-756]** chat path `/v1/chat/completions` — `build_chat_response` now runs
   `finalize_chat_text()`, applying the shared `truncate_at_stop()` helper across ALL 7
   `build_chat_response` call sites (gpu/quantized/cached/q4k/qwen3_moe/registry) AND the
   inline `try_safetensors_cuda_backend` builder (which bypasses `build_chat_response` and
   was caught as a non-streaming gap by adversarial re-review) — setting `finish_reason="stop"`
   when a stop string truncated (precedence over "length"). Helper promoted to `pub(crate)`.
   Covered by FALSIFY-CHAT-STOP-756 (pmat756_chat_stop_tests). With this, **stop sequences
   are honored on every NON-STREAMING completion+chat backend**. Streaming stop — applying
   stop mid-SSE (pregenerated/true_streaming/chat_completions_stream) — remains OPEN, grouped
   with item 2's cross-token stream-buffer work (same incremental-detection machinery).

## Param plumbing dropped (HIGH/MEDIUM) — OpenAI fidelity
7. **[TODO]** `n` accepted but ignored (mod_create_demo.rs:94) — n>1 silently returns 1
   choice. Min fix: reject n>1 with 400; full: generate n choices.
8. **[TODO]** `seed` accepted but not plumbed (mod_create_demo.rs:92) — non-reproducible.
   MoE path already plumbs it (cuda_chat_backend.rs:748); mirror to dense backends.
9. **[TODO]** `top_k` ignored, hardcoded 40/1 (cuda_chat_backend.rs:252) — mirror MoE plumbing.
10. **[TODO]** temperature default 0.7 vs OpenAI 1.0 (openai_handlers.rs:99) — the repo's
    own `default_temperature()`=1.0; chat path hardcodes 0.7. Use the canonical default.

## finish_reason wrong (HIGH/MEDIUM)
11. **[TODO]** Streaming `finish_reason` always "stop", never "length" when max_tokens hit
    (mod_create_demo.rs:398).
12. **[TODO]** `true_streaming_sse_response` lacks max_tokens → can't set finish_reason
    correctly (openai_handlers.rs:211).

## Method
All confirmed by skeptic verification (default-refute). Several cite the codebase's OWN
correct reference (openai_handlers.rs SSE form; default_temperature()=1.0;
try_cuda_gguf_completions stop truncation; MoE seed/top_k plumbing) — i.e. the bugs are
drift where one backend/path diverged from a sibling that does it right.
