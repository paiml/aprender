# #3791 — `apr serve` on an APR Q4K file, GPU (the ALB-095 pool path): before and after, both hosts

Measured by aprender-c7, 2026-09-21. File: `/home/noah/models/qwen2.5-coder-1.5b-instruct-q4k.apr` (Qwen2, 28 layers, 169 Q4_K + 29 Q6_K
tensors — `attn_v`/`ffn_down` in 14 layers and `output.weight` at Q6_K — and 84 GGUF-named QKV biases). Command:
`apr serve run <file> --gpu --port P`, then `GET /health` and one `/v1/chat/completions` request, greedy, `max_tokens` 24,
user *"Write one sentence about the sea."* (`apr_probe.sh`).

| binary | host | path taken | `/health` | chat | reply | prompt tokens |
|---|---|---|---|---|---|---|
| 0.69.0 (225b2a9ab) | lambda (RTX 4090) | `[GH-471] Entering Q4K GPU path` | **503** `{"status":"loading","model_loaded":false}` | **500** | `Invalid token ID: 151935` (argmax over NaN) | — |
| 9ae04b54a (this row) | lambda (RTX 4090) | `[GH-471] Entering Q4K GPU path` | 200 `{"status":"ok","compute_mode":"gpu","model_loaded":true}` | 200 | "The vast, blue sea stretches across the horizon, reflecting the sun's warm rays and creating a mesmerizing display of colors" | 18 |
| 9ae04b54a (this row) | gx10 (GB10) | `[GH-471] Entering Q4K GPU path` | 200 `{"status":"ok","compute_mode":"gpu","model_loaded":true}` | 200 | the same reply, word for word | 18 |

Reference: the CPU forward of the same file (`apr run --no-gpu`) answers *"The sea is a vast and mysterious body of water that
stretches across the globe."* The GPU forward's agreement with it is the unit test
`q4k_gpu_forward_matches_the_cpu_forward_under_the_f2_rule`: 64 real positions, F2 rule accepted, min cosine 0.9999.

## Unmeasured, and recorded as unmeasured (aprender-c7, 2026-09-22)

The head of this branch, `069fb38a8`, carries one commit this receipt does NOT cover: the #2762 module move in
`crates/apr-cli/src/commands/serve/handler_gpu_completion.rs`, cherry-picked from unit (2)'s `c9691a7d6` (itself identical to c3's
`aae778d15`, already folded into batch-1). **No one has measured it on this branch.** It is a pure cherry-pick — 118 insertions,
118 deletions, `fn resolve_serve_max_seq_len` at :573 now above the first `#[cfg(test)]` at :743 — and the identical move on unit (2)
is green (`cargo test -p apr-cli --lib --features cuda commands::serve::`, 274 passed, including
`the_gguf_cuda_serve_path_reads_the_context_length_flag`), so I expect it green here. Expecting is not measuring, and the cop's
ruling was to leave it unmeasured rather than rush a green during the wind-down. Whoever folds this runs that suite.

The F2 ruling for this path is on the issue: https://github.com/paiml/aprender/issues/3791#issuecomment-5771248200 — F2 at load with a
#3604-style receipt on #3748's key, behind a capacity preflight, reusing `3e[44388e]`'s `capacity::plan`. It is NOT implemented on this
branch; what is here is the parity test, which is the shape the ruling declined. That gap is deliberate and recorded.
