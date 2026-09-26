# #4026 live control: `apr serve` chat `top_logprobs` compared with a llama.cpp oracle

**Verdict: RED on the pre-registered magnitude bar.** The feature reports apr's distribution, but apr's Q4_K CPU logits differ from llama.cpp's by more than 0.25 nats on 25 of 29 steps.

- The report is faithful. Rank and alignment agree with the oracle, the refusals work, and the planted cases fail.
- The magnitude gap belongs to forward parity (apr vs llama.cpp on the same GGUF), not to the logprobs report. That attribution is an inference and is untested here; see *What this does not prove*.

## Setup (mechanism proven, not assumed)

| | |
|---|---|
| apr | `apr 0.69.3 (6f909ca30)`, a release build of rex/001 HEAD. The `apr serve` log says `gpu-layers: requested=none resolved=0 total=24 (backend=cpu)` |
| oracle | llama.cpp `llama-server` build 11050 (commit 60b06ab9a), `-ngl 0`, `CUDA_VISIBLE_DEVICES=""` |
| model | `Qwen3.5-0.8B-Q4_K_M.gguf`, sha256 `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517` |
| oracle prompt | The GGUF's own jinja template through `/apply-template` with `enable_thinking:false`, which is the mode apr served. It is tokenized by `/tokenize` with `parse_special:true` and sent to `/completion` as **ids** |
| teacher forcing | For step *i*, the oracle gets the prompt ids plus apr's first *i* generated ids, with `n_predict:1`, `n_probs:20`, `post_sampling_probs:false` (raw full-vocab softmax) and `cache_prompt:false` |
| request | `temperature:0`, `max_tokens:8`, `logprobs:true`, `top_logprobs:20` |

Prompt identity: apr's `usage.prompt_tokens` equals the oracle's token count on all 6 prompts (24/23/17/17/22/17).

- **This is a count, not an id diff.** apr exposes no chat-path prompt ids, so an id-level comparison was not possible.
- The first run was also a count mismatch (apr 24, oracle 22), caused by the oracle template's thinking-on default. That is how the thinking mode was pinned.

## Result: 6 prompts, 29 teacher-forced steps

| prompt | steps | top-1 agrees | top-20 overlap | max \|Δlogprob\| over shared ids |
|---|---|---|---|---|
| p1 capital of France | 1 | **0/1** | 14 | **1.65** |
| p2 three primes | 7 | 7/7 | 19–20 | 0.37–0.81 |
| p3 primary color | 8 | 8/8 | 17–20 | 0.19–0.33 |
| p4 hello → Spanish | 2 | 2/2 | 19–20 | 0.38–0.56 |
| p5 12 × 12 | 8 | 8/8 | 18–20 | 0.37–1.13 |
| p6 say blue | 3 | 3/3 | 19–20 | 0.25–0.40 |

Pre-registered bar (set in `c4026_live.sh` before any data): top-1 equal, overlap ≥ 16, max |Δ| ≤ 0.25, chosen |Δ| ≤ 0.25, and Σexp(top-20) ≤ 1.
- Overlap ≥ 16 and top-1 each pass on 28 of 29. Both misses are p1 step 0 (overlap 14).
- **|Δ| ≤ 0.25 fails on 25 of 29 steps.** The four within it are all p3 steps (0.19–0.24).

p1 step 0 is the top-1 flip:
- apr: `" Paris"` at −0.506, then `"Paris"` at −1.183.
- oracle: `"Paris"` at −0.390, then `" Paris"` at −1.651.

The step-0 prefill logits are not the culprit, because the other five step-0s look like decode steps (0.31–0.40). One flip is an anecdote, not a named cause.

## Controls

| control | expected | observed |
|---|---|---|
| positive: the comparator on the oracle against itself (p2, p3, p5) | pass, Δ = 0 | pass, max Δ 0 |
| planted: the oracle shifted one step late (p1) | fail | fail |
| planted: p1's record against p2's oracle | fail | fail |
| `top_logprobs:21` / `top_logprobs` without `logprobs` / `stream` with logprobs | 400, named | 400 / 400 / 400 |

## Findings (for the cop; only the cop files issues)

1. **Forward parity, apr vs llama.cpp, Qwen3.5-0.8B Q4_K_M on CPU.** Top-20 logprobs differ by 0.19–1.65 nats, and one top-1 flips (p1). Attribution is open: apr's Q4_K/Q8_K dot products, accumulation order, or the SSM/linear-attention kernels. It belongs to the parity contract, not #4026.
2. **`POST /tokenize` on apr serve splits `<|im_end|>`** while it parses `<|im_start|>` (248045) as one special. Given this prompt, it splits `<|im_end|>` into 6 literal pieces (28 ids, where the chat path and the oracle have 24). Evidence: `p1.apr_tok.json`.

## What this does not prove

- That the 0.25 bar is the right bar for #4026. It tests forward parity and the logprobs report together. A report-only oracle, meaning the same session's full logits dumped independently, is not in this receipt.
- GPU (the lambda 4090 is excluded; gx10 was not run), other quants, streaming, and `/v1/completions`, which are not implemented yet.

Reproduce: `bash c4026_live.sh <apr> <gguf> <out>`. It uses bash, curl and jq only.
