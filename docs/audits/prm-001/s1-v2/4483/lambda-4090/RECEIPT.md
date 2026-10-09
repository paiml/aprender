# #4483 RCA-SRV-001 fix 1: tensor-core prefill projections, PRM-S1 v2 receipt on RTX 4090

Set `set-a4910a65.jsonl` (sha256 `a4910a65…b30632`), 200 items (50 each at 2k/8k/16k/32k), cell `lambda-cuda` (RTX 4090, sm_89).
GGUF `Qwen3.5-4B-Q4_K_M`, sha256 `00fe7986…ef11a4`, the same file as the gx10 receipt (`../RECEIPT.md`).
Competitor: llama.cpp `d1d3c3396`, same host and GGUF, one server per leg, run in sequence under the GPU lock.
Prompt ids are identical on all four legs: `prompt_ids_sha256` and `input_tokens` match item by item, 200/200.
The rows hold timings, token counts, RSS and the parsed verdict state only. There are no prompts or completions.

**Binary:** `apr 0.70.3`, `3824eba424` = `main` `acd599e52a` plus the gx10 receipt commit, which is docs only;
sha256 `eb417ad9171002d67251174f09608def391fa15a1da8b935b95f568c23ef8944`. Unlike the gx10 binary, this is
`main` itself, with no car commits. The f16 leg is the default; the f32 leg is the same binary with
`APR_QWEN35_PREFILL_GEMM=f32`. Server: `--gpu-layers all --context-length 36864`.

**Engaged:**
- f16 log: `[qwen35] fp16 prefill weights prewarmed: 6807 MiB in 48 ms (#4313)`. The first prompt prefills at 9748 tok/s.
- f32 log: no prewarm line. The same first prompt prefills at 2585 tok/s.

## Where apr's prefill time comes from

`main` has no SRV-TIM-001 timings block on the Qwen3.5 chat route (it is open on #4487), so `prompt_ms` is
`null` in the apr rows. apr's prefill time is taken from the server's own log line,
`[qwen35] batched prefill: N tokens in M ms (… from position P)`. Each row is paired with the log lines in order,
up to and including the line where `P + N` equals the row's `input_tokens`, and their `M` values are summed.
All 200 rows pair on every leg, with prefilled tokens equal to `input_tokens` on every row. The pairing is in
`prefill-apr-<leg>.tsv`; `log_lines` is 2 on every row (see the last section).

llama.cpp's `prompt_ms` comes from its own timings. It leaves out a cached prefix: 49 tokens on 150 rows and
10 tokens on 15. That favours llama.cpp slightly; it does not favour apr.

apr's decode tok/s is derived as `output_tokens / (wall − prefill)`. It includes HTTP and detokenisation, so it
is a lower bound. llama.cpp's figure is its own `decode_tps`.

## Speed (p50 over 200 rows)

| leg | prefill ms | prefill tok/s | decode tok/s | wall ms | prefill ratio vs llama | wall ratio vs llama |
|---|---|---|---|---|---|---|
| apr f16 (fix) | 1030.0 | 7458 | 94.1 | 3800.7 | **1.68×** | **1.74×** |
| apr f16, pass 2 | 1016.5 | 7446 | 96.1 | 3750.9 | 1.66× | 1.71× |
| apr f32 (same binary) | 2472.5 | 3093 | 96.0 | 5264.5 | 4.04× | 2.40× |
| llama.cpp | 611.4 | 12698 | 196.6 | 2189.1 | 1 | 1 |

Prefill p50 per stratum, in ms (f16 / f32 / llama), with the f16 ratio vs llama:
- 2k: 119.5 / 346.0 / 96.6 (1.24×)
- 8k: 446.5 / 1182.5 / 305.7 (1.46×)
- 16k: 1588.5 / 3537.0 / 881.9 (1.80×)
- 32k: 3851.5 / 7440.5 / 1603.5 (2.40×)

f32 / f16 prefill, paired item by item, p50 (min–max): all 2.42 (1.69–3.50); 2k 2.80 (2.62–3.50);
8k 2.64 (2.44–3.21); 16k 2.23 (2.02–2.40); 32k 1.93 (1.69–2.04).

Result: on the 4090, prefill is 2.42× faster than f32 on the same binary. The gap to llama.cpp shrinks from 4.04×
to 1.68×. The f16 speed-up falls as context grows, and the gap to llama.cpp grows with it, from 1.24× at 2k to
2.40× at 32k, while llama.cpp's own rate stays flat (10–13k tok/s). That points to attention, which grows with
context, more than to the projections. This receipt does not measure that split.
Decode is unchanged at 0.48× llama. That gap is the decode GEMV, a later RCA-SRV-001 item, not this fix.

## Quality falsifier (paired by `diff_sha256`)

- f16 vs f32: the parsed verdict state is identical on **200/200** items. The mix is 39 Fail / 161 Pass on both
  legs, so the match is not vacuous. Output token count is identical on 177/200, so greedy text is **not**
  bit-identical: the fp16 GEMM inputs move near-tie argmaxes.
  Stated tolerance: the verdict must be identical on every item. **Met.**
- f16 vs f16 pass 2: verdict and output token count are identical on 200/200. So f16 is deterministic run to
  run, and the 23 count differences above come from precision, not noise. Per-item prefill, pass 2 / pass 1,
  p50 0.999 (min 0.894, max 2.254).
- llama.cpp gives 37 Fail / 163 Pass. Its verdict agrees with apr on 194/200 items for both f16 and f32.

## Larger models on 24 GB: the fp16 cache does not always fit

The fp16 weight cache is prewarmed only when it fits (`f16_prewarm_fits`: cache + 1 GiB ≤ free VRAM). Otherwise
prefill runs the f32 leg. Probed at load on this host, `--context-length 36864`:

| model | prewarm line | VRAM used after load |
|---|---|---|
| Qwen3.5-9B-Q4_K_M | `fp16 prefill weights prewarmed: 13196 MiB in 105 ms (#4313)` | 19449 / 24564 MiB |
| Qwen3.5-27B-Q4_K_M | `fp16 prefill weights NOT prewarmed: 46445 MiB needed, 7223 MiB free; prefill uses f32` | 16817 / 24564 MiB |

So on a 24 GB card the 27B still runs its prefill projections on no tensor core. A whole-model fp16 cache can
never fit there. Closing that needs a leg that dequantizes each weight into an fp16 scratch per call and then
runs the same `gemm_f16_to_f32`.

## Each request is prefilled in two calls

On every row the prompt is prefilled as the body, then a 7-token tail from the body's end position. The tail
costs 25.0 ms mean on f16 and 51.5 ms on f32, and it is included in the prefill times above. At 2k it is
about a fifth of the f16 p50.
