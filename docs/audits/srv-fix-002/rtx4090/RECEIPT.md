# SRV-FIX-002 — RTX 4090 receipt (#4484, the 4090 half)

The protocol is `../PREREG.md`, run unchanged on a second cell. The prereg names gx10 as its
cell, so this is the same pre-registered protocol applied to lambda-vector (RTX 4090, sm_89,
driver 580.119.02, nsys 2026.1.3). It is not a new pre-registration. aprender-79,
2026-09-27 (Madrid 15:40–16:05).

- **Code:** `origin/79/4484-flash-prefill` @ `3b8fc87675`, built from this branch. This branch
  adds only docs on top of that sha, so `apr --version` prints `apr 0.70.0 (baf9a5f261)`.
  The build was `--features cuda` with `CARGO_INCREMENTAL=0`. Binary sha256s are in `sums.txt`.
- **Model:** `Qwen3.5-4B-Q4_K_M.gguf`, sha256 `00fe7986…a11f9`. This is the same file gx10 used.
- **Prompts:** `../gx10/prompts.sh`. All 5 sha256s match gx10's byte for byte (`sums.txt`).
- **GPU proof:** each run's stderr names its path. It shows `Backend: GPU (CUDA, NVIDIA GeForce
  RTX 4090 …)` and `attention flash (f16 inputs, f32 accumulation)` or `attention cuBLAS f32`.
  The nsys kernel names below confirm it.

| Gate | Result | Evidence |
|---|---|---|
| P1 greedy parity, flash vs cuBLAS f32, 64 tokens, 5 prompts (~30 to 28,931 tokens) | **PASS 5/5** byte-identical | `parity-verdicts.txt`, `base/rc.txt` |
| P2 planted mutant (`../gx10/mutant.diff`, softmax scale ×1.25 at both `c_exp` sites) | **RED as required**: 0/5 identical | `mutant-verdicts.txt`, `mutant/rc.txt` |
| P3 prefill attention vs llama.cpp `d1d3c3396`, 30k prompt, bar ≤ 2.0× | **MISS on this cell: 2.09× and 2.92×** over two runs | `p3/`, `p3-rep2/` |

## P3 detail

The numbers come from nsys `cuda_gpu_kern_sum`. Runs were sequential under
`/tmp/apr-gpu.lock`. The kernel sets are the same as gx10's.

| run | apr flash (`gdn_prefill_flash_attention_256` + `_combine_256`) | llama.cpp (`flash_attn_ext_f16<256…>` + `flash_attn_stream_k_fixup_*`) | ratio |
|---|---|---|---|
| p3 | 756.1 + 11.8 = **767.9 ms** | 360.7 + 5.5 + 1.2 = **367.4 ms** | 2.09× |
| p3-rep2 | **885.9 ms** | **303.5 ms** | 2.92× |

Run-to-run noise is large on this box: apr moved 15% and llama.cpp 17% between the two runs.
The direction does not change, and both runs are over the bar. On gx10 the same code was
1.15×. The difference is llama.cpp's side: its attention is 2.1–2.5× faster on the 4090 than
on gx10 (756 ms there), while apr's kernel is roughly the same speed on both.

Deviation: `p3.sh` differs from `../gx10/p3.sh` in paths only, plus one change to how the
server is stopped. It stops only the recorded nsys child and has no `pgrep -f` pattern
fallback. If no child is found, it sends SIGINT to the recorded nsys pid. That happened on
both runs, and nsys exited 0 both times.

## Also measured (not gated)

- **End-to-end prefill at 28,931 tokens.** These are apr's own stderr lines.
  - Without nsys (P1 runs): flash 10.99 s (2,634 tok/s) vs f32 15.41 s (1,878 tok/s).
  - Under nsys: flash 11.37 s / 8.76 s vs f32 13.20 s / 14.96 s.
  - llama.cpp under nsys: 3.16 s / 2.49 s (9,154 / 11,604 tok/s).
  - Flash beats f32 on every pair. apr is still 3.5× behind llama.cpp end to end.
- **Where apr's time goes.** `ampere_sgemm_128x64_tn` takes 5,406 ms of apr-flash's 9,551 ms
  kernel total (57%), in run p3. That is f32 SIMT GEMM, as on gx10. It is the next lever, not
  attention.
- **f32 path, for comparison.** `causal_mask_softmax` is 1,619.7 ms, plus
  `ampere_sgemm_128x128_{nn,tn}` 3,475.5 ms for QKᵀ/PV. Flash replaces all of this with 768 ms.
