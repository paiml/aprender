# Implementation receipt — PMAT-4258 (#4258), carried onto batch PR #4322

## Provenance
The fix was written on `fix/q8-flag-qwen35-4258` (PR #4302). That round was AGREED 3/3 PASS on head 4482ab98b. The
PR was then parked by the emergency fold #4311 because of a conflict against release/0.69.3. Here it is cherry-picked
(`-x`) onto main-based `perf/0694-qwen35-persistent-embed`:

- 263958fc3 ← 1f14aea95 (fix)
- 56c9f9112 ← e9d074b83 (test inputs)
- 4d4a7841c ← 4482ab98b (roadmap)

The code commits applied cleanly. The roadmap conflict was resolved as the union of both sides.

## Change carried
`ensure_q8_activation` keys the cached Q8_1 activation by `(ptr, len)`. The qwen35 writer kernels invalidate it:
`rmsnorm_into`, `per_head_rmsnorm_into`, `residual_add_into`, `fused_swiglu_into` and `gdn_sigmoid_gate_into`.

## New on this branch (555a6774d)
`qwen35_cuda_dp4a_gemv_is_catastrophic_through_the_recurrence` asserted DP4A breaks parity. It now fails, as its own
message intends. It becomes `qwen35_cuda_dp4a_gemv_holds_parity_through_the_recurrence`, which asserts 0 broken
positions, so a return of the stale reuse is caught. The pin comment in `with_max_seq_len` now states the real cause.
The float pin itself is NOT changed: lifting it waits on a 2B/4B re-measure (#4030).

## Gates (head 555a6774d)
- Build (`--release --features cuda --no-run`): rc=0.
- `clippy --lib --features cuda -D warnings`: rc=0.
- `fmt --check`: rc=0.
- Roadmap sorted/unique: PASS.
- GPU tests on RTX 4090 via gpu-q, with compute-apps empty at start (logs in `/mnt/nvme-raid0/tmp/q8-gpu/`):
  - `qwen35_cuda*`: 17 passed, 0 failed.
  - `tests_q8_activation_staleness`: 2 passed. Max |DP4A − float| is 3.119e-3 for the second buffer and 6.488e-4
    after `rmsnorm_into`.
  - DP4A on 0.8B over 6 positions: argmax equal at every position, cosine 0.997423 to 0.999476.
