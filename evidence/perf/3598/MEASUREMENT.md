# PMAT-3598 row 1 (#3542) — what `apr run --json` now says, and what it found

Host `noah-Lambda-Vector`, RTX 4090 sm_89, **GPU occupancy recorded at start: 639 MiB used, 0 %
utilisation** (a shared box; a timing taken without recording occupancy is a number nobody can stand
on). Binary built from this branch with `--features cuda`. Every row is one `apr run … --json`.

## Qwen3.5-4B-Q4_K_M, `--max-tokens 1`, backend `cuda-qwen35`

| prompt | load_ms | h2d_ms | **validate_ms** | prefill_ms | decode_ms | unattributed_ms | wall_ms |
|---|---|---|---|---|---|---|---|
| `Hi` | 1757.5 | 141.4 | **2129.5** | 125.6 | 9.7 | 1185.9 | 5349.7 |
| 8 words | — | — | **3164.6** | 190.3 | 9.7 | — | 6470.4 |
| 32 words | — | — | **6642.2** | 421.0 | 9.8 | — | 10183.9 |
| 72 words | — | — | **9569.6** | 808.3 | 9.9 | — | 13489.5 |
| 144 words | 1807.7 | 143.1 | **9526.3** | 1518.6 | 10.1 | 1178.1 | 14183.8 |
| 288 words | — | — | **9553.0** | 2983.3 | 10.7 | — | 15616.1 |
| `Hi`, 32 tokens | 1814.5 | 141.7 | **2035.5** | 124.1 | 87.7 | 1176.4 | 5379.8 |

**The mechanism.** `validate_ms` is `f2_validate_qwen35` — a CPU reference forward plus a GPU probe,
run before the user's first token, on every invocation. It is **67 % of the 14 s** at 144 words and
costs **3.2×–16.9× the prefill it guards**.

**It saturates at ~9.55 s, and the reason is in the code, not inferred:**
`QWEN35_F2_PROBE_MAX = 64` (`forward_qwen35.rs:1396`) caps the guard at 64 positions on both
backends. Cost rises with the prompt to that cap, then flattens — which is the shape of the
previously unexplained curve on #3596 (`Hi` 6.77 → 24w 12.76 → 72w 17.69 → 144w ~17 → 288w ~19).

**What the split rules out.** `load_ms` is flat at ~1.8 s across every prompt length, so it is not
load. GPU `prefill_ms` is linear at ~10.3 ms/word and 126 ms at two characters, so it is not a
170×-slow prefill. `h2d_ms` is flat at 141–143 ms, so the transfer is not a candidate at all.
`decode_ms` is 9.7 ms for one token and 87.7 ms for 32 — about **2.7 ms/token (~370 tok/s)**, against
llama.cpp's measured 179–189 tok/s on the same box and file. **apr's generation is not the gap; the
gap is entirely pre-generation.**

## Which half of the guard owns the time (operator-ruled addition, for #3604's receipt)

`validate_ms` is reported with its two halves, both INSIDE it and excluded from the stage sum so the
books still close:

| prompt | validate_ms | **validate_ref_ms** (CPU reference forward) | validate_probe_ms (GPU probe) | wall_ms |
|---|---|---|---|---|
| `Hi` | 2227.8 | **2016.1 (90.5 %)** | 204.8 (9.2 %) | 5775.4 |
| 144 words | 10135.0 | **9444.3 (93.2 %)** | 656.1 (6.5 %) | 15138.1 |

**The guard is a CPU forward, not a GPU probe.** 90–93 % of it is running the prompt through the CPU
implementation to have something to compare against. So #3604's cache removes a CPU forward, and its
before/after receipt can say which mechanism disappeared rather than only that the total fell.

**Not established here**, and deliberately left open for the 0.70 fix row: why the guard runs per
invocation rather than once per (model, build, device), and whether a cheaper equivalent guard exists.

**The instrument's own gap.** `unattributed_ms` is a flat ~1.18 s that no stage claims. It is
reported rather than smeared into a neighbouring stage, which is the point: a large
`unattributed_ms` is a finding that says the instrument is missing a stage, and this one is.

## The dense path, and a separate defect this row only reveals

qwen2.5-coder, `Hi`, `--max-tokens 1`, GPU vs `--no-gpu`:

| model | GPU attempt (wall) | `--no-gpu` (wall) | validate_ms | backend reported |
|---|---|---|---|---|
| 0.5b | 32009 / 35798 | **17549** | 15124 / 17645 | `cpu` |
| 1.5b | 10393 | **3700** | 1907.9 | `cpu` |

The dense GPU validation **fails on correctness** — stderr: `GPU output diverges from CPU at position
1 (argmax 20840 != 73562, cosine 0.4153); min cosine 0.4153, validated via batched prefill — falling
back to CPU` — and the run then falls back through wgpu to CPU. **The user pays 15–17 s for a
rejected GPU attempt and then pays for the CPU run anyway, making the default 2–3× slower than
`--no-gpu`.** The warning was on stderr; the cost was invisible until this row.

This is not row 1's to fix and nothing here touches it. It resembles the #3483 batched-prefill FP8
class and needs its own row.
