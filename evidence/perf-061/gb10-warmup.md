# PERF-061 — GB10 decode shortfall: warm-up, or real deficit?

**Question** (paiml/aprender#2786, under APR-PERF-GATE-001 / #2706): `apr` measured
`ratio_median = 0.579x` against ollama on gx10 (GB10 Blackwell, aarch64) after four
consecutive passes on Ada. `ratio_best = 0.767x` sits much closer to the 0.90 floor than
the median, which reads like warm-up or clock settling specific to GB10 rather than a flat
kernel deficit. Two discriminators were briefed: (1) plot the trial ordering, (2) see
whether a longer warmup moves the median.

## Verdict: REAL DEFICIT. Not warm-up, not settling, not thermal.

`apr`'s steady-state decode rate on GB10 is **106.4 tok/s [95% CI 103.1–110.0]**, measured
by a 24-point token ladder that does not use the gate's differencing estimator at all.
ollama on the same box, same weights, same session is **182.9 tok/s**. That is **0.582x** —
which is, to three digits, what the gate already reported. The gate's number is correct.

The `ratio_best` / `ratio_median` gap that motivated the warm-up hypothesis is a property
of the estimator, not a glimpse of a faster regime. It is reproduced exactly by simulating
a **constant** 106.4 tok/s.

| Discriminator | Result |
|---|---|
| 1. Trend in the ordered trials | **No trend.** Spearman ρ = +0.393 (p=0.198) and +0.179 (p=0.357); Fisher-pooled **p = 0.445** |
| 2. Does a longer warmup move the median? | **+6.4%, p = 0.650.** Cold 0.579x → warm 0.616x. The floor needs +55% |
| 3. (added) Is the steady-state rate itself 106? | **Yes**, 106.4 tok/s by ladder fit; flat from token 128 to 1024 |
| 4. (added) Does `best > median` need a warm regime? | **No.** A constant-rate simulation predicts every observed statistic |
| 5. (added) Is the GPU clocking down? | **No.** Lifetime `HW/SW Thermal Slowdown = 0 µs`; P8→P0 in one 120 ms sample |

---

## Provenance (pinned, proven by content)

| | |
|---|---|
| Host | `gx10-a5b5`, NVIDIA GB10, aarch64, driver 590.48.01, CUDA 13.1 |
| Binary | `/home/noah/perf061-target/release/apr`, built `--release --features cuda --bin apr` |
| Source | `/home/noah/perf061-wt`, detached worktree at `a866988e4`, `git status --porcelain` = 0 lines |
| **Provenance by content** | `strings -a $BIN \| grep -c '^/home/noah/perf061-wt/'` → **16**; `grep -c '^/home/noah/src/aprender/'` → **0**. The other `apr` on this box inverts both counts (0 / 15). Version strings were not used — `apr --version` can be wrong in both directions in a worktree (#2768) |
| CUDA by content | `libcuda.so` ×1, `cuModuleLoad` ×6, `cuLaunchKernel` ×1 |
| `APR_LAYER_TRACE` | **0** `[CB-006-OUT]` lines across all 66 apr invocations (#2764) |
| `HW_DP4A_Q4K` | absent from gx10's environment — confirms #2786's cause (3) |
| Comparator | ollama 0.33.2, `qwen2.5-coder:1.5b-instruct-q4_K_M`, resident daemon |
| Weights | `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` (the same file the gate hands apr) |
| Box state | idle at start (load 0.00, 0 compute procs, GPU 0% / 208 MHz / 4.5 W) |

**Protocol honesty.** This is **not** APR-PERF-GATE-001 §4.4.2. §4.4.2 (`2 × c` warmup +
5 s quiesce) governs the perf-matrix *band* harness, and §4.4.3 defines `decode_tok_s`
*inside* a request. `beat_ollama_decode_throughput_speed.rs` is a different harness: it has
**no warmup step at all**, and it derives a rate by differencing two independent cold
one-shot process invocations. There was therefore no §4.4.2 warmup knob to enlarge, so
discriminator 2 was implemented directly — a full discard invocation before each measured
pair — and backed by a ladder that removes the differencing entirely.

---

## Discriminator 1 — the ordering does not ramp

Full ordered series, extracted verbatim from both failing runs (`ci-run-extracts.txt`):

```
33292383055  apr [111.80, 96.78, 96.41, 99.01, 111.43, 105.62, 139.95]  argmax at trial 7
33298580389  apr [ 90.47, 107.42, 142.71, 132.88, 124.47, 106.76, 117.86]  argmax at trial 3
```

Spearman ρ against trial index, with an **exact** permutation p over all 5040 orderings:

| run | ρ | p (one-tailed) | argmax | argmin |
|---|---|---|---|---|
| 33292383055 | +0.393 | 0.198 | trial 7 | trial 3 |
| 33298580389 | +0.179 | 0.357 | trial 3 | trial 1 |

Fisher-combined **p = 0.445**. The two runs put the maximum in different places. There is no
ramp. Warm-up was already weak before any GPU time was spent.

The incumbent, by contrast, is stable to the third digit *across runs*:

| | run 33292383055 | run 33298580389 | moved |
|---|---|---|---|
| apr median | 105.6 | 117.9 | **11.6%** |
| ollama median | 182.3 | 182.3 | **0.04%** |

---

## Discriminator 2 — a longer warmup does not move the median

7 trials per arm, arms **interleaved** (cold, warm, cold, warm, …) against host drift.
`cold` is exactly the gate's protocol. `warm` prepends a full 384-token discard invocation
immediately before the measured pair, so the GPU is already in P0 and the page cache is hot.
Every replicate:

| trial | arm | t128 (ms) | t384 (ms) | Δ (ms) | tok/s | ratio |
|---|---|---|---|---|---|---|
| 1 | cold | 10231.4 | 12340.9 | 2109.5 | 121.36 | 0.664 |
| 1 | warm | 10259.7 | 12531.3 | 2271.6 | 112.70 | 0.616 |
| 2 | cold | 9826.2 | 12301.7 | 2475.5 | 103.41 | 0.566 |
| 2 | warm | 9954.2 | 12700.2 | 2746.0 | 93.23 | 0.510 |
| 3 | cold | 10175.1 | 12382.2 | 2207.1 | 115.99 | 0.634 |
| 3 | warm | 10256.2 | 12359.1 | 2102.9 | 121.74 | 0.666 |
| 4 | cold | 10125.7 | 12598.2 | 2472.5 | 103.54 | 0.566 |
| 4 | warm | 10046.5 | 12024.4 | 1977.9 | 129.43 | 0.708 |
| 5 | cold | 9917.5 | 12333.5 | 2416.0 | 105.96 | 0.579 |
| 5 | warm | 9853.1 | 12291.5 | 2438.4 | 104.99 | 0.574 |
| 6 | cold | 9829.4 | 12482.3 | 2652.9 | 96.50 | 0.528 |
| 6 | warm | 10256.2 | 12528.6 | 2272.4 | 112.66 | 0.616 |
| 7 | cold | 10248.7 | 12101.2 | 1852.5 | 138.19 | 0.756 |
| 7 | warm | 10167.6 | 12148.6 | 1981.0 | 129.23 | 0.707 |

| arm | n | median tok/s | **ratio** | min | max | CV | best ratio |
|---|---|---|---|---|---|---|---|
| cold | 7 | 105.96 | **0.579x** | 96.50 | 138.19 | 12.7% | 0.756x |
| warm | 7 | 112.70 | **0.616x** | 93.23 | 129.43 | 11.5% | 0.708x |

- **The cold arm reproduces CI run 33292383055 exactly: 0.579x.** The gate is repeatable.
- Warm − cold = **+6.74 tok/s (+6.4%)**, two-sided permutation **p = 0.650** (20 000 shuffles).
- The floor needs **+55%**. A 6.4% non-significant shift is not the explanation.

---

## Why the median is right: the token ladder

The gate's differencing estimator is not needed to get apr's decode rate. Run
`apr run --gpu --benchmark --max-tokens N` for N ∈ {128, 256, 384, 512, 768, 1024}, 4 reps,
and fit `t(N) = C + N/R`. Every replicate (apr-reported ms):

| N | rep1 | rep2 | rep3 | rep4 |
|---|---|---|---|---|
| 128 | 10239.9 | 9812.5 | 9611.3 | 9686.5 |
| 256 | 10945.1 | 10857.5 | 10866.0 | 11079.8 |
| 384 | 11791.9 | 12178.5 | 12098.1 | 12531.0 |
| 512 | 13217.3 | 13164.7 | 13251.4 | 13725.8 |
| 768 | 15676.7 | 15639.5 | 15862.1 | 15955.9 |
| 1024 | 18054.6 | 18029.9 | 18055.2 | 18725.7 |

```
OLS  over n=24:  R = 106.4 tok/s  [95% CI 103.1 .. 110.0]   ratio 0.582x
Theil-Sen     :  R = 107.1 tok/s                            ratio 0.586x
C  (fixed per-invocation cost, apr's own timer) = 8566 ms
residual sd about the fit                       =  233 ms
```

The marginal rate is **flat with sequence depth** — there is no regime the gate's 128→384
window is missing:

| segment | rep1 | rep2 | rep3 | rep4 | median |
|---|---|---|---|---|---|
| 128→256 | 181.5 | 122.5 | 102.0 | 91.9 | 112.3 |
| 256→384 | 151.2 | 96.9 | 103.9 | 88.2 | 100.4 |
| 384→512 | 89.8 | 129.8 | 111.0 | 107.1 | 109.1 |
| 512→768 | 104.1 | 103.4 | 98.1 | 114.8 | 103.8 |
| 768→1024 | 107.7 | 107.1 | 116.7 | 92.4 | 107.4 |

A quadratic fit finds only mild depth decay (109.4 tok/s at N=256 → 102.6 at N=1024) and
does not improve the residual (233 → 236 ms), so the linear model stands.

**Three independent estimators agree:** ladder OLS 0.582x, ladder Theil-Sen 0.586x, cold
gate arm 0.579x — against the CI's 0.579x / 0.647x. The deficit is real and reproducible.

---

## The `best` vs `median` gap needs no warm regime — predict, then verify

`rate = 256000 / Δ` is the **reciprocal** of a noisy difference, so its sampling
distribution is right-skewed: whenever Δ lands low, the reported rate blows up. `best` will
sit far above `median` even when the underlying rate is perfectly constant.

Simulate a **constant** 106.4 tok/s (the ladder fit) with the measured per-invocation
jitter (233 ms each side ⇒ 330 ms on the difference), take median-of-7 and best-of-7,
200 000 times:

| statistic | predicted (95% band) | observed 33292383055 | observed 33298580389 |
|---|---|---|---|
| `ratio_median` | 0.584x (0.520 – 0.665) | 0.579x ✓ | 0.647x ✓ |
| `ratio_best` | 0.712x (0.602 – 0.924) | 0.768x ✓ | 0.783x ✓ |
| `best / median` | 1.212 (1.052 – 1.582) | 1.326 ✓ | 1.210 ✓ |

**All six observed values fall inside the constant-rate prediction.** There is nothing left
for warm-up, scheduling or clock settling to explain.

---

## The GPU is not clocking down

100 ms clock trace of one cold `apr run --max-tokens 1024` (`gb10-clock-trace-1024tok.txt`):

| phase | window | state | what |
|---|---|---|---|
| 1 | 0 → 3.45 s | P8, 208 MHz, 4.4 W, util 0 | host-side load; GPU asleep |
| 2 | 3.45 → 9.58 s | **P0, 2411 MHz**, ~11 W, util 0 | CUDA context up, GPU idle |
| 3 | 9.58 → 12.46 s | P0, 2411 MHz, 15→60 W, util 0 | weight cache / graph capture |
| 4 | 12.46 → 20.38 s | P0, 2522 MHz, ~60 W, util 96 | decode |

**P8 → P0 is a single 120 ms sample: 208 MHz → 2411 MHz.** There is no gradual ramp to
settle into. Decode runs at 2411–2522 MHz against a 3003 MHz ceiling, at 43–56 °C.

Matched-clock probe, both engines decoding on the same GPU ~5 s apart:

| engine | median SM clock | median W | max °C | result |
|---|---|---|---|---|
| apr | 2411 MHz | 48.8 | 96 | 1024 tokens |
| ollama | 2411 MHz | 53.1 | 93 | 1278 tokens in 7.059 s → 181.0 tok/s |

Same clock, comparable power, ~1.7× the throughput for ollama.

After ~1 hour of continuous benchmarking peaking at 96 °C, the **lifetime** counters were
still `HW Thermal Slowdown 0 µs`, `SW Thermal Slowdown 0 µs`, `HW Power Braking 0 µs`. The
GPU never throttled. (`SW Power Cap: Active` reads at *idle* on this part at 4.5 W and goes
*Not Active* under load — it is an idle-state artifact, not a workload throttle.)

---

## What the harness *does* get wrong — noise, not bias

The differential's signal is Δ ≈ 2.4 s sitting on a fixed cost of C ≈ 8.6 s in apr's own
timer (≈12.5 s of process wall before the first decoded token — 61% of a 20 s invocation).
Per-invocation jitter of 233 ms therefore becomes **14% CV** in the reported rate:

- apr CV 12.7–15.0% vs ollama CV 0.41–0.75%, same box, minutes apart.
- apr's median moved 11.6% between the two CI runs; ollama's moved 0.04%.

This makes the gate **noisy**, not **wrong**. The noise is symmetric and the median is
unbiased: cold gate median 106.0 vs ladder fit 106.4. It does mean a single night's
`ratio_median` carries a ±0.07 band around the truth (simulated 95% band 0.520–0.665), so
the gate should not be read to two decimal places.

**Consequence for I-9.** These gx10 runs were *not* spent through an inadequate warmup.
The measurement is sound and the cells it informs are not burned.

---

## Two side findings, both cheap and both real

**A. The yield-to-training guard is blind on GB10.** Both failing runs logged
`GPU memory.used=0 MiB, existing compute procs=0` while PID 32612 —
`./target/release/apr serve run …q4_k_m.gguf --gpu-layers all --port 8451` — had been
resident since 2026-08-27 17:45 (66 h at the time of measurement).
`nvidia-smi --query-compute-apps` returns nothing on GB10's unified memory, and
`memory.used` reads `[N/A]`. `cuda-nightly.yml` calls `procs` "the primary, portable signal
— it works on both sm_89 and GB10". On this silicon it does not. This did **not** cause the
shortfall (that process is idle: GPU measured 0% / 208 MHz / 4.5 W), but the guard cannot
currently see a busy GB10 and so cannot yield to training on it.

**B. The gate's ollama arm can run away.** `ollama run <model> <prompt> --verbose` is
unbounded; ollama context-shifts and keeps generating. One of five trials here ran **283 s**
before being killed by PID, while its neighbours produced 758–1280 tokens in 4–7 s. Run
33292383055's `ollama trials=[154.99, 182.79, 182.9, 182.35, 100.85]` — a first-trial
warm-up and a 45% collapse on trial 5 — is consistent with this. Median-of-5 absorbed it
that night, but it is a live flake source in the comparator arm.

---

## What this does and does not say

- **Says:** on GB10, `apr` decodes this model at ~106 tok/s against ollama's ~183, at the
  same SM clock, on the same weights, with no throttling and no warm-up effect. The
  ≥0.90x no-collapse floor is genuinely violated, by a genuine ~1.7× kernel/pipeline gap.
- **Does not say** anything about Ada. It also does not say GB10 is a slow machine — the
  gate is a *ratio on one box*, and ollama reaches 183 tok/s on that same box.
- **Does not** identify the responsible kernel. That is the next question, and it is now a
  kernel question rather than a measurement question.
- The spec's own note that "GB10 legitimately loses ~4× on decode while winning prefill"
  (§4.3.1) is about absolute decode against other silicon; it does not excuse a ratio
  measured against a comparator on the same GB10.

## Files

| file | what |
|---|---|
| `gb10-raw-trials.tsv` | every replicate of every arm, as emitted |
| `gb10-analysis.txt` | full analysis output (all arms, all replicates) |
| `ci-trial-ordering-analysis.txt` | discriminator 1, exact permutation tests |
| `ci-run-extracts.txt` | the two failing runs' `BEAT-OLLAMA-…` lines, verbatim |
| `estimator-skew-simulation.txt` | constant-rate prediction of median / best / best-median |
| `gb10-clock-trace-1024tok.txt` | 100 ms clock/power/temp trace of one cold `apr run` |
| `matched-clock-probe.txt`, `matched-clock-samples.txt` | apr vs ollama at matched clock |
| `gb10exp.sh`, `ramp.sh`, `matched.sh` | the measurement harnesses, exactly as run |

Analysis scripts are not committed — `evidence/` on main holds data artifacts, not
tooling. Everything above is reproducible from `gb10-raw-trials.tsv` with the methods
named in place: OLS and Theil-Sen on `t(N) = C + N/R` over the 24 ladder points; a
three-term quadratic for the curvature check; exact permutation over all 5040 orderings
for discriminator 1's Spearman p, Fisher-combined across runs; a 20 000-shuffle two-sided
permutation on the median difference for discriminator 2; and 200 000 draws of
median-of-7 / best-of-7 from `Delta ~ N(256000/106.4, 233*sqrt(2))` ms for the skew
simulation.
