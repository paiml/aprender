# Spike Wrap-Up Summary

## Session 4 — 2026-09-25

**Spikes processed:** 6. 021 PARTIAL; 022, 023 VALIDATED (`mcp-model-hosting-aws`); 024, 025, 026 VALIDATED (`llm-decision-classifier`, Laya)
**Feature areas:** AWS hosting for Rust model MCP servers · Laya decision model · Laya inference in Rust
**Skill output:** `./.claude/skills/spike-findings-aprender/` (SKILL.md now 3 ideas, 17 references, sources for 25 spikes)

### Processed Spikes

| # | Name | Type | Verdict | Feature Area |
|---|------|------|---------|--------------|
| 021 | kev-mcp-default-lambda | comparison | PARTIAL | AWS hosting (`aws-mcp-model-hosting.md`) |
| 022 | kev-mcp-fargate-scale-from-zero | comparison | VALIDATED | AWS hosting |
| 023 | lmi-minimum-footprint | standard | VALIDATED | AWS hosting |
| 024 | laya-vs-kev-few-shot | comparison | VALIDATED | Laya decision model (`laya-decision-model.md`) |
| 025 | laya-rust-forward-parity | standard | VALIDATED | Laya inference in Rust (`laya-rust-inference.md`) |
| 026 | laya-mcp-default-lambda | standard | VALIDATED | AWS hosting + Laya inference |

### Key Findings

**Cold start is bytes ÷ bandwidth, and the platform sets the bandwidth (021–023, 026).** The same instrumented pmcp
server ran on every host:
- Default Lambda caps S3 at ~80–95 MB/s per environment, so Kev-0.8B (3 GB) takes 43 s cold and Laya (0.84 GB) 12 s.
- Fargate pulls at 675–790 MB/s but spends 13–21 s provisioning every wake-up (Kev: 22–30 s).
- Weights baked into images are 7–18× worse on Lambda (800 s / 309 s via the lazy image store) and 4× worse on
  Fargate.

**LMI is fast but bills whole hosts (023).**
- There's no request-path cold start, and it's the fastest CPU measured (c9g Neoverse-V3: 0.34 s for Kev).
- An 8-vCPU function got 32-vCPU `c9g.8xlarge` hosts, three by default. **One is possible** after a first
  successful publish.
- LMI hosts are hidden from default `DescribeInstances`, and a failed first publish wedged its capacity provider.

**Laya is the better few-shot base, with a different adaptation shape (024).**
- A full fine-tune beats SetFit on stance (0.538 / 0.608 vs 0.512 / 0.561 at 16 / 64) and ties it on emotion (0.697
  vs 0.705), where Kev-4B got 0.577.
- Zero-shot it is only Kev-0.8B level, and head-only adapters cannot move its shared marker scorer (train accuracy
  0.52–0.55 at any lr).
- Per-tenant artifact: a full 0.84 GB checkpoint. Every fine-tuned run is over-confident.

**Laya ports cleanly to Rust (025–026).**
- ~400 lines on `gemm_blis`; probabilities to 3.8e-6 and ids 14/14 on the first run; the window rule proven by
  mutation.
- 1.6× faster than Rust Kev-0.8B; on Lambda, 0.75 s warm on Graviton2 at ≈ $0.0001 per decision.
- It is spike code: aprender has no ModernBERT.

### Open Items Surfaced (not spiked)
- **Productise Laya in-tree**: choose a home (realizar-first → `aprender-serve` unless an exception is argued),
  workspace lints, a contract, CI tests from the ladder/tokenizer/mutation rungs, a fine-tuned checkpoint, and the
  typed-decisions and multilingual variants.
- **Batch a request's questions into one pass** (3-question ticket: 3.8 s on G2); refit temperature after FT.
- **MCP Task front door + Fargate worker** for multi-GB models; LMI with several models packed on one provider.
- Land the spike-016 upstream sync (the trueno build every Rust spike since 017 depends on is local-only).

---

## Session 3 — 2026-09-23

**Spikes processed:** 5 (015 PARTIAL, 016 VALIDATED, 017 VALIDATED, 020 VALIDATED, 019 PARTIAL); 018 was dropped before it was built
**Idea:** `llm-decision-classifier`
**Feature areas:** Kev few-shot evaluation · Upstream sync · Qwen3.5 decision inference in Rust · LLM classifier on Lambda / Lambda Managed Instances
**Skill output:** `./.claude/skills/spike-findings-aprender/` (SKILL.md now 3 ideas, 14 references, sources for 19 spikes)

### Processed Spikes

| # | Name | Type | Verdict | Feature Area |
|---|------|------|---------|--------------|
| 015 | kev-vs-setfit-few-shot | standard | PARTIAL | Kev few-shot evaluation (`kev-few-shot-evaluation.md`) |
| 016 | upstream-sync-qwen35 | standard | VALIDATED | Upstream sync (`upstream-sync.md`) |
| 017 | kev-rust-forward-parity | standard | VALIDATED | Qwen3.5 decision inference (`qwen35-decision-inference.md`) |
| 020 | qwen35-batched-prefill | standard | VALIDATED | Qwen3.5 decision inference (`qwen35-decision-inference.md`) |
| 019 | kev-lambda-inference | standard | PARTIAL | LLM classifier deployment (`llm-classifier-lambda-deployment.md`) |

### Key Findings

**Kev complements SetFit (015).** Kev wins at 0–16 shots on both tasks. On stance, Kev-4B zero-shot
(F_avg 0.607) already beats SetFit trained on 64 shots (0.561), and 8 shots of per-class bias (3
floats) lift it to 0.642. SetFit wins emotion at 32+ shots (0.705 vs 0.577 at 64): Kev's
frozen-backbone adapters plateau, while SetFit's contrastive fine-tune keeps scaling. 0.8B is only
SetFit-level, so the gains need 4B, which is 5× slower and 100× bigger. The strongest adapters (bias,
head_ft) train in seconds on CPU on top of unchanged released weights. head_ft is over-confident
(ECE up to 0.37) until its temperature is refit.

**The upstream sync is tractable (016).** 280 commits produced 34 conflicts, resolved by five rules
plus three fixes that only `cargo check` surfaces. Upstream's `--features setfit` sweep fixed contract
commands of ours that had been running zero tests. The one red test (a stale SetFit golden) is
already red on the un-merged fork, as a control worktree proved. The merge `895c654de` is local and
not pushed.

**The handoff works with off-the-shelf tools (017).** PEFT fp32 merge → llama.cpp converter → upstream
`Qwen35Model` with a hidden-state readout → 40-line pointer head. Probabilities match to 1.5e-6 and
argmax is 12/12 on the first run. It needed two upstream patches, one of them an upstream-worthy
defect: llama.cpp counts the MTP block in `block_count`, so every freshly converted Qwen3.5 GGUF fails
to load. As shipped, token-at-a-time costs 79 ms per token, 5–7 s per decision.

**The batched prefill makes Kev servable on CPU (020).** Projections run as one GEMM per layer over
the row, with W borrowed zero-copy as the GEMM's A operand. An 87-token decision drops from 6.6 s to
**0.36 s on 6 threads** (18×) with parity unchanged. The decisive fix was running the DeltaNet
recurrence **in parallel per head**, found by phase timers after a sampling profile blamed trueno's
thread cap. The prefill is F32-only; a BF16 GEMM is needed for 4B.

**Kev-0.8B fits default Lambda; Kev-4B does not (019).** The binary is 1.4 MB and the weights 3 GB;
steady RSS is 3.9 GB (7.1 GB transient); the first decision completes 0.82 s after process start, and
short decisions take 0.2–0.4 s on 6 M4 threads (est. 1–1.5 s on Graviton2). Kev-4B needs 16.8 GB f32,
which exceeds the 10 GB ceiling. Cheap loader wins: skip the unused 1 GB `lm_head`; don't keep both
the mmap and owned copies (−2.9 GB).

**Lambda Managed Instances, added at wrap-up on the user's direction.** LMI lifts the ceiling to
32 GB / 16 vCPU with Graviton4, has no per-request cold starts, and runs concurrent requests in one
Rust process. That makes Kev-4B f32 placeable, but only after the loader fixes, because the as-built
~2.35× load transient would be ~39 GB. It costs always-on capacity (min 3 environments) plus a 15 %
management fee, so the small MCP servers stay on scale-to-zero Lambda. Nothing on LMI is measured.

### Open Items Surfaced (not spiked)

- **Build before 4B**: BF16 prefill GEMM; zero-copy F32 loading and skipping the tied `lm_head`; a
  state-prefix cache (a 3-question ticket pays the state 3×); a Rust Qwen tokenizer with Kev's
  `<|x|>` → `<¦x¦>` escaping.
- **Deploy spike on LMI**: a real Graviton4 capacity provider with Kev-0.8B, then 4B. Measure init
  weight load from S3, decision latency per vCPU setting, and per-environment concurrency vs throttles.
  Settle the unverified items: packaging limit, `/tmp` size, Function URL ingress, pmcp.run support.
- **Business-shaped evaluation**: support routing with policy criteria, where Kev's descriptions should
  matter more than on academic sets.
- **Upstream contributions (checkpoints)**: the MTP `nextn_predict_layers` loader fix; the batched
  prefill (time-to-first-token for `apr run`/`apr chat` on the hybrid); regenerate the stale SetFit
  golden on the fork before the real sync PR.

---

## Session 2 — 2026-09-20

**Spikes processed:** 4 (011–014, all VALIDATED)
**Idea:** `forecast-exogenous-inputs`
**Feature areas:** Prophet external regressors · NeuralProphet exogenous inputs (events, regressors) · No-argument bitwise invariance gate
**Skill output:** `./.claude/skills/spike-findings-aprender/` (SKILL.md now 2 ideas, 10 references, sources for 14 spikes)

### Processed Spikes

| # | Name | Type | Verdict | Feature Area |
|---|------|------|---------|--------------|
| 011 | prophet-regressor-parity | standard | VALIDATED | Prophet external regressors (`prophet-external-regressors.md`) |
| 012 | no-arg-bitwise-invariance | standard | VALIDATED | No-argument invariance gate (`no-argument-invariance-gate.md`) |
| 013 | np-events-autograd | standard | VALIDATED | NeuralProphet exogenous inputs (`neuralprophet-exogenous-inputs.md`) |
| 014 | np-gap-imputation-regressors | standard | VALIDATED | NeuralProphet exogenous inputs (`neuralprophet-exogenous-inputs.md`) |

### Key Findings

**The change request was re-priced before a line was written (011).** Four premises checked against
the consumer's own pinned tag `aprender-forecast-v0.63.0` (`fdf6b1802`), not HEAD. Three refuted:
NeuralProphet does not silently ignore `holidays` (`forecast.rs:117-130` already refuses); a
holidays parity fixture *is* committed (`peyton_holidays_prophet140.json`); and `fitted_forecast`
does not exist — the public door is `forecast(&ForecastArgs)`.

**Prophet regressors are an additive change, not a restructure (011).** A 219-line prototype that
touches nothing under `crates/` reaches full parity on both fixtures: column order identical at 24
and 30 columns, `prior_scales`/`s_a`/`s_m` exact `0.0`, yhat at Python MAP 4.5e-16 of `y_scale`,
9–12 components worst 3.6e-11, fit slack −3.42 / −2.22 against a 0.5 bar. The optimiser and gradient
needed no change. Column order is **seasonalities → holidays (name-sorted) → regressors (insertion
order)**; `std` is pandas `Series.std()`, **ddof = 1**, not numpy's default. The only real signature
change is a per-row value channel on `predict` — `feature_row` is untouched.

**A collinear driver has no reproducible lift (011).** `discount`, a cosine of period 6 months, is
r = +0.999 with `yearly_delim_4` — a harmonic of the yearly seasonality. Two optimisers at
equal-or-better objective report its coefficient 2× apart. This lands on the CR's own verification
plan, which backtests "with and without" each driver and reports lift. A pairwise correlation cutoff
is too loose (0.759 `price` was flagged identifiable and is still 5× off) — use condition number or
per-column VIF, and surface the diagnostic beside each regressor.

**No-argument invariance falls out of the column-order decision (012).** Because regressor columns
append, no existing index moves and the plumbing is provably inert at zero regressors: `Design`,
`yhat`, `trend` and all shared components bit-identical against the crate's untouched `predict` on
three datasets. 8/8 door cases reproduce on repeat; the signature is mutation-proven to detect a
1-ULP change and an extra component key. `fit_seconds`/`predict_seconds` are excluded (wall-clock);
f64s hash by `to_bits()`. Bands are **not** in the mechanism test — add them when the feature lands
in-crate. The harness plus `baseline.json` is the release gate for every tag.

**DEFECT — the door's budget promise does not survive events (013).** An additive `Linear(E,1)`
block composes beside `NpModel` without forking it, recovers a planted `+8.0` to within 5.8 % on all
six columns, is seed-42 bit-identical (seed 43 differs), and grows the tape by a fixed +5 that does
not scale with E. But `train_cost(n_samples, epochs, n_lags)` has **no event term**: measured work is
linear in E (`µs/step ≈ 12.9 + 0.085·E`) while the priced cost stays at 132 000, so at
`MAX_HOLIDAY_COLUMNS` (1 000) a request buys **7.6×** the work it was priced at. `request_train_cost`
must take `n_event_cols` and `a_neuralprophet_request_over_the_train_cost_bound_is_refused` must be
re-derived **before** events ship. The shape transfers; the constant is one machine, lag-free.

**The proposed `RegressorArg` contract does not cover NeuralProphet with lags (014).** NP trains on
an imputed daily grid denser than `ds`. Lag-free trains on observed rows only, so the imputed-day
regressor value is **never read** — all four fill rules including a garbage probe are bit-identical.
With `n_lags = 7` every grid row is a sample: two *defensible* rules differ by 10.48 on scale 35.32
(~30 %), and a garbage value on 12 % of days **flips the sign of both coefficients**. The CR's
`values.len() == ds.len() + horizon` leaves 109 values undefined. Three ways out, ranked: refuse /
require grid-complete values / impute-and-disclose (carry-forward for binary — linear interpolation
of a binary driver invents a "0.5 promo" day).

**Spike 012 voided spike 014's original premise mid-session.** `forecast.rs:421-423` refuses
`freq != "D"` on the neuralprophet arm, pinned by `neuralprophet_refuses_non_daily_freq` — so the
CR's P5 Q3 ("how do weekly and month-start series map onto NP's daily grid?") is answered by a
refusal, and 014 was re-scoped to the daily-gap case before building.

### Open Items Surfaced (not spiked)

- **Blocking events:** add an event-column term to `train_cost`/`request_train_cost`, calibrate the
  per-column constant on the deployment target, round the bound up, re-derive the C-08 invariant in
  `contracts/forecast-tool-boundary-v1.yaml`.
- **Unmeasured:** NP event blocks with `n_lags > 0` (`predict_ar_recursive` / `predict_ar_1step`
  interaction with stationarised lags); multiplicative event mode; uncertainty bands in the
  part-C inertness comparison.
- **Contract hygiene:** bind the components rung relative to `y_scale` on large-scale series
  (`components_via_python_params_abs` passes with only 3× headroom on `retail_sales`).
- **Diagnostics:** a per-regressor identifiability figure (condition number / VIF) in the response,
  so an operator cannot read a lift that will not hold.

---

## Session 1 — 2026-09-05

**Spikes processed:** 10 (001–010, all VALIDATED)
**Idea:** `prophet-forecast-mcp`
**Feature areas:** Prophet port · NeuralProphet-lite · Forecast MCP thin server · Chronos zero-shot ports · Chronos MCP server · Forecaster evaluation and routing · NEON GEMM kernel
**Skill output:** `./.claude/skills/spike-findings-aprender/` (SKILL.md, 7 references, sources for 10 spikes)

### Processed Spikes

| # | Name | Type | Verdict | Feature Area |
|---|------|------|---------|--------------|
| 001 | prophet-map-fit-lbfgs | standard | VALIDATED | Prophet port (`prophet-fit-and-predict.md`) |
| 002 | neuralprophet-autograd | standard | VALIDATED | NeuralProphet-lite (`neuralprophet-autograd.md`) |
| 003 | prophet-intervals-and-components | standard | VALIDATED | Prophet port (`prophet-fit-and-predict.md`) |
| 004 | forecast-mcp-thin-server | standard | VALIDATED | Forecast MCP thin server (`forecast-mcp-thin-server.md`) |
| 005 | chronos-bolt-tiny-parity | standard | VALIDATED | Chronos zero-shot ports (`chronos-zero-shot-port.md`) |
| 006 | chronos-vs-prophet-holdout | standard | VALIDATED | Forecaster evaluation and routing (`forecaster-evaluation-and-routing.md`) |
| 007 | chronos-mcp-thin-server | standard | VALIDATED | Chronos MCP server (`chronos-mcp-server.md`) |
| 008 | neon-gemm-microkernel-upstream | standard | VALIDATED | NEON GEMM kernel (`gemm-neon-kernel-upstream.md`) |
| 009 | chronos-2-parity | standard | VALIDATED | Chronos zero-shot ports (`chronos-zero-shot-port.md`) |
| 010 | forecast-server-concurrency | standard | VALIDATED | Forecast MCP thin server (`forecast-mcp-thin-server.md`) |

### Key Findings

**Prophet (001, 003).** The Rust port of Prophet 1.4.0's Stan model is bit-for-bit on data prep and
predict and reaches the MAP with `aprender::optim::LbfgsF64` — but only with objective ÷ T, a
non-finite guard, `Stalled` accepted as success, restarts from the stall point (≤ 8 rounds) and an
x-keyed value+gradient cache. Forecasts land inside Prophet's own Newton-vs-L-BFGS band on four
datasets; intervals match to ~1 %; components reconstruct yhat to 1e-16. Constant `y` diverges and
must be refused before fitting. Two core candidates: Stan-style initial step / non-finite
backtracking in `WolfeSearch`, and stop re-evaluating `f`/`∇f` at `x` inside the line search.

**NeuralProphet (002).** NP's default model trains on the existing f32 autograd to NP-level holdout
error (0.451 vs 0.461) in 0.04 s vs 20.9 s. Core `SmoothL1Loss` is detached from the graph (defect);
Huber is built from ops with a constant mask. Full-batch training collapses; lr is selected by train
loss; ~4× NP's auto epochs for linear AR. The AR-Net (0.25 vs naive 0.35 one step ahead) is where NP
earns its keep; trend + seasonality alone is no better than Prophet.

**Forecast MCP server (004, 010).** One typed stateless `forecast` tool over both models, refusals at
the boundary, per-round iteration cap + 15 s wall-clock budget (a 20k fit ran 66 s uncapped),
diagnostics in the response, and a same-origin page that is itself an MCP client. Peyton round trip
1.41 s. Under load every response is bit-identical (8/8 ×3, 16/16) — but pmcp 2.19's streamable-HTTP
router holds one `Arc<Mutex<Server>>` across each tool call, so a single router serialises fits;
a pool of 8 routers behind a round-robin `fallback` gives 3.9× / 6×.

**Chronos ports (005, 009).** Chronos-Bolt (tiny, small) and Chronos-2 run in Rust with no torch and
match `chronos-forecasting` 2.3.1 to f32 rounding (Bolt 1e-6 abs; Chronos-2 2e-5 of scale) on the
main series, 6–7 edge probes and the autoregressive rollout — because a bottom-up parity ladder made
every architectural assumption checkable at its own rung. The 2025 rollout scheme (9 re-quantiled
paths) differs from the median-only scheme by 0.5. Keep only transposed weights (Chronos-2 peaked at
1.45 GB with both). The rayon GEMM splits M only: ≤ 1.4× below ~500 tokens.

**Chronos server (007).** Embedded tiny-f16: 24 MB binary, 52 ms cold start to first forecast,
18.5 ms per 2048-point forward, parity 9.5e-7 through the server; f16 costs 0.11 % of std. Small-f16:
103 MB / 280 ms / 98 ms (S3 or container, not a Lambda zip). Horizon > 64 refused unless
`allow_long_horizon`, then a warning citing the 006 measurement.

**Evaluation (006).** 17 rolling-origin windows on four series: NP-lite 1.029, Prophet 1.094,
Bolt-small 1.178, Bolt-tiny 1.238 mean MASE; baselines 1.69 / 2.02. Chronos wins monthly (0.74–0.80)
and is within 0.1 MASE inside 64 steps; past 64 its rollout degrades to 1.73. Every nominal 80 % band
covers 0.60–0.69 out of sample. Routing rule: monthly or ≤ 64 steps → Chronos, else NP-lite/Prophet.
A rolling-origin gate belongs in the release check.

**NEON GEMM (008).** `gemm_blis` on aarch64 ran the scalar microkernel (7.4 GFLOP/s, 14× behind faer);
the one NEON kernel in the tree was uncallable (wrong panel stride). A 78-line 8×6 kernel honouring
the packing contract gives 65–77 GFLOP/s (0.7× faer), 7.3–10.7× on the crate's own criterion bench,
Chronos forward 137 → 21 ms, rollout 6.4 → 0.97 s, parity unchanged. Packaged with
`contracts/neon-blis-v1.yaml`, three tests, and a PR body on `perf/neon-gemm-8x6-microkernel`
(worktree `~/Development/machine-learning/aprender-neon-upstream`); the only red test is a
pre-existing `Instant::now()` resolution flake reproduced 4/8 on untouched `upstream/main`.
Opening the PR is a checkpoint.

### Open Items Surfaced (not spiked)

- Core: fix `SmoothL1Loss` graph connectivity + a connectivity test for every `nn::loss`; add
  `where`/`clamp`, `cat` to the autograd; `WolfeSearch` initial-step scaling / non-finite backtracking;
  `LbfgsF64` double evaluation at `x`; f64 proximal solver.
- Compute (upstream): 8×12 / 12×8 NEON tile with per-arch `NR`; partition the parallel GEMM along N
  for M ≤ 256; `test_brick_profiler_reset_v2` timer flake; streaming safetensors loader.
- pmcp (upstream): take the router lock only to route, not across the tool future.
- Forecasting features: `freq` H (fractional days), country-holiday calendars, NP quantile regression
  and `n_forecasts > 1`, Chronos-2 long-horizon unrolling and covariates, `model: auto`.
