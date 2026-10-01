# Spike Manifest

## Ideas

### prophet-forecast-mcp
Implement Facebook Prophet (piecewise-linear/logistic trend with Laplace-prior changepoints,
Fourier seasonality, holidays, MAP fit via L-BFGS, simulated-changepoint uncertainty) and
NeuralProphet (the same decomposition plus AR-Net and lagged/future regressors, trained by
gradient descent) natively in aprender, and expose them the way SetFit is exposed: a thin,
single-purpose MCP server (pmcp, Lambda/pmcp.run) for time-series forecasting.

**Requirements:**
- MCP serving shape is a STATELESS `forecast` tool: one call carries `ds[]`, `y[]`, horizon
  (and freq); the server fits and forecasts inside that call. No fit → artifact → forecast
  round-trip (decided 2026-09-04 at spike alignment).
- Both Prophet (MAP / L-BFGS) and NeuralProphet (autograd / AdamW) are in scope; NeuralProphet
  is spiked in this session, not deferred.
- Chronos (zero-shot foundation model) joins the idea as a THIRD forecaster with its own thin server (one model per server); Chronos-Bolt first, Chronos-2 later (decided 2026-09-04).
- Correctness bar for the Prophet port is parity with Python Prophet 1.4.0 on the Peyton Manning
  dataset (fixture: `001-prophet-map-fit-lbfgs/fixtures/peyton_manning_prophet140.json`), not
  self-consistency.
- Frontier session 2026-09-05: build order 008 (NEON GEMM kernel, an upstream contribution) → 007 (Chronos thin server) → 009 (Chronos-2) → 010 (concurrency probe). The 008 kernel lives in `crates/aprender-compute` on a branch cut from `upstream/main`; opening the PR is a checkpoint, not automatic.

### forecast-exogenous-inputs
Let operators supply the information the forecast models cannot infer: known future events
(holidays) and numeric drivers (price, promotions). Requested by Forecast Coach, an MCP app that
consumes `aprender-forecast` as a git dependency pinned by tag and calls the crate's one stateless
door for both `prophet` and `neuralprophet`. Extends the ports built under `prophet-forecast-mcp`.
Two of the change request's four premises were refuted against its own pinned tag
(`aprender-forecast-v0.63.0` = `fdf6b1802`) before any spike was built: the NeuralProphet
`holidays` silent-ignore does not exist (`forecast.rs` already refuses with "holidays is
prophet-only"), and the Prophet parity ladder already carries a holidays fixture
(`peyton_holidays_prophet140.json`, two holidays with `upper_window: 1`, measured
`X 0.00e0 (K=30)` at rung 1 and `playoff 0.0e0, superbowl 0.0e0, holidays 0.0e0` at rung 3).
The real work is regressors on both models, events on NeuralProphet, and proving a tag bump is safe.

**Requirements:**
- **Byte-identical when the new arguments are absent.** Every committed fixture must reproduce its
  pre-change `ForecastResponse` exactly after the plumbing lands. This is the consumer's acceptance
  gate, not a nice-to-have (CR constraint 1, 2026-09-20).
- **No new required arguments.** Every new field is `Option<_>` with a serde default;
  `#[serde(deny_unknown_fields)]` stays as it is.
- **Errors, not silence.** An unsupported combination (a driver on a model that cannot use it) is
  refused at the door with a message naming the limitation — the existing D-11 pattern.
- **One `RegressorArg` shape serves both models**, so the caller sends one argument to `prophet`
  and `neuralprophet` alike.
- **Correctness bar for regressors is parity with Python Prophet 1.4.0**, measured by extending the
  existing rung ladder (data prep -> predict-at-Python-params -> components), not self-consistency.
- **One tag per release**; the consumer bumps one line plus the lock entry and builds `--locked`.
- The public entry point is `forecast(&ForecastArgs)`. `fitted_forecast` does not exist at HEAD or
  at the pinned tag — the CR names an API the crate does not ship (recorded 2026-09-20).

### llm-decision-classifier
Evaluate Kev (https://github.com/jaredpalmer/kev, Apache-2.0), the open replica of the Jev decision
model, as a classifier family in aprender alongside SetFit. Kev is a Qwen3.5 base plus a rank-16 LoRA
and a pointer head that scores each option's `</opt>` hidden state against the question's `<decide>`
hidden state; it answers yes/no (`noul`), `choice` and `score` questions with calibrated
probabilities and never generates text. Upstream aprender (`paiml/aprender`) already runs the
Qwen3.5 Gated-DeltaNet hybrid on CPU and CUDA, but our fork is 280 commits behind it (2026-09-23),
and upstream's `Qwen35Model` is GGUF, token-at-a-time and logits-only: no hidden-state readout and no
training path through DeltaNet.

**Requirements:**
- **Few-shot steering is the product.** The business supplies a handful of labelled examples
  (SetFit's 8–64 per class regime) to add its own knowledge and bias; massive datasets are out of
  scope. Zero-shot Kev with `criteria` descriptions is the baseline few-shot must beat, since it
  already carries some steering (recorded 2026-09-23).
- **Training may stay in Python; inference must be Rust on aprender.** Fine-tuning is a back-office
  process whose only output is weights; the served path (speed, security, AWS Lambda) is Rust.
  Rust-side training is a bonus, not a requirement (decided 2026-09-23 at spike alignment).
- **The Python-to-Rust weight handoff is part of the contract**: a Kev checkpoint (adapter + head)
  must export to an artifact the Rust inference path loads, with probability parity to Python fp32.
- **Qwen3.5 support comes from upstream, not a fork-local re-port.** Sync `paiml/aprender` first
  (spike 016) and extend its `Qwen35Model` rather than writing a second implementation (OPS-03).
- **Lambda is the deployment target**: size, memory, cold start and latency are measured against
  Lambda limits, not assumed.
- **Kev needs a batched prefill before it is servable on CPU** (spike 017): upstream's token-at-a-time
  Qwen3.5 path costs 5–7 s per decision. Spike 020 prototypes it before 019 measures Lambda; spike 018
  (Rust head training) is dropped because training stays in Python (decided 2026-09-23).
- **Lambda is measured by a local proxy** (6 threads, 10 GB cap), not a real AWS deploy, for this session
  (decided 2026-09-23).
- **Laya (github.com/NandhaKishorM/laya, Apache-2.0) is evaluated as an alternative base to Kev**: an encoder-only
  ModernBERT-large (421M) decision model with the same `choice`/`score`/`noul` types and `/v1/systemone` wire. The
  English root and `typed-decisions` checkpoints are in scope (multilingual is not). Build order 024 (quality vs
  Kev/SetFit on the 015 rows) -> 025 (Rust port, parity) -> 026 (MCP on default Lambda); stop and report after 024
  if Laya loses (decided 2026-09-25).
- **Lambda Managed Instances is the intended host for Qwen-sized models** (Kev-4B and further Qwen-based
  models), alongside the existing smaller Rust MCP servers on AWS (recorded 2026-09-23 at wrap-up). LMI lifts
  the per-function ceiling from 10 GB / ~6 vCPU to 32 GB / 16 vCPU (Graviton4 available), removes per-request
  cold starts and serves requests concurrently in one Rust process. Nothing on LMI has been measured yet.
- **Laya's calibration gate fails on split shift, not on the cap** (spike 027, 2026-09-26): on TweetEval stance the
  slice-fitted T calibrates in-distribution held-out rows (ECE 0.05–0.07) but not the SemEval test split, which wants
  T 8–30; raising `TEMP_MAX` 5 -> 10 rescues 1/3 runs at best. Choosing the demo's eval distribution is a user decision.

### mcp-model-hosting-aws
Host aprender's Rust models on AWS as thin, stateless pmcp MCP servers that AI agents call — small models
(SetFit, Chronos-Bolt, forecasters) and the multi-GB Qwen-based decision models (Kev and successors) alike — and
choose, per model, between default Lambda, ECS on Fargate scaled to zero, and Lambda Managed Instances on measured
cold start, latency and idle cost rather than on assumptions. Extends `llm-decision-classifier` (Kev is the
test model) and the pmcp.run deployment of the forecast servers.

**Requirements:**
- **Every model is wrapped as an MCP server on the pmcp SDK** (`~/Development/mcp/sdk/rust-mcp-sdk`) so it
  integrates with AI agents; the thin one-model-per-server rule holds (recorded 2026-09-23).
- **Idle cost matters**: 24/7 capacity x 3 AZs + the 15 % LMI fee is the cost to beat; scale-to-zero is preferred
  when its cold start is acceptable, and the cold start is measured, not assumed (2026-09-23).
- **Candidate hosts are default Lambda, ECS on Fargate (scale to zero) and Lambda Managed Instances**; the
  decision is per model and may differ between small and large models (2026-09-23).
- Measured in us-east-1 (where pmcp.run deploys), arm64, all resources tagged `spike=021-023` and torn down or
  left idle-free after measuring.

## Spikes

| # | Idea | Name | Type | Validates | Verdict | Tags |
|---|------|------|------|-----------|---------|------|
| 001 | prophet-forecast-mcp | prophet-map-fit-lbfgs | standard | Given Peyton Manning, when Prophet's Stan objective + analytic gradient is minimized with `LbfgsF64`, then it converges and yhat matches Python Prophet 1.4.0 within tolerance | VALIDATED ✓ (inside Prophet's own Newton-vs-LBFGS band on 3 datasets; needs f/T scaling + non-finite guard) | prophet, lbfgs, changepoints, fourier |
| 002 | prophet-forecast-mcp | neuralprophet-autograd | standard | Given the same series, when trend + Fourier + AR-Net is trained with AdamW + SmoothL1 on the f32 autograd, then holdout MAE is comparable to 001 using only existing ops | VALIDATED ✓ (test MAE 0.451 vs NP 0.461, 500× faster; core SmoothL1Loss is detached) | neuralprophet, autograd, ar-net |
| 003 | prophet-forecast-mcp | prophet-intervals-and-components | standard | Given the 001 fit, when future changepoints are simulated and components decomposed, then the 80% band covers ~80% of a holdout and components sum to yhat; logistic, multiplicative and holidays fit | VALIDATED ✓ (bands within ~1% of Python, holdout coverage 0.81–0.83 both; L-BFGS needs restarts) | prophet, uncertainty, holidays |
| 004 | prophet-forecast-mcp | forecast-mcp-thin-server | standard | Given a pmcp thin server with one stateless `forecast` tool, when called with ds/y/horizon, then fit + forecast returns in under 2s for 3k points and a browser page charts it | VALIDATED ✓ (Peyton 1.41 s round trip; 3k pts 0.21 s; iteration cap + budget for 20k; page is an MCP client) | mcp, pmcp, latency, ui |
| 005 | prophet-forecast-mcp | chronos-bolt-tiny-parity | standard | Given `amazon/chronos-bolt-tiny` safetensors, when its T5 encoder-decoder (patch embedding, instance scaling, REG token, relative position bias, quantile head) is run in Rust, then the 9 quantiles match the Python `chronos-forecasting` pipeline on Peyton and air passengers within a committed tolerance, including the autoregressive rollout past 64 steps | VALIDATED ✓ (quantiles to 1e-6, rollout to 1.5e-5, 6 edge probes; 36 ms/forward plain Rust vs 5.9 ms torch; trueno GEMMs 4× slower than loops) | chronos, t5, zero-shot, safetensors, parity |
| 006 | prophet-forecast-mcp | chronos-vs-prophet-holdout | standard | Given the spike-001/003 series with rolling-origin holdouts (Peyton daily; air, retail monthly; wp_log_R daily), when zero-shot Chronos-Bolt (tiny and small) and the fitted Prophet / NeuralProphet-lite ports forecast the same windows, then MAE/MASE, 80% coverage/width and latency are compared per series with naive and seasonal-naive baselines, so the value of a Chronos server versus the fitted models is measured, not assumed | VALIDATED ✓ (17 windows: NP-lite 1.03, Prophet 1.09, Bolt-small 1.18 mean MASE; Chronos wins monthly and ≤64 steps, loses past 64; all 80% bands cover 0.60–0.69) | chronos, prophet, benchmark, holdout, coverage |
| 008 | prophet-forecast-mcp | neon-gemm-microkernel-upstream | standard | Given `gemm_blis` on aarch64 falls to `microkernel_scalar` because the only NEON kernel is 8×8 while panels are packed 8×6, when an 8×6 NEON FMA kernel honouring the packed-panel contract is added and dispatched under `cfg(aarch64)`, then it agrees with the scalar kernel within FMA rounding on the existing microkernel test matrix, `cargo test -p aprender-compute --lib` stays green, and a before/after table on spike-005 shapes (129×256×1024) plus square perf-gate shapes shows the packed GEMM beating plain loops, with the Chronos forward dropping from 36 ms toward torch's 6 ms | VALIDATED ✓ (7.4 → 65–77 GFLOP/s, 7.3–10.7× on the crate's own bench, 0.7× faer; suite green except a pre-existing timer flake reproduced upstream; Chronos forward 137 → 21 ms; PR branch ready) | gemm, neon, blis, upstream, performance |
| 007 | prophet-forecast-mcp | chronos-mcp-thin-server | standard | Given Chronos-Bolt embedded in a pmcp thin server exposing the spike-004 `forecast` shape, when called over streamable-HTTP and stdio with ds/y/horizon, then native quantile bands return in under 100 ms for a 2048-point context at horizon ≤ 64, horizon > 64 is refused unless explicitly allowed (with a warning in the response), the browser page charts it, and binary size and cold start with embedded weights (f32 vs f16) are measured for Lambda | VALIDATED ✓ (embedded tiny-f16: 24 MB binary, 52 ms cold start to first forecast, 18.5 ms forward on 2048 pts, parity 9.5e-7 through the server; small-f16 103 MB / 280 ms / 98 ms; f16 costs 0.1 % of std; horizon > 64 gated by allow_long_horizon) | chronos, mcp, pmcp, lambda, latency |
| 009 | prophet-forecast-mcp | chronos-2-parity | standard | Given `amazon/chronos-2` safetensors (120M, RoPE, arcsinh scaling, 21 quantiles, covariates), when ported on the spike-005 ladder (scaling → embeddings → hidden states → quantiles), then Python parity holds on Peyton, air and the edge probes, forward cost is measured with and without 008, and a verdict is given on whether 120M is servable in one binary | VALIDATED ✓ (Rust port matches Python to 2e-5 of scale on Peyton at 64/365/1024 steps and on 7 edge probes; 0.59 s/forward single-thread at 65 GFLOP/s vs torch 0.05 s on 10 threads; f16 costs 0.3–0.6 % of scale; 228 MB f16 / 1.45 GB peak RAM; parallel GEMM ≤ 2.1× because it splits M only) | chronos-2, t5, rope, parity |
| 010 | prophet-forecast-mcp | forecast-server-concurrency | standard | Given the spike-004 server, when 8 NeuralProphet and Prophet requests arrive concurrently over streamable-HTTP, then every response matches its sequential result and no fit is corrupted by another thread's autograd tape | VALIDATED ✓ (8/8 and 16/16 responses bit-identical under load; but pmcp's router holds one Arc<Mutex<Server>> across each tool call so the shipped server serialises — a pool of 8 routers gives 3.9× / 6× with identical outputs) | mcp, concurrency, autograd |
| 011 | forecast-exogenous-inputs | prophet-regressor-parity | standard | Given Python Prophet 1.4.0 with `add_regressor` on retail_sales plus binary, continuous and multiplicative regressors, when the shipped design-matrix path grows regressor columns, then column order, standardisation constants, `s_a`/`s_m`, X, predict-at-Python-MAP and every named component match, and the fit stays inside `fitted_objective_slack` | VALIDATED ✓ (additive change, no restructure: column order identical at 24 and 29 cols, `prior_scales`/`s_a`/`s_m` exact 0.0, yhat 4.5e-16 of y_scale, 9/11 components worst 3.6e-11, fit slack −3.42/−0.16 vs bar 0.5; std is pandas ddof=1 not numpy ddof=0; a driver collinear with trend/seasonality has no reproducible lift) | prophet, regressors, parity, design-matrix, identifiability |
| 012 | forecast-exogenous-inputs | no-arg-bitwise-invariance | standard | Given eight door cases across both models, growths, holidays and AR lags, when the regressor plumbing is present but no new argument is passed, then every deterministic `ForecastResponse` field is bit-identical — and the signature saying so is proven able to fail | VALIDATED ✓ (8/8 repeat calls bit-identical; mutation proof detects 1-ULP yhat/trend changes and an extra component key; zero-regressor plumbing bit-identical to untouched `predict` on 3 datasets. Bands not in the mechanism test. Found: NP refuses freq != D, which answers CR P5 Q3 and voids spike 014's premise) | invariance, determinism, delivery-gate, mutation-proof |
| 013 | forecast-exogenous-inputs | np-events-autograd | standard | Given NP-lite's `forward()`, when an additive `Linear(E,1)` event block is trained jointly on the f32 autograd, then a known synthetic event effect is recovered, fixed-seed runs are bit-identical, and the D-10 train-cost bound is measured against real work | VALIDATED ✓ (block composes without forking NpModel; all 6 weights recover the planted +8.0 within 5.8%, train loss 6.8x lower; seed 42 bit-identical twice, seed 43 differs; tape +5 fixed, no leak. **DEFECT: `train_cost` has no event term — cost is linear in E and at E=1001 (MAX_HOLIDAY_COLUMNS) a request buys 7.6x the priced work**. Lag-free only; AR paths unmeasured) | neuralprophet, events, autograd, train-cost, determinism |
| 014 | forecast-exogenous-inputs | np-gap-imputation-regressors | standard | Given a daily series with gaps, when a regressor is supplied on the caller's rows and NP trains on its imputed daily grid, then whether the imputed-day value is read is determined per `n_lags` and the forecast's sensitivity to the fill rule is measured | VALIDATED ✓ (RE-SCOPED after 012 refuted the W/MS premise. **Lag-free never reads it** — all 4 rules incl. a garbage probe are bit-identical. **Lagged reads all of it** — defensible rules differ by 10.5 on scale 35.3 (~30%), garbage flips both coefficient signs. The CR's `values.len() == ds.len() + horizon` does not cover NP's denser grid; refuse / require-grid-complete / impute-and-disclose) | neuralprophet, regressors, imputation, gaps, api-contract |
| 015 | llm-decision-classifier | kev-vs-setfit-few-shot | standard | Given the SetFit tasks at 8–64 shots/class, when SetFit, zero-shot Kev (0.8B/4B) and Kev with bias / head / LoRA adaptation are scored on identical rows, then we know whether few-shot Kev beats SetFit | PARTIAL ⚠ (Kev wins 0–16 shots on both tasks; Kev-4B zero-shot 0.607 beats SetFit@64 0.561 on stance and +3 floats of bias → 0.642; SetFit wins emotion at 32+ shots, 0.705 vs 0.577 @64; 0.8B is only SetFit-level; head adapters train in seconds on CPU; LoRA r2 works on MPS) | kev, setfit, few-shot, benchmark |
| 016 | llm-decision-classifier | upstream-sync-qwen35 | standard | Given the fork is 280 commits behind paiml/aprender, when upstream/main is merged in a worktree, then conflicts are enumerated, the workspace builds, and forecast/SetFit/Qwen3.5 suites stay green | VALIDATED ✓ (34 conflicts, 5 rules + 3 one-line semantic fixes; upstream now 0.69/0.70; 4305 tests pass; the one red SetFit golden is already red on the fork HEAD, proven by a control worktree; merge committed locally 895c654de, not pushed) | upstream, merge, qwen3.5 |
| 017 | llm-decision-classifier | kev-rust-forward-parity | standard | Given Kev-0.8B trained in Python, LoRA merged and exported to GGUF, when a decision row runs through upstream's Qwen35Model + a hidden-state readout + the pointer head in Rust, then probabilities match Python fp32 to ~1e-5 and CPU latency is measured | VALIDATED ✓ (probs 1.5e-6, 12/12 argmax, PEFT merge → llama.cpp converter → upstream loader; 2 upstream patches incl. an MTP block_count defect; BUT 79 ms/token token-at-a-time = 5–7 s per decision, dtype buys ≤1.8×; batched-prefill GEMM bound 0.73 s @85 tok on 14 cores) | kev, qwen3.5, parity, latency |
| 020 | llm-decision-classifier | qwen35-batched-prefill | standard | Given upstream's Qwen3.5 CPU forward is token-at-a-time, when each layer's projections run as one GEMM over the row and only the mixers per token, then parity holds and a short decision drops below 1 s on 6 threads | VALIDATED ✓ (85-token decision 6.6 s → 0.36 s on 6 threads, 18×; 915 tokens 73 → 3.9 s; parity vs torch unchanged 1.3e-6, 12/12; key fix was parallelising the DeltaNet recurrence per head, found by phase timers after a profile misled; F32-only — BF16 GEMM needed for 4B) | qwen3.5, prefill, gemm, performance |
| 019 | llm-decision-classifier | kev-lambda-inference | standard | Given the Rust Kev path (017 + 020), when it runs as a Lambda would (6 threads, 10,240 MB), then size, peak memory, cold start and latency are measured against Lambda limits | PARTIAL ⚠ (0.8B fits: 1.4 MB binary + 3 GB f32 weights, 3.9 GB steady / 7.1 GB transient, first decision 0.82 s from process start, 185–375 ms per short decision on 6 M4 threads, est. 1–1.5 s on Graviton2; 4B does not fit — 16.8 GB f32, BF16 GEMM missing, est. 6–9 s/decision on Lambda CPU; unused 1 GB lm_head and mmap copies are cheap wins) | kev, lambda, memory, latency |
| 021 | mcp-model-hosting-aws | kev-mcp-default-lambda | comparison | Given Kev-0.8B as a stateless pmcp MCP server on arm64 Lambda (10,240 MB), when tools/call hits a cold environment, then the init -> weights -> first-decision timeline and Graviton latency are measured for weights baked into the image vs streamed from S3 | PARTIAL ⚠ (S3: cold 42.8–43.2 s, 37.7 s of it the S3 download capped at ~80 MB/s per environment regardless of parallelism; warm 1.13–1.16 s on Graviton2 (once Graviton3); baked image 800 s first, 309 s second — 794 s reading 3 GB through the lazy image store; parity 4.8e-7) | lambda, cold-start, mcp, graviton |
| 022 | mcp-model-hosting-aws | kev-mcp-fargate-scale-from-zero | comparison | Given the same server as an arm64 Fargate task (8 vCPU / 16 GB), when started from zero, then RunTask -> pulled -> started -> weights -> first MCP answer is measured for S3 vs baked weights | VALIDATED ✓ (S3: first answer 22–30 s, of which 13–21 s is provisioning; S3 674–789 MB/s = 3 GB in ~4 s; warm 0.51 s G4 / 0.64 s G3; baked image 95–105 s (2.4 GB pull + unpack 75–81 s); weights never go in the image) | fargate, ecs, scale-to-zero, cold-start |
| 023 | mcp-model-hosting-aws | lmi-minimum-footprint | standard | Given an arm64 LMI capacity provider and the Kev function (16 GB / 8 vCPU), when published and scaled to min=max=1, then the instances run, time to Active, warm latency and whether one instance suffices are measured | VALIDATED ✓ (one instance IS possible after the first publish — 3 -> 1 in ~5 min; but 8-vCPU envs land on c9g.8xlarge (32 vCPU / 64 GB): default 3 hosts = 96 vCPU; warm 0.34 s, S3 875–962 MB/s, no request-path cold start; LMI hosts hidden from DescribeInstances without IncludeManagedResources; a failed first publish wedged its provider; /tmp 512 MB default; us-east-1c refused) | lambda-managed-instances, cost, graviton |
| 024 | llm-decision-classifier | laya-vs-kev-few-shot | comparison | Given the spike-015 stance/emotion rows paired with SetFit and Kev, when Laya (en + typed-decisions) is scored zero-shot, with 015's head-only adaptations and with a pre-declared full fine-tune, then we know whether it matches Kev and SetFit | VALIDATED ✓ (full FT from the en root beats SetFit on stance 0.538/0.608 vs 0.512/0.561 @16/64 and ties emotion 0.697 vs 0.705 @64 — where Kev-4B got 0.577; zero-shot ≈ Kev-0.8B; head-only adaptation cannot move its shared marker scorer; 0.84 GB fp16, 65 ms/decision torch CPU vs Rust Kev-0.8B 360 ms; FT over-confident ECE 0.17–0.38; per-tenant artifact is a full checkpoint) | laya, modernbert, few-shot, benchmark |
| 025 | llm-decision-classifier | laya-rust-forward-parity | standard | Given Laya's F16 safetensors, when ModernBERT + head + scorer and the sequence builder are ported to Rust on trueno BLIS, then ids equal Python's, probs match torch fp32 to ~1e-5 and CPU latency is measured | VALIDATED ✓ (ids 14/14 incl. injection/unicode/512-truncation; probs 3.8e-6, argmax 14/14 on the first run; local window |i-j|<=64 proven by a 63/65 mutation that breaks only layer 1; 226 ms @85 tok on 6 threads = 1.6x faster than Rust Kev-0.8B; GEMM-bound at ~61 GFLOP/s/core; torch's 62 ms is Apple AMX; no conversion step — safetensors + tokenizer.json load directly) | laya, modernbert, parity, rust |
| 026 | llm-decision-classifier | laya-mcp-default-lambda | standard | Given the Rust Laya behind a stateless pmcp `decide` tool taking a real /v1/systemone request, when run on default arm64 Lambda (10 GB and 4 GB) cold and warm, then cold start, Graviton latency, memory and cost are measured against Kev's 43 s / 1.15 s | VALIDATED ✓ (10 GB: cold 11.3–12.0 s — 846 MB F16 from S3 at 94 MB/s — vs Kev 43 s; warm tweet 0.75 s G2 / 0.49 s G3 vs Kev 1.15 s; 3.5 GB peak vs 8 GB; ≈$0.00010 vs $0.00015 per decision; 4 GB is the same GB-s price and 2.4x slower; parity ≤ 4.5e-6 through the real text API; multi-question and 512-token requests are multi-second on G2) | laya, lambda, cold-start, mcp, cost |
| 027 | llm-decision-classifier | laya-calibration-slice-and-tcap | standard | Given TweetEval stance, when Laya en-root is fully fine-tuned on s16 and s64 (12- vs 48-row calibration slices) with the fixed 12-epoch and early-stopping recipes over seeds 13/17/23, then post-calibration ECE at T<=5 and T<=10, fitted-T vs eval-optimal T and macro-F1 margin show whether any recipe clears ECE<=0.10 with margin>=0.05 within Laya's cap | PARTIAL ⚠ (MEASUREMENTS, not gate runs. 0/12 main runs pass at T<=5 or T<=10; s64 early-stop passes T<=5 on 1/6 incl. replicates (ECE 0.143 ± 0.028), s64 4-epoch passes T<=10 on 1/3; the cap binds only for fixed-epoch recipes (slice T 12–26), early-stop slices fit T<5 in 8/9; cause is the SemEval train->test shift — same checkpoints at their slice T score ECE 0.05–0.07 on 459 in-distribution held-out rows; an MPS replicate flipped a FAIL to a PASS (margin ±0.07, ECE ±0.05); candidates for 028: s64-es12-seed17-rep, s64-r1-seed13, s16-r1-seed13) | laya, calibration, temperature-scaling, gate, distribution-shift, mps-noise |
| 028 | llm-decision-classifier | laya-packability-noise-floor | standard | Given spike 027's candidate checkpoints, when each is re-scored over the full eval set in Rust fp32, torch fp32 and float64, then we know whether pack's 1e-5 probability bar and the ladder's 1e-4 final-block bar leave room for a gate-passing model, and if not, what noise-based bar the measurements support | INVALIDATED ✗ (the only gate-passing checkpoint, s64-es12-seed17-rep, is REFUSED RescoreDrift 1.445e-5 by `just laya-pack` while torch's own fp32 is 6.3e-5 from float64 and Rust is closer to exact; 2.98e-5 on 027's 459-row in-dist set; ECE/F1/argmax unchanged to 1e-7; final-norm 1e-4 has 0.00–0.04× headroom on real rows; proposed bar max(1e-5, 4 × max\|torch32−f64\|), min margin 3.2× on every measured checkpoint, or fixed 1e-4 at 2.1×; x86_64 unmeasured) | laya, parity, pack, rescore, fp32-noise-floor, float64-reference, tolerance |
