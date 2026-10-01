# Phase 6: Native Time-Series Forecasting Stack - Context

**Gathered:** 2026-09-05
**Status:** Ready for planning
**Provenance:** Assembled from the user's spike-alignment decisions in `.planning/spikes/MANIFEST.md`
(idea `prophet-forecast-mcp`, decided 2026-09-04/05) and the wrap-up skill
`.claude/skills/spike-findings-aprender/` (ten VALIDATED spikes), not from a `/gsd-discuss-phase`
session. D-01..D-05 are the user's recorded decisions verbatim; D-06 onward are the build rules the
spikes established and the user ratified by asking for "the real build" from them. Run
`/gsd-discuss-phase 6` to amend any of them.

<domain>
## Phase Boundary

This phase turns ten validated spikes into shipped crates: a `aprender-forecast` library holding
the pure-Rust Prophet 1.4.0 port, the NeuralProphet-lite port and the Chronos-Bolt zero-shot
forward; a thin `aprender-mcp-forecast` pmcp server exposing Prophet and NeuralProphet behind ONE
stateless `forecast` tool; and a thin `aprender-mcp-chronos` server exposing Chronos-Bolt behind the
same request/response shape with the weights embedded in the binary. Each port is proven against its
Python original by committed oracle fixtures, and each server is exercised end-to-end over stdio
and streamable-HTTP, the way `crates/aprender-mcp-setfit` already is.

**In scope:** the three crates and their contracts; porting the spike sources (`prophet.rs`,
`np.rs`, `dates.rs`, `bolt.rs`, `safetensors.rs`, the server `lib.rs`/`main.rs`/`static/index.html`/
`tests/e2e.rs`, `build.rs`) into workspace members that pass the workspace gates; the parity
fixtures as CI tests; the router pool for the fit server; the horizon gate and embedded-weight
build for the Chronos server; a Realizar-first table row in CLAUDE.md documenting the exception;
README/CLAUDE.md drift updates the new crates force.

**Not in scope (and why):**
- Chronos-2 serving — MANIFEST: "Chronos-Bolt first, Chronos-2 later". Spike 009 proved the port
  (228 MB f16, ~0.5 s per forecast); it is a follow-up tier of the same server, not this phase.
- `freq` H (needs fractional days through the whole Prophet pipeline), country-holiday calendars,
  NeuralProphet quantile regression and `n_forecasts > 1`, multivariate/covariate inputs — not spiked.
- `model: auto` routing across servers — the policy is measured (spike 006) but it is a product
  decision that spans two servers; documented, not built.
- An `apr forecast` CLI subcommand — touches the 111-command registry and its contract; own ticket.
- Opening the upstream NEON PR (MANIFEST: a checkpoint, not automatic), the pmcp router-lock
  report, the parallel-GEMM N-split, the streaming safetensors loader, the 8×12 tile, the
  `test_brick_profiler_reset_v2` timer flake — all upstream/follow-up items surfaced by the spikes.
- pmcp.run / Lambda deployment itself — the crates are Lambda-shaped; deploying is operations.

</domain>

<decisions>
## Implementation Decisions

### Serving Shape (user decisions, spike alignment 2026-09-04/05 — MANIFEST Requirements)

- **D-01: The MCP serving shape is a STATELESS `forecast` tool.** One call carries `ds[]`, `y[]`,
  `horizon` (and `freq`); the server fits and forecasts inside that call. No fit → artifact →
  forecast round-trip. Spikes 001–004 showed fits take 0.04–1.5 s, so the artifact round-trip buys
  nothing.

- **D-02: Both Prophet (MAP / L-BFGS) and NeuralProphet (autograd / AdamW) ship behind the one tool**, selected by `model: prophet | neuralprophet` (`prophet` default). NeuralProphet was spiked,
  not deferred (spike 002); its value is the AR-Net for short-horizon nowcasting (`n_lags`).

- **D-03: Chronos (zero-shot foundation model) is a THIRD forecaster with its OWN thin server — one model per server.** Chronos-Bolt first (`aprender-mcp-chronos`, this phase); Chronos-2 later.
  All servers share the spike-004 request/response shape so a client can switch by endpoint.

- **D-04: The correctness bar for the Prophet port is parity with Python Prophet 1.4.0 on the Peyton Manning dataset** (fixture
  `.planning/spikes/001-prophet-map-fit-lbfgs/fixtures/peyton_manning_prophet140.json`), not
  self-consistency. Parity means: data prep 0.0 diff, objective at Python's MAP ≤ 1e-9, predict
  path ≤ 1e-10, fitted objective ≤ Python's + 0.5, forecast inside Prophet's own Newton-vs-L-BFGS
  band (spike 001 §7). The same bar extends to the spike-003 fixtures (holidays, logistic,
  multiplicative) and to Chronos (`chronos-forecasting` 2.3.1 oracle) and NeuralProphet (0.9.0
  oracle) with the tolerances recorded in the spike-findings references.

- **D-05: Build order 008 → 007 → 009 → 010 is already satisfied for this phase's inputs.** The
  spike-008 NEON kernel is in this tree (`crates/aprender-compute/src/blis/microkernels/neon.rs`,
  `contracts/neon-blis-v1.yaml`) and on `perf/neon-gemm-8x6-microkernel`; **opening the upstream
  PR is a human checkpoint, not a Phase 6 task.** Plans assume the kernel; they do not modify it.

### Crate Shape (the user's ask: "aprender-forecast lib + aprender-mcp-forecast + aprender-mcp-chronos")

- **D-06: Three workspace crates, following the SetFit trio's pattern.**
  `crates/aprender-forecast` (library: Prophet, NeuralProphet-lite, Chronos-Bolt forward,
  safetensors loader, civil-date helpers, shared request/response types);
  `crates/aprender-mcp-forecast` (thin pmcp server, `TOOL_NAME = "forecast"`, Prophet + NP);
  `crates/aprender-mcp-chronos` (thin pmcp server, same tool name and shape, embedded Bolt weights).
  Template for the servers: `crates/aprender-mcp-setfit` (typed tool, `deny_unknown_fields`,
  `spawn_blocking`, `run_stdio`) and `crates/aprender-mcp-setfit-lambda/build.rs` (embedding).
  Library and module names inside the crates are Claude's discretion.

- **D-07: Forecasting inference lives in `aprender-forecast`, NOT in `aprender-serve` (realizar) — a documented Realizar-first EXCEPTION, like the SetFit row (Phase 4 D-09).** Rationale: Prophet
  and NeuralProphet "inference" IS a fit (L-BFGS / autograd) — training-side machinery by
  CLAUDE.md's own table; Chronos-Bolt is an 8.7M-parameter T5 with no tokenizer, no KV cache and
  no LLM kernels, whose ONLY conformance-proven implementation is the spike port; serving a second,
  unproven port through realizar would violate OPS-03 (one implementation per operation). The phase
  adds the row to CLAUDE.md's Realizar-first table with this rationale. `aprender-serve` is not
  touched.

### Porting Rules (spike-established; the findings skill is the source of truth)

- **D-08: Port the spike sources as-is; the parity fixtures become the tests.** `prophet.rs` and
  `np.rs` from spike 004 (the evolved copies), `fit.rs` from 006, `dates.rs`, `bolt.rs`,
  `safetensors.rs`, `build.rs`, `static/index.html`, `tests/e2e.rs` from 007, the router pool from
  010. Sources are in `.claude/skills/spike-findings-aprender/sources/NNN-*/`; oracle fixtures stay
  in `.planning/spikes/NNN-*/fixtures/` and are referenced (or copied into the crate's `tests/
  fixtures/` if the test harness needs them in-crate — the planner decides; the 5 MB Prophet
  fixtures and the 2.6 MB Chronos fixtures are already committed either way). Rewriting a proven
  port is a regression risk, not an improvement; clippy `-D warnings`, `unwrap()` ban and fmt are
  the only reasons to touch a line.

- **D-09: The Prophet fit config is exactly the spike-004/006 recipe, every piece load-bearing.**
  objective ÷ T; non-finite guard (return `1e300` with zero gradient); exact L1 on δ;
  `LbfgsF64::new(2_000, 1e-7, 20)`; `Stalled` accepted as the success status; restart from the
  stall point with fresh history until relative improvement < 1e-6 or 8 rounds; x-keyed
  `value_and_grad` cache; 15 s wall-clock budget reported as `budget_hit` in diagnostics; constant
  `y` refused BEFORE fitting; interval RNG seeded per request. See
  `references/prophet-fit-and-predict.md`.

- **D-10: NeuralProphet training rules are the spike-002 recipe.** Huber built from ops with a constant mask (core
  `SmoothL1Loss` is detached from the graph — never use it); `clear_graph()` after every optimiser
  step; training on a `spawn_blocking` thread, never two fits interleaved on one thread; learning
  rate selected by TRAIN loss from {0.01, 0.03, 0.1} lag-free / {0.03, 0.1} with lags; auto batch
  and epochs from NeuralProphet's formulas, never full-batch; Fourier on days since 1900-01-01.
  See `references/neuralprophet-autograd.md`.

- **D-11: The tool boundary refuses, never defaults.** `#[serde(deny_unknown_fields)]`;
  `MIN_POINTS 10 / MAX_POINTS 20_000 / MAX_HORIZON 3_650` (fit server), `MIN_POINTS 4 / MAX_HORIZON
  1_024` (Chronos); `ds` strictly ascending, unique, real calendar dates; `ds`/`y` same length;
  unknown `freq` refused (D, W, MS only); `interval_width ∈ (0,1)`; logistic requires `cap >
  max(y)`; constant `y` refused; `y` nullable only on the Chronos server. Refusals are
  `pmcp::Error::validation`; each has an e2e case. Responses carry `fit_seconds`,
  `predict_seconds` and a `diagnostics` object; `seed` defaults to 42.

- **D-12: The Prophet/NeuralProphet HTTP app pools K pmcp routers behind a round-robin `fallback` handler** (spike 010: pmcp 2.19's streamable-HTTP router holds one `Arc<Mutex<Server>>` across
  the whole tool future, so a single router serialises fits — 1.0× on 8 concurrent requests; a pool
  of 8 gives 3.9×). K is configurable (`--pool`), default 8; the stdio path is unaffected. The
  release check is the spike-010 equality test: JSON signature of `ds/yhat/bands/trend/components`
  alone vs under load, bit-identical.

- **D-13: The Chronos server embeds weights the `aprender-mcp-setfit-lambda` way.** `build.rs`
  stages `model.safetensors` + `config.json` from `CHRONOS_EMBED_DIR` into `OUT_DIR` for
  `include_bytes!`, writes empty markers when unset, and the runtime falls back to
  `CHRONOS_MODEL_DIR`. Default embed is Bolt-**tiny f16** (24 MB binary, 52 ms cold start, f16
  costs 0.11 % of std); small-f16 is a build option (103 MB — S3/container, not a Lambda zip).
  Horizon policy from spike 006: accept `horizon ≤ 64` by default; `allow_long_horizon: true` up
  to 1024 with a `warning` in the response. Response: `yhat`/`yhat_lower`/`yhat_upper` = q50/q10/
  q90 plus the full `quantiles` map, `context_used`, `forwards`, `rollouts`, `missing_values`,
  weights dtype and source. Only transposed `[in, out]` weights are kept in memory.

- **D-14: Every multi-row product goes through the packed BLIS GEMM, single rows through the 8-accumulator dot.** `trueno::blis::gemm_blis` for rows > 1, `dot8` for one row. Never `trueno::Matrix::matmul` (measured 4× slower than loops before the
  kernel). The rayon `blis::gemm` is not used in the servers (≤ 1.4× below ~500 tokens; Lambda has
  1–2 vCPUs).

- **D-15: Contracts ship with the crates and carry the tolerances.** Each new crate has at least
  one `contracts/*.yaml` (`KernelContract` shape: `equations`, `proof_obligations`,
  `falsification_tests`) validated by `pv validate` — the forecast tool boundary, Prophet parity
  tolerances, Chronos parity tolerances and horizon policy. Tolerances live in the contract and
  are cited by the tests, not hardcoded in two places. Project rule: coverage without contracts is
  rejected.

- **D-16: Documentation states EMPIRICAL coverage, not nominal.** Spike 006 measured every
  method's nominal 80 % band covering 0.60–0.69 on rolling origins. Tool descriptions and READMEs
  say so. The spike-006 rolling-origin MASE harness is preserved as a bench/example, not a
  per-commit CI test (the CI gate is the parity fixtures).

- **D-17: Dates are `i64` days since epoch via civil-date arithmetic** (`days_from_civil` /
  `civil_from_days`, ~60 lines, spike-proven to exact parity); no `chrono`. `future_days` covers D,
  W and MS; H is refused with a message.

- **D-18: Model weights are never committed; a weight-dependent test never passes vacuously.**
  Chronos weights (34.6 MB f32 / 17.3 MB f16 for tiny) come from the Hub (`amazon/chronos-bolt-tiny`,
  Apache-2.0) at a pinned revision. The planner chooses the mechanism (a `just`/script fetch with
  a sha256 check into a gitignored `models/` dir, a CI cache step, or both), under two constraints:
  a missing-weights run SKIPS with a visible reason and a non-zero count of skipped tests in the
  summary (never a silent green), and at least one CI leg actually exercises the embedded-weights
  build. Fixtures (oracle outputs) ARE committed.

### Claude's Discretion

- Library/lib names and module layout inside the three crates; whether `aprender-forecast` exposes
  one `forecast()` entry or per-model modules.
- Whether to add `aprender-mcp-forecast-lambda` / `aprender-mcp-chronos-lambda` wrapper crates now
  (thin copies of `aprender-mcp-setfit-lambda`) or defer them — include if they fit in one plan.
- Whether to fix core `nn::loss::SmoothL1Loss` (detached from the graph, spike 002) and add a
  gradient-connectivity test for every loss in `nn::loss` in this phase, or file it as a core ticket.
- Test organisation: unit vs `tests/*.rs` integration targets. Remember CI runs `--lib` across the
  workspace plus ONE explicit line of `--test` targets in `.github/workflows/ci.yml`; a new
  integration target is dark until added there — and modifying CI workflows is a human check-in
  per CLAUDE.md, so plans must surface that as an `autonomous: false` step or keep parity tests
  as `--lib` tests.
- Whether the spike-004 `--bench` latency table and the spike-007 `--coldstart` harness ship as
  binary flags, examples, or `just` recipes.
- Router pool size default and how `--pool` is exposed.

</decisions>

<canonical_refs>
## Canonical References

### The spike evidence (read first)
- `.claude/skills/spike-findings-aprender/SKILL.md` — requirements, feature-area index
- `.claude/skills/spike-findings-aprender/references/prophet-fit-and-predict.md` — Prophet blueprint
- `.claude/skills/spike-findings-aprender/references/neuralprophet-autograd.md` — NeuralProphet blueprint
- `.claude/skills/spike-findings-aprender/references/forecast-mcp-thin-server.md` — fit server + router pool
- `.claude/skills/spike-findings-aprender/references/chronos-zero-shot-port.md` — Bolt/Chronos-2 forwards, parity ladder
- `.claude/skills/spike-findings-aprender/references/chronos-mcp-server.md` — embedding, f16, horizon gate, sizing
- `.claude/skills/spike-findings-aprender/references/forecaster-evaluation-and-routing.md` — rolling-origin evidence
- `.claude/skills/spike-findings-aprender/references/gemm-neon-kernel-upstream.md` — kernel state, parity-work protocol
- `.claude/skills/spike-findings-aprender/sources/NNN-*/` — the code to port (README, Cargo.toml, src/, tests/, static/, tools/)
- `.planning/spikes/MANIFEST.md` — idea + requirements; `.planning/spikes/CONVENTIONS.md` — how the spikes were built;
  `.planning/spikes/WRAP-UP-SUMMARY.md` — consolidated findings and open items
- `.planning/spikes/NNN-*/fixtures/` — oracle fixtures (Prophet 1.4.0, NeuralProphet 0.9.0, chronos-forecasting 2.3.1)

### Code the phase builds on
- `crates/aprender-mcp-setfit/` (`Cargo.toml`, `src/lib.rs`, `src/main.rs`, `tests/`) — the thin-server template
- `crates/aprender-mcp-setfit-lambda/build.rs`, `src/main.rs` — embedding + stateless streamable-HTTP loopback
- `crates/aprender-core/src/optim/` — `LbfgsF64`, `ConvergenceStatus`; `crates/aprender-core/src/autograd/`, `src/nn/` — `Tensor`, `Linear`, `AdamW`
- `crates/aprender-compute/src/blis/` — `gemm_blis`, `microkernels/neon.rs` (spike 008 kernel)
- `contracts/setfit-apr-v1.yaml`, `contracts/neon-blis-v1.yaml` — contract shape to mirror
- Root `Cargo.toml` — workspace members, `[workspace.lints]` (`unsafe_code = "forbid"`, pedantic), `.clippy.toml` (`unwrap()` banned)

### Project rules that bite here
- CLAUDE.md "Derive the numbers": adding 3 crates moves the README crate count; `crates/aprender-core/tests/readme_contract.rs`
  gates README.md counts and every path cited in CLAUDE.md. (4 README failures pre-exist on this branch: 82→83 crates,
  1778→1786 contracts, `aprender-mcp-setfit` README lacks the monorepo link, one crate has no README — `test_every_crate_has_readme`
  means every new crate needs a README with the monorepo link.)
- CLAUDE.md "Contract Validation: DOGFOOD `pv`" — validate contracts with `pv validate`, never scripts.
- CLAUDE.md "Verification Discipline" — never `$?` through a pipe; pin binaries; prove the mechanism engaged.
- CLAUDE.md "Modifying CI workflows" is a check-in-before-acting item.
- `.planning/spikes/CONVENTIONS.md` — `rtk` hides `println!` from `cargo test`; run the deps test binary directly.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `LbfgsF64` (f64 L-BFGS with Wolfe search) — proven for the Prophet MAP with the D-09 wrapper.
- f32 autograd ops proven sufficient for NeuralProphet: `matmul` (2-D), `transpose`, `broadcast_add`, `add`, `sub`, `mul`,
  `mul_scalar`, `abs`, `pow`, `mean`, `relu`, `view`, `backward`, `no_grad`, `clear_graph`, `nn::Linear`, `optim::AdamW`.
- `trueno::blis::gemm_blis` with the NEON 8×6 kernel: 65–77 GFLOP/s on aarch64 (0.7× faer).
- Workspace lock already has `pmcp 2.19.3` (`streamable-http`, `schema-generation`), `schemars 1.0`, `axum 0.8`, `tower 0.5`,
  `reqwest 0.12`, `half 2.7`, `tokio`, `serde`, `serde_json`.

### Established Patterns
- `Server::builder().tool_typed_with_description::<Args,_,_>(...)` with `#[serde(deny_unknown_fields)]` args, CPU work in
  `tokio::task::spawn_blocking`, `run_stdio()` locally, `StreamableHttpServerConfig::stateless()` on HTTP.
- Same-origin demo page: `pmcp::axum::router_with_config(server, RouterConfig{ stateless, allowed_origins: localhost })`
  nested under `/mcp` next to `GET /` (`include_str!("../static/index.html")`).
- In-process e2e tests over streamable-HTTP (`tests/e2e.rs`), refusal cases included.
- Embedded artifacts via `build.rs` + env var + `include_bytes!`, runtime-path fallback.

### Integration Points
- Root `Cargo.toml` `[workspace] members` — three new entries (check `exclude` and publish settings; CB-510 root-anchored ignores).
- `.github/workflows/ci.yml` — new `--test` targets are dark until listed (human check-in to edit).
- `contracts/` — new YAMLs counted by `find contracts -name '*.yaml'`; README/CLAUDE counts drift.
- `CLAUDE.md` Realizar-first table — one new row (D-07); `readme_contract` verifies cited paths exist.
- `README.md` claims table — crate and contract counts must be re-derived after the crates land.

</code_context>

<specifics>
## Specific Ideas

- **"Training is expensive" does not hold here.** Prophet fits in 0.2–1.4 s, NeuralProphet in 0.04–0.16 s lag-free;
  that is why the tool is stateless (D-01) and why a fit-per-request server is safe with a budget (D-09).
- **Parity over self-consistency.** Every port matched Python because a bottom-up ladder made each assumption checkable
  at its own rung; the tests should keep that ladder shape (data prep → objective/embeddings → predict → forecast), not
  just an end-to-end tolerance.
- **The tool description is a contract with the client.** Refusal messages name the fix (`allow_long_horizon`,
  "use D, W or MS"); the description states bounds and empirical coverage (D-16).
- **Weights in, fixtures in, reports out.** Commit oracle fixtures; never weights; never HTML reports.
- **Four `readme_contract` failures already exist on this branch** — the phase must not add to them and should fix the ones
  its own crates touch (README counts, crate READMEs with the monorepo link).

</specifics>

<deferred>
## Deferred Ideas

- **Chronos-2 tier** of `aprender-mcp-chronos` (spike 009: 21 quantiles, 1024-step direct horizon, 8192 context, 228 MB f16,
  ~0.5 s/forecast, needs a streaming loader to avoid the 1.45 GB load peak) — MANIFEST: "Chronos-2 later".
- **`model: auto`** routing ("monthly or ≤ 64 steps → Chronos, else NeuralProphet/Prophet") — measured in spike 006; a
  cross-server product decision.
- **`freq` H** (fractional days), **country-holiday calendars**, **NeuralProphet quantile regression and `n_forecasts > 1`**,
  **multivariate/covariate inputs** — not spiked.
- **`apr forecast` CLI subcommand** — touches `contracts/apr-cli-commands-v1.yaml` and the 111-command registry.
- **Core fixes surfaced by the spikes** (unless taken under Claude's Discretion): `SmoothL1Loss` graph connectivity + loss
  connectivity tests; `where`/`clamp`/`cat` autograd ops; `WolfeSearch` initial-step scaling / non-finite backtracking;
  `LbfgsF64` double evaluation at `x`; an f64 proximal solver.
- **Upstream items:** opening the NEON PR (human checkpoint); pmcp router lock report (paiml/pmcp); parallel GEMM N-split
  for M ≤ 256; streaming safetensors loader; 8×12 NEON tile; `test_brick_profiler_reset_v2` timer-resolution flake.
- **Deployment** to pmcp.run / Lambda and the MCP client configs — operations, after the crates exist.

</deferred>

---

*Phase: 6-Native Time-Series Forecasting Stack*
*Context assembled: 2026-09-05 from `.planning/spikes/MANIFEST.md` and the spike-findings skill*
