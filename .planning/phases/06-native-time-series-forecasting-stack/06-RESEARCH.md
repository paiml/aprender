# Phase 6: Native Time-Series Forecasting Stack - Research

**Researched:** 2026-09-05
**Domain:** Porting ten VALIDATED spike crates (Prophet 1.4.0 / NeuralProphet-lite / Chronos-Bolt ports + two thin pmcp servers) into aprender workspace members that pass every workspace gate
**Confidence:** HIGH for everything read from the tree this session (tagged `[VERIFIED: path:lines]`); MEDIUM for the spike-measured numbers (cited from the skill, not re-run except where noted); LOW for the three web look-ups (tagged `[CITED]`)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

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

### Deferred Ideas (OUT OF SCOPE)

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
</user_constraints>

<phase_requirements>
## Phase Requirements

`phase_req_ids` is null: `.planning/REQUIREMENTS.md` is the SetFit milestone's document and carries no forecasting REQ-IDs (confirmed by grep: zero hits for `forecast|prophet|chronos` — `[VERIFIED: .planning/REQUIREMENTS.md]`). No IDs are invented here. The binding requirements are D-01..D-18 above and the five ROADMAP Success Criteria (SC1–SC5, `.planning/ROADMAP.md` §Phase 6). The Validation Architecture section maps SC1–SC5 to tests.
</phase_requirements>

## Summary

The research for *what to build* is done — ten VALIDATED spikes (`.claude/skills/spike-findings-aprender/`) proved every port to parity and measured every cost. This document covers the gap between "validated standalone spike crate" and "workspace member that passes every gate", and it found that gap is mostly **repo policy, not code**: 2 249 lines of spike Rust port as-is (`sources/004/*.rs` 1 273 lines, `sources/007/*.rs` + `tests/e2e.rs` 847, `sources/006/fit.rs` 29 — `[VERIFIED: wc -l over the skill sources]`), every core/compute API they call exists under the same name today (Finding F3), every dependency they need is already resolved in `Cargo.lock` (F9), and the SetFit trio is a faithful template for manifests, server wiring and embedding (F1, F2).

The load-bearing surprises are the gates. **Five CI-wired drift tests already fail on this branch** — measured this session by running them, not read off the CONTEXT: `test_no_unauthorized_binaries` (FALSIFY-MONO-011) rejects four SetFit crates' `[[bin]]`s against a **shrink-only 27-entry allowlist**, and `readme_contract` fails on crate count (README says 82, cargo says 83), contract count (1778 vs 1786), two crates without a README (`aprender-mcp-setfit-lambda`, `aprender-contrastive-data` — the CONTEXT said one) and `aprender-mcp-setfit`'s README lacking the `paiml/aprender` link (F7). Phase 6 adds two more `[[bin]]` crates, so the allowlist ratchet is a **human decision the planner must surface**, not a line an executor may edit (Open Question 1). Two more findings change how the planner should shape tests: `contracts/neon-blis-v1.yaml` — named as a shape reference — validates clean only because `registry: true` disables PROVABILITY-001 and its tests sit under a `falsification:` key the parser silently drops (`pv status` reports 0/0/0), so `setfit-apr-v1.yaml` is the only safe template (F4); and a debug-profile Prophet fit is **35× slower than release** (9.15 s vs 0.259 s on the single-round Peyton fit, measured this session), so the parity suite needs a per-package `opt-level` override or CI's nextest leg pays ~50 s per Peyton fit (F10, Pitfall 9).

For D-18 a positive probe proved the mechanism: a `build.rs` that emits `cargo::rustc-check-cfg=cfg(chronos_weights)` and `cargo:rustc-cfg=chronos_weights` when `CHRONOS_MODEL_DIR/model.safetensors` exists, with `#[cfg_attr(not(chronos_weights), ignore = "<reason>")]` on weight-dependent tests, yields `1 ignored, <reason>` under libtest and `1 skipped` in nextest's summary when weights are absent, and runs the tests normally when armed (F6).

**Primary recommendation:** Port the spike sources verbatim into three `publish = false` crates modelled line-for-line on `crates/aprender-mcp-setfit/Cargo.toml`; keep every parity and in-process e2e test as a `--lib` `#[cfg(test)]` module (CI's nextest reaches it without a workflow edit); template the contracts on `setfit-apr-v1.yaml` (not `neon-blis-v1.yaml`); fetch `amazon/chronos-bolt-tiny@a0e552de…` via a `just` recipe with sha256 pins into `/models/`; and open the phase with a human checkpoint on the FALSIFY-MONO-011 allowlist and the four README fixes, because CI cannot go green without them.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Prophet MAP fit + predict, NeuralProphet-lite train + predict, civil dates, shared request/response types | `aprender-forecast` library (training-side crate family; depends on `aprender-core` for `LbfgsF64`/autograd) | — | D-07: "inference" IS a fit; the only parity-proven implementation is the spike port; one implementation per operation (OPS-03) |
| Chronos-Bolt forward, safetensors F32/F16/BF16 decode, `dot8` + `gemm_blis` routing | `aprender-forecast` library (depends on `aprender-compute` as `trueno`) | `aprender-compute` (`blis::gemm_blis`, NEON 8×6 kernel) | D-14; kernel already in tree (D-05); no realizar kernels apply (no tokenizer, no KV cache) |
| Tool boundary (typed `ForecastArgs`, `deny_unknown_fields`, refusals as `pmcp::Error::validation`) | Thin server crates (`aprender-mcp-forecast`, `aprender-mcp-chronos`) | — | D-11; SetFit precedent: bounds enforced at the reading surface (`crates/aprender-mcp-setfit/src/lib.rs:20-27`) |
| stdio transport (`run_stdio`) | Thin server `main.rs` | — | What MCP clients spawn; template `crates/aprender-mcp-setfit/src/main.rs:81` |
| Streamable-HTTP + same-origin demo page, router pool (fit server only) | Thin server `lib.rs` (`http_app`, `axum` nest under `/mcp`) | Lambda loopback wrapper (Claude's discretion) | D-12; pmcp router lock is per-router, so the pool lives where the routers are built |
| Weight embedding (`build.rs` → `OUT_DIR` → `include_bytes!`), runtime `CHRONOS_MODEL_DIR` fallback | `aprender-mcp-chronos` build.rs + lib.rs | `just` fetch recipe (`/models/`, gitignored) | D-13, D-18; template `crates/aprender-mcp-setfit-lambda/build.rs` |
| Parity evidence (oracle fixtures → tests), contracts with tolerances | `aprender-forecast` `#[cfg(test)]` + `contracts/*.yaml` | Makefile `$(CONTRACTS)` list (tier3 `contract-validate`) | D-04, D-15; CI reaches `--lib` tests only (F5) |
| Drift-gate updates (README counts, per-crate READMEs, CLAUDE.md row, bin allowlist) | Repo docs + `crates/aprender-core/tests/{readme_contract,monorepo_invariants}.rs` | Human checkpoint for the allowlist | F7, Open Question 1 |

## Findings — the gap between spike and workspace member

### F1. Workspace integration (root `Cargo.toml`, lints, toolchain)

**Members and exclusions.** Members are an explicit list under `[workspace] members = [ ".", "crates/aprender-core", … ]` with section comments per merge phase; the MCP block reads verbatim `[VERIFIED: Cargo.toml:46-53]`:

```toml
    # --- MCP server (Model Context Protocol) ---
    "crates/aprender-mcp",
    # Thin single-model MCP server: SetFit classification via pmcp (pmcp.run template)
    "crates/aprender-mcp-setfit",
    "crates/aprender-mcp-setfit-lambda",
    # Thin single-algorithm MCP TRAINING server: SetFit training as an async MCP Task
    "crates/aprender-mcp-setfit-train",
```

The three new crates go in this block. There is **no `default-members`** key (`[VERIFIED: Cargo.toml:2-158]` — grep for `default-members` returns nothing), so `cargo build --release` builds every member. `exclude` (lines 116-157) lists only `crates/aprender-train-canary`, `crates/aprender-zram/bins/trueno-ublk`, `tools/ccpa-sft-export`, `fuzz`, three old shells and `crates/facades` — nothing that affects new members. `resolver = "2"`.

**Inheritance.** `[workspace.package]` `[VERIFIED: Cargo.toml:159-168]`:

```toml
version = "0.63.0"
edition = "2021"
license = "MIT"
repository = "https://github.com/paiml/aprender"
authors = ["Noah Gift <noah@paiml.com>"]
rust-version = "1.91"
```

Toolchain pin `[VERIFIED: rust-toolchain.toml:1-3]`: `channel = "1.93.0"`, components `["rustfmt", "clippy"]`. Installed: `rustc 1.93.0`, `cargo 1.93.0` (probed). MSRV 1.91 ≥ everything the spike code uses (`f64::round_ties_even` 1.77, `cargo::rustc-check-cfg` needs Cargo ≥ 1.80 — `[ASSUMED]` for the exact Cargo floor; the `cargo::` double-colon syntax is what the probe used on 1.93).

**Workspace dependencies the new crates should inherit** `[VERIFIED: Cargo.toml:170-215]`:

```toml
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1", features = ["full"] }
axum = { version = "0.8", features = ["ws"] }
tower = "0.5"
reqwest = { version = "0.12", default-features = false }
half = { version = "2.4", default-features = false, features = ["std"] }
rayon = "1.10"
```

`pmcp` and `schemars` are **not** workspace dependencies (grep `^pmcp|^schemars` in root `Cargo.toml` → none); each MCP crate declares them (F2). `half` and `tower` are therefore **not new direct deps** in policy terms — both are workspace deps and `apr-cli` already depends on them directly (`crates/apr-cli/Cargo.toml:259` `half = "2.4.1"`, `:314` `tower = { version = "0.5", features = ["util"] }` `[VERIFIED]`). The spike's `half = "2.7"` request resolves to the locked `2.7.1` (F9); declare it as `half = { workspace = true }`.

**Lints.** `[workspace.lints.rust]` `[VERIFIED: Cargo.toml:378-398]`: `unsafe_code = "deny"` (note: CLAUDE.md says "forbid"; the manifest says `deny` with a comment about the mmap module — the manifest wins), `unexpected_cfgs = { level = "warn", check-cfg = ['cfg(kani)', 'cfg(coverage_nightly)', 'cfg(feature, values("explainable-monitor-integration"))'] }`, `unsafe_op_in_unsafe_fn = "warn"`, `rust_2018_idioms = warn`, `trivial_numeric_casts = warn`, `unused_import_braces = warn`, `unused_lifetimes = warn`. `[workspace.lints.clippy]` `[VERIFIED: Cargo.toml:400-450]`: `all = { level = "warn", priority = -1 }`, `pedantic = { level = "warn", priority = -1 }`, `undocumented_unsafe_blocks = "warn"`, `checked_conversions = "warn"`, `manual_ok_or = "warn"`, `explicit_deref_methods = "warn"`, `inconsistent_struct_constructor = "warn"`, `unnested_or_patterns = "warn"`; allowed: `many_single_char_names`, `cast_precision_loss`, `cast_possible_truncation`, `cast_possible_wrap`, `cast_sign_loss`, `similar_names`, `doc_markdown`, `missing_const_for_fn`, `module_name_repetitions`, `must_use_candidate`, `missing_errors_doc`, `missing_panics_doc`, `implicit_clone`, `explicit_iter_loop`, `used_underscore_binding`, `redundant_closure_for_method_calls`, `inefficient_to_string`. Everything is `warn`, so `cargo clippy -p <crate> --lib -- -D warnings` (SC5) turns all of it into errors — the spike code was written for a standalone crate without these lints and **will** need pedantic clean-up (Pitfall 1).

`.clippy.toml` `[VERIFIED: .clippy.toml:1-40]`: `cognitive-complexity-threshold = 15`, `type-complexity-threshold = 250`, and four `[[disallowed-methods]]` — `core::option::Option::unwrap`, `core::result::Result::unwrap` ("Use .expect() with descriptive message or proper error handling with ?. See GH-41."), plus the LAYOUT-001 column-major q4k/q6k bans. Consequence for the port: the spike-010 pool's `.unwrap_or_else(|e| match e {})` is fine (not `unwrap`), but every `.unwrap()` in the spike sources must become `expect`/`?`, and `schemars::JsonSchema` derive + `serde_json::json!` expand to `.unwrap()` internally — the template opts out at file scope with `#![allow(clippy::disallowed_methods)]` `[VERIFIED: crates/aprender-mcp-setfit/src/lib.rs:29-32]`:

```rust
// schemars' JsonSchema derive and serde_json::json! both expand to .unwrap()
// internally, and the derive's generated impl lands at file scope where a
// struct-level allow cannot reach it. Same precedent as aprender-mcp's tools.
#![allow(clippy::disallowed_methods)]
```

`rustfmt.toml` `[VERIFIED: rustfmt.toml:1-10]`: `max_width = 100`, `use_field_init_shorthand = true`, `use_try_shorthand = true`. Spike lines longer than 100 columns (the one-line `RouterConfig { … }` in `sources/007/src/lib.rs:165`) get reflowed by `cargo fmt --all`; run it before `--check`.

**Gates a new member must satisfy** (`crates/aprender-core/tests/monorepo_invariants.rs`, on the CI `--test` line — F5):
- FALSIFY-MONO-010 (`:36-77`): package name must start with `aprender` (allowlist `["apr-cli", "apr-format"]`). `aprender-forecast`, `aprender-mcp-forecast`, `aprender-mcp-chronos` comply.
- FALSIFY-MONO-012 (`:79`): flat layout — direct children of `crates/`.
- FALSIFY-MONO-011 (`:235-395`): **`[[bin]]` allowlist, shrink-only** — see F7 and Open Question 1. Quoted assertion `[VERIFIED: :353]`: `"FALSIFY-MONO-011: Unauthorized [[bin]] sections found in: {:?}\nOnly apr-cli should produce user-facing binaries."` and `const ALLOWLIST_BASELINE: usize = 27;` with `"It is shrink-only: migrate the capability to an apr subcommand instead of granting a new exemption."`
- FALSIFY-BUILD-005 (`:429`): every member's `manifest_path` exists.

### F2. The `aprender-mcp-setfit` template, in detail

**Manifest** `[VERIFIED: crates/aprender-mcp-setfit/Cargo.toml:1-42]` — copy this shape:

```toml
[package]
name = "aprender-mcp-setfit"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
authors.workspace = true
rust-version.workspace = true
description = "Thin single-model MCP server: SetFit classification over the pmcp SDK"
keywords = ["mcp", "setfit", "classification", "aprender", "pmcp"]
categories = ["science", "web-programming"]
readme = "README.md"
# Not published until the pmcp.run pilot proves the template. The crate is a
# deployment unit, not a library other crates should depend on.
publish = false

[lib]
name = "aprender_mcp_setfit"
path = "src/lib.rs"

[[bin]]
name = "aprender-mcp-setfit"
path = "src/main.rs"

[dependencies]
aprender = { path = "../aprender-core", version = "0.63.0", package = "aprender-core", features = ["setfit"] }
pmcp = { version = "2.9", features = ["streamable-http", "schema-generation"] }
schemars = "1.0"
serde = { workspace = true }
serde_json = { workspace = true }
tokio = { workspace = true }

[dev-dependencies]
serde_json = { workspace = true }

[lints]
workspace = true
```

Note `pmcp = "2.9"` resolves to `2.19.3` in the lock (F9); `aprender-mcp-setfit-train` already declares `pmcp = { version = "2.19", features = ["streamable-http", "schema-generation"] }` and `schemars = "1.0"` `[VERIFIED: crates/aprender-mcp-setfit-train/Cargo.toml:41-42]` — use the `2.19` form (the spikes' `router_with_config` needs it).

**Lambda wrapper manifest** `[VERIFIED: crates/aprender-mcp-setfit-lambda/Cargo.toml:1-45]`: `[[bin]] name = "bootstrap"` ("AWS Lambda's Custom Runtime API requires the binary to be named `bootstrap` exactly"), `[lib] name = "aprender_mcp_setfit_lambda"`, deps `aprender-mcp-setfit = { path = "../aprender-mcp-setfit" }`, `pmcp = { version = "2.9", … }`, `lambda_http = "0.13"`, `tokio = { workspace = true }`, `reqwest = { version = "0.12", default-features = false, features = ["json", "rustls-tls"] }`, `once_cell = "1.19"`, `serde_json`, `tracing = "0.1"`, `tracing-subscriber = { version = "0.3", features = ["env-filter"] }`, `publish = false`, `[lints] workspace = true`.

**Server API shapes used** `[VERIFIED: crates/aprender-mcp-setfit/src/lib.rs:41-42, 198-224]`:

```rust
use pmcp::types::capabilities::ServerCapabilities;
use pmcp::Server;
// …
pub fn build_server(model: Arc<VerifiedSetFitModel>, name: &str, version: &str) -> pmcp::Result<Server> {
    Server::builder()
        .name(name)
        .version(version)
        .capabilities(ServerCapabilities::tools_only())
        .tool_typed_with_description::<ClassifyArgs, _, _>(
            TOOL_NAME,
            TOOL_DESCRIPTION,
            move |args, _extra| {
                let model = Arc::clone(&model);
                async move {
                    let document = precheck(args)?;
                    let response = tokio::task::spawn_blocking(move || model.classify(&document))
                        .await
                        .map_err(|e| pmcp::Error::internal(format!("classify task join: {e}")))?
                        .map_err(|e| classify_error(&e))?;
                    serde_json::to_value(&response)
                        .map_err(|e| pmcp::Error::internal(format!("response serialization: {e}")))
                }
            },
        )
        .build()
}
```

`main.rs` `[VERIFIED: crates/aprender-mcp-setfit/src/main.rs:49-86]`: `#[tokio::main] async fn main() -> ExitCode`; parses `--model <FILE>` else `APRENDER_SETFIT_MODEL`; everything human-readable to **stderr** ("stdout belongs to the protocol", `:6-7`); `server.run_stdio().await`. The spike servers' `--stdio` default / `--http PORT` / `--bench` / `--coldstart` flags extend this same hand-rolled argv loop (no clap — keep it that way; `clap` would be a new direct dep for a 60-line parser).

**pmcp 2.19.3 API the spikes add on top** (verified in the registry source, `~/.cargo/registry/src/index.crates.io-*/pmcp-2.19.3/`):
- `pub fn router_with_config(server: Arc<tokio::sync::Mutex<Server>>, config: RouterConfig) -> Router` `[VERIFIED: src/server/axum_router.rs:91]`
- `pub struct RouterConfig { pub allowed_origins: Option<AllowedOrigins>, pub security_headers: SecurityHeadersLayer, pub server_config: StreamableHttpServerConfig }` `[VERIFIED: src/server/axum_router.rs:47-55]` — it implements `Default`, so the spikes' `..Default::default()` is valid.
- `AllowedOrigins::localhost()` `[VERIFIED: src/server/tower_layers/dns_rebinding.rs:25, 88]`
- `StreamableHttpServerConfig::stateless()` `[VERIFIED: src/server/streamable_http_server.rs:451-462]` — sets `enable_json_response: true`; the doc comment calls it "the serverless/Lambda constructor".
- `Server::run_stdio(self)` `[VERIFIED: src/server/mod.rs:1093]`; `tool_typed_with_description` `[VERIFIED: src/server/mod.rs:3663]`; `ServerCapabilities::tools_only()` `[VERIFIED: src/types/capabilities.rs:623]`.
- The router lock spike 010 measured: `ServerState::server` is "one `Arc<tokio::sync::Mutex<Server>>` shared by every session" `[VERIFIED: src/server/streamable_http_server.rs:1566-1567]` and is taken with `state.server.lock().await` at `:2094` and `:2122` — matching the reference's citation.

**Lambda wrapper** `[VERIFIED: crates/aprender-mcp-setfit-lambda/src/lib.rs:18-55]`:

```rust
pub static EMBEDDED_MODEL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/model.apr"));
pub fn server_config() -> StreamableHttpServerConfig { StreamableHttpServerConfig::stateless() }
pub fn resolve_model() -> Result<Arc<aprender_mcp_setfit::Model>, ModelLoadError> {
    if !EMBEDDED_MODEL.is_empty() { return aprender_mcp_setfit::load_model_from_bytes(EMBEDDED_MODEL).map(Arc::new); }
    let path = std::env::var_os("APRENDER_SETFIT_MODEL").ok_or_else(|| ModelLoadError::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound, "no embedded model in this build and APRENDER_SETFIT_MODEL is unset")))?;
    aprender_mcp_setfit::load_model_from_path(std::path::Path::new(&path)).map(Arc::new)
}
```

`build.rs` `[VERIFIED: crates/aprender-mcp-setfit-lambda/build.rs:12-33]` — `println!("cargo:rerun-if-env-changed=APRENDER_SETFIT_MODEL")`, copy to `OUT_DIR/model.apr` when set (panicking on a bad path), else `std::fs::write(&staged, [])` ("write empty embed marker"). The spike-007 `build.rs` is the same pattern with two files and **no `".."` path join** (`[VERIFIED: grep '"\.\."' sources/007/build.rs → 0]`), so `scripts/check_build_rs_paths.sh` (wired in CI as "Build.rs crate-root escape check") skips it entirely — its rule only inspects build scripts that join `".."` `[VERIFIED: scripts/check_build_rs_paths.sh:39-45]`.

`main.rs` of the wrapper `[VERIFIED: crates/aprender-mcp-setfit-lambda/src/main.rs:48-53]` starts `pmcp::server::streamable_http_server::StreamableHttpServer::with_config(addr, server, server_config())` on loopback and proxies each `lambda_http` event to it. A forecast wrapper would be a copy with the model resolution removed (fit server) or swapped for `resolve_model()` (Chronos).

**Tests.** The template has 7 unit tests in `lib.rs` `#[cfg(test)] mod tests` (schema strictness, bounds, document conversion — `[VERIFIED: lib.rs:226-321]`) and one integration target `tests/e2e_stdio.rs` that spawns `env!("CARGO_BIN_EXE_aprender-mcp-setfit")` and is env-gated with a **visible `println!` SKIP + early return, "never `#[ignore]`"** `[VERIFIED: tests/e2e_stdio.rs:3-9, 60-68]`. That target is **not** on the CI `--test` line (grep `aprender-mcp-setfit` in ci.yml → 0 hits), i.e. it is dark in CI — the precedent the phase should *not* repeat for parity tests (F5). `tests/embed.rs` in the lambda crate self-gates on `EMBEDDED_MODEL.is_empty()` with `println!("EMBED SKIP: …")` `[VERIFIED: crates/aprender-mcp-setfit-lambda/tests/embed.rs:12-25]`. Both SKIP styles report `test result: ok … 0 ignored` — a green with no skipped count, exactly what D-18 forbids; F6 gives the mechanism that produces a counted skip.

**README requirement.** `test_every_readme_links_monorepo` checks `content.contains("paiml/aprender")` `[VERIFIED: crates/aprender-core/tests/readme_contract.rs:275]` — any `github.com/paiml/aprender` URL satisfies it. Existing phrasing: `Model Context Protocol (MCP) server for [aprender](https://github.com/paiml/aprender).` `[VERIFIED: crates/aprender-mcp/README.md:3]`. `crates/aprender-mcp-setfit/README.md` links `paiml/rust-mcp-sdk` but never `paiml/aprender` `[VERIFIED: file read]` — that is the pre-existing failure.

### F3. Core / compute API surface the ports call — all present under the spike's names

| Spike usage | Current definition | Provenance |
|---|---|---|
| `use aprender::optim::{ConvergenceStatus, LbfgsF64};` (`sources/004/src/lib.rs:8`, `sources/006/src/fit.rs:3`) | `pub struct LbfgsF64` · `pub fn new(max_iter: usize, tol: f64, m: usize) -> Self` · `pub fn minimize<F, G>(&mut self, objective: F, gradient: G, x0: &Vector<f64>) -> OptimizationResultF64 where F: Fn(&Vector<f64>) -> f64, G: Fn(&Vector<f64>) -> Vector<f64>` | `[VERIFIED: crates/aprender-core/src/optim/lbfgs.rs:766, 796, 811-819]` |
| `ConvergenceStatus::{Converged, Stalled}` | `pub enum ConvergenceStatus { Converged, MaxIterations, Stalled, NumericalError, Running, UserTerminated }` | `[VERIFIED: crates/aprender-core/src/optim/mod.rs:198-211]` (verbatim variants) |
| `r.objective_value`, `r.solution`, `r.status`, `r.iterations` | `pub struct OptimizationResultF64 { pub solution: Vector<f64>, pub objective_value: f64, pub iterations: usize, pub status: ConvergenceStatus, pub gradient_norm: f64 }` | `[VERIFIED: crates/aprender-core/src/optim/mod.rs:183-194]` |
| `aprender::primitives::Vector::from_vec`, `.as_slice()` | `pub struct Vector<T>` · `pub fn from_vec(data: Vec<T>) -> Self` · `pub fn as_slice(&self) -> &[T]` | `[VERIFIED: crates/aprender-core/src/primitives/vector.rs:18, 33, 51]` |
| `use aprender::autograd::{clear_graph, graph_tape_len, no_grad, Tensor};` (`sources/004/src/np.rs:6`) | `pub fn no_grad<F, R>(f: F) -> R` `:89` · `pub fn clear_graph()` `:117` · `pub fn graph_tape_len() -> usize` `:142` | `[VERIFIED: crates/aprender-core/src/autograd/mod.rs]` |
| `Tensor::from_vec`, `Tensor::new`, `Tensor::zeros`, `.data()`, `.shape()`, `.numel()`, `.grad()`, `.backward()`, `.requires_grad()` | `pub fn new(data: &[f32], shape: &[usize])` `:82` · `pub fn from_vec(data: Vec<f32>, shape: &[usize])` `:109` · `pub fn zeros(shape: &[usize])` `:139` · `requires_grad(mut self)` `:167` · `shape()` `:198` · `numel()` `:204` · `data() -> &[f32]` `:216` · `grad() -> Option<&Tensor>` `:231` · `pub fn backward(&self)` `:317` | `[VERIFIED: crates/aprender-core/src/autograd/tensor.rs]` |
| `add`, `sub`, `mul`, `mul_scalar`, `pow`, `abs`, `mean` | `ops/mod.rs:28, 55, 86, 174, 247, 294, 342` | `[VERIFIED: crates/aprender-core/src/autograd/ops/mod.rs]` |
| `relu`, `matmul`, `transpose`, `broadcast_add`, `view` | `ops/activation.rs:9, 274, 328, 376, 428` | `[VERIFIED: crates/aprender-core/src/autograd/ops/activation.rs]` |
| `use aprender::nn::{Linear, Module};` · `Linear::new(in, out)`, `Linear::without_bias(in, out)`, `set_bias`, `.forward()` | `pub fn new(in_features: usize, out_features: usize)` `:68` · `pub fn without_bias(in_features, out_features)` `:117` · `pub fn set_bias(&mut self, bias: Tensor)` `:177` · `Module::forward(&self, input: &Tensor) -> Tensor` | `[VERIFIED: crates/aprender-core/src/nn/linear.rs; nn/module.rs:31-50; nn/mod.rs:87 pub use linear::Linear]` |
| `use aprender::nn::optim::{AdamW, Optimizer};` · `AdamW::new(params, lr).weight_decay(wd)` · `opt.step()` · `opt.zero_grad()` | `pub struct AdamW` (`nn/optim/mod.rs:391`); `impl AdamW` with `pub fn new(params: Vec<&mut Tensor>, lr: f32) -> Self` and `pub fn weight_decay(mut self, wd: f32) -> Self`; `impl Optimizer for AdamW` (`step`, `zero_grad`) | `[VERIFIED: crates/aprender-core/src/nn/optim/rm_sprop.rs:4, 10, 40, 96]` — the impl lives in `rm_sprop.rs`, an unexpected file name; the spike compiled against it 2026-09-03 |
| `trueno::blis::gemm_blis(m, n, k, a, b, c, None)` (`sources/007/src/bolt.rs:164, 175`) | `pub fn gemm_blis(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32], mut profiler: Option<&mut BlisProfiler>) -> Result<(), TruenoError>` — seven args; the prompt's guessed `alpha?` does not exist | `[VERIFIED: crates/aprender-compute/src/blis/compute.rs:838-846]`; `pub mod blis;` `[VERIFIED: crates/aprender-compute/src/lib.rs:88]`; `[lib] name = "trueno"` `[VERIFIED: crates/aprender-compute/Cargo.toml:155-156]` |

Module gating: `pub mod autograd;` (`:83`), `pub mod nn;` (`:144`), `pub mod optim;` (`:147`), `pub mod primitives;` (`:151`) are **unconditional** `[VERIFIED: crates/aprender-core/src/lib.rs]` — the spikes built with `aprender = { path = …, package = "aprender-core", default-features = false }` (`[VERIFIED: sources/004/Cargo.toml]`) and that is the right dependency line for `aprender-forecast` (no `setfit`, no `parallel`). In-workspace form: `aprender = { path = "../aprender-core", version = "0.63.0", package = "aprender-core", default-features = false }` and `trueno = { path = "../aprender-compute", version = "0.63.0", package = "aprender-compute" }` (the `parallel` feature the spike enabled "only to measure" is not needed — D-14).

**`SmoothL1Loss` is still detached** `[VERIFIED: crates/aprender-core/src/nn/loss.rs:171-195]` — quoted verbatim:

```rust
    pub fn forward(&self, pred: &Tensor, target: &Tensor) -> Tensor {
        assert_eq!(pred.shape(), target.shape());

        let diff = pred.sub(target);
        let loss_data: Vec<f32> = diff
            .data()
            .iter()
            .map(|&x| {
                let abs_x = x.abs();
                if abs_x < self.beta {
                    0.5 * x * x / self.beta
                } else {
                    abs_x - 0.5 * self.beta
                }
            })
            .collect();

        let loss = Tensor::new(&loss_data, pred.shape());

        match self.reduction {
            Reduction::None => loss,
            Reduction::Mean => loss.mean(),
            Reduction::Sum => loss.sum(),
        }
    }
```

`Tensor::new(&loss_data, …)` builds a fresh leaf from raw `f32`s; the graph edge from `pred` is severed. D-10 (build Huber from ops) stands. Fixing it is Claude's discretion; if taken, the fix is to compose `diff.abs()`, `diff.pow(2.0).mul_scalar(0.5 / beta)` and a constant mask exactly as `weighted_huber` in `sources/004/src/np.rs` does, plus a connectivity test per loss (`loss.backward(); assert!(pred.grad().is_some())`).

### F4. Contract schema — what `pv validate` actually requires, and why `neon-blis-v1.yaml` is the wrong template

The parser is `serde_yaml::from_str::<Contract>` `[VERIFIED: crates/aprender-contracts/src/schema/parser.rs:22]`; `Contract` is **not** `deny_unknown_fields`, so an unknown key is silently dropped. Structure `[VERIFIED: crates/aprender-contracts/src/schema/types.rs:12-52]`: `metadata: Metadata` (required), `equations: BTreeMap<String, Equation>` (default; map or sequence form), `proof_obligations: Vec<ProofObligation>`, `falsification_tests: Vec<FalsificationTest>`, `kani_harnesses: Vec<KaniHarness>`, `qa_gate: Option<QaGate>`, … all `#[serde(default)]`. `Metadata` `[VERIFIED: types.rs:202-230]`: `version: String` and `description: String` are **required** (no default); `references: Vec<String>`, `kind: ContractKind` (default `Kernel`), `registry: bool`, `depends_on`, `enforcement_level` are defaulted. `ContractKind` `[VERIFIED: kind.rs:46-64]`: `Kernel, Registry, ModelFamily, ModelFamilyVariant, Tokenizer, TrainingLoop, PretrainingCorpus, TrainingPreconditionGate, CorpusAssembly, Pattern, Schema, BeatBenchmark`; parser test `parse_contract_defaults_to_kernel_kind` `[VERIFIED: parser.rs:313-317]`.

Validator rules `[VERIFIED: crates/aprender-contracts/src/schema/validator.rs:189-260, 320-380]`:
- SCHEMA-001 error: `metadata.references must not be empty — every contract must cite its source paper(s)`
- SCHEMA-002 error: `metadata.version must not be empty`
- SCHEMA-003 error: `equations must contain at least one equation`; SCHEMA-004 error: each `formula` non-empty
- SCHEMA-005 error: each `proof_obligations[i].property` non-empty; SCHEMA-006 warn: duplicate `formal`
- SCHEMA-007 error: duplicate falsification `id`; SCHEMA-008 error: `prediction` non-empty; SCHEMA-009 **warning**: `if_fails` empty
- SCHEMA-010/011 error: duplicate Kani `id` / empty `obligation`
- PROVABILITY-001 (error) from `Contract::provability_violations()` `[VERIFIED: types.rs:168-197]`: applies only when `self.kind() == ContractKind::Kernel`; then requires non-empty `proof_obligations`, non-empty `falsification_tests`, non-empty `kani_harnesses`, and `falsification_tests.len() >= proof_obligations.len()`.

**Why `neon-blis-v1.yaml` passes `pv validate` while proving nothing** `[VERIFIED: contracts/neon-blis-v1.yaml:1-60 read; target/debug/pv status run this session]`: it sets `metadata.registry: true` (legacy flag) and top-level `kind: KernelContract` (an unknown key — dropped), and lists its tests under `falsification:` (unknown key — dropped). `pv status contracts/neon-blis-v1.yaml` prints `Proof obligations: 0 / Falsification tests: 0 / Kani harnesses: 0`, yet `pv validate` says `0 error(s), 0 warning(s)` because `kind()` resolves the legacy flag to Registry and PROVABILITY-001 never fires. **Template on `contracts/setfit-apr-v1.yaml` instead** (`pv validate` → `0 error(s), 0 warning(s)`; keys `contract:`, `metadata:` (`kind: kernel`), `equations:` `:80`, `proof_obligations:` `:1010`, `falsification_tests:` `:1103`, `kani_harnesses:` `:1243`, `qa_gate:` `:1276` `[VERIFIED: grep of top-level keys]`). Its Kani block states the house rule verbatim `[VERIFIED: contracts/setfit-apr-v1.yaml:1245-1247]`: `# DECLARED, NOT EXECUTED. cargo-kani is not installed in this repository and no #[kani::proof] harness exists anywhere under crates/. Each entry names the runnable, identically bounded test that is the ACTUAL evidence.`

**Minimal valid skeleton** (every field below is one the parser reads; validated shape, values are the planner's):

```yaml
contract: forecast-tool-boundary            # informational; parser ignores unknown top-level keys
metadata:
  version: 1.0.0                            # SCHEMA-002
  created: '2026-09-05'
  author: PAIML Engineering
  kind: kernel                              # PROVABILITY-001 applies -> obligations, tests, harnesses all required
  description: >
    …what this contract pins and why (tolerances live HERE and are cited by tests)…
  references:                               # SCHEMA-001: at least one
    - "Taylor & Letham (2018). Forecasting at Scale. The American Statistician."
    - ".claude/skills/spike-findings-aprender/references/prophet-fit-and-predict.md (spike 001/003 parity)"
equations:                                  # SCHEMA-003/004: >= 1, each with formula
  prophet_objective_parity:
    formula: "|f_rust(theta_py) - f_python(theta_py)| <= 1e-9"
    domain: "theta_py = Prophet 1.4.0 MAP from the committed fixture"
    invariants: ["data-prep max|diff| == 0.0"]
proof_obligations:                          # >= 1; each `property` non-empty (SCHEMA-005)
  - type: bound
    property: 'Rust objective at the Python MAP is within 1e-9 of the Python -lp on every committed Prophet fixture'
    formal: 'forall fixture: |f_rust(theta_py) - f_py| <= 1e-9'
    applies_to: all
falsification_tests:                        # count >= proof_obligations count; unique ids; prediction non-empty
  - id: FALSIFY-FORECAST-001
    rule: 'Objective parity at the Python MAP'
    prediction: 'objective_at_python_map_within_1e9 passes for peyton, air, retail, wp_log_R'
    test: 'cargo test -p aprender-forecast --lib prophet::parity::objective_at_python_map'
    if_fails: 'Design matrix, changepoint placement or prior terms diverged from Prophet 1.4.0 Stan model'
kani_harnesses:                             # >= 1; id + obligation non-empty
  - id: KANI-FORECAST-001
    obligation: 'future_days(last, h, freq) is strictly increasing for freq in {D, W, MS}'
    property: 'forall last, h <= 3650: ds[i+1] > ds[i]'
    bound: 16
    strategy: bounded_int
    harness: 'NOT EXECUTED — the evidence is the bounded proptest `future_days_strictly_increasing` (FALSIFY-FORECAST-00N)'
```

**Invocation** (pv is not on PATH — `command -v pv` → nothing; a debug binary exists at `target/debug/pv`, 0.63.0, built 2026-09-05): `cargo run -p aprender-contracts-cli --bin pv -- validate contracts/<file>.yaml`. The Makefile uses `PV_BIN := cargo run --release -p aprender-contracts-cli --bin pv --` `[VERIFIED: Makefile:1815]`. **A contract that merely exists is validated by nothing**: `contract-validate` iterates the hardcoded `$(CONTRACTS)` list `[VERIFIED: Makefile:1817-1866, 1929-1935]` and the comment says so (`"Every future phase contract needs its own line here or it is decoration."` `:1920-1927`). Phase pattern: `PHASE4_CONTRACTS := contracts/setfit-apr-v1.yaml` (`:1905`), `PHASE5_CONTRACTS := …` (`:1918`) each with a scoped `contract-audit-phaseN` target. Phase 6 adds its files to `$(CONTRACTS)` and a `PHASE6_CONTRACTS` list. Binding entries (`contracts/aprender/binding.yaml` `[VERIFIED: :1-12]`) have the shape `- contract: <bare filename>.yaml / equation: <key> / module_path: … / function: … / signature: '…' / status: implemented|partial|not_implemented|pending`; STATE.md records the trap that `contract:` must be the bare filename and `status` accepts only those values.

### F5. CI and test wiring — verbatim

The `workspace-test` job's lib leg `[VERIFIED: .github/workflows/ci.yml:289]`:

```
cargo nextest run --profile ci --workspace --lib --exclude aprender-gpu --exclude aprender-cuda-edge --exclude aprender-compute
```

runs inside `docker run … localhost:5000/sovereign-ci:stable` on `[self-hosted, X64, Linux, clean-room]` with `CARGO_INCREMENTAL=0`, `CARGO_BUILD_JOBS=8`, `CARGO_PROFILE_TEST_DEBUG=line-tables-only` (`:270-289`). **Every `#[cfg(test)]` unit test in a new member's `src/` runs here automatically** — no workflow edit. nextest `[profile.ci]` `[VERIFIED: .config/nextest.toml]`: `retries = 2`, `fail-fast = true`, `slow-timeout = { period = "60s", terminate-after = 20 }` (a test is killed at 1200 s), `status-level = "slow"`.

The ONE explicit integration line `[VERIFIED: .github/workflows/ci.yml:471]`:

```
bash -c 'cargo test -p aprender-core --test monorepo_invariants && cargo test -p aprender-core --test readme_contract && cargo test -p apr-cli --test cli_commands && cargo test -p aprender-train-inspect --test falsify_no_fabricated_metadata_2519 && cargo test -p aprender-train-bench --test falsify_no_fabricated_benchmarks_2519 && cargo test -p aprender-train-shell --test falsify_no_fabricated_fetch_2519 && cargo test -p aprender-core --test beat_sklearn_iris && cargo test -p aprender-core --test beat_sklearn_nmi && cargo test -p aprender-core --test beat_sklearn_metrics_parity && cargo test -p aprender-core --test beat_sklearn_gaussiannb_accuracy && cargo test -p aprender-core --test beat_sklearn_svc_accuracy && cargo test -p aprender-core --test beat_sklearn_pipeline_encoder && cargo test -p aprender-serve --test beat_fail_closed_garbage && cargo test -p aprender-compute --lib beat_nf4_bitsandbytes_equivalence && cargo test -p aprender-core --test beat_pytorch_autograd_grad && cargo test -p aprender-train-lora --lib beat_lora_merge_forward_equivalence && cargo test -p apr-cli --release --test beat_pytorch_deploy_footprint && cargo test -p aprender-serve --test beat_fail_closed_structural && cargo test -p aprender-serve --test ollama_http_compat && cargo test -p apr-cli --test ollama_ndjson_streaming && cargo test -p apr-cli --test falsification_chat_http_cli && cargo test -p aprender-contracts --test apr_serve_api_key_auth_contract && cargo test -p apr-cli --test falsify_auth_001 --test falsify_auth_002 --test falsify_auth_003 --no-fail-fast && cargo test -p apr-cli --test beat_apr_data_alimentar_reach && cargo test -p apr-cli --test beat_apr_sibling_cli_reach && cargo build --examples --workspace --keep-going'
```

The comment above it: "only one PR at a time may edit this single physical line" (`:440-441`) and `readme_contract.rs:11-16` records that "547 of 573 integration targets in this repo are dark for exactly this reason". Note the trailing `cargo build --examples --workspace --keep-going` — any `examples/*.rs` the new crates ship (e.g. the spike-006 MASE harness as an example, D-16) **is compiled in CI**, so it must build clean.

**Recommendation (Claude's discretion, decided by the evidence):** keep parity tests, the in-process streamable-HTTP e2e (spike 004/007 `tests/e2e.rs` uses an in-process `axum` app + `reqwest`, so it needs no binary) and the spike-010 concurrency/equality test as `#[cfg(test)]` modules inside `src/` — they run under the nextest leg without touching ci.yml. Only the **spawned-binary stdio** e2e (needs `env!("CARGO_BIN_EXE_…")`, available only to integration targets) must be a `tests/*.rs` file; it is then dark in CI exactly like `aprender-mcp-setfit/tests/e2e_stdio.rs`, and adding it to line 471 is an `autonomous: false` step. `reqwest` becomes a `[dev-dependencies]` entry (`reqwest = { workspace = true, features = ["json", "rustls-tls"] }`, matching the lambda crate's TLS choice).

**Fixture loading precedent** `[VERIFIED: crates/aprender-core/tests/setfit_conformance.rs:137; crates/apr-cli/tests/setfit_cli_lifecycle.rs:392]`: `PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/setfit")` and cross-crate `.join("../aprender-core/tests/fixtures/setfit")`. `include_str!` is used by the spike servers for the page and sample CSVs (`include_str!("../static/index.html")`, `include_str!("../fixtures/peyton_manning.csv")`, `include_str!("../fixtures/air_passengers.csv")` `[VERIFIED: sources/007/src/lib.rs:168-170; sources/004/src/lib.rs:303-305]`) — so each server crate needs `static/index.html` **and** a `fixtures/` dir with those two CSVs (85 KB + 2 KB) inside the crate. **The `static/index.html` pages are NOT in the skill's `sources/`** — they live only at `.planning/spikes/004-forecast-mcp-thin-server/static/index.html` (10.3 KB) and `.planning/spikes/007-chronos-mcp-thin-server/static/index.html` (10.0 KB) `[VERIFIED: ls]`; D-08's "port `static/index.html`" must copy from there. `scripts/check_test_fixture_paths.sh` `[VERIFIED: :40]` flags only `"/home/…"` / `"/Users/…"` literals in `tests?/` files (baseline 85); a `CARGO_MANIFEST_DIR/../../.planning/spikes/…` path is workspace-relative and passes, but see Pitfall 6 for why copying the oracle JSON into `crates/aprender-forecast/tests/fixtures/` (≈ 7 MB, all 18 fixture files are git-tracked `[VERIFIED: git ls-files]`) is the safer choice.

### F6. Model weights in CI (D-18) — mechanism, pin, and the skip that counts

**Repo conventions found.** `.gitignore:48` `*.safetensors` (global) and `:51-53` `# Root-level models/ only (NOT src/models/ which is source code)` / `/models/` `[VERIFIED]` — root-anchored per CB-510, so `/models/chronos-bolt-tiny/` is already ignored and no `.gitignore` edit is needed. A `justfile` **exists** (root, 16 KB, `[VERIFIED: justfile:1-8]`: "The repo's quality gates live in the Makefile (tier1..tier4, coverage, contract audits) and stay there — this file is the deployment surface"; `set shell := ["bash", "-uc"]`; recipes `build-apr-arm64`, `build-trainer-asset`, `pmcp-train-deploy`, …). A weight-fetch recipe is deployment-surface work and belongs there, matching CLAUDE.md's `just` preference. The SetFit precedent for un-committed artifacts is `scripts/setfit_fixtures/fetch_full_weights.py` (`hf_hub_download` at a pinned `REVISION`, sha256 recorded in a manifest, "NEVER called from CI", target `~/.cache/aprender/…` via env var `[VERIFIED: :1-45]`) and `hf-hub` is only used by `apr-cli` for publishing (`crates/apr-cli/src/commands/publish.rs`) — there is no in-repo fetch-for-tests helper to reuse.

**The pin** (from the local HF cache, `[VERIFIED: ~/.cache/huggingface/hub/models--amazon--chronos-bolt-tiny/refs/main + snapshots/]`):
- revision `a0e552de83495b5c28c14c71c374f3e33280b340`
- `model.safetensors` (F32, 33.0 MB) sha256 `75068728d376d2bec670379eeef4bfb4d24c0cfe24d957451f8d19b447030a32`
- `config.json` (1.1 KB) sha256 `278f0086733031635fb1c861cb01c1bad6477420c7fcb19381a2993e335785e0`
- derived tiny-f16 `model.safetensors` (16.5 MB) sha256 `f9a033b42bc516e17ae5756317cb946121afdb59c94b4acfcd30fef93317cd4c` (`[VERIFIED: .planning/spikes/007-chronos-mcp-thin-server/models/tiny-f16/]` — produced by `tools/to_f16.py`; the f16 file is a deterministic function of the f32 file, so pin the f32 sha and re-derive f16 rather than downloading it)
- Model card: license `apache-2.0`, 8.65M params F32, files `model.safetensors` + `config.json` `[CITED: https://huggingface.co/amazon/chronos-bolt-tiny]` (LOW per the classify seam; consistent with the local files)
- `huggingface_hub.hf_hub_download(repo_id: str, filename: str, *, …, revision: str | None = None, …, local_dir: … = None, …) -> str` `[VERIFIED: uv run --with huggingface_hub (1.30.0) inspect.signature]`

**Recommended mechanism** (`just fetch-chronos-tiny`): `uv run --python 3.12 --with huggingface_hub --with safetensors --with numpy python - <<'PY' …` downloads the two files at the pinned revision into `/models/chronos-bolt-tiny/f32/`, verifies both sha256s (abort on mismatch), converts to `/models/chronos-bolt-tiny/f16/` with the spike's `to_f16.py` logic, and verifies the f16 sha. Then `CHRONOS_MODEL_DIR=$PWD/models/chronos-bolt-tiny/f16` arms the runtime path and `CHRONOS_EMBED_DIR=…` arms the embedded build (D-13).

**The skip that counts — positively probed this session** (`[VERIFIED: scratchpad crate on rustc 1.93.0 / nextest 0.9.102]`). `build.rs` of the Chronos server (and of `aprender-forecast` for its Bolt parity tests):

```rust
fn main() {
    println!("cargo::rustc-check-cfg=cfg(chronos_weights)");   // declares the cfg -> no unexpected_cfgs warning
    println!("cargo:rerun-if-env-changed=CHRONOS_MODEL_DIR");
    if let Some(dir) = std::env::var_os("CHRONOS_MODEL_DIR") {
        if std::path::Path::new(&dir).join("model.safetensors").is_file() {
            println!("cargo:rustc-cfg=chronos_weights");
        }
    }
}
```

```rust
#[test]
#[cfg_attr(not(chronos_weights), ignore = "CHRONOS_MODEL_DIR unset or has no model.safetensors — run `just fetch-chronos-tiny` to arm")]
fn bolt_quantiles_match_oracle_within_1e6() { /* … */ }
```

Observed output, weights absent — libtest: `test weight_dependent_parity ... ignored, CHRONOS_MODEL_DIR unset or has no model.safetensors — run \`just fetch-chronos-tiny\` to arm` / `test result: ok. 1 passed; 0 failed; 1 ignored; …`; nextest: `Starting 1 test across 1 binary (1 test skipped)` / `Summary [0.006s] 1 test run: 1 passed, 1 skipped`. Weights armed (`CHRONOS_MODEL_DIR=… cargo test`): `test result: ok. 2 passed; 0 failed; 0 ignored`. This satisfies D-18's "visible reason and a non-zero count of skipped tests" without `--ignored` flags and without the `println!`-and-return style that reports `0 ignored`. The web digest agrees that `cargo::rustc-check-cfg` is the sanctioned way to declare a build-script cfg `[CITED: https://doc.rust-lang.org/rustc/check-cfg/cargo-specifics.html]`.

**"At least one CI leg exercises the embedded-weights build"** cannot be satisfied without either (a) a CI step that runs `just fetch-chronos-tiny` then `CHRONOS_EMBED_DIR=… cargo test -p aprender-mcp-chronos` (a `.github/workflows/*.yml` edit → human check-in, `autonomous: false`), or (b) a local release-gate recipe (`just chronos-gate`) recorded in the phase's VALIDATION evidence. The planner should schedule (b) as the phase gate and file (a) as the CI follow-up with the exact step text. The CI runner is X64 Linux (`:96`), where `gemm_blis` runs the scalar/AVX2 kernels, not NEON — the spike's Apple-M4 timing numbers (SC4's "under 100 ms", "under 150 ms cold start") are **not** portable to that box; assert parity in CI and timing only on the aarch64 host (Validation Architecture).

### F7. README / CLAUDE.md drift gates — measured state on this branch

Run this session: `rtk proxy cargo test -p aprender-core --test monorepo_invariants --test readme_contract --no-fail-fast` → **`monorepo_invariants`: 10 passed, 1 failed; `readme_contract`: 11 passed, 4 failed** (`[VERIFIED: run output, scratchpad/drift-gates.log]`). Verbatim failures:

1. `FALSIFY-MONO-011: Unauthorized [[bin]] sections found in: ["aprender-mcp-setfit", "aprender-mcp-setfit-lambda", "aprender-mcp-setfit-train", "aprender-setfit-train-lambda"]` — **pre-existing and not in the CONTEXT's list**. The allowlist (`monorepo_invariants.rs:238-266`) has 27 entries and `ALLOWLIST_BASELINE = 27` with a shrink-only assertion; the four SetFit deployment crates were never added. Phase 6's `aprender-mcp-forecast` and `aprender-mcp-chronos` each ship a `[[bin]]` → two more violations. See Open Question 1.
2. `FALSIFY-README-CRATE-001: Crates missing README.md: ["aprender-mcp-setfit-lambda", "aprender-contrastive-data"]` — **two** crates, not one (the test skips dirs without `Cargo.toml` `[VERIFIED: readme_contract.rs:245]`; `crates/deploy/` has none and is not counted).
3. `FALSIFY-README-CRATE-002: READMEs without monorepo link: ["aprender-mcp-setfit"]`.
4. `FALSIFY-README-005: README claims-table row does not match cargo. expected: | Workspace crates | **83** workspace crates |` — README line 43 says `**82**` `[VERIFIED: README.md:43]`. The row format is fixed by `format!("| Workspace crates | **{crate_count}** workspace crates |")` `[VERIFIED: readme_contract.rs:130]`. After +3 crates: **86** (+5 with two lambda wrappers: **88**).
5. `FALSIFY-README-007: README lacks **1786** provable contracts` — README line 44 says `**1778**`; the checked string is `format!("**{contract_count}** provable contracts")` `[VERIFIED: readme_contract.rs:171]`, counting `*.yaml` recursively under `contracts/`. After Phase 6: 1786 + N new contracts.

Derived counts this session `[VERIFIED: commands run]`: `cargo metadata --no-deps … | python3 -c …` → **83**; `find contracts -name '*.yaml' | wc -l` → **1786**.

FALSIFY-DOCS-CLAUDE-001 (`test_documented_paths_exist`, currently passing) extracts every backticked token in `CLAUDE.md` and `docs/BEATS.md` that contains `/`, has no `..`, none of `' ', '*', '<', '>', '|', '(', ')', '[', ']', '{', '}', '?'`, does not start with `http`, `hf://`, `~`, `/`, `target/`, and ends in one of `.rs .yaml .yml .toml .sh .md` `[VERIFIED: readme_contract.rs:456-466]`; lines containing `DELETED` or `[gitignored]` are skipped (`:474`). Consequence: every path the new CLAUDE.md row cites (e.g. `crates/aprender-forecast/src/lib.rs`, `contracts/forecast-tool-boundary-v1.yaml`) must exist **in the same commit** the row lands, and the gate requires ≥ 20 extracted paths overall (`:499-503`).

### F8. Realizar-first exception row (D-07) — text for the executor

CLAUDE.md table header and SetFit row `[VERIFIED: CLAUDE.md:209-216]`:

```
| Responsibility | aprender | realizar | trueno |
|---------------|----------|----------|--------|
…
| SetFit Classification Inference | **Primary** (aprender-core: loader + `VerifiedSetFitModel::classify`) | HTTP transport ONLY (route/`AppState`/readiness — calls core) | Compute |
```

Proposed row (insert after line 216):

```
| Time-Series Forecasting (Prophet / NeuralProphet fit+predict, Chronos-Bolt zero-shot) | **Primary** (`crates/aprender-forecast`; thin pmcp servers `crates/aprender-mcp-forecast` and `crates/aprender-mcp-chronos` are transport only) | Never | Compute (`blis::gemm_blis`, NEON 8×6 kernel) |
```

Proposed paragraph (after the SetFit paragraph, CLAUDE.md line 226), modelled on lines 218-226:

```
**The Forecasting row is a second deliberate EXCEPTION (Phase 6 D-07), for a different reason.**
Prophet and NeuralProphet "inference" IS a fit — L-BFGS on a Stan-shaped objective and AdamW on
the autograd — which is training-side machinery by this table's own first row, and the fit runs
inside every stateless `forecast` call (D-01). Chronos-Bolt is an 8.65M-parameter T5 with no
tokenizer, no KV cache and no LLM kernels; its ONLY conformance-proven implementation is the
spike-005/007 port (parity 9.5e-7 against `chronos-forecasting` 2.3.1), so the evidence lives in
`crates/aprender-forecast`. Serving a second, unproven port through realizar would violate OPS-03
(one implementation per operation). The thin servers own the tool boundary and transport only;
`aprender-serve` is untouched. Tolerances and the tool boundary: `contracts/forecast-*.yaml`
(see the phase's contracts list).
```

Every backticked path in that text must exist when it lands (F7). Keep parameter counts as measured (8.65M `[CITED: model card]` / "8.7M" in D-07 is a rounding of the same number).

### F9. Dependency availability — `Cargo.lock` resolved versions

| Crate | Resolved in `Cargo.lock` | Spike request | Direct dep already? |
|---|---|---|---|
| `pmcp` | **2.19.3** (single version) | `2.19` | yes — `aprender-mcp-setfit`(2.9→2.19.3), `-setfit-train` (2.19) |
| `schemars` | 1.2.1 (also 0.8.22, 0.9.0 transitively) | `1.0` | yes — both MCP crates |
| `axum` | 0.8.9 (also 0.7.9) | `0.8` | workspace dep `axum = { version = "0.8", features = ["ws"] }` |
| `tower` | 0.5.3 (also 0.4.13) | `0.5` | workspace dep; `apr-cli` direct |
| `reqwest` | 0.12.28 (also 0.13.4) | `0.12` (dev) | workspace dep `default-features = false`; lambda crate adds `["json", "rustls-tls"]` |
| `half` | 2.7.1 | `2.7` | workspace dep `2.4` (resolves 2.7.1); `apr-cli` direct |
| `tokio` | 1.52.3 | `1` full | workspace dep |
| `serde` / `serde_json` | 1.0.228 / 1.0.150 | `1` | workspace deps |
| `lambda_http` / `lambda_runtime` | 0.13.0 | — | `aprender-mcp-setfit-lambda` |
| `safetensors` (crate) | 0.4.5 / 0.7.0 / 0.8.0 | **not used** — the spike ships its own 41-line decoder | n/a |

`[VERIFIED: Cargo.lock awk over [[package]] blocks; crates/*/Cargo.toml grep]`. **No new registry package enters the graph**, so the Package Legitimacy Gate has nothing new to check (audit below records the existing ones). `deny.toml` `[VERIFIED: :56-75]` allows MIT / Apache-2.0 / BSD / ISC / Unicode / Zlib / MPL-2.0 / … and bans `wildcards = "deny"` with `allow-wildcard-paths = true` (path deps without `version` are fine — the sibling dev-dep convention). The Chronos weights are Apache-2.0 model files, not a crate, so cargo-deny does not see them; record the license in the crate README.

### F10. Validation architecture inputs — measured runtimes

Release (spike, Apple M4 Pro, `[CITED: sources/001/RUN-OUTPUT.md:149; sources/004/BENCH.md]`): Peyton single-round L-BFGS fit 0.259 s; full spike-004 fit with restarts 1.41 s round-trip; 3 000 daily points 0.215 s; NP lag-free 0.16 s; NP 30 lags 2.75 s; Chronos-tiny forward 18.5 ms / 2 048 points; embedded tiny-f16 cold start 52 ms.

**Debug profile (what CI's nextest leg builds)** — measured this session by building `.planning/spikes/001-prophet-map-fit-lbfgs` with `cargo build` (dev) and timing its driver `[VERIFIED: scratchpad/spike001-debug.md vs sources/001/RUN-OUTPUT.md]`: the same `exact, f/T, guard, m=20, tol 1e-7` Peyton fit took **9.152 s** (726 iterations, 2 677 evals) vs **0.259 s** in release — **35.3× slower**; the `smooth 1e-4` variant 522.265 s vs 13.504 s (**38.7×**); the whole 3-dataset driver did not finish inside a 600 s cap (release: seconds). Every debug row's iteration count, eval count, objective (`-8004.9295`) and Δf equal the release row **exactly**, so parity assertions are profile-independent — only wall time changes. A Peyton parity test with the D-09 restart loop (1 312 iterations in release, 1.4 s) therefore costs ≈ 50 s in debug, and the four Prophet fixtures together several minutes — under nextest's 1 200 s kill but a real tax on the 75-minute job. The root `Cargo.toml` has **no `[profile.release]`/`[profile.test]` opt-level overrides**, only `[profile.test.package.proptest] debug-assertions = false` and the dev twin `[VERIFIED: Cargo.toml:649-653]` — that precedent shows per-package profile overrides are already accepted in this tree. Recommendation: add `[profile.dev.package.aprender-forecast] opt-level = 3` (inherited by `profile.test`) in the root `Cargo.toml` **after** measuring the debug parity-suite time in Wave 0 (`[ASSUMED]` that the override alone recovers most of the 35×; the Prophet objective is pure f64 loops in the same crate, so it should).

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `aprender-core` (lib `aprender`) | workspace 0.63.0, `default-features = false` | `LbfgsF64`, `Vector<f64>`, autograd `Tensor`, `nn::Linear`, `nn::optim::AdamW` | The spike ports compiled against exactly these (F3) |
| `aprender-compute` (lib `trueno`) | workspace 0.63.0 | `blis::gemm_blis` with the NEON 8×6 kernel | D-14; D-05 says the kernel is already in tree |
| `pmcp` | 2.19.3 `features = ["streamable-http", "schema-generation"]` | `Server::builder`, `tool_typed_with_description`, `run_stdio`, `pmcp::axum::router_with_config`, `StreamableHttpServerConfig::stateless` | The SetFit trio and both spike servers (F2) |
| `schemars` | 1.2.1 (`"1.0"`) | `JsonSchema` derive on `ForecastArgs` | pmcp's schema-generation feature uses this major |
| `axum` | 0.8.9 (workspace) | Same-origin app: `GET /` page + `nest("/mcp", …)`; pool `fallback` | Spike 004/007/010 |
| `tower` | 0.5.3 (workspace) | `ServiceExt::oneshot` for the round-robin router pool | Spike 010 |
| `tokio` | 1.52.3 (workspace, `full`) | runtime, `spawn_blocking` | Template |
| `serde` / `serde_json` | workspace | args/response, `deny_unknown_fields` | Template |
| `half` | 2.7.1 (workspace) | F16/BF16 → f32 in the 41-line safetensors decoder | Spike 007 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `reqwest` | 0.12.28 (workspace, dev-dep, `["json", "rustls-tls"]`) | in-process streamable-HTTP e2e client | tests only |
| `lambda_http`, `once_cell`, `tracing`, `tracing-subscriber` | 0.13.0 / 1.19 / 0.1 / 0.3 | Lambda loopback wrapper | only if the wrapper crates are taken (Claude's discretion) |
| `uv` + `huggingface_hub` 1.30.0 + `safetensors` + `numpy` (Python, via `uv run --with`) | probed: uv 0.9.5, Python 3.13.7 | `just fetch-chronos-tiny`, `to_f16.py`, oracle regeneration | developer/gate workflow, never on the request path |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Own 41-line safetensors decoder (`sources/007/src/safetensors.rs`) | `safetensors` crate 0.4/0.7/0.8 (already in lock) | The crate does not decode F16/BF16 to f32 or transpose; the spike loader works on a byte slice for both embedded and on-disk weights and is parity-proven. Keep the port (D-08). |
| Civil-date arithmetic (`dates.rs`, 49 lines) | `chrono` 0.4 (workspace dep) | D-17 forbids `chrono`; the arithmetic is exact-parity proven. |
| Hand-rolled argv loop in `main.rs` | `clap` 4.5 (workspace dep) | Template uses a 30-line loop; clap adds compile time and a new direct dep to a 7 MB binary. Keep the loop. |
| `#[cfg_attr(not(chronos_weights), ignore = "…")]` | `println!("SKIP…"); return;` (template style) | The template style reports `0 ignored` — a silent green, forbidden by D-18. Use the cfg_attr form (F6). |

**Installation:** none — every crate is already in `Cargo.lock`; new members declare the lines in F1/F2 and `cargo build` resolves offline against the lock.

**Version verification:** performed against `Cargo.lock` (F9); `cargo search` was not run because no new registry package is introduced.

## Package Legitimacy Audit

No package new to the dependency graph is introduced (F9). The `gsd-tools query package-legitimacy check` seam was therefore not needed for slopsquat screening; the table records the crates the new manifests will name, all already resolved and building in this workspace.

| Package | Registry | Resolved | Already in graph via | Verdict | Disposition |
|---------|----------|----------|----------------------|---------|-------------|
| `pmcp` | crates.io | 2.19.3 | `aprender-mcp-setfit`, `-setfit-train`, `-setfit-lambda` | OK (in lock, builds) | Approved |
| `schemars` | crates.io | 1.2.1 | same | OK | Approved |
| `axum` | crates.io | 0.8.9 | workspace dep | OK | Approved |
| `tower` | crates.io | 0.5.3 | workspace dep, `apr-cli` | OK | Approved |
| `half` | crates.io | 2.7.1 | workspace dep, `apr-cli`, core `format-quantize` | OK | Approved |
| `reqwest` | crates.io | 0.12.28 | workspace dep | OK | Approved (dev) |
| `tokio`, `serde`, `serde_json` | crates.io | 1.52.3 / 1.0.228 / 1.0.150 | workspace deps | OK | Approved |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
MCP client (Claude Desktop / Code, browser page, Lambda proxy)
   │  JSON-RPC  tools/call { name: "forecast", arguments: ForecastArgs }
   ├── stdio ──────────────► Server::run_stdio ─┐
   └── streamable-HTTP ──► axum Router          │
        GET /  → static/index.html (MCP client) │
        /mcp   → [fit server] round-robin fallback over K pmcp routers ──┐
                 [chronos]    one pmcp router (Arc<Model> immutable)     │
                                                                          ▼
                                    pmcp Server (tool_typed_with_description::<ForecastArgs>)
                                                │  serde deny_unknown_fields → pmcp::Error::validation on any defect
                                                │  precheck: MIN/MAX_POINTS, horizon, ds ascending+unique+real dates,
                                                │            freq ∈ {D,W,MS}, interval_width, cap > max(y), constant y
                                                ▼
                                    tokio::task::spawn_blocking
                                                │
              ┌─────────────────────────────────┼─────────────────────────────────────┐
              ▼ model: prophet                   ▼ model: neuralprophet                 ▼ (aprender-mcp-chronos)
   aprender_forecast::prophet             aprender_forecast::np                aprender_forecast::bolt
   Spec→columns→make_design               features → NpModel(Linear…)          instance-norm → patches →
   Model(obj ÷ T, guard) → fit_prophet    weighted_huber (ops) → AdamW         ResidualBlock → T5 enc/dec
   (LbfgsF64 2000/1e-7/20, Stalled=ok,    one-cycle lr, clear_graph/step       (gemm_blis multi-row, dot8 single)
    restarts ≤8, 15 s budget, cache)      → predict_ts / predict_ar_*         → 9 quantiles [× rollout >64]
   → predict(yhat, band[seeded], comps)                                        ← weights: include_bytes!(OUT_DIR) or CHRONOS_MODEL_DIR
              └─────────────────────────────────┬─────────────────────────────────────┘
                                                ▼
                       ForecastResponse { ds, yhat, yhat_lower, yhat_upper, trend, components|quantiles,
                                          fit_seconds, predict_seconds, diagnostics, warning? }
                                                │  serde_json::to_value → content[0].text (+ structuredContent)
                                                ▼
                                            client
```

### Recommended Project Structure

```
crates/aprender-forecast/                 # library; publish = false; [lib] name = "aprender_forecast"
├── Cargo.toml                            # aprender (core, default-features=false) + trueno (compute) + half + serde
├── README.md                             # must contain "paiml/aprender"; states EMPIRICAL coverage (D-16)
├── build.rs                              # emits cfg(chronos_weights) when CHRONOS_MODEL_DIR has model.safetensors (F6)
├── src/lib.rs                            # pub mod prophet; pub mod np; pub mod bolt; pub mod safetensors; pub mod dates; pub mod fit; pub mod types
├── src/prophet.rs                        # sources/004/src/prophet.rs (520 lines) as-is
├── src/np.rs                             # sources/004/src/np.rs (384) as-is
├── src/fit.rs                            # sources/006/src/fit.rs (29) + Cached/x-keyed cache from 004 lib.rs
├── src/bolt.rs, src/safetensors.rs, src/dates.rs   # sources/007/src/* (323 + 41 + 49)
├── src/types.rs                          # ForecastArgs/ForecastResponse/HolidayArg (shared by both servers), refusals
├── src/parity/                           # #[cfg(test)] parity ladders: prophet (4 fixtures), np (1), bolt (2) — run by CI's --lib leg
├── tests/fixtures/                       # COPIED oracle JSON (≈7 MB, see Pitfall 6) — or referenced from .planning/spikes
└── examples/mase_rolling_origin.rs       # spike-006 harness (D-16) — compiled by CI's `cargo build --examples --workspace`
crates/aprender-mcp-forecast/             # thin server; [[bin]] name = "aprender-mcp-forecast"
├── Cargo.toml, README.md, static/index.html (from .planning/spikes/004/static), fixtures/{peyton_manning,air_passengers}.csv
├── src/lib.rs                            # TOOL_NAME/TOOL_DESCRIPTION, build_server, http_app, pooled_app(K) ; #[cfg(test)] mod e2e (in-process HTTP + refusals + spike-010 equality)
├── src/main.rs                           # --stdio (default) | --http PORT [--pool K] | --bench
└── tests/e2e_stdio.rs                    # spawned binary; dark in CI unless ci.yml:471 is edited (autonomous: false)
crates/aprender-mcp-chronos/              # thin server; [[bin]] name = "aprender-mcp-chronos"; build.rs stages CHRONOS_EMBED_DIR → OUT_DIR
├── Cargo.toml, README.md, build.rs, static/index.html (from .planning/spikes/007/static), fixtures/*.csv
├── src/lib.rs                            # EMBEDDED_WEIGHTS/EMBEDDED_CONFIG, resolve_model, horizon gate, build_server, http_app; #[cfg(test)] e2e (cfg_attr-ignored when no weights)
└── src/main.rs                           # --stdio | --http PORT | --coldstart N
contracts/forecast-tool-boundary-v1.yaml, contracts/prophet-parity-v1.yaml, contracts/chronos-bolt-parity-v1.yaml   # setfit-apr-v1 shape (F4); appended to Makefile $(CONTRACTS) + PHASE6_CONTRACTS
justfile                                  # + fetch-chronos-tiny, chronos-embed-build, chronos-gate, forecast-bench, chronos-coldstart
```

### Pattern 1: Thin server = transport over one library implementation
**What:** The server crate owns only `ForecastArgs` → refusals → `spawn_blocking(forecast)` → `serde_json::to_value`; every numeric path lives in `aprender-forecast`.
**When to use:** Both servers. The CLAUDE.md SetFit exception paragraph is the precedent ("does not reimplement the tokenizer, pooling or head").
**Example:** F2's `build_server` verbatim, with `ClassifyArgs` → `ForecastArgs` and `model.classify(&document)` → `aprender_forecast::forecast(&args)` (fit server) / `bolt::forecast(&model, &args)` (Chronos, `Arc<Model>` cloned into the closure like `Arc::clone(&model)` in the template).

### Pattern 2: Router pool behind a round-robin `fallback` (fit server only)
**What:** Build K `http_app(build_server(…))` routers; a `fallback` handler dispatches `req` to `routers[i].clone().oneshot(req)` with `i = next.fetch_add(1) % K`.
**When to use:** The HTTP app of `aprender-mcp-forecast` (D-12). Not the Chronos server (18 ms calls, immutable model) and not stdio.
**Example:** `sources/010-forecast-server-concurrency/src/main.rs` `fn app(pool: usize)`, quoted in `references/forecast-mcp-thin-server.md` §4. Note it uses `.unwrap_or_else(|e| match e {})` on `Infallible` — allowed by `.clippy.toml` (only `unwrap` is banned).

### Pattern 3: Build-time embed with runtime fallback and a compile-time-gated test
**What:** `build.rs` stages `CHRONOS_EMBED_DIR/{model.safetensors,config.json}` into `OUT_DIR` (empty markers otherwise); `resolve_model()` prefers `EMBEDDED_WEIGHTS` else `CHRONOS_MODEL_DIR`; the same `build.rs` emits `cfg(chronos_weights)` so weight tests are counted-skipped without weights (F6).
**When to use:** `aprender-mcp-chronos`, and `aprender-forecast`'s Bolt parity tests (runtime path only).
**Example:** `references/chronos-mcp-server.md` §1 + F6's `build.rs`/`cfg_attr` snippets.

### Pattern 4: Parity ladder as the test shape
**What:** One test per rung — data prep 0.0 diff → objective at oracle MAP ≤ 1e-9 → oracle params through Rust predict ≤ 1e-10 → fitted objective ≤ oracle + 0.5 → forecast within the Newton-vs-L-BFGS band → components rebuild `yhat` → band widths within 2 % (SC2); Chronos: loc/scale → patches → embeddings → hidden states → quantiles → rollout (SC4). Tolerances read from the contract YAML at test time (D-15) rather than duplicated.
**When to use:** All parity tests. The spike drivers (`sources/00{1,3,5}/src/main.rs`) already compute every rung; the port turns each printed table row into an `assert!`.

### Anti-Patterns to Avoid
- **Rewriting `prophet.rs`/`np.rs`/`bolt.rs` "idiomatically"** while porting — D-08; every deviation re-opens the ladder. Only clippy/fmt/`unwrap` edits.
- **`println!("SKIP"); return;` for weight-gated tests** — reports `0 ignored`; forbidden by D-18 (F6).
- **Putting parity tests in `tests/*.rs`** — dark in CI (F5); keep them `--lib`.
- **Copying `contracts/neon-blis-v1.yaml`'s keys** (`falsification:`, `registry: true`) — parses to 0 tests and skips PROVABILITY-001 (F4).
- **Editing `ALLOWLIST_BASELINE` or the allowlist autonomously** — it is a shrink-only ratchet with an explicit policy sentence; surface it (Open Question 1).
- **Citing a path in CLAUDE.md before the file exists** — FALSIFY-DOCS-CLAUDE-001 fails the CI-wired `readme_contract` target (F7).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| f64 L-BFGS with Wolfe search | a new optimiser | `aprender::optim::LbfgsF64` + the D-09 wrapper (÷T, guard, restarts, cache) | Proven to Prophet parity on four fixtures; spike 001 §4 shows the raw solver fails at iteration 0 without the wrapper — the wrapper is the deliverable, not a new solver |
| Huber loss on the autograd | `nn::loss::SmoothL1Loss` | `weighted_huber` from ops with a constant mask (`sources/004/src/np.rs`) | Core's is detached (F3 quote) |
| GEMM | `trueno::Matrix::matmul`, plain loops, rayon `blis::gemm` | `trueno::blis::gemm_blis` (rows > 1), `dot8` (one row) | D-14; 65–77 GFLOP/s vs 9 for loops (`references/gemm-neon-kernel-upstream.md`) |
| Torch-style `quantile` | ad-hoc percentile | `torch_quantile` in `bolt.rs` (linear interpolation) | The 9-path re-quantiled rollout is off by 0.5 with any other scheme (`references/chronos-zero-shot-port.md`) |
| Streamable-HTTP + CORS + DNS-rebinding protection | custom axum handlers for `/mcp` | `pmcp::axum::router_with_config(server, RouterConfig{ server_config: stateless(), allowed_origins: Some(AllowedOrigins::localhost()), ..Default::default() })` | pmcp 2.19.3 ships security headers + origin allowlist (`axum_router.rs:47-55`) |
| MCP tool schema | hand-written JSON schema | `#[derive(JsonSchema)]` + `tool_typed_with_description` | Template; `additionalProperties: false` surfaces from `deny_unknown_fields` (`lib.rs:305-320` test) |
| Weight download | `reqwest` in the binary, or a bash `curl` | `just fetch-chronos-tiny` → `uv run --with huggingface_hub` `hf_hub_download(revision=…)` + sha256 | Never network on the request path (`references/chronos-mcp-server.md` "What to Avoid"); mirrors `scripts/setfit_fixtures/fetch_full_weights.py` |
| Contract validation | yq/python/bash | `cargo run -p aprender-contracts-cli --bin pv -- validate` | CLAUDE.md "DOGFOOD `pv`, NEVER bash" |
| Counted skip without weights | env-var `println!` + return | `build.rs` `rustc-cfg` + `#[cfg_attr(not(chronos_weights), ignore = "…")]` | Only form that produces `N ignored` / `N skipped` (F6 probe) |

**Key insight:** every numeric component already exists in a proven form; the phase's engineering is in the seams — manifests, lints, gates, embedding, and the skip semantics — where the repo's own drift tests are unforgiving and mostly already red.

## Common Pitfalls

The spikes' "What to Avoid" sections are the numerical pitfalls list — cite, don't restate: `references/prophet-fit-and-predict.md` (Stalled ≠ failure; no FD at the exact MAP; sort/de-dup `ds`; CR-only `air_passengers.csv`), `references/neuralprophet-autograd.md` (never full-batch; select lr by train loss; `SmoothL1Loss`), `references/forecast-mcp-thin-server.md` (router lock; no fit→artifact flow; same-origin page; iteration cap + budget; `rtk` hides `println!`), `references/chronos-zero-shot-port.md` (rollout scheme; T5 norm/scaling; both weight layouts), `references/chronos-mcp-server.md` (no lazy Hub download; 103 MB > 50 MB zip; nominal vs empirical bands), `references/gemm-neon-kernel-upstream.md` (dead `microkernel_8x8_neon`; `test_brick_profiler_reset_v2` flake). The pitfalls below are the **integration** ones this research found.

### Pitfall 1: Pedantic clippy under `-D warnings` on 2 249 lines written for a bare crate
**What goes wrong:** SC5 requires `cargo clippy -- -D warnings` on the new crates; the workspace inherits `clippy::all` + `clippy::pedantic` as `warn` (F1), so dozens of `needless_pass_by_value`, `cast_lossless`, `items_after_statements`, `too_many_lines`, `struct_excessive_bools` … become errors, and every `.unwrap()` is a `disallowed_methods` error.
**Why it happens:** The spikes had no `[lints]` table (`sources/*/Cargo.toml`).
**How to avoid:** First task of each port plan: `cargo clippy -p <crate> --all-targets -- -D warnings` and fix mechanically; add `#![allow(clippy::disallowed_methods)]` **only** at the file scope where `JsonSchema`/`json!` expand (template `lib.rs:29-32`), never crate-wide. Budget it — it is the only reason to touch a ported line (D-08).
**Warning signs:** A plan that ports and "then runs clippy" as an afterthought.

### Pitfall 2: FALSIFY-MONO-011 makes CI red for any new `[[bin]]`
**What goes wrong:** `monorepo_invariants` (CI line 471) already fails on 4 SetFit crates; Phase 6 adds 2 (or 4). The allowlist is asserted shrink-only at 27 with a policy message (F1, F7).
**Why it happens:** The thin-MCP-server pattern (Phase 4, user-ratified) and the "only apr-cli ships binaries" ratchet were never reconciled.
**How to avoid:** Open Question 1 — a human decision at plan start, before any executor edits `monorepo_invariants.rs`.
**Warning signs:** A plan task that raises `ALLOWLIST_BASELINE` without an `autonomous: false` marker.

### Pitfall 3: The `neon-blis-v1.yaml` template is hollow
**What goes wrong:** Mirroring it yields contracts with 0 obligations/tests/harnesses that still pass `pv validate` — D-15 unmet while green.
**How to avoid:** F4 skeleton; verify with `pv status <file>` that counts are non-zero, and `pv validate` after adding `kind: kernel` (which makes PROVABILITY-001 bite).
**Warning signs:** `pv status` showing `Falsification tests: 0`.

### Pitfall 4: A contract file that exists but is validated by nothing
**What goes wrong:** `make contract-validate`/tier3 iterate only `$(CONTRACTS)` (`Makefile:1817-1866`).
**How to avoid:** Append each new YAML to `$(CONTRACTS)` and add `PHASE6_CONTRACTS` + a scoped `contract-audit-phase6` (pattern at `:1905-1918`). Bindings need bare filenames and `status ∈ {implemented, partial, not_implemented, pending}`.

### Pitfall 5: README/CLAUDE.md counts and paths
**What goes wrong:** Four `readme_contract` failures exist; the phase must land `**86** workspace crates` (or 88), `**<1786+N>** provable contracts`, a README with `paiml/aprender` in each new crate, fix `aprender-mcp-setfit`'s README, add READMEs for `aprender-mcp-setfit-lambda` and `aprender-contrastive-data` (or accept those two as out-of-phase debt — but SC5's "every gate is green" argues for fixing them), and cite only existing paths in the CLAUDE.md row (F7, F8).
**Warning signs:** Crate count derived from `ls crates/` (82 dirs ≠ 83 packages).

### Pitfall 6: Oracle fixtures referenced from `.planning/`
**What goes wrong:** `.planning/` is a planning artifact tree (`auto_prune_state` exists in `.planning/config.json`); a test path into it survives `check_test_fixture_paths.sh` (only `/home`, `/Users` literals are flagged) but ties test evidence to non-source directories, and `cargo package` would exclude it if a crate ever flips `publish`.
**How to avoid:** Copy the 18 fixture files (≈ 7 MB, already committed) into `crates/aprender-forecast/tests/fixtures/` and load via `PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/…")` (precedent `setfit_conformance.rs:137`). Fail loudly (`expect`) when a fixture is missing — fixtures are committed, so absence is a defect, unlike weights.

### Pitfall 7: `*.safetensors` is gitignored globally
**What goes wrong:** Any in-crate fixture weights would be silently untracked (CB-510 class).
**How to avoid:** D-18 forbids committing weights anyway; keep weights under `/models/` (already `/models/`-ignored) and commit only JSON oracles. After adding `include_str!`/`include_bytes!` files run `git check-ignore -v <file>` (CLAUDE.md Publishing Safety) — note both `check_include_files.sh` guards are vacuous on macOS (STATE.md D-ITEM-01), so do the `git check-ignore` by hand.

### Pitfall 8: The embedded-weights leg has no CI home without a workflow edit
**What goes wrong:** D-18's "at least one CI leg exercises the embedded-weights build" needs either a new ci.yml step (human check-in) or a local gate recorded as evidence.
**How to avoid:** Plan a `just chronos-gate` recipe (fetch → embed build → `cargo test -p aprender-mcp-chronos` with `CHRONOS_EMBED_DIR` → assert `0 ignored`) as the phase gate, plus an `autonomous: false` task proposing the ci.yml step text. Also note the CI box is X64 (no NEON): assert parity there, timing only on aarch64.

### Pitfall 9: Debug-profile fits are ~35× slower
**What goes wrong:** CI's nextest leg builds `--lib` tests unoptimised; a Peyton parity fit measured 9.15 s (debug) vs 0.259 s (release) this session (F10). Four Prophet fixtures × restart loop ≈ minutes; the NP 30-lag case and Chronos rollout add more.
**How to avoid:** Wave 0 measures `time cargo test -p aprender-forecast --lib parity::`; if > ~60 s, add `[profile.dev.package.aprender-forecast] opt-level = 3` to the root `Cargo.toml` (precedent `[profile.dev.package.proptest]`, `Cargo.toml:652`). Keep the 20 000-point budget case out of the default test set (it is an 8 s release / minutes debug run).
**Warning signs:** nextest `SLOW [>60s]` lines on parity tests.

### Pitfall 10: `rtk` swallows `println!` and rewrites `$?`
**What goes wrong:** Timing tables and SKIP lines vanish under the Bash-tool hook; a `cargo test | grep` reports grep's status (CLAUDE.md Verification Discipline #1).
**How to avoid:** `rtk proxy cargo test …`, or run `target/debug/deps/<test>-*` directly (spike CONVENTIONS); redirect to a file and read `rc=$?` on the next line — as done for the drift gates this session.

### Pitfall 11: Two crates named `*-lambda` confuse `cargo pmcp deploy`
**What goes wrong:** The justfile records that cargo-pmcp's `find_lambda_package_dir` falls back to "the FIRST `*-lambda` workspace package with a `bootstrap` binary" and shipped the wrong server once (`justfile`, `pmcp-train-deploy` comment `[VERIFIED]`).
**How to avoid:** If the wrapper crates are taken, each deploy recipe must pass `--manifest-path` to the exact package, and every wrapper adds another `[[bin]] bootstrap` to the FALSIFY-MONO-011 count (Pitfall 2). Recommendation: defer the wrappers (Open Question 3).

## Code Examples

### `crates/aprender-forecast/Cargo.toml` (derived from the template, F1/F2/F3)
```toml
[package]
name = "aprender-forecast"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
authors.workspace = true
rust-version.workspace = true
description = "Pure-Rust time-series forecasting: Prophet 1.4.0 and NeuralProphet-lite ports, Chronos-Bolt zero-shot forward"
keywords = ["forecasting", "prophet", "chronos", "time-series", "aprender"]
categories = ["science", "mathematics"]
readme = "README.md"
publish = false            # same reason as aprender-mcp-setfit: deployment unit under the Phase 6 gate, not a published API yet

[lib]
name = "aprender_forecast"
path = "src/lib.rs"

[dependencies]
aprender = { path = "../aprender-core", version = "0.63.0", package = "aprender-core", default-features = false }
trueno = { path = "../aprender-compute", version = "0.63.0", package = "aprender-compute" }
half = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
schemars = "1.0"           # JsonSchema on the shared ForecastArgs so both servers advertise one schema

[lints]
workspace = true
```

### Server crate dependencies (both servers)
```toml
[dependencies]
aprender-forecast = { path = "../aprender-forecast" }
pmcp = { version = "2.19", features = ["streamable-http", "schema-generation"] }
schemars = "1.0"
serde = { workspace = true }
serde_json = { workspace = true }
tokio = { workspace = true }
axum = { workspace = true }
tower = { workspace = true }          # fit server only (pool)

[dev-dependencies]
reqwest = { workspace = true, features = ["json", "rustls-tls"] }
```

### `http_app` with pmcp 2.19.3 (verified API names, F2)
```rust
// Source: sources/007-chronos-mcp-thin-server/src/lib.rs:163-171, reflowed for rustfmt max_width = 100
use axum::response::{Html, IntoResponse};
use axum::routing::get;
use pmcp::server::streamable_http_server::StreamableHttpServerConfig;

pub fn http_app(server: pmcp::Server) -> axum::Router {
    let server = std::sync::Arc::new(tokio::sync::Mutex::new(server));
    let config = pmcp::axum::RouterConfig {
        server_config: StreamableHttpServerConfig::stateless(),
        allowed_origins: Some(pmcp::axum::AllowedOrigins::localhost()),
        ..Default::default()
    };
    let mcp = pmcp::axum::router_with_config(server, config);
    axum::Router::new()
        .route("/", get(|| async { Html(include_str!("../static/index.html")) }))
        .route("/sample/peyton", get(|| async {
            ([("content-type", "text/csv")], include_str!("../fixtures/peyton_manning.csv")).into_response()
        }))
        .nest("/mcp", mcp)
}
```

### Weight fetch recipe (justfile — deployment surface per its own header)
```make
# Fetch amazon/chronos-bolt-tiny at the pinned revision, verify, derive f16. Never run from CI's default gate.
chronos_rev := "a0e552de83495b5c28c14c71c374f3e33280b340"
chronos_dir := "models/chronos-bolt-tiny"

fetch-chronos-tiny:
    #!/usr/bin/env bash
    set -euo pipefail
    uv run --python 3.12 --with huggingface_hub --with safetensors --with numpy python - "{{chronos_rev}}" "{{chronos_dir}}" <<'PY'
    import hashlib, json, os, shutil, sys
    from huggingface_hub import hf_hub_download
    import numpy as np
    from safetensors.numpy import load_file, save_file
    rev, root = sys.argv[1], sys.argv[2]
    pins = {"model.safetensors": "75068728d376d2bec670379eeef4bfb4d24c0cfe24d957451f8d19b447030a32",
            "config.json":       "278f0086733031635fb1c861cb01c1bad6477420c7fcb19381a2993e335785e0"}
    f32 = os.path.join(root, "f32"); f16 = os.path.join(root, "f16"); os.makedirs(f32, exist_ok=True); os.makedirs(f16, exist_ok=True)
    for name, want in pins.items():
        p = hf_hub_download("amazon/chronos-bolt-tiny", name, revision=rev, local_dir=f32)
        got = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if got != want: sys.exit(f"sha256 mismatch for {name}: {got} != {want}")
    t = load_file(os.path.join(f32, "model.safetensors"))
    save_file({k: v.astype(np.float16) for k, v in t.items()}, os.path.join(f16, "model.safetensors"),
              metadata={"format": "pt", "converted": "f32->f16"})
    shutil.copy(os.path.join(f32, "config.json"), os.path.join(f16, "config.json"))
    got16 = hashlib.sha256(open(os.path.join(f16, "model.safetensors"), "rb").read()).hexdigest()
    print(f"f16 sha256 {got16}  (spike 007 recorded f9a033b42bc516e17ae5756317cb946121afdb59c94b4acfcd30fef93317cd4c)")
    PY
    echo "  CHRONOS_MODEL_DIR=$PWD/{{chronos_dir}}/f16   CHRONOS_EMBED_DIR=$PWD/{{chronos_dir}}/f16"
```
(`[ASSUMED]`: that `hf_hub_download(..., local_dir=)` places the file directly under `local_dir` with its repo filename in huggingface_hub 1.30 — the signature was verified, the on-disk layout was not; the sha256 check catches any surprise. `[ASSUMED]`: numpy's `astype(np.float16)` reproduces the spike's f16 sha byte-for-byte — same script, but the safetensors header ordering could differ across library versions; treat the f16 sha as advisory and the f32 shas as the pins.)

### README monorepo link line (every new crate, plus the `aprender-mcp-setfit` fix)
```markdown
Part of the [aprender](https://github.com/paiml/aprender) monorepo — see `crates/aprender-forecast` for the library and the spike evidence in `.claude/skills/spike-findings-aprender/`.
```

### Counted skip for weight-dependent tests — see F6 (`build.rs` + `#[cfg_attr(not(chronos_weights), ignore = "…")]`), probed on this toolchain.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `pmcp = "2.9"` in the SetFit template | `pmcp = "2.19"` (lock 2.19.3) with `pmcp::axum::router_with_config` | `aprender-mcp-setfit-train` already on 2.19 | Same resolved crate; use 2.19 so the axum router API is guaranteed |
| `cargo test --workspace --lib` in CI | `cargo nextest run --profile ci --workspace --lib …` (ci.yml:289) | 2026-06-24 | Per-test processes; `retries = 2`; 1200 s kill; skipped tests are counted in the Summary line |
| `println!("SKIP"); return` (SetFit e2e) | `cfg_attr(…, ignore = "reason")` from a build-script cfg | this research (D-18) | Skips are counted, not silent |
| Median-only Chronos rollout | 9-path re-quantiled rollout (`chronos-forecasting` 2.3.1) | spike 005 | The port already implements the current scheme |
| `trueno::Matrix::matmul` / plain loops | `blis::gemm_blis` + NEON 8×6 kernel | spike 008, in tree | 137 → 21 ms Bolt forward |

**Deprecated/outdated:**
- `contracts/neon-blis-v1.yaml` as a contract *shape* reference — its keys are not the parser's (F4). It remains valid evidence for the kernel.
- CLAUDE.md's "`unsafe_code = "forbid"`" — the workspace lint is `deny` (`Cargo.toml:382`).
- CONTEXT's "one crate has no README" — two do (F7).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `[profile.dev.package.aprender-forecast] opt-level = 3` recovers most of the 35× debug slowdown | F10 / Pitfall 9 | Parity suite stays minutes-long in CI; fallback is `--release` for a scoped `--test` target (needs the ci.yml line) |
| A2 | `hf_hub_download(local_dir=…)` writes `<local_dir>/<filename>` directly | Code Examples | Recipe path wrong; sha check fails loudly, fix the path |
| A3 | numpy `astype(float16)` + `safetensors.numpy.save_file` reproduces the spike's f16 sha | Code Examples | Only the f16 sha is advisory; f32 pins hold |
| A4 | Cargo ≥ 1.80 for `cargo::rustc-check-cfg`; MSRV 1.91 covers it | F1 | If a consumer built on < 1.80, build.rs output would be a warning not an error; irrelevant under the 1.93 pin |
| A5 | The 4 pre-existing FALSIFY-MONO-011 violations are unknown to the human (not deliberate debt) | Open Question 1 | If deliberate, the decision may already exist somewhere; still needs surfacing |
| A6 | Spike-measured latencies (release, Apple M4 Pro) transfer to the release build of the ported crates | Validation Architecture | SC1/SC4 timing thresholds may need re-measurement after the pedantic-clippy edits; re-run `--bench`/`--coldstart` |
| A7 | pmcp 2.19.3's `router_with_config` behaves identically when nested under `/mcp` via `axum::Router::nest` in a workspace crate as it did in the spike | Pattern 2 | Same crate version, same code; low risk |

## Open Questions (RESOLVED)

_All five were resolved by the 2026-09-05 plan set; the resolving plan is marked inline on each item._

1. **FALSIFY-MONO-011 `[[bin]]` allowlist (human decision, blocks CI green).**
   - What we know: shrink-only at 27; 4 SetFit crates already violate (measured); Phase 6 adds `aprender-mcp-forecast` and `aprender-mcp-chronos` (+2), or +4 with lambda wrappers. Policy text: "migrate the capability to an `apr` subcommand instead of granting a new exemption." An `apr forecast` subcommand is explicitly deferred (CONTEXT).
   - What's unclear: whether the human wants (a) a documented exemption class "thin MCP deployment units, `publish = false`" with the baseline raised to cover the six (or eight) crates, (b) the SetFit precedent fixed separately first, or (c) binaries moved under `apr` (contradicts D-06 and the deferral).
   - Recommendation: first plan task, `autonomous: false`, presenting (a) with the exact diff (add the six names in a new comment block; `ALLOWLIST_BASELINE = 33`; keep the stale-entry check). Nothing else in the phase can make `monorepo_invariants` green. **RESOLVED: 06-02** (Task 1 is the blocking human decision with four options — deployment-unit-class recommended over the single-list-33 above; Task 2 applies it with a two-sided ratchet control).

2. **Fixture location** — copy into `crates/aprender-forecast/tests/fixtures/` (≈ 7 MB duplicated in git) vs reference `.planning/spikes/*/fixtures/`. Recommendation: copy (Pitfall 6); D-08 leaves this to the planner. **RESOLVED: 06-01** (Task 2 copies the 17 files into `crates/aprender-forecast/tests/fixtures/`, byte-verified with `cmp` against the spike originals).

3. **Lambda wrapper crates now or later** (Claude's discretion). Recommendation: defer — each adds a `bootstrap` `[[bin]]` (Open Question 1), cargo-pmcp's `*-lambda` fallback is already ambiguous (Pitfall 11), and deployment is out of scope. Ship `http_app`/`stateless()` so the wrapper is a 60-line copy later. **RESOLVED: 06-01** (deferred as recommended, recorded in the plan; 06-09 Task 3 files it as D-ITEM-06-01).

4. **Fix `SmoothL1Loss` in core now?** Recommendation: file as a core ticket with the F3 quote; the phase does not depend on it (D-10), and touching `aprender-core` pulls the 14 285-test lib into every PR cycle. **RESOLVED: 06-09** (Task 3 files the core ticket — D-ITEM-06-02; 06-04 asserts `weighted_huber` graph connectivity instead of fixing core).

5. **Which parity numbers are CI assertions vs. aarch64-only gates** — timing (SC1 "< 2 s", SC4 "< 100 ms", "< 150 ms cold start", "< 30 MB binary") cannot be asserted on the X64 CI box or in debug builds. Recommendation: assert parity + refusals in CI; timings and binary size in `just forecast-bench` / `just chronos-coldstart` / `just chronos-gate` recorded in the phase VALIDATION evidence on the M4 host. **RESOLVED: 06-08** (host-gated `just` recipes + `06-EVIDENCE.md` measured on aarch64 release; CI asserts parity and refusals via 06-03/04/05/06/07; the one CI-workflow edit is 06-08's blocking human decision, applied by 06-09).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| rustc / cargo (pinned) | everything | ✓ | 1.93.0 / 1.93.0 (pin `rust-toolchain.toml`) | — |
| cargo-nextest | reproduce CI lib leg | ✓ | 0.9.102 | `cargo test --lib` |
| `pv` (aprender-contracts-cli) | D-15 | ✓ via `cargo run -p aprender-contracts-cli --bin pv` (debug binary at `target/debug/pv`, 0.63.0) | 0.63.0 | — (never bash) |
| `just` | fetch/gate recipes | ✓ | 1.46.0 | plain bash script under `scripts/` |
| `uv` + Python | weight fetch, f16 conversion, oracle regeneration | ✓ | uv 0.9.5, Python 3.13.7 (`--python 3.12` used by the spikes) | — |
| huggingface_hub | fetch recipe | ✓ via `uv run --with` | 1.30.0 | — |
| Chronos-Bolt-tiny weights | Bolt parity tests, embedded build | ✓ locally (HF cache snapshot `a0e552de…`; f16 copies in `.planning/spikes/007/models/tiny-f16/`) | f32 33.0 MB / f16 16.5 MB | counted skip (F6) when absent |
| aarch64 host (NEON kernel) for timing gates | SC1/SC4 timings | ✓ (this box, `arm64`) | Apple M4 Pro | none — CI runner is X64; timings are host-gated |
| pmat | code search policy | ✓ | 3.15.0 (CLAUDE.md says 3.30.0 — drift) | grep fallback used this session per the orchestrator's instruction |
| bashrs | lint any new `scripts/*.sh` | ✓ | 6.66.3 | — |
| cargo-deny | license/ban policy | ✓ | 0.18.3 | — |
| Disk | builds | ⚠ 57 GiB free of 926 GiB (94 % used); `.cargo/config.toml` sets `incremental = false` | — | STATE.md records three ENOSPC halts; keep `CARGO_INCREMENTAL=0` (already in config) |
| `.planning/graphs/graph.json` | graph context | ✗ | — | skipped (Step 1.3) |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** knowledge graph (absent; skipped).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust libtest via `cargo test` 1.93.0; CI runs `cargo nextest run --profile ci` 0.9.102 (`.config/nextest.toml`: retries 2, fail-fast, slow-timeout 60 s × 20) |
| Config file | `.config/nextest.toml` (existing); per-package profile override to add in root `Cargo.toml` if Wave 0 timing demands (Pitfall 9) |
| Quick run command | `cargo test -p aprender-forecast --lib prophet::parity::peyton` (one fixture, one ladder) |
| Full suite command | `cargo nextest run --workspace --lib --exclude aprender-gpu --exclude aprender-cuda-edge --exclude aprender-compute` (CI's exact leg) + `cargo test -p aprender-core --test monorepo_invariants --test readme_contract` (the two CI-wired drift gates) |

### Phase Requirements → Test Map
| Req | Behavior | Test Type | Automated Command | File Exists? |
|-----|----------|-----------|-------------------|--------------|
| D-04 / SC2 | Prophet data prep 0.0 diff; objective at MAP ≤ 1e-9; predict ≤ 1e-10; fit ≤ Py + 0.5; forecast within band; components rebuild yhat; band widths ±2 % — on peyton, air, retail, wp_log_R (+ holidays, logistic, multiplicative) | unit (`--lib`), release-profile in CI via package override | `cargo test -p aprender-forecast --lib prophet::parity` | ❌ Wave 0 (`src/parity/prophet.rs`; fixtures copied) |
| D-10 / SC3 | NP data prep matches `np_oracle_peyton.json`; lag-free 365-day MAE ≤ 0.47; `n_lags = 30` beats naive one-step; Huber gradient connected (`pred.grad().is_some()`) | unit | `cargo test -p aprender-forecast --lib np::parity` | ❌ Wave 0 |
| D-13 / SC4 | Bolt ladder vs `chronos_bolt_tiny_fixture.json` (1e-6 abs f32), `chronos_probes.json` edge cases, 365-step rollout 1.5e-5; f16 within 2 % of std through the server | unit, **cfg_attr-ignored without weights** | `CHRONOS_MODEL_DIR=models/chronos-bolt-tiny/f32 cargo test -p aprender-forecast --lib bolt::parity` | ❌ Wave 0 + `just fetch-chronos-tiny` |
| D-11 / SC1 | Every refusal is `pmcp::Error::validation` with a fix-naming message (unknown field, <10 / >20 000 pts, unsorted/duplicate/impossible dates, constant y, horizon 0 / >3650, unknown freq, logistic without valid cap, interval_width ∉ (0,1)); happy paths prophet + NP over in-process streamable-HTTP | unit `#[cfg(test)] mod e2e` (in-process axum + reqwest) | `cargo test -p aprender-mcp-forecast --lib e2e` | ❌ Wave 0 (port `sources/004/tests/e2e.rs` into `src/`) |
| D-11 / SC4 | Chronos refusals (<4 pts, impossible date, horizon 0, all-null y, horizon > 64 without flag) + flag → `warning` + `forwards: 46`; parity through the server | unit, cfg_attr-ignored without weights | `CHRONOS_MODEL_DIR=… cargo test -p aprender-mcp-chronos --lib e2e` | ❌ Wave 0 (port `sources/007/tests/e2e.rs`) |
| D-12 / SC5 | 8 concurrent requests bit-identical to sequential (JSON signature of ds/yhat/bands/trend/components); wall time < ½ sequential with pool K = 8 | unit (equality) + host-gated timing | `cargo test -p aprender-mcp-forecast --lib pool_equality` (equality asserted; speed-up asserted only when `cfg!(target_arch = "aarch64")` && release, else logged) | ❌ Wave 0 (port `sources/010/src/main.rs` as a test) |
| D-18 | Missing weights → counted skip with reason; armed → tests run | meta-test of the mechanism | `cargo test -p aprender-mcp-chronos --lib 2>&1 \| grep -c "ignored,"` (expect ≥ 1 unarmed, 0 armed) | ❌ Wave 0 (`build.rs` cfg) |
| D-15 | Every new contract validates; counts non-zero | shell gate | `cargo run -p aprender-contracts-cli --bin pv -- validate contracts/<f>.yaml && … pv status …` ; `make contract-validate` | ❌ Wave 0 (3 YAMLs + Makefile lines) |
| SC5 lint/fmt | clean on new crates | shell | `cargo clippy -p aprender-forecast -p aprender-mcp-forecast -p aprender-mcp-chronos --all-targets -- -D warnings && cargo fmt --all -- --check` | ✅ commands exist |
| F7 gates | README counts, per-crate READMEs, CLAUDE.md paths, bin allowlist | integration (CI-wired) | `cargo test -p aprender-core --test monorepo_invariants --test readme_contract` | ✅ exists — **currently 5 failures** |
| SC1/SC4 timing, SC4 binary size, cold start | < 2 s Peyton round trip; < 100 ms 2 048-pt forward; < 30 MB release binary; < 150 ms cold start over stdio | **manual / host-gated** (aarch64 release) | `just forecast-bench`, `just chronos-coldstart 3`, `ls -l target/release/aprender-mcp-chronos` after `CHRONOS_EMBED_DIR=… cargo build --release -p aprender-mcp-chronos` | ❌ Wave 0 recipes |
| Demo page | page performs initialize → tools/list → tools/call and charts the result | manual (browser) | `cargo run -p aprender-mcp-forecast -- --http 8787` then open `http://127.0.0.1:8787/` | — (spike-proven page copied verbatim) |

### Sampling Rate
- **Per task commit:** `cargo clippy -p <crate> --all-targets -- -D warnings && cargo fmt --all -- --check && cargo test -p <crate> --lib`
- **Per wave merge:** CI's nextest lib leg (F5 verbatim) + `cargo test -p aprender-core --test monorepo_invariants --test readme_contract` + `make contract-validate`
- **Phase gate:** all of the above green, plus `just chronos-gate` (embedded build, `0 ignored`), `just forecast-bench`, `just chronos-coldstart` on the aarch64 host, results pasted into the VALIDATION evidence; the human decision of Open Question 1 recorded before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `crates/aprender-forecast/tests/fixtures/` — copy the 18 committed oracle files (001, 002, 003, 005, 007 spikes; ≈ 7 MB) or decide to reference `.planning/spikes/` (Open Question 2)
- [ ] `crates/aprender-forecast/build.rs` + `crates/aprender-mcp-chronos/build.rs` — `cfg(chronos_weights)` emission (F6) and, for the server, `CHRONOS_EMBED_DIR` staging (D-13)
- [ ] `justfile` recipes: `fetch-chronos-tiny`, `chronos-gate`, `forecast-bench`, `chronos-coldstart`
- [ ] `models/chronos-bolt-tiny/{f32,f16}/` populated locally (gitignored) — run the recipe once; record shas in the crate README
- [ ] Root `Cargo.toml`: three member lines; `[profile.dev.package.aprender-forecast] opt-level = 3` **after** measuring debug parity time (Pitfall 9)
- [ ] `contracts/forecast-tool-boundary-v1.yaml`, `contracts/prophet-parity-v1.yaml`, `contracts/chronos-bolt-parity-v1.yaml` skeletons (F4) + Makefile `$(CONTRACTS)` / `PHASE6_CONTRACTS` lines
- [ ] Human checkpoint task for FALSIFY-MONO-011 (Open Question 1)
- [ ] README fixes: `README.md:43-44` counts; `crates/aprender-mcp-setfit/README.md` link; READMEs for `aprender-mcp-setfit-lambda` and `aprender-contrastive-data`; three new crate READMEs with `paiml/aprender`
- [ ] Framework install: none — cargo/nextest/just/uv present

## Security Domain

`security_enforcement: true`, ASVS level 1 (`.planning/config.json`). The servers accept untrusted JSON over stdio and HTTP and run CPU-bound fits per request.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no (this phase) | pmcp.run API Gateway fronts the Lambda; local HTTP binds `127.0.0.1` and allows only `AllowedOrigins::localhost()` — document that `--http` is a dev/loopback surface |
| V3 Session Management | no | `StreamableHttpServerConfig::stateless()` — no sessions by construction (`streamable_http_server.rs:451`) |
| V4 Access Control | no | single public tool; no per-user data |
| V5 Input Validation | **yes** | `serde` `deny_unknown_fields` + `schemars` schema (`additionalProperties: false`); D-11 bounds (`MIN/MAX_POINTS`, `MAX_HORIZON`, date validity, `freq` allowlist, `interval_width`, `cap`); refusals are `pmcp::Error::validation`, never defaults; body cap from pmcp stateless config (4 MB per the spike reference) |
| V6 Cryptography | no runtime crypto | sha256 of weights at fetch time only (Python `hashlib`); no hand-rolled crypto |
| V10 Malicious Code / supply chain | yes | no new registry crates (F9); weights pinned by revision + sha256 (F6); `cargo-deny` licenses; `*.safetensors` never committed |
| V12 Files & Resources | yes | `CHRONOS_MODEL_DIR` read once at startup, never per request; embedded bytes preferred; no path taken from a request |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| CPU exhaustion via a 20 000-point fit or `n_lags` AR-Net | Denial of Service | `MAX_POINTS 20_000`, per-round iteration cap 2 000, 15 s wall budget → `budget_hit` (D-09); `spawn_blocking` keeps the protocol loop responsive; router pool bounds concurrency to K blocking fits |
| Memory blow-up from huge `horizon` or `quantiles` | DoS | `MAX_HORIZON 3_650` / `1_024`; Chronos context truncated to 2 048 (`context_used` reported) |
| NaN/Inf in `y` driving the objective non-finite | Tampering / DoS | Non-finite guard returns `1e300` with zero gradient (D-09); Chronos `y` nullable only there and all-null refused; serde rejects non-finite JSON numbers natively |
| DNS rebinding / cross-origin browser calls to the local HTTP server | Spoofing | pmcp `AllowedOrigins::localhost()` + `SecurityHeadersLayer` defaults (`axum_router.rs:47-55`) |
| Unknown argument silently ignored (client believes a knob took effect) | Tampering | `deny_unknown_fields` end-to-end, with an e2e case per D-11 (template `e2e_stdio.rs:188-207`) |
| Poisoned weights on the request path | Tampering | Never download at runtime (spike rule); pinned revision + sha256 at fetch; embedded bytes compiled in |
| Non-determinism masking a race (autograd tape shared across threads) | Repudiation | Per-request `seed` (default 42); spike-010 equality test under 8–16 concurrent requests; fits never interleave on one thread (`spawn_blocking`) |

## Sources

### Primary (HIGH confidence — read in-tree this session)
- Root `Cargo.toml` (members, exclude, `[workspace.package]`, `[workspace.dependencies]`, `[workspace.lints.*]`, profiles), `.clippy.toml`, `rustfmt.toml`, `rust-toolchain.toml`, `.gitignore`, `.config/nextest.toml`, `justfile`, `Makefile:311-345, 1815-1960`, `deny.toml`, `README.md:38-48`, `CLAUDE.md:209-226`
- `crates/aprender-mcp-setfit/{Cargo.toml, README.md, src/lib.rs, src/main.rs, tests/e2e_stdio.rs}`; `crates/aprender-mcp-setfit-lambda/{Cargo.toml, build.rs, src/lib.rs, src/main.rs, tests/embed.rs}`; `crates/aprender-mcp-setfit-train/Cargo.toml:41-42`
- `crates/aprender-core/src/optim/{lbfgs.rs, mod.rs}`, `autograd/{mod.rs, tensor.rs, ops/mod.rs, ops/activation.rs}`, `nn/{linear.rs, module.rs, mod.rs, loss.rs, optim/mod.rs, optim/rm_sprop.rs}`, `primitives/vector.rs`, `lib.rs`, `Cargo.toml`; `crates/aprender-compute/src/{lib.rs, blis/compute.rs}`, `Cargo.toml`
- `crates/aprender-contracts/src/schema/{types.rs, validator.rs, parser.rs, kind.rs}`; `contracts/setfit-apr-v1.yaml`, `contracts/neon-blis-v1.yaml`, `contracts/aprender/binding.yaml`; `target/debug/pv` (0.63.0) `validate`/`status` runs
- `crates/aprender-core/tests/{readme_contract.rs, monorepo_invariants.rs}` — read fully and **executed** (`rtk proxy cargo test …`, log in scratchpad)
- `.github/workflows/ci.yml:95-300, 315-345, 390-480` (nextest leg `:289`, `--test` line `:471`); `scripts/{check_test_fixture_paths.sh, check_build_rs_paths.sh, setfit_fixtures/fetch_full_weights.py, setfit_fixtures/README.md}`
- pmcp 2.19.3 registry source: `src/server/{axum_router.rs, streamable_http_server.rs, mod.rs, tower_layers/dns_rebinding.rs}`, `src/types/capabilities.rs`
- `Cargo.lock` (resolved versions); `cargo metadata --no-deps` (83 packages); `find contracts -name '*.yaml' | wc -l` (1786)
- `.claude/skills/spike-findings-aprender/` — `SKILL.md`, all 7 `references/*.md`, `sources/00{4,6,7,10}/*` (Cargo.toml, src, tests, build.rs, BENCH.md, RUN-OUTPUT.md); `.planning/spikes/{CONVENTIONS,MANIFEST,WRAP-UP-SUMMARY}.md`; fixture inventory and `static/index.html` locations under `.planning/spikes/`
- HF cache `~/.cache/huggingface/hub/models--amazon--chronos-bolt-tiny/` (revision, sha256s); `uv run --with huggingface_hub` signature probe
- Scratchpad probes: `skipprobe/` (cfg_attr-ignore under libtest and nextest), `spike001-debug.md` (debug-profile fit timing)

### Secondary (MEDIUM confidence — spike measurements cited, not re-run)
- Latency, parity and size numbers from the spike references and `sources/004/BENCH.md`, `sources/001/RUN-OUTPUT.md`

### Tertiary (LOW confidence — web, per the classify-confidence seam)
- [AWS Lambda limits guide](https://middleware.io/blog/aws-lambda-limits/), [Dashbird: exploring Lambda limitations](https://dashbird.io/blog/exploring-lambda-limitations/), [AWS re:Post on package quotas](https://repost.aws/questions/QUIbLMSIa5Q3O9dQLjLoca1g/increase-of-lambda-deployment-package-quota-max) — 50 MB zip / 250 MB unzipped / 10 GB image
- [rustc book: check-cfg Cargo specifics](https://doc.rust-lang.org/rustc/check-cfg/cargo-specifics.html), [Cargo book: build scripts](https://docs.adacore.com/live/wave/rust/html/rust_ug/_static/cargo/reference/build-scripts.html) — `cargo::rustc-check-cfg` / `rustc-cfg`
- [amazon/chronos-bolt-tiny model card](https://huggingface.co/amazon/chronos-bolt-tiny) — apache-2.0, 8.65M params, files

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every dependency resolved in `Cargo.lock`; every API name verified in source this session
- Architecture: HIGH — template crates read in full; pmcp API confirmed in the registry source; spike blueprints are VALIDATED
- Pitfalls: HIGH for the gate failures (executed), the contract-template hollowness (`pv status` run), the skip mechanism (probed) and the debug slowdown (timed); MEDIUM for clippy volume (not yet run on the ported code)

**Research date:** 2026-09-05
**Valid until:** 2026-10-05 for the repo-state findings (re-run the two drift gates and re-derive counts before planning if later); the spike findings are stable until the skill's `processed_spikes` list changes
