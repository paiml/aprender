# Phase 6: Native Time-Series Forecasting Stack - Pattern Map

**Mapped:** 2026-09-05
**Files analyzed:** 34 (three new crates + workspace/contract/doc edits)
**Analogs found:** 31 / 34 (3 files have no in-tree analog: the router pool, the spike-006 MASE example, the demo page)

Every analog path below passed `git ls-files` (tracked source). The spike sources under
`.claude/skills/spike-findings-aprender/sources/` are tracked too (90 files) and are the
SOURCE-OF-TRUTH for *what* each new file contains (D-08: port as-is); the workspace crates are
the analog for *how* it must be shaped (manifest fields, lint attributes, module layout, test
placement, README link). Line numbers are from the files as read this session.

**The single most important fact for the planner:** the spike sources already have **zero
`.unwrap()` calls in every `src/*.rs`** (measured: 004 `lib.rs/main.rs/np.rs/prophet.rs` = 0,
007 `bolt.rs/dates.rs/lib.rs/main.rs/safetensors.rs` = 0). Only the test/harness files carry
them: `sources/010/src/main.rs` 2, `sources/004/tests/e2e.rs` 4, `sources/007/tests/e2e.rs` 1.
So the `disallowed_methods` clean-up is small; the pedantic-clippy and rustfmt (`max_width =
100` — the spikes are written in long one-liners) reflow is the real volume (RESEARCH Pitfall 1).

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/aprender-forecast/Cargo.toml` | config (lib manifest) | — | `crates/aprender-mcp-setfit/Cargo.toml` (drop `[[bin]]`) | role-match |
| `crates/aprender-forecast/src/lib.rs` | library root (module exports + shared types) | transform | `sources/004/src/lib.rs` 1-128 (types, `parse_date`, `future_days`) + `crates/aprender-mcp-setfit/src/lib.rs` 29-32 (file-scope allow) | exact (spike) |
| `crates/aprender-forecast/src/prophet.rs` | model (Prophet 1.4.0 port) | batch/transform | `sources/004/src/prophet.rs` (520 lines, verbatim) | exact (spike) |
| `crates/aprender-forecast/src/np.rs` | model (NeuralProphet-lite) | batch/transform | `sources/004/src/np.rs` (384 lines, verbatim) | exact (spike) |
| `crates/aprender-forecast/src/fit.rs` | service (L-BFGS wrapper) | batch | `sources/004/src/lib.rs` 130-172 (`Cached`, `FitInfo`, `fit_prophet`) — supersedes `sources/006/src/fit.rs` | exact (spike) |
| `crates/aprender-forecast/src/bolt.rs` | model (Chronos-Bolt forward) | batch/transform | `sources/007/src/bolt.rs` (323 lines) | exact (spike) |
| `crates/aprender-forecast/src/safetensors.rs` | utility (decoder) | file-I/O | `sources/007/src/safetensors.rs` (41 lines) | exact (spike) |
| `crates/aprender-forecast/src/dates.rs` | utility (civil dates) | transform | `sources/007/src/dates.rs` (49 lines) | exact (spike) |
| `crates/aprender-forecast/src/types.rs` | model (shared `ForecastArgs`/`ForecastResponse`/`HolidayArg`) | request-response | `sources/004/src/lib.rs` 30-103 + `crates/aprender-mcp-setfit/src/lib.rs` 62-92 | exact |
| `crates/aprender-forecast/src/parity/*.rs` (`#[cfg(test)]`) | test (parity ladders) | batch | `crates/aprender-core/tests/setfit_conformance.rs:137` (fixture path) + `crates/aprender-mcp-setfit/src/lib.rs` 226-321 (in-lib test module) | role-match |
| `crates/aprender-forecast/build.rs` | config (cfg emission) | — | `crates/aprender-train/build.rs:75` (`rustc-check-cfg`) + RESEARCH F6 snippet | role-match |
| `crates/aprender-forecast/tests/fixtures/*.json` | test fixtures | file-I/O | `crates/aprender-core/tests/fixtures/setfit/` (layout) | role-match |
| `crates/aprender-forecast/examples/mase_rolling_origin.rs` | example (bench harness) | batch | `sources/006/src/main.rs` (190 lines) | exact (spike); no workspace example analog for forecasting |
| `crates/aprender-forecast/README.md` | docs | — | `crates/aprender-mcp/README.md:3` (monorepo link line) | exact |
| `crates/aprender-mcp-forecast/Cargo.toml` | config (server manifest) | — | `crates/aprender-mcp-setfit/Cargo.toml` (whole file) | exact |
| `crates/aprender-mcp-forecast/src/lib.rs` | service (pmcp server + http_app + pool) | request-response | `crates/aprender-mcp-setfit/src/lib.rs` 189-224 (`build_server`) + `sources/004/src/lib.rs` 278-307 + `sources/010/src/main.rs` 55-68 (pool) | exact |
| `crates/aprender-mcp-forecast/src/main.rs` | controller (argv → stdio/http) | request-response | `crates/aprender-mcp-setfit/src/main.rs` (whole) + `sources/004/src/main.rs` 41-62 | exact |
| `crates/aprender-mcp-forecast/src/lib.rs` `#[cfg(test)] mod e2e` | test (in-process HTTP) | request-response | `sources/004/tests/e2e.rs` (100 lines) + `sources/010/src/main.rs` 70-141 (equality) | exact (spike) — must move from `tests/` into `src/` (F5) |
| `crates/aprender-mcp-forecast/tests/e2e_stdio.rs` | test (spawned binary) | request-response | `crates/aprender-mcp-setfit/tests/e2e_stdio.rs` (217 lines) | exact |
| `crates/aprender-mcp-forecast/static/index.html` | asset | — | `.planning/spikes/004-forecast-mcp-thin-server/static/index.html` (tracked, 10.3 KB — NOT the 65-line copy in sources/) | exact (spike) |
| `crates/aprender-mcp-forecast/fixtures/{peyton_manning,air_passengers}.csv` | asset | — | `.planning/spikes/004-forecast-mcp-thin-server/fixtures/*.csv` (tracked) | exact |
| `crates/aprender-mcp-forecast/README.md` | docs | — | `crates/aprender-mcp-setfit/README.md` 1-10 + `crates/aprender-mcp/README.md:3` | role-match |
| `crates/aprender-mcp-chronos/Cargo.toml` | config (server manifest, `build = "build.rs"`) | — | `crates/aprender-mcp-setfit/Cargo.toml` + `sources/007/Cargo.toml:6` | exact |
| `crates/aprender-mcp-chronos/build.rs` | config (embed staging + cfg) | file-I/O | `crates/aprender-mcp-setfit-lambda/build.rs` (33 lines) + `sources/007/build.rs` (23 lines) | exact |
| `crates/aprender-mcp-chronos/src/lib.rs` | service (embedded model + server) | request-response | `crates/aprender-mcp-setfit-lambda/src/lib.rs` 18-55 (`EMBEDDED_*`, `resolve_model`) + `sources/007/src/lib.rs` | exact |
| `crates/aprender-mcp-chronos/src/main.rs` | controller | request-response | `crates/aprender-mcp-setfit/src/main.rs` + `sources/007/src/main.rs` 77-103 | exact |
| `crates/aprender-mcp-chronos/src/lib.rs` `#[cfg(test)] mod e2e` | test (weight-gated) | request-response | `sources/007/tests/e2e.rs` + `crates/aprender-mcp-setfit-lambda/tests/embed.rs` (what NOT to copy: println-SKIP) | exact (spike) |
| `crates/aprender-mcp-chronos/static/index.html`, `fixtures/*.csv` | asset | — | `.planning/spikes/007-chronos-mcp-thin-server/static/index.html` (tracked) | exact |
| `crates/aprender-mcp-chronos/README.md` | docs | — | as above | role-match |
| Root `Cargo.toml` (`[workspace] members`, `[profile.dev.package.*]`) | config | — | `Cargo.toml:46-53` (MCP member block), `Cargo.toml:649-653` (per-package profile override) | exact |
| `contracts/forecast-tool-boundary-v1.yaml`, `contracts/prophet-parity-v1.yaml`, `contracts/chronos-bolt-parity-v1.yaml` | contract | — | `contracts/setfit-apr-v1.yaml` (keys at 1, 2, 80, 1010, 1103, 1243, 1276) | exact |
| `Makefile` (`CONTRACTS`, `PHASE6_CONTRACTS`, `contract-audit-phase6`) | config | — | `Makefile:1815-1866` + `PHASE4_CONTRACTS` at `:1905` | exact |
| `justfile` (`fetch-chronos-tiny`, `chronos-gate`, `forecast-bench`, `chronos-coldstart`) | config (deployment recipes) | — | `justfile:1-40` header + `build-apr-arm64` recipe `:40-46` | role-match |
| `README.md:43-44`, `CLAUDE.md:216-226`, per-crate READMEs | docs | — | `crates/aprender-core/tests/readme_contract.rs:130, 171, 245, 275, 456-466` (what the gates check) | exact |

## Pattern Assignments

### `crates/aprender-forecast/Cargo.toml` (config, library manifest)

**Analog:** `crates/aprender-mcp-setfit/Cargo.toml` lines 1-19, 44-45 — with `[[bin]]` removed and deps swapped.

**Manifest header pattern** (`crates/aprender-mcp-setfit/Cargo.toml:1-19`):
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
```

**Lints inheritance** (`crates/aprender-mcp-setfit/Cargo.toml:44-45`) — every new crate ends with this:
```toml
[lints]
workspace = true
```

**Dependency lines** — take the in-workspace path form from the SetFit manifest (`:29`) and the
feature set from the spike manifests (`sources/004/Cargo.toml:18`, `sources/007/Cargo.toml:19-20`):
```toml
[dependencies]
aprender = { path = "../aprender-core", version = "0.63.0", package = "aprender-core", default-features = false }
trueno = { path = "../aprender-compute", version = "0.63.0", package = "aprender-compute" }   # NO `parallel` feature (D-14)
half = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
schemars = "1.0"
```
Do NOT copy `crates/aprender-data/Cargo.toml` — it predates workspace inheritance (hardcoded
`edition`, `license`, `repository = ".../alimentar"`, `rust-version = "1.75"`) and is the
anti-pattern here.

---

### `crates/aprender-forecast/src/lib.rs` + `src/types.rs` (library root; shared request/response types)

**Analog A (spike, what goes in):** `sources/004-forecast-mcp-thin-server/src/lib.rs`.

**Module layout + file-scope allow** (`sources/004/src/lib.rs:1-16`; the allow comment text
comes from `crates/aprender-mcp-setfit/src/lib.rs:29-32`):
```rust
// schemars' JsonSchema derive and serde_json::json! both expand to .unwrap()
// internally, and the derive's generated impl lands at file scope where a
// struct-level allow cannot reach it. Same precedent as aprender-mcp's tools.
#![allow(clippy::disallowed_methods)]
pub mod np;
pub mod prophet;

use aprender::optim::{ConvergenceStatus, LbfgsF64};
use aprender::primitives::Vector;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::time::Instant;
```
Put the `#![allow(clippy::disallowed_methods)]` ONLY in files that derive `JsonSchema` or use
`json!` (the types file and the server `lib.rs`), never crate-wide (RESEARCH Pitfall 1).

**Typed-args pattern** (`sources/004/src/lib.rs:45-81` — the doc comments become the tool schema descriptions):
```rust
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ForecastArgs {
    /// Timestamps, YYYY-MM-DD (a time part is ignored), ascending, unique.
    pub ds: Vec<String>,
    /// Observed values, same length as `ds`.
    pub y: Vec<f64>,
    /// Number of future periods to forecast (1 … 3650).
    pub horizon: usize,
    /// Period of the future steps: "D" (default), "W", or "MS" (month start).
    #[serde(default)]
    pub freq: Option<String>,
    /// "prophet" (default) or "neuralprophet".
    #[serde(default)]
    pub model: Option<String>,
    // … growth, cap, seasonality_mode, interval_width, holidays, n_lags, seed — all `#[serde(default)] Option<_>`
}
```
The Chronos variant differs in exactly three fields (`sources/007/src/lib.rs:58-74`): `y: Vec<Option<f64>>`
(nullable, D-11), `allow_long_horizon: bool` (`#[serde(default)]`), and no model/growth/holiday knobs.
Both servers must still advertise `additionalProperties: false` — the SetFit unit test that pins
this is the template (`crates/aprender-mcp-setfit/src/lib.rs:305-320`).

**Response + error pattern** (`sources/004/src/lib.rs:83-103`):
```rust
#[derive(Debug, Serialize)]
pub struct ForecastResponse {
    pub model: String,
    pub freq: String,
    pub n_history: usize,
    pub fit_seconds: f64,
    pub predict_seconds: f64,
    pub ds: Vec<String>,
    pub yhat: Vec<f64>,
    pub yhat_lower: Vec<f64>,
    pub yhat_upper: Vec<f64>,
    pub trend: Vec<f64>,
    pub components: serde_json::Map<String, serde_json::Value>,
    pub diagnostics: serde_json::Value,
}

#[derive(Debug)]
pub enum ForecastError { Validation(String), Internal(String) }
```
Chronos response adds `context_used`, `quantiles`, `#[serde(skip_serializing_if = "Option::is_none")] warning: Option<String>`
(`sources/007/src/lib.rs:76-91`).

**Validation door pattern** (`sources/004/src/lib.rs:174-190` — "the transport re-checks nothing; this is THE door"):
```rust
pub fn forecast(args: &ForecastArgs) -> Result<ForecastResponse, ForecastError> {
    if args.ds.len() != args.y.len() { return Err(ForecastError::Validation(format!("ds has {} entries but y has {}", args.ds.len(), args.y.len()))); }
    if args.ds.len() < MIN_POINTS { return Err(ForecastError::Validation(format!("need at least {MIN_POINTS} points, got {}", args.ds.len()))); }
    if args.ds.len() > MAX_POINTS { return Err(ForecastError::Validation(format!("{} points exceeds max_points {MAX_POINTS}", args.ds.len()))); }
    if args.horizon == 0 || args.horizon > MAX_HORIZON { return Err(ForecastError::Validation(format!("horizon must be 1..={MAX_HORIZON}, got {}", args.horizon))); }
    if args.y.iter().any(|v| !v.is_finite()) { return Err(ForecastError::Validation("y contains a non-finite value".into())); }
    let ds: Vec<i64> = args.ds.iter().map(|s| parse_date(s)).collect::<Result<_, _>>()?;
    if !ds.windows(2).all(|w| w[0] < w[1]) { return Err(ForecastError::Validation("ds must be strictly ascending with no duplicates".into())); }
    let freq = args.freq.clone().unwrap_or_else(|| "D".into());
    let fut = future_days(ds[ds.len() - 1], args.horizon, &freq)?;
    // …
    if !(0.0 < interval_width && interval_width < 1.0) { return Err(ForecastError::Validation("interval_width must be in (0, 1)".into())); }
    // …
    if y_min == y_max { return Err(ForecastError::Validation("y is constant; nothing to fit (Prophet special-cases this too)".into())); }
```
Every branch here maps to one D-11 refusal and one e2e case. Reflow for rustfmt (100 cols);
do not change the messages — the e2e tests string-match them (`"at least"`, `"calendar"`).

**Constants** (`sources/004/src/lib.rs:26-28`, `sources/007/src/lib.rs:25-27`): `MAX_POINTS 20_000 / MAX_HORIZON 3_650 / MIN_POINTS 10`
(fit) and `MAX_POINTS 20_000 / MIN_POINTS 4 / MAX_HORIZON 1_024` (Chronos). Per D-15 these
values are also written into the contract YAML; the refusal message should cite the contract
the way SetFit does (`crates/aprender-mcp-setfit/src/lib.rs:157-161`):
```rust
return Err(pmcp::Error::validation(format!(
    "batch of {} texts exceeds max_batch_texts {MAX_BATCH_TEXTS} \
     (contracts/setfit-apr-v1.yaml item 11); split the batch",
    args.texts.len()
)));
```

---

### `crates/aprender-forecast/src/fit.rs` (service, L-BFGS wrapper — D-09)

**Analog:** `sources/004-forecast-mcp-thin-server/src/lib.rs:130-172` (the evolved copy with
`evals` counter, `FitInfo`, `budget_hit`; `sources/006/src/fit.rs` is the 29-line reduced form
and lacks `FitInfo` — port the 004 version).

```rust
/// Objective + gradient evaluated ONCE per distinct point; L-BFGS's line search asks for both
/// at the same x, and re-asks at the accepted point next iteration.
struct Cached<'a> { model: Model<'a>, last: RefCell<Option<(Vec<f64>, f64, Vec<f64>)>>, evals: RefCell<usize> }
impl<'a> Cached<'a> {
    fn ensure(&self, x: &[f64]) {
        let hit = self.last.borrow().as_ref().map_or(false, |(k, _, _)| k.as_slice() == x);
        if !hit {
            *self.evals.borrow_mut() += 1;
            let (f, g) = self.model.value_and_grad(x);
            *self.last.borrow_mut() = Some((x.to_vec(), f, g));
        }
    }
    fn f(&self, x: &[f64]) -> f64 { self.ensure(x); self.last.borrow().as_ref().expect("cached").1 }
    fn g(&self, x: &[f64]) -> Vec<f64> { self.ensure(x); self.last.borrow().as_ref().expect("cached").2.clone() }
}

pub struct FitInfo { pub rounds: usize, pub iterations: usize, pub evals: usize, pub objective: f64, pub status: String, pub budget_hit: bool }
pub const MAX_ITERS_PER_ROUND: usize = 2_000;
pub const FIT_BUDGET_SECS: f64 = 15.0;

pub fn fit_prophet(design: &Design, max_rounds: usize) -> (Params, FitInfo) {
    let model = Model::new(design);
    let init = model.init();
    let mut x = Vector::from_vec(model.pack(&init));
    let cache = Cached { model, last: RefCell::new(None), evals: RefCell::new(0) };
    let mut best_f = cache.f(x.as_slice());
    let mut opt = LbfgsF64::new(MAX_ITERS_PER_ROUND, 1e-7, 20);
    let (mut rounds, mut iters, mut status, mut budget_hit) = (0, 0, String::new(), false);
    let t0 = Instant::now();
    for _ in 0..max_rounds {
        let r = opt.minimize(|v: &Vector<f64>| cache.f(v.as_slice()), |v: &Vector<f64>| Vector::from_vec(cache.g(v.as_slice())), &x);
        rounds += 1; iters += r.iterations; status = format!("{:?}", r.status);
        let improved = r.objective_value < best_f - 1e-6 * best_f.abs().max(1.0);
        if improved { best_f = r.objective_value; x = r.solution; }
        if !improved || r.status == ConvergenceStatus::Converged { break; }
        if t0.elapsed().as_secs_f64() > FIT_BUDGET_SECS { budget_hit = true; break; }
    }
    let p = cache.model.unpack(x.as_slice());
    let evals = *cache.evals.borrow();
    (p, FitInfo { rounds, iterations: iters, evals, objective: best_f / cache.model.scale, status, budget_hit })
}
```
Clippy will flag `map_or(false, …)` (`unnecessary_map_or` → `is_some_and`) and the tuple
`let (mut …) = (…)`. Those are the mechanical edits D-08 permits. Core API confirmed present:
`LbfgsF64::new(max_iter, tol, m)` at `crates/aprender-core/src/optim/lbfgs.rs:796`, `minimize` `:811-819`,
`ConvergenceStatus` at `crates/aprender-core/src/optim/mod.rs:198-211`.

---

### `crates/aprender-forecast/src/{prophet,np,bolt,safetensors,dates}.rs` (models / utilities)

**Analogs (verbatim ports):** `sources/004/src/prophet.rs` (520), `sources/004/src/np.rs` (384),
`sources/007/src/bolt.rs` (323), `sources/007/src/safetensors.rs` (41), `sources/007/src/dates.rs` (49).

**Note on duplication:** `prophet.rs:8-36` already defines `days_from_civil`, `civil_from_days`,
`parse_ymd`, `format_ymd`; `dates.rs` defines the same functions plus `parse_date` (fallible) and
`future_days`. In the merged crate keep ONE copy in `dates.rs` and `pub use` it from `prophet`
(spike 010 reaches them as `forecast_mcp::prophet::{days_from_civil, format_ymd, Rng}` —
`sources/010/src/main.rs:32-39`; keep re-exports so the ported tests compile).

**Civil-date core** (`sources/007/src/dates.rs:2-10`, D-17, exact-parity proven):
```rust
pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}
```

**`future_days` D/W/MS with H refused** (`sources/004/src/lib.rs:117-128`):
```rust
pub fn future_days(last: i64, horizon: usize, freq: &str) -> Result<Vec<i64>, ForecastError> {
    Ok(match freq {
        "D" => (1..=horizon as i64).map(|i| last + i).collect(),
        "W" => (1..=horizon as i64).map(|i| last + 7 * i).collect(),
        "MS" => { /* civil_from_days → month arithmetic → days_from_civil(…, 1) */ }
        other => return Err(ForecastError::Validation(format!("unsupported freq {other:?}: use D, W or MS"))),
    })
}
```

**Safetensors decoder door** (`sources/007/src/safetensors.rs:1-12` — one byte-slice door for
embedded and on-disk weights):
```rust
pub struct Tensor { pub shape: Vec<usize>, pub data: Vec<f32> }
pub type Weights = HashMap<String, Tensor>;

/// Returns the tensors and the dtype they were stored in.
pub fn load_bytes(bytes: &[u8]) -> Result<(Weights, String), String> {
    if bytes.len() < 8 { return Err("safetensors: file too short".into()); }
    let n = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
    let header: serde_json::Value = serde_json::from_slice(bytes.get(8..8 + n).ok_or("safetensors: truncated header")?).map_err(|e| format!("header json: {e}"))?;
```

**GEMM routing (D-14)** — `sources/007/src/bolt.rs:35` (`pub fn dot8`), `:53` (`linear_fast`),
`:164, :175` (`trueno::blis::gemm_blis(lq, lk, dk, &qh, &kt, &mut scores, None).expect("gemm")`).
`gemm_blis` signature confirmed at `crates/aprender-compute/src/blis/compute.rs:838-846` (7 args,
`Option<&mut BlisProfiler>` last). `bolt.rs:49-61` has a `PARALLEL_GEMM` atomic + rayon path
behind the `parallel` feature — drop that branch (D-14 says the rayon path is not used and the
manifest will not enable `parallel`).

**np.rs load-bearing lines** (D-10): `:6` `use aprender::autograd::{clear_graph, graph_tape_len, no_grad, Tensor};`
and `:206-208` `weighted_huber` ("Exact SmoothL1 (Huber, β) × sample weight, mean — built from
graph-connected ops"). Never swap for `aprender::nn::loss::SmoothL1Loss`
(`crates/aprender-core/src/nn/loss.rs:171-195` builds a fresh leaf; detached).

---

### `crates/aprender-forecast/src/parity/*.rs` (tests, `#[cfg(test)]` — parity ladders)

**Analog A — in-lib test module shape:** `crates/aprender-mcp-setfit/src/lib.rs:226-229`:
```rust
#[cfg(test)]
#[allow(clippy::disallowed_methods)] // serde_json::json! expands to .unwrap() internally
mod tests {
    use super::*;
```

**Analog B — fixture path resolution:** `crates/aprender-core/tests/setfit_conformance.rs:137`:
```rust
PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/setfit")
```
For a `--lib` test the same expression works (`CARGO_MANIFEST_DIR` is the crate dir); fixtures
land in `crates/aprender-forecast/tests/fixtures/` (RESEARCH Pitfall 6, 18 committed JSON files).
Fixtures are committed, so use `expect("fixture <name> is committed; absence is a defect")` —
never a skip.

**Analog C — weight-gated tests (Bolt only):** RESEARCH F6 probe (no in-tree precedent for
`cfg_attr(…, ignore = "…")`; the only in-tree `rustc-check-cfg` emitter is
`crates/aprender-train/build.rs:75`):
```rust
#[test]
#[cfg_attr(not(chronos_weights), ignore = "CHRONOS_MODEL_DIR unset or has no model.safetensors — run `just fetch-chronos-tiny` to arm")]
fn bolt_quantiles_match_oracle_within_1e6() { /* … */ }
```
**Anti-pattern to NOT copy:** `crates/aprender-mcp-setfit-lambda/tests/embed.rs:14-20` and
`crates/aprender-mcp-setfit/tests/e2e_stdio.rs:62-68` (`println!("… SKIP …"); return;`) — they
report `0 ignored`, which D-18 forbids.

**Ladder shape:** the spike drivers already compute each rung; turn each printed row into an
assert. Tolerances are read from the contract YAML (D-15), not literal in two places.

---

### `crates/aprender-forecast/build.rs` and `crates/aprender-mcp-chronos/build.rs` (config; embed staging + cfg)

**Analog A — staging with empty marker fallback:** `crates/aprender-mcp-setfit-lambda/build.rs:12-33`:
```rust
fn main() {
    println!("cargo:rerun-if-env-changed=APRENDER_SETFIT_MODEL");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo always sets OUT_DIR"));
    let staged = out.join("model.apr");
    match std::env::var_os("APRENDER_SETFIT_MODEL") {
        Some(source) => {
            println!("cargo:rerun-if-changed={}", PathBuf::from(&source).display());
            std::fs::copy(&source, &staged).unwrap_or_else(|e| {
                panic!("APRENDER_SETFIT_MODEL={} could not be staged for embedding: {e}", PathBuf::from(&source).display())
            });
        }
        None => {
            std::fs::write(&staged, []).expect("write empty embed marker");
        }
    }
}
```
**Analog B — two-file version already written for Chronos:** `sources/007/build.rs:7-23`
(`CHRONOS_EMBED_DIR`, loops over `["model.safetensors", "config.json"]`). It joins no `".."`, so
`scripts/check_build_rs_paths.sh` skips it.

**Analog C — cfg emission (add to BOTH build scripts):** `crates/aprender-train/build.rs:75`
(`println!("cargo:rustc-check-cfg=cfg(feature, values(\"…\"))")`) shows the tree already
emits check-cfg from a build script; RESEARCH F6 (probed on rustc 1.93 / nextest 0.9.102):
```rust
println!("cargo::rustc-check-cfg=cfg(chronos_weights)");
println!("cargo:rerun-if-env-changed=CHRONOS_MODEL_DIR");
if let Some(dir) = std::env::var_os("CHRONOS_MODEL_DIR") {
    if std::path::Path::new(&dir).join("model.safetensors").is_file() {
        println!("cargo:rustc-cfg=chronos_weights");
    }
}
```
The `build.rs` of `aprender-mcp-chronos` combines B + C; `aprender-forecast`'s has only C.
Manifest needs `build = "build.rs"` (`sources/007/Cargo.toml:6`) — the lambda crate relies on
auto-discovery, either is fine.

---

### `crates/aprender-mcp-forecast/Cargo.toml` and `crates/aprender-mcp-chronos/Cargo.toml` (config, server manifests)

**Analog:** `crates/aprender-mcp-setfit/Cargo.toml` (whole file, quoted above) with:
- `[[bin]] name = "aprender-mcp-forecast"` / `"aprender-mcp-chronos"`, `path = "src/main.rs"` (`:21-23`)
- `pmcp = { version = "2.19", features = ["streamable-http", "schema-generation"] }` — the
  `2.19` form from `crates/aprender-mcp-setfit-train/Cargo.toml:41`, not SetFit's `2.9`
  (`router_with_config` needs 2.19).
- `aprender-forecast = { path = "../aprender-forecast" }` instead of the `aprender` line.
- `axum = { workspace = true }`; fit server adds `tower = { workspace = true }` (pool).
- `[dev-dependencies] reqwest = { workspace = true, features = ["json", "rustls-tls"] }` for the
  in-process e2e (the setfit-train crate's dev-dep comment at `Cargo.toml:47-53` explains why
  `default-features = false` is the workspace form).
- Keep `publish = false` with the SetFit comment (`:13-15`) and `[lints] workspace = true`.

**Gate consequence:** each `[[bin]]` is a FALSIFY-MONO-011 violation until the allowlist at
`crates/aprender-core/tests/monorepo_invariants.rs:264-297` (`ALLOWLIST_BASELINE = 27` at `:366`)
is amended — a human decision, `autonomous: false` (RESEARCH Open Question 1).

---

### `crates/aprender-mcp-forecast/src/lib.rs` (service: pmcp server + `http_app` + router pool)

**Analog A — server builder with `spawn_blocking`:** `crates/aprender-mcp-setfit/src/lib.rs:198-224`:
```rust
pub fn build_server(
    model: Arc<VerifiedSetFitModel>,
    name: &str,
    version: &str,
) -> pmcp::Result<Server> {
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
The forecast version is `sources/004/src/lib.rs:278-292` (no model argument; `forecast(&args)`
inside `spawn_blocking`); the Chronos version is `sources/007/src/lib.rs:144-157` (`Arc<Model>`
cloned into the closure exactly like the SetFit template).

**Error taxonomy mapping** (`sources/004/src/lib.rs:278`; SetFit equivalent `crates/aprender-mcp-setfit/src/lib.rs:180-187`):
```rust
fn map_error(e: ForecastError) -> pmcp::Error {
    match e { ForecastError::Validation(s) => pmcp::Error::validation(s), ForecastError::Internal(s) => pmcp::Error::internal(s) }
}
```

**Analog B — `http_app` (pmcp 2.19.3 API names verified in RESEARCH F2):** `sources/004/src/lib.rs:294-307`
(same in `sources/007/src/lib.rs:159-172`); reflowed for `max_width = 100`:
```rust
pub fn http_app(server: Server) -> axum::Router {
    use axum::response::{Html, IntoResponse};
    use axum::routing::get;
    let server = std::sync::Arc::new(tokio::sync::Mutex::new(server));
    let config = pmcp::axum::RouterConfig {
        server_config: pmcp::server::streamable_http_server::StreamableHttpServerConfig::stateless(),
        allowed_origins: Some(pmcp::axum::AllowedOrigins::localhost()),
        ..Default::default()
    };
    let mcp = pmcp::axum::router_with_config(server, config);
    axum::Router::new()
        .route("/", get(|| async { Html(include_str!("../static/index.html")) }))
        .route("/sample/peyton", get(|| async {
            ([("content-type", "text/csv")], include_str!("../fixtures/peyton_manning.csv")).into_response()
        }))
        .route("/sample/air", get(|| async {
            ([("content-type", "text/csv")], include_str!("../fixtures/air_passengers.csv")).into_response()
        }))
        .nest("/mcp", mcp)
}
```
`include_str!` paths are crate-relative, so each server crate ships `static/index.html` and
`fixtures/*.csv`. Copy the page from `.planning/spikes/004-forecast-mcp-thin-server/static/index.html`
(10.3 KB, tracked) — the `sources/004/static/index.html` in the skill is a 65-line stub, NOT the
spike-proven page. Same for 007 (`.planning/spikes/007-chronos-mcp-thin-server/static/index.html`).
After copying run `git check-ignore -v <file>` by hand (CLAUDE.md Publishing Safety; the
`check_include_files.sh` guard is vacuous on macOS).

**Analog C — router pool (D-12, no in-tree analog; spike 010 is the only source):** `sources/010/src/main.rs:55-68`:
```rust
/// `pool = 1`: the spike-004 app as shipped (one `Arc<Mutex<Server>>` behind pmcp's router).
/// `pool > 1`: K independent pmcp routers, each over its own `Server`, with a round-robin front
/// handler — K tool calls can then be in flight at once.
fn app(pool: usize) -> axum::Router {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tower::ServiceExt;
    if pool <= 1 { return forecast_mcp::http_app(forecast_mcp::build_server("aprender-forecast-concurrency", "0.0.0").expect("server")); }
    let routers: std::sync::Arc<Vec<axum::Router>> = std::sync::Arc::new((0..pool).map(|_| forecast_mcp::http_app(forecast_mcp::build_server("aprender-forecast-concurrency", "0.0.0").expect("server"))).collect());
    let next = std::sync::Arc::new(AtomicUsize::new(0));
    axum::Router::new().fallback(move |req: axum::extract::Request| {
        let routers = routers.clone(); let next = next.clone();
        async move {
            let i = next.fetch_add(1, Ordering::Relaxed) % routers.len();
            routers[i].clone().oneshot(req).await.unwrap_or_else(|e| match e {})
        }
    })
}
```
Port as `pub fn pooled_app(pool: usize, name: &str, version: &str) -> pmcp::Result<axum::Router>`
(return the builder error instead of `expect`). `.unwrap_or_else(|e| match e {})` on `Infallible`
is not `unwrap()` and passes `.clippy.toml`.

---

### `crates/aprender-mcp-forecast/src/main.rs` and `crates/aprender-mcp-chronos/src/main.rs` (controllers)

**Analog A — repo conventions:** `crates/aprender-mcp-setfit/src/main.rs`:
- stderr-only for humans (`:6-7` "Everything human-readable goes to stderr: stdout belongs to the protocol")
- hand-rolled argv loop, no clap (`:21-47`)
- `#[tokio::main] async fn main() -> ExitCode` with `ExitCode::from(2)` for usage errors, `FAILURE` for load/serve errors (`:49-86`)
- `server.run_stdio().await` (`:81`)
```rust
    let server =
        match aprender_mcp_setfit::build_server(model, SERVER_NAME, env!("CARGO_PKG_VERSION")) {
            Ok(server) => server,
            Err(error) => {
                eprintln!("error: server construction refused: {error}");
                return ExitCode::FAILURE;
            }
        };

    if let Err(error) = server.run_stdio().await {
        eprintln!("error: stdio server terminated: {error}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
```

**Analog B — the extra modes:** `sources/004/src/main.rs:41-62` (`--bench`, `--http PORT`, default
stdio) and `sources/007/src/main.rs:77-103` (`--bench`, `--coldstart N`, `--http`, stdio):
```rust
        Some("--http") => {
            let port: u16 = args.get(1).and_then(|p| p.parse().ok()).unwrap_or(8765);
            let server = match forecast_mcp::build_server(SERVER_NAME, env!("CARGO_PKG_VERSION")) { Ok(s) => s, Err(e) => { eprintln!("error: {e}"); return ExitCode::FAILURE; } };
            let app = forecast_mcp::http_app(server);
            let listener = match tokio::net::TcpListener::bind(("127.0.0.1", port)).await { Ok(l) => l, Err(e) => { eprintln!("error: bind {port}: {e}"); return ExitCode::FAILURE; } };
            eprintln!("{SERVER_NAME}: demo page http://127.0.0.1:{port}/  — MCP streamable-http at http://127.0.0.1:{port}/mcp");
            if let Err(e) = axum::serve(listener, app).await { eprintln!("error: {e}"); return ExitCode::FAILURE; }
            ExitCode::SUCCESS
        }
```
Add `--pool K` (default 8) to the fit server's `--http` branch → `pooled_app(K, …)`. The
`--bench`/`--coldstart` functions (`sources/004/src/main.rs:8-39`, `sources/007/src/main.rs:11-75`)
use `println!` deliberately (reports, not protocol) and read `fixtures/peyton_manning.csv` by
relative path — switch them to `include_str!` or `CARGO_MANIFEST_DIR` so they work from any cwd.
`sources/007/src/main.rs:24-32` toggles the `fast/attn_gemm/row1_gemv/PARALLEL_GEMM` variants —
drop the `PARALLEL_GEMM` row (D-14).

---

### `crates/aprender-mcp-forecast/src/lib.rs` `#[cfg(test)] mod e2e` and the Chronos equivalent (tests, in-process streamable-HTTP)

**Analog:** `sources/004/tests/e2e.rs` (100 lines) and `sources/007/tests/e2e.rs` (142 lines).
Both are in-process (`axum::serve` on `127.0.0.1:0` + `reqwest`), so they need no binary and can
live as `#[cfg(test)]` modules inside `src/` where CI's `--lib` nextest leg runs them (RESEARCH
F5). The client/harness shape (`sources/004/tests/e2e.rs:5-23`):
```rust
struct Client { http: reqwest::Client, url: String, next: u64 }
impl Client {
    async fn call(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
        self.next += 1;
        let body = serde_json::json!({"jsonrpc": "2.0", "id": self.next, "method": method, "params": params});
        let r = self.http.post(&self.url).header("content-type", "application/json").header("accept", "application/json, text/event-stream").body(body.to_string()).send().await.expect("send");
        let status = r.status();
        let text = r.text().await.expect("text");
        assert!(status.is_success(), "{method}: HTTP {status}\n{text}");
        let payload = text.lines().find_map(|l| l.strip_prefix("data: ")).unwrap_or(&text);
        serde_json::from_str(payload).unwrap_or_else(|e| panic!("{method}: non-JSON: {e}\n{text}"))
    }
}

fn tool_output(v: &serde_json::Value) -> serde_json::Value {
    if let Some(s) = v["result"].get("structuredContent") { return s.clone(); }
    let text = v["result"]["content"][0]["text"].as_str().expect("text content");
    serde_json::from_str(text).expect("tool JSON")
}
```
Server bring-up (`sources/004/tests/e2e.rs:33-39`):
```rust
    let server = forecast_mcp::build_server("aprender-forecast-test", "0.0.0").expect("server");
    let app = forecast_mcp::http_app(server);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move { axum::serve(listener, app).await.expect("serve"); });
    let mut c = Client { http: reqwest::Client::new(), url: format!("http://{addr}/mcp"), next: 0 };
```
Refusal cases (`:68-78`): unknown field, `ds[..5]` ("at least"), `"2008-02-30"` ("calendar").
The 5 `.unwrap()`s in these two files (`sources/004/tests/e2e.rs:28, 62, 65, 98`;
`sources/007/tests/e2e.rs` 1) become `expect` — or the test module carries
`#[allow(clippy::disallowed_methods)]` as `crates/aprender-mcp-setfit/src/lib.rs:227` does.

**Equality-under-load test (D-12):** port `sources/010/src/main.rs:43-49` (`signature`,
`max_abs_diff`) and the sequential-then-concurrent loop at `:95-119` into a `#[tokio::test]`;
assert `identical == reqs.len()`; assert the speed-up only under
`cfg!(target_arch = "aarch64") && !cfg!(debug_assertions)`, else log (RESEARCH Validation table).

**Chronos e2e gating:** wrap each weight-dependent `#[tokio::test]` with
`#[cfg_attr(not(chronos_weights), ignore = "…")]`; the `serve(model_dir)` helper
(`sources/007/tests/e2e.rs:29-38`) should read `CHRONOS_MODEL_DIR` instead of the literal
`"models/tiny"`. Refusal-only cases that need no weights should not be gated — but note
`forecast()` takes `&Model` for `cfg.prediction_length`, so plan a weights-free constructor or
gate them too.

---

### `crates/aprender-mcp-forecast/tests/e2e_stdio.rs` (test, spawned binary — dark in CI)

**Analog:** `crates/aprender-mcp-setfit/tests/e2e_stdio.rs` (217 lines). Copy the harness whole:
`KillOnDrop` (`:29-39`), `request`/`send`/`recv_json` (`:41-58`), `env!("CARGO_BIN_EXE_aprender-mcp-setfit")`
spawn (`:76-85`), reader thread (`:89-100`), the `tools/list` "exactly ONE tool" +
`additionalProperties == false` assertions (`:118-132`), the unknown-key probe (`:188-207`), and
the shutdown note (`:209-216` — pmcp's `run_stdio` does not exit on EOF; the client kills).
The fit server needs no model, so drop the `ENV_MODEL` gate (`:20, 60-74`) entirely — this test
can always run. It is still dark in CI until `.github/workflows/ci.yml:471` lists it
(`autonomous: false`).

---

### `crates/aprender-mcp-chronos/src/lib.rs` (service: embedded weights + runtime fallback)

**Analog A — the embed/fallback door:** `crates/aprender-mcp-setfit-lambda/src/lib.rs:18, 44-55`:
```rust
pub static EMBEDDED_MODEL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/model.apr"));

pub fn resolve_model() -> Result<Arc<aprender_mcp_setfit::Model>, ModelLoadError> {
    if !EMBEDDED_MODEL.is_empty() {
        return aprender_mcp_setfit::load_model_from_bytes(EMBEDDED_MODEL).map(Arc::new);
    }
    let path = std::env::var_os("APRENDER_SETFIT_MODEL").ok_or_else(|| {
        ModelLoadError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no embedded model in this build and APRENDER_SETFIT_MODEL is unset",
        ))
    })?;
    aprender_mcp_setfit::load_model_from_path(std::path::Path::new(&path)).map(Arc::new)
}
```
**Analog B — the Chronos version already written:** `sources/007/src/lib.rs:29-56`:
```rust
pub static EMBEDDED_WEIGHTS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/model.safetensors"));
pub static EMBEDDED_CONFIG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/config.json"));

pub struct Model { pub bolt: Bolt, pub name: String, pub n_params: usize, pub dtype: String, pub source: String, pub load_seconds: f64 }

pub fn load_model_from_bytes(weights: &[u8], config: &[u8], source: &str) -> Result<Model, String> { /* safetensors::load_bytes → Bolt::load */ }
pub fn load_model_from_dir(dir: &std::path::Path) -> Result<Model, String> { /* fs::read both files → load_model_from_bytes */ }
pub fn resolve_model() -> Result<Model, String> {
    if !EMBEDDED_WEIGHTS.is_empty() { return load_model_from_bytes(EMBEDDED_WEIGHTS, EMBEDDED_CONFIG, "embedded"); }
    let dir = std::env::var_os("CHRONOS_MODEL_DIR").ok_or_else(|| "no embedded model in this build and CHRONOS_MODEL_DIR is unset".to_string())?;
    load_model_from_dir(std::path::Path::new(&dir))
}
```
Replace `Result<_, String>` with a `ModelLoadError` enum implementing `Display + Error` as the
SetFit crate does (`crates/aprender-mcp-setfit/src/lib.rs:94-112`) — pedantic clippy and the
`main.rs` error printing both prefer it.

**Horizon gate + warning (D-13):** `sources/007/src/lib.rs:108-110, 130-131`:
```rust
    if args.horizon > cfg.prediction_length && !args.allow_long_horizon {
        return Err(bad(format!("horizon {} exceeds the model's native {} steps; beyond that the model rolls its own forecast forward and accuracy degrades — pass allow_long_horizon: true to accept (max {MAX_HORIZON})", args.horizon, cfg.prediction_length)));
    }
    // …
    let rollouts = forwards.saturating_sub(1) / cfg.quantiles.len();
    let warning = (args.horizon > cfg.prediction_length).then(|| format!("horizon {} exceeds the model's native {} steps: steps {}+ come from {} autoregressive rollout(s) …", args.horizon, cfg.prediction_length, cfg.prediction_length + 1, rollouts));
```

---

### Root `Cargo.toml` (workspace members + per-package profile override)

**Analog — member block to extend:** `Cargo.toml:46-53`:
```toml
    # --- MCP server (Model Context Protocol) ---
    "crates/aprender-mcp",
    # Thin single-model MCP server: SetFit classification via pmcp (pmcp.run template)
    "crates/aprender-mcp-setfit",
    "crates/aprender-mcp-setfit-lambda",
    # Thin single-algorithm MCP TRAINING server: SetFit training as an async MCP Task
    "crates/aprender-mcp-setfit-train",
```
Add after it, same comment style:
```toml
    # Phase 6: pure-Rust forecasting library + two thin stateless `forecast` MCP servers
    "crates/aprender-forecast",
    "crates/aprender-mcp-forecast",
    "crates/aprender-mcp-chronos",
```
**Analog — per-package profile override (RESEARCH Pitfall 9, only after measuring):** `Cargo.toml:649-653`:
```toml
[profile.test.package.proptest]
debug-assertions = false

[profile.dev.package.proptest]
debug-assertions = false
```
→ `[profile.dev.package.aprender-forecast] opt-level = 3` (inherited by `profile.test`).

---

### `contracts/{forecast-tool-boundary,prophet-parity,chronos-bolt-parity}-v1.yaml` (contracts)

**Analog:** `contracts/setfit-apr-v1.yaml` — the ONLY safe template (`contracts/neon-blis-v1.yaml`
parses to 0/0/0 under `pv status`; RESEARCH F4). Top-level keys in order: `contract:` (1),
`metadata:` (2), `equations:` (80), `proof_obligations:` (1010), `falsification_tests:` (1103),
`kani_harnesses:` (1243), `qa_gate:` (1276).

**Metadata head** (`contracts/setfit-apr-v1.yaml:1-7`):
```yaml
contract: setfit-apr
metadata:
  version: 1.0.0
  created: '2026-08-15'
  author: PAIML Engineering
  kind: kernel
  description: >
```
`kind: kernel` is what makes PROVABILITY-001 bite (obligations, tests, harnesses all required;
`falsification_tests.len() >= proof_obligations.len()`). `metadata.references` must be non-empty
(SCHEMA-001).

**Equation shape** (`:87-92`):
```yaml
  artifact_storage_map:
    formula: '…'
    domain: '…'
    codomain: '…'
    preconditions:
      - '…'
```

**Proof obligation shape** (`:1012-1015`):
```yaml
  - type: completeness
    property: 'Every APR-01 item has exactly one storage location in the storage map, …'
    formal: 'forall item in APR_01_ITEMS: exists! loc in storage_map: loc.item == item'
    applies_to: all
```

**Falsification test shape** (`:1105-1109`) — the `test:` field names the runnable cargo test;
tolerances go in `prediction:`:
```yaml
  - id: FALSIFY-APR-001
    rule: 'Head coefficients are tensors'
    prediction: 'An artifact written by write_setfit_apr contains tensor index entries named exactly …'
    test: 'Write a fixture artifact; assert both index entries exist with the declared shapes and dtypes; …'
    if_fails: 'The writer put the head in metadata (review B1 regression), …'
```

**Kani block house rule** (`:1243-1253`, keep the comment verbatim):
```yaml
kani_harnesses:

  # DECLARED, NOT EXECUTED. cargo-kani is not installed in this repository and no
  # #[kani::proof] harness exists anywhere under crates/. Each entry names the
  # runnable, identically bounded test that is the ACTUAL evidence.
  - id: KANI-APR-001
    obligation: '…'
    property: '…'
    bound: 5
    strategy: bounded_int
    harness: 'NOT EXECUTED — the evidence is the bounded proptest `…` (FALSIFY-APR-004), …'
```

**Validate with:** `cargo run -p aprender-contracts-cli --bin pv -- validate contracts/<f>.yaml`
then `… pv status contracts/<f>.yaml` and confirm non-zero counts. Binding rows (if added to
`contracts/aprender/binding.yaml:4-10`) use the bare filename and `status ∈ {implemented, partial, not_implemented, pending}`:
```yaml
- contract: softmax-kernel-v1.yaml
  equation: softmax
  module_path: aprender::nn::functional::softmax
  function: softmax
  signature: 'fn softmax(x: &Tensor, dim: i32) -> Tensor'
  status: implemented
```

---

### `Makefile` (`CONTRACTS`, `PHASE6_CONTRACTS`, `contract-audit-phase6`)

**Analog:** `Makefile:1815-1866` (the `CONTRACTS :=` list ends with
`contracts/setfit-apr-v1.yaml \` / `contracts/setfit-benchmark-claims-v1.yaml`) and the
one-entry phase list pattern at `:1905`:
```make
PHASE4_CONTRACTS := contracts/setfit-apr-v1.yaml
```
and `:1918`:
```make
PHASE5_CONTRACTS := contracts/setfit-benchmark-claims-v1.yaml
```
The comment at `:1920-1927` is the rule: "A contract file that merely EXISTS in contracts/ is
validated by nothing … Every future phase contract needs its own line here or it is decoration."
Add the three files to `CONTRACTS` (continuation-backslash list) and a `PHASE6_CONTRACTS` +
`contract-audit-phase6` target mirroring `contract-audit-phase4/5`. The validate loop
(`:1929-1935`) is the gate that reaches them:
```make
contract-validate: ## Validate all kernel contracts (schema + staleness)
	@echo "Validating kernel contracts..."
	@for contract in $(CONTRACTS); do \
		echo "  $$contract"; \
		$(PV_BIN) validate "$$contract" || exit 1; \
	done
	@echo "Contract validation passed"
```

---

### `justfile` (`fetch-chronos-tiny`, `chronos-gate`, `forecast-bench`, `chronos-coldstart`)

**Analog:** `justfile:1-6` header defines the split ("The repo's quality gates live in the Makefile
… this file is the deployment surface"), `:21` `set shell := ["bash", "-uc"]`, `:26-28`
variables via `env_var_or_default`, and the recipe shape at `:40-46`:
```make
build-apr-arm64:
    @command -v cargo-zigbuild >/dev/null || cargo install cargo-zigbuild
    cargo zigbuild --release --target {{target}} \
        --bin apr --no-default-features --features setfit
    @file target/{{target}}/release/apr | grep -q 'ARM aarch64' \
        || { echo "ERROR: not an aarch64 binary — check the target"; exit 1; }
    @ls -lh target/{{target}}/release/apr | awk '{print "  apr (arm64): " $5}'
```
Multi-line bash recipes use the shebang form (`:70` `#!/usr/bin/env bash` under
`build-trainer-asset`). Weight fetch: RESEARCH "Code Examples" gives the full `fetch-chronos-tiny`
recipe (pinned revision `a0e552de…`, f32 sha256 `75068728…`, config sha `278f0086…`, f16
derived). `/models/` is already gitignored root-anchored (`.gitignore:51-52`) and
`*.safetensors` globally (`:48`). `chronos-gate` must capture `rc=$?` on its own line, never
through a pipe (CLAUDE.md Verification Discipline #1), and assert `0 ignored` in the test
summary.

---

### README / CLAUDE.md drift edits

**What the gates check** (`crates/aprender-core/tests/readme_contract.rs`):
- `:130` `format!("| Workspace crates | **{crate_count}** workspace crates |")` → README.md:43 (`**82**` now; 83 measured; 86 after +3).
- `:171` `format!("**{contract_count}** provable contracts")` → README.md:44 (`**1778**` now; 1786 measured; +3).
- `:245` every `crates/*/Cargo.toml` dir must have `README.md` (currently missing: `aprender-mcp-setfit-lambda`, `aprender-contrastive-data`).
- `:275` `content.contains("paiml/aprender")` in every crate README (currently failing: `aprender-mcp-setfit`).
- `:456-466` every backticked `CLAUDE.md` token that contains `/`, ends in `.rs/.yaml/.yml/.toml/.sh/.md`, has no `..`/metacharacters, and does not start with `http`/`~`/`/`/`target/` must exist on disk.

**README link line analog:** `crates/aprender-mcp/README.md:3`:
```markdown
Model Context Protocol (MCP) server for [aprender](https://github.com/paiml/aprender).
```
**README opening analog (thin-server framing):** `crates/aprender-mcp-setfit/README.md:1-7`
(it links `paiml/rust-mcp-sdk` but not `paiml/aprender` — that is the pre-existing failure; the
new READMEs must include the aprender URL, and D-16 requires the empirical-coverage sentence).

**CLAUDE.md row analog:** `CLAUDE.md:216`:
```
| SetFit Classification Inference | **Primary** (aprender-core: loader + `VerifiedSetFitModel::classify`) | HTTP transport ONLY (route/`AppState`/readiness — calls core) | Compute |
```
followed by the exception paragraph at `:218-226`. RESEARCH F8 supplies the proposed row and
paragraph text; every backticked path in them must exist in the same commit.

---

## Shared Patterns

### File-scope `disallowed_methods` allow (only where `JsonSchema`/`json!` expand)
**Source:** `crates/aprender-mcp-setfit/src/lib.rs:29-32`
**Apply to:** `aprender-forecast/src/types.rs`, both server `src/lib.rs`, test modules that use `json!`
```rust
// schemars' JsonSchema derive and serde_json::json! both expand to .unwrap()
// internally, and the derive's generated impl lands at file scope where a
// struct-level allow cannot reach it. Same precedent as aprender-mcp's tools.
#![allow(clippy::disallowed_methods)]
```
Test-module form: `#[cfg(test)] #[allow(clippy::disallowed_methods)] // serde_json::json! expands to .unwrap() internally` (`:226-227`).

### Refusal, never default — `pmcp::Error::validation` naming the fix
**Source:** `crates/aprender-mcp-setfit/src/lib.rs:155-174` (`precheck`) and `sources/004/src/lib.rs:174-190`
**Apply to:** every D-11 bound in `aprender-forecast::forecast` / `bolt::forecast`; `map_error` in both servers.
Validation → `pmcp::Error::validation(msg)`; everything else → `pmcp::Error::internal(msg)`; join errors → `internal(format!("… task join: {e}"))`.

### CPU work on `spawn_blocking`
**Source:** `crates/aprender-mcp-setfit/src/lib.rs:214-217`
**Apply to:** both `build_server` closures (fits and forwards). Also the D-10 guarantee that two autograd fits never interleave on one thread.

### stderr for humans, stdout for the protocol
**Source:** `crates/aprender-mcp-setfit/src/main.rs:6-7, 54, 62, 66-70, 76, 82`
**Apply to:** both `main.rs`. `--bench`/`--coldstart` reports may `println!` because they never run alongside the protocol.

### Strict tool schema pinned by a unit test
**Source:** `crates/aprender-mcp-setfit/src/lib.rs:305-320`
**Apply to:** both server crates (one test each on `ForecastArgs`)
```rust
    #[test]
    fn the_tool_schema_is_strict_and_names_both_fields() {
        let schema =
            serde_json::to_value(schemars::schema_for!(ClassifyArgs)).expect("schema serializes");
        assert_eq!(
            schema["additionalProperties"],
            serde_json::json!(false),
            "deny_unknown_fields must surface in the advertised schema"
        );
        let required = schema["required"].as_array().expect("required array");
        assert!(required.contains(&serde_json::json!("texts")));
```

### Embedded-or-env artifact resolution
**Source:** `crates/aprender-mcp-setfit-lambda/build.rs:12-33` + `src/lib.rs:18, 44-55`
**Apply to:** `aprender-mcp-chronos` (`CHRONOS_EMBED_DIR` at build, `CHRONOS_MODEL_DIR` at runtime).

### Counted skip for weight-dependent tests
**Source:** RESEARCH F6 probe; in-tree check-cfg precedent `crates/aprender-train/build.rs:75`
**Apply to:** every Bolt parity / Chronos e2e test. `build.rs` emits `cargo::rustc-check-cfg=cfg(chronos_weights)` + conditional `cargo:rustc-cfg=chronos_weights`; tests carry `#[cfg_attr(not(chronos_weights), ignore = "…run `just fetch-chronos-tiny` to arm")]`. Never the `println!("SKIP"); return;` form (`crates/aprender-mcp-setfit-lambda/tests/embed.rs:14-20`).

### Keep tests in `--lib` reach
**Source:** `.github/workflows/ci.yml:289` (nextest `--workspace --lib`) vs `:471` (the one `--test` line, human-edited)
**Apply to:** parity ladders, in-process e2e, pool-equality test — all `#[cfg(test)]` in `src/`. Only the spawned-binary stdio test is a `tests/*.rs` target (dark until the line is edited).

## No Analog Found

Files with no close in-tree match (planner should use the spike source + RESEARCH patterns):

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `crates/aprender-mcp-forecast/src/lib.rs::pooled_app` | service (router pool) | request-response | No workspace crate pools pmcp routers; `sources/010/src/main.rs:55-68` is the only source (D-12) |
| `crates/aprender-forecast/examples/mase_rolling_origin.rs` | example (rolling-origin MASE) | batch | No forecasting example in tree; source is `sources/006/src/main.rs` (190 lines) — compiled by CI's `cargo build --examples --workspace --keep-going` (ci.yml:471), so it must build clean |
| `static/index.html` (both servers) | asset (browser MCP client) | request-response | No in-tree same-origin MCP demo page; copy the tracked spike pages from `.planning/spikes/00{4,7}-*/static/index.html`, not the stubs under `sources/` |

## Metadata

**Analog search scope:** `crates/aprender-mcp-setfit/`, `crates/aprender-mcp-setfit-lambda/`, `crates/aprender-mcp-setfit-train/Cargo.toml`, `crates/aprender-data/Cargo.toml`, `crates/aprender-train/build.rs`, `crates/aprender-core/tests/{readme_contract,monorepo_invariants,setfit_conformance}.rs`, `contracts/setfit-apr-v1.yaml`, `contracts/aprender/binding.yaml`, root `Cargo.toml`, `Makefile:1812-1940`, `justfile`, `README.md:38-48`, `CLAUDE.md:205-228`, `.gitignore`, and the spike sources `sources/00{4,6,7,10}-*/` (all tracked; 90 files)
**Files scanned:** 38
**Tracked-source gate:** every analog path verified with `git ls-files` (non-empty); no `.gsd/capabilities/` mirror paths emitted
**Pattern extraction date:** 2026-09-05
