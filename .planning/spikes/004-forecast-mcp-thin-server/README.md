---
spike: 004
idea: prophet-forecast-mcp
name: forecast-mcp-thin-server
type: standard
validates: "Given a pmcp thin server with one stateless `forecast` tool wrapping the spike-001/003 Prophet port and the spike-002 NeuralProphet-lite, when called with ds/y/horizon over streamable-HTTP (and stdio), then fit + forecast returns in under 2 s for ~3k daily points with bands and components, malformed input is refused at the tool boundary, and a same-origin browser page lets you try your own series"
verdict: VALIDATED
related: [001, 002, 003]
tags: [mcp, pmcp, latency, ui, streamable-http, stateless]
---

# Spike 004: stateless `forecast` MCP tool + demo page

## What This Validates

Given the MANIFEST requirement "stateless `forecast` tool: fit and forecast inside the call", when
the three earlier spikes are put behind ONE typed pmcp tool served on stdio (for Claude Desktop /
Claude Code) and on streamable-HTTP (what pmcp.run's Lambda loopback serves), then a real MCP client
(the page, and the end-to-end test) gets a forecast with bands and components in under two seconds
for Peyton Manning, and every bad input is a validation error, not a silent default.

## Research

- **Template:** `crates/aprender-mcp-setfit` — `Server::builder().tool_typed_with_description::<Args,_,_>`,
  `#[serde(deny_unknown_fields)]` on the args, `spawn_blocking` for CPU work, `run_stdio()` locally;
  the Lambda wrapper (`aprender-mcp-setfit-lambda`) runs `StreamableHttpServer` with
  `StreamableHttpServerConfig::stateless()` (no session ids, JSON responses, 4 MB body cap).
- **Same-origin page:** `pmcp::axum::router_with_config(server, RouterConfig{server_config: stateless(),
  allowed_origins: localhost, ..})` returns an axum `Router` (routes at `/`), which nests under `/mcp`
  next to `GET /` (the page) and `GET /sample/*` — no CORS gymnastics, and the page is a genuine MCP
  client: `initialize` → `notifications/initialized` → `tools/list` → `tools/call`.
- **pmcp 2.19.3** resolves from the workspace lock; the `axum` feature is part of `streamable-http`.

| Approach | Pros | Cons | Status |
|---|---|---|---|
| Page talks to `StreamableHttpServer` on another port | Uses the exact Lambda server type | Cross-origin; needs CORS config, two processes | Rejected |
| axum app = pmcp router at `/mcp` + static page | One process, same origin, page is a real MCP client | Local demo only (Lambda uses the loopback wrapper) | **Chosen** |
| Fit → artifact → forecast tools | Mirrors setfit-train | Spikes 001–003 showed fits take 0.1–1.5 s; the artifact round-trip buys nothing | Rejected per MANIFEST requirement |

## How to Run

```bash
cd .planning/spikes/004-forecast-mcp-thin-server
export CARGO_TARGET_DIR=../../../target
cargo build --release
../../../target/release/forecast-mcp --http 8765      # demo: open http://127.0.0.1:8765/
../../../target/release/forecast-mcp                  # stdio MCP server (add to an MCP client config)
../../../target/release/forecast-mcp --bench          # latency table → BENCH.md
cargo test --release --test e2e -- --nocapture        # live streamable-HTTP end-to-end
```

Note: the `rtk` hook summarises `cargo test` output; run the test binary from `target/release/deps/e2e-*`
directly to see the timing lines.

## What to Expect

The page loads Peyton Manning or air passengers into a textarea, you pick model/horizon/freq/growth/
cap/seasonality/interval/n_lags/holidays, press **Forecast**, and get the chart (history, yhat, band,
trend), timing (fit / predict / round trip), the diagnostics JSON, and an event log with Export. The
e2e test prints `PEYTON … fit 1.4s` and the NeuralProphet lines.

## Observability

The page keeps an event log (ISO timestamps, categories `ui`, `rpc`, `mcp`, `forecast`, `error`,
per-call latency and byte counts) with an Export button producing JSON with a summary. The tool
response carries `fit_seconds`, `predict_seconds` and a `diagnostics` object (seasonalities,
changepoints active, σ, L-BFGS rounds/iterations/evaluations/objective/status/budget flag; for
NeuralProphet epochs/batch/steps/params/selected lr/final loss/residual sd).

## Investigation Trail

1. **One tool, one door.** `ForecastArgs` mirrors the request document idea from SetFit:
   `deny_unknown_fields`, bounds (`MIN_POINTS` 10, `MAX_POINTS` 20 000, `MAX_HORIZON` 3650,
   `interval_width ∈ (0,1)`, cap > max(y) for logistic, holiday windows, strictly ascending unique
   dates validated as real calendar dates). Constant `y` is refused up front (spike 001's divergence).
2. **First measurement: Peyton 2.55 s round trip** — over the 2 s bar. 1312 L-BFGS iterations but
   **11 312 evaluations**: the Wolfe search bisects hard near the L1 kinks, and objective and
   gradient were computed by separate passes.
3. **Three cheap fixes, measured one at a time:** row-major β-gradient pass (2.55 → 2.09 s); restart
   stop rule 1e-9 → 1e-6 relative (5 rounds instead of 6, same objective to 0.003); a single
   `value_and_grad` pass behind an x-keyed cache (2.09 → **1.40 s**).
4. **Scaling bench (synthetic daily series, trend + yearly + weekly + noise):** Prophet 1k 0.15 s,
   3k 0.21 s, 10k 1.19 s — but **20k took 66 s** with two rounds hitting the 10 000-iteration cap.
   Added `MAX_ITERS_PER_ROUND = 2000` and a 15 s wall-clock budget (reported in diagnostics):
   20k → 8.2 s, 3 rounds. The Peyton fit is unchanged.
5. **NeuralProphet with lags** was 4.2 s at 3k because the lr sweep trains three models with a hidden
   layer; two rates (0.03, 0.1 — 0.1 won every time in spike 002) → 2.75 s. Lag-free NP: 0.16 s.
6. **Transport.** Stateless pmcp on `/mcp`; the page needed no session id and gets plain JSON
   (`enable_json_response`). `notifications/initialized` is posted as a fire-and-forget. The response
   comes back as `content[0].text` JSON (the page also accepts `structuredContent`).
7. **Freq.** D, W and MS futures are civil-date arithmetic (no chrono). Hourly needs fractional days
   throughout the Prophet pipeline — not spiked; the tool refuses unknown freqs.

## Results

**Verdict: VALIDATED.** One stateless tool, two forecasters, bands and components, under two seconds
for Peyton, and a page that is itself an MCP client.

| series | model | fit | predict | round trip |
|---|---|---|---|---|
| Peyton Manning, 2905 daily → 365 | prophet | 1.40 s | 15 ms | 1.41 s |
| Peyton | neuralprophet | 0.16 s | 1 ms | 0.16 s |
| Peyton | neuralprophet n_lags=30 (30→32→1) | 2.74 s | 12 ms | 2.75 s |
| synthetic 3 000 daily | prophet | 0.20 s | 15 ms | 0.21 s |
| synthetic 10 000 daily | prophet | 1.17 s | 13 ms | 1.19 s |
| synthetic 20 000 daily | prophet | 8.2 s (capped) | 13 ms | 8.2 s |

Full table: `BENCH.md`. E2E test: 1 passed (initialize, tools/list, forecast, three refusals,
MS + multiplicative, logistic + holidays, NeuralProphet ×2).

**Surprises**
- The dominant cost was not the model but the line search's evaluation count; caching and one-pass
  value+gradient were worth 45 %. Core's `LbfgsF64` re-evaluates `f` and `∇f` at `x` inside every
  line search — a core fix would make the cache unnecessary.
- A 20k-point fit can run away (66 s); a stateless tool needs both an iteration cap and a wall-clock
  budget, and must say so in the response.
- pmcp's axum router nests cleanly under a path; the whole "demo page + MCP" is one binary.

**Signal for the build**
- Crate shape: `aprender-mcp-forecast` (thin, `TOOL_NAME = "forecast"`), `forecast()` in core or a
  `aprender-forecast` lib crate holding `prophet.rs` + `np.rs`; the Lambda loopback wrapper is a copy
  of `aprender-mcp-setfit-lambda` with no embedded model.
- Keep the bounds and the diagnostics object; add `MAX_POINTS`-dependent budgets to the contract.
- Provide `freq` H via fractional days; add country-holiday calendars as an optional feature.
- NeuralProphet bands are residual-sd based here; NP's quantile regression is a follow-up spike.
