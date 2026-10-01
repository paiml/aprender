# Forecast MCP Thin Server (Prophet + NeuralProphet), Stateless, with Concurrency

One pmcp server, one typed `forecast` tool, no model artifact: the request carries the series,
the server fits and forecasts inside the call. Serves stdio (Claude Desktop / Code) and
streamable-HTTP (pmcp.run Lambda loopback) plus a same-origin demo page that is itself an MCP
client (spike 004). Spike 010 measured concurrency and found the transport lock that must be
pooled around.

## Requirements

From the `prophet-forecast-mcp` idea (MANIFEST.md):

- **MCP serving shape is a STATELESS `forecast` tool**: one call carries `ds[]`, `y[]`, horizon
  (and freq); the server fits and forecasts inside that call. No fit → artifact → forecast
  round-trip (decided 2026-09-04).
- Both Prophet and NeuralProphet are behind the one tool (`model: prophet | neuralprophet`).
- One model per server (the Chronos forecaster gets its own thin server; see
  `chronos-mcp-server.md`). All servers share the request/response shape defined here.

## How to Build It

**Crate shape:** `aprender-mcp-forecast` (thin; `TOOL_NAME = "forecast"`) over a library crate
holding `prophet.rs` + `np.rs` + `dates.rs` (`aprender-forecast`, or core). The Lambda loopback
wrapper is a copy of `aprender-mcp-setfit-lambda` with no embedded model. Template for everything:
`crates/aprender-mcp-setfit`.

**Dependencies** (all already in the workspace lock):

```toml
pmcp = { version = "2.19", features = ["streamable-http", "schema-generation"] }
schemars = "1.0"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["full"] }
axum = "0.8"
tower = "0.5"                      # router pool `oneshot`
reqwest = { version = "0.12", features = ["json"] }   # dev: e2e
```

**1. The tool boundary** — `sources/004-forecast-mcp-thin-server/src/lib.rs`:

```rust
pub const MAX_POINTS: usize = 20_000;
pub const MAX_HORIZON: usize = 3_650;
pub const MIN_POINTS: usize = 10;

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ForecastArgs {
    pub ds: Vec<String>,                 // YYYY-MM-DD, ascending, unique, real calendar dates
    pub y: Vec<f64>,
    pub horizon: usize,                  // 1 … 3650
    #[serde(default)] pub freq: Option<String>,             // "D" | "W" | "MS"
    #[serde(default)] pub model: Option<String>,            // "prophet" | "neuralprophet"
    #[serde(default)] pub growth: Option<String>,           // linear | logistic (needs cap) | flat
    #[serde(default)] pub cap: Option<f64>,
    #[serde(default)] pub seasonality_mode: Option<String>, // additive | multiplicative
    #[serde(default)] pub interval_width: Option<f64>,      // (0,1), default 0.8
    #[serde(default)] pub holidays: Option<Vec<HolidayArg>>,
    #[serde(default)] pub n_lags: Option<usize>,            // NeuralProphet AR-Net
    #[serde(default)] pub seed: Option<u64>,                // default 42 — determinism per request
}

pub struct ForecastResponse { model, freq, n_history, fit_seconds, predict_seconds,
    ds, yhat, yhat_lower, yhat_upper, trend, components: Map, diagnostics: Value }
```

Refuse at the boundary, as `pmcp::Error::validation`: unknown fields; < 10 or > 20 000 points;
`ds`/`y` length mismatch; non-ascending, duplicate or impossible dates; horizon 0 or > 3650;
unknown `freq`; `interval_width` outside (0,1); logistic without `cap` or `cap ≤ max(y)`;
**constant `y`** (Prophet's objective diverges — see `prophet-fit-and-predict.md`).

**2. Registration and blocking:**

```rust
pub fn build_server(name: &str, version: &str) -> pmcp::Result<Server> {
    Server::builder().name(name).version(version)
        .capabilities(ServerCapabilities::tools_only())
        .tool_typed_with_description::<ForecastArgs, _, _>(TOOL_NAME, TOOL_DESCRIPTION, move |args, _extra| async move {
            let response = tokio::task::spawn_blocking(move || forecast(&args)).await
                .map_err(|e| pmcp::Error::internal(format!("forecast task join: {e}")))?
                .map_err(map_error)?;
            serde_json::to_value(&response).map_err(|e| pmcp::Error::internal(format!("response serialization: {e}")))
        })
        .build()
}
```

**3. Same-origin HTTP app** (demo + MCP in one binary; the page is a real MCP client doing
`initialize` → `notifications/initialized` → `tools/list` → `tools/call`):

```rust
pub fn http_app(server: Server) -> axum::Router {
    let server = Arc::new(tokio::sync::Mutex::new(server));
    let config = pmcp::axum::RouterConfig {
        server_config: StreamableHttpServerConfig::stateless(),     // no session ids, JSON responses
        allowed_origins: Some(pmcp::axum::AllowedOrigins::localhost()), ..Default::default() };
    let mcp = pmcp::axum::router_with_config(server, config);
    axum::Router::new()
        .route("/", get(|| async { Html(include_str!("../static/index.html")) }))
        .nest("/mcp", mcp)
}
```

`main.rs`: `--stdio` default (`run_stdio()`), `--http PORT`, `--bench`. `tests/e2e.rs` drives the
app in-process over streamable-HTTP: initialize, tools/list, a forecast, three refusals, MS +
multiplicative, logistic + holidays, NeuralProphet ×2. The response is `content[0].text` JSON
(clients should also accept `structuredContent`).

**4. Concurrency — pool the router** (`sources/010-forecast-server-concurrency/src/main.rs`):

```rust
fn app(pool: usize) -> axum::Router {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tower::ServiceExt;
    let routers: Arc<Vec<axum::Router>> = Arc::new((0..pool).map(|_| http_app(build_server(NAME, VERSION).expect("server"))).collect());
    let next = Arc::new(AtomicUsize::new(0));
    axum::Router::new().fallback(move |req: axum::extract::Request| {
        let routers = routers.clone(); let next = next.clone();
        async move {
            let i = next.fetch_add(1, Ordering::Relaxed) % routers.len();
            routers[i].clone().oneshot(req).await.unwrap_or_else(|e| match e {})
        }
    })
}
```

Size K to the blocking-thread budget. Measured: 8 concurrent requests 5.41 s → **1.44 s** (3.9×),
16 NeuralProphet fits 19.4 s → 3.16 s, outputs bit-identical (8/8 ×3, 16/16).

**5. Observability to keep.** Response carries `fit_seconds`, `predict_seconds` and a `diagnostics`
object (seasonalities, active changepoints, σ, L-BFGS rounds/iterations/evals/objective/status,
`budget_hit`; for NP epochs/batch/steps/params/selected lr/final loss/residual sd). The page keeps
an exportable event log (ISO timestamps; `ui`, `rpc`, `mcp`, `forecast`, `error`; latency, bytes).

**6. Determinism check for release:** the spike-010 equality test — JSON signature of
`ds/yhat/bands/trend/components` for a request run alone vs under load must be identical.
Per-request seeds make this hold; never share a fit across requests.

## What to Avoid

- **pmcp 2.19.3's streamable-HTTP router holds one `Arc<tokio::sync::Mutex<Server>>` across the
  whole tool future** (`pmcp-2.19.3/src/server/streamable_http_server.rs:2094`, `:2122`; comment at
  `:1566`). `spawn_blocking` inside the tool buys nothing — the next request waits for the lock.
  A single router serialises every fit (8 concurrent = 1.0× sequential). Use the pool above, or an
  upstream pmcp change (lock only to route). On Lambda (one request per container) it does not matter.
- **Do not build a fit → artifact → forecast flow.** Fits take 0.04–1.5 s; the round-trip buys
  nothing and the MANIFEST forbids it.
- **Do not serve the page from another port** — cross-origin CORS gymnastics for nothing; nest the
  pmcp router under `/mcp` in the same axum app.
- **Do not skip the iteration cap and wall-clock budget** — a 20k-point Prophet fit ran 66 s uncapped.
- **Do not accept `freq` H** until the Prophet pipeline handles fractional days end to end.
- The `rtk` hook summarises `cargo test` output; run `target/release/deps/e2e-*` directly to see the
  timing `println!` lines.

## Constraints

- Latency (release, M4 Pro): Peyton 2905 daily → 365: Prophet 1.41 s round trip; NP lag-free 0.16 s;
  NP 30 lags (30→32→1) 2.75 s. Synthetic daily Prophet: 1k 0.15 s, 3k 0.21 s, 10k 1.19 s, 20k 8.2 s
  (capped, `budget_hit`). Full table: `sources/004-forecast-mcp-thin-server/BENCH.md`.
- `freq` D, W, MS via civil-date arithmetic (no `chrono`); `dates.rs` in spike 007 is the shared copy.
- Transport: `StreamableHttpServerConfig::stateless()` (no session ids, `enable_json_response`,
  4 MB body cap) — the same config the Lambda loopback wrapper uses.
- Bands: Prophet simulated (nominal 0.8 covers 0.60 on rolling origins), NP residual-sd (0.66).
  Report empirical coverage.

## Origin

Synthesized from spikes: 004, 010
Source files available in: `sources/004-forecast-mcp-thin-server/` (lib.rs, main.rs, prophet.rs, np.rs, static/index.html, tests/e2e.rs, BENCH.md),
`sources/010-forecast-server-concurrency/`
Sample CSVs (not copied): `.planning/spikes/004-forecast-mcp-thin-server/fixtures/`
