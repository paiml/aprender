# Chronos MCP Thin Server: Embedded Weights, f16, Horizon Gating, Lambda Sizing

Chronos-Bolt compiled into a pmcp thin server with one stateless `forecast` tool (the spike-004
shape, quantiles native). Exact parity with the Python pipeline through the server, 52 ms cold
start, 18.5 ms per 2048-point forward, 24 MB binary with embedded tiny-f16 weights (spike 007).

## Requirements

From the `prophet-forecast-mcp` idea (MANIFEST.md):

- **Chronos is a THIRD forecaster with its own thin server (one model per server)**; Chronos-Bolt
  first, Chronos-2 later.
- **Stateless `forecast` tool**: `ds[]`, `y[]`, horizon (and freq) in; forecast out; no artifact.
- Build order: the 008 NEON kernel lands before this server is sized (its numbers assume it).

## How to Build It

**Crate:** `aprender-mcp-chronos` = `sources/007-chronos-mcp-thin-server/src/{lib,main,bolt,safetensors,dates}.rs`
+ `build.rs` + `static/index.html` + `tests/e2e.rs`. Default embed **tiny-f16**; small-f16 as a
build option. Same pmcp/axum/tower stack as `forecast-mcp-thin-server.md`, plus `half = "2.7"`
and `trueno = { path = "crates/aprender-compute", package = "aprender-compute" }`.

**1. Embed the weights — the `aprender-mcp-setfit-lambda` pattern** (`build.rs`):

```rust
fn main() {
    println!("cargo:rerun-if-env-changed=CHRONOS_EMBED_DIR");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    match std::env::var_os("CHRONOS_EMBED_DIR") {
        Some(dir) => for f in ["model.safetensors", "config.json"] {
            let src = PathBuf::from(&dir).join(f);
            println!("cargo:rerun-if-changed={}", src.display());
            std::fs::copy(&src, out.join(f)).unwrap_or_else(|e| panic!("CHRONOS_EMBED_DIR: cannot stage {f}: {e}"));
        },
        None => for f in ["model.safetensors", "config.json"] { std::fs::write(out.join(f), []).expect("empty embed marker"); },
    }
}
```

```rust
pub static EMBEDDED_WEIGHTS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/model.safetensors"));
pub static EMBEDDED_CONFIG:  &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/config.json"));

/// Embedded bytes if the build staged them, else `CHRONOS_MODEL_DIR` read as a runtime path.
pub fn resolve_model() -> Result<Model, String> {
    if !EMBEDDED_WEIGHTS.is_empty() { return load_model_from_bytes(EMBEDDED_WEIGHTS, EMBEDDED_CONFIG, "embedded"); }
    let dir = std::env::var_os("CHRONOS_MODEL_DIR").ok_or("no embedded model in this build and CHRONOS_MODEL_DIR is unset")?;
    load_model_from_dir(std::path::Path::new(&dir))
}
```

Build: `CHRONOS_EMBED_DIR=models/tiny-f16 cargo build --release`. CI builds without the env var
and tests via the runtime path. The safetensors loader works on a byte slice, so embedded and
on-disk weights share one door (`safetensors::load_bytes` decodes F32/F16/BF16 via `half`).

**2. f16 weights.** Convert once with `tools/to_f16.py`
(`uv run --python 3.12 --with safetensors --with numpy tools/to_f16.py models/tiny models/tiny-f16`).
Measured cost: max |Δ| vs f32 on Peyton's nine quantiles 9.55e-4 = **0.11 % of the series std**;
load 27 ms vs 29 ms. The e2e test bars it at 2 %. (Chronos-2 pays 0.3–0.6 %; measure per model.)

**3. Horizon policy — from the spike-006 measurement, not opinion:**

```rust
pub const MIN_POINTS: usize = 4;
pub const MAX_POINTS: usize = 20_000;
pub const MAX_HORIZON: usize = 1_024;

pub struct ForecastArgs { ds, y: Vec<Option<f64>> /* null = missing */, horizon, freq,
    #[serde(default)] pub allow_long_horizon: bool, /* quantile levels, seed … */ }

if args.horizon > MAX_HORIZON { return Err(bad(format!("horizon {} exceeds the maximum {MAX_HORIZON}", args.horizon))); }
if args.horizon > cfg.prediction_length && !args.allow_long_horizon {
    return Err(bad(format!("horizon {} exceeds the model's native {} steps; beyond that the model rolls its own \
        forecast forward and accuracy degrades — pass allow_long_horizon: true to accept (max {MAX_HORIZON})", ...)));
}
let warning = (args.horizon > cfg.prediction_length).then(|| format!("horizon {} exceeds the model's native {} steps: \
    steps {}+ come from {} autoregressive rollout(s); spike 006 measured MASE 1.7 past step 64 vs 1.1 within it", ...));
```

Within 64 steps Chronos is within 0.1 MASE of the fitted models; past 64 its rollout degrades to
MASE 1.7. Default accept ≤ 64; `allow_long_horizon: true` up to 1024 with a `warning` in the response.

**4. Response shape.** Spike-004 fields (`ds`, `yhat`/`yhat_lower`/`yhat_upper` = q50/q10/q90,
`predict_seconds`) plus the full `quantiles` map, `context_used`, `forwards`, `rollouts`,
`missing_values`, weights dtype and source, `warning`. The server prints one banner line at start:
model, params, dtype, source, load time.

**5. Concurrency is free here.** The model is an immutable `Arc<Model>` with no per-thread state;
the forward runs on `spawn_blocking`. The pmcp router lock (see `forecast-mcp-thin-server.md`)
still serialises calls, but at 18 ms each it does not matter; on Lambda it never does.

**6. Refusals proven in `tests/e2e.rs`:** unknown field; < 4 points; impossible date; horizon 0;
all-null `y`; horizon > 64 without the flag (message names `allow_long_horizon`); with the flag
the response carries the warning and `forwards: 46`. Parity through the server: Peyton 64 steps
9.54e-7, 365 steps 1.91e-5 vs the Python oracle.

**7. Cold-start harness:** `chronos-mcp --coldstart 3` spawns itself over stdio and times
spawn → `initialize` reply → first forecast. Keep it; it is the Lambda number.

## What to Avoid

- **Do not lazy-download weights from the Hub at cold start** — network on the request path,
  against the offline vision. Embed (A/B) or runtime path (C) only.
- **Do not ship the untransposed weight copies or the loop paths** — cold start is dominated by the
  transposes at load; storing only `[in, out]` halves memory and cuts load time.
- **Do not embed small-f16 for a direct-upload Lambda zip** — 103 MB exceeds the 50 MB limit;
  deploy via S3 or a container image.
- **Do not expect threading to help** — `blis/parallel.rs` keeps products under 64 MFLOP on ≤ 2–4
  threads and thin ones serial, which is every GEMM in a 129-token transformer; rayon bought 4 %.
  A thin server on Lambda's 1–2 vCPUs loses nothing.
- **Do not report the nominal 80 % band** — Chronos-tiny covered 0.65 and small 0.69 on rolling
  origins (spike 006). Document empirical coverage.
- After GEMM routing the encoder is 80 % of the forward and sits at the kernel's rate; more routing
  is not worth it — the next lever is an 8×12 NEON tile (spike 008 headroom, 0.7× faer).

## Constraints

| build | binary (stripped) | cold start → first forecast | forward, 2048 pts | horizon 365 (46 forwards) | parity |
|---|---|---|---|---|---|
| tiny f16 embedded (default) | 24.4 MB (23.1) | 52 ms (85 cold cache) | 18.5 ms | 0.85 s | 1e-3 of std (f16) / 9.5e-7 (f32) |
| small f16 embedded | 103 MB (101.8) | 280 ms (448 cold) | 98 ms | 4.5 s | spike-006 parity |
| tiny f32 embedded | 41.9 MB (40.5) | ~55 ms | 18.5 ms | – | 9.5e-7 |
| no embed (runtime path) | 7.0 MB | + weight read | – | – | – |

- Latency grid (median of 5, tiny-f16 / small-f16): context 100 → 2.7 / 15.8 ms; 512 → 6.3 / 35 ms;
  2048 → 18.5 / 98 ms. Stage breakdown (tiny): patch embedding 1.3, encoder 14.7, decoder 2.3, head 0.1 ms.
- Context is capped at 2048 points by the model; longer `y` is truncated to the last 2048 and
  `context_used` says so.
- Prefer Bolt-**small** for accuracy (0.06 MASE better, best WQL3) when the deploy target can carry
  103 MB; tiny is the Lambda-zip deployable.
- Chronos-2 as a third "large" tier: 228 MB f16, ~0.5 s per forecast on one core, container image only.

## Origin

Synthesized from spike: 007 (horizon policy from 006; kernel numbers from 008; Chronos-2 sizing from 009)
Source files available in: `sources/007-chronos-mcp-thin-server/` (build.rs, src/, static/, tests/, tools/to_f16.py, RUN-OUTPUT.md)
Oracle fixture (not copied): `.planning/spikes/007-chronos-mcp-thin-server/fixtures/peyton_tiny_oracle.json`
