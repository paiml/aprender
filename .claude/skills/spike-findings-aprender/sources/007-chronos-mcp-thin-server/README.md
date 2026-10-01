---
spike: 007
idea: prophet-forecast-mcp
name: chronos-mcp-thin-server
type: standard
validates: "Given Chronos-Bolt embedded in a pmcp thin server exposing the spike-004 `forecast` shape, when called over streamable-HTTP and stdio with ds/y/horizon, then native quantile bands return in under 100 ms for a 2048-point context at horizon ≤ 64, horizon > 64 is refused unless explicitly allowed (with a warning in the response), the browser page charts it, and binary size and cold start with embedded weights (f32 vs f16) are measured for Lambda"
verdict: VALIDATED
related: [004, 005, 006, 008]
tags: [chronos, mcp, pmcp, lambda, latency, embedded-weights, f16]
---

# Spike 007: Chronos MCP Thin Server

## What This Validates

Given the spike-005 Chronos-Bolt port compiled into a pmcp thin server with one stateless
`forecast` tool (the spike-004 request/response shape, quantiles native), when it is called over
streamable-HTTP and stdio, then (a) the quantiles match the Python oracle through the whole
server, (b) a 2048-point context forecasts in well under 100 ms, (c) horizons past the model's
native 64 steps are refused unless `allow_long_horizon` is set and then carry a warning, (d) the
same-origin page charts the nine-quantile fan, and (e) binary size and cold start are measured
for f32 and f16 embedded weights, tiny and small, the way the SetFit Lambda wrapper embeds its
artifact.

## Research

No new external dependencies. `pmcp` 2.19 / `schemars` 1.0 / `axum` 0.8 as in spike 004; `half`
2.7 (already in the workspace lock) decodes F16/BF16 safetensors.

**Embedding pattern** (`crates/aprender-mcp-setfit-lambda/build.rs`): a build-time env var names
the artifact, `build.rs` stages it into `OUT_DIR`, `include_bytes!` compiles it in; unset ⇒ empty
marker and a runtime path from the same env var. Mirrored here as `CHRONOS_EMBED_DIR` (build) /
`CHRONOS_MODEL_DIR` (runtime). The Lambda transport is `StreamableHttpServerConfig::stateless()`
behind a loopback proxy; this spike's `http_app` uses the same stateless config.

| Approach | Weights in the deployable | Pros | Cons | Status |
|----------|---------------------------|------|------|--------|
| A. Embed f32 safetensors | 34.6 MB (tiny) / 182 MB (small) | Zero load I/O, exact parity with spike 005 | Small is large for a Lambda zip | Measured |
| B. Embed f16 safetensors, decode at start | 17.3 MB / 95.5 MB | Halves the binary | Weight rounding; decode at cold start | **Chosen default; parity cost measured** |
| C. Runtime path via env var | none | CI builds without weights | Needs a volume or download on Lambda | Fallback, as SetFit |
| D. Lazy download from the Hub | none | Smallest artifact | Network on cold start, against the offline vision | Rejected |

**Horizon policy** comes from spike 006: within 64 steps Chronos is within 0.1 MASE of the fitted
models; past 64 its rollout degrades to MASE 1.7. So the tool accepts `horizon ≤ 64` by default,
`allow_long_horizon: true` up to 1024 with a `warning` in the response.

## How to Run

```bash
# weights (gitignored): models/tiny -> spike 005's models/, models/small -> spike 006's models/small,
# f16 copies via tools/to_f16.py:
uv run --python 3.12 --with safetensors --with numpy tools/to_f16.py models/tiny models/tiny-f16

# runtime-path build (no embed), bench and tests
CARGO_TARGET_DIR=../../../target cargo build --release
../../../target/release/chronos-mcp --bench models/tiny models/small-f16
CARGO_TARGET_DIR=../../../target cargo test --release          # e2e: parity, refusals, f16 vs f32
# (the rtk hook hides println! — run target/release/deps/e2e-* --nocapture for the numbers)

# embedded build, demo page, cold start
CHRONOS_EMBED_DIR=models/tiny-f16 CARGO_TARGET_DIR=../../../target cargo build --release
../../../target/release/chronos-mcp --http 8766     # http://127.0.0.1:8766/  (MCP at /mcp)
../../../target/release/chronos-mcp --coldstart 3   # spawns itself over stdio, times first replies
../../../target/release/chronos-mcp                 # stdio MCP server (Claude Desktop / Code)
```

## What to Expect

The page loads a sample, calls `tools/call forecast`, and draws the history with a four-band
quantile fan (q10–q90 … q40–q60) and the median; long horizons show the warning. `RUN-OUTPUT.md`
holds the bench, size and cold-start tables.

## Observability

The page keeps an event log (rpc timings, forecast diagnostics, errors) with Export JSON; the
response carries `predict_seconds`, `context_used`, `forwards`, `rollouts`, `missing_values`,
weights dtype and source; the server prints one banner line with model, params, dtype, source and
load time.

## Investigation Trail

1. **Server.** `bolt.rs`/`safetensors.rs` from spike 005 (F16/BF16 decode added, loader over a byte
   slice), `dates.rs` and the `http_app`/page/e2e pattern from spike 004. `y` accepts `null` (the
   model's own NaN handling). The model is an `Arc` with no per-thread state, so concurrent calls
   share it and the forward runs on a blocking thread.
2. **Parity through the server, not just the model.** Peyton, 64 steps: max |Δ| vs the Python
   oracle **9.54e-7** across all nine quantiles; 365 steps (46 forwards): **1.91e-5**. Round trip
   20 ms in the release e2e test; 4 ms for air passengers from an external client.
3. **Where the time goes.** Variants on the 2048-point Peyton context (tiny / small, ms):
   plain loops 37.0 / 229; GEMM projections + FF + patch embedding 19.7 / 103; + attention
   scores/context via `gemm_blis` 18.6 / 98.5; + single-row projections as `dot8` instead of
   `gemv` 18.5 / 98.6; + rayon-parallel `blis::gemm` 17.8 / 91.9. Stage breakdown (tiny): patch
   embedding 1.3, **encoder 14.7**, decoder 2.3, head 0.1. The encoder's GEMMs alone cost 12.2 ms
   at spike 008's 65 GFLOP/s, so the forward is now kernel-bound; the attention GEMM bought 1 ms,
   not the 5 estimated, and threading barely engages because `blis/parallel.rs` keeps products
   under 64 MFLOP on ≤ 2–4 threads (`max_threads` ladder, line 110) and thin ones serial.
4. **Latency grid** (median of 5, tiny-f16 / small-f16): context 100 → 2.7 / 15.8 ms; 512 → 6.3 /
   35 ms; 2048 → 18.5 / 98 ms; horizon 365 at 2048 → 848 ms / 4.5 s (46 forwards).
5. **f16 weights.** Max |Δ| vs f32 on Peyton's nine quantiles 9.55e-4 = **0.11 % of the series
   std**; load 27 ms vs 29 ms. The e2e test bars it at 2 %.
6. **Sizes and cold start.** Base binary 7.0 MB; tiny f32 41.9 MB (40.5 stripped); **tiny f16
   24.4 MB (23.1)**; small f16 103.2 MB (101.8). Spawn → `initialize` reply → first forecast over
   stdio: tiny-f16 **33 / 52 ms** (64 / 85 ms on the first, cold-cache run); small-f16 180 / 280
   ms (347 / 448 cold). Load is dominated by the f32 decode and the weight transposes.
7. **Refusals proven** in the e2e test: unknown field, < 4 points, impossible date, horizon 0,
   all-null `y`, horizon > 64 without the flag (message names `allow_long_horizon`); with the flag
   the response carries the warning and `forwards: 46`.

## Results

**Verdict: VALIDATED.** One embedded zero-shot model, one stateless tool, native quantiles, exact
parity with the Python pipeline through the server, and Lambda-sized numbers for both weight sets.

| build | binary | cold start to first forecast | forward, 2048 pts | horizon 365 | parity vs Python |
|---|---|---|---|---|---|
| tiny f16 embedded (default) | 24.4 MB | 52 ms | 18.5 ms | 0.85 s | 1e-3 of std (f16) / 9.5e-7 (f32) |
| small f16 embedded | 103 MB | 280 ms | 98 ms | 4.5 s | same code, spike 006 parity |
| no embed (runtime path) | 7.0 MB | + weight read | – | – | – |

**Surprises**
- Once the projections use the packed GEMM, nothing else in the forward is worth optimising: the
  encoder is 80 % of the time and sits at the kernel's rate. The next step is spike 008's 8×12
  tile, not more routing.
- The crate's parallel GEMM is tuned for large square products and deliberately stays near-serial
  below 64 MFLOP, which is every GEMM in a 129-token transformer. A thin server on Lambda's 1–2
  vCPUs loses nothing; a desktop build would want a lower threshold or layer-level parallelism.
- f16 weights are free in accuracy terms (0.1 % of std) and halve the artifact; the decode costs
  nothing measurable at load.
- Cold start is dominated by transposing the weights at load (the spike keeps both layouts for
  the A/B); storing only the transposed copy would halve memory and cut load time.

**Signal for the build**
- `aprender-mcp-chronos` = this crate's `lib.rs` + `bolt.rs` + `safetensors.rs` + `dates.rs`;
  `build.rs` with `CHRONOS_EMBED_DIR`; default embed tiny-f16, small-f16 as a build option (its
  103 MB binary exceeds Lambda's 50 MB direct-upload zip; deploy via S3 or a container image).
- Keep the spike-004 shape (`ds`, `y`, `horizon`, `freq`) plus `allow_long_horizon`; `y` nullable.
  Return `yhat`/`yhat_lower`/`yhat_upper` as q50/q10/q90 and the full `quantiles` map.
- Drop the untransposed weight copies and the loop paths in the product; keep `dot8` for rows = 1.
- Report empirical coverage (spike 006) in the docs, not the nominal 80 %.
- Concurrency is free here (immutable `Arc` model); spike 010 is about the Prophet/NP server.
