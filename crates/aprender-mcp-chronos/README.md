# aprender-mcp-chronos

A **thin, single-model MCP server**: one stateless `forecast` tool over the Chronos-Bolt
port in [`aprender-forecast`](../aprender-forecast), built on
[pmcp](https://github.com/paiml/rust-mcp-sdk) and deployable to pmcp.run. Part of the
[aprender](https://github.com/paiml/aprender) monorepo; it follows the
`aprender-mcp-setfit` template — one model per server, transport only, no numerics.

Zero-shot: there is nothing to train and nothing to store. The request carries the series,
the model has already seen enough series to guess yours, and the answer comes back in one
~18 ms forward. Same tool NAME and same response core as
[`aprender-mcp-forecast`](../aprender-mcp-forecast) — the shared shape is a test
(`shared_forecast_shape_holds_for_every_server`), not a convention.

## Run it

```bash
# stdio — what Claude Code / Claude Desktop / Cursor spawn directly
aprender-mcp-chronos

# loopback HTTP: same-origin demo page at / and MCP streamable-http at /mcp
aprender-mcp-chronos --http 8766

# cold-start table: spawns itself over stdio and times exec → initialize → first forecast
aprender-mcp-chronos --coldstart 5

# kernel-variant and latency tables on the embedded Peyton Manning series
aprender-mcp-chronos --bench models/chronos-bolt-tiny/f16
```

## The tool

`forecast` takes exactly `ds`, `y` and `horizon`, plus optional `freq` and
`allow_long_horizon`. Unknown fields are **refused**, never ignored
(`additionalProperties: false` is advertised in `tools/list`).

| field | type | notes |
|---|---|---|
| `ds` | `string[]` | `YYYY-MM-DD`, strictly ascending, unique. 4 … 20 000 points |
| `y` | `(number \| null)[]` | same length as `ds`; `null` is a genuine gap — the model handles it |
| `horizon` | `integer` | 1 … 64 direct; up to 1 024 with `allow_long_horizon` |
| `freq` | `string?` | `D` (default), `W`, `MS` |
| `allow_long_horizon` | `bool?` | see the horizon policy below |

The response carries `yhat` (= `quantiles["0.5"]`), `yhat_lower` (`0.1`), `yhat_upper`
(`0.9`), the full nine-quantile map, the future `ds`, `context_used`, `predict_seconds`,
an optional `warning`, and `diagnostics` (weights dtype and source, `forwards`,
`rollouts`, `missing_values`, `context_length`, `native_horizon`).

### Band honesty (D-16)

The q10–q90 band is **nominally** an 80 % interval. Measured on 17 rolling-origin windows
in spike 006, it covered **0.65** of held-out points. Read it as a ~65 % band. The tool
description says the same thing to every client that lists it.

### Horizon policy

Chronos-Bolt-tiny predicts **64 steps directly**. Past that it must roll its own forecast
forward and feed it back in, and the accuracy cost is measured, not assumed: spike 006 saw
**MASE 1.7 past step 64 against 1.1 within it** on daily data. So a horizon above 64 is
**refused** unless the caller passes `allow_long_horizon: true`, and a request that does
pass it gets a `warning` naming the number of rollouts. A 365-step forecast is 46 forwards.

The hard ceiling is 1 024 steps; `allow_long_horizon` does not lift it.

## Weights

| | |
|---|---|
| Repository | [`amazon/chronos-bolt-tiny`](https://huggingface.co/amazon/chronos-bolt-tiny) |
| Revision (pinned) | `a0e552de83495b5c28c14c71c374f3e33280b340` |
| `f32/model.safetensors` sha256 | `75068728d376d2bec670379eeef4bfb4d24c0cfe24d957451f8d19b447030a32` |
| `config.json` sha256 | `278f0086733031635fb1c861cb01c1bad6477420c7fcb19381a2993e335785e0` |
| `f16/model.safetensors` sha256 | `f5dc2ef53533c8896bcb120a754c52c39d8917c15750a9e845192014dfa74a67` (DERIVED locally from the verified f32, never downloaded) |
| License | **Apache-2.0** (the weights, as published by Amazon) |
| Fetch | `just fetch-chronos-tiny` — verifies both sha256s and refuses to overwrite a present file |

Weights are **never** downloaded at cold start. They are either compiled into the binary
at build time or read once, at startup, from a directory you name.

## Build modes

| mode | how | binary | cold start → first forecast | when |
|---|---|---|---|---|
| **tiny-f16 embedded** (default) | `CHRONOS_EMBED_DIR=$PWD/models/chronos-bolt-tiny/f16 cargo build --release -p aprender-mcp-chronos` | 24.4 MB (23.1 stripped) | 52 ms (85 cold cache) | the Lambda-zip deployable |
| small-f16 embedded | same, pointing at a Bolt-**small** directory | 103 MB (101.8) | 280 ms (448 cold) | **S3 / container image only** — 103 MB does not fit a 50 MB Lambda zip. Documented option, never the default |
| tiny-f32 embedded | same, pointing at `…/f32` | 41.9 MB (40.5) | ~55 ms | when you want the f32 parity numbers in the shipped binary |
| runtime path (no embed) | `CHRONOS_MODEL_DIR=$PWD/models/chronos-bolt-tiny/f16 cargo run -p aprender-mcp-chronos` | 7.0 MB | + one weight read | development, and any deploy that mounts weights |

f16 costs **0.11 % of the series std** against f32 (measured; the contract bar is 2 %) and
halves the binary. That is why the default embedded build is f16.

`CHRONOS_EMBED_DIR` (build) and `CHRONOS_MODEL_DIR` (runtime, and *also* what arms the
weight-dependent tests) are independent: embedding a model does not arm the parity suite,
because the oracle comparison needs the f32 weights **directory**.

### What the embedded path was measured doing (plan 06-07, debug build)

The table above is the **release** shape, which plan 06-08 measures. What is proven here is
the **mechanism**, on a debug build:

- `build.rs` staged `17 316 992` weight bytes + `1 120` config bytes into `OUT_DIR` —
  byte-for-byte the f16 directory it was pointed at.
- With `CHRONOS_EMBED_DIR` set and **`CHRONOS_MODEL_DIR` unset**, the server answered a
  64-step `forecast` over stdio from `include_bytes!` alone: `resolve_model` reported
  `source = embedded`, `dtype = F16`, 8 652 672 params. Cold start, median of 3:
  **32 ms to `initialize`, 1 402 ms to the first forecast** — debug, so read it as a
  mechanism check, not a latency claim.
- With `CHRONOS_EMBED_DIR` set and `CHRONOS_MODEL_DIR` **also** set, the embedded bytes
  still won (`source = embedded`) *and* the four parity tests ran (`0 ignored`). With the
  embed alone they went back to being counted skips. That is the independence D-13/D-18
  intend, demonstrated rather than asserted.

## Tests

```bash
# unarmed: the weights-free invariants run, the weight tests are COUNTED skips
cargo test -p aprender-mcp-chronos --lib

# armed: everything runs, including oracle parity through the server
CHRONOS_MODEL_DIR=$PWD/models/chronos-bolt-tiny/f32 cargo test -p aprender-mcp-chronos --lib
```

An unarmed run reports `N ignored` with the arming reason printed — never a silent green
(D-18). The parity bars are read from `contracts/chronos-bolt-parity-v1.yaml` and the
bounds from `contracts/forecast-tool-boundary-v1.yaml`; neither is written as a literal in
a test.

## Not here

**Chronos-2** (228 MB f16, ~0.5 s per forecast on one core, container image only) is a
deliberate deferral, not an oversight — it is a third size tier with a different deployment
story. So is the Lambda `bootstrap` wrapper crate: `aprender-mcp-setfit-lambda` is the
template when it lands.
