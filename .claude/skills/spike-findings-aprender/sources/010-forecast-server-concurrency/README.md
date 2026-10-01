---
spike: 010
idea: prophet-forecast-mcp
name: forecast-server-concurrency
type: standard
validates: "Given the spike-004 server, when 8 NeuralProphet and Prophet requests arrive concurrently over streamable-HTTP, then every response matches its sequential result and no fit is corrupted by another thread's autograd tape"
verdict: VALIDATED
related: [002, 004, 007]
tags: [mcp, pmcp, concurrency, autograd, throughput]
---

# Spike 010: Forecast Server Concurrency

## What This Validates

Given the spike-004 Prophet / NeuralProphet server (in-process, streamable-HTTP), when eight
distinct forecast requests arrive at once — four series × {Prophet, NeuralProphet with 30 lags} —
three rounds over, and then sixteen simultaneous NeuralProphet fits, then every response is
bit-identical to the same request run alone (the autograd tape is thread-local and per-request
seeds fix the RNG), and the wall time shows whether the fits ran in parallel.

## How to Run

```bash
CARGO_TARGET_DIR=../../../target cargo run --release        # pool = 1 (as shipped) then pool = 8
CARGO_TARGET_DIR=../../../target cargo run --release -- 1   # only the shipped configuration
```

## What to Expect

`RUN-OUTPUT.md`: sequential baseline per request, three concurrent rounds with identical-response
counts and wall time, the 16-fit stress, for each pool size.

## Investigation Trail

1. **Correctness under load holds.** 8/8 identical in each of three rounds; 16/16 in the stress;
   max |Δ yhat| = 0. Prophet (seeded interval simulation) and NeuralProphet (seeded init and
   shuffling, `clear_graph()` per step on a thread-local tape) are deterministic per request no
   matter what else runs on the reused `spawn_blocking` threads.
2. **But nothing ran in parallel.** Eight concurrent requests: 5.41 s wall against a 5.53 s
   sequential total (1.0×); sixteen 1.2-second fits: 19.4 s. The server processes one tool call
   at a time.
3. **Cause.** `http_app` hands pmcp's router one `Arc<tokio::sync::Mutex<Server>>` (the same shape
   the SetFit Lambda wrapper and `StreamableHttpServer` use); the POST handler locks it
   (`pmcp-2.19.3/src/server/streamable_http_server.rs:2094`, `:2122`) and the JSON-RPC dispatch
   awaits the tool future inside the guard, so `spawn_blocking` inside the tool buys nothing —
   the next request waits for the lock. pmcp's own comment (line 1566) calls it "one
   `Arc<tokio::sync::Mutex<Server>>` shared by every session".
4. **Fix, measured.** A pool of K independent pmcp routers, each over its own `Server`, behind a
   round-robin `fallback` handler that `oneshot`s the request into one of them (25 lines, `tower`).
   With K = 8: eight concurrent requests **1.44 s** (3.9×, the wall is the longest single fit,
   1.38 s); sixteen stress fits **3.16 s** vs 19.4 s; still 8/8 and 16/16 identical, so K servers
   on shared blocking threads do not disturb each other's tape either.

## Results

**Verdict: VALIDATED** for the question asked, with a transport finding the build must act on.

| configuration | 8 concurrent, wall | vs sequential (5.5 s) | 16 NP fits, wall | identical responses |
|---|---|---|---|---|
| as shipped (one `Arc<Mutex<Server>>`) | 5.41 s | 1.0× | 19.4 s | 8/8 ×3, 16/16 |
| pool of 8 routers, round-robin | 1.44 s | 3.9× | 3.16 s | 8/8 ×3, 16/16 |

**Surprises**
- The thing this spike was written to catch (tape corruption) never happened; the thing it was
  not written to catch (a global lock in the transport) showed up in the very first wall-time
  column. Concurrency probes need a throughput row, not just an equality row.
- The Chronos server (spike 007) has the same lock; it did not matter there only because each
  call is 18 ms. On Lambda (one request per container) neither matters.

**Signal for the build**
- Any long-running MCP server built on pmcp's streamable-HTTP router needs the pool (or an
  upstream pmcp change: take the lock only to route, not across the tool future). Size K to the
  blocking-thread budget; the tape-per-thread model has no other constraint.
- Keep per-request seeds; never share a fit across requests. The equality test here (JSON
  signature of `ds/yhat/bands/trend/components`) is the release check for determinism.
- Report this to paiml/pmcp with the table: it is a measurement, not an opinion.
