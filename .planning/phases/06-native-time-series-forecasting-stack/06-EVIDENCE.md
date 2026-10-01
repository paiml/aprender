# Phase 06 — Host-Gated Measured Evidence (SC1 / SC4 / SC5)

Every number below was MEASURED by running the `just` recipe named in its own row on the
host described in the header. Nothing here is carried over from the spikes; where a spike
number exists it is shown only in the "spike" column, beside the measurement, and it never
decides a pass. A FAILING bar is recorded as FAIL with its number — this file does not
explain bars away.

CLAUDE.md Verification Discipline #2 ("never label a run by intent — prove the mechanism
engaged") is why every row carries the command, the profile, and a log path, and why the
mechanism proofs are quoted verbatim in §5 rather than asserted.

---

## 1. Provenance

| Fact | Value | How derived |
|---|---|---|
| Host arch | `arm64` | `uname -m` |
| Host triple | `aarch64-apple-darwin` | `rustc -vV` → `host:` |
| CPU | Apple M4 Pro, 14 logical CPUs | `sysctl -n machdep.cpu.brand_string` / `hw.ncpu` |
| OS | macOS 26.6.2 (build 25G83), Darwin 25.6.0 | `sw_vers`, `uname -r` |
| Toolchain | `rustc 1.93.0 (254b59607 2026-01-19)`, `cargo 1.93.0 (083ac5135 2025-12-15)` | `rustc -vV`, `cargo --version` |
| `just` | 1.46.0 | `just --version` |
| Commit | **`adc8a560a`** (`adc8a560a68184c6f40d6cdbad284892fdd1b896`), branch `gsd/phase-2-contract-gate` | `git rev-parse HEAD` |
| Date (UTC) | 2026-09-06 (runs between 22:57Z and 23:08Z) | `date -u` |
| Build profile | `--release` for every timing/size row; `cargo test --release` for SC5 | see each row's command |
| `CARGO_INCREMENTAL` | `0` — set in the recipes AND in `.cargo/config.toml` (`incremental = false`) | |

**This host is NOT `lambda-vector` and NOT the CI runner.** The bars below are aarch64
claims and are stated as such. CI is `[self-hosted, X64, Linux, clean-room]`
(`.github/workflows/ci.yml:96,516,988,1039`) and builds debug for the lib leg, so it asserts
PARITY and REFUSALS instead of these numbers (06-RESEARCH Open Question 5). On x86_64
`trueno::blis::gemm_blis` does not dispatch the spike-008 NEON microkernel, so neither the
latency numbers nor the f32 quantile margin transfer — see §7.

**WORKING-TREE CAVEAT (read before citing the commit alone).** The measurements ran with
`adc8a560a` checked out AND with pre-existing uncommitted modifications in the tree that
this plan did not create and did not commit (14 source files; `crates/aprender-forecast/`,
`crates/aprender-mcp-forecast/`, `contracts/forecast-tool-boundary-v1.yaml`,
`crates/aprender-core/tests/monorepo_invariants.rs`). They are review-hardening changes left
behind by an earlier pass over plans 06-06/06-07. So the exact bytes measured are
`adc8a560a` **plus** that delta, whose digest is recorded here so the state is reproducible:

```
git diff -- crates contracts | shasum -a 256
bd42a46dda0f0c660cc450b1c97d50f2c017d9111a6164a73272793b7dbcb2b6   (59 848 bytes)
```

Naming this is the point of Verification Discipline #2: a commit hash alone would have
described code that was not what ran.

---

## 2. The measured bars

| gate | bar | measured (min / median / max over N) | pass | spike | command | log |
|---|---|---|---|---|---|---|
| `chronos-gate` | rc 0, and both `test result:` lines contain `0 ignored` with >= 1 passed | 8 passed / **0 ignored**, 9 passed / **0 ignored**, rc 0 (N = 1) | **PASS** | – | `just chronos-gate` | `target/p06-chronos-gate-weights.log`, `target/p06-chronos-gate-forecast.log`, `target/p06-chronos-gate-server.log` |
| `chronos-embed-build` | embedded release binary **< 30 000 000 bytes** (SC4) | 24 671 472 / 24 671 472 / 24 671 472 bytes (N = 5 builds, byte-identical) | **PASS** (82.2 % of the bar) | 24.4 MB unstripped | `just chronos-embed-build` | `target/p06-chronos-embed-build.log` |
| `chronos-bench` | tiny-f16 forward at 2 048 context **< 100 ms** (SC4) | 18.0 / 18.1 / 18.2 ms (N = 3) | **PASS** (18 % of the bar) | 18.5 ms | `just chronos-bench` | `target/p06-chronos-bench.log` |
| `chronos-coldstart` | median exec → first forecast reply **< 150 ms** (SC4) | 52 / **53** / 91 ms (N = 5 spawns, first is cold-cache) | **PASS** | 52 ms (85 cold cache) | `just chronos-coldstart 5` | `target/p06-chronos-coldstart.log` |
| `forecast-bench` | 3 000-point Prophet fit + 365-step predict **< 2 s** total (SC1) | 0.212 / 0.212 / 0.213 s (N = 3) | **PASS** (11 % of the bar) | 0.215 s synthetic | `just forecast-bench` | `target/p06-forecast-bench.log` |
| `forecast-bench` (real series) | same bar, on the real 2 905-point Peyton Manning series | 0.31 / 0.49 / 0.65 s over the 5 Peyton windows (N = 5, 2 200–2 815 training points) | **PASS** | 1.41 s | `just mase-rolling-origin` (Prophet `secs` column, peyton rows) | `target/p06-mase-rolling-origin.log` |
| `forecast-pool-ratio` | sequential wall / concurrent wall **>= 2.0**, best of three (SC5) | 5.146 / 5.150 / **5.162** (N = 3 attempts) | **PASS** (2.6x the bar) | 3.9x | `just forecast-pool-ratio` | `target/p06-forecast-pool-ratio-{1,2,3}.log` |
| `mase-rolling-origin` | informational (D-16) — no bar | 17 windows, 4 series, 5 models; mean MASE 1.029 (NP-lite) … 2.017 (naive) | n/a | reproduces spike-006 to 3 dp | `just mase-rolling-origin` | `target/p06-mase-rolling-origin.log` |

Every "measured" cell above came from the recipe's own stdout on this host at this commit.
No cell was derived from the spike column.

---

## 3. SC5 — the pool ratio, and who asserts it

REVIEW-06-04, both cross-AI reviewers independently: **the >= 2.0 wall-clock ratio is
asserted by `just forecast-pool-ratio` and by NO unit test.** `pool_equality` asserts only
that responses are bit-identical under load (a correctness claim that holds on every host)
and PRINTS the ratio. Verified in the tree: `crates/aprender-mcp-forecast/src/lib.rs`
contains exactly one occurrence of the string `assert!(speedup`, and it is inside the doc
comment that explains why there is no such assertion — zero occurrences in code.

All three attempt lines, verbatim (each carries its own arch/profile/workers/cpus, which is
what makes a later comparison mean anything):

```
POOL SPEEDUP: 5.150x seq=7448ms conc=1446ms arch=aarch64 profile=release workers=4 cpus=14
POOL SPEEDUP: 5.162x seq=7447ms conc=1442ms arch=aarch64 profile=release workers=4 cpus=14
POOL SPEEDUP: 5.146x seq=7438ms conc=1445ms arch=aarch64 profile=release workers=4 cpus=14
```

Best attempt: **5.162x**, the second line above. `profile=release` is the mechanism proof —
a debug-profile ratio would be measuring the wrong binary and would mean nothing.

Each attempt also reported `test result: ok. 2 passed; 0 failed; 0 ignored` — the equality
assertions (8/8 and 16/16 bit-identical) passed on every attempt. A non-zero exit here would
have failed the recipe on the spot without retrying: a response differing under load is a
correctness failure, and the retries exist only for CPU throttling.

**Why this is 5.16x and not 06-06's 2.070x.** Different measurement, not a contradiction.
06-06 measured a **debug** build on the **synthetic** 1 000/500-point requests
(`POOL=1 → 1.002x` control vs `POOL=8 → 2.070x`). This is a **release** build on the real
**2 905-point Peyton** series (`FORECAST_POOL_SERIES=peyton`), where each fit is heavier and
the router mutex serialises more work, so the pool has more to recover. The `1.002x`
single-router control from 06-06 remains the thing that makes either number evidence rather
than an assertion.

---

## 4. Chronos gate — the weights it verified before it tested them

REVIEW-06-03 (verified MEDIUM-HIGH): `just chronos-gate` calls `just fetch-chronos-tiny`
UNCONDITIONALLY, never `if [ ! -f ... ]`. These are the sha256 lines that call printed on
this run, echoed into the gate's own stdout — the evidence that the weights the parity tests
then loaded are the pinned ones, not merely present ones:

```
  f32/config.json        278f0086733031635fb1c861cb01c1bad6477420c7fcb19381a2993e335785e0  pin 278f0086733031635fb1c861cb01c1bad6477420c7fcb19381a2993e335785e0
  f32/model.safetensors  75068728d376d2bec670379eeef4bfb4d24c0cfe24d957451f8d19b447030a32  pin 75068728d376d2bec670379eeef4bfb4d24c0cfe24d957451f8d19b447030a32
  f16/model.safetensors  f5dc2ef53533c8896bcb120a754c52c39d8917c15750a9e845192014dfa74a67  pin f5dc2ef53533c8896bcb120a754c52c39d8917c15750a9e845192014dfa74a67
      (spike 007 recorded f9a033b42bc516e17ae5756317cb946121afdb59c94b4acfcd30fef93317cd4c — same tensors, __metadata__ key order differs)
```

All three files were already present, so **nothing was downloaded and everything was
re-hashed** — which is exactly the direction REVIEW-06-03 cared about, and exactly the
direction the proposed CI mount would exercise.

The two armed summaries:

```
  aprender-forecast (bolt::parity + chronos::parity): test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 85 filtered out; finished in 73.11s
  aprender-mcp-chronos (--lib): test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 63.68s
CHRONOS GATE: PASS
```

`0 ignored` in both is the non-vacuity check: an armed weights test that skipped itself would
report `1 ignored` and fail the gate. The two positional filters on the first command both
matched (8 tests ran, 85 filtered out), which also settles the rejected multi-filter finding
empirically on cargo 1.93.0.

---

## 5. Mechanism proofs (Verification Discipline #2)

| Claim | Proof, quoted |
|---|---|
| the cold start timed the EMBEDDED release binary | server banner: `aprender-chronos: chronos-bolt-tiny (8652672 params, F16, embedded) loaded in 62 ms` — `source == embedded`, `dtype == F16` |
| the pool ratio measured a release build | `profile=release` inside the `POOL SPEEDUP:` line itself |
| the pool ratio measured the real Peyton series | `FORECAST_POOL_SERIES=peyton` is in the recipe and `seq=7447ms` for 8 requests is ~40x the synthetic-series wall |
| the bench measured the f16 model on the D-14 production routing | section header `### chronos-bolt-tiny — 8652672 params, weights F16, load 27 ms`; the parsed row is `+ single-row projections as contiguous dot8 instead of gemv (D-14 production)` |
| the size bar measured the linked artifact, not RSS | `wc -c < target/release/aprender-mcp-chronos` → 24 671 472 |
| the demo servers are up and answering MCP | `GET :8787/` and `GET :8788/` → HTTP 200 with `tools/call` in the body; `POST /mcp` `initialize` → `{"serverInfo":{"name":"aprender-forecast","version":"0.63.0"}}` and `{"name":"aprender-chronos",...}`; a live `tools/call` on :8788 returned a 12-step forecast with `quantiles` |

The full `chronos-bench` table for the f16 model, as measured (run 1 of 3):

```
### chronos-bolt-tiny — 8652672 params, weights F16, load 27 ms (parse + f32 decode + transposes)

| variant | forward ms (2048 ctx) | max abs Δ vs all-GEMM (q, 64 steps) |
|---|---|---|
| plain loops (spike 005 default) | 36.0 | 9.5e-7 |
| GEMM projections + FF + embedding, single rows via trueno gemv | 19.6 | 9.5e-7 |
| + attention scores/context via GEMM | 18.2 | 9.5e-7 |
| + single-row projections as contiguous dot8 instead of gemv (D-14 production) | 18.2 | 0.0e0 |

Stage breakdown (2048 ctx, all-GEMM, dot8 single rows): patch embedding 1.3 ms · encoder 14.4 ms · decoder (1 token) 2.4 ms · quantile head 0.1 ms

| context | horizon | forwards | predict ms (median of 5) |
|---|---|---|---|
| 100  | 12  | 1  | 2.6   |
| 100  | 64  | 1  | 2.6   |
| 100  | 365 | 46 | 192.1 |
| 512  | 12  | 1  | 6.2   |
| 512  | 64  | 1  | 6.1   |
| 512  | 365 | 46 | 336.3 |
| 2048 | 12  | 1  | 18.1  |
| 2048 | 64  | 1  | 18.0  |
| 2048 | 365 | 46 | 832.6 |
```

The `365`-horizon rows are why D-13 keeps long horizons behind `allow_long_horizon`: 46
forwards, not 1, and 0.83 s at 2 048 context.

The `chronos-coldstart 5` table, as measured:

```
| run | exec → initialize response ms | → forecast response ms |
|---|---|---|
| 1 | 69 | 91 |
| 2 | 33 | 53 |
| 3 | 33 | 53 |
| 4 | 33 | 53 |
| 5 | 33 | 52 |

median: initialize 33 ms, forecast 53 ms (n = 5)
```

Run 1 is the cold page cache; runs 2–5 are within 1 ms of each other. Reporting only the
median would have hidden the 91 ms first spawn, which is the number a Lambda cold start
would actually see.

---

## 6. D-16 — the rolling-origin accuracy table

`just mase-rolling-origin` — 4 series x 17 rolling origins, 5 models, `CHRONOS_MODEL_DIR`
armed. Measured here, beside spike-006's table for comparison.

**Measured (this run, commit `adc8a560a`, aarch64 release):**

| model | peyton | wp_log_r | air | retail | mean | MASE 1–64 | MASE 65+ | cov80 | width/σ | WQL3 | total s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| naive (last value) | 1.772 | 2.246 | 2.603 | 1.448 | **2.017** | 1.963 | 2.094 | – | – | – | 0.0 |
| seasonal naive | 2.117 | 2.075 | 1.463 | 1.116 | **1.693** | 1.672 | 2.211 | – | – | – | 0.0 |
| Prophet (Rust port) | 1.045 | 1.030 | 1.092 | 1.208 | **1.094** | 1.070 | 1.058 | 0.60 | 0.83 | 0.0343 | 4.7 |
| NeuralProphet-lite (Rust) | 1.021 | 1.026 | 1.055 | 1.013 | **1.029** | 1.011 | 1.083 | 0.66 | 0.86 | 0.0320 | 2.0 |
| chronos-bolt-tiny (zero-shot) | 1.242 | 1.802 | 0.980 | 0.928 | **1.238** | 1.193 | 1.640 | 0.65 | 0.95 | 0.0332 | 3.0 |

**Spike 006 (`sources/006-chronos-vs-prophet-holdout/RUN-OUTPUT.md`), same splits:**

| model | peyton | wp_log_r | air | retail | mean | MASE 1–64 | MASE 65+ | cov80 | width/σ | WQL3 |
|---|---|---|---|---|---|---|---|---|---|---|
| naive (last value) | 1.772 | 2.246 | 2.603 | 1.448 | **2.017** | 1.963 | 2.094 | – | – | – |
| seasonal naive | 2.117 | 2.075 | 1.463 | 1.116 | **1.693** | 1.672 | 2.211 | – | – | – |
| Prophet (Rust port) | 1.045 | 1.030 | 1.092 | 1.208 | **1.094** | 1.070 | 1.058 | 0.60 | 0.83 | 0.0343 |
| NeuralProphet-lite (Rust) | 1.021 | 1.026 | 1.055 | 1.013 | **1.029** | 1.011 | 1.083 | 0.66 | 0.86 | 0.0320 |
| chronos-bolt-tiny (zero-shot) | 1.242 | 1.802 | 0.980 | 0.928 | **1.238** | 1.193 | 1.640 | 0.65 | 0.95 | 0.0332 |

**Every accuracy cell is identical to 3 decimal places.** The spike's numbers came from a
standalone driver with its own `bolt.rs`/`fit.rs`/`np.rs`; these come from the shipped
`aprender_forecast::{fit, np, prophet, chronos}` behind the crate's public API. That the two
agree bit-for-bit on 17 windows is the strongest available evidence that the monorepo port
did not change the numerics. The only differing column is `total secs` (3.0 s here vs 5.9 s
in the spike for Chronos), i.e. speed, not accuracy. The spike's `chronos-bolt-small` row is
absent: only the tiny weights are pinned in-tree.

Cross-check against the Python oracle (`tests/fixtures/chronos_holdout_oracle.json`,
`chronos-forecasting` 2.3.1): **17 forecasts compared, max |MAE(Rust) − MAE(Python)| =
7.8e-3**; total Chronos time Rust 3.0 s vs torch 0.4 s.

What the table says, which is what the routing rule in the example's header documents:
Chronos wins on the monthly series (air 0.980, retail 0.928) and loses badly on daily data
past 64 steps (MASE 65+ = 1.640 vs 1.058 for Prophet), while inside 64 steps all three are
within 0.18. `model: auto` remains DEFERRED — the example documents the rule, it does not
implement a router.

---

## 7. What these numbers do NOT establish

1. **They are not x86_64 numbers.** No latency, size or ratio row transfers to the CI
   runner. `gemm_blis` does not dispatch the spike-008 NEON microkernel there.
2. **The f32 quantile margin is unmeasured off aarch64.** `quantiles_abs_f32 = 1.0e-6` was
   frozen on 9.54e-7 measured here (visible again in §5's bench table as `9.5e-7`) — about
   4.6 % of headroom. 06-05's `quantiles_abs_f32_nonaarch64 = 5.0e-6` is explicitly
   PROVISIONAL-UNMEASURED. This run did not exercise it and could not.
3. **`forecast-bench` measures a synthetic series.** The bar it enforces uses
   `--bench 1000 3000`'s generated series (0.212 s). The real-series row in §2 comes from a
   different command and is 1.5–3x slower; both are recorded rather than the flattering one.
4. **The demo pages have not been judged by a human yet.** §5 proves the servers answer
   `initialize` and `tools/call`; whether the page charts a forecast legibly and shows the
   `allow_long_horizon` refusal is the end-of-phase human check.
5. **One host, one session.** N >= 3 per timing row bounds run-to-run noise on THIS box; it
   says nothing about a different machine or a loaded one.

---

## 8. Reproducing this file

```bash
just chronos-gate                 # the D-18 gate; must print CHRONOS GATE: PASS
just chronos-embed-build          # < 30 MB
just chronos-bench                # < 100 ms   (run 3x for the min/median/max above)
just chronos-coldstart 5          # < 150 ms median
just forecast-bench               # < 2 s      (run 3x)
just forecast-pool-ratio          # >= 2.0, best of 3 (the recipe does the 3 itself)
just mase-rolling-origin          # informational
```

Each recipe exits non-zero when its bar is missed and names the number; none of them can
pass by reading a status through a pipe (`rc=$?` is on its own line in all of them — 10
captures, checked by plan 06-08 Task 1's verify).

`bashrs lint` on the seven extracted recipe bodies: **0 errors, 4 files with warnings**
(SC2046/SC2198/SC2047 on assignments and scalars — false positives; PERF002 on an intended
in-loop command substitution). Just recipes do not live under `scripts/`, so
`scripts/*.sh`-scoped lint gates do not reach them; this extraction is how they were linted.
