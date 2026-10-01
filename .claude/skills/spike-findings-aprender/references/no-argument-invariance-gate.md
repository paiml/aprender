# No-Argument Bitwise Invariance — the Release Gate

Forecast Coach consumes `aprender-forecast` as a git dependency pinned by tag, and made one
constraint non-negotiable: *"Our deployed results must reproduce to the last bit after the bump
with no new arguments passed."*

**The constraint is satisfiable, and cheaply.** Because Prophet's column order puts extra
regressors last (spike 011), regressor columns **append** — no existing column index moves, `k` is
unchanged with zero regressors, and nothing downstream (the objective, the gradient, the L-BFGS
path, the component roll-ups) sees a different matrix. **No-argument invariance falls out of the
column-order decision rather than needing to be engineered.**

What spike 012 delivers is the **harness**: a signature proven able to fail, and a captured
baseline at the current commit. Ship it alongside the feature and run it as the release gate for
every tag.

## Requirements

From the `forecast-exogenous-inputs` idea (`.planning/spikes/MANIFEST.md`):

- **Byte-identical when the new arguments are absent.** Every committed fixture must reproduce its
  pre-change `ForecastResponse` exactly after the plumbing lands. This is the consumer's
  acceptance gate, not a nice-to-have (CR constraint 1, 2026-09-20).
- **No new required arguments.** Every new field is `Option<_>` with a serde default;
  `#[serde(deny_unknown_fields)]` stays as it is.
- **One tag per release**; the consumer bumps one line plus the lock entry and builds `--locked`.

## How to Build It

### 1. The signature — bits, and the right field set

```rust
pub fn signature(r: &ForecastResponse) -> u64 {
    let mut h = Hasher::new();                 // FNV-1a 64
    h.str(&r.model);  h.str(&r.freq);
    h.bytes(&(r.n_history as u64).to_le_bytes());
    h.bytes(&(r.ds.len() as u64).to_le_bytes());
    for d in &r.ds { h.str(d); }
    h.f64s(&r.yhat);  h.f64s(&r.yhat_lower);  h.f64s(&r.yhat_upper);  h.f64s(&r.trend);
    h.bytes(&(r.components.len() as u64).to_le_bytes());
    for (k, v) in &r.components { h.str(k); json(&mut h, v); }
    json(&mut h, &r.diagnostics);
    h.finish()
}

pub fn f64(&mut self, v: f64) {
    let v = if v == 0.0 { 0.0 } else { v };    // normalise -0.0; every other bit pattern matters
    self.bytes(&v.to_bits().to_le_bytes());    // BITS, never a printed decimal
}
```

Two decisions carry the gate:

- **`fit_seconds` and `predict_seconds` are EXCLUDED.** They are wall-clock; a signature
  containing them can never be stable, which would make the gate vacuous in the opposite
  direction. Everything else is included.
- **f64s hash by `to_bits()`.** A formatted comparison silently accepts any change below the print
  precision.

`diagnostics` was checked and *is* fully deterministic (`forecast.rs:395-414`): no timings, and
`serde_json`'s default `Map` is a `BTreeMap`, so key order does not depend on insertion.

### 2. Three parts, because a baseline alone proves nothing

**A — determinism.** Run each case through the public `forecast()` door twice; signatures must
match. Eight cases span both models, linear / logistic / multiplicative growth, holidays with
windows, and AR lags.

**B — mutation proof.** Mutate the response and require the signature to *change*. Without this, a
signature function that returned a constant would produce the same green table. Three mutations,
then revert and require the clean signature back:

| mutation | detected |
|---|---|
| `yhat[0] += 1 ULP` (`8.56059168341344368e0 → …545e0`) | **yes** |
| `trend[last] += 1 ULP` | **yes** |
| one extra component key | **yes** |
| all reverted | back to `aa669c2352dd376a` |

**C — the mechanism test, not a proxy.** Rather than *assert* that appending columns is harmless,
run the spike-011 `splice` with an **empty** regressor list and compare the resulting `Design` and
forecast against the crate's own untouched `make_design` / `predict`, field by field, bit by bit.

```rust
if r == 0 { return; }   // splice() is a no-op at zero regressors — this is what C proves empirically
```

### 3. Running it as the release gate

```bash
CARGO_TARGET_DIR=../../../target cargo run --release --quiet > RUN-OUTPUT.md
```

Signatures are captured in `sources/012-no-arg-bitwise-invariance/baseline.json`. **The real
re-check is a build step, not a spike step** — re-run this after the regressor work lands in
`crates/aprender-forecast` and diff against the captured baseline.

## What to Avoid

- **A green invariance table with no falsification probe beside it.** It is indistinguishable
  from a broken harness. Part B is not optional.
- **Including timings in the signature.** Vacuous-always-red is as useless as vacuous-always-green.
- **Comparing formatted decimals.** See above.
- **Asserting inertness instead of measuring it.** Part C runs the real plumbing against the real
  untouched path; a comment saying "appending is harmless" is not a gate.
- **Treating a lone `budget_hit` difference as a regression.** `budget_hit` inside `diagnostics`
  is wall-clock dependent. It did not flip across these runs, but on a loaded machine it could,
  and it would read as a signature change. That is an environment result.
- **Assuming the case list can be written from the docs.** One planned case
  (`retail/neuralprophet/MS`) panicked with `Validation("neuralprophet supports freq D only")` —
  `forecast.rs:421-423`, pinned by `neuralprophet_refuses_non_daily_freq`. The refusal **was** the
  finding (it answers the change request's P5 Q3); the case was swapped for a second daily series.
- **Parsing the fixture CSVs naively.** They quote their fields (`"2007-12-10"`); strip quotes or
  the date validator refuses the row. Sort by `ds` and de-duplicate at load — `wp_log_R.csv` is
  not chronological.

## Constraints

**A — 8/8 identical on repeat** (`sources/012-no-arg-bitwise-invariance/RUN-OUTPUT.md`):

| case | signature | fit s |
|---|---|---|
| peyton/prophet/default | `aa669c2352dd376a` | 1.39 |
| air/prophet/multiplicative | `3053244dcb27492c` | 0.01 |
| retail/prophet/default | `73b171523eb3fa3b` | 0.19 |
| wp_log_R/prophet/logistic | `331d67a8f8c924bb` | 0.76 |
| peyton/prophet/holidays+windows | `e0a13f9a10e3c3cc` | 1.29 |
| peyton/neuralprophet/lag0 | `d5a583795591a3db` | 0.16 |
| peyton/neuralprophet/lag7 | `726043759dde45dc` | 0.47 |
| wp_log_R/neuralprophet/lag0 | `72528357591ad4d6` | 0.15 |

**C — inert at zero regressors** on `peyton_manning`, `retail_sales`, `air_passengers`: `Design`
identical (`k`, `cols`, `x`, `s_a`, `s_m`, `prior_scales` all bit-equal), `yhat` and `trend`
bit-identical, all 3 shared components bit-identical.

### What this does NOT prove

- **The uncertainty bands are outside part C.** The spike-011 prototype computes point estimates
  only, so `yhat_lower` / `yhat_upper` are not in the mechanism test. They *are* covered by parts A
  and B, which go through the real door. The band path resamples changepoints around a `yhat` that
  depends on `beta` and `X`, so **add the bands to the part-C comparison when the feature lands
  in-crate**.
- **The baseline is pinned to one commit.** It is evidence for the bump it was captured against,
  not a permanent certificate.

## Origin

Synthesized from spike: 012.
Source files available in: `sources/012-no-arg-bitwise-invariance/`
(README.md, `src/sig.rs` — the bit-exact signature, `src/main.rs` — the A/B/C driver and the eight
door cases, `src/regressors.rs` — the spike-011 prototype under test, `baseline.json` — captured
signatures, RUN-OUTPUT.md).
