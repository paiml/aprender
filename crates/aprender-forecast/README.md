# aprender-forecast

Pure-Rust time-series forecasting for [aprender](https://github.com/paiml/aprender):
a port of Facebook **Prophet 1.4.0** (piecewise-linear / logistic / flat trend with
Laplace-prior changepoints, Fourier seasonality, holiday windows, additive and
multiplicative modes, MAP fit by L-BFGS, simulated-changepoint uncertainty) behind one
stateless entry point. NeuralProphet-lite (`model: "neuralprophet"`) and the Chronos-Bolt
zero-shot forward ship alongside it.

```rust
use aprender_forecast::{forecast, ForecastArgs};

let response = forecast(&ForecastArgs {
    ds: ds, y: y, horizon: 365,
    ..Default::default()
})?;
```

**Use the `..Default::default()` rest, not an exhaustive literal.** `ForecastArgs` is a
public struct that gains optional fields over time (`holidays`, `n_lags`, `regressors` …);
an exhaustive literal fails to compile on every such addition, while the rest form does not.
See [*Compatibility*](#compatibility-what-a-version-bump-actually-costs-you) below.

One call carries the series and the horizon; the fit runs inside the call. There is no
fit → artifact → forecast round-trip, and no model to store. `aprender-mcp-forecast`
wraps this crate as a thin MCP server.

## Read the bands as measured, not as nominal

**The nominal 80 % interval covered 0.60 (Prophet) and 0.66 (NeuralProphet-lite) of
held-out points across 17 rolling-origin windows** (spike 006). Report and consume the
*empirical* coverage: `yhat_lower`/`yhat_upper` are a roughly 60–66 % band, not an 80 %
one. This is a property of Prophet's uncertainty model on real series, not a defect in
the port — Python Prophet behaves the same way — but a number labelled 80 % that covers
0.60 is a number that will mislead someone.

## Why forecasting lives here and not in `realizar`

CLAUDE.md's Realizar-first rule sends all *inference* through `realizar`. Forecasting is
a documented exception class, for the same reason SetFit is one: a Prophet forecast **is
a fit**, there is no trained artifact to serve, and the only conformance-proven
implementation of these numerics is this crate's, measured against Python Prophet 1.4.0
on committed oracle fixtures. `aprender-mcp-forecast` owns the transport — tool schema,
routes, readiness — and calls `forecast()`; it re-implements nothing (OPS-03).

## Correctness bar

Not self-consistency: **parity with Python Prophet 1.4.0**. The ladder's load-bearing
rung, `prophet::parity::peyton_manning_objective_at_python_map`, evaluates the Rust
objective at Python's MAP on the Peyton Manning series and compares it to Python's own
unnormalised log posterior. The oracle fixtures, their provenance, their generating
environments and the commands that regenerate them are documented in
[`tests/fixtures/README.md`](tests/fixtures/README.md). They are committed; their absence
is a defect, and no test in this crate skips because one is missing.

## Bounds

The library — not the transport — is the door, so every caller gets the same refusals:
at least 10 and at most 20 000 points, a horizon of 1…3 650, strictly ascending unique
`YYYY-MM-DD` dates, finite non-constant `y`, `freq` in `D`/`W`/`MS`, and
`interval_width` strictly inside (0, 1). The L-BFGS fit is capped at 2 000 iterations per
round; `FIT_BUDGET_SECS` (15 s) is a **cooperative** round-boundary budget, not a hard
wall-clock cap.

Dates are `i64` days since the epoch via civil-date arithmetic — no calendar-library
dependency anywhere in the Phase 6 crates.

## Exogenous inputs: `regressors` and `holidays`

Two optional arguments carry outside information into the fit, and **both are accepted on
both model arms**. A caller switches `model` between `prophet` and `neuralprophet` and
changes nothing else about the argument (D-26, D-29) — the arm restrictions below are
refusals, never silent drops.

### `regressors` — caller-supplied numeric drivers

An array of objects. **Insertion order is load-bearing**: the regressor columns are
appended to the design in the order you send them, which is what the Python Prophet 1.4.0
column-for-column parity assertion compares against. (A name-keyed map was rejected for
exactly this reason — `serde_json::Map` is a `BTreeMap` and iterates key-sorted.)

| Field | Required | Default | Meaning |
|---|---|---|---|
| `name` | yes | — | Becomes a response component key on the prophet arm, so it must be unique in the request and must not collide with a generated design-column name or a reserved response key. At most 200 bytes. |
| `values` | yes | — | One value per HISTORY row **and** per HORIZON row, in `ds` order then future order: `values.len()` must equal `ds.len() + horizon` exactly. The future half is required because the model needs the covariate over the period it is forecasting. |
| `mode` | no | `"additive"` | `"additive"` or `"multiplicative"`. Multiplicative is **prophet-only**. |
| `prior_scale` | no | `10.0` (Prophet's own `holidays_prior_scale`) | The prior scale on this regressor's coefficient. **Prophet-only** — the neuralprophet trainer has one global weight decay and no per-parameter prior. |
| `standardize` | no | absent = Prophet's **auto** rule | `true` / `false` force the choice. Absent means auto: standardise the column *unless* its history values are exactly the two-element set `{0, 1}` — the binary carve-out, which leaves an indicator alone (`mu = 0`, `std = 1`). Standardisation is pandas `Series.std()`, **ddof = 1**, computed over **history rows only** (numpy's ddof = 0 is a silent ~0.17 % shift in every coefficient). |

Ceilings: at most **200** regressors per request (`MAX_REGRESSORS`), and at most **25 000**
regressor design feature cells (`(rows + horizon) x n_regressors`,
`MAX_REGRESSOR_DESIGN_COST`). `rows` is the caller's point count on the prophet arm and the
**span in days** on the neuralprophet arm, because that arm reads the regressor over an
imputed daily grid — inheriting the prophet operand would under-price a gappy series by
exactly `span / points`.

### `holidays` on both arms — one argument, two models

`holidays` is a list of named events with optional `lower_window` / `upper_window` day
offsets. The same list is now accepted by both models; what each publishes back:

| Arm | Components published |
|---|---|
| `prophet` | one component per holiday **name**, plus a `holidays` roll-up, plus `additive_terms` / `multiplicative_terms` as before |
| `neuralprophet` | one component per event **name** (grouped by name, not by column, and not by index — two `HolidayArg`s sharing a name sum into one component, matching prophet), plus a `holidays` roll-up. Requires `n_lags = 0`. |

Regressor components are **prophet-only**: that arm publishes one component per regressor
name plus the `extra_regressors_additive` / `extra_regressors_multiplicative` roll-ups. On
the neuralprophet arm the regressors enter the single additive block and no per-regressor
component is emitted.

### What the door refuses, and what to do about it

Every bound lives in the library, so a direct Rust caller and an MCP client get identical
answers. The door **refuses, never defaults** (D-21): a knob you set is never silently
ignored.

| Refused when | The limitation | The fix |
|---|---|---|
| `values.len() != ds.len() + horizon` | the covariate has to cover the rows being fitted *and* the rows being forecast; a short array would silently offset every future feature | send `points + horizon` values, history first |
| any value is non-finite | JSON `1e400` parses to infinity, and the fit's arithmetic has no defined answer for it | send finite numbers (the row index is named in the message; the value is not echoed back) |
| `mode` is neither `"additive"` nor `"multiplicative"` | the accepted set is closed and is named in the message | use one of the two |
| `prior_scale` outside `[1e-153, 1e150]` | the objective and gradient **square** it into a denominator, so a smaller value underflows to exactly `0.0` and the initial zero coefficients meet `0/0` — L-BFGS then runs zero iterations and the regressor contributes exactly nothing while the response still looks normal | leave it at the `10.0` default, or stay inside the representable range |
| two regressors share a `name` | the response component map is keyed by name, so the second would silently replace the first | give each a distinct name |
| `name` collides with a generated design-column name, a response component name or a reserved response key | same map-insert overwrite, from the other direction | rename the regressor (the message says which part of the response it collided with) |
| `name` exceeds 200 bytes | the name is cloned per design column, compared during component dedup, and becomes a JSON key | use a shorter label (the name is never echoed back — its *length* is what is at issue) |
| a column is constant over the history rows | a constant driver is collinear with the intercept and carries no information; standardising it divides by zero | drop the regressor |
| more than 200 regressors | the identifiability diagnostic factorises a K x K correlation matrix, `O(K^3)` in the **count alone** — at 2 000 regressors it is 96 % of the whole request | send fewer regressors |
| `(rows + horizon) x n_regressors` over 25 000 | every fit iteration is `O(rows x K)` over exactly that matrix, and the three individual bounds do not bound their product | send fewer regressors, a shorter history, or a shorter horizon (the message states which operand it used, so a caller refused on one arm and accepted on the other can see which arithmetic refused them) |
| `mode: "multiplicative"` with `model: "neuralprophet"` | NP-lite composes exogenous terms with a single **additive** block; there is no multiplicative path and none was ever measured | set `mode` to `"additive"`, or use `model: "prophet"` |
| `prior_scale` present with `model: "neuralprophet"` | the NP trainer applies one global weight decay and has no per-regressor prior, so there is no mechanism by which the field could take effect | omit the field, or use `model: "prophet"` |
| a regressor on a **gappy** series with `model: "neuralprophet"` and `n_lags > 0` | with lags the model trains on an imputed **daily grid** and reads a regressor value on every one of those days, including days you did not send — and there is no defensible value to invent (measured: two defensible fill rules 10.48 apart on a series of scale 35.32, with a garbage probe flipping the sign of both coefficients) | supply a gap-free daily series, or set `n_lags` to 0 — at `n_lags = 0` the imputed-day value is unread by construction, so the case does not arise |
| `holidays` with `model: "neuralprophet"` and `n_lags > 0` | the event block and the AR path were not measured together | set `n_lags` to 0, or use `model: "prophet"` |

### `diagnostics.regressors` — the identifiability report (prophet arm)

When a prophet request carries regressors, the response's `diagnostics` gains two keys.
When it does not, **the keys are absent** — not `null`, not `[]`. (The mechanism is "the
key is never constructed", deliberately: a `skip_serializing_if` that misfires emits
`"regressors": []` on every response and changes the recorded invariance signature.)

`diagnostics.regressors` is one row per regressor:

| Field | Meaning |
|---|---|
| `name`, `mode` | as sent (`mode` resolved, so the default reads back as `"additive"`) |
| `mu`, `std` | the standardisation constants actually applied, over the history rows |
| `vif` | variance-inflation factor for that column against the rest of the diagnostic's column set. `null` means the number is **withheld**, not zero |
| `warning` | present only when the column's interpretation is suspect (VIF over 10.0). **Absent when clean** — a key that is always present with a `null` is a key every consumer has to branch on |

`diagnostics.regressors_identifiability` is the design-level sibling: `scope` (prose naming
exactly which columns were used), `condition_number` (and `condition_number_warning` over
30.0), `ridge` / `regularized` (whether a ridge rung was needed to factorise), and `status`
— `"ok"`, `"singular"` (an exactly duplicated or collinear column; every `vif` and the
condition number are `null`), or `"not_converged"` (an eigenvalue loop hit its 100-iteration
cap; the condition number is `null`, the Cholesky-derived VIFs are unaffected and still
reported).

**A warning is about the coefficient, not about the forecast.** A high VIF says the
*attribution* between correlated columns is unreliable; `yhat` and the bands are unaffected.

**Holiday indicator columns are outside the diagnostic's column set, and the `scope` string
says so.** It is computed over the trend proxy `t`, the seasonality columns and the
regressor columns only. The holiday block can reach 1 000 design columns and this diagnostic
is `O(n*K^2 + K^3)`, so including them would make its cubic term the dominant cost of the
request — and every collinearity the spikes measured was regressor-against-seasonality or
regressor-against-trend.

### Caution: AR absorption on the neuralprophet arm

**With `n_lags > 0`, a continuous driver competes with the AR term, so a regressor
coefficient from the neuralprophet arm is not a clean effect estimate.** The autoregressive
block can explain a smooth covariate's contribution out of the regressor weight.

The measured instance: at `n_lags = 7` the `price` weight collapsed to **-0.0048** from
**-0.0493** lag-free (spike 014). That number is **illustrative of the magnitude and is
explicitly not a bound** — it is one series, one gap pattern, `n_lags ∈ {0, 7}`, additive
mode, and it moves with gap fraction, gap run length and driver volatility.

**No VIF and no condition number are computed on the neuralprophet arm, and that is
deliberate** (D-37). AR absorption is a *training dynamic*, not column collinearity, so a
green VIF there would reassure about the wrong thing; and with no bound to calibrate
against there is no threshold to defend. The caution is prose rather than a number on
purpose. If you need an interpretable coefficient, read it off the prophet arm — which does
publish the diagnostic — or set `n_lags = 0`.

## Compatibility: what a version bump actually costs you

The promise is **scoped**, and the scope is measured rather than asserted.

**A JSON / MCP caller bumps one pinned line plus the lock entry and changes nothing else.**
Every field this crate added is `Option<_>` with a serde default and
`#[serde(deny_unknown_fields)]` is unchanged, so a request body that predates the new
arguments deserialises and runs exactly as before (D-20). This half is verified by a compile
probe that builds an unchanged JSON-constructing consumer against the candidate revision.

**A Rust caller may need a change.** `ForecastArgs` is a public struct: a caller
constructing one with an **exhaustive struct literal** (no `..Default::default()` rest) will
fail to compile when a field is added — by Rust's rules, not by an oversight. The same probe
asserts that failure, because an unobserved caveat is not a measurement. This repository hit
it twice in this phase, at `crates/aprender-mcp-forecast/src/main.rs:71` and
`crates/aprender-forecast/examples/mase_rolling_origin.rs:244`. Direct callers of
`prophet::predict` are in the same position — see *Public API changes* below.

A bare "one-line bump" claim would be false for the Rust surface, so it is not made.

## Chronos-Bolt weights

Weights are **never committed** (D-18). The Chronos-Bolt zero-shot path reads
[`amazon/chronos-bolt-tiny`](https://huggingface.co/amazon/chronos-bolt-tiny) — **Apache-2.0**,
8.65 M parameters — at one pinned revision:

```
just fetch-chronos-tiny
```

| What | Value |
|---|---|
| Repo | `amazon/chronos-bolt-tiny` |
| Revision | `a0e552de83495b5c28c14c71c374f3e33280b340` |
| `f32/model.safetensors` sha256 | `75068728d376d2bec670379eeef4bfb4d24c0cfe24d957451f8d19b447030a32` (33.0 MB, upstream pin) |
| `f32/config.json` sha256 | `278f0086733031635fb1c861cb01c1bad6477420c7fcb19381a2993e335785e0` (1.1 KB, upstream pin) |
| `f16/model.safetensors` sha256 | `f5dc2ef53533c8896bcb120a754c52c39d8917c15750a9e845192014dfa74a67` (16.5 MB, **derived locally**) |

The recipe writes into `/models/chronos-bolt-tiny/`, which is root-anchored gitignored
(CB-510), and it **verifies on every run, not only on download** — a file that is already
present, cached, or mounted is re-hashed, and a mismatch exits non-zero naming the file.
The f32 shas pin *upstream provenance*; the f16 sha is re-derived from the verified f32 and
so pins *local derivation integrity* only. Spike 007 recorded
`f9a033b42bc516e17ae5756317cb946121afdb59c94b4acfcd30fef93317cd4c` for the same weights: the
two f16 files differ in exactly six bytes — the key order inside the `__metadata__` object —
and their tensor bodies are byte-identical. Newer `safetensors` writes those two keys in the
other order.

Two environment variables, two different jobs:

| Variable | Read at | Does |
|---|---|---|
| `CHRONOS_MODEL_DIR` | runtime **and** build time | The directory the model is loaded from. At build time its presence is what arms the weight-dependent tests: `build.rs` emits `cfg(chronos_weights)` only when `$CHRONOS_MODEL_DIR/model.safetensors` is a file, so without weights those tests are **counted, reasoned skips** (`N ignored`), never a silent green. |
| `CHRONOS_EMBED_DIR` | build time, in the server crate | Stages `model.safetensors` + `config.json` into `OUT_DIR` for `include_bytes!`, so the deployed binary carries its own weights (D-13). |

## Public API changes

Two items in the 0.63.0 line are **breaking for external callers** of this crate. Both were
raised by the Phase 6 incremental code review (`IN-02`, `IN-04`) and are recorded here
because a consumer reads the crate README, not a plan. The release-time capture belongs in
the repository-root `CHANGELOG.md`.

| Symbol | Change | Breaking? | Why |
|---|---|---|---|
| `prophet::feature_row` | Gained a fourth parameter, `hol_sets: &[HashSet<i64>]`, before `out` | **Yes** — the signature changed | The function used to answer "is `day` in this holiday's window at offset `off`?" with a linear `.any()` over `days`, i.e. `rows x holiday_columns x dates` comparisons per design build. The caller now hoists the membership-set construction out of the row loop and passes it in — `prophet::holiday_day_sets(spec)` builds exactly what the parameter wants. This is a measured design-build improvement (plan 06-11) and is deliberately **not** being reverted. |
| `safetensors::load` | **Removed** (`pub fn load(path: &str) -> Result<(Weights, String), String>`) | **Yes** — the item is gone | It had no remaining caller in `crates/` or `src/` and was deleted as dead code. `safetensors::load_bytes` remains and is what the Chronos path actually uses (`chronos::load_model_from_dir` reads the file and hands over bytes), so a caller that needs the old behaviour composes `std::fs::read` with `load_bytes`. |

**On `feature_row`'s index safety.** `hi` comes from `Column.holiday`, i.e. from the `cols`
argument, and it looks up `hol_sets`, a *different* argument that no type ties to it. Since
every `Design` field is `pub`, a caller can construct one whose `cols` and `spec.holidays`
disagree. That lookup is now **total** (`hol_sets.get(hi)`), so a mismatched slice yields a
zero column instead of an out-of-bounds panic raised inside this library. Note what that
does and does not buy: neither a panic nor a zero column is *correct output* for a caller
who built the slice wrong — the point is only that a library should not abort your process
over it. Build `hol_sets` with `holiday_day_sets` from the same `Spec` you pass in, which is
what both in-crate callers do.
