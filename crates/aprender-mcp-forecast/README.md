# aprender-mcp-forecast

A **thin, single-purpose MCP server**: one stateless `forecast` tool over
[`aprender-forecast`](../aprender-forecast), built on
[pmcp](https://github.com/paiml/rust-mcp-sdk) and deployable to pmcp.run. Part of the
[aprender](https://github.com/paiml/aprender) monorepo; it follows the
`aprender-mcp-setfit` template — one model per server, transport only, no numerics.

The request carries the series; the server fits and forecasts inside the call. There is
no model to train first and no artifact to store.

## Run it

```bash
# stdio — what Claude Code / Claude Desktop / Cursor spawn directly
aprender-mcp-forecast

# loopback HTTP: same-origin demo page at / and MCP streamable-http at /mcp
aprender-mcp-forecast --http 8765

# K concurrent fits in flight (default 8); --pool 1 is the pre-pool behaviour
aprender-mcp-forecast --http 8765 --pool 16

# timing table on synthetic series (a report, not a protocol path)
aprender-mcp-forecast --bench 1000 3000
```

Everything human-readable goes to **stderr**: stdout belongs to the protocol. `--http`
binds `127.0.0.1` only, with `AllowedOrigins::localhost()` — a dev/loopback surface, not
an exposed one.

## Sizing `--pool`

pmcp's streamable-HTTP router holds **one** `Arc<Mutex<Server>>` across the whole tool
future, and a fit is seconds of CPU inside `spawn_blocking`. With a single router those
seconds are spent holding the mutex, so concurrent requests wait on the lock rather than
on the CPU — measured in-tree at **1.002×** the sequential wall for eight simultaneous
fits. `--pool K` puts K independent routers behind a round-robin front handler; the same
eight fits then measure **2.070×** on a 14-core debug build, and the spike measured 3.9×
in release.

**K is the number of fits that can be in flight at once, so size it to the CPU and the
blocking-thread budget you are willing to give this process, not to your expected client
count.** Each in-flight fit is bounded by `MAX_POINTS` (20 000), `MAX_HORIZON` (3 650) and
the per-round L-BFGS iteration cap; `FIT_BUDGET_SECS` is a *cooperative* budget inspected
at round boundaries, **not** a hard wall-clock cap, so it is K and those three bounds —
not the budget — that bound the work a burst of requests can buy. A K far above your core
count buys queueing, not throughput. `--pool 1` restores the single-router behaviour.

Responses do not depend on load: every request seeds its own simulation (default 42) and
shares no fit state, and `pool_equality` asserts that eight (and sixteen) concurrent
responses are **bit-identical** to their sequential results.

## The tool

`forecast` takes parallel `ds` (dates, `YYYY-MM-DD`) and `y` arrays plus `horizon`, and
optionally `freq` (`D`/`W`/`MS`), `model`, `growth`, `cap`, `seasonality_mode`,
`interval_width`, `holidays`, `regressors`, `n_lags` and `seed`. It returns future `ds`,
`yhat`, `yhat_lower`, `yhat_upper`, `trend`, named `components`, `fit_seconds`,
`predict_seconds` and `diagnostics`.

The advertised schema is strict (`additionalProperties: false`), so an unknown key is
**refused, not ignored**.

## Exogenous inputs: `regressors` and `holidays`

Both arguments are accepted on **both** model arms. Switch `model` between `"prophet"` and
`"neuralprophet"` and send the same argument — the arm restrictions below are explicit
refusals with a named fix, never a silent drop.

### `regressors`

An array of objects, in **insertion order** — the order you send them is the order their
design columns are appended, and the Prophet 1.4.0 parity fixtures are asserted against it.

| Field | Required | Default | Meaning |
|---|---|---|---|
| `name` | yes | — | Becomes a response component key on the prophet arm. Unique within the request; must not collide with a generated column name or a reserved response key. At most 200 bytes. |
| `values` | yes | — | One value per HISTORY row **and** per HORIZON row, history first: `len(values) == len(ds) + horizon`, exactly. You supply the driver over the period being forecast, because the model has to read it there. |
| `mode` | no | `"additive"` | `"additive"` or `"multiplicative"`. Multiplicative is **prophet-only**. |
| `prior_scale` | no | `10.0` | Prior scale on the coefficient (Prophet's `holidays_prior_scale`). **Prophet-only.** |
| `standardize` | no | absent = Prophet's **auto** rule | `true`/`false` force it. Absent standardises the column *unless* its history values are exactly `{0, 1}` — a binary indicator is left as-is. Standardisation uses pandas `Series.std()` (ddof = 1) over history rows only. |

```json
{
  "ds": ["2024-01-01", "..."], "y": [10.0, "..."], "horizon": 14,
  "regressors": [
    { "name": "promo", "values": [0, 1, "... len(ds)+horizon values ..."] },
    { "name": "price", "values": ["..."], "mode": "additive", "prior_scale": 10.0 }
  ]
}
```

Ceilings: **200** regressors per request, and **25 000** regressor design feature cells
(`(rows + horizon) x n_regressors`) — where `rows` is the point count on `prophet` and the
**span in days** on `neuralprophet`, which reads the driver over an imputed daily grid.

### `holidays` on both arms

The same list of named events with optional `lower_window` / `upper_window` offsets now goes
to either model. What comes back:

| `model` | Components |
|---|---|
| `"prophet"` | one component per holiday **name**, plus a `holidays` roll-up |
| `"neuralprophet"` | one component per event **name** (grouped by name, so two entries sharing a name sum into one component, exactly as prophet does), plus a `holidays` roll-up. Requires `n_lags = 0`. |

Regressor components are prophet-only: that arm publishes one component per regressor name
plus the `extra_regressors_additive` / `extra_regressors_multiplicative` roll-ups. On
`neuralprophet` the drivers enter the single additive block and no per-regressor component
is emitted.

## Read the bands as measured, not as nominal

**The nominal 80 % interval covered 0.60 (Prophet) and 0.66 (NeuralProphet-lite) of
held-out points across 17 rolling-origin windows** (spike 006). The tool description says
so too, because a client that trusts the label rather than the measurement will
under-estimate its own risk.

## What it refuses

Every bound lives in the library, not here — the transport re-checks nothing, so a direct
library caller and an MCP client get identical answers. Refusals: fewer than 10 or more
than 20 000 points; a horizon outside 1…3 650; mismatched `ds`/`y` lengths; a date that
is not exactly ten ASCII bytes of `YYYY-MM-DD`, or is not a real calendar date; `ds` not
strictly ascending; non-finite or constant `y`; a `freq` other than `D`, `W` or `MS`;
`interval_width` outside (0, 1); logistic growth without a `cap`, or a `cap` not
exceeding `max(y)`.

Unknown `model`, `growth` and `seasonality_mode` values are refused with the accepted set
named in the message — the tool boundary **refuses, never defaults**, so a knob you set is
never silently ignored. Every refusal above has its own end-to-end test against a live
server, and the bounds themselves are asserted equal to
[`contracts/forecast-tool-boundary-v1.yaml`](../../contracts/forecast-tool-boundary-v1.yaml)
rather than written twice.

`model: "neuralprophet"` is live (plan 06-04): it adds an AR-Net over the last `n_lags`
values, supports `freq: "D"` only, and reports a residual-sd band rather than
NeuralProphet's quantile regression — the `diagnostics.band` field says so.

### Exogenous-input refusals, each with its fix

| Refused when | The limitation | The fix |
|---|---|---|
| `len(values) != len(ds) + horizon` | the driver must cover the rows being fitted *and* the rows being forecast; a short array would offset every future feature | send `points + horizon` values, history first |
| a value is non-finite | JSON `1e400` parses to infinity and the fit has no defined answer for it | send finite numbers (the message names the row index; it does not echo the value) |
| `mode` is not `"additive"` / `"multiplicative"` | the accepted set is closed and is named in the message | use one of the two |
| `prior_scale` outside `[1e-153, 1e150]` | the objective squares it into a denominator, so a smaller value underflows to `0.0` and the fit runs **zero** iterations with the regressor contributing exactly nothing — while the response still looks normal | omit it (default `10.0`) or stay in range |
| duplicate regressor `name` | the component map is keyed by name; the second would silently replace the first | give each a distinct name |
| `name` collides with a generated design-column or reserved response key | same overwrite, from the other direction | rename the regressor — the message says what it collided with |
| `name` over 200 bytes | the name is cloned per column, compared during dedup and becomes a JSON key | use a shorter label (the name is never echoed back; its *length* is the issue) |
| a column is constant over the history rows | it is collinear with the intercept and carries no information | drop the regressor |
| more than 200 regressors | the identifiability diagnostic is `O(K^3)` in the **count alone** — at 2 000 it is 96 % of the request | send fewer |
| `(rows + horizon) x n_regressors` over 25 000 | every fit iteration is `O(rows x K)` over that matrix, and the individual bounds do not bound their product | fewer regressors, shorter history, or shorter horizon — the message states which operand it used |
| `mode: "multiplicative"` with `model: "neuralprophet"` | NP-lite composes exogenous terms with one **additive** block; no multiplicative path exists or was measured | use `"additive"`, or `model: "prophet"` |
| `prior_scale` with `model: "neuralprophet"` | the NP trainer has one global weight decay and no per-regressor prior, so the field could not take effect | omit it, or `model: "prophet"` |
| a regressor on a **gappy** series with `model: "neuralprophet"` and `n_lags > 0` | with lags the model trains on an imputed **daily grid** and reads the driver on days you did not send — and there is no defensible value to invent (two defensible fill rules measured 10.48 apart on a series of scale 35.32; a garbage probe flipped the sign of both coefficients) | send a gap-free daily series, or set `n_lags` to 0 — at 0 the imputed value is unread by construction |
| `holidays` with `model: "neuralprophet"` and `n_lags > 0` | the event block and the AR path were not measured together | set `n_lags` to 0, or use `model: "prophet"` |

### `diagnostics.regressors` (prophet arm)

A prophet response that carried regressors gains two `diagnostics` keys. A response that did
not carry any has **neither key** — absent, not `null`, not `[]`.

`diagnostics.regressors` is one row per regressor: `name`, resolved `mode`, the `mu` / `std`
actually applied, `vif` (`null` means **withheld**, not zero), and `warning` — present only
when that column's interpretation is suspect (VIF over 10.0), absent when it is clean.

`diagnostics.regressors_identifiability` is the design-level sibling: `scope` (prose naming
exactly which columns were used), `condition_number` with `condition_number_warning` over
30.0, `ridge` / `regularized`, and `status` — `"ok"`, `"singular"` (a duplicated or exactly
collinear column; every `vif` and the condition number are `null`) or `"not_converged"` (an
eigenvalue loop hit its iteration cap; the condition number is `null`, the VIFs still stand).

**A warning is about the coefficient, not about the forecast.** It says the *attribution*
between correlated columns is unreliable; `yhat` and the bands are unaffected.

**Holiday indicator columns are outside the diagnostic's column set, and the `scope` string
says so.** It covers the trend proxy, the seasonality columns and the regressor columns. The
holiday block can reach 1 000 columns while the diagnostic is `O(n*K^2 + K^3)`, so including
them would make its cubic term the dominant cost of the request — and every collinearity the
spikes measured was regressor-against-seasonality or regressor-against-trend.

### Caution: AR absorption on the `neuralprophet` arm

**With `n_lags > 0`, a continuous driver competes with the AR term, so a regressor
coefficient from the neuralprophet arm is not a clean effect estimate** — the autoregressive
block can explain the driver's contribution out of its weight.

The measured instance: at `n_lags = 7` the `price` weight collapsed to **-0.0048** from
**-0.0493** lag-free (spike 014). That number is **illustrative of the magnitude and is
explicitly not a bound**: one series, one gap pattern, `n_lags ∈ {0, 7}`, additive mode, and
it moves with gap fraction, gap run length and driver volatility.

**No identifiability number — no VIF, no condition number — is computed on this arm, on
purpose.** AR absorption is a *training dynamic* rather than column collinearity, so a green
VIF would reassure about the wrong thing, and with no bound to calibrate against there is no
threshold to defend. The caution is prose, not a number. For an interpretable coefficient,
read it off the `prophet` arm, which does publish the diagnostic, or set `n_lags = 0`.

## Compatibility: what a version bump costs a caller

**Scoped, and measured.** A JSON / MCP caller bumps one pinned line plus the lock entry and
changes nothing else: every new field is optional with a serde default and
`additionalProperties: false` is unchanged, so a body written before these arguments existed
deserialises and runs as before. A **Rust** caller constructing `ForecastArgs` with an
exhaustive struct literal, or calling `prophet::predict` directly, does need a change —
adding a public field breaks exhaustive literals by Rust's own rules. Both halves are
observed by a compile probe, so the bare "one-line bump" claim is not made here.

## Deployment

The Lambda wrapper crate is **deferred**: each one adds a `bootstrap` `[[bin]]` to the
`FALSIFY-MONO-011` allowlist count, and cargo-pmcp's `*-lambda` discovery is already
ambiguous. `http_app` and `StreamableHttpServerConfig::stateless()` ship here so that
wrapper is a short copy when it is wanted.
