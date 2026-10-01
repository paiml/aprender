//! THE validation door. The transport re-checks nothing: every bound, every refusal and
//! every default lives here (D-11), so the stdio server, the streamable-HTTP server and a
//! direct library caller all get identical answers.
//!
//! Ported from `sources/004-forecast-mcp-thin-server/src/lib.rs:174-275` (D-08). Every
//! refusal message is verbatim — plan 06-06's e2e cases string-match them.

// serde_json::json! expands to .unwrap() internally, and the expansion lands at file
// scope where a statement-level allow cannot reach it. Same precedent as
// aprender-mcp-setfit/src/lib.rs:29-32.
#![allow(clippy::disallowed_methods)]

use std::time::Instant;

use crate::dates::{format_ymd, future_days, parse_date};
use crate::events;
use crate::fit::{fit_prophet, FIT_BUDGET_SECS, MAX_ITERS_PER_ROUND};
use crate::np;
use crate::prophet::{
    auto_seasonalities, changepoint_count, make_design, predict, Growth, Holiday, Mode,
    Seasonality, Spec,
};
use crate::types::{
    ForecastArgs, ForecastError, ForecastResponse, MAX_HOLIDAY_COLUMNS, MAX_HOLIDAY_DATES,
    MAX_HOLIDAY_DATES_TOTAL, MAX_HOLIDAY_DESIGN_COST, MAX_HOLIDAY_NAME_LEN, MAX_HOLIDAY_WINDOW,
    MAX_HORIZON, MAX_LOGISTIC_CHANGEPOINT_LAMBDA, MAX_NP_TRAIN_COST, MAX_POINTS, MAX_REGRESSORS,
    MAX_REGRESSOR_DESIGN_COST, MAX_SPAN_DAYS, MIN_POINTS, REGRESSOR_CONDITION_NUMBER_WARN,
    REGRESSOR_PRIOR_SCALE_MAX, REGRESSOR_PRIOR_SCALE_MIN, REGRESSOR_VIF_WARN,
};

/// Response keys a regressor name must not shadow — the THIRD part of the collision
/// reserved set, and the only part that is not derivable from any column.
///
/// Parts (a) generated column names and (b) per-component names both come out of the spec,
/// so a check built from the spec alone finds them. These eleven do not: they are pushed by
/// [`crate::prophet::predict`] AFTER the per-component loop, or they are top-level
/// [`ForecastResponse`] fields, and neither is a `Column.component` value anywhere.
///
/// Why it matters that this is a refusal and not a rename: the response component map is
/// built by `components.insert(name, value)` — a MAP INSERT — so a regressor named
/// `additive_terms` is computed as its own component and then silently OVERWRITTEN by the
/// real aggregate that `predict` pushes afterwards. The regressor's own contribution
/// vanishes from the response with no error anywhere, which is the D-21 silent-ignore class
/// this phase exists to close (threat T-06.1-13).
///
/// The first five are component keys `predict` pushes unconditionally or on the regressor
/// path (`prophet.rs`'s `holidays`, `extra_regressors_additive`,
/// `extra_regressors_multiplicative`, `additive_terms`, `multiplicative_terms`); the last
/// six are top-level `ForecastResponse` fields a component key must not shadow in a consumer
/// that flattens the response. `the_reserved_response_keys_slice_is_not_empty` pins the list
/// so a later edit cannot empty it and make the check vacuous.
/// A finite `f64` as a JSON number, and ANYTHING ELSE as an explicit `null`.
///
/// `serde_json` already renders a non-finite `f64` as `null`, so an accidental infinity and
/// a DELIBERATE withholding produce the SAME wire bytes — and only one of them is an
/// intended statement. Routing every emitted number through here makes the null a decision
/// taken in one place instead of an accident that is invisible on the wire. The same defect
/// is on record in this project as CR-03 (`UpdateEvidence::table_hash` not injective over
/// inf/NaN), which is why the fix is a funnel rather than a comment.
fn finite_or_null(v: f64) -> serde_json::Value {
    if v.is_finite() {
        serde_json::Number::from_f64(v).map_or(serde_json::Value::Null, serde_json::Value::Number)
    } else {
        serde_json::Value::Null
    }
}

/// The WIRE spelling of a mode — the string the caller sent, or the default.
///
/// `format!("{mode:?}")` would emit `Additive`, which is the Rust spelling and not the one
/// the caller used or the schema advertises.
fn mode_wire_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Additive => "additive",
        Mode::Multiplicative => "multiplicative",
    }
}

pub(crate) const RESERVED_RESPONSE_KEYS: [&str; 11] = [
    "additive_terms",
    "multiplicative_terms",
    "holidays",
    "extra_regressors_additive",
    "extra_regressors_multiplicative",
    "trend",
    "yhat",
    "yhat_lower",
    "yhat_upper",
    "ds",
    "cap",
];

/// Fit and forecast in ONE stateless call (D-01: no fit -> artifact -> forecast round-trip).
///
/// # Errors
///
/// [`ForecastError::Validation`] for anything the caller can fix (shape, bounds, dates,
/// unsupported options); [`ForecastError::Internal`] for a failure inside a fit.
pub fn forecast(args: &ForecastArgs) -> Result<ForecastResponse, ForecastError> {
    // ---- validation (the transport re-checks nothing; this is THE door) ----
    if args.ds.len() != args.y.len() {
        return Err(ForecastError::Validation(format!(
            "ds has {} entries but y has {}",
            args.ds.len(),
            args.y.len()
        )));
    }
    if args.ds.len() < MIN_POINTS {
        return Err(ForecastError::Validation(format!(
            "need at least {MIN_POINTS} points, got {}",
            args.ds.len()
        )));
    }
    if args.ds.len() > MAX_POINTS {
        return Err(ForecastError::Validation(format!(
            "{} points exceeds max_points {MAX_POINTS}",
            args.ds.len()
        )));
    }
    if args.horizon == 0 || args.horizon > MAX_HORIZON {
        return Err(ForecastError::Validation(format!(
            "horizon must be 1..={MAX_HORIZON}, got {}",
            args.horizon
        )));
    }
    if args.y.iter().any(|v| !v.is_finite()) {
        return Err(ForecastError::Validation(
            "y contains a non-finite value".into(),
        ));
    }
    let ds: Vec<i64> = args
        .ds
        .iter()
        .map(|s| parse_date(s))
        .collect::<Result<_, _>>()?;
    if !ds.windows(2).all(|w| w[0] < w[1]) {
        return Err(ForecastError::Validation(
            "ds must be strictly ascending with no duplicates".into(),
        ));
    }
    // MAX_POINTS bounds how MANY points arrive, never how far apart they are, and the
    // NeuralProphet path materialises an imputed DAILY grid over the whole span — ten
    // points a millennium apart bought a 3.6-million-row grid. Bound the span too.
    let span_days = ds[ds.len() - 1] - ds[0] + 1;
    if span_days > MAX_SPAN_DAYS {
        return Err(ForecastError::Validation(format!(
            "ds spans {span_days} days, which exceeds max_span_days {MAX_SPAN_DAYS}"
        )));
    }
    let freq = args.freq.clone().unwrap_or_else(|| "D".into());
    let fut = future_days(ds[ds.len() - 1], args.horizon, &freq)?;
    let model_name = args.model.clone().unwrap_or_else(|| "prophet".into());
    let interval_width = args.interval_width.unwrap_or(0.8);
    if !(0.0 < interval_width && interval_width < 1.0) {
        return Err(ForecastError::Validation(
            "interval_width must be in (0, 1)".into(),
        ));
    }
    let seed = args.seed.unwrap_or(42);
    let (y_min, y_max) = args
        .y
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| {
            (a.min(*v), b.max(*v))
        });
    if y_min == y_max {
        return Err(ForecastError::Validation(
            "y is constant; nothing to fit (Prophet special-cases this too)".into(),
        ));
    }

    // An option that belongs to the OTHER model is refused, never silently dropped: the
    // worst failure class at this door is the caller getting a plausible answer to a
    // question they did not ask (D-11). `deny_unknown_fields` catches a misspelt key; only
    // this catches a well-formed key aimed at the wrong arm.
    if model_name == "prophet" && args.n_lags.is_some() {
        return Err(ForecastError::Validation(
            "n_lags is neuralprophet-only; set model to \"neuralprophet\"".into(),
        ));
    }
    if model_name == "neuralprophet" {
        // `holidays` is NOT in this list any more (D-29): it is one field serving both
        // models, and a holiday IS an event indicator column in each of them. `growth`,
        // `cap` and `seasonality_mode` stay, because they are genuinely prophet-only.
        for (name, present) in [
            ("growth", args.growth.is_some()),
            ("cap", args.cap.is_some()),
            ("seasonality_mode", args.seasonality_mode.is_some()),
        ] {
            if present {
                return Err(ForecastError::Validation(format!(
                    "{name} is prophet-only; set model to \"prophet\""
                )));
            }
        }
        // ---- the SCOPED opening: events are proven LAG-FREE and refused WITH LAGS ----
        //
        // D-30 put events in scope at `n_lags > 0` and made SC3's 10 % per-column recovery
        // bar binding "at every configuration that ships". MEASURED (plan 06.1-05 FINDING 1,
        // four seeds, learning rate selected by train loss in every cell): lag-free recovery
        // is 0.0528-0.0677, comfortably inside the bar — while at `n_lags = 7` under the
        // epoch budget THIS DOOR configures the worst per-column error runs 0.2191-0.4660,
        // and one run produced a NEGATIVE weight. The AR term can read a planted effect out
        // of the lags, so the split between the two terms is under-determined until the fit
        // converges, and it is the trailing columns of each window that pay for it.
        //
        // Three measured facts rule out simply buying convergence, and they are recorded
        // here because this refusal looks arbitrary without them:
        //   1. `np::door_epochs` at `n_lags > 0` is `min(auto_epochs(n), 320)`, and
        //      `auto_epochs` DECREASES with the point count — 110 at 1 200 points, 90 at
        //      2 400, 50 at 20 000. The 320 cap binds only at n <= 50 (`MIN_POINTS` is 10),
        //      so 320 — the only budget at which the bar was observed to hold — is one this
        //      door configures for no realistic series.
        //   2. Even at 320 the bar does not close: 2 400 points, seed 7, worst error 0.1663.
        //   3. Raising the budget collides with C-08, a SHIPPED bound. At 5 000 points and
        //      six event columns, 320 epochs prices at 191.7 % of `MAX_NP_TRAIN_COST` —
        //      requests the door accepts today would start refusing. A cost bound is a
        //      published surface; moving it to make an internal bar pass is the wrong
        //      direction of fix.
        //
        // So the combination is REFUSED rather than shipped behind a disclaimer. That is
        // D-21's own posture — refuse rather than silently ignore — applied to the
        // COMBINATION instead of to the whole arm, and it satisfies both halves of SC3 at
        // once: `holidays` on neuralprophet WORKS (lag-free) and keeps REFUSING with a
        // message naming the limitation (lagged).
        //
        // Placed HERE, at the door and above the dispatch, beside the bounds it travels
        // with — not buried in the arm — and before the holiday loop parses any date, since
        // a request that cannot proceed should not buy that work.
        // NON-EMPTY, not merely `is_some()`. `"holidays": []` configures zero event columns
        // and is priced at zero — every later use normalises emptiness the same way
        // (`as_deref().unwrap_or(&[])` in the bounds loop, `if holidays.is_empty()` at the
        // event-design build) — so refusing it denied `n_lags` to any client that always
        // serialises the key, and told them to drop a field they had already emptied.
        if args.holidays.as_deref().is_some_and(|h| !h.is_empty())
            && args.n_lags.is_some_and(|l| l > 0)
        {
            return Err(ForecastError::Validation(
                "holidays on model \"neuralprophet\" are supported only at n_lags = 0: at \
                 n_lags > 0 the event block and the autoregressive term are jointly \
                 identified only once the fit converges, and the per-column effect recovery \
                 that requires is NOT established at the epoch budget this door configures \
                 (worst per-column error 0.2191-0.4660 against a 0.10 bar, one run \
                 negative). Set n_lags to 0 to use holidays, or drop holidays to use lags"
                    .into(),
            ));
        }
    }

    // ---- D-33: HOISTED HOLIDAY BOUNDS — one site, BOTH arms ----
    //
    // All four holiday bounds live HERE, above `match model_name`, so the neuralprophet
    // arm pays every one of them. Bounds first, acceptance second: hoisting after the
    // prophet-only refusal was dropped would have opened a window in which a ~200-byte
    // request buys an unbounded event surface.
    //
    // Plan 06.1-06 Task 2 has since removed that refusal (D-29), so the neuralprophet
    // branch of the design-cost operand below is LIVE and is what refuses real requests —
    // it is not retained dead code. The four bounds were re-mutated in the opened scope
    // (CLAUDE.md Verification Discipline rule 4: the old proof does not transfer).
    //
    // The ORDER inside the loop is unchanged and load-bearing. The name-length check is
    // FIRST because every other message in the loop formats `h.name` back to the caller.
    let mut holidays = Vec::new();
    // `prophet::columns` emits one design column per offset in the window, so an
    // unbounded window is an unbounded column count from a ~200-byte request.
    let mut holiday_columns = 0usize;
    let mut holiday_dates_total = 0usize;
    for h in args.holidays.as_deref().unwrap_or(&[]) {
        // FIRST in the loop, deliberately (C-07). `prophet::columns` clones this
        // name TWICE per design column and then makes it the key of an O(C log C)
        // byte-wise comparison sort, so at MAX_HOLIDAY_COLUMNS one payload
        // occurrence is amplified ~2 000:1 — and nothing bounded it at HEAD. Being
        // first also bounds every OTHER refusal message in this loop, each of which
        // formats `h.name` back to the caller.
        //
        // BYTES, not chars: `String::len` is bytes, and bytes are exactly what the
        // clone and the comparison cost. Do NOT "fix" this to `chars().count()` —
        // `a_holiday_name_whose_char_count_fits_but_whose_byte_length_does_not_is_refused`
        // is the test that catches that rewrite.
        //
        // The message names the POSITION and the LENGTH and never the name itself:
        // echoing an oversized string back re-materialises the very bytes the bound
        // refuses and reflects attacker-controlled content into logs (T-06-38).
        if h.name.len() > MAX_HOLIDAY_NAME_LEN {
            return Err(ForecastError::Validation(format!(
                "holiday at index {}: name is {} bytes, which exceeds \
                     max_holiday_name_len {MAX_HOLIDAY_NAME_LEN}; use a shorter label \
                     (the name is not echoed back — its LENGTH is what is at issue)",
                holidays.len(),
                h.name.len()
            )));
        }
        // ---- the RESERVED-KEY rule, for the OTHER caller-controlled name family ----
        //
        // A holiday name becomes a response component key on BOTH arms — `Column.component`
        // via `prophet::columns` on prophet, and the D-31 per-event insert on neuralprophet.
        // The component map is built with `components.insert(name, value)`, a MAP INSERT, so
        // a holiday named `trend` is computed as its own component and then silently
        // OVERWRITES (or is overwritten by) the server-owned one, with no error anywhere.
        // Measured before this check existed: `{"model":"neuralprophet","holidays":[{"name":
        // "trend",...}]}` returned `components.trend` as the event indicator — all zeros away
        // from the holiday date — in place of the real trend series `[16.016, 16.110, ...]`.
        //
        // That is the D-21 silent-ignore class (threat T-06.1-13) that `RESERVED_RESPONSE_KEYS`
        // was introduced to close. It was applied to regressor names only; holidays are the
        // other family the caller names, so the same rule is paid here. Placed AFTER the
        // length bound so an oversized name is still refused without being echoed back.
        if RESERVED_RESPONSE_KEYS.contains(&h.name.as_str()) {
            return Err(ForecastError::Validation(format!(
                "holiday at index {} is named {:?}, which is already a reserved response \
                 key; the component map is keyed by name and one would silently overwrite \
                 the other, so rename the holiday",
                holidays.len(),
                h.name
            )));
        }
        if h.lower_window > 0 || h.upper_window < 0 {
            return Err(ForecastError::Validation(format!(
                "holiday {:?}: lower_window ≤ 0 ≤ upper_window",
                h.name
            )));
        }
        if h.lower_window < -MAX_HOLIDAY_WINDOW || h.upper_window > MAX_HOLIDAY_WINDOW {
            return Err(ForecastError::Validation(format!(
                "holiday {:?}: windows must be within ±{MAX_HOLIDAY_WINDOW} days",
                h.name
            )));
        }
        if h.dates.len() > MAX_HOLIDAY_DATES {
            return Err(ForecastError::Validation(format!(
                "holiday {:?}: {} dates exceeds max_holiday_dates {MAX_HOLIDAY_DATES}",
                h.name,
                h.dates.len()
            )));
        }
        // Each term is now <= MAX_HOLIDAY_DATES; the running sum is refused HERE,
        // the moment the aggregate ceiling is passed, so the remaining holidays'
        // dates are never parsed and never allocated (WR-03). The post-loop check
        // below is KEPT: this one can only report a PARTIAL sum, and a partial sum
        // reported as "the total" would be a false statement in an error message.
        holiday_dates_total += h.dates.len();
        if holiday_dates_total > MAX_HOLIDAY_DATES_TOTAL {
            return Err(ForecastError::Validation(format!(
                "holidays carry {holiday_dates_total} dates in the first {} holidays \
                     alone (a running total, not the request's total), which already \
                     exceeds max_holiday_dates_total {MAX_HOLIDAY_DATES_TOTAL}; send \
                     fewer holidays or fewer dates per holiday",
                holidays.len() + 1
            )));
        }
        // Both windows are now within ±MAX_HOLIDAY_WINDOW, so the width is small
        // enough that this sum cannot overflow before the ceiling refuses it.
        holiday_columns += (h.upper_window - h.lower_window + 1) as usize;
        if holiday_columns > MAX_HOLIDAY_COLUMNS {
            return Err(ForecastError::Validation(format!(
                "holiday windows expand to more than max_holiday_columns \
                     {MAX_HOLIDAY_COLUMNS} design columns"
            )));
        }
        let days: Vec<i64> = h
            .dates
            .iter()
            .map(|s| parse_date(s))
            .collect::<Result<_, _>>()?;
        holidays.push(Holiday {
            name: h.name.clone(),
            days,
            lower_window: h.lower_window,
            upper_window: h.upper_window,
            prior_scale: 10.0,
        });
    }
    // ---- the PRODUCT bounds, at THE door and BEFORE make_design ----
    //
    // Every per-holiday bound above has now fired with its own message, so
    // `holiday_columns <= MAX_HOLIDAY_COLUMNS` and each `dates.len() <=
    // MAX_HOLIDAY_DATES`. What was never checked is what they multiply to.
    // `FIT_BUDGET_SECS` cannot cover it: it is a COOPERATIVE ROUND-BOUNDARY
    // budget, so it is structurally blind to `make_design` (which runs below,
    // before `fit_prophet` is entered) and it overshoots by a whole round inside
    // the fit — a 20 000-point, 1 000-column request measured 70.089 s against a
    // 15 s budget. The refusal has to be HERE.
    // KEPT alongside the in-loop refusal above (WR-03). The in-loop one fires
    // early and therefore knows only a RUNNING total; this one has seen every
    // holiday, so it is the only one that can honestly report the request's EXACT
    // total. Both are O(1) and they say different true things. Unreachable for a
    // request whose sum crosses the ceiling mid-loop — which is exactly why the
    // position test asserts on the MESSAGE and not on the constant's presence.
    if holiday_dates_total > MAX_HOLIDAY_DATES_TOTAL {
        return Err(ForecastError::Validation(format!(
            "holidays carry {holiday_dates_total} dates in total, which exceeds \
                 max_holiday_dates_total {MAX_HOLIDAY_DATES_TOTAL}; send fewer holidays \
                 or fewer dates per holiday"
        )));
    }
    // ONE ceiling, TWO operands, selected by arm (D-33).
    //
    // On prophet the design rows are the caller's POINT COUNT: `make_design` builds one
    // row per supplied `ds` entry plus one per horizon step, and a gap in `ds` costs
    // nothing because no row is materialised for it.
    //
    // On neuralprophet the design rows are the SPAN IN DAYS, because the event design is
    // built over the IMPUTED DAILY GRID that `np::NpData::new` materialises over
    // `first..=last` — at least as long as the point count, and longer on every gappy
    // series. Inheriting the prophet operand verbatim would knowingly under-price the
    // gappy NP case by exactly `span / points`, closing one under-pricing while opening
    // another. `span_days` is already computed above for the `MAX_SPAN_DAYS` check, so
    // this reuses that value rather than recomputing it.
    //
    // NOT a second constant: a `MAX_NP_EVENT_DESIGN_COST` would be a second ceiling, a
    // second contract row and a second re-derivation whenever either moves.
    //
    // `ds.len() <= MAX_POINTS` (20 000), `span_days <= MAX_SPAN_DAYS` (also 20 000),
    // `args.horizon <= MAX_HORIZON` (3 650) and `holiday_columns <=
    // MAX_HOLIDAY_COLUMNS` (1 000) are all already refused above, so this product is at
    // most 23 650 000 under EITHER operand — six orders of magnitude below `usize::MAX`
    // on every supported target. A plain multiply therefore cannot overflow here, and
    // `saturating_mul` would only obscure that the factors are bounded rather than add
    // safety.
    let (design_rows, design_operand) = if model_name == "neuralprophet" {
        // `span_days >= 1` and is bounded by MAX_SPAN_DAYS above, so the cast is exact.
        (span_days as usize, "span_days")
    } else {
        (ds.len(), "points")
    };
    let design_cells = (design_rows + args.horizon) * holiday_columns;
    if design_cells > MAX_HOLIDAY_DESIGN_COST {
        // The message STATES THE OPERAND IT USED, so a caller refused on one arm and
        // accepted on the other can see which arithmetic refused them (D-21).
        return Err(ForecastError::Validation(format!(
            "holidays expand to {design_cells} design feature cells \
                 (({design_operand} + horizon) x holiday_columns = ({design_rows} + {}) x \
                 {holiday_columns}), which exceeds max_holiday_design_cost \
                 {MAX_HOLIDAY_DESIGN_COST}; reduce the holiday windows, the number of \
                 holidays, the history length or the horizon",
            args.horizon
        )));
    }
    // With no holidays `holiday_columns` and `holiday_dates_total` are both 0, so
    // neither refusal above can fire and the no-holiday SC1 path is untouched.

    // ---- D-33's PATTERN, APPLIED AGAIN: the SPEC is assembled ABOVE the dispatch ----
    //
    // It is hoisted for ONE reason, and it is the reserved-set reason below: part (a) and
    // part (b) of the regressor name-collision check are built from `prophet::columns(&spec)`
    // and the `Column.component` values behind those columns, so a SHARED collision check
    // needs the spec. Rebuilding an equivalent spec at a second site would be two copies of
    // one assembly rule, which is the drift hazard the whole hoisting pattern exists to
    // close.
    //
    // Nothing about a PROPHET request changes. The relative order of every refusal below is
    // exactly what it was inside the arm, and the hoisted holiday bounds still fire first.
    // On the NEURALPROPHET arm `growth`, `cap` and `seasonality_mode` are already refused
    // above, so all three parse to their defaults here and the only cost is one
    // `auto_seasonalities` call and one holiday-list clone — paid so that a regressor name
    // is refused by the SAME rule on both arms rather than by two rules that can drift.
    let mode = match args.seasonality_mode.as_deref() {
        None | Some("additive") => Mode::Additive,
        Some("multiplicative") => Mode::Multiplicative,
        Some(o) => {
            return Err(ForecastError::Validation(format!(
                "seasonality_mode {o:?}: additive or multiplicative"
            )))
        }
    };
    let growth = match args.growth.as_deref() {
        None | Some("linear") => Growth::Linear,
        Some("logistic") => Growth::Logistic,
        Some("flat") => Growth::Flat,
        Some(o) => {
            return Err(ForecastError::Validation(format!(
                "growth {o:?}: linear, logistic or flat"
            )))
        }
    };
    // `cap` belongs to the LOGISTIC growth arm, not merely to the prophet MODEL:
    // `make_design` matches only `(Growth::Logistic, Some(c))` and falls to
    // `_ => None` for every other pair, so a cap sent on the linear, flat or
    // DEFAULTED arm was accepted and provably inert — the caller got a plausible
    // answer to a question they did not ask (D-11, FALSIFY-BOUNDARY-005). This
    // runs AFTER the `growth` enum is parsed, so an unknown growth string still
    // refuses with its own message first.
    if growth != Growth::Logistic && args.cap.is_some() {
        return Err(ForecastError::Validation(
            "cap is logistic-only; set growth to \"logistic\"".into(),
        ));
    }
    // The ONLY path that puts a `Some` here is the one below that validated it,
    // so `Spec` structurally cannot carry a cap the door did not check.
    let mut checked_cap: Option<f64> = None;
    if growth == Growth::Logistic {
        let cap = args
            .cap
            .ok_or_else(|| ForecastError::Validation("logistic growth needs cap".into()))?;
        // `y` is checked for finiteness above; `cap` was not, and JSON `1e400`
        // parses to `f64::INFINITY`, for which `cap <= y_max` is false. An infinite
        // cap makes every trend value infinite, which serialises as JSON `null`.
        if !cap.is_finite() {
            return Err(ForecastError::Validation(
                "cap must be a finite number".into(),
            ));
        }
        if cap <= y_max {
            return Err(ForecastError::Validation(format!(
                "cap {cap} must exceed max(y) = {y_max}"
            )));
        }
        checked_cap = Some(cap);
    }
    let mut spec = Spec::default_linear(auto_seasonalities(&ds, 10.0, mode));
    spec.growth = growth;
    spec.cap = checked_cap;
    // CLONED, not moved: the neuralprophet arm reads the same hoisted list below to
    // build its EventSpec list, which is what makes D-29's one-argument-two-models claim
    // literal.
    spec.holidays = holidays.clone();
    spec.holidays_mode = mode;
    spec.interval_width = interval_width;
    if spec.seasonalities.is_empty() && spec.holidays.is_empty() {
        // Prophet fits a 'zeros' column here; the design needs K >= 1 — give it a
        // harmless weekly term with a tiny prior.
        spec.seasonalities.push(Seasonality {
            name: "weekly".into(),
            period: 7.0,
            order: 1,
            prior_scale: 1e-3,
            mode,
        });
    }

    // ---- the HOLIDAY half of check 10, parts (a) and (b) ----
    //
    // The reserved-KEY half is paid in the holiday loop above, which is the earliest point
    // it can be. This half cannot be: it needs the assembled spec, so it waits until the
    // seasonalities (including the empty-design `weekly` fallback just above) are final.
    //
    // A holiday shares the component namespace with the seasonalities. `prophet::predict`
    // deduplicates components BY NAME, so a holiday named `weekly` does not overwrite the
    // weekly seasonality — it MERGES into it: one `components.weekly` is published carrying
    // the sum of both, the holiday never appears as its own component, and the roll-up takes
    // the mode of whichever column came first, so an additive holiday under
    // `seasonality_mode: "multiplicative"` is published without its `y_scale` factor.
    // Silent, and wrong in three ways at once.
    //
    // The set is built from a HOLIDAY-FREE view of the spec: `prophet::columns(&spec)`
    // includes the holiday columns themselves, so every holiday would collide with its own
    // name. This is the same two parts the regressor check pays — a generated design column
    // name and a response component name — restricted to the sources a holiday does not
    // itself produce.
    {
        let mut seasonal_only = spec.clone();
        seasonal_only.holidays = Vec::new();
        let mut reserved: std::collections::BTreeMap<String, &'static str> =
            std::collections::BTreeMap::new();
        for c in crate::prophet::columns(&seasonal_only) {
            reserved.insert(c.name, "a generated design column name");
            reserved.insert(c.component, "a response component name");
        }
        if let Some((i, h, part)) = spec
            .holidays
            .iter()
            .enumerate()
            .find_map(|(i, h)| reserved.get(h.name.as_str()).map(|part| (i, h, *part)))
        {
            return Err(ForecastError::Validation(format!(
                "holiday at index {i} is named {:?}, which is already {part} in the \
                 response; the component map is keyed by name and the two would be \
                 merged into one component, so rename the holiday",
                h.name
            )));
        }
    }

    // ---- SHARED REGRESSOR VALIDATION — one site, BOTH arms (D-22, D-33's pattern) ----
    //
    // Plan 06.1-01 Task 2 placed the length tie, the finiteness checks, the mode allowlist,
    // the prior-scale range, the duplicate-name check and the zero-spread refusal INSIDE the
    // `"prophet" =>` arm, and plan 06.1-03 added the name-byte bound, the count ceiling, the
    // design-cost product and the three-part name-collision check in the same place. The
    // dispatch below separates the two execution paths, so deleting the neuralprophet
    // refusal without MOVING these first would open that arm onto a regressor surface with
    // no length tie, no finiteness guarantee, no mode allowlist, no duplicate-name check, no
    // name bound, no count ceiling and no design-cost ceiling — reopening, on the second
    // arm, every defect the first two waves closed on the first.
    //
    // Bounds first, acceptance second, in separate commits: hoisting AFTER the refusal was
    // dropped would open a window in which a ~200-byte request buys an unbounded regressor
    // surface. Plan 06.1-06 hoisted the four HOLIDAY bounds for exactly this reason and the
    // argument reaches here unchanged.
    //
    // Every message below is BYTE-IDENTICAL to the one that shipped inside the arm — they
    // are verbatim contracts that e2e cases string-match. The ONLY arm-dependent piece is
    // the design-cost OPERAND, which follows D-33's rule: the caller's point count on
    // prophet, the imputed daily grid's span in days on neuralprophet, because the NP
    // regressor design is read over that grid on the lagged path. The message STATES the
    // operand it used, as the holiday check does.
    //
    // ---- EXTERNAL REGRESSORS (D-22) ----
    //
    // Split each caller array into the HISTORY prefix (`ds.len()` rows, what the
    // standardisation constants are computed over) and the FUTURE tail (`horizon`
    // rows, what `predict` is handed). `fut` has length `horizon`, NOT
    // `ds.len() + horizon`, so handing `predict` the whole array would silently
    // offset every future feature — `predict` re-checks the length for that reason.
    let mut reg_std: Vec<crate::regressors::Standardized> = Vec::new();
    let mut reg_hist: Vec<Vec<f64>> = Vec::new();
    let mut reg_fut: Vec<Vec<f64>> = Vec::new();
    if let Some(regs) = args.regressors.as_ref() {
        // ---- 8. NAME LENGTH, in BYTES, FIRST in the block ----
        //
        // FIRST for the reason the holiday loop's identical check is first (C-07):
        // every OTHER refusal below formats caller data — six of them format the
        // index, two format the NAME — so an unbounded name would be reflected back
        // through whichever check fires. The length check bounds all of them.
        //
        // Reuses MAX_HOLIDAY_NAME_LEN rather than adding a fourth name ceiling: the
        // amplification argument is the SAME one C-07 records. `regressors::splice`
        // clones the name TWICE per design column (the `Column.name` and the
        // `Column.component`), `predict`'s component-name dedup compares it
        // O(C * distinct_components) times, and it becomes a key of the serialized
        // `components` map. One ceiling, one derivation, one place to raise it.
        //
        // BYTES, not chars: `String::len` is bytes and bytes are what the clone and
        // the comparison cost. The message names the INDEX and the LENGTH and NEVER
        // the name — echoing an oversized string back re-materialises the very bytes
        // the bound refuses and reflects attacker-controlled content into logs
        // (T-06.1-14).
        if let Some((i, r)) = regs
            .iter()
            .enumerate()
            .find(|(_, r)| r.name.len() > MAX_HOLIDAY_NAME_LEN)
        {
            return Err(ForecastError::Validation(format!(
                "regressor at index {i}: name is {} bytes, which exceeds \
                 max_holiday_name_len {MAX_HOLIDAY_NAME_LEN}; use a shorter label \
                 (the name is not echoed back — its LENGTH is what is at issue)",
                r.name.len()
            )));
        }
        // ---- 7. THE COUNT CEILING, MEASURED (replaces the wave-1 interim 50) ----
        //
        // Before any allocation proportional to the payload. `fit_max_regressors` is
        // its own ceiling rather than a consequence of the product below, because
        // the product is satisfiable by trading rows for columns while the
        // identifiability diagnostic's O(K^3) factorisation is a function of the
        // COUNT alone (cost axis C-17).
        if regs.len() > MAX_REGRESSORS {
            return Err(ForecastError::Validation(format!(
                "the request carries {} regressors, which exceeds \
                 max_regressors {MAX_REGRESSORS}; send fewer regressors",
                regs.len()
            )));
        }
        // ---- 9. THE DESIGN-COST PRODUCT, and it STATES ITS OPERANDS ----
        //
        // `ds.len() <= MAX_POINTS` (20 000), `args.horizon <= MAX_HORIZON` (3 650)
        // and `regs.len() <= MAX_REGRESSORS` are all already refused above, so this
        // product is at most 4 730 000 — six orders of magnitude below `usize::MAX`
        // on every supported target. A plain multiply cannot overflow here, and
        // `saturating_mul` would only obscure that the factors are bounded.
        //
        // ONE ceiling, TWO operands, selected by arm — D-33's rule, the same one the
        // hoisted holiday design-cost check above follows, and for the same reason.
        // On prophet the design rows are the caller's POINT COUNT: `regressors::splice`
        // writes one cell per supplied `ds` entry and `predict` one per horizon step,
        // and a gap in `ds` costs nothing because no row is materialised for it. On
        // neuralprophet the rows are the SPAN IN DAYS, because the regressor design is
        // read over the IMPUTED DAILY GRID `np::NpData::new` materialises over
        // `first..=last` on the lagged path — at least as long as the point count, and
        // longer on every gappy series. Inheriting the prophet operand verbatim would
        // knowingly under-price the gappy NP case by exactly `span / points`.
        //
        // `span_days` is already computed above for the `MAX_SPAN_DAYS` check and is
        // bounded by it, so the cast is exact and this product is at most 23 650 000
        // under EITHER operand — six orders of magnitude below `usize::MAX`.
        // The SAME arm operand the hoisted holiday design-cost check already selected
        // (D-33, :390), used DIRECTLY rather than through a rebinding alias. Re-deriving
        // it per ceiling is how two ceilings come to disagree about what an "NP row" is —
        // the drift the hoisting pattern exists to close — and an alias only hides the
        // shared-ness the identifiers are supposed to show.
        let regressor_cells = (design_rows + args.horizon) * regs.len();
        if regressor_cells > MAX_REGRESSOR_DESIGN_COST {
            // The message STATES THE OPERAND IT USED, so a caller refused on one arm
            // and accepted on the other can see which arithmetic refused them (D-21).
            return Err(ForecastError::Validation(format!(
                "regressors expand to {regressor_cells} design feature cells \
                 (({design_operand} + horizon) x n_regressors = ({} + {}) x {}), which \
                 exceeds max_regressor_design_cost {MAX_REGRESSOR_DESIGN_COST}; \
                 send fewer regressors, a shorter history or a shorter horizon",
                design_rows,
                args.horizon,
                regs.len()
            )));
        }
        // ---- 10. NAME COLLISION WITH A RESPONSE COMPONENT KEY ----
        //
        // The response component map is `components.insert(name, value)` — a MAP
        // INSERT — so a duplicate key silently OVERWRITES the earlier entry and the
        // operator sees ONE component where TWO were computed. Check 5 below closes
        // regressor-against-regressor; this closes regressor-against-everything-else,
        // over a reserved set with THREE parts (T-06.1-13):
        //
        //   (a) GENERATED COLUMN NAMES — `prophet::columns(&spec)`, i.e. the
        //       `{seasonality}_delim_{n}` and `{holiday}_delim_{sign}{offset}` forms.
        //   (b) PER-COMPONENT NAMES — the `Column.component` values behind those
        //       columns. `predict` pushes one component per DISTINCT `component`
        //       value, so a regressor named `yearly` collides even though `yearly`
        //       is not itself a generated column name; and a regressor named after a
        //       declared holiday collides for the same reason.
        //   (c) RESERVED RESPONSE KEYS — [`RESERVED_RESPONSE_KEYS`], which are
        //       invisible to (a) and (b) because they are pushed after the
        //       per-component loop or are top-level response fields.
        //
        // The spec is fully assembled by this point (seasonalities, holidays and the
        // empty-design weekly fallback are all settled above), so the reserved set is
        // built ONCE from the real spec rather than from a reconstruction of it.
        let mut reserved: std::collections::BTreeMap<String, &'static str> =
            std::collections::BTreeMap::new();
        for c in crate::prophet::columns(&spec) {
            reserved.insert(c.name, "a generated design column name");
            reserved.insert(c.component, "a response component name");
        }
        for k in RESERVED_RESPONSE_KEYS {
            reserved.insert(k.to_string(), "a reserved response key");
        }
        if let Some((i, r, part)) = regs
            .iter()
            .enumerate()
            .find_map(|(i, r)| reserved.get(r.name.as_str()).map(|part| (i, r, *part)))
        {
            return Err(ForecastError::Validation(format!(
                "regressor {i} is named {:?}, which is already {part} in the \
                 response; the component map is keyed by name and one would \
                 silently overwrite the other, so rename the regressor",
                r.name
            )));
        }
        // ---- 5. DUPLICATE NAMES ----
        //
        // O(R log R) via a BTreeSet, not a quadratic scan: the component map is
        // keyed by name, so the second of a pair would silently overwrite the first.
        let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for r in regs {
            if !seen.insert(r.name.as_str()) {
                return Err(ForecastError::Validation(format!(
                    "regressor name {:?} appears more than once; the response \
                     component map is keyed by name and the second would silently \
                     replace the first, so give each regressor a distinct name",
                    r.name
                )));
            }
        }

        for (i, r) in regs.iter().enumerate() {
            // ---- 1. LENGTH TIE, before anything is copied ----
            let want = ds.len() + args.horizon;
            if r.values.len() != want {
                return Err(ForecastError::Validation(format!(
                    "regressor {i} carries {} values but {want} are required \
                     (points + horizon = {} + {}); the array must cover the \
                     history rows AND the horizon rows",
                    r.values.len(),
                    ds.len(),
                    args.horizon
                )));
            }
            // ---- 2. INPUT FINITENESS ----
            //
            // The same class the `cap` finiteness check closed: JSON `1e400` parses
            // to infinity. The row index is named, the VALUE is not echoed.
            if let Some(bad) = r.values.iter().position(|v| !v.is_finite()) {
                return Err(ForecastError::Validation(format!(
                    "regressor {i} carries a non-finite value at row {bad}; every \
                     value must be finite (note that JSON 1e400 parses to infinity)"
                )));
            }
            // ---- 3. MODE ALLOWLIST, refused BY NAME ----
            let mode = match r.mode.as_deref() {
                None | Some("additive") => Mode::Additive,
                Some("multiplicative") => Mode::Multiplicative,
                Some(other) => {
                    return Err(ForecastError::Validation(format!(
                        "regressor {i} mode {other:?} is not supported; use \
                         \"additive\" or \"multiplicative\""
                    )))
                }
            };
            // ---- 4. PRIOR SCALE, in a NUMERICALLY USABLE range ----
            //
            // Not merely "> 0": the objective and gradient SQUARE this into a
            // denominator (prophet.rs:536, :685), so f64::MIN_POSITIVE squares to
            // exactly 0.0 and the initial zero coefficients meet 0.0/0.0.
            let prior_scale = r.prior_scale.unwrap_or(10.0);
            if !prior_scale.is_finite()
                || prior_scale < REGRESSOR_PRIOR_SCALE_MIN
                || prior_scale > REGRESSOR_PRIOR_SCALE_MAX
            {
                return Err(ForecastError::Validation(format!(
                    "regressor {i} prior_scale {prior_scale:e} is outside the \
                     usable range [{REGRESSOR_PRIOR_SCALE_MIN:e}, \
                     {REGRESSOR_PRIOR_SCALE_MAX:e}]; the objective squares it into \
                     a denominator, so a smaller value underflows to zero and \
                     yields a NaN fit"
                )));
            }

            let spec_r = crate::regressors::RegressorSpec {
                name: r.name.clone(),
                mode,
                prior_scale,
                standardize: r.standardize,
            };
            let (hist, futv) = r.values.split_at(ds.len());
            let st = crate::regressors::standardize_one(&spec_r, hist);

            // ---- 6a. ZERO SPREAD ----
            //
            // Only when the column was actually STANDARDISED: the auto {0,1}
            // carve-out legitimately returns std = 1.0 and must not be caught here.
            if st.std == 0.0 {
                return Err(ForecastError::Validation(format!(
                    "regressor {i} is constant over the history rows and carries \
                     no information; drop the regressor"
                )));
            }
            // ---- 6b. POST-ARITHMETIC FINITENESS ----
            //
            // A SEPARATE check from 2, and both are needed: finite inputs are not
            // sufficient, because 1e300 values overflow the sum of squares in the
            // variance. Every DERIVED quantity is re-checked.
            if !st.mu.is_finite() {
                return Err(ForecastError::Validation(format!(
                    "regressor {i} has a non-finite mean after standardisation; \
                     the values are individually finite but their sum overflows"
                )));
            }
            if !st.std.is_finite() {
                return Err(ForecastError::Validation(format!(
                    "regressor {i} has a non-finite standard deviation after \
                     standardisation; the values are individually finite but the \
                     sum of squares overflows"
                )));
            }
            // ---- 6a-bis. ZERO SPREAD THAT 6a CANNOT SEE ----
            //
            // 6a reads `st.std`, which only reports a constant column when the column was
            // actually standardised. A caller sending `standardize: false` gets
            // `(mu, std) = (0.0, 1.0)` unconditionally, so a perfectly constant driver —
            // collinear with the intercept, carrying no information — was admitted by the
            // flag alone, while the SAME values without the flag were refused.
            // `contracts/forecast-tool-boundary-v1.yaml`'s `standardize` row states the
            // guarantee this restores: "A single-valued column is therefore NOT exempt ...
            // which is refused as a constant column carrying no information". On the
            // neuralprophet arm there is no `diagnostics.regressors[]` block (D-37), so
            // nothing downstream would have surfaced it either.
            //
            // AFTER the two finiteness checks, not beside 6a: a column of 1e308 values is
            // constant AND overflows the mean, and the overflow is the more specific
            // diagnosis — `the_regressor_refusal_table_is_complete` pins that ordering.
            // The auto {0,1} carve-out is unaffected: a binary column holds both 0 and 1,
            // so it has spread and is correctly not caught here.
            if hist
                .split_first()
                .is_some_and(|(first, rest)| rest.iter().all(|v| v == first))
            {
                return Err(ForecastError::Validation(format!(
                    "regressor {i} is constant over the history rows and carries \
                     no information; drop the regressor"
                )));
            }
            // EVERY row, history AND the horizon tail — not `hist` alone.
            //
            // `mu`/`std` are fitted on the history, but the SAME standardisation is applied
            // to the future values the caller supplies over the horizon (`prophet::predict`
            // on one arm, `regressors::standardise_cell` on the other). Scanning only the
            // history left the tail unchecked, so a history in [0, 1] paired with a single
            // huge future value — each individually finite, so check 2 passes — produced a
            // non-finite standardised cell that no check refused. `standardise_cell`'s own
            // doc states the invariant this restores: "the door has already refused
            // `std == 0.0` and re-checked every derived cell for finiteness, which is why
            // there is no guard here".
            //
            // The f32 bound is the neuralprophet arm's: it narrows each cell with `as f32`,
            // and a cell that is finite in f64 but exceeds f32::MAX narrows to infinity,
            // which reaches `RegressorBlock::forward`, makes the AdamW step produce NaN
            // weights, and returns an all-`null` yhat as a SUCCESS. Both arms pay the bound
            // because the door is above the dispatch and the wider arm loses nothing: a
            // standardised cell above 3.4e38 is pathological under either arithmetic.
            if let Some(bad) = r.values.iter().position(|v| {
                let z = (v - st.mu) / st.std;
                !z.is_finite() || !(z as f32).is_finite()
            }) {
                return Err(ForecastError::Validation(format!(
                    "regressor {i} has a non-finite standardised cell at row \
                     {bad}; the value is finite but (value - mu) / std is not"
                )));
            }

            reg_std.push(st);
            reg_hist.push(hist.to_vec());
            reg_fut.push(futv.to_vec());
        }
    }

    match model_name.as_str() {
        "prophet" => {
            // ---- the LOGISTIC CHANGEPOINT LAMBDA bound, at THE door and BEFORE make_design ----
            //
            // `predict`'s `Growth::Logistic` arm draws `poisson(lambda)` NEW changepoints on
            // every one of the 1 000 simulation rows, with
            // `lambda = changepoints_t.len() * (t_max - 1)`, and the per-row work
            // (`logistic_gammas` + a sort + a full `piecewise_logistic`) is LINEAR in that
            // count. Nothing bounded lambda: `MAX_POINTS`, `MAX_SPAN_DAYS` and `MAX_HORIZON`
            // are checked in isolation and it is their RATIO that sizes this, with
            // `dates::future_days` multiplying the horizon COUNT by 7 for "W" and ~30.44 for
            // "MS". `FIT_BUDGET_SECS` cannot cover it either — that budget is entered inside
            // `fit_prophet`, and this cost is spent in `predict`, which runs after the fit
            // returns with no budget at all. A 1 132-byte accepted request walled at 2.334 s
            // against SC1's 2 s bar (06-REVIEW.md CR-01).
            //
            // The count comes from `prophet::changepoint_count`, the SAME function
            // `make_design` is tied to, so the door's lambda IS the lambda `predict` draws
            // rather than a second inline copy of the arithmetic that could drift from it.
            if growth == Growth::Logistic {
                // `ds` is strictly ascending and `ds.len() >= MIN_POINTS` (10), both already
                // refused above, so `t_scale_days` is strictly positive and no division
                // guard is needed — adding one would imply a case that cannot occur.
                let t_scale_days = (ds[ds.len() - 1] - ds[0]) as f64;
                let future_span_days = fut[fut.len() - 1] - ds[0];
                let t_max = future_span_days as f64 / t_scale_days;
                let n_changepoints = changepoint_count(ds.len(), &spec);
                let lambda = n_changepoints as f64 * (t_max - 1.0);
                if lambda > MAX_LOGISTIC_CHANGEPOINT_LAMBDA {
                    return Err(ForecastError::Validation(format!(
                        "the logistic uncertainty simulation would draw a Poisson mean of \
                         {lambda:.1} new changepoints per sample \
                         (changepoints x (t_max - 1) = {n_changepoints} x ({t_max:.3} - 1), \
                         where t_max is the {future_span_days}-day future span over the \
                         {t_scale_days}-day history span), which exceeds \
                         max_logistic_changepoint_lambda {MAX_LOGISTIC_CHANGEPOINT_LAMBDA}; \
                         shorten the horizon, use freq \"D\" instead of \"W\" or \"MS\", or \
                         send a longer history"
                    )));
                }
            }

            let mut design = make_design(&ds, &args.y, &spec);
            crate::regressors::splice(&mut design, &reg_std, &reg_hist);
            let t0 = Instant::now();
            // UNCHANGED: the optimiser and the gradient need no regressor awareness — the
            // spliced columns are ordinary design columns with ordinary prior scales.
            let (p, info) = fit_prophet(&design, 8);
            let fit_seconds = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let fc = predict(
                &design,
                &p,
                &fut,
                seed,
                &crate::regressors::RegressorChannel {
                    specs: &reg_std,
                    values: &reg_fut,
                },
            )
            // The door BUILT this channel (`reg_std`/`reg_fut` above); the caller cannot
            // shape it. `predict`'s three channel invariants are `Validation` because a
            // direct Rust caller really is supplying the channel — but here a breach is the
            // door's own arithmetic, so reporting it as the caller's bad input would be a
            // lie the transport then repeats (`map_error`: "the caller's fault stays the
            // caller's fault"). Matches 06.1-07's rule for the n_grid/n_train guard:
            // `Internal` rather than `Validation` because the input was already accepted.
            .map_err(|e| {
                ForecastError::Internal(format!("the door built a malformed regressor channel: {e}"))
            })?;
            let predict_seconds = t1.elapsed().as_secs_f64();
            let mut components = serde_json::Map::new();
            for (n, v) in &fc.components {
                components.insert(n.clone(), serde_json::json!(v));
            }
            // ---- THE IDENTIFIABILITY DIAGNOSTIC (D-35, D-36) ----
            //
            // Computed AFTER the fit and only when the request actually carried regressors.
            // Both keys are built CONDITIONALLY and inserted into the map below rather than
            // being fields with a `skip_serializing_if`: a serialisation attribute that
            // misfires emits `"regressors": []` on EVERY response, and `invariance::
            // signature` hashes the WHOLE `diagnostics` object, so that would change every
            // recorded baseline and break SC2. The mechanism has to be "the key is never
            // constructed", not "the key is usually omitted".
            //
            // Nothing equivalent exists on the neuralprophet arm, deliberately (D-37): VIF
            // and the condition number are properties of the design matrix PROPHET builds,
            // and AR absorption is a training dynamic rather than column collinearity, so a
            // green VIF there would reassure about the wrong thing.
            let identifiability = if reg_std.is_empty() {
                None
            } else {
                Some(crate::regressors::identifiability(
                    &design,
                    &reg_std,
                    &p.beta,
                    REGRESSOR_VIF_WARN,
                    REGRESSOR_CONDITION_NUMBER_WARN,
                ))
            };
            let mut response = ForecastResponse {
                model: "prophet".into(),
                freq,
                n_history: ds.len(),
                fit_seconds,
                predict_seconds,
                ds: fut.iter().map(|d| format_ymd(*d)).collect(),
                yhat: fc.yhat,
                yhat_lower: fc.yhat_lower,
                yhat_upper: fc.yhat_upper,
                trend: fc.trend,
                components,
                diagnostics: serde_json::json!({
                    "growth": format!("{growth:?}"),
                    "seasonality_mode": format!("{mode:?}"),
                    "seasonalities": spec.seasonalities.iter().map(|s| format!("{} (order {})", s.name, s.order)).collect::<Vec<_>>(),
                    "n_changepoints": design.changepoints_t.len(),
                    "active_changepoints": p.delta.iter().filter(|d| d.abs() > 1e-3).count(),
                    "sigma_obs": p.sigma_obs,
                    "lbfgs": {
                        "rounds": info.rounds,
                        "iterations": info.iterations,
                        "evaluations": info.evals,
                        "objective": info.objective,
                        "last_status": info.status,
                        "budget_hit": info.budget_hit,
                        "iters_per_round_cap": MAX_ITERS_PER_ROUND,
                        "budget_secs": FIT_BUDGET_SECS
                    },
                    "interval_width": interval_width,
                    "uncertainty_samples": spec.uncertainty_samples
                }),
            };
            if let Some(id) = identifiability {
                let diag = response
                    .diagnostics
                    .as_object_mut()
                    .expect("the prophet diagnostics value is built as a JSON object above");
                let mut rows = Vec::with_capacity(id.regressors.len());
                for r in &id.regressors {
                    let mut o = serde_json::Map::new();
                    o.insert("name".into(), serde_json::Value::String(r.name.clone()));
                    o.insert(
                        "mode".into(),
                        serde_json::Value::String(mode_wire_name(r.mode).into()),
                    );
                    o.insert("mu".into(), finite_or_null(r.mu));
                    o.insert("std".into(), finite_or_null(r.std));
                    o.insert(
                        "vif".into(),
                        r.vif.map_or(serde_json::Value::Null, finite_or_null),
                    );
                    // `warning` is ABSENT, not null, when the column is clean: a key that is
                    // always present with a null is a key a consumer has to branch on.
                    if let Some(w) = &r.warning {
                        o.insert("warning".into(), serde_json::Value::String(w.clone()));
                    }
                    rows.push(serde_json::Value::Object(o));
                }
                diag.insert("regressors".into(), serde_json::Value::Array(rows));
                let mut sib = serde_json::Map::new();
                sib.insert("scope".into(), serde_json::Value::String(id.scope.clone()));
                sib.insert(
                    "condition_number".into(),
                    id.condition_number
                        .map_or(serde_json::Value::Null, finite_or_null),
                );
                sib.insert(
                    "condition_number_warning".into(),
                    id.condition_number_warning
                        .clone()
                        .map_or(serde_json::Value::Null, serde_json::Value::String),
                );
                sib.insert(
                    "ridge".into(),
                    id.ridge.map_or(serde_json::Value::Null, finite_or_null),
                );
                sib.insert(
                    "regularized".into(),
                    serde_json::Value::Bool(id.regularized),
                );
                sib.insert(
                    "status".into(),
                    serde_json::Value::String(id.status.as_str().into()),
                );
                diag.insert(
                    "regressors_identifiability".into(),
                    serde_json::Value::Object(sib),
                );
            }
            Ok(response)
        }
        // Ported verbatim from `sources/004-forecast-mcp-thin-server/src/lib.rs:226-262`
        // (D-08), replacing 06-01's refusing tracer stub (REVIEW-06-06). Every training
        // rule below is D-10 and is pinned by an invariant test in `np::parity`.
        "neuralprophet" => {
            if freq != "D" {
                return Err(ForecastError::Validation(
                    "neuralprophet supports freq D only".into(),
                ));
            }
            // ---- THE ARM RULES FOR REGRESSORS (D-26, D-28, D-21) ----
            //
            // The temporary refusal plan 06.1-01 left here is GONE; these are the real
            // rules. Every SHARED check has already run above the dispatch, so what is left
            // is only what is true of THIS arm.
            //
            // REFUSAL PRECEDENCE IS DECIDED HERE AND STATED, not left to evaluation order.
            // A request can satisfy several of these at once — a multiplicative regressor
            // carrying a prior_scale on a gappy series with lags — and which message the
            // caller receives must be a DECISION, because the messages are verbatim
            // contracts that e2e cases string-match. The order is:
            //
            //   (1) the SHARED hoisted checks, in the order that site keeps them (above);
            //   (2) multiplicative mode;
            //   (3) an explicitly present prior_scale;
            //   (4) the gappy-series-at-lags predicate (below, after the n_lags range
            //       checks, because it needs the parsed dates and the span).
            //
            // Cheapest and most unconditional first, most input-dependent last: (2) and (3)
            // are single field reads on data already validated above, while (4) walks the
            // day span. `the_refusal_precedence_is_stable_when_several_rules_apply` pins it,
            // so a refactor cannot silently reorder them.
            //
            // (2) D-28. NP-lite has NO multiplicative composition path: the event block is
            // one additive `.add()` on the forward and multiplicative mode was never built
            // or measured. A field may be arm-restricted the way `cap` is growth-restricted
            // without breaking D-22's one-shape requirement; accepting the field and
            // ignoring it would violate the refuse-never-default rule outright (D-21).
            // Read from `reg_std`, which carries the RESOLVED mode the shared allowlist
            // above produced — not from the raw string, so `None` (the default) is additive
            // here by exactly the rule the prophet arm uses.
            if let Some(i) = reg_std.iter().position(|s| s.mode == Mode::Multiplicative) {
                return Err(ForecastError::Validation(format!(
                    "regressor {i} has mode \"multiplicative\", which is prophet-only: the \
                     neuralprophet model composes exogenous terms with a single additive \
                     block and has no multiplicative path. Set mode to \"additive\", or use \
                     model \"prophet\""
                )));
            }
            // (3) D-21 applied to `prior_scale`. The NP trainer has ONE global
            // `weight_decay` and no per-parameter prior, so there is no mechanism by which
            // a per-regressor prior scale could take effect and no measurement anywhere for
            // what it should mean here. Accepting the field and dropping it is exactly the
            // "plausible answer to a question they did not ask" failure D-21 names.
            //
            // The check keys on `Some(_)`, NEVER on the defaulted value: a check keyed on
            // the default would refuse every neuralprophet regressor request. The positive
            // control is `a_neuralprophet_regressor_without_a_prior_scale_is_accepted`.
            if let Some(i) = args
                .regressors
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .position(|r| r.prior_scale.is_some())
            {
                return Err(ForecastError::Validation(format!(
                    "regressor {i} carries prior_scale, which is prophet-only: the \
                     neuralprophet trainer applies one global weight decay and has no \
                     per-regressor prior. Omit the field, or use model \"prophet\""
                )));
            }
            let n_lags = args.n_lags.unwrap_or(0);
            if n_lags > 365 {
                return Err(ForecastError::Validation("n_lags ≤ 365".into()));
            }
            // (4) D-26, the gappy-series-at-lags predicate — computed from the span and the
            // observed days the door already holds, BEFORE `NpData::new` materialises
            // anything.
            //
            // THE PREDICATE IS WHOLE-SERIES, AND THAT IS NOT A SIMPLIFICATION. With
            // `n_lags > 0` the training sample list is `(n_lags..n_train_grid)` — EVERY grid
            // row, imputed rows included — so a gap ANYWHERE enters the loss. A narrower
            // "only gaps inside a lag window" rule collapses to the same predicate.
            //
            // Three alternatives were rejected, one line each so the shape is legible here:
            //   - refuse whenever `n_lags > 0`: refuses a well-defined case (a gap-free
            //     daily series, where the grid equals the caller's rows) to avoid a cheap
            //     check;
            //   - require grid-complete values on this arm: makes `values.len()` differ
            //     between the two models, breaking D-22's one-shape requirement outright;
            //   - impute and disclose: spike 014 measured two DEFENSIBLE rules 10.48 apart
            //     on a series of scale 35.32 (~30 %), with the garbage probe moving the
            //     forecast by 192 and flipping the sign of BOTH coefficients. There is no
            //     defensible value to invent.
            //
            // At `n_lags = 0` this does not fire at all: D-27 makes the imputed-day value
            // unread BY CONSTRUCTION there, so there is nothing to argue about.
            if n_lags > 0 && !reg_std.is_empty() {
                // `span_days` is the same quantity, computed once at :144 for the
                // MAX_SPAN_DAYS check and reused by the design-cost operand at :606. Two
                // expressions for one bounded quantity is how the ceiling and the gap
                // predicate come to disagree about the span.
                let span = span_days as usize;
                if span != ds.len() {
                    // The FIRST missing day, in the same `YYYY-MM-DD` form the caller sent.
                    // `ds` is strictly ascending (refused above), so the first gap is the
                    // first place consecutive entries differ by more than one day.
                    let first_missing = ds
                        .windows(2)
                        .find(|w| w[1] - w[0] > 1)
                        .map_or(ds[0], |w| w[0] + 1);
                    return Err(ForecastError::Validation(format!(
                        "this request carries {} regressor(s) with n_lags {n_lags}, but the \
                         series is missing {} day(s) between {} and {} — the first is {}. \
                         With lags the neuralprophet model trains on an imputed DAILY grid \
                         and reads a regressor value on every one of those days, and there \
                         is no defensible value to invent for a day you did not send \
                         (measured: two defensible fill rules 10.48 apart on a series of \
                         scale 35.32). Supply a gap-free daily series, or set n_lags to 0",
                        reg_std.len(),
                        span - ds.len(),
                        format_ymd(ds[0]),
                        format_ymd(ds[ds.len() - 1]),
                        format_ymd(first_missing)
                    )));
                }
            }
            let n_train = ds.len();
            let d = np::NpData::new(&ds, &args.y, n_train, 10, 0.8);
            // The channel `np::train` and the predict paths read. Its constructor pays the
            // lagged path's grid-equals-caller-rows invariant in RELEASE and returns
            // `ForecastError::Internal` if it is broken — which, given the predicate above,
            // can only mean that predicate regressed.
            let np_regs = if reg_std.is_empty() {
                None
            } else {
                // Moved, not cloned: the prophet arm's readers of these three are in the
                // mutually exclusive `match` arm, and nothing on THIS arm reads them after
                // the channel is built.
                Some(crate::regressors::NpRegressors::new(
                    &d, n_lags, reg_std, reg_hist, reg_fut,
                )?)
            };
            if n_lags >= d.n_train_grid {
                return Err(ForecastError::Validation(
                    "n_lags must be smaller than the series span in days".into(),
                ));
            }
            // ---- the NEURALPROPHET TRAINING COST bound, at THE door and BEFORE np::train ----
            //
            // Cost axis C-08, and the ONLY axis in this door that had no budget of any kind
            // on its path: `fit::FIT_BUDGET_SECS` is read at exactly one place, inside
            // `fit::fit_prophet`, and this arm never enters that function. `n_lags <= 365`,
            // `n_samples <= n_train_grid <= MAX_SPAN_DAYS` and `epochs <= 500` are each
            // enforced on their own; what was never checked is their PRODUCT, times the 2-or-3
            // learning-rate sweep run below. MEASURED: the worst legal request (20 000
            // contiguous daily points, n_lags 365, horizon 3650) walls at 47.924 s on release,
            // 24x over SC1's 2 s bar.
            //
            // The price is computed by `np::request_train_cost`, built out of the SAME three
            // functions used to configure the sweep below (`door_lr_sweep`, `door_epochs`,
            // `n_training_samples`), so the door cannot price a request differently from the
            // work it then spends.
            //
            // The event design is expanded HERE, before the price, so a request carrying
            // events is priced FOR them before the first `np::train` (SC4, C-08). The
            // caller's `HolidayArg` list was already validated and parsed by the hoisted
            // bounds above; `EventSpec` mirrors `Holiday` field for field, which is what
            // makes D-29's one-argument-two-models claim literal.
            let event_design = if holidays.is_empty() {
                None
            } else {
                Some(events::EventDesign::new(
                    holidays
                        .iter()
                        .map(|h| events::EventSpec {
                            name: h.name.clone(),
                            days: h.days.clone(),
                            lower_window: h.lower_window,
                            upper_window: h.upper_window,
                        })
                        .collect(),
                ))
            };
            let n_event_cols = event_design.as_ref().map_or(0, events::EventDesign::dim);
            // The count the door PRICES must be the count the block is BUILT at. The
            // hoisted `MAX_HOLIDAY_COLUMNS` loop accumulated exactly the same sum from the
            // same windows, so a disagreement here would mean one of the two rules drifted.
            //
            // This is a RELEASE-mode check that RETURNS, not a `debug_assert_eq!`.
            // `debug_assert!` is compiled out of `--release`, the profile the door ships in,
            // so the guard would be absent exactly where it matters: the door would price
            // C-08 on one count and train on another, and `MAX_NP_TRAIN_COST` would be
            // evadable by precisely the disagreement. This is the same rule
            // `regressors::NpRegressors::new` states for the structurally identical
            // priced-vs-built invariant, and `np.rs`'s sample-count derivation states again
            // — stated in two places and then not followed in this third.
            //
            // `Internal`, not `Validation`: by this line the caller's input has already been
            // accepted, so a breach is a server-side invariant break and the message should
            // say so rather than blaming the caller.
            if n_event_cols != holiday_columns {
                return Err(ForecastError::Internal(format!(
                    "the priced event-column count ({n_event_cols}) must equal the bounded \
                     holiday column count ({holiday_columns}); the door bounded one \
                     expansion and priced another"
                )));
            }
            let n_reg_cols = np_regs
                .as_ref()
                .map_or(0, crate::regressors::NpRegressors::dim);
            let np_cost = np::request_train_cost(&d, n_train, n_lags, n_event_cols, n_reg_cols);
            if np_train_cost_is_over(np_cost) {
                let n_samples = np::n_training_samples(&d, n_lags);
                let epochs = np::door_epochs(n_train, n_samples, n_lags);
                let sweep = np::door_lr_sweep(n_lags).len();
                // The per-sample WIDTH is reported as the door computes it, event term
                // included, so a request refused BECAUSE of its events can see that term
                // rather than being shown an arithmetic that does not reproduce {np_cost}.
                // At zero event columns the suffix is empty and the message is exactly the
                // one this door has always sent (the e2e cases string-match it).
                let event_term = if n_event_cols == 0 {
                    String::new()
                } else {
                    format!(
                        " + {n_event_cols} event columns priced at \
                         fit_np_event_cost_per_column"
                    )
                };
                return Err(ForecastError::Validation(format!(
                    "this neuralprophet request buys {np_cost} units of training work \
                     (learning-rate sweep {sweep} x epochs {epochs} x samples {n_samples} x \
                     (n_lags + 1) {}{event_term}), which exceeds max_np_train_cost \
                     {MAX_NP_TRAIN_COST}; reduce n_lags, shorten the history, or narrow the \
                     series span",
                    n_lags + 1
                )));
            }
            let t0 = Instant::now();
            // spike-002 lesson: a short lr sweep selected by TRAIN loss stands in for NP's
            // range test (D-10 — never select by test error); 4x the auto epochs when lags
            // are on (the linear AR case needs the budget), capped at 320.
            let mut best: Option<(f64, f64, np::NpModel, np::TrainLog)> = None;
            // The sweep and the epoch rule come from `np`, not from literals here, so the
            // cost priced above is the cost this loop actually spends (C-08).
            let lrs: &[f64] = np::door_lr_sweep(n_lags);
            let n_samples = np::n_training_samples(&d, n_lags);
            for &lr in lrs {
                let cfg = np::TrainConfig {
                    n_lags,
                    ar_layers: if n_lags > 0 { vec![32] } else { vec![] },
                    max_lr: lr,
                    epochs: Some(np::door_epochs(n_train, n_samples, n_lags)),
                    batch: None,
                    weight_decay: 1e-3,
                    huber_beta: 0.3,
                    newer_w: 2.0,
                    seed,
                    // The expanded design, or `None` when the caller sent no holidays.
                    // `None` and an empty design are the same fit bit for bit, so a
                    // request without events produces exactly what it produced before this
                    // plan — which is what keeps both recorded neuralprophet invariance
                    // signatures reproducing.
                    //
                    // Cloned per learning rate because `TrainConfig` owns its design and
                    // the sweep builds one config per rate. The clone is the columns and
                    // the membership sets, both bounded by the hoisted ceilings above.
                    event_design: event_design.clone(),
                    // Cloned per learning rate for the reason `event_design` is: `TrainConfig`
                    // owns its channel and the sweep builds one config per rate. The clone is
                    // the caller's own arrays, bounded by the hoisted design-cost ceiling.
                    regressors: np_regs.clone(),
                };
                let (m, log) = np::train(&d, &cfg, false);
                let fl = *log.epoch_loss.last().unwrap_or(&f64::INFINITY);
                if !fl.is_finite() {
                    continue;
                }
                if best.as_ref().is_none_or(|b| fl < b.0) {
                    best = Some((fl, lr, m, log));
                }
            }
            let (train_loss, selected_lr, m, log) = best.ok_or_else(|| {
                ForecastError::Internal("training diverged for every learning rate".into())
            })?;
            let fit_seconds = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            // The channel the predict helpers take: the DESIGN (which days are events) and
            // the trained BLOCK (what an active column is worth). Weights alone cannot
            // build an indicator row for a future day, which is why both travel together.
            // `None` whenever the caller sent no holidays, so every event-free path is
            // byte-identical to what it was before this plan.
            let ev: np::EventChannel<'_> = event_design.as_ref().zip(log.events.as_ref());
            // The regressor channel: the caller's OWN rows — the history days and the
            // horizon days, `ds.len() + horizon` of them, exactly the array the caller
            // sent — standardised with the constants the FIT used (`log.regressor_specs`
            // travel with the weights for that reason), never a grid-shaped array (D-27).
            let reg_rows = np_regs.as_ref().map(|r| r.rows_for(&ds, &fut));
            let rc: np::NpRegChannel<'_> = reg_rows.as_ref().zip(log.regressors.as_ref());
            // `predict_trend` is branch-independent; only the yhat path differs.
            let trend = np::predict_trend(&d, &m, &fut);
            let yhat = if n_lags == 0 {
                np::predict_ts(&d, &m, &fut, ev, rc)
            } else {
                np::predict_ar_recursive(&d, &m, &fut, ev, rc)
            };
            // Residual-based band. NeuralProphet itself would use quantile regression; that
            // was NOT spiked (CONTEXT deferred), and the diagnostics say so rather than
            // implying a coverage guarantee this band does not have.
            let fitted = if n_lags == 0 {
                np::predict_ts(&d, &m, &ds, ev, rc)
            } else {
                let idx: Vec<usize> = ds
                    .iter()
                    .map(|day| (day - d.t0) as usize)
                    .filter(|i| *i >= n_lags)
                    .collect();
                let pr = np::predict_ar_1step(&d, &m, &idx, ev, rc);
                let mut out = vec![f64::NAN; ds.len()];
                let mut k = 0;
                for (i, day) in ds.iter().enumerate() {
                    if (day - d.t0) as usize >= n_lags {
                        out[i] = pr[k];
                        k += 1;
                    }
                }
                out
            };
            let resid: Vec<f64> = fitted
                .iter()
                .zip(&args.y)
                .filter(|(f, _)| f.is_finite())
                .map(|(f, y)| y - f)
                .collect();
            let sd = (resid.iter().map(|r| r * r).sum::<f64>() / resid.len().max(1) as f64).sqrt();
            let z = normal_quantile((1.0 + interval_width) / 2.0);
            let predict_seconds = t1.elapsed().as_secs_f64();
            let mut components = serde_json::Map::new();
            components.insert("trend".into(), serde_json::json!(trend));
            // ---- D-31: per-EVENT components plus a `holidays` roll-up ----
            //
            // One component per event NAME, matching the prophet arm's grouping exactly
            // (`prophet::columns` sets `Column.component = h.name`, and `predict` dedups by
            // that string), plus the `holidays` roll-up prophet emits when its holiday list
            // is non-empty. NOT one component per COLUMN: that map could reach
            // MAX_HOLIDAY_COLUMNS entries and would disagree with the prophet arm's naming
            // for the very same argument.
            //
            // Grouped by NAME rather than by event INDEX, because two `HolidayArg`s may
            // share a name and prophet sums them into one component. Indexing would emit
            // the same key twice and silently keep only the last.
            //
            // THE CONVERSION IS THE SUBSTANCE HERE. The block trains on the NORMALISED
            // target — `NpData` carries `shift = min` and `scale = q95 - min`, and `d.norm`
            // is applied to `y` before training — so `w_j * x_j` is in normalised units.
            // The published component is therefore
            //
            //     component[event][i] = d.scale * SUM_{j in that event} ( w_j * x_j[i] )
            //
            // multiplying by `scale` and NEVER adding `shift`. A component is a
            // CONTRIBUTION, not a level; the shift belongs to the trend component emitted
            // beside it, and adding it per event would make the components sum to one shift
            // per event more than the forecast, so the additive decomposition the prophet
            // arm publishes would stop reconciling. `d.denorm` is deliberately NOT used
            // here for exactly that reason — it adds the shift.
            //
            // Emitted ONLY when the request carried holidays, so an event-free
            // neuralprophet response is the map it has always been and both recorded
            // invariance signatures still reproduce.
            if let (Some(design), Some(block)) = (event_design.as_ref(), log.events.as_ref()) {
                let w = block.weights();
                let dim = design.dim();
                let rows = design.rows(&fut);
                // One accumulator per DISTINCT name, in first-appearance order, plus the
                // roll-up across every column.
                let mut names: Vec<String> = Vec::new();
                let mut sums: Vec<Vec<f64>> = Vec::new();
                let mut roll_up = vec![0.0f64; fut.len()];
                for (ci, &(ei, _)) in design.cols.iter().enumerate() {
                    let Some(name) = design.events.get(ei).map(|e| e.name.as_str()) else {
                        continue;
                    };
                    let slot = names.iter().position(|n| n == name).unwrap_or_else(|| {
                        names.push(name.to_string());
                        sums.push(vec![0.0f64; fut.len()]);
                        names.len() - 1
                    });
                    // `w` is read through `get` rather than indexed: `weights()` and `cols`
                    // are produced by two different objects and nothing in the type system
                    // ties their lengths (IN-02). A short weight vector yields a zero
                    // contribution instead of panicking inside a library.
                    let wj = f64::from(w.get(ci).copied().unwrap_or(0.0));
                    for i in 0..fut.len() {
                        let x = f64::from(rows[i * dim + ci]);
                        let term = d.scale * wj * x;
                        sums[slot][i] += term;
                        roll_up[i] += term;
                    }
                }
                for (name, v) in names.into_iter().zip(sums) {
                    components.insert(name, serde_json::json!(v));
                }
                components.insert("holidays".into(), serde_json::json!(roll_up));
            }
            // Bound before the move so the literal stays in declaration order
            // (clippy::inconsistent_struct_constructor is a workspace `warn`).
            let yhat_lower: Vec<f64> = yhat.iter().map(|v| v - z * sd).collect();
            let yhat_upper: Vec<f64> = yhat.iter().map(|v| v + z * sd).collect();
            Ok(ForecastResponse {
                model: "neuralprophet".into(),
                freq,
                n_history: ds.len(),
                fit_seconds,
                predict_seconds,
                ds: fut.iter().map(|d| format_ymd(*d)).collect(),
                yhat,
                yhat_lower,
                yhat_upper,
                trend,
                components,
                diagnostics: serde_json::json!({
                    "n_lags": n_lags,
                    "ar_layers": if n_lags > 0 { vec![32] } else { vec![] },
                    "epochs": log.epochs,
                    "batch": log.batch,
                    "steps": log.steps,
                    "params": log.n_params,
                    "selected_lr": selected_lr,
                    "final_train_loss": train_loss,
                    "residual_sd": sd,
                    "band": "residual-sd based, not NeuralProphet's quantile regression",
                    "seasonalities": d.seasons.iter().map(|s| format!("{} (order {})", s.name, s.order)).collect::<Vec<_>>()
                }),
            })
        }
        other => Err(ForecastError::Validation(format!(
            "model {other:?}: prophet or neuralprophet"
        ))),
    }
}

/// The door's C-08 comparison, named ONCE so the boundary can be tested without paying it.
///
/// It is `>`, not `>=`: a request pricing EXACTLY at the bound is accepted. That one-off is
/// invisible to any near-miss control this crate can afford to run — at the bound a single
/// request costs 45.619 s on a debug profile — so
/// [`tests::the_np_train_cost_bound_is_exclusive_not_inclusive`] pins the comparison here
/// instead.
fn np_train_cost_is_over(cost: u64) -> bool {
    cost > MAX_NP_TRAIN_COST
}

/// Acklam's inverse normal CDF (enough for band z-scores).
///
/// The clamp is load-bearing and lives in `aprender::monte_carlo::engine::inverse_normal_cdf`,
/// which this now delegates to. Without it the tails divide infinity by infinity:
/// `interval_width = 0.9999999999999999` is the largest value the door accepts, and
/// `(1.0 + w) / 2.0` rounds to EXACTLY 1.0, so `(1.0 - p).ln()` is `-inf`, `q` is `inf`
/// and the returned z is `NaN` — which `serde_json` then writes as JSON `null` for every
/// `yhat_lower`/`yhat_upper` in an otherwise successful response.
#[must_use]
pub fn normal_quantile(p: f64) -> f64 {
    // Delegates rather than transcribes: proven bit-identical to core over a
    // 100k-point grid plus both clamp shoulders and both branch boundaries.
    aprender::monte_carlo::engine::inverse_normal_cdf(p)
}

#[cfg(test)]
mod tests {
    use super::{forecast, normal_quantile};
    use crate::dates::{days_from_civil, format_ymd};
    use crate::prophet::Rng;
    use crate::test_support::equation_tolerance;
    use crate::types::{
        ForecastArgs, ForecastError, MAX_NP_TRAIN_COST, REGRESSOR_PRIOR_SCALE_MAX,
        REGRESSOR_PRIOR_SCALE_MIN,
    };
    use aprender::autograd::{clear_graph, graph_tape_len};

    /// A 120-point synthetic DAILY series: linear trend + a weekly sine + a little noise.
    /// Deliberately short — these tests prove dispatch, refusals and tape hygiene, never
    /// accuracy. Correctness against the NeuralProphet 0.9.0 oracle is `np::parity`.
    fn synthetic_daily(n: usize) -> (Vec<String>, Vec<f64>) {
        let mut rng = Rng::new(7);
        let t0 = days_from_civil(2020, 1, 1);
        let mut ds = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            ds.push(format_ymd(t0 + i as i64));
            let t = i as f64;
            y.push(
                10.0 + 0.01 * t
                    + (2.0 * std::f64::consts::PI * t / 7.0).sin()
                    + 0.05 * rng.normal(),
            );
        }
        (ds, y)
    }

    fn np_args(n: usize, horizon: usize) -> ForecastArgs {
        let (ds, y) = synthetic_daily(n);
        ForecastArgs {
            ds,
            y,
            horizon,
            freq: None,
            model: Some("neuralprophet".into()),
            growth: None,
            cap: None,
            seasonality_mode: None,
            interval_width: None,
            holidays: None,
            n_lags: None,
            seed: None,
            regressors: None,
        }
    }

    #[test]
    fn neuralprophet_arm_dispatches_and_returns_a_band() {
        clear_graph();
        let args = np_args(120, 14);
        let r = forecast(&args).expect("the neuralprophet arm must dispatch, not refuse");
        assert_eq!(r.model, "neuralprophet");
        assert_eq!(r.ds.len(), 14, "one row per horizon step");
        assert_eq!(r.yhat.len(), 14);
        assert_eq!(r.trend.len(), 14);
        for i in 0..14 {
            assert!(
                r.yhat_lower[i] < r.yhat_upper[i],
                "row {i}: band must be strictly ordered ({} !< {})",
                r.yhat_lower[i],
                r.yhat_upper[i]
            );
            assert!(r.yhat[i].is_finite(), "row {i}: yhat must be finite");
        }
        let lr = r.diagnostics["selected_lr"]
            .as_f64()
            .expect("diagnostics.selected_lr");
        assert!(
            [0.01, 0.03, 0.1].iter().any(|c| (c - lr).abs() < 1e-12),
            "the lag-free sweep is {{0.01, 0.03, 0.1}} selected by TRAIN loss (D-10); got {lr}"
        );
        assert!(
            r.diagnostics["band"]
                .as_str()
                .expect("diagnostics.band")
                .contains("not NeuralProphet's quantile regression"),
            "the band must SAY it is residual-sd based, not quantile regression"
        );
        // D-10 tape hygiene: `clear_graph()` after every step means a completed fit leaves
        // the thread-local tape empty for the next caller on this thread.
        assert_eq!(
            graph_tape_len(),
            0,
            "the autograd tape must be empty after a completed neuralprophet fit"
        );
    }

    #[test]
    fn neuralprophet_refuses_non_daily_freq() {
        let mut args = np_args(120, 14);
        args.freq = Some("W".into());
        refusal(&args, "supports freq D only");
    }

    #[test]
    fn neuralprophet_refuses_n_lags_above_365() {
        let mut args = np_args(120, 14);
        args.n_lags = Some(366);
        refusal(&args, "365");
    }

    #[test]
    fn neuralprophet_refuses_n_lags_at_or_above_span() {
        // 120 consecutive daily points => n_train_grid == 120, so n_lags 120 leaves no
        // complete window. This must refuse at the door, never panic inside the fit.
        let mut args = np_args(120, 14);
        args.n_lags = Some(120);
        refusal(&args, "smaller than the series span");
    }

    // ---------------------------------------------------------------- door hardening ---
    // Every case below reached a crash, an unbounded allocation or a silently wrong
    // answer before it was closed. They live in `--lib` (which CI runs) rather than in
    // `tests/`, which is not on ci.yml's explicit `--test` line.

    fn refusal(args: &ForecastArgs, needle: &str) {
        match forecast(args) {
            Err(ForecastError::Validation(m)) => assert!(
                m.contains(needle),
                "message must name the fix; wanted {needle:?}, got {m:?}"
            ),
            other => panic!(
                "expected a Validation refusal, got {:?}",
                other.map(|r| r.model)
            ),
        }
    }

    /// D-29 INVERTED — the `"neuralprophet"` arm ACCEPTS `holidays` and publishes them.
    ///
    /// This test replaces `the_neuralprophet_arm_still_refuses_holidays` in the SAME commit
    /// that removes the refusal it pinned. That ordering is the point: a removed refusal
    /// with no positive test in its place is a window in which an event could be accepted
    /// and silently ignored, which is the D-21 failure class this phase exists to close.
    ///
    /// Acceptance alone is not the claim. The control asserts the same request WITHOUT
    /// holidays forecasts DIFFERENTLY, so "accepted" cannot be satisfied by an argument
    /// that is parsed, bounded, priced and then inert.
    #[test]
    fn holidays_on_the_neuralprophet_arm_are_accepted_and_publish_per_event_components() {
        let args = planted_np_args(PLANTED_POINTS, PLANTED_HORIZON, 0, 42);
        let r = forecast(&args).expect("holidays must be ACCEPTED on the neuralprophet arm");
        assert_eq!(r.model, "neuralprophet");
        for key in ["trend", "promo", "blackfriday", "holidays"] {
            assert!(
                r.components.contains_key(key),
                "the components map must carry {key:?}; got {:?}",
                r.components.keys().collect::<Vec<_>>()
            );
        }
        // The EVENT keys match the prophet arm's per-name grouping and its `holidays`
        // roll-up exactly. The two arms' FULL maps do not match and are not claimed to:
        // prophet also emits seasonality names, `additive_terms` and
        // `multiplicative_terms`, while this arm emits `trend` and the event keys.
        assert_eq!(
            r.components.len(),
            4,
            "one component per event NAME plus the roll-up plus trend — never one per \
             COLUMN, which could reach max_holiday_columns entries; got {:?}",
            r.components.keys().collect::<Vec<_>>()
        );

        // CONTROL: the same request without holidays must forecast DIFFERENTLY, or
        // "accepted" would be indistinguishable from "accepted and ignored".
        let mut off = args.clone();
        off.holidays = None;
        let r_off = forecast(&off).expect("the event-free control must still be accepted");
        assert!(
            !r_off.components.contains_key("holidays"),
            "an event-free neuralprophet response must publish NO event components, or the \
             two recorded invariance signatures would move"
        );
        let moved = r
            .yhat
            .iter()
            .zip(&r_off.yhat)
            .filter(|(a, b)| (*a - *b).abs() > 1e-9)
            .count();
        assert!(
            moved > 0,
            "holidays were accepted but changed nothing: all {} forecast rows are \
             bit-identical with and without them",
            r.yhat.len()
        );
    }

    /// The SCOPED half of D-29, and the reason this arm opened at `n_lags = 0` only.
    ///
    /// MEASURED (06.1-05 FINDING 1): at `n_lags > 0` the per-column recovery runs
    /// 0.2191-0.4660 against SC3's 0.10 bar under the epoch budget this door configures,
    /// with one run producing a negative weight. The combination is REFUSED rather than
    /// shipped behind a disclaimer — D-21's posture applied to the COMBINATION rather than
    /// to the whole arm.
    ///
    /// The message must name the LIMITATION, not merely say "unsupported": a caller who is
    /// told `n_lags = 0` works can act on it, and a caller told "unsupported" cannot.
    #[test]
    fn the_neuralprophet_arm_refuses_holidays_with_lags_naming_the_limitation() {
        let mut args = planted_np_args(200, 7, 7, 42);
        assert_eq!(
            args.n_lags,
            Some(7),
            "this case is only about the LAGGED arm"
        );
        match forecast(&args) {
            Err(ForecastError::Validation(m)) => {
                assert!(
                    m.contains("n_lags = 0"),
                    "the refusal must name the configuration that DOES work; got {m:?}"
                );
                assert!(
                    m.contains("recovery") || m.contains("recover"),
                    "the refusal must name the LIMITATION — effect recovery — rather than \
                     saying 'unsupported'; got {m:?}"
                );
                assert!(
                    !m.contains("prophet-only"),
                    "this is no longer a prophet-only refusal; got {m:?}"
                );
            }
            other => panic!(
                "expected a Validation refusal, got {:?}",
                other.map(|r| r.model)
            ),
        }

        // CONTROL A: the SAME lag count without holidays is still accepted, so the refusal
        // is about the COMBINATION and not about lags.
        let mut lags_only = args.clone();
        lags_only.holidays = None;
        forecast(&lags_only).expect("n_lags alone must still be accepted on this arm");

        // CONTROL B: the SAME holidays without lags are still accepted, so the refusal is
        // not quietly re-closing the arm.
        args.n_lags = None;
        forecast(&args).expect("holidays alone must still be accepted on this arm");
    }

    /// D-31 — the published component is in RESPONSE units, not normalised training units.
    ///
    /// The block trains on `d.norm(y)`, so a learned weight times an indicator is in
    /// normalised units and must be multiplied by `NpData::scale` before publication (never
    /// plus `shift`, which belongs to trend). This is the ONLY class of check that catches
    /// the omission: component PRESENCE and a CHANGED forecast both stay green on an
    /// unscaled component.
    ///
    /// The assertion is against the planted magnitude in RESPONSE units. The control makes
    /// the test non-vacuous by asserting the normalised value would MISS the same bar by a
    /// wide margin — if `scale` were ever near 1.0 the two would coincide and this case
    /// would prove nothing, so the separation is asserted rather than assumed.
    #[test]
    fn the_event_components_are_in_response_units() {
        let bar = equation_tolerance("neuralprophet-parity-v1", "event_effect_recovery_rel");
        let args = planted_np_args(PLANTED_POINTS, PLANTED_HORIZON, 0, 42);
        let r = forecast(&args).expect("the planted request must be accepted");
        let (peak, active) = event_component_peak(&r, "promo");
        assert!(
            active > 0,
            "the forecast window contains no active promo day, so this case would pass \
             vacuously on an all-zero component"
        );
        let rel = (peak - PLANTED_EFFECT).abs() / PLANTED_EFFECT;
        assert!(
            rel <= bar,
            "the published component peak is {peak}, which is {rel} away from the planted \
             {PLANTED_EFFECT} in RESPONSE units against a {bar} bar — a component left in \
             normalised training units misses by a factor of NpData::scale"
        );
        // The separation control: the same peak divided by a plausible series scale must
        // NOT also satisfy the bar, or the check could not tell the two unit systems apart.
        let scale = planted_series_scale();
        assert!(
            scale > 2.0,
            "this control needs the series scale to be far from 1.0 to separate the two \
             unit systems; got {scale}"
        );
        let unscaled_rel = (peak / scale - PLANTED_EFFECT).abs() / PLANTED_EFFECT;
        assert!(
            unscaled_rel > bar,
            "a component left in normalised units would read {} and must FAIL the {bar} \
             bar, or this test cannot distinguish the two unit systems",
            peak / scale
        );
    }

    /// D-31 — the components RECONCILE with the forecast they decompose.
    ///
    /// Asserted to a stated tolerance rather than exactly, and the reason is measured
    /// rather than hand-waved: `np::train` draws the event block from the same `Rng` AFTER
    /// the model, so an events-OFF run initialises the model identically — but the block's
    /// draws shift the subsequent shuffle stream, so the two fits diverge slightly. The
    /// difference between the ON and OFF forecasts is therefore the event contribution
    /// PLUS a small base-model difference, and the bar is SC3's own recovery bar applied to
    /// the planted magnitude.
    ///
    /// A component scaled wrongly by `NpData::scale` misses this by ~scale x, far outside
    /// any tolerance this test could plausibly carry.
    #[test]
    fn the_event_components_reconcile_with_the_forecast() {
        let bar = equation_tolerance("neuralprophet-parity-v1", "event_effect_recovery_rel");
        let args = planted_np_args(PLANTED_POINTS, PLANTED_HORIZON, 0, 42);
        let on = forecast(&args).expect("the events-ON request must be accepted");
        let mut off_args = args.clone();
        off_args.holidays = None;
        let off = forecast(&off_args).expect("the events-OFF control must be accepted");

        let roll = component_values(&on, "holidays");

        // The BASE-MODEL LEVEL SHIFT, measured from the rows where the roll-up is exactly
        // zero. On those rows the event term contributes nothing by construction, so
        // whatever separates the two forecasts there is the two fits differing — and it
        // must be removed before the remainder can be attributed to the events.
        //
        // MEASURED on this configuration: mean -1.7864 over 32 inactive rows with a mean
        // absolute deviation of 0.0703, i.e. 3.9 % of the shift. That is a LEVEL offset,
        // not noise, which is what makes subtracting a single number a correction rather
        // than a fudge — and the character is ASSERTED below, not assumed, so a future
        // change that made the two fits diverge in SHAPE would fail here instead of being
        // silently absorbed.
        let inactive: Vec<f64> = (0..on.yhat.len())
            .filter(|&i| roll[i].abs() < 1e-9)
            .map(|i| on.yhat[i] - off.yhat[i])
            .collect();
        assert!(
            inactive.len() >= 8,
            "too few event-free rows ({}) to measure the base-model shift against",
            inactive.len()
        );
        let n = inactive.len() as f64;
        let shift = inactive.iter().sum::<f64>() / n;
        let mad = inactive.iter().map(|v| (v - shift).abs()).sum::<f64>() / n;
        assert!(
            mad < 0.25 * shift.abs(),
            "the two fits differ in SHAPE, not merely in LEVEL (mad {mad} against shift \
             {shift}), so a single-number correction is not justified and this \
             reconciliation would be measuring the wrong thing"
        );

        let mut checked = 0usize;
        let mut worst = 0.0f64;
        for i in 0..on.yhat.len() {
            if roll[i].abs() < 1e-9 {
                continue; // inactive rows carry no event contribution to reconcile
            }
            let attributable = (on.yhat[i] - off.yhat[i]) - shift;
            let err = (attributable - roll[i]).abs() / PLANTED_EFFECT;
            worst = worst.max(err);
            assert!(
                err <= bar,
                "row {i}: the events-ON minus events-OFF movement attributable to the \
                 events is {attributable} but the published roll-up says {}, a relative \
                 miss of {err} against the {bar} bar",
                roll[i]
            );
            checked += 1;
        }
        assert!(
            checked > 0,
            "no forecast row carried a non-zero event contribution, so this case \
             reconciled nothing"
        );

        // THE CONTROL THAT MAKES THIS A UNITS CHECK. A roll-up left in normalised
        // training units would be smaller by a factor of NpData::scale, and it must FAIL
        // the same comparison the correct one just passed — otherwise this test would be
        // green on the very defect it exists to catch.
        let scale = planted_series_scale();
        let mut unscaled_worst = 0.0f64;
        for i in 0..on.yhat.len() {
            if roll[i].abs() < 1e-9 {
                continue;
            }
            let attributable = (on.yhat[i] - off.yhat[i]) - shift;
            unscaled_worst =
                unscaled_worst.max((attributable - roll[i] / scale).abs() / PLANTED_EFFECT);
        }
        assert!(
            unscaled_worst > bar,
            "a roll-up in normalised units would reconcile to {unscaled_worst}, inside the \
             {bar} bar the scaled one passed at {worst} — this case cannot tell the two \
             unit systems apart and proves nothing"
        );
    }

    /// SC3 THROUGH THE PUBLIC DOOR, at the configuration that actually ships.
    ///
    /// Plan 06.1-05's `events::tests` run against a hand-built `TrainConfig`. The door uses
    /// its OWN configuration — `np::door_lr_sweep`, `np::door_epochs`, empty `ar_layers`
    /// lag-free, `weight_decay: 1e-3`, `huber_beta: 0.3`, `newer_w: 2.0`, and the learning
    /// rate selected by TRAIN loss — and SC3's bar binds on what SHIPS. A recovery bar
    /// proven only under a test harness's own hyperparameters is a bar about the harness.
    ///
    /// Only `n_lags = 0` appears here because `n_lags > 0` with holidays is now REFUSED at
    /// the door; that half is
    /// [`tests::the_neuralprophet_arm_refuses_holidays_with_lags_naming_the_limitation`].
    /// The bar is not being asserted over a configuration it fails — that configuration is
    /// refused.
    #[test]
    fn a_planted_event_effect_is_recovered_through_the_door() {
        let bar = equation_tolerance("neuralprophet-parity-v1", "event_effect_recovery_rel");
        let args = planted_np_args(PLANTED_POINTS, PLANTED_HORIZON, 0, 42);
        let r = forecast(&args).expect("the planted request must be accepted");
        let mut worst = 0.0f64;
        for name in ["promo", "blackfriday"] {
            let (peak, active) = event_component_peak(&r, name);
            assert!(
                active > 0,
                "{name} has no active day in the forecast window, so its recovery is \
                 untested rather than proven"
            );
            // EVERY active row of a per-event component must recover the planted effect —
            // the peak alone could hide a column that recovered nothing.
            for (i, v) in component_values(&r, name).iter().enumerate() {
                if v.abs() < 1e-9 {
                    continue;
                }
                let rel = (v - PLANTED_EFFECT).abs() / PLANTED_EFFECT;
                worst = worst.max(rel);
                assert!(
                    rel <= bar,
                    "{name} row {i}: recovered {v} against a planted {PLANTED_EFFECT}, a \
                     relative error of {rel} over the {bar} bar (peak {peak})"
                );
            }
        }
        assert!(
            worst > 0.0,
            "no component row was compared, so the bar was never exercised"
        );
    }

    /// D-10 determinism, THROUGH THE DOOR, with a control that makes it able to fail.
    ///
    /// Two runs at one seed must be bit-identical in `yhat` AND in every component; a run
    /// at a DIFFERENT seed must differ. A determinism check without the second half cannot
    /// fail — it would pass on an implementation that ignored the seed entirely.
    #[test]
    fn door_level_event_training_is_deterministic_at_a_fixed_seed() {
        let args = planted_np_args(DETERMINISM_POINTS, 20, 0, 42);
        let a = forecast(&args).expect("run A must be accepted");
        let b = forecast(&args).expect("run B must be accepted");
        assert_eq!(
            a.yhat.to_bits_vec(),
            b.yhat.to_bits_vec(),
            "two runs at seed 42 must be BIT-identical in yhat"
        );
        for name in ["promo", "blackfriday", "holidays", "trend"] {
            assert_eq!(
                component_values(&a, name).to_bits_vec(),
                component_values(&b, name).to_bits_vec(),
                "two runs at seed 42 must be BIT-identical in the {name:?} component"
            );
        }

        // CONTROL: a different seed must produce a different fit, or the check above is
        // satisfied by an implementation that never reads the seed.
        let mut other = args.clone();
        other.seed = Some(43);
        let c = forecast(&other).expect("the differing-seed control must be accepted");
        assert_ne!(
            a.yhat.to_bits_vec(),
            c.yhat.to_bits_vec(),
            "seed 43 must NOT reproduce seed 42's forecast, or the determinism assertion \
             above cannot fail"
        );
    }

    /// SC4 / C-08 — the event columns are PRICED, before the first `np::train`.
    ///
    /// Arithmetic only: no fit is run, because what is under test is the number the door
    /// compares against `MAX_NP_TRAIN_COST`, not the work it then spends.
    #[test]
    fn the_event_column_count_raises_the_priced_cost() {
        let ds: Vec<i64> = (0..600).map(|i| days_from_civil(2020, 1, 1) + i).collect();
        let y: Vec<f64> = (0..600).map(|i| 100.0 + f64::from(i) * 0.01).collect();
        let d = crate::np::NpData::new(&ds, &y, ds.len(), 10, 0.8);
        let without = crate::np::request_train_cost(&d, ds.len(), 0, 0, 0);
        let with = crate::np::request_train_cost(&d, ds.len(), 0, 6, 0);
        assert!(
            with > without,
            "a request carrying 6 event columns must be priced ABOVE the same request \
             without them; got {with} <= {without}"
        );
        assert_eq!(
            crate::np::request_train_cost(&d, ds.len(), 0, 0, 0),
            without,
            "the price at ZERO event columns must be bit-identical to the pre-event \
             arithmetic"
        );
    }

    // ---- the four hoisted bounds, RE-MUTATED IN THE NEURALPROPHET SCOPE ----
    //
    // CLAUDE.md Verification Discipline rule 4: extending a guard's SCOPE requires
    // re-mutating in the new scope; the old proof does not transfer. Each of the four
    // cases below is paired with a POSITIVE CONTROL that the same shape just inside the
    // bound is ACCEPTED on this arm, so a bound cannot pass by refusing everything.

    /// A neuralprophet request carrying `n_holidays` holidays, each `width` columns wide and
    /// each carrying `dates` occurrences, over a `points`-day CONTIGUOUS daily series.
    fn np_holiday_args(
        points: usize,
        horizon: usize,
        n_holidays: usize,
        width: i64,
        dates: usize,
    ) -> ForecastArgs {
        let mut args = np_args(points, horizon);
        let t0 = days_from_civil(2020, 1, 1);
        let lower = -((width - 1) / 2);
        args.holidays = Some(
            (0..n_holidays)
                .map(|h| crate::types::HolidayArg {
                    name: format!("h{h}"),
                    dates: (0..dates).map(|k| format_ymd(t0 + k as i64)).collect(),
                    lower_window: lower,
                    upper_window: width - 1 + lower,
                })
                .collect(),
        );
        args
    }

    #[test]
    fn the_neuralprophet_arm_pays_the_holiday_name_ceiling() {
        let mut args = np_holiday_args(60, 7, 1, 1, 1);
        args.holidays.as_mut().expect("holidays")[0].name =
            "n".repeat(crate::types::MAX_HOLIDAY_NAME_LEN + 1);
        refusal(&args, "max_holiday_name_len");

        let mut ok = np_holiday_args(60, 7, 1, 1, 1);
        ok.holidays.as_mut().expect("holidays")[0].name =
            "n".repeat(crate::types::MAX_HOLIDAY_NAME_LEN);
        forecast(&ok).expect("a name exactly AT the bound must be accepted on this arm too");
    }

    #[test]
    fn the_neuralprophet_arm_pays_the_holiday_column_ceiling() {
        // MAX_HOLIDAY_COLUMNS + 1 columns from two holidays, each far inside every other
        // bound, so only the column ceiling can refuse.
        let width = (crate::types::MAX_HOLIDAY_COLUMNS / 2 + 1) as i64;
        let args = np_holiday_args(60, 7, 2, width, 1);
        refusal(&args, "max_holiday_columns");

        let ok = np_holiday_args(60, 7, 1, 9, 1);
        forecast(&ok).expect("a column count far inside the ceiling must be accepted");
    }

    #[test]
    fn the_neuralprophet_arm_pays_the_aggregate_holiday_dates_ceiling() {
        let args = np_holiday_args(60, 7, 11, 1, 1_000);
        refusal(&args, "max_holiday_dates_total");

        let ok = np_holiday_args(60, 7, 2, 1, 50);
        forecast(&ok).expect("an aggregate far inside the ceiling must be accepted");
    }

    /// D-33's REASON FOR EXISTING: a GAPPY neuralprophet series whose POINT count clears the
    /// design-cost ceiling but whose imputed-grid SPAN does not.
    ///
    /// This is the case the operand change exists for, and it is the one a verbatim
    /// inheritance of the prophet operand would have accepted while the arm paid
    /// `span/points` times the priced work. The refusal must NAME the span, and the paired
    /// DENSE control — the same point count and the same column count, no gaps — must be
    /// ACCEPTED, so the refusal cannot pass by refusing everything.
    #[test]
    fn a_gappy_neuralprophet_series_is_refused_on_the_holiday_span_operand() {
        // 40 points, 30 columns, horizon 10: dense, the operand is 40 + 10 = 50 rows and
        // 50 x 30 = 1 500 cells, far inside max_holiday_design_cost (50 000).
        let dense = np_gappy_holiday_args(40, 10, 30, 1);
        let r = forecast(&dense)
            .expect("the DENSE control at the same point and column count must be accepted");
        assert_eq!(r.yhat.len(), 10, "the control must really have forecast");

        // The SAME 40 points and 30 columns, spread 100 days apart: the point count is
        // unchanged but the imputed grid is 3 901 days, so (3 901 + 10) x 30 = 117 330
        // cells, over the ceiling. Only the OPERAND distinguishes these two requests.
        let gappy = np_gappy_holiday_args(40, 10, 30, 100);
        match forecast(&gappy) {
            Err(ForecastError::Validation(m)) => {
                assert!(
                    m.contains("max_holiday_design_cost"),
                    "the design-cost ceiling must refuse, naming its key; got {m:?}"
                );
                assert!(
                    m.contains("span_days"),
                    "the refusal must NAME the span operand it used, or a caller cannot \
                     tell why the same point count is accepted on the prophet arm; got {m:?}"
                );
                assert!(
                    !m.contains("(points + horizon)"),
                    "the neuralprophet arm must NOT claim it priced on the point count; \
                     got {m:?}"
                );
            }
            other => panic!(
                "expected a Validation refusal, got {:?}",
                other.map(|r| r.model)
            ),
        }

        // CONTROL: the very same GAPPY request on the PROPHET arm is ACCEPTED, because its
        // design really is built per supplied row. This is what proves the two operands
        // differ rather than that the gappy series is simply too big for anything.
        let mut on_prophet = gappy.clone();
        on_prophet.model = None;
        forecast(&on_prophet).expect(
            "the same gappy request must be ACCEPTED on the prophet arm, whose \
                     design is built per supplied row",
        );
    }

    /// A neuralprophet request whose `points` rows are `step` days apart, carrying one
    /// holiday `width` columns wide. `step = 1` is the dense control; `step > 1` is gappy.
    fn np_gappy_holiday_args(points: usize, horizon: usize, width: i64, step: i64) -> ForecastArgs {
        let t0 = days_from_civil(2020, 1, 1);
        let mut rng = crate::prophet::Rng::new(7);
        let mut args = np_args(points, horizon);
        args.ds = (0..points)
            .map(|i| format_ymd(t0 + i as i64 * step))
            .collect();
        args.y = (0..points)
            .map(|i| 10.0 + 0.01 * i as f64 + 0.05 * rng.normal())
            .collect();
        args.holidays = Some(vec![crate::types::HolidayArg {
            name: "promo".into(),
            dates: vec![format_ymd(t0 + 5)],
            lower_window: -((width - 1) / 2),
            upper_window: width - 1 - ((width - 1) / 2),
        }]);
        args
    }

    // ---- the planted-effect harness the door-level SC3 cases share ----

    /// The planted per-indicator magnitude, in RESPONSE units. The recovery BAR is
    /// contract-read; this is the harness's own INPUT, which is not a bar.
    const PLANTED_EFFECT: f64 = 8.0;
    /// The series length the door-level recovery cases use. 06.1-05 measured the lag-free
    /// recovery at this length (worst 0.0602 against the 0.10 bar), so the configuration
    /// these cases assert over is the one that has been measured.
    const PLANTED_POINTS: usize = 1_200;
    /// A horizon long enough to contain several active event days — asserted, not assumed,
    /// by the `active > 0` guard in every case that reads a component.
    const PLANTED_HORIZON: usize = 40;
    /// The determinism case does not read a weight's VALUE, so it does not pay for
    /// convergence: a shorter series is the same wiring claim for less compute.
    const DETERMINISM_POINTS: usize = 240;

    /// Two non-overlapping recurring events: `promo` every 30 days with a +/-1 window and
    /// `blackfriday` 15 days later with a 0..2 window — six columns in all, and never two
    /// active on the same day, so a component row reads ONE column's weight and a recovery
    /// result is a statement about that column.
    fn planted_events(t0: i64, span: usize) -> Vec<crate::types::HolidayArg> {
        let last = span as i64 + 120;
        vec![
            crate::types::HolidayArg {
                name: "promo".into(),
                // A BOUNDED range, not `(0..).take_while(..)`: clippy's
                // `maybe_infinite_iter` denies the open form, and the occurrence count is
                // exactly derivable from the span anyway.
                dates: (0..=last / 30).map(|k| format_ymd(t0 + k * 30)).collect(),
                lower_window: -1,
                upper_window: 1,
            },
            crate::types::HolidayArg {
                name: "blackfriday".into(),
                dates: (0..=last / 30)
                    .map(|k| format_ymd(t0 + 15 + k * 30))
                    .collect(),
                lower_window: 0,
                upper_window: 2,
            },
        ]
    }

    /// A synthetic daily series carrying a KNOWN additive effect on every active indicator,
    /// planted THROUGH the same window expansion the block reads — so a recovery result is
    /// a statement about the block, not about the harness agreeing with itself.
    fn planted_np_args(points: usize, horizon: usize, n_lags: usize, seed: u64) -> ForecastArgs {
        let t0 = days_from_civil(2018, 1, 1);
        let holidays = planted_events(t0, points);
        let active = planted_active_days(&holidays);
        let mut rng = crate::prophet::Rng::new(7);
        let mut ds = Vec::with_capacity(points);
        let mut y = Vec::with_capacity(points);
        for i in 0..points {
            let day = t0 + i as i64;
            let t = i as f64;
            let mut v = 100.0
                + 0.02 * t
                + 6.0 * (2.0 * std::f64::consts::PI * t / 365.25).sin()
                + 3.0 * (2.0 * std::f64::consts::PI * t / 7.0).sin();
            if active.contains(&day) {
                v += PLANTED_EFFECT;
            }
            v += 0.4 * rng.normal();
            ds.push(format_ymd(day));
            y.push(v);
        }
        let mut args = np_args(points, horizon);
        args.ds = ds;
        args.y = y;
        args.holidays = Some(holidays);
        args.n_lags = if n_lags == 0 { None } else { Some(n_lags) };
        args.seed = Some(seed);
        args
    }

    /// Every day on which some planted indicator column is active.
    fn planted_active_days(
        holidays: &[crate::types::HolidayArg],
    ) -> std::collections::HashSet<i64> {
        let mut out = std::collections::HashSet::new();
        for h in holidays {
            for dt in &h.dates {
                let d = crate::dates::parse_date(dt).expect("the harness builds valid dates");
                for off in h.lower_window..=h.upper_window {
                    out.insert(d + off);
                }
            }
        }
        out
    }

    /// The `NpData::scale` the planted series produces — the factor a component left in
    /// normalised units would be wrong by.
    fn planted_series_scale() -> f64 {
        let args = planted_np_args(PLANTED_POINTS, PLANTED_HORIZON, 0, 42);
        let ds: Vec<i64> = args
            .ds
            .iter()
            .map(|s| crate::dates::parse_date(s).expect("date"))
            .collect();
        crate::np::NpData::new(&ds, &args.y, ds.len(), 10, 0.8).scale
    }

    /// One component's values, or a panic NAMING the key: a component that silently
    /// defaulted to zeros is the vacuous-guard class these tests refuse.
    fn component_values(r: &crate::types::ForecastResponse, key: &str) -> Vec<f64> {
        r.components
            .get(key)
            .and_then(|v| v.as_array())
            .unwrap_or_else(|| panic!("the response must carry a {key:?} component"))
            .iter()
            .map(|v| v.as_f64().unwrap_or(f64::NAN))
            .collect()
    }

    /// `(largest absolute value, count of non-zero rows)` for one event component.
    fn event_component_peak(r: &crate::types::ForecastResponse, key: &str) -> (f64, usize) {
        let v = component_values(r, key);
        let peak = v
            .iter()
            .fold(0.0f64, |a, b| if b.abs() > a.abs() { *b } else { a });
        let active = v.iter().filter(|x| x.abs() > 1e-9).count();
        (peak, active)
    }

    /// Bit-pattern comparison for a float vector — `assert_eq!` on `f64` would compare
    /// `-0.0 == 0.0` as equal and would not notice a NaN reproducing.
    trait ToBitsVec {
        fn to_bits_vec(&self) -> Vec<u64>;
    }
    impl ToBitsVec for [f64] {
        fn to_bits_vec(&self) -> Vec<u64> {
            self.iter().map(|v| v.to_bits()).collect()
        }
    }
    impl ToBitsVec for Vec<f64> {
        fn to_bits_vec(&self) -> Vec<u64> {
            self.as_slice().to_bits_vec()
        }
    }

    /// `MAX_POINTS` bounds how MANY points arrive, never how far apart they are, and the
    /// NeuralProphet path materialises an imputed daily grid over the whole span.
    #[test]
    fn a_span_far_wider_than_the_point_count_is_refused() {
        let mut args = np_args(12, 7);
        args.ds = (0..12)
            .map(|i| format_ymd(days_from_civil(1500, 1, 1) + i * 30_000))
            .collect();
        refusal(&args, "max_span_days");
    }

    /// The two window fields were only SIGN-checked, so two integers expanded into an
    /// unbounded number of design columns from a ~200-byte request.
    #[test]
    fn an_enormous_holiday_window_is_refused() {
        let (ds, y) = synthetic_daily(60);
        let mut args = np_args(60, 7);
        args.model = None;
        args.ds = ds;
        args.y = y;
        args.holidays = Some(vec![crate::types::HolidayArg {
            name: "x".into(),
            dates: vec!["2020-02-01".into()],
            lower_window: -2_000_000_000,
            upper_window: 0,
        }]);
        refusal(&args, "365");
    }

    /// A prophet request carrying `n_holidays` holidays, each `width` design columns wide
    /// and each carrying `dates` occurrences. Every knob here is INSIDE its own bound; the
    /// tests below vary only which PRODUCT goes over.
    fn holiday_args(
        points: usize,
        horizon: usize,
        n_holidays: usize,
        width: i64,
        dates: usize,
    ) -> ForecastArgs {
        let (ds, y) = synthetic_daily(points);
        let mut args = np_args(points, horizon);
        args.model = None;
        args.ds = ds;
        args.y = y;
        let t0 = days_from_civil(2020, 1, 1);
        let lower = -((width - 1) / 2);
        args.holidays = Some(
            (0..n_holidays)
                .map(|h| crate::types::HolidayArg {
                    name: format!("h{h}"),
                    dates: (0..dates).map(|k| format_ymd(t0 + k as i64)).collect(),
                    lower_window: lower,
                    upper_window: width - 1 + lower,
                })
                .collect(),
        );
        args
    }

    /// MAX_POINTS, MAX_HORIZON, MAX_HOLIDAY_COLUMNS and MAX_HOLIDAY_DATES are each checked
    /// in ISOLATION; their product was not. 200 points, a 100-step horizon and one
    /// 400-column holiday are all comfortably legal and together buy 120 000 design
    /// feature cells — and every fit iteration is `O(rows * K)` over exactly that matrix.
    #[test]
    fn an_in_bounds_holiday_spec_whose_product_is_not_is_refused() {
        let args = holiday_args(200, 100, 1, 400, 5);
        refusal(&args, "max_holiday_design_cost");
    }

    /// `MAX_HOLIDAY_COLUMNS` on the PROPHET arm — and it had NO test on EITHER arm until
    /// this plan.
    ///
    /// Found by the rule-4 re-mutation rather than by reading: disabling the column
    /// ceiling at the shared hoisted site reddened the neuralprophet case and NOTHING
    /// else, where every other bound reddened a prophet case beside it. MEASURED at the
    /// phase base (`94ce60bb7`): `max_holiday_columns` occurs exactly once in the whole
    /// file, inside the refusal message, so no test asserted on it. The ceiling has been
    /// shipping unguarded.
    ///
    /// This is the POSITIVE CONTROL half of the re-mutation pair — it proves a mutation of
    /// the shared site engages on the prophet arm too — and it is a real missing guard in
    /// its own right.
    #[test]
    fn holiday_columns_over_the_ceiling_are_refused_on_the_prophet_arm() {
        // Two holidays, 501 columns each: 1 002 > MAX_HOLIDAY_COLUMNS (1 000). The in-loop
        // ceiling fires at the second holiday, before the post-loop design-cost check.
        let args = holiday_args(60, 7, 2, 501, 1);
        refusal(&args, "max_holiday_columns");

        // The NEAR MISS: exactly AT the ceiling must still be accepted, or the bound is
        // proven only to refuse and not to refuse just what it claims. 1 000 columns over
        // (60 + 7) rows is 67 000 design cells, so the design-cost ceiling would refuse
        // this — the near miss is therefore taken on the COLUMN count alone, at a horizon
        // and history short enough to clear the product.
        let ok = holiday_args(10, 10, 2, 500, 1);
        let r = forecast(&ok).expect("a column count exactly AT the ceiling must be accepted");
        assert_eq!(r.yhat.len(), 10, "one row per horizon step");
    }

    /// D-33 — the design-cost refusal STATES THE OPERAND IT USED.
    ///
    /// One ceiling now serves two arms with different arithmetic: the caller's POINT COUNT
    /// on prophet and the imputed-grid SPAN IN DAYS on neuralprophet. A refusal that named
    /// neither would leave a caller unable to tell why the same series is accepted on one
    /// arm and refused on the other. This half pins the PROPHET form; the neuralprophet
    /// span form is pinned by
    /// [`tests::a_gappy_neuralprophet_series_is_refused_on_the_holiday_span_operand`], which cannot
    /// exist until the arm accepts holidays at all.
    #[test]
    fn the_design_cost_refusal_names_the_operand_it_used_on_the_prophet_arm() {
        let args = holiday_args(200, 100, 1, 400, 5);
        match forecast(&args) {
            Err(ForecastError::Validation(m)) => {
                assert!(
                    m.contains("(points + horizon) x holiday_columns"),
                    "the prophet arm prices on the POINT COUNT and the message must say so; \
                     got {m:?}"
                );
                assert!(
                    !m.contains("span_days"),
                    "the prophet arm must NOT claim it priced on the span; got {m:?}"
                );
                assert!(
                    m.contains("(200 + 100) x 400"),
                    "the message must report both factors and their VALUES; got {m:?}"
                );
            }
            other => panic!(
                "expected a Validation refusal, got {:?}",
                other.map(|r| r.model)
            ),
        }
    }

    /// The NEAR MISS. Exactly at the bound — (50 + 50) x 500 = 50 000 — which the door
    /// must ACCEPT, because a bound proven only to refuse is not proven to refuse just
    /// what it claims.
    #[test]
    fn a_holiday_spec_just_under_the_design_cost_bound_is_accepted() {
        let args = holiday_args(50, 50, 1, 500, 5);
        let r = forecast(&args).expect("a request exactly at the bound must fit, not refuse");
        assert_eq!(r.yhat.len(), 50, "one row per horizon step");
        assert_eq!(r.yhat_lower.len(), 50);
        assert_eq!(r.yhat_upper.len(), 50);
    }

    /// MAX_HOLIDAY_DATES bounds ONE holiday's list; nothing bounded the SUM. Eleven
    /// holidays at the per-holiday ceiling are 11 000 dates from one request — and only
    /// 11 design columns, so the design-cost bound cannot see them.
    #[test]
    fn holidays_carrying_more_dates_than_the_total_bound_are_refused() {
        let args = holiday_args(60, 7, 11, 1, 1_000);
        refusal(&args, "max_holiday_dates_total");
    }

    /// WR-03 — the aggregate ceiling is enforced INSIDE the loop, not after it.
    ///
    /// This test discriminates POSITION, not presence. The running sum crosses
    /// [`MAX_HOLIDAY_DATES_TOTAL`](crate::types::MAX_HOLIDAY_DATES_TOTAL) at the ELEVENTH
    /// holiday, and every LATER holiday carries a date string `parse_date`'s SHAPE gate
    /// rejects (`"2020-1-01"` — nine bytes, so it is refused for its shape and not for an
    /// out-of-range calendar field, which is a different message). If the aggregate refusal
    /// fired after the loop, those later holidays would be parsed first and the caller would
    /// receive the DATE-SHAPE refusal instead. The message that comes back therefore names
    /// which check ran first — something no grep for the constant could establish.
    #[test]
    fn the_aggregate_dates_refusal_fires_inside_the_holiday_loop() {
        let (ds, y) = synthetic_daily(60);
        let mut args = np_args(60, 7);
        args.model = None;
        args.ds = ds;
        args.y = y;
        let base = days_from_civil(1990, 1, 1);
        // 11 x 1 000 = 11 000 > MAX_HOLIDAY_DATES_TOTAL (10 000): the running sum crosses at
        // the eleventh holiday, with four holidays still unparsed behind it.
        let mut holidays: Vec<crate::types::HolidayArg> = (0..11i64)
            .map(|h| crate::types::HolidayArg {
                name: format!("h{h}"),
                dates: (0..1_000i64)
                    .map(|d| format_ymd(base + h * 1_000 + d))
                    .collect(),
                lower_window: 0,
                upper_window: 0,
            })
            .collect();
        for h in 11..15i64 {
            holidays.push(crate::types::HolidayArg {
                name: format!("late{h}"),
                dates: vec!["2020-1-01".into()],
                lower_window: 0,
                upper_window: 0,
            });
        }
        args.holidays = Some(holidays);
        match forecast(&args) {
            Err(ForecastError::Validation(m)) => {
                assert!(
                    !m.contains("want YYYY-MM-DD"),
                    "a DATE-SHAPE refusal means the loop kept parsing holidays after the door \
                     already had the information to refuse — the aggregate check is still \
                     AFTER the loop; got {m:?}"
                );
                assert!(
                    m.contains("max_holiday_dates_total"),
                    "the aggregate ceiling must refuse, naming its own key; got {m:?}"
                );
            }
            other => panic!(
                "expected a Validation refusal, got {:?}",
                other.map(|r| r.model)
            ),
        }
    }

    /// C-08 — the NeuralProphet training path had NO budget of any kind.
    ///
    /// The worst LEGAL request (20 000 contiguous daily points, `n_lags` 365, horizon 3650)
    /// prices at 718 641 000 and walled at **47.924 s** on a release build — 24x SC1's 2 s
    /// bar — because `fit::FIT_BUDGET_SECS` is read only inside `fit::fit_prophet`, which
    /// this arm never enters. The refusal names the observed cost, all four factors, the
    /// bound key and what to reduce.
    #[test]
    fn a_neuralprophet_request_over_the_train_cost_bound_is_refused() {
        // Same shape as the structural maximum, at a history short enough that the REFUSAL
        // (which fires before any training) stays instant.
        let mut args = np_args(2_000, 365);
        args.n_lags = Some(365);
        refusal(&args, "max_np_train_cost");
    }

    /// The positive control the always-run suite CAN afford: a real NeuralProphet request
    /// with lags on, priced under the bound, accepted through the door and returning the
    /// full response shape. The at-the-bound near miss is in `np::wall::np_train_wall`
    /// (`NP_WALL_MODE=at_bound_*`, three compositions at 93-99% of the bound, 1.402-1.642 s
    /// on release) because at the bound ONE request costs 45.619 s on a debug profile.
    #[test]
    fn a_neuralprophet_request_under_the_train_cost_bound_is_accepted() {
        let mut args = np_args(120, 7);
        args.n_lags = Some(7);
        let r = forecast(&args).expect("a request under the training-cost bound must fit");
        assert_eq!(r.model, "neuralprophet");
        assert_eq!(r.yhat.len(), 7, "one row per horizon step");
        assert_eq!(r.yhat_lower.len(), 7);
        assert_eq!(r.yhat_upper.len(), 7);
    }

    /// The bound must not refuse the geometry the NeuralProphet parity ladder proves the
    /// model CORRECT on.
    ///
    /// `np::parity` runs Peyton Manning (2 905 rows over a 2 964-day span) at `n_lags` 0 and
    /// 30. Through the door those price at 697 200 and 14 552 640, and the second is 97% of
    /// the bound — so a value of 14 000 000, which the wall measurements would also have
    /// allowed, would refuse the ladder's own request. This is arithmetic on the door's own
    /// pricing functions, so it costs nothing to run and turns red the moment the bound is
    /// lowered past the ladder.
    #[test]
    fn the_np_parity_ladder_geometry_prices_under_the_train_cost_bound() {
        const PEYTON_ROWS: usize = 2_905;
        const PEYTON_SPAN_DAYS: usize = 2_964;
        for n_lags in [0usize, 30] {
            let n_samples = if n_lags == 0 {
                PEYTON_ROWS
            } else {
                PEYTON_SPAN_DAYS - n_lags
            };
            let epochs = crate::np::door_epochs(PEYTON_ROWS, n_samples, n_lags);
            let sweep = crate::np::door_lr_sweep(n_lags).len() as u64;
            let cost = crate::np::train_cost(n_samples, epochs, n_lags, 0, 0) * sweep;
            assert!(
                cost <= MAX_NP_TRAIN_COST,
                "np::parity's own Peyton rung at n_lags={n_lags} prices at {cost}, which the \
                 bound {MAX_NP_TRAIN_COST} would REFUSE — a bound below the geometry the \
                 ladder proves correctness on is wrong"
            );
        }
    }

    /// The comparison is `>`, not `>=`: a request pricing EXACTLY at the bound is accepted.
    ///
    /// No near-miss this crate can afford to run reaches the boundary itself — at the bound
    /// one request costs 45.619 s on a debug profile — so the off-by-one is pinned on the
    /// door's own comparison instead of on a request nobody will pay for.
    #[test]
    fn the_np_train_cost_bound_is_exclusive_not_inclusive() {
        assert!(
            !super::np_train_cost_is_over(MAX_NP_TRAIN_COST),
            "a request pricing EXACTLY at max_np_train_cost must be accepted"
        );
        assert!(
            super::np_train_cost_is_over(MAX_NP_TRAIN_COST + 1),
            "one unit over the bound must be refused"
        );
    }

    /// The door prices a request with the SAME functions it then uses to configure the
    /// sweep, so the bound cannot be evaded by the two disagreeing — the drift hazard
    /// `prophet::changepoint_count` was extracted to avoid in 06-14.
    #[test]
    fn the_door_prices_a_request_at_exactly_the_work_it_configures() {
        let (ds, y) = synthetic_daily(120);
        let days: Vec<i64> = ds
            .iter()
            .map(|s| crate::dates::parse_date(s).expect("valid"))
            .collect();
        for n_lags in [0usize, 7, 30] {
            let d = crate::np::NpData::new(&days, &y, days.len(), 10, 0.8);
            let n_points = days.len();
            let n_samples = crate::np::n_training_samples(&d, n_lags);
            let epochs = crate::np::door_epochs(n_points, n_samples, n_lags);
            let sweep = crate::np::door_lr_sweep(n_lags).len() as u64;
            assert_eq!(
                crate::np::request_train_cost(&d, n_points, n_lags, 0, 0),
                crate::np::train_cost(n_samples, epochs, n_lags, 0, 0) * sweep,
                "request_train_cost must be exactly sweep x train_cost at n_lags={n_lags}"
            );
        }
    }

    /// C-07 — `holidays[].name` had NO enforcement of any kind at the close of 06-14.
    ///
    /// One payload occurrence becomes TWO owned `String`s per design column in
    /// `prophet::columns` and then the sort key of an `O(C log C)` byte-wise comparison
    /// sort, so at `MAX_HOLIDAY_COLUMNS` a single name is amplified ~2 000:1. The refusal
    /// names the holiday's POSITION and the observed LENGTH and never the name itself —
    /// echoing an oversized string back re-materialises the very bytes the bound refuses
    /// (T-06-38).
    #[test]
    fn a_holiday_name_over_the_length_bound_is_refused() {
        let args = named_holiday_args("n".repeat(crate::types::MAX_HOLIDAY_NAME_LEN + 1));
        refusal(&args, "max_holiday_name_len");
    }

    /// The NEAR MISS: exactly at the bound, which must still be accepted and must still
    /// return the full response. A comparison written as `>=` fails here and nowhere else.
    #[test]
    fn a_holiday_name_at_the_length_bound_is_accepted() {
        let name = "n".repeat(crate::types::MAX_HOLIDAY_NAME_LEN);
        assert_eq!(
            name.len(),
            crate::types::MAX_HOLIDAY_NAME_LEN,
            "this control is only a NEAR MISS if it sits exactly ON the bound"
        );
        let args = named_holiday_args(name);
        let r = forecast(&args).expect("a holiday name exactly at the bound must fit");
        assert_eq!(r.yhat.len(), 7, "one row per horizon step");
    }

    /// The bound is on BYTES, and this is the case that says so. `é` is two UTF-8 bytes, so
    /// this name's CHARACTER count is comfortably under the bound while its BYTE length is
    /// over it — and bytes are what the clone and the byte-wise comparison actually cost.
    /// A rewrite of `h.name.len()` to `h.name.chars().count()` turns this test red and
    /// nothing else in the suite.
    #[test]
    fn a_holiday_name_whose_char_count_fits_but_whose_byte_length_does_not_is_refused() {
        let chars = crate::types::MAX_HOLIDAY_NAME_LEN / 2 + 1;
        let name = "é".repeat(chars);
        assert!(
            name.chars().count() <= crate::types::MAX_HOLIDAY_NAME_LEN,
            "the CHAR count must sit under the bound, or this case proves nothing about \
             which quantity is measured"
        );
        assert!(
            name.len() > crate::types::MAX_HOLIDAY_NAME_LEN,
            "the BYTE length must sit over the bound"
        );
        let args = named_holiday_args(name);
        refusal(&args, "max_holiday_name_len");
    }

    /// T-06-38 — the refusal names the LENGTH, never the name.
    ///
    /// Echoing an oversized attacker-controlled string back re-materialises the very bytes
    /// the bound refuses and reflects that content into every log the refusal reaches. The
    /// assertion is non-vacuous by construction: `TOKEN` is what the message WOULD contain
    /// if `h.name` were interpolated, and every OTHER refusal in the same loop does
    /// interpolate `h.name` — which is exactly why this check is FIRST in the loop.
    #[test]
    fn the_oversized_name_refusal_names_the_length_and_not_the_name() {
        const TOKEN: &str = "SECRETHOLIDAYLABEL";
        let over = crate::types::MAX_HOLIDAY_NAME_LEN + 1;
        let name = TOKEN.repeat(over.div_ceil(TOKEN.len()))[..over].to_string();
        assert!(
            name.contains(TOKEN),
            "the probe token must survive truncation"
        );
        let args = named_holiday_args(name.clone());
        match forecast(&args) {
            Err(ForecastError::Validation(m)) => {
                assert!(
                    m.contains("max_holiday_name_len"),
                    "the refusal must name the bound key; got {m:?}"
                );
                assert!(
                    m.contains(&over.to_string()),
                    "the refusal must report the OBSERVED byte length ({over}); got {m:?}"
                );
                assert!(
                    m.contains("index 0"),
                    "the refusal must name WHICH holiday; got {m:?}"
                );
                assert!(
                    !m.contains(TOKEN),
                    "the refusal must NOT echo the oversized name back (T-06-38); got {m:?}"
                );
            }
            other => panic!(
                "expected a Validation refusal, got {:?}",
                other.map(|r| r.model)
            ),
        }
    }

    /// The bound is ONE-SIDED by design. An empty name is a separate question (does a
    /// component need a label at all?) that this plan does not open, and a length bound
    /// that quietly grew a lower side would be answering it by accident.
    #[test]
    fn an_empty_holiday_name_is_not_refused_by_the_length_bound() {
        let args = named_holiday_args(String::new());
        let r = forecast(&args).expect("the length bound is one-sided: it must not refuse 0");
        assert_eq!(r.yhat.len(), 7);
    }

    /// A prophet request carrying exactly one holiday whose NAME is the variable under test.
    /// Every other knob is comfortably inside its own bound, so only the name can refuse.
    fn named_holiday_args(name: String) -> ForecastArgs {
        let (ds, y) = synthetic_daily(60);
        let mut args = np_args(60, 7);
        args.model = None;
        args.ds = ds;
        args.y = y;
        args.holidays = Some(vec![crate::types::HolidayArg {
            name,
            dates: vec![format_ymd(days_from_civil(2020, 2, 1))],
            lower_window: 0,
            upper_window: 0,
        }]);
        args
    }

    /// A HOLIDAY name that is a reserved response key is refused, on BOTH arms.
    ///
    /// The regressor family had this rule from the start; holidays are the OTHER
    /// caller-controlled name family and did not, although a holiday name becomes a
    /// component key on both arms — `Column.component` via `prophet::columns` on prophet,
    /// the D-31 per-event insert on neuralprophet. The map is built with
    /// `components.insert(name, value)`, so the collision was a silent overwrite, not an
    /// error.
    ///
    /// MEASURED before the fix, on the neuralprophet arm: `{"holidays":[{"name":"trend",
    /// ..}]}` returned `components.trend` as the event indicator — `[0.0, 0.0, 0.0, ..]` —
    /// in place of the real trend series `[16.016, 16.110, 16.204, ..]` that the same
    /// request with the holiday named `xmas` returns. That is the D-21 silent-ignore class
    /// (T-06.1-13), so it is a refusal and not a rename.
    ///
    /// Every key in `RESERVED_RESPONSE_KEYS` is exercised, so the check cannot be narrowed
    /// to the two that happen to be live component keys today.
    #[test]
    fn a_holiday_named_after_a_reserved_response_key_is_refused_on_both_arms() {
        for model in [None, Some("neuralprophet".to_string())] {
            for key in crate::forecast::RESERVED_RESPONSE_KEYS {
                let mut args = named_holiday_args((*key).to_string());
                args.model.clone_from(&model);
                if model.is_some() {
                    args.n_lags = Some(0);
                }
                let err = forecast(&args).expect_err(&format!(
                    "a holiday named {key:?} must be refused on model {model:?}: it would \
                     silently overwrite the response component of the same name"
                ));
                let msg = format!("{err:?}");
                assert!(
                    msg.contains("already a reserved response key"),
                    "refusal for {key:?} on {model:?} must NAME the rule, got: {msg}"
                );
            }
        }
    }

    /// A holiday named after a SEASONALITY is refused — the other half of check 10.
    ///
    /// The reserved-key list does not contain `weekly`, so this case survived that check.
    /// `prophet::predict` deduplicates components BY NAME, so the holiday does not overwrite
    /// the seasonality, it MERGES into it: one `components.weekly` carrying the sum of both,
    /// no holiday component at all, and the roll-up mode taken from whichever column came
    /// first. The same name sent as a REGRESSOR is refused by
    /// `a_regressor_colliding_with_a_seasonality_component_name_is_refused`; holidays now
    /// pay the same rule.
    #[test]
    fn a_holiday_named_after_a_seasonality_component_is_refused() {
        let args = named_holiday_args("weekly".to_string());
        let err =
            forecast(&args).expect_err("a holiday named after a live seasonality must be refused");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("a response component name"),
            "the message must say WHICH part matched, got: {msg}"
        );
    }

    /// An EMPTY holiday list is not an event surface, so it does not block `n_lags`.
    ///
    /// The refusal keyed on `is_some()`, so a client that always serialises the key — or a
    /// caller who cleared the list without dropping it — could not use lags at all, and was
    /// told to "drop holidays" for a field they had already emptied. Zero event columns are
    /// configured and zero are priced; every later use normalises emptiness the same way.
    #[test]
    fn an_empty_holiday_list_does_not_block_lags_on_neuralprophet() {
        let mut args = np_args(60, 7);
        args.model = Some("neuralprophet".to_string());
        args.n_lags = Some(7);
        args.holidays = Some(vec![]);
        forecast(&args).expect("an empty holiday list must not refuse a lagged request");
    }

    /// A CONSTANT regressor is refused even when the caller turns standardisation off.
    ///
    /// `standardize: false` makes `standardize_one` return `(0.0, 1.0)` unconditionally, so
    /// the `st.std == 0.0` test could not see the column and the flag alone admitted a
    /// driver that is collinear with the intercept. The contract's `standardize` row
    /// promises the opposite. The accepted control is the same flag on a column that DOES
    /// vary, which must still be accepted — otherwise this is a ban on the flag, not a
    /// constant-column check.
    #[test]
    fn a_constant_regressor_is_refused_even_with_standardize_false() {
        let mut flat = good_reg("promo", 60, 7);
        flat.values = vec![1.0; 67];
        flat.standardize = Some(false);
        let err = forecast(&reg_args(60, 7, vec![flat]))
            .expect_err("a constant column must be refused however it is standardised");
        assert!(
            format!("{err:?}").contains("constant over the history rows"),
            "got: {err:?}"
        );

        let mut varying = good_reg("promo", 60, 7);
        varying.standardize = Some(false);
        forecast(&reg_args(60, 7, vec![varying]))
            .expect("standardize:false on a VARYING column is still accepted");
    }

    /// The positive control for the rule above: an ordinary holiday name is still accepted,
    /// and on the neuralprophet arm it still publishes its own component beside a `trend`
    /// that is the real trend and not the event indicator.
    #[test]
    fn an_ordinary_holiday_name_is_accepted_and_does_not_disturb_the_trend() {
        let mut args = named_holiday_args("xmas".to_string());
        args.model = Some("neuralprophet".to_string());
        args.n_lags = Some(0);
        let r = forecast(&args).expect("an ordinary holiday name must be accepted");
        let v = serde_json::to_value(&r).expect("the response serialises");
        let comps = v["components"]
            .as_object()
            .expect("the response carries a component map");
        assert!(
            comps.contains_key("xmas"),
            "the event keeps its own component"
        );
        let trend = comps["trend"]
            .as_array()
            .expect("the trend component is an array");
        assert!(
            trend.iter().any(|x| x.as_f64().is_some_and(|f| f != 0.0)),
            "`trend` must be the TREND, not an all-zero event indicator: {trend:?}"
        );
    }

    /// The two-sided control for WR-03's move: a request whose aggregate is EXACTLY at
    /// `MAX_HOLIDAY_DATES_TOTAL` is still accepted. The in-loop comparison must be the same
    /// `>` the post-loop one uses; written as `>=` it would over-refuse by one date.
    #[test]
    fn holidays_carrying_exactly_the_total_bound_are_accepted() {
        let (ds, y) = synthetic_daily(60);
        let mut args = np_args(60, 7);
        args.model = None;
        args.ds = ds;
        args.y = y;
        let base = days_from_civil(1990, 1, 1);
        // 10 x 1 000 == MAX_HOLIDAY_DATES_TOTAL exactly; 10 columns and (60 + 7) x 10 = 670
        // design cells, both far inside their own bounds, so only the aggregate is at issue.
        args.holidays = Some(
            (0..10i64)
                .map(|h| crate::types::HolidayArg {
                    name: format!("h{h}"),
                    dates: (0..1_000i64)
                        .map(|d| format_ymd(base + h * 1_000 + d))
                        .collect(),
                    lower_window: 0,
                    upper_window: 0,
                })
                .collect(),
        );
        let total: usize = args
            .holidays
            .as_deref()
            .expect("holidays")
            .iter()
            .map(|h| h.dates.len())
            .sum();
        assert_eq!(
            total,
            crate::types::MAX_HOLIDAY_DATES_TOTAL,
            "this control is only a NEAR MISS if it sits exactly ON the bound"
        );
        let r = forecast(&args).expect("a request exactly at the aggregate bound must fit");
        assert_eq!(r.yhat.len(), 7, "one row per horizon step");
    }

    /// A well-formed option aimed at the wrong arm was silently DROPPED: the caller got a
    /// plausible answer to a question they did not ask.
    #[test]
    fn an_option_belonging_to_the_other_model_is_refused_not_dropped() {
        let mut prophet = np_args(60, 7);
        prophet.model = None;
        prophet.n_lags = Some(7);
        refusal(&prophet, "neuralprophet-only");

        let mut np = np_args(60, 7);
        np.growth = Some("logistic".into());
        refusal(&np, "prophet-only");

        // The cross-MODEL refusal keeps its OWN older message: a cap aimed at the
        // neuralprophet arm still says `cap is prophet-only`, never the new growth-arm one.
        let mut np_cap = np_args(60, 7);
        np_cap.cap = Some(100.0);
        refusal(&np_cap, "cap is prophet-only");

        // `cap` belongs to the LOGISTIC growth arm, not merely to the prophet MODEL. It
        // was accepted and provably dropped on every other growth arm: `make_design`
        // matches only `(Growth::Logistic, Some(c))`, so a cap sent with linear or flat
        // growth fell to `_ => None` and the caller got a plausible answer to a question
        // they did not ask. Four shapes, because one failing input is an anecdote.
        let mut linear_cap = np_args(60, 7);
        linear_cap.model = None;
        linear_cap.growth = Some("linear".into());
        linear_cap.cap = Some(100.0);
        refusal(&linear_cap, "logistic-only");

        // The DEFAULTED arm — no `growth` key at all — is the one a real caller hits, and
        // only this case proves the check is not keyed on the PRESENCE of `growth`.
        let mut bare_cap = np_args(60, 7);
        bare_cap.model = None;
        bare_cap.cap = Some(100.0);
        refusal(&bare_cap, "logistic-only");

        let mut flat_cap = np_args(60, 7);
        flat_cap.model = None;
        flat_cap.growth = Some("flat".into());
        flat_cap.cap = Some(100.0);
        refusal(&flat_cap, "logistic-only");

        // A cap BELOW max(y) off the logistic arm must refuse for being off-arm, not
        // sneak through the `cap <= y_max` rule that only guards the logistic branch.
        let mut linear_low_cap = np_args(60, 7);
        linear_low_cap.model = None;
        linear_low_cap.growth = Some("linear".into());
        linear_low_cap.cap = Some(0.5);
        refusal(&linear_low_cap, "logistic-only");

        // POSITIVE CONTROL: the refusal must not swallow the logistic happy path. A cap
        // strictly above the series maximum, computed from the args themselves so the
        // helper series may change without silently disarming this assertion.
        let mut logistic_ok = np_args(60, 7);
        logistic_ok.model = None;
        logistic_ok.growth = Some("logistic".into());
        let y_max = logistic_ok
            .y
            .iter()
            .fold(f64::NEG_INFINITY, |a, v| a.max(*v));
        logistic_ok.cap = Some(y_max + 1.0);
        let r = forecast(&logistic_ok)
            .expect("logistic growth with a cap above max(y) is a legal request");
        assert_eq!(r.yhat.len(), 7, "the logistic arm must still return a band");
    }

    // ------------------------------------------ the logistic changepoint lambda bound ---
    // `06-REVIEW.md` CR-01. A 1 132-byte request, inside every door bound, buys a Poisson
    // mean of ~86 790 new changepoints on each of 1 000 simulation rows and walls at
    // 2.334 s against SC1's 2 s bar. Three cases, because a one-sided assertion would pass
    // for a bound that refuses everything and for a bound keyed on the wrong arm.

    /// The review's exact geometry, as a library request: 33 daily points (the tightest
    /// history that still earns all 25 changepoints), `horizon: 3650`, `freq: "MS"`.
    fn logistic_lambda_args(points: usize, horizon: usize, freq: &str) -> ForecastArgs {
        let t0 = days_from_civil(2015, 1, 1);
        let ds: Vec<String> = (0..points as i64).map(|i| format_ymd(t0 + i)).collect();
        let y: Vec<f64> = (0..points)
            .map(|i| {
                let t = i as f64;
                10.0 + 0.01 * t + (2.0 * std::f64::consts::PI * t / 7.0).sin()
            })
            .collect();
        ForecastArgs {
            ds,
            y,
            horizon,
            freq: Some(freq.into()),
            growth: Some("logistic".into()),
            cap: Some(50.0),
            ..ForecastArgs::default()
        }
    }

    /// The lambda a given request will actually make `predict` draw, computed the way the
    /// door computes it — through `prophet::changepoint_count`, so a test named "over"
    /// cannot quietly become a test of something under the bound if the constant moves.
    fn lambda_of(args: &ForecastArgs) -> f64 {
        use crate::dates::parse_date;
        use crate::prophet::{auto_seasonalities, changepoint_count, Mode, Spec};
        let ds: Vec<i64> = args
            .ds
            .iter()
            .map(|s| parse_date(s).expect("the helper builds valid dates"))
            .collect();
        let fut = crate::dates::future_days(
            ds[ds.len() - 1],
            args.horizon,
            args.freq.as_deref().unwrap_or("D"),
        )
        .expect("the helper builds a valid freq");
        let t_scale = (ds[ds.len() - 1] - ds[0]) as f64;
        let t_max = (fut[fut.len() - 1] - ds[0]) as f64 / t_scale;
        let spec = Spec::default_linear(auto_seasonalities(&ds, 10.0, Mode::Additive));
        changepoint_count(ds.len(), &spec) as f64 * (t_max - 1.0)
    }

    /// The measured CR-01 request is refused, and the refusal names the bound key.
    #[test]
    fn a_logistic_request_over_the_changepoint_lambda_bound_is_refused() {
        let args = logistic_lambda_args(33, 3650, "MS");
        let lambda = lambda_of(&args);
        assert!(
            lambda > crate::types::MAX_LOGISTIC_CHANGEPOINT_LAMBDA,
            "the OVER geometry must actually be over the bound it is testing: \
             lambda={lambda:.1} vs {}",
            crate::types::MAX_LOGISTIC_CHANGEPOINT_LAMBDA
        );
        refusal(&args, "max_logistic_changepoint_lambda");
    }

    /// The NEAR MISS, on the review's own control axis. Same 33 points, same `MS`
    /// frequency, a horizon that pulls lambda just under the bound — and it must still
    /// FIT and still return a full-length band. A bound proven only to refuse is not
    /// proven to refuse just what it claims.
    #[test]
    fn a_logistic_request_just_under_the_changepoint_lambda_bound_is_accepted() {
        let horizon = 840usize;
        let args = logistic_lambda_args(33, horizon, "MS");
        let lambda = lambda_of(&args);
        assert!(
            lambda <= crate::types::MAX_LOGISTIC_CHANGEPOINT_LAMBDA,
            "the UNDER geometry must actually sit under the bound: lambda={lambda:.1}"
        );
        assert!(
            lambda > 0.9 * crate::types::MAX_LOGISTIC_CHANGEPOINT_LAMBDA,
            "the near miss must be NEAR: lambda={lambda:.1} is not within 10% of the bound, \
             so this control would pass for a bound an order of magnitude away"
        );
        let r = forecast(&args).expect("a request just under the bound must fit, not refuse");
        assert_eq!(r.yhat.len(), horizon, "one row per horizon step");
        assert_eq!(r.yhat_lower.len(), horizon);
        assert_eq!(r.yhat_upper.len(), horizon);
        assert_eq!(r.trend.len(), horizon);
        assert!(r.yhat.iter().all(|v| v.is_finite()));
    }

    /// The SAME geometry on the LINEAR arm is still accepted.
    ///
    /// `predict`'s linear arm never calls `poisson` — the review measured it flat at
    /// 0.149 s while the logistic arm went to 2.334 s — so the bound must be keyed on
    /// `Growth::Logistic`. Without this case, a refusal that fired for every growth arm
    /// would pass both cases above while silently refusing the majority of real requests.
    #[test]
    fn the_linear_arm_at_the_same_geometry_is_still_accepted() {
        let mut args = logistic_lambda_args(33, 3650, "MS");
        assert!(
            lambda_of(&args) > crate::types::MAX_LOGISTIC_CHANGEPOINT_LAMBDA,
            "the geometry must be one the LOGISTIC arm would refuse, or this proves nothing"
        );
        args.growth = Some("linear".into());
        // `cap` is logistic-only (06-10), so the linear request drops it.
        args.cap = None;
        let r = forecast(&args).expect("the linear arm has no changepoint-lambda cost to bound");
        assert_eq!(r.yhat.len(), 3650, "one row per horizon step");
    }

    /// JSON `1e400` parses to `f64::INFINITY`, for which `cap <= y_max` is false.
    #[test]
    fn an_infinite_cap_is_refused() {
        let mut args = np_args(60, 7);
        args.model = None;
        args.growth = Some("logistic".into());
        args.cap = Some(f64::INFINITY);
        refusal(&args, "finite");
    }

    /// `(1.0 + w) / 2.0` rounds to EXACTLY 1.0 for the largest accepted `interval_width`,
    /// and the un-clamped tail then divided infinity by infinity.
    #[test]
    fn the_widest_accepted_interval_still_yields_a_finite_z() {
        let w = 0.999_999_999_999_999_9_f64;
        assert!(w < 1.0, "the door accepts anything strictly below 1.0");
        assert!(
            ((1.0 + w) / 2.0 - 1.0).abs() < f64::EPSILON,
            "the midpoint really does round to 1.0"
        );
        let z = normal_quantile((1.0 + w) / 2.0);
        assert!(z.is_finite(), "band z must stay finite, got {z}");
    }

    /// 20 points seven days apart select NO auto seasonality, and `season_dim() == 0`
    /// tripped trueno's `Contract transpose: input is empty` inside the fit.
    #[test]
    fn a_weekly_spaced_series_still_fits_instead_of_panicking() {
        let mut args = np_args(20, 7);
        args.ds = (0..20)
            .map(|i| format_ymd(days_from_civil(2020, 1, 1) + i * 7))
            .collect();
        args.y = (0..20).map(|i| 10.0 + f64::from(i) * 0.5).collect();
        let r = forecast(&args).expect("a weekly-spaced series is a legal request");
        assert_eq!(r.yhat.len(), 7);
        assert!(r.yhat.iter().all(|v| v.is_finite()));
    }

    // ---------------------------------------------------------------------------------
    // Plan 06.1-01 Task 2: the six external-regressor door refusals.
    //
    // Every refusal is paired with a POSITIVE CONTROL that is ACCEPTED, so no check can
    // pass by refusing everything — the failure mode a refusal-only suite cannot see.
    // ---------------------------------------------------------------------------------

    /// A prophet request over `n` daily points with `horizon` future steps and the given
    /// regressors, none of which is refused by anything except what the test is probing.
    fn reg_args(n: usize, horizon: usize, regs: Vec<crate::types::RegressorArg>) -> ForecastArgs {
        let (ds, y) = synthetic_daily(n);
        ForecastArgs {
            ds,
            y,
            horizon,
            seed: Some(42),
            regressors: Some(regs),
            ..ForecastArgs::default()
        }
    }

    /// A well-formed regressor of exactly the required length, varying enough to have a
    /// non-zero spread.
    fn good_reg(name: &str, n: usize, horizon: usize) -> crate::types::RegressorArg {
        crate::types::RegressorArg {
            name: name.into(),
            values: (0..n + horizon)
                .map(|i| f64::from(i as u32) * 0.5 + 1.0)
                .collect(),
            mode: None,
            prior_scale: None,
            standardize: None,
        }
    }

    /// CONTROL for every refusal below: a well-formed four-regressor request is ACCEPTED
    /// and its regressors reach the response as components.
    #[test]
    fn regressors_that_are_well_formed_are_accepted() {
        let args = reg_args(60, 7, vec![good_reg("promo", 60, 7)]);
        let r = forecast(&args).expect("a well-formed regressor request must be accepted");
        assert!(
            r.components.contains_key("promo")
                && r.components.contains_key("extra_regressors_additive"),
            "the regressor must reach the response; got {:?}",
            r.components.keys().collect::<Vec<_>>()
        );
    }

    /// CHECK 1, the length tie, from BOTH sides plus the exact value.
    ///
    /// Three shapes because one input is an anecdote (CLAUDE.md rule 6): a check keyed on
    /// the wrong side of the comparison passes two of the three.
    #[test]
    fn a_regressor_whose_values_length_is_wrong_is_refused_from_both_sides() {
        for (label, len) in [("too short", 66usize), ("too long", 68)] {
            let mut reg = good_reg("promo", 60, 7);
            reg.values = vec![1.0; len];
            let args = reg_args(60, 7, vec![reg]);
            match forecast(&args) {
                Err(ForecastError::Validation(m)) => {
                    assert!(
                        m.contains("67"),
                        "{label}: must state the required length, got {m:?}"
                    );
                    assert!(
                        m.contains(&len.to_string()),
                        "{label}: must state the length received, got {m:?}"
                    );
                }
                other => panic!("{label} must be refused, got {:?}", other.map(|r| r.model)),
            }
        }
        // POSITIVE CONTROL: exactly points + horizon is accepted.
        let args = reg_args(60, 7, vec![good_reg("promo", 60, 7)]);
        assert_eq!(args.regressors.as_ref().expect("regs")[0].values.len(), 67);
        forecast(&args).expect("exactly points + horizon must be accepted");
    }

    /// CHECK 2, input finiteness. JSON `1e400` parses to infinity — the same class the
    /// `cap` finiteness check closed.
    #[test]
    fn a_regressor_with_a_non_finite_value_is_refused() {
        for bad in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let mut reg = good_reg("promo", 60, 7);
            reg.values[13] = bad;
            let args = reg_args(60, 7, vec![reg]);
            refusal(&args, "row 13");
        }
        // POSITIVE CONTROL: a large-but-finite value is accepted.
        let mut reg = good_reg("promo", 60, 7);
        reg.values[13] = 1.0e12;
        forecast(&reg_args(60, 7, vec![reg])).expect("a large finite value must be accepted");
    }

    /// CHECK 3, the mode allowlist, refused BY NAME.
    #[test]
    fn a_regressor_mode_outside_the_allowlist_is_refused_by_name() {
        let mut reg = good_reg("promo", 60, 7);
        reg.mode = Some("exponential".into());
        refusal(&reg_args(60, 7, vec![reg]), "exponential");
        // POSITIVE CONTROLS: both accepted values.
        for m in ["additive", "multiplicative"] {
            let mut reg = good_reg("promo", 60, 7);
            reg.mode = Some(m.into());
            forecast(&reg_args(60, 7, vec![reg]))
                .unwrap_or_else(|e| panic!("mode {m:?} must be accepted: {e}"));
        }
    }

    /// CHECK 4, the prior-scale domain.
    ///
    /// The bound is not "greater than zero": the objective SQUARES it into a denominator
    /// (`prophet.rs:536`, `:685`), so `f64::MIN_POSITIVE` squares to exactly `0.0` and the
    /// initial zero coefficients meet `0.0 / 0.0`.
    #[test]
    fn a_regressor_prior_scale_outside_the_usable_range_is_refused() {
        for (label, bad) in [
            ("MIN_POSITIVE squares to zero", f64::MIN_POSITIVE),
            (
                "one ULP below the floor",
                f64::from_bits(REGRESSOR_PRIOR_SCALE_MIN.to_bits() - 1),
            ),
            ("above the ceiling", REGRESSOR_PRIOR_SCALE_MAX * 10.0),
            ("zero", 0.0),
            ("negative", -1.0),
            ("infinite", f64::INFINITY),
            ("nan", f64::NAN),
        ] {
            let mut reg = good_reg("promo", 60, 7);
            reg.prior_scale = Some(bad);
            match forecast(&reg_args(60, 7, vec![reg])) {
                Err(ForecastError::Validation(m)) => {
                    assert!(m.contains("prior_scale"), "{label}: got {m:?}");
                }
                other => panic!("{label} must be refused, got {:?}", other.map(|r| r.model)),
            }
        }
    }

    /// The POSITIVE CONTROL for check 4: a bound that has not been shown to keep the
    /// arithmetic finite is a guess.
    ///
    /// NOTE what this does and does not establish. It pins FINITENESS at both ends, which
    /// is this plan's acceptance criterion. It does NOT establish that a prior scale at the
    /// floor produces a meaningful fit — measurement says it does not, and
    /// `the_prior_scale_floor_is_a_representability_bound_not_a_usability_bound` below
    /// records that explicitly rather than leaving it as a comment nobody re-runs.
    #[test]
    fn a_regressor_prior_scale_at_the_floor_fits_finitely() {
        for (label, ps) in [
            ("floor", REGRESSOR_PRIOR_SCALE_MIN),
            ("ceiling", REGRESSOR_PRIOR_SCALE_MAX),
            ("default-ish", 10.0),
        ] {
            let mut reg = good_reg("promo", 60, 7);
            reg.prior_scale = Some(ps);
            let r = forecast(&reg_args(60, 7, vec![reg]))
                .unwrap_or_else(|e| panic!("{label} ({ps:e}) must be accepted: {e}"));
            assert!(
                r.yhat.iter().all(|v| v.is_finite()) && r.trend.iter().all(|v| v.is_finite()),
                "{label} ({ps:e}): every yhat and trend value must be finite — a NaN here is \
                 the 0/0 the bound exists to prevent"
            );
            let objective = r.diagnostics["lbfgs"]["objective"]
                .as_f64()
                .expect("the prophet arm reports its objective");
            assert!(
                objective.is_finite(),
                "{label} ({ps:e}): the fitted objective must be finite, got {objective}"
            );
        }
    }

    /// CHECK 5, duplicate names. The component map is keyed by name.
    #[test]
    fn regressors_sharing_a_name_are_refused() {
        let args = reg_args(
            60,
            7,
            vec![good_reg("promo", 60, 7), good_reg("promo", 60, 7)],
        );
        refusal(&args, "promo");
        // POSITIVE CONTROL: two DISTINCT names are accepted.
        let args = reg_args(
            60,
            7,
            vec![good_reg("promo", 60, 7), good_reg("price", 60, 7)],
        );
        forecast(&args).expect("two distinct names must be accepted");
    }

    /// CHECK 6a, zero spread over the history rows.
    ///
    /// An all-zero column has ONE unique value, so it is NOT auto-exempt, IS standardised,
    /// and comes out at `std = 0.0`. This is what stops `splice` dividing by zero.
    #[test]
    fn a_regressor_constant_over_the_history_is_refused() {
        for constant in [0.0, 1.0, 7.5] {
            let mut reg = good_reg("flat", 60, 7);
            reg.values = vec![constant; 67];
            refusal(&reg_args(60, 7, vec![reg]), "constant over the history");
        }
        // POSITIVE CONTROL, and the OTHER half of the auto rule: a {0,1} column has TWO
        // unique values, is auto-exempt, keeps std = 1.0, and is accepted.
        let mut reg = good_reg("binary", 60, 7);
        reg.values = (0..67).map(|i| f64::from(i as u32 % 2)).collect();
        forecast(&reg_args(60, 7, vec![reg]))
            .expect("a {0,1} indicator column is auto-exempt and must be accepted");
    }

    /// CHECK 6b, post-arithmetic finiteness. A SEPARATE check from input finiteness, and
    /// both are needed: these inputs are each finite, but their sum of squares overflows.
    #[test]
    fn a_regressor_whose_derived_statistics_overflow_is_refused() {
        let mut reg = good_reg("huge", 60, 7);
        reg.values = (0..67)
            .map(|i| if i % 2 == 0 { 1.0e300 } else { -1.0e300 })
            .collect();
        assert!(
            reg.values.iter().all(|v| v.is_finite()),
            "the probe's own inputs must be finite, or it is testing check 2 by accident"
        );
        refusal(&reg_args(60, 7, vec![reg]), "non-finite");
        // POSITIVE CONTROL at a large-but-safe magnitude, so the check cannot pass by
        // refusing every large column.
        let mut reg = good_reg("big", 60, 7);
        reg.values = (0..67)
            .map(|i| 1.0e100 + f64::from(i as u32) * 1.0e98)
            .collect();
        forecast(&reg_args(60, 7, vec![reg]))
            .expect("a large but non-overflowing column must be accepted");
    }

    /// CHECK 7, the MEASURED count ceiling — replaces plan 06.1-01's interim literal 50.
    ///
    /// The ceiling is read from `MAX_REGRESSORS` rather than written as a literal, so
    /// lowering the constant after a second measurement cannot leave this test pinning a
    /// number the door no longer enforces.
    #[test]
    fn regressors_beyond_the_measured_count_ceiling_are_refused() {
        let over: Vec<_> = (0..=crate::types::MAX_REGRESSORS)
            .map(|i| good_reg(&format!("r{i}"), 60, 7))
            .collect();
        // The probe must be INSIDE the product ceiling, or the refusal it triggers is the
        // product check (which runs just after) and this test would be green for the wrong
        // reason — a real risk, because both ceilings move together in Task 2 step 5.
        assert!(
            (60 + 7) * over.len() <= crate::types::MAX_REGRESSOR_DESIGN_COST,
            "the over-count probe must violate ONLY the count ceiling"
        );
        refusal(&reg_args(60, 7, over), "max_regressors");
        // POSITIVE CONTROL: exactly AT the ceiling is accepted. Re-derived against the
        // MEASURED constant, so a refusal that refused everything would show here.
        let at: Vec<_> = (0..crate::types::MAX_REGRESSORS)
            .map(|i| good_reg(&format!("r{i}"), 60, 7))
            .collect();
        assert_eq!(at.len(), crate::types::MAX_REGRESSORS);
        assert!(
            (60 + 7) * at.len() <= crate::types::MAX_REGRESSOR_DESIGN_COST,
            "the control must be inside the PRODUCT ceiling too, or it would be refused by \
             the other check and prove nothing about this one"
        );
        forecast(&reg_args(60, 7, at)).expect("exactly the count ceiling must be accepted");
    }

    /// The fixed history geometry the product-ceiling probes use.
    ///
    /// 2 000 points and a 365-step horizon: both individually legal by a wide margin, so
    /// the only thing a refusal here can be about is the PRODUCT.
    const PRODUCT_PROBE_POINTS: usize = 2_000;
    const PRODUCT_PROBE_HORIZON: usize = 365;

    /// The smallest regressor count whose product EXCEEDS the ceiling, and the largest that
    /// does not — both DERIVED from the constant.
    ///
    /// Derived rather than written down because the ceiling MOVES: plan 06.1-03 Task 1
    /// derives a provisional value and Task 2 step 5 re-derives it with the identifiability
    /// diagnostic live. A hardcoded pair silently stops straddling the boundary the moment
    /// the constant changes — which is exactly what happened when the Task 1 sweep lowered
    /// it from the starting candidate, and is why this is a function.
    fn product_probe_counts() -> (usize, usize) {
        let rows = PRODUCT_PROBE_POINTS + PRODUCT_PROBE_HORIZON;
        let under = crate::types::MAX_REGRESSOR_DESIGN_COST / rows;
        (under + 1, under)
    }

    /// CHECK 9, the design-cost PRODUCT, from both sides.
    ///
    /// The point of a product bound is that each factor is individually legal: this request
    /// is inside `fit_max_points`, inside `fit_max_horizon` and inside `fit_max_regressors`,
    /// and only their product is refused.
    #[test]
    fn a_regressor_design_cost_over_the_product_ceiling_is_refused() {
        let (n, horizon) = (PRODUCT_PROBE_POINTS, PRODUCT_PROBE_HORIZON);
        let (over_count, under_count) = product_probe_counts();
        let over: Vec<_> = (0..over_count)
            .map(|i| good_reg(&format!("r{i}"), n, horizon))
            .collect();
        assert!(
            n <= crate::types::MAX_POINTS
                && horizon <= crate::types::MAX_HORIZON
                && over.len() <= crate::types::MAX_REGRESSORS,
            "every factor must be individually legal, or this tests the wrong check"
        );
        assert!(
            (n + horizon) * over.len() > crate::types::MAX_REGRESSOR_DESIGN_COST,
            "the probe must actually be over the product ceiling"
        );
        let args = reg_args(n, horizon, over);
        refusal(&args, "max_regressor_design_cost");
        // The message STATES THE OPERANDS it used, the way the holiday design-cost message
        // does — a bound that reports only its own value leaves the caller guessing which
        // factor to shrink.
        match forecast(&args) {
            Err(ForecastError::Validation(m)) => {
                let cells = (n + horizon) * over_count;
                for needle in [
                    "(points + horizon) x n_regressors".to_string(),
                    format!("({n} + {horizon}) x {over_count}"),
                    cells.to_string(),
                ] {
                    assert!(
                        m.contains(&needle),
                        "message must contain {needle:?}, got {m:?}"
                    );
                }
            }
            other => panic!(
                "expected a Validation refusal, got {:?}",
                other.map(|r| r.model)
            ),
        }
        // POSITIVE CONTROL, at the LARGEST count the same geometry admits — one below the
        // refused one, so the two straddle the boundary rather than sitting near it.
        let under: Vec<_> = (0..under_count)
            .map(|i| good_reg(&format!("r{i}"), n, horizon))
            .collect();
        assert_eq!(
            under.len() + 1,
            over_count,
            "the pair must straddle the ceiling"
        );
        assert!((n + horizon) * under.len() <= crate::types::MAX_REGRESSOR_DESIGN_COST);
        forecast(&reg_args(n, horizon, under))
            .expect("the largest request under the product ceiling must be accepted");
    }

    /// CHECK 8, the name byte bound — and it must not echo the name back (T-06.1-14).
    ///
    /// Reuses `fit_max_holiday_name_len`: the amplification argument is the same one C-07
    /// records, so a fourth name ceiling would be a second number to keep in step.
    #[test]
    fn a_regressor_name_longer_than_the_byte_ceiling_is_refused_without_echoing_it() {
        let long = "z".repeat(crate::types::MAX_HOLIDAY_NAME_LEN + 1);
        let mut reg = good_reg(&long, 60, 7);
        reg.name.clone_from(&long);
        match forecast(&reg_args(60, 7, vec![reg])) {
            Err(ForecastError::Validation(m)) => {
                assert!(
                    m.contains("the name is not echoed back"),
                    "the message must say the name is withheld, got {m:?}"
                );
                assert!(
                    !m.contains(&long),
                    "the message must NOT echo the oversized name — echoing it \
                     re-materialises the very bytes the bound refuses"
                );
                assert!(
                    m.contains("index 0") && m.contains("201 bytes"),
                    "the message must name the INDEX and the LENGTH, got {m:?}"
                );
            }
            other => panic!(
                "expected a Validation refusal, got {:?}",
                other.map(|r| r.model)
            ),
        }
        // POSITIVE CONTROL: exactly AT the ceiling is accepted.
        let at = "z".repeat(crate::types::MAX_HOLIDAY_NAME_LEN);
        forecast(&reg_args(60, 7, vec![good_reg(&at, 60, 7)]))
            .expect("a name of exactly the ceiling must be accepted");
    }

    /// BYTES, not characters — the rewrite this test exists to catch.
    ///
    /// The holiday side already carries this case
    /// (`a_holiday_name_whose_char_count_fits_but_whose_byte_length_does_not_is_refused`);
    /// the regressor side needs its own, because a rewrite to `chars().count()` would be
    /// made in one place and would have to be caught in both.
    #[test]
    fn a_regressor_name_whose_char_count_fits_but_whose_byte_length_does_not_is_refused() {
        // 'é' is two UTF-8 bytes: 120 chars, 240 bytes.
        let name = "é".repeat(120);
        assert!(
            name.chars().count() <= crate::types::MAX_HOLIDAY_NAME_LEN
                && name.len() > crate::types::MAX_HOLIDAY_NAME_LEN,
            "the probe must fit on chars and NOT on bytes, or it tests nothing"
        );
        refusal(&reg_args(60, 7, vec![good_reg(&name, 60, 7)]), "bytes");
    }

    /// CHECK 10 part (a): a regressor named exactly like a GENERATED design column.
    #[test]
    fn a_regressor_colliding_with_a_generated_column_name_is_refused() {
        // 60 daily points gives a weekly seasonality, so `weekly_delim_1` is a real
        // generated column name for this request — derived, not assumed.
        let args = reg_args(60, 7, vec![good_reg("promo", 60, 7)]);
        let probe = forecast(&args).expect("the control request must be accepted");
        let generated = probe
            .components
            .keys()
            .find(|k| k.contains("_delim_"))
            .cloned();
        // The components map carries COMPONENT names, not column names, so derive the
        // column name from the spec the same way the door does.
        let ds: Vec<i64> = args
            .ds
            .iter()
            .map(|s| crate::dates::parse_date(s).expect("valid"))
            .collect();
        let spec = crate::prophet::Spec::default_linear(crate::prophet::auto_seasonalities(
            &ds,
            10.0,
            crate::prophet::Mode::Additive,
        ));
        let col = crate::prophet::columns(&spec)
            .into_iter()
            .map(|c| c.name)
            .find(|n| n.contains("_delim_"))
            .expect("this geometry must generate at least one _delim_ column");
        assert!(
            generated.is_none(),
            "a _delim_ name is a COLUMN name, not a component key; if one appeared as a \
             component this test's premise changed"
        );
        match forecast(&reg_args(60, 7, vec![good_reg(&col, 60, 7)])) {
            Err(ForecastError::Validation(m)) => assert!(
                m.contains("a generated design column name"),
                "the message must say WHICH part of the reserved set matched, got {m:?}"
            ),
            other => panic!(
                "expected a Validation refusal, got {:?}",
                other.map(|r| r.model)
            ),
        }
    }

    /// CHECK 10 part (b): a regressor named after a SEASONALITY component.
    ///
    /// `weekly` is not itself a generated column name — the columns are `weekly_delim_1`,
    /// `weekly_delim_2`, … — so part (a) alone would let this through, and `predict` would
    /// push a `weekly` component that the regressor's own entry then overwrote.
    #[test]
    fn a_regressor_colliding_with_a_seasonality_component_name_is_refused() {
        match forecast(&reg_args(60, 7, vec![good_reg("weekly", 60, 7)])) {
            Err(ForecastError::Validation(m)) => assert!(
                m.contains("a response component name"),
                "the message must say WHICH part matched, got {m:?}"
            ),
            other => panic!(
                "expected a Validation refusal, got {:?}",
                other.map(|r| r.model)
            ),
        }
    }

    /// CHECK 10 part (b), the HOLIDAY half — and its accepted control, which is what makes
    /// the check a collision test rather than a blocklist.
    #[test]
    fn a_regressor_colliding_with_a_declared_holiday_name_is_refused_but_not_otherwise() {
        let mut args = reg_args(60, 7, vec![good_reg("blackfriday", 60, 7)]);
        // WITHOUT the holiday declared, `blackfriday` is an ordinary name and is ACCEPTED.
        forecast(&args)
            .expect("`blackfriday` collides with nothing when no such holiday is declared");
        // WITH it declared, the same name is refused.
        args.holidays = Some(vec![crate::types::HolidayArg {
            name: "blackfriday".into(),
            dates: vec!["2020-01-10".into()],
            lower_window: 0,
            upper_window: 0,
        }]);
        match forecast(&args) {
            Err(ForecastError::Validation(m)) => assert!(
                m.contains("a response component name") && m.contains("blackfriday"),
                "the message must name the collision and its part, got {m:?}"
            ),
            other => panic!(
                "expected a Validation refusal, got {:?}",
                other.map(|r| r.model)
            ),
        }
    }

    /// CHECK 10 part (c): a RESERVED RESPONSE KEY, which is invisible to parts (a) and (b).
    ///
    /// This is the case the review found: `components.insert(name, value)` is a MAP INSERT,
    /// so a regressor named `additive_terms` is computed as its own component and then
    /// silently OVERWRITTEN by the aggregate `predict` pushes afterwards.
    #[test]
    fn a_regressor_colliding_with_a_reserved_response_key_is_refused() {
        for key in [
            "additive_terms",
            "extra_regressors_additive",
            "yhat",
            "trend",
        ] {
            match forecast(&reg_args(60, 7, vec![good_reg(key, 60, 7)])) {
                Err(ForecastError::Validation(m)) => assert!(
                    m.contains("a reserved response key") && m.contains(key),
                    "{key}: the message must name the collision and its part, got {m:?}"
                ),
                other => panic!(
                    "{key}: expected a Validation refusal, got {:?}",
                    other.map(|r| r.model)
                ),
            }
        }
        // POSITIVE CONTROL across all three parts: an ordinary name clears every one.
        forecast(&reg_args(60, 7, vec![good_reg("promo", 60, 7)]))
            .expect("`promo` must clear all three parts of the reserved set");
    }

    /// The part-(c) list cannot be silently emptied into vacuity.
    ///
    /// Parts (a) and (b) are DERIVED from the spec, so they cannot rot. Part (c) is a
    /// hand-written slice, which is exactly the shape that goes stale — so it is pinned
    /// here, by name, in one place.
    #[test]
    fn the_reserved_response_keys_slice_is_not_empty() {
        assert_eq!(
            super::RESERVED_RESPONSE_KEYS.len(),
            11,
            "the reserved response key list must carry all eleven names"
        );
        for want in [
            "additive_terms",
            "multiplicative_terms",
            "holidays",
            "extra_regressors_additive",
            "extra_regressors_multiplicative",
            "trend",
            "yhat",
            "yhat_lower",
            "yhat_upper",
            "ds",
            "cap",
        ] {
            assert!(
                super::RESERVED_RESPONSE_KEYS.contains(&want),
                "{want} must be in the reserved response key list"
            );
        }
    }

    /// THE REFUSAL COVERAGE, pinned by MESSAGES EXERCISED rather than by test-function names.
    ///
    /// A floor on the number of test FUNCTIONS is coupled to how an executor chose to group
    /// them: six checks can legitimately be six functions or one table. This test drives one
    /// probe per refusal the regressor surface can produce, asserts each one actually fires
    /// with the expected needle, and asserts its OWN row count — so the coverage number is a
    /// property of the refusals rather than of the file's layout.
    #[test]
    fn the_regressor_refusal_table_is_complete() {
        type Probe = fn() -> ForecastArgs;
        let long_name = "z".repeat(crate::types::MAX_HOLIDAY_NAME_LEN + 1);
        let table: Vec<(&str, &str, Box<dyn Fn() -> ForecastArgs>)> = vec![
            (
                "name byte length",
                "max_holiday_name_len",
                Box::new(move || reg_args(60, 7, vec![good_reg(&long_name, 60, 7)])),
            ),
            (
                "count ceiling",
                "max_regressors",
                Box::new(|| {
                    reg_args(
                        60,
                        7,
                        (0..=crate::types::MAX_REGRESSORS)
                            .map(|i| good_reg(&format!("r{i}"), 60, 7))
                            .collect(),
                    )
                }),
            ),
            (
                "design cost product",
                "max_regressor_design_cost",
                Box::new(|| {
                    let (over_count, _) = product_probe_counts();
                    reg_args(
                        PRODUCT_PROBE_POINTS,
                        PRODUCT_PROBE_HORIZON,
                        (0..over_count)
                            .map(|i| {
                                good_reg(
                                    &format!("r{i}"),
                                    PRODUCT_PROBE_POINTS,
                                    PRODUCT_PROBE_HORIZON,
                                )
                            })
                            .collect(),
                    )
                }),
            ),
            (
                "collision: generated column name",
                "a generated design column name",
                Box::new(|| reg_args(60, 7, vec![good_reg("weekly_delim_1", 60, 7)])),
            ),
            (
                "collision: component name",
                "a response component name",
                Box::new(|| reg_args(60, 7, vec![good_reg("weekly", 60, 7)])),
            ),
            (
                "collision: reserved response key",
                "a reserved response key",
                Box::new(|| reg_args(60, 7, vec![good_reg("additive_terms", 60, 7)])),
            ),
            (
                "duplicate names",
                "appears more than once",
                Box::new(|| {
                    reg_args(
                        60,
                        7,
                        vec![good_reg("promo", 60, 7), good_reg("promo", 60, 7)],
                    )
                }),
            ),
            (
                "length tie",
                "values but 67 are required",
                Box::new(|| {
                    let mut r = good_reg("promo", 60, 7);
                    r.values.truncate(10);
                    reg_args(60, 7, vec![r])
                }),
            ),
            (
                "input finiteness",
                "non-finite value at row",
                Box::new(|| {
                    let mut r = good_reg("promo", 60, 7);
                    r.values[3] = f64::INFINITY;
                    reg_args(60, 7, vec![r])
                }),
            ),
            (
                "mode allowlist",
                "is not supported",
                Box::new(|| {
                    let mut r = good_reg("promo", 60, 7);
                    r.mode = Some("exponential".into());
                    reg_args(60, 7, vec![r])
                }),
            ),
            (
                "prior scale range",
                "outside the",
                Box::new(|| {
                    let mut r = good_reg("promo", 60, 7);
                    r.prior_scale = Some(0.0);
                    reg_args(60, 7, vec![r])
                }),
            ),
            (
                "zero spread",
                "carries no information",
                Box::new(|| {
                    let mut r = good_reg("flat", 60, 7);
                    r.values = vec![2.0; 67];
                    reg_args(60, 7, vec![r])
                }),
            ),
            (
                "post-arithmetic mean overflow",
                "non-finite mean after standardisation",
                Box::new(|| {
                    let mut r = good_reg("huge", 60, 7);
                    r.values = (0..67).map(|i| 1.0e307 + f64::from(i as u32)).collect();
                    reg_args(60, 7, vec![r])
                }),
            ),
            // The neuralprophet ARM rules replace the temporary blanket refusal plan
            // 06.1-01 left there. Three rows, because the arm now has three rules and a
            // single row would leave two of them uncovered by this table.
            (
                "neuralprophet arm: multiplicative mode",
                "which is prophet-only: the neuralprophet model composes",
                Box::new(|| {
                    let mut a = reg_args(60, 7, vec![good_reg("promo", 60, 7)]);
                    a.model = Some("neuralprophet".into());
                    a.freq = Some("D".into());
                    if let Some(regs) = a.regressors.as_mut() {
                        regs[0].mode = Some("multiplicative".into());
                    }
                    a
                }),
            ),
            (
                "neuralprophet arm: an explicit prior_scale",
                "carries prior_scale, which is prophet-only",
                Box::new(|| {
                    let mut a = reg_args(60, 7, vec![good_reg("promo", 60, 7)]);
                    a.model = Some("neuralprophet".into());
                    a.freq = Some("D".into());
                    if let Some(regs) = a.regressors.as_mut() {
                        regs[0].prior_scale = Some(10.0);
                    }
                    a
                }),
            ),
            (
                "neuralprophet arm: a regressor on a gappy series with lags",
                "Supply a gap-free daily series, or set n_lags to 0",
                Box::new(|| {
                    let mut a = gappy_np_args(90, &[13], 7);
                    a.n_lags = Some(7);
                    a
                }),
            ),
        ];
        // The floor is asserted on the TABLE, so the coverage claim survives any regrouping
        // of the test functions around it.
        assert!(
            table.len() >= 14,
            "the regressor refusal table must cover at least 14 distinct refusals, has {}",
            table.len()
        );
        let _: Option<Probe> = None;
        for (label, needle, build) in &table {
            match forecast(&build()) {
                Err(ForecastError::Validation(m)) => assert!(
                    m.contains(needle),
                    "{label}: the refusal must name {needle:?}, got {m:?}"
                ),
                other => panic!(
                    "{label}: this probe must be REFUSED — a row that no longer refuses is a \
                     silently removed check, got {:?}",
                    other.map(|r| r.model)
                ),
            }
        }
    }

    // ================================================================================
    // D-22 / D-26 / D-28: the neuralprophet arm ACCEPTS regressors, and the two arm
    // rules that scope that acceptance to what is measured.
    // ================================================================================

    /// A gappy daily series: `n` calendar days with `missing` of them removed, plus a
    /// regressor array of the required `kept + horizon` length. Returns the args.
    ///
    /// `y` varies enough to have a non-zero spread on both the target and the driver, and
    /// the series is long enough for `auto_epochs` to be cheap.
    fn gappy_np_args(days: usize, missing: &[usize], horizon: usize) -> ForecastArgs {
        let t0 = days_from_civil(2021, 1, 1);
        let mut ds = Vec::new();
        let mut y = Vec::new();
        let mut rng = Rng::new(11);
        for k in 0..days {
            if missing.contains(&k) {
                continue;
            }
            ds.push(format_ymd(t0 + k as i64));
            let t = k as f64;
            y.push(
                20.0 + 0.05 * t
                    + (2.0 * std::f64::consts::PI * t / 7.0).sin()
                    + 0.05 * rng.normal(),
            );
        }
        let n = ds.len();
        ForecastArgs {
            ds,
            y,
            horizon,
            model: Some("neuralprophet".into()),
            freq: Some("D".into()),
            seed: Some(42),
            regressors: Some(vec![crate::types::RegressorArg {
                name: "price".into(),
                values: (0..n + horizon)
                    .map(|i| 2.0 + (i as f64 * 0.37).sin() * 3.0)
                    .collect(),
                mode: None,
                prior_scale: None,
                standardize: None,
            }]),
            ..ForecastArgs::default()
        }
    }

    /// D-26, CASE 1 of 4: a GAP-FREE daily series with lags and a regressor is ACCEPTED.
    ///
    /// Four cases exist because ONE failing input is an anecdote (CLAUDE.md rule 6) and a
    /// predicate keyed on the wrong conjunct passes three of them: keyed on `n_lags > 0`
    /// alone it refuses this one; keyed on the gap alone it refuses case 3; keyed on the
    /// regressor alone it refuses case 4.
    #[test]
    fn a_gap_free_neuralprophet_series_with_lags_and_a_regressor_is_accepted() {
        let mut args = gappy_np_args(90, &[], 7);
        args.n_lags = Some(7);
        let r = forecast(&args).expect("a gap-free daily series with lags must be ACCEPTED");
        assert_eq!(r.model, "neuralprophet");
        assert!(
            r.yhat.len() == 7 && r.yhat.iter().all(|v| v.is_finite()),
            "the forecast must be finite: {:?}",
            r.yhat
        );
    }

    /// D-26, CASE 2 of 4: the SAME series with days removed and the same lags is REFUSED,
    /// and the message names the count, the first missing date and the fix.
    #[test]
    fn a_gappy_neuralprophet_series_with_lags_and_a_regressor_is_refused() {
        let mut args = gappy_np_args(90, &[13], 7);
        args.n_lags = Some(7);
        match forecast(&args) {
            Err(ForecastError::Validation(m)) => {
                // The COUNT, exactly: one day removed is one missing day, not "the grid
                // length minus something approximate".
                assert!(
                    m.contains("missing 1 day(s)"),
                    "the message must name the EXACT missing-day count, got {m:?}"
                );
                // The FIRST MISSING DATE, in the same YYYY-MM-DD form the caller sent.
                let expected = format_ymd(days_from_civil(2021, 1, 1) + 13);
                assert_eq!(expected, "2021-01-14", "fixture arithmetic");
                assert!(
                    m.contains(&expected),
                    "the message must name the first missing date {expected:?}, got {m:?}"
                );
                assert!(
                    m.contains("Supply a gap-free daily series, or set n_lags to 0"),
                    "the message must state the fix, got {m:?}"
                );
            }
            other => panic!(
                "a gappy series with lags and a regressor must be REFUSED, got {:?}",
                other.map(|r| r.model)
            ),
        }
    }

    /// D-26, CASE 3 of 4: the SAME gappy series at `n_lags = 0` is ACCEPTED — D-27 makes
    /// the imputed-day value unread by construction there, so there is nothing to refuse.
    #[test]
    fn a_gappy_neuralprophet_series_without_lags_and_a_regressor_is_accepted() {
        let args = gappy_np_args(90, &[13, 40, 41], 7);
        let r = forecast(&args).expect("a gappy series at n_lags = 0 must be ACCEPTED");
        assert_eq!(r.model, "neuralprophet");
        assert!(r.yhat.iter().all(|v| v.is_finite()));
    }

    /// D-26, CASE 4 of 4: the SAME gappy series with lags but NO regressor is ACCEPTED —
    /// the refusal is about the REGRESSOR, not about gaps in general.
    #[test]
    fn a_gappy_neuralprophet_series_with_lags_and_no_regressor_is_accepted() {
        let mut args = gappy_np_args(90, &[13], 7);
        args.n_lags = Some(7);
        args.regressors = None;
        let r = forecast(&args)
            .expect("a gappy series with lags and NO regressor must still be ACCEPTED");
        assert_eq!(r.model, "neuralprophet");
    }

    /// D-28: a multiplicative regressor on the neuralprophet arm is REFUSED, with an
    /// ADDITIVE control on the same series that is accepted.
    ///
    /// The control is what keeps the refusal from swallowing the happy path: a check that
    /// refused every regressor would pass the first half alone.
    #[test]
    fn a_multiplicative_regressor_on_the_neuralprophet_arm_is_refused_but_additive_is_not() {
        let mut args = gappy_np_args(90, &[], 7);
        if let Some(regs) = args.regressors.as_mut() {
            regs[0].mode = Some("multiplicative".into());
        }
        refusal(
            &args,
            "which is prophet-only: the neuralprophet model composes",
        );
        refusal(&args, "Set mode to \"additive\", or use model \"prophet\"");

        // The ADDITIVE control, explicit rather than defaulted, and at both lag settings —
        // D-28 is unconditional in `n_lags`.
        for lags in [None, Some(7)] {
            let mut ok = gappy_np_args(90, &[], 7);
            ok.n_lags = lags;
            if let Some(regs) = ok.regressors.as_mut() {
                regs[0].mode = Some("additive".into());
            }
            forecast(&ok).unwrap_or_else(|e| {
                panic!("an ADDITIVE regressor at n_lags {lags:?} must be accepted: {e}")
            });
        }
    }

    /// D-21 applied to `prior_scale`: REFUSED when explicitly present, ACCEPTED when
    /// absent.
    ///
    /// The positive control is the whole point. The neuralprophet trainer has one global
    /// weight decay and no per-regressor prior, so accepting the field and dropping it is
    /// the silent-ignore failure this phase exists to close — but a check keyed on the
    /// DEFAULTED value rather than on `Some(_)` would refuse every neuralprophet regressor
    /// request, which is a different bug with the same green test.
    #[test]
    fn a_neuralprophet_regressor_carrying_a_prior_scale_is_refused_but_an_absent_one_is_not() {
        let mut args = gappy_np_args(90, &[], 7);
        if let Some(regs) = args.regressors.as_mut() {
            // The PROPHET DEFAULT, sent explicitly. A check keyed on the value would not
            // fire here; one keyed on `Some(_)` does, which is the distinction.
            regs[0].prior_scale = Some(10.0);
        }
        refusal(&args, "carries prior_scale, which is prophet-only");
        refusal(&args, "Omit the field, or use model \"prophet\"");

        let absent = gappy_np_args(90, &[], 7);
        assert!(
            absent
                .regressors
                .as_ref()
                .is_some_and(|r| r[0].prior_scale.is_none()),
            "the control must genuinely omit the field"
        );
        forecast(&absent).expect("an ABSENT prior_scale must be accepted on this arm");
    }

    /// The refusal PRECEDENCE, pinned: which message a caller receives when several arm
    /// rules apply at once is a DECISION, not an artefact of evaluation order.
    ///
    /// The messages are verbatim contracts that e2e cases string-match, so a refactor that
    /// reordered the three `if`s would silently change what a caller is told. The order is
    /// multiplicative, then `prior_scale`, then the gap predicate — cheapest and most
    /// unconditional first.
    #[test]
    fn the_refusal_precedence_is_stable_when_several_rules_apply() {
        // ALL THREE at once: multiplicative mode, an explicit prior_scale, and a gappy
        // series with lags. The multiplicative message wins.
        let mut all_three = gappy_np_args(90, &[13], 7);
        all_three.n_lags = Some(7);
        if let Some(regs) = all_three.regressors.as_mut() {
            regs[0].mode = Some("multiplicative".into());
            regs[0].prior_scale = Some(10.0);
        }
        refusal(
            &all_three,
            "which is prophet-only: the neuralprophet model composes",
        );

        // (3) and (4): prior_scale plus the gap predicate. The prior_scale message wins.
        let mut two = gappy_np_args(90, &[13], 7);
        two.n_lags = Some(7);
        if let Some(regs) = two.regressors.as_mut() {
            regs[0].prior_scale = Some(10.0);
        }
        refusal(&two, "carries prior_scale, which is prophet-only");

        // And the SHARED hoisted checks still precede all three: the same request with a
        // wrong `values` length is refused by the length tie, not by an arm rule.
        let mut shared_first = all_three.clone();
        if let Some(regs) = shared_first.regressors.as_mut() {
            regs[0].values.truncate(3);
        }
        refusal(&shared_first, "are required (points + horizon");
    }

    /// D-22 as a LIVE test rather than a claim: ONE payload literal, both arms, both
    /// succeed.
    ///
    /// The length contract must not differ between the models — `values.len()` is
    /// `ds.len() + horizon` on both, with no amendment. Building the payload once and
    /// sending it twice is what makes that a comparison instead of two independent
    /// assertions that happen to agree.
    #[test]
    fn the_same_regressor_payload_is_accepted_by_both_arms() {
        let base = gappy_np_args(90, &[], 14);
        let payload = base
            .regressors
            .clone()
            .expect("the fixture carries one regressor");
        assert_eq!(
            payload[0].values.len(),
            base.ds.len() + base.horizon,
            "the ONE length contract, stated before it is exercised"
        );

        let mut np = base.clone();
        np.regressors = Some(payload.clone());
        let np_r = forecast(&np).expect("the neuralprophet arm must accept this payload");

        let mut pr = base.clone();
        pr.model = Some("prophet".into());
        pr.regressors = Some(payload);
        let pr_r = forecast(&pr).expect("the prophet arm must accept the IDENTICAL payload");

        assert_eq!(np_r.model, "neuralprophet");
        assert_eq!(pr_r.model, "prophet");
        assert_eq!(np_r.yhat.len(), pr_r.yhat.len());
    }

    /// D-33's operand rule for REGRESSORS: the neuralprophet arm prices the design over the
    /// imputed grid's SPAN IN DAYS, the prophet arm over the caller's POINT COUNT.
    ///
    /// The rule-4 re-mutation of this operand at the shared hoisted site returned an EMPTY
    /// failing set before this test existed — the arm-dependent operand had no test on
    /// either arm. It has one now.
    ///
    /// The DENSE CONTROL is what makes the claim about the operand rather than about the
    /// regressor count: the same point count and the same regressor count on a contiguous
    /// series is ACCEPTED, so what refuses the first request is the span.
    #[test]
    fn the_regressor_design_cost_uses_the_span_operand_on_the_neuralprophet_arm() {
        use crate::types::MAX_REGRESSOR_DESIGN_COST;
        // 400 calendar days, every 4th day kept: 100 points over a 397-day span.
        const SPAN: usize = 400;
        const HORIZON: usize = 7;
        const REGS: usize = 100;
        let missing: Vec<usize> = (0..SPAN).filter(|k| k % 4 != 0).collect();
        let mut sparse = gappy_np_args(SPAN, &missing, HORIZON);
        let points = sparse.ds.len();
        // Derived from the fixture rather than asserted from the constants above, so a
        // change to the generator cannot silently make the premise below false.
        let first = crate::dates::parse_date(&sparse.ds[0]).expect("a fixture date");
        let last = crate::dates::parse_date(&sparse.ds[points - 1]).expect("a fixture date");
        let span_days = usize::try_from(last - first + 1).expect("a positive span");
        // The PREMISE, asserted rather than assumed: the two operands straddle the ceiling.
        assert!(
            (points + HORIZON) * REGS <= MAX_REGRESSOR_DESIGN_COST,
            "the point operand must CLEAR the ceiling: ({points} + {HORIZON}) x {REGS}"
        );
        assert!(
            (span_days + HORIZON) * REGS > MAX_REGRESSOR_DESIGN_COST,
            "the span operand must EXCEED it: ({span_days} + {HORIZON}) x {REGS}"
        );

        let make = |n: usize, len: usize| -> Vec<crate::types::RegressorArg> {
            (0..n)
                .map(|j| crate::types::RegressorArg {
                    name: format!("r{j}"),
                    values: (0..len)
                        .map(|i| 2.0 + (i as f64 * 0.37 + j as f64).sin() * 3.0)
                        .collect(),
                    mode: None,
                    prior_scale: None,
                    standardize: None,
                })
                .collect()
        };
        sparse.regressors = Some(make(REGS, points + HORIZON));
        match forecast(&sparse) {
            Err(ForecastError::Validation(m)) => {
                assert!(
                    m.contains("exceeds max_regressor_design_cost"),
                    "the SPAN operand must refuse this request, got {m:?}"
                );
                assert!(
                    m.contains("(span_days + horizon) x n_regressors"),
                    "the message must NAME the operand it used, got {m:?}"
                );
            }
            other => panic!(
                "a sparse series whose SPAN product clears the ceiling must be refused, got \
                 {:?}",
                other.map(|r| r.model)
            ),
        }

        // THE DENSE CONTROL: the same point count and the same regressor count, contiguous,
        // is accepted — so what refused above was the span and not the count.
        let mut dense = gappy_np_args(points, &[], HORIZON);
        assert_eq!(
            dense.ds.len(),
            points,
            "the control must match the point count"
        );
        dense.regressors = Some(make(REGS, points + HORIZON));
        forecast(&dense)
            .expect("the DENSE series of the same point count and regressor count is accepted");
    }

    /// Every new refusal message separates the LIMITATION from the FIX with a semicolon,
    /// matching the existing door message shape, and none echoes a caller-supplied VALUE.
    #[test]
    fn a_regressor_refusal_names_the_limitation_and_the_fix() {
        let mut short = good_reg("promo", 60, 7);
        short.values = vec![1.0; 3];
        let mut bad_mode = good_reg("promo", 60, 7);
        bad_mode.mode = Some("exponential".into());
        let mut flat = good_reg("flat", 60, 7);
        flat.values = vec![2.0; 67];
        for (label, reg) in [("length", short), ("mode", bad_mode), ("constant", flat)] {
            match forecast(&reg_args(60, 7, vec![reg])) {
                Err(ForecastError::Validation(m)) => assert!(
                    m.contains(';'),
                    "{label}: the message must separate the limitation from the fix with a \
                     semicolon, got {m:?}"
                ),
                other => panic!("{label} must be refused, got {:?}", other.map(|r| r.model)),
            }
        }
    }

    /// The measured LIMIT of the prior-scale floor, recorded as a test so it cannot quietly
    /// become folklore.
    ///
    /// `REGRESSOR_PRIOR_SCALE_MIN` is a REPRESENTABILITY bound: it keeps `sc * sc` normal
    /// and the objective finite. It is NOT a usability bound. At the floor the fit is
    /// degenerate — L-BFGS performs zero iterations and the regressor contributes exactly
    /// nothing — while the door returns an ordinary-looking forecast.
    ///
    /// This test asserts BOTH halves, so the day either changes it goes red:
    ///   - at the floor, the regressor contribution is exactly zero (the degenerate case
    ///     the door currently ACCEPTS);
    ///   - at a normal prior scale, it is not (so the assertion above is about the floor
    ///     and not about the whole mechanism being broken).
    ///
    /// The contract records the sweep this came from: every `prior_scale <= 1e-9` measured
    /// 0-1 iterations with a zero contribution; `>= 1e-7` fits normally. Closing that gap
    /// needs a measurement campaign across series shapes and belongs to a later plan.
    #[test]
    fn the_prior_scale_floor_is_a_representability_bound_not_a_usability_bound() {
        let contribution = |ps: f64| -> f64 {
            let mut reg = good_reg("promo", 60, 7);
            reg.prior_scale = Some(ps);
            let r = forecast(&reg_args(60, 7, vec![reg]))
                .unwrap_or_else(|e| panic!("prior_scale {ps:e} must be accepted: {e}"));
            r.components["extra_regressors_additive"]
                .as_array()
                .expect("extra_regressors_additive is an array")
                .iter()
                .map(|v| v.as_f64().unwrap_or(f64::NAN).abs())
                .fold(0.0_f64, f64::max)
        };
        assert_eq!(
            contribution(REGRESSOR_PRIOR_SCALE_MIN),
            0.0,
            "at the representability floor the regressor contributes exactly nothing — if \
             this ever becomes non-zero the floor has become a usability bound and the \
             contract comment must be re-measured"
        );
        assert!(
            contribution(10.0) > 0.0,
            "at a normal prior scale the regressor MUST contribute, or the assertion above \
             is passing because the whole mechanism is dead rather than because the floor \
             is degenerate"
        );
    }

    // =====================================================================
    // D-33's pattern applied to REGRESSORS: the shared validation is ONE site
    // above the dispatch, and this is the executable form of that claim.
    // =====================================================================

    /// EVERY hoisted regressor check fires on the NEURALPROPHET arm, with the SAME refusal
    /// substring the prophet arm produces.
    ///
    /// Without this, "the checks are shared" is a claim about where the code SITS rather
    /// than about what the neuralprophet arm DOES. Plan 06.1-01 put the length tie, the two
    /// finiteness checks, the mode allowlist, the prior-scale range, the duplicate-name
    /// check and the zero-spread refusal inside the `"prophet" =>` arm, and plan 06.1-03
    /// added the name-byte bound, the count ceiling, the design-cost product and the
    /// three-part name-collision check in the same place. Removing the neuralprophet
    /// refusal without moving them would have opened that arm onto a regressor surface with
    /// none of them.
    ///
    /// Each row is asserted on BOTH arms, which is what makes "the SAME refusal" a
    /// comparison rather than an assertion about one side. The prophet column is the
    /// control: a needle that stopped matching there would mean the row is testing a
    /// message that no longer exists rather than a check that no longer runs.
    ///
    /// The test asserts its own ROW COUNT, so a check added to the hoisted site without a
    /// row here makes it RED rather than silently uncovered.
    #[test]
    fn the_shared_regressor_validation_reaches_the_neuralprophet_arm() {
        use crate::types::{RegressorArg, MAX_HOLIDAY_NAME_LEN, MAX_REGRESSORS};
        const N: usize = 120;
        const H: usize = 14;
        const LEN: usize = N + H;

        /// A well-formed column with a non-zero spread, of exactly the required length.
        fn ok_reg(name: &str) -> RegressorArg {
            good_reg(name, N, H)
        }
        fn many(n: usize) -> Vec<RegressorArg> {
            (0..n).map(|i| ok_reg(&format!("r{i}"))).collect()
        }

        // ---- the hoisted checks, one row each, in the order the site keeps them ----
        let mut short = ok_reg("price");
        short.values.truncate(LEN - 1);

        let mut nonfinite = ok_reg("price");
        nonfinite.values[5] = f64::INFINITY;

        let mut overflows = ok_reg("price");
        // Individually FINITE, but the sum of squares overflows: the separate
        // post-arithmetic check exists for exactly this, and check 2 above passes it.
        overflows.values = (0..LEN)
            .map(|i| if i % 2 == 0 { 1e300 } else { -1e300 })
            .collect();

        let mut bad_mode = ok_reg("price");
        bad_mode.mode = Some("bogus".into());

        let mut bad_prior = ok_reg("price");
        bad_prior.prior_scale = Some(1e-300);

        let mut constant = ok_reg("price");
        constant.values = vec![7.0; LEN];

        let long_name = ok_reg(&"a".repeat(MAX_HOLIDAY_NAME_LEN + 1));

        // DERIVED from the ceiling, never written down. A hardcoded column count stops
        // straddling the boundary the moment the constant moves, and the row then fails as
        // "the hoisted check did not run there" — blaming the check rather than the stale
        // probe. `product_probe_counts()` cannot serve here: it is parameterised by
        // PRODUCT_PROBE_POINTS/HORIZON (2 000 + 365), a different geometry from this test's
        // N + H. Trips the design-cost product on BOTH operands — the point count on
        // prophet and the span in days on neuralprophet, equal on this contiguous series.
        let design_cost_cols = crate::types::MAX_REGRESSOR_DESIGN_COST / (N + H) + 1;
        assert!(
            (N + H) * design_cost_cols > crate::types::MAX_REGRESSOR_DESIGN_COST
                && (N + H) * (design_cost_cols - 1) <= crate::types::MAX_REGRESSOR_DESIGN_COST
                && design_cost_cols <= MAX_REGRESSORS,
            "the probe must straddle the design-cost ceiling from above while staying \
             inside the COUNT ceiling, or it stops testing the product: {design_cost_cols} \
             columns over {} rows",
            N + H
        );
        let design_cost = many(design_cost_cols);
        let over_count = many(MAX_REGRESSORS + 1);

        let rows: Vec<(&str, Vec<RegressorArg>, &str)> = vec![
            (
                "8. name length in bytes",
                vec![long_name],
                "exceeds max_holiday_name_len",
            ),
            (
                "7. the count ceiling",
                over_count,
                "exceeds max_regressors 200",
            ),
            (
                "9. the design-cost product",
                design_cost,
                "exceeds max_regressor_design_cost",
            ),
            (
                "10a. collision with a generated design column name",
                vec![ok_reg("weekly_delim_1")],
                "which is already a generated design column name",
            ),
            (
                "10b. collision with a per-component name",
                vec![ok_reg("weekly")],
                "which is already a response component name",
            ),
            (
                "10c. collision with a reserved response key",
                vec![ok_reg("yhat")],
                "which is already a reserved response key",
            ),
            (
                "5. duplicate names",
                vec![ok_reg("price"), ok_reg("price")],
                "appears more than once",
            ),
            (
                "1. the length tie",
                vec![short],
                "values but 134 are required",
            ),
            (
                "2. input finiteness",
                vec![nonfinite],
                "carries a non-finite value at row 5",
            ),
            (
                "3. the mode allowlist",
                vec![bad_mode],
                "is not supported; use",
            ),
            (
                "4. the prior-scale range",
                vec![bad_prior],
                "is outside the usable range",
            ),
            (
                "6a. zero spread",
                vec![constant],
                "is constant over the history rows",
            ),
            (
                "6b. derived-value finiteness",
                vec![overflows],
                "non-finite standard deviation after standardisation",
            ),
        ];

        // The ROW COUNT, asserted. Ten checks plus all THREE parts of the name-collision
        // set. A check added to the hoisted site without a row here turns this red rather
        // than leaving the neuralprophet arm silently uncovered.
        assert_eq!(
            rows.len(),
            13,
            "one row per hoisted regressor check, the collision set counted as its three \
             parts; add the row when you add the check"
        );

        for (label, regs, needle) in rows {
            let mut np = reg_args(N, H, regs.clone());
            np.model = Some("neuralprophet".into());
            np.freq = Some("D".into());
            let mut pr = reg_args(N, H, regs);
            pr.model = Some("prophet".into());
            for (arm, args) in [("neuralprophet", &np), ("prophet", &pr)] {
                match forecast(args) {
                    Err(ForecastError::Validation(m)) => assert!(
                        m.contains(needle),
                        "{label}: the {arm} arm must refuse with the SHARED message. Wanted \
                         {needle:?}, got {m:?}"
                    ),
                    other => panic!(
                        "{label}: the {arm} arm must refuse — the hoisted check did not run \
                         there. Got {:?}",
                        other.map(|r| r.model)
                    ),
                }
            }
        }
    }

    // -------------------------------------------------------------------------------
    // The 06.1-05 guard regex, FIXED and shipped as a live guard rather than left in a
    // plan file. `request_train_cost\([^)]*\)` stops at the FIRST `)`, so every call
    // site with a nested parenthesised argument — 3 of the 5 in this file — came back
    // TRUNCATED, and a guard meant to prove an argument is threaded everywhere was
    // blind at exactly the sites this plan changes. CLAUDE.md rule 7: it ships with a
    // must-match / must-not-match case table, and the table is what gets re-run.
    // -------------------------------------------------------------------------------

    /// Every CALL of `name` in `src`, by a BALANCED-PAREN scan from the opening paren.
    ///
    /// Definitions (`pub fn NAME(`) are skipped: a definition is not a call site, and
    /// including it would make the argument assertions below assert about a signature.
    fn cost_call_sites(src: &str, name: &str) -> Vec<String> {
        let mut out = Vec::new();
        let needle = format!("{name}(");
        let mut from = 0usize;
        while let Some(rel) = src[from..].find(&needle) {
            let i = from + rel;
            from = i + needle.len();
            // a call, not a definition, and not a longer identifier ending in `name`
            let before = src[..i].trim_end();
            if before.ends_with("fn") {
                continue;
            }
            if src[..i]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric() || c == '_')
            {
                continue;
            }
            let open = i + needle.len() - 1;
            let mut depth = 0i32;
            for (k, c) in src[open..].char_indices() {
                if c == '(' {
                    depth += 1;
                } else if c == ')' {
                    depth -= 1;
                    if depth == 0 {
                        out.push(src[i..=(open + k)].to_string());
                        break;
                    }
                }
            }
        }
        out
    }

    /// The TOP-LEVEL comma-separated arguments of a call extracted by [`cost_call_sites`].
    fn cost_call_args(call: &str) -> Vec<String> {
        let open = call
            .find('(')
            .expect("an extracted call has an opening paren");
        let body = &call[open + 1..call.len() - 1];
        let mut args = Vec::new();
        let mut depth = 0i32;
        let mut cur = String::new();
        for c in body.chars() {
            match c {
                // Parens and square brackets only. Counting `<`/`>` as depth drives it
                // NEGATIVE on any argument containing `->` or a comparison, after which a
                // genuine top-level comma stops splitting. Not reachable from today's call
                // sites, which pass bare identifiers — elaboration bought for nothing, and
                // wrong in the direction that silently merges two arguments into one.
                '(' | '[' => {
                    depth += 1;
                    cur.push(c);
                }
                ')' | ']' => {
                    depth -= 1;
                    cur.push(c);
                }
                ',' if depth == 0 => args.push(std::mem::take(&mut cur)),
                _ => cur.push(c),
            }
        }
        if !cur.trim().is_empty() {
            args.push(cur);
        }
        args.into_iter().map(|a| a.trim().to_string()).collect()
    }

    /// The BROKEN 06.1-05 form, reproduced so its blindness is an executable fact rather
    /// than a sentence in a SUMMARY: `NAME\([^)]*\)` stops at the FIRST `)`.
    fn naive_truncating_scan(src: &str, name: &str) -> Vec<String> {
        let needle = format!("{name}(");
        let mut out = Vec::new();
        let mut from = 0usize;
        while let Some(rel) = src[from..].find(&needle) {
            let i = from + rel;
            from = i + needle.len();
            if src[..i].trim_end().ends_with("fn") {
                continue;
            }
            if let Some(close) = src[i..].find(')') {
                out.push(src[i..=(i + close)].to_string());
            }
        }
        out
    }

    /// The door prices with the REAL exogenous counts, and every other call site in this
    /// file passes literal zeros — a claim the compiler cannot make, because arity is
    /// satisfied by any expression.
    ///
    /// This is the 06.1-05 check, corrected and extended to the regressor axis. Its
    /// instrument is a balanced-paren scan; see the case table below for why, and re-run
    /// the table rather than re-reading the scanner.
    #[test]
    fn every_cost_call_site_is_found_whole_and_prices_what_it_should() {
        const SRC: &str = include_str!("forecast.rs");
        // The SHIPPED region only: this test's own case-table literals below would
        // otherwise be scanned as call sites.
        let shipped = SRC
            .split_once("\n#[cfg(test)]\nmod tests {")
            .map_or(SRC, |(head, _)| head);

        // ---- THE CASE TABLE (CLAUDE.md rule 7) ----
        //
        // MUST MATCH, and must come back WHOLE. The first row is the exact shape the
        // `[^)]*` form truncates, and it is a real shape from this crate.
        for (probe, want_args) in [
            ("let a = request_train_cost(&d, ds.len(), 0, 0, 0);", 5usize),
            (
                "request_train_cost(&d, p.points, p.n_lags, p.columns, 0)",
                5,
            ),
            (
                "let c = request_train_cost(\n    &d,\n    n_train,\n    n_lags,\n    \
                 n_event_cols,\n    n_reg_cols,\n);",
                5,
            ),
        ] {
            let found = cost_call_sites(probe, "request_train_cost");
            assert_eq!(found.len(), 1, "MUST MATCH exactly once: {probe:?}");
            assert_eq!(
                cost_call_args(&found[0]).len(),
                want_args,
                "the scan must return the WHOLE argument list of {probe:?}, got {:?}",
                cost_call_args(&found[0])
            );
        }
        // MUST NOT MATCH.
        for probe in [
            "pub fn request_train_cost(d: &NpData, n_points: usize) -> u64 {",
            "log.train_cost",
            "let x = my_request_train_cost(1, 2);",
        ] {
            assert!(
                cost_call_sites(probe, "request_train_cost").is_empty()
                    && cost_call_sites(probe, "train_cost").is_empty(),
                "MUST NOT MATCH: {probe:?}"
            );
        }
        // AND THE DEFECT ITSELF, pinned: the naive form TRUNCATES the nested-call shape.
        let nested = "let a = request_train_cost(&d, ds.len(), 0, 0, 0);";
        let naive = naive_truncating_scan(nested, "request_train_cost");
        assert_eq!(
            naive[0], "request_train_cost(&d, ds.len()",
            "the 06.1-05 regex's blindness must stay observable: if this ever stops \
             truncating, the balanced scan is no longer buying anything and the comment \
             above is wrong"
        );

        // ---- THE LIVE CLAIM ----
        let sites = cost_call_sites(shipped, "request_train_cost");
        assert_eq!(
            sites.len(),
            1,
            "the door prices a neuralprophet request at exactly ONE place, so the priced \
             cost cannot disagree with the work spent. Found {sites:?}"
        );
        let args = cost_call_args(&sites[0]);
        assert_eq!(
            args.len(),
            5,
            "the priced call must carry BOTH exogenous counts"
        );
        assert_eq!(
            args[3], "n_event_cols",
            "the door must price the REAL event-column count"
        );
        assert_eq!(
            args[4], "n_reg_cols",
            "the door must price the REAL regressor-column count — a literal 0 here is the \
             SC4 under-pricing this plan closed"
        );
        // And no OTHER cost call escapes into the shipped door.
        assert!(
            // `is_empty`, NOT `all(starts_with("request_train_cost("))`: the scanner's
            // identifier-prefix guard already drops every `train_cost(` preceded by a word
            // character, so a `request_train_cost` occurrence can never reach the predicate.
            // Written as `all` it reads as if it tolerated them and could not be observed
            // false for that reason — the assertion still does real work (a BARE
            // `train_cost(` in the shipped door fails it), so say that instead.
            cost_call_sites(shipped, "train_cost").is_empty(),
            "no bare `train_cost` call may appear in the shipped door — price through \
             request_train_cost only: {:?}",
            cost_call_sites(shipped, "train_cost")
        );
    }
}

// ------------------------------------------- the WR-03 wall harness (ignored) ----

/// Wall-clock harness for **WR-03**: the work the in-loop aggregate-dates refusal avoids.
///
/// `#[ignore]`d, because it is a MEASUREMENT and not an assertion about correctness. Run it
/// deliberately, on a RELEASE build:
///
/// ```text
/// cargo test --release -p aprender-forecast --lib wr03_aggregate_dates_wall \
///     -- --ignored --nocapture
/// ```
///
/// It drives the review's exact trigger — 1 000 holidays, each with `lower_window: 0,
/// upper_window: 0` (so `holiday_columns` reaches only 1 000 and never trips) and each
/// carrying 1 000 dates. To reproduce the BEFORE side, delete the in-loop refusal in
/// `forecast` and re-run: the same payload then parses ~1 000 000 dates and allocates
/// ~8 MB of `Vec<i64>` before the post-loop check discards all of it.
#[cfg(test)]
mod wr03_wall {
    /// One machine-parsable line. `profile=` is derived, never asserted (CLAUDE.md rule 2).
    #[test]
    #[ignore = "wall-clock measurement; run with --release -- --ignored --nocapture"]
    fn wr03_aggregate_dates_wall() {
        // A short valid series; the refusal fires at the door, so the history is only
        // required to be legal in the dimension this payload is not perturbing.
        let t0d = crate::dates::days_from_civil(2020, 1, 1);
        let ds: Vec<String> = (0..60i64)
            .map(|i| crate::dates::format_ymd(t0d + i))
            .collect();
        let y: Vec<f64> = (0..60)
            .map(|i| 10.0 + 0.05 * f64::from(i) + f64::from(i % 7))
            .collect();
        let base = crate::dates::days_from_civil(1990, 1, 1);
        let holidays: Vec<crate::types::HolidayArg> = (0..1_000i64)
            .map(|h| crate::types::HolidayArg {
                name: format!("h{h}"),
                dates: (0..1_000i64)
                    .map(|d| crate::dates::format_ymd(base + h * 1_000 + d))
                    .collect(),
                lower_window: 0,
                upper_window: 0,
            })
            .collect();
        let dates_sent: usize = holidays.iter().map(|h| h.dates.len()).sum();
        let args = crate::types::ForecastArgs {
            ds,
            y,
            horizon: 7,
            holidays: Some(holidays),
            ..crate::types::ForecastArgs::default()
        };
        let t0 = std::time::Instant::now();
        let outcome = match super::forecast(&args) {
            Err(crate::types::ForecastError::Validation(_)) => "refused",
            Ok(_) => "accepted",
            Err(e) => panic!("unexpected {e:?}"),
        };
        println!(
            "WR03 AGGREGATE DATES WALL: holidays=1000 dates_per_holiday=1000 \
             dates_sent={dates_sent} outcome={outcome} total_s={:.6} profile={}",
            t0.elapsed().as_secs_f64(),
            if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
        );
    }
}
