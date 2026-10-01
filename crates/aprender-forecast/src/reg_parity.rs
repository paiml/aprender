//! The external-regressor parity ladder (SC1, D-23), at BOTH committed widths.
//!
//! ONE TEST PER RUNG PER FIXTURE, so a regression names the RUNG and the FIXTURE rather
//! than "regressors broke". The rungs, in the order they must be believed:
//!
//! 0. `<f>_regressor_standardization_exact` — `mu` and `std` from the history rows alone,
//!    ddof = 1, with the auto-binary carve-out pinned exactly.
//! 1. `<f>_column_order_and_scales_exact` — the column NAME list, `prior_scales`, `s_a`
//!    and `s_m` against the oracle, and the design-matrix CELLS at their own measured bar.
//! 3. `<f>_predict_path_via_python_params` — the oracle's own MAP parameters through the
//!    regressor-aware Rust `predict`, plus the `trend*(1+mul)+add == yhat` identity.
//! 3. `<f>_components_via_python_params` — every named component the oracle publishes,
//!    including both `extra_regressors_*` roll-ups, bound RELATIVE to `y_scale`.
//! 4. `<f>_fit_objective_and_identifiability` — the Rust fit's own objective, one-sided,
//!    with the coefficient ratio table RECORDED rather than asserted.
//!
//! Rung 2 (the objective at the oracle's published log posterior) is resolved by
//! measurement rather than skipped; see `regressor_fitted_objective_slack`'s `invariants:`
//! in `contracts/prophet-parity-v1.yaml` and the note at [`oracle`] below.
//!
//! # Two bars at rung 1, deliberately not one
//!
//! Column NAMES, the `s_a` / `s_m` masks and `prior_scales` are compared at EXACT zero.
//! The design-matrix CELLS are compared at their own MEASURED bar, because the committed
//! spike records `max abs d X (first3 + last3 rows)` of about three parts in a quadrillion
//! at both widths — float association order, not disagreement. An exact bar on the cells
//! would make this ladder red against the implementation it exists to protect. Names and
//! mask bits have no arithmetic excuse and keep the exact bar.
//!
//! # NO TOLERANCE LITERAL LIVES IN THIS FILE (D-15)
//!
//! Every bar is read at test time from `contracts/prophet-parity-v1.yaml` through
//! [`equation_tolerance`]. A bar that lives in a test can be loosened without the contract
//! ever noticing; a bar that lives in the contract moves only as a `pv diff`-visible edit.
//! That claim is ENFORCED, not asserted — see the region gate at the bottom of this file
//! and the case table beside it.
//!
//! # Why this is a `--lib` module and not `tests/reg_parity.rs`
//!
//! This crate ships zero integration-test targets, and a new `tests/*.rs` file is DARK
//! until someone adds it to the one explicit `--test` line in `.github/workflows/ci.yml`
//! — a workflow edit, which is a check-in-before-acting item. In `src/` it runs inside
//! CI's existing `--lib` workspace sweep with no workflow change at all.

use crate::dates::parse_ymd;
use crate::fit::fit_prophet;
use crate::prophet::{
    make_design, predict, Design, Forecast, Holiday, Mode, Model, Params, Seasonality, Spec,
};
use crate::regressors::{splice, standardize_one, RegressorChannel, RegressorSpec, Standardized};
use crate::test_support::{constant_u64, equation_tolerance, load_json};
use serde_json::Value;
use std::collections::BTreeSet;
use std::sync::{Arc, OnceLock};

const CONTRACT: &str = "prophet-parity-v1";

/// The uncertainty seed `predict` draws with. Every rung here is a POINT-ESTIMATE
/// comparison, so this only has to be fixed; no band is asserted at either width.
const SEED: u64 = 42;

/// One committed spike-011 oracle and everything about it that is structural rather than
/// numeric — counts and names, never tolerances.
struct Oracle {
    /// The test-name stem, so a failure names the fixture.
    stem: &'static str,
    file: &'static str,
    /// The design column count the oracle publishes. Asserted EXPLICITLY: a bar on names
    /// alone would pass on a list that is a prefix of the other.
    k: usize,
    /// The components both the oracle and Rust `predict` must produce, enumerated. Asserted
    /// as SET EQUALITY and then as a count EQUALITY — an intersection plus a minimum count
    /// is satisfiable by comparing the wrong nine.
    components: &'static [&'static str],
}

/// The two committed oracles.
///
/// Both are `prophet==1.4.0` under Python 3.12, both are byte-identical to
/// `.planning/spikes/011-prophet-regressor-parity/fixtures/`, and both carry `params.lp__`
/// rather than the `log_posterior_at_map_unnormalized` the spike-001 family publishes.
const FIXTURES: [Oracle; 2] = [
    Oracle {
        stem: "retail_regressors",
        file: "retail_regressors_prophet140.json",
        k: 24,
        components: &[
            "yearly",
            "promo",
            "price",
            "discount",
            "weather",
            "extra_regressors_additive",
            "extra_regressors_multiplicative",
            "additive_terms",
            "multiplicative_terms",
        ],
    },
    Oracle {
        stem: "retail_regressors_holidays",
        file: "retail_regressors_holidays_prophet140.json",
        k: 30,
        components: &[
            "yearly",
            "blackfriday",
            "newyear",
            "holidays",
            "promo",
            "price",
            "discount",
            "weather",
            "extra_regressors_additive",
            "extra_regressors_multiplicative",
            "additive_terms",
            "multiplicative_terms",
        ],
    },
];

fn oracle(stem: &str) -> &'static Oracle {
    FIXTURES
        .iter()
        .find(|o| o.stem == stem)
        .unwrap_or_else(|| panic!("{stem} is not one of the two committed regressor oracles"))
}

// ------------------------------------------------------------------ readers ----

fn f64s(v: &Value, what: &str) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("{what} must be a JSON array"))
        .iter()
        .map(|x| {
            x.as_f64()
                .unwrap_or_else(|| panic!("{what} must hold only numbers"))
        })
        .collect()
}

fn strings(v: &Value, what: &str) -> Vec<String> {
    v.as_array()
        .unwrap_or_else(|| panic!("{what} must be a JSON array"))
        .iter()
        .map(|x| {
            x.as_str()
                .unwrap_or_else(|| panic!("{what} must hold only strings"))
                .to_string()
        })
        .collect()
}

/// Read a scalar the oracle stores as a ONE-ELEMENT ARRAY, or bare.
///
/// Not defensive coding: spike 011 writes `params.k` as a one-element array while the seven
/// spike-001 / spike-003 fixtures write it bare, so `prophet::parity`'s `python_params`
/// would panic on these two. Accepting both keeps the fixtures byte-identical to the spike
/// originals, which the `cmp` verify requires.
fn scalar(v: &Value, what: &str) -> f64 {
    v.as_f64()
        .or_else(|| v.as_array().and_then(|a| a.first()).and_then(Value::as_f64))
        .unwrap_or_else(|| panic!("{what} must be a number or a one-element array, got {v}"))
}

fn max_abs_diff(a: &[f64], b: &[f64], what: &str) -> f64 {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    assert!(!a.is_empty(), "{what}: refusing to compare empty vectors");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Flatten a JSON array-of-rows into one row-major vector.
fn flat_rows(v: &Value, what: &str) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("{what} must be an array of rows"))
        .iter()
        .flat_map(|r| f64s(r, what))
        .collect()
}

// -------------------------------------------------------------------- rig ----

/// One oracle, parsed once and rebuilt into the design the port would build itself.
struct Rig {
    fx: Value,
    /// The SPLICED history design: base columns from the unchanged `make_design`, regressor
    /// columns appended by `splice`.
    design: Design,
    std_regs: Vec<Standardized>,
    /// One array per regressor over the FULL forecast grid (history + horizon).
    values_all: Vec<Vec<f64>>,
    forecast_days: Vec<i64>,
    n_hist: usize,
    y_scale: f64,
}

/// Build the `Spec` from the fixture ALONE — never from a hand-copied default.
///
/// The spike-011 oracles publish `seasonalities` as an OBJECT keyed by name, where the
/// spike-001 / spike-003 family publishes an array, so `prophet::parity::spec_of` cannot be
/// reused verbatim. Holidays arrive as one flat row per (name, date) and are grouped here,
/// preserving first-appearance order and carrying the per-name window.
fn spec_of(fx: &Value) -> Spec {
    let seas = fx["seasonalities"]
        .as_object()
        .expect("the regressor oracles publish `seasonalities` as an object keyed by name");
    // Both oracles carry exactly one seasonality. Asserted rather than assumed: a second
    // one would make the iteration order of this map load-bearing, and `serde_json`'s
    // default map is key-sorted rather than insertion-ordered.
    assert_eq!(
        seas.len(),
        1,
        "these oracles carry ONE seasonality; with more, the column order would depend on \
         this map's iteration order, which is key-sorted and not Prophet's insertion order"
    );
    let seasonalities: Vec<Seasonality> = seas
        .iter()
        .map(|(name, s)| Seasonality {
            name: name.clone(),
            period: s["period"].as_f64().expect("seasonality.period"),
            order: usize::try_from(s["fourier_order"].as_u64().expect("fourier_order"))
                .expect("a fourier order fits usize"),
            prior_scale: s["prior_scale"].as_f64().expect("seasonality.prior_scale"),
            mode: if s["mode"].as_str() == Some("multiplicative") {
                Mode::Multiplicative
            } else {
                Mode::Additive
            },
        })
        .collect();

    let mut spec = Spec::default_linear(seasonalities);
    spec.changepoint_prior_scale = fx["changepoint_prior_scale"]
        .as_f64()
        .expect("changepoint_prior_scale");

    // Prophet's own default when the request names no holiday prior scale.
    let holiday_prior = fx["holidays_prior_scale"].as_f64().unwrap_or(10.0);
    let none: Vec<Value> = Vec::new();
    let mut holidays: Vec<Holiday> = Vec::new();
    for h in fx["holidays_requested"].as_array().unwrap_or(&none) {
        let name = h["holiday"].as_str().expect("holiday.holiday").to_string();
        let day = parse_ymd(h["ds"].as_str().expect("holiday.ds"));
        if let Some(existing) = holidays.iter_mut().find(|x| x.name == name) {
            existing.days.push(day);
        } else {
            holidays.push(Holiday {
                name,
                days: vec![day],
                lower_window: h["lower_window"].as_i64().expect("lower_window"),
                upper_window: h["upper_window"].as_i64().expect("upper_window"),
                prior_scale: holiday_prior,
            });
        }
    }
    spec.holidays = holidays;
    spec
}

fn build(o: &Oracle) -> Rig {
    let fx = load_json(o.file);
    assert_eq!(
        fx["prophet_version"], "1.4.0",
        "{}: the bar is Python Prophet 1.4.0, not whatever regenerated the fixture",
        o.stem
    );
    let hist_days: Vec<i64> = strings(&fx["history"]["ds"], "history.ds")
        .iter()
        .map(|s| parse_ymd(s))
        .collect();
    let hist_y = f64s(&fx["history"]["y"], "history.y");
    let n_hist = hist_days.len();
    let forecast_days: Vec<i64> = strings(&fx["forecast"]["ds"], "forecast.ds")
        .iter()
        .map(|s| parse_ymd(s))
        .collect();

    let mut design = make_design(&hist_days, &hist_y, &spec_of(&fx));

    // The regressor list, in the oracle's own INSERTION order. Sorting it here would make
    // the column-order rung pass for the wrong reason.
    let requested = fx["regressors_requested"]
        .as_array()
        .expect("regressors_requested");
    let specs: Vec<RegressorSpec> = requested
        .iter()
        .map(|r| RegressorSpec {
            name: r["name"].as_str().expect("regressor.name").to_string(),
            mode: if r["mode"].as_str() == Some("multiplicative") {
                Mode::Multiplicative
            } else {
                Mode::Additive
            },
            prior_scale: r["prior_scale"].as_f64().expect("regressor.prior_scale"),
            // The oracle records the literal string "auto"; on the crate's wire, auto is
            // the ABSENCE of a setting. All four oracle regressors are "auto".
            standardize: match &r["standardize"] {
                Value::Bool(b) => Some(*b),
                Value::String(s) if s == "auto" => None,
                other => panic!("regressor.standardize must be a bool or \"auto\", got {other}"),
            },
        })
        .collect();
    let values_all: Vec<Vec<f64>> = specs
        .iter()
        .map(|s| f64s(&fx["regressor_values"][&s.name], "regressor_values"))
        .collect();
    let values_hist: Vec<Vec<f64>> = values_all.iter().map(|v| v[..n_hist].to_vec()).collect();
    let std_regs: Vec<Standardized> = specs
        .iter()
        .zip(&values_hist)
        .map(|(s, h)| standardize_one(s, h))
        .collect();
    splice(&mut design, &std_regs, &values_hist);

    let y_scale = fx["y_scale"].as_f64().expect("y_scale");
    Rig {
        fx,
        design,
        std_regs,
        values_all,
        forecast_days,
        n_hist,
        y_scale,
    }
}

/// One parse and one splice per fixture per test binary, not one per test.
///
/// The two oracles are 126 KB and 136 KB; ten tests re-parsing them would be pure repeat
/// work. `OnceLock::get_or_init` blocks the second caller rather than duplicating it.
fn rig(stem: &'static str) -> Arc<Rig> {
    static CELLS: [OnceLock<Arc<Rig>>; FIXTURES.len()] =
        [const { OnceLock::new() }; FIXTURES.len()];
    let idx = FIXTURES
        .iter()
        .position(|o| o.stem == stem)
        .unwrap_or_else(|| panic!("{stem} is not a committed regressor oracle"));
    CELLS[idx]
        .get_or_init(|| Arc::new(build(&FIXTURES[idx])))
        .clone()
}

impl Rig {
    /// Python Prophet 1.4.0's MAP, verbatim from the fixture.
    fn python_params(&self) -> Params {
        Params {
            k: scalar(&self.fx["params"]["k"], "params.k"),
            m: scalar(&self.fx["params"]["m"], "params.m"),
            delta: f64s(&self.fx["params"]["delta"], "params.delta"),
            beta: f64s(&self.fx["params"]["beta"], "params.beta"),
            sigma_obs: scalar(&self.fx["params"]["sigma_obs"], "params.sigma_obs"),
        }
    }

    /// The UNSCALED negative log posterior at a parameter vector, on the SPLICED design.
    ///
    /// `Model::new` scales the objective by 1/T (the D-09 recipe) and `FitInfo.objective`
    /// is already unscaled, so the scale is undone here for the two to be comparable.
    fn objective_at(&self, p: &Params) -> f64 {
        let model = Model::new(&self.design);
        model.objective(&model.pack(p)) / model.scale
    }

    /// The oracle's own MAP parameters through the regressor-aware Rust `predict`.
    ///
    /// Fit-independent BY CONSTRUCTION: no optimiser disagreement can leak into any rung
    /// that reads this, which is what makes rungs 1 and 3 measurements of the regressor
    /// arithmetic alone.
    fn python_forecast(&self) -> Forecast {
        predict(
            &self.design,
            &self.python_params(),
            &self.forecast_days,
            SEED,
            &RegressorChannel {
                specs: &self.std_regs,
                values: &self.values_all,
            },
        )
        .expect("the full-grid value channel matches the full-grid forecast")
    }
}

// ----------------------------------------------------------------- rung 0 ----

/// Rung 0: the standardisation constants, over the HISTORY rows only, ddof = 1.
fn regressor_standardization_exact(stem: &'static str) {
    let r = rig(stem);
    let bar = equation_tolerance(CONTRACT, "regressor_standardization_abs");
    assert_eq!(
        r.std_regs.len(),
        4,
        "{stem}: both oracles request four regressors"
    );

    for s in &r.std_regs {
        let want = &r.fx["extra_regressors"][&s.name];
        let want_mu = want["mu"].as_f64().expect("extra_regressors.<name>.mu");
        let want_std = want["std"].as_f64().expect("extra_regressors.<name>.std");
        let (d_mu, d_std) = ((s.mu - want_mu).abs(), (s.std - want_std).abs());
        println!("  {stem} rung 0 {}: mu {d_mu:.2e}, std {d_std:.2e}", s.name);
        assert!(
            d_mu <= bar,
            "{stem}: {} mu {} vs oracle {want_mu}, differ by {d_mu:e} over bar {bar:e}",
            s.name,
            s.mu
        );
        assert!(
            d_std <= bar,
            "{stem}: {} std {} vs oracle {want_std}, differ by {d_std:e} over bar {bar:e}. \
             A POPULATION divisor (ddof = 0) gives 6.500320 where the oracle stores \
             6.511441581253706 for `price` — a 0.17 percent shift in every coefficient \
             that reads as arithmetic noise rather than as a wrong divisor",
            s.name,
            s.std
        );
    }

    // The auto-binary carve-out, pinned EXACTLY rather than inside the bar above. A
    // tolerance alone cannot tell "correctly exempted" from "standardised to constants that
    // happen to be close", and `promo` is the only column where the rule is observable.
    let promo = r
        .std_regs
        .iter()
        .find(|s| s.name == "promo")
        .expect("both oracles request a `promo` regressor");
    assert_eq!(
        promo.mu.to_bits(),
        0.0_f64.to_bits(),
        "{stem}: `promo` is a {{0,1}} column under auto, so it is NOT standardised and its \
         mu is exactly 0.0; got {}",
        promo.mu
    );
    assert_eq!(
        promo.std.to_bits(),
        1.0_f64.to_bits(),
        "{stem}: `promo` is a {{0,1}} column under auto, so its std is exactly 1.0; got {}",
        promo.std
    );
}

#[test]
fn retail_regressors_regressor_standardization_exact() {
    regressor_standardization_exact("retail_regressors");
}

#[test]
fn retail_regressors_holidays_regressor_standardization_exact() {
    regressor_standardization_exact("retail_regressors_holidays");
}

// ----------------------------------------------------------------- rung 1 ----

/// Rung 1: the column list, the masks and the design cells — at TWO bars, deliberately.
///
/// Names, `prior_scales`, `s_a` and `s_m` are EXACT: they are strings, echoed caller
/// numbers and mask bits, and a non-zero difference in any of them is a structural defect
/// with no arithmetic excuse. The design CELLS get their own measured bar, because the
/// committed spike records a non-zero residual there at both widths and an exact bar would
/// reject the validated port.
fn column_order_and_scales_exact(stem: &'static str) {
    let o = oracle(stem);
    let r = rig(stem);
    let exact = equation_tolerance(CONTRACT, "regressor_column_order_exact");
    let cells = equation_tolerance(CONTRACT, "regressor_design_cells_abs");

    let want_cols = strings(&r.fx["columns"], "columns");
    // COUNTS FIRST. An element-for-element name comparison alone would pass on a list that
    // is a prefix of the other, and the whole point of the 30-column fixture is that the
    // holiday block sits BETWEEN the seasonality block and the regressor block.
    assert_eq!(
        want_cols.len(),
        o.k,
        "{stem}: the oracle fixture must publish exactly {} columns",
        o.k
    );
    assert_eq!(
        r.design.k, o.k,
        "{stem}: the spliced design must have exactly {} columns, got {}",
        o.k, r.design.k
    );

    let got_cols: Vec<String> = r.design.cols.iter().map(|c| c.name.clone()).collect();
    assert_eq!(
        got_cols, want_cols,
        "{stem}: column NAMES must match the oracle element-for-element. Prophet's order is \
         seasonalities in insertion order, then holiday columns sorted by the generated \
         `{{name}}_delim_{{±offset}}` string (so +0, +1, +2, -1, because '+' sorts before \
         '-' in ASCII), then extra regressors in INSERTION order.\n  rust:   {got_cols:?}\n  \
         oracle: {want_cols:?}"
    );

    for (label, got, want) in [
        (
            "prior_scales",
            &r.design.prior_scales,
            f64s(&r.fx["prior_scales"], "prior_scales"),
        ),
        ("s_a", &r.design.s_a, f64s(&r.fx["s_a"], "s_a")),
        ("s_m", &r.design.s_m, f64s(&r.fx["s_m"], "s_m")),
    ] {
        let d = max_abs_diff(got, &want, label);
        assert!(
            d <= exact,
            "{stem}: {label} differs by {d:e} against an EXACT bar of {exact:e}. These are \
             echoed caller numbers and mask bits, not arithmetic — a non-zero difference \
             here is a structural defect, and relaxing this to the design-cell bar would \
             hide it"
        );
    }

    // The design CELLS, at their own MEASURED bar. The committed spike records a worst
    // residual of about three parts in a quadrillion at BOTH widths; the residual is float
    // association order in `(v - mu) / std`, not disagreement. Printed beside the bar so
    // drift is visible long before it is a failure.
    let k = r.design.k;
    let d_first = max_abs_diff(
        &r.design.x[..3 * k],
        &flat_rows(&r.fx["X_first3"], "X_first3"),
        "X first 3 rows",
    );
    let d_last = max_abs_diff(
        &r.design.x[(r.n_hist - 3) * k..r.n_hist * k],
        &flat_rows(&r.fx["X_last3"], "X_last3"),
        "X last 3 rows",
    );
    let worst = d_first.max(d_last);
    println!(
        "  {stem} rung 1: K={k}, names equal, prior_scales/s_a/s_m exact, \
         X cells worst {worst:.3e} against bar {cells:.3e}"
    );
    assert!(
        worst <= cells,
        "{stem}: design cells differ by {worst:e}, over the MEASURED bar {cells:e}. If this \
         is order unity the column order or the standardisation divisor is wrong; if it is \
         near the bar, the arithmetic moved and the bar must be RE-MEASURED in the \
         contract, never widened in this file"
    );
}

#[test]
fn retail_regressors_column_order_and_scales_exact() {
    column_order_and_scales_exact("retail_regressors");
}

#[test]
fn retail_regressors_holidays_column_order_and_scales_exact() {
    column_order_and_scales_exact("retail_regressors_holidays");
}

// ----------------------------------------------------------------- rung 3 ----

/// Rung 3a: the oracle's MAP parameters through Rust `predict`, plus the reconstruction
/// identity. Fit-independent, so no optimiser difference can excuse a miss.
fn predict_path_via_python_params(stem: &'static str) {
    let r = rig(stem);
    let f = r.python_forecast();

    assert_eq!(
        r.forecast_days.len(),
        f.yhat.len(),
        "{stem}: predict must return one row per requested day"
    );
    assert_eq!(
        &r.forecast_days[..r.n_hist],
        &strings(&r.fx["history"]["ds"], "history.ds")
            .iter()
            .map(|s| parse_ymd(s))
            .collect::<Vec<_>>()[..],
        "{stem}: the oracle's forecast grid must start with the history"
    );

    let bar = equation_tolerance(CONTRACT, "regressor_predict_path_rel_yscale") * r.y_scale;
    let d_yhat = max_abs_diff(&f.yhat, &f64s(&r.fx["forecast"]["yhat"], "yhat"), "yhat");
    let d_trend = max_abs_diff(
        &f.trend,
        &f64s(&r.fx["forecast"]["trend"], "trend"),
        "trend",
    );
    println!(
        "  {stem} rung 3: yhat {d_yhat:.2e}, trend {d_trend:.2e} \
         (y_scale {}, relative bar {bar:.2e})",
        r.y_scale
    );
    assert!(
        d_yhat <= bar && d_trend <= bar,
        "{stem}: the oracle's params through Rust predict differ by yhat {d_yhat:e} / trend \
         {d_trend:e}, over the y_scale-relative bar {bar:e}. The bar is RELATIVE on purpose: \
         at a y_scale of {} an absolute bar would be measuring float size rather than parity",
        r.y_scale
    );

    // The decomposition must reconstruct the number the tool returns — an INTERNAL
    // identity, not an oracle comparison, and the one that proves a caller charting
    // components is charting the same model as the forecast.
    let comp = |name: &str| -> Vec<f64> {
        f.components
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.clone())
            .unwrap_or_else(|| panic!("{stem}: predict must return a `{name}` component"))
    };
    let (add, mul) = (comp("additive_terms"), comp("multiplicative_terms"));
    let recon: Vec<f64> = (0..f.yhat.len())
        .map(|i| f.trend[i] * (1.0 + mul[i]) + add[i])
        .collect();
    let d_recon = max_abs_diff(&recon, &f.yhat, "reconstruction");
    assert!(
        d_recon <= equation_tolerance(CONTRACT, "components_rebuild_yhat_abs"),
        "{stem}: trend*(1+multiplicative_terms)+additive_terms misses yhat by {d_recon:e}"
    );
}

#[test]
fn retail_regressors_predict_path_via_python_params() {
    predict_path_via_python_params("retail_regressors");
}

#[test]
fn retail_regressors_holidays_predict_path_via_python_params() {
    predict_path_via_python_params("retail_regressors_holidays");
}

/// Rung 3b: every named component the oracle publishes, bound RELATIVE to `y_scale`.
///
/// THE EXPECTED KEY SET IS ASSERTED FIRST, as set EQUALITY against an explicit enumeration,
/// and the compared count is asserted as EQUALITY afterwards. An intersection plus a
/// minimum count is satisfiable by comparing the wrong nine components, which is how a
/// value comparison silently stops comparing.
fn components_via_python_params(stem: &'static str) {
    let o = oracle(stem);
    let r = rig(stem);
    let f = r.python_forecast();
    let bar = equation_tolerance(CONTRACT, "regressor_components_rel_yscale") * r.y_scale;

    // Derived quantities have no Rust point-estimate twin and are skipped BY NAME, exactly
    // as the seven-fixture ladder does. Neither of these two oracles publishes them today;
    // the filter stays so a regenerated fixture that grew them cannot silently change the
    // compared set.
    let derived = |n: &str| n.ends_with("_lower") || n.ends_with("_upper") || n == "cap";

    let want_keys: BTreeSet<String> = o.components.iter().map(|s| (*s).to_string()).collect();
    assert_eq!(
        want_keys.len(),
        o.components.len(),
        "{stem}: the expected component list must not repeat a name"
    );
    let got_keys: BTreeSet<String> = f
        .components
        .iter()
        .map(|(n, _)| n.clone())
        .filter(|n| !derived(n))
        .collect();
    assert_eq!(
        got_keys, want_keys,
        "{stem}: Rust `predict` must produce EXACTLY the enumerated component set"
    );

    let py = r.fx["forecast"]["components"]
        .as_object()
        .expect("forecast.components");
    let py_keys: BTreeSet<String> = py.keys().filter(|n| !derived(n)).cloned().collect();
    assert_eq!(
        py_keys, want_keys,
        "{stem}: the oracle must publish EXACTLY the enumerated component set"
    );

    let mut checked = 0_usize;
    let mut report: Vec<String> = Vec::new();
    for name in &want_keys {
        let got = f
            .components
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.clone())
            .unwrap_or_else(|| panic!("{stem}: missing component {name}"));
        let want = f64s(&py[name], name);
        let d = max_abs_diff(&got, &want, name);
        assert!(
            d <= bar,
            "{stem}: component `{name}` differs by {d:e}, over the y_scale-relative bar \
             {bar:e}. `extra_regressors_additive` is SCALED BY y_scale and \
             `extra_regressors_multiplicative` is a FRACTION that is not rescaled — a \
             roll-up that rescaled the multiplicative arm is wrong by five orders of \
             magnitude and lands here"
        );
        report.push(format!("{name} {d:.1e}"));
        checked += 1;
    }
    println!(
        "  {stem} rung 3 components ({checked}): {}",
        report.join(", ")
    );
    assert_eq!(
        checked,
        o.components.len(),
        "{stem}: exactly {} components must be compared. This is EQUALITY, not a floor: a \
         rung that grew or shrank its comparison has to say so",
        o.components.len()
    );
}

#[test]
fn retail_regressors_components_via_python_params() {
    components_via_python_params("retail_regressors");
}

#[test]
fn retail_regressors_holidays_components_via_python_params() {
    components_via_python_params("retail_regressors_holidays");
}

// ----------------------------------------------------------------- rung 2 ----

/// Rung 2: the Rust objective at the oracle's MAP against the oracle's OWN published
/// objective — asserted as an EXACT digit-for-digit identity, with no tolerance at all.
///
/// # Why this rung exists here, and why it does not bind `objective_at_python_map_abs`
///
/// The plan that produced this module recorded rung 2 as unavailable, on the premise that
/// neither regressor oracle publishes a Python objective. That premise is half wrong: both
/// publish `params.lp__`, Stan's log posterior at the MAP. Measured once, against the
/// existing rung-2 arithmetic (`Model::objective(pack(theta_py)) / Model::scale`, compared
/// as a SUM because the fixture stores a positive log posterior while the model computes a
/// negative one):
///
/// | fixture | `f_rust` | `lp__` | `f_rust + lp__` |
/// |---|---|---|---|
/// | 24 columns | -1022.494294424 | 1022.4943 | 5.576e-6 |
/// | 30 columns | -1022.462291772 | 1022.4623 | 8.228e-6 |
///
/// That is far outside the crate-wide `objective_at_python_map_abs` bar of one part in a
/// billion, so extending that equation's domain to these two fixtures would be writing
/// something FALSE into the contract. But the residual is not a model disagreement either,
/// and the guessed explanation — that Stan's `lp__` carries Jacobian and prior terms this
/// port's objective does not — is REFUTED by the numbers above: the oracle stores `lp__`
/// rounded to FOUR DECIMAL PLACES, and both residuals are bounded by half of that last
/// place. `lp__` is exactly the Rust objective rounded to four decimals, at both widths.
///
/// So the honest rung is the exact one: assert that the Rust objective agrees with the
/// oracle's own objective in EVERY DIGIT THE ORACLE PUBLISHED. It reads no bar, because
/// there is no tolerance in it — which is also why it is not a `float_tolerance` key in
/// `contracts/prophet-parity-v1.yaml`.
///
/// What this buys the rest of the ladder: rung 4's reference is the Rust objective at the
/// oracle's MAP rather than a published Python number. That stand-in is legitimate only if
/// the quantity really IS Python's objective — and this rung is the proof, on the same two
/// fixtures, exactly as `objective_at_python_map_abs` is the proof for the spike-001 family.
fn objective_at_python_map(stem: &'static str) {
    let r = rig(stem);
    let f_rust = r.objective_at(&r.python_params());
    let lp = scalar(&r.fx["params"]["lp__"], "params.lp__");
    assert!(
        f_rust.is_finite() && lp.is_finite(),
        "{stem}: f_rust {f_rust} and lp__ {lp} must both be finite before they are compared"
    );

    // The oracle publishes `lp__` to four decimals. Compare at that precision by scaling to
    // integers: `(x * 10_000.0).round()` is exact at these magnitudes, where reconstructing
    // a rounded f64 by dividing back down is not — and a bit-comparison of the
    // reconstruction would fail for a reason with nothing to do with parity.
    let scaled = |x: f64| (x * 10_000.0).round();
    let (got, want) = (scaled(-f_rust), scaled(lp));
    println!(
        "  {stem} rung 2: f_rust {f_rust:.9}, oracle lp__ {lp:.4}, sum {:.3e} \
         (bounded by the oracle's own last published place)",
        f_rust + lp
    );
    assert!(
        got.to_bits() == want.to_bits(),
        "{stem} rung 2: the Rust objective at the oracle's MAP is {f_rust:.9}, so rounded to \
         the four decimals the oracle publishes it is {:.4}; the oracle published {lp:.4}. \
         These must agree in EVERY DIGIT the oracle wrote down. A disagreement means the \
         priors, the exact L1 on `delta`, the sigma terms, the Fourier column order or the \
         changepoint grid diverged — and nothing downstream, INCLUDING rung 4 whose \
         reference is this same quantity, means anything until it is green",
        -f_rust
    );
}

#[test]
fn retail_regressors_objective_at_python_map() {
    objective_at_python_map("retail_regressors");
}

#[test]
fn retail_regressors_holidays_objective_at_python_map() {
    objective_at_python_map("retail_regressors_holidays");
}

// ----------------------------------------------------------------- rung 4 ----

/// Rung 4: what the Rust MAP fit itself reaches on the SPLICED design, ONE-SIDED, plus the
/// coefficient identifiability table — RECORDED, never asserted.
///
/// # The rule this rung enforces, and the one it refuses
///
/// ENFORCED: the fitted objective may not land more than the contract's slack ABOVE the
/// objective at the oracle's MAP. REFUSED: any assertion on a coefficient VALUE. Rust's
/// `price` beta came out about 4.7x from Python's while the objective slack was about
/// -3.42 — Rust found a BETTER optimum on a flat ridge. `discount` is a six-month cosine
/// and therefore a harmonic of the yearly seasonality; its coefficient is not identifiable
/// at all. Asserting on either would make this ladder red for being right, so the table is
/// printed and carried in the assertion message instead.
///
/// FINITENESS IS ASSERTED FIRST, and is not a value assertion. A NaN compares false against
/// every bound, so an ill-conditioned solve would make the one-sided slack below, the ratio
/// table and every downstream comparison pass while proving nothing.
fn fit_objective_and_identifiability(stem: &'static str) {
    let r = rig(stem);
    let rounds = usize::try_from(constant_u64(CONTRACT, "max_rounds")).expect("max_rounds");
    let (fitted, info) = fit_prophet(&r.design, rounds);

    // ---- finiteness, before any tolerance is applied ----
    assert_eq!(
        fitted.beta.len(),
        r.design.k,
        "{stem}: the fit must return one coefficient per design column"
    );
    if let Some(i) = fitted.beta.iter().position(|v| !v.is_finite()) {
        panic!(
            "{stem}: fitted beta[{i}] is {} — non-finite. A NaN compares false against every \
             bound, so the one-sided slack below and the ratio table would BOTH pass while \
             proving nothing",
            fitted.beta[i]
        );
    }
    assert!(
        info.objective.is_finite(),
        "{stem}: the fitted objective is {}, not a finite number",
        info.objective
    );

    // ---- the identifiability table: RECORDED, not asserted ----
    let py = r.python_params();
    let base_k = r.design.k - r.std_regs.len();
    let table = r
        .std_regs
        .iter()
        .enumerate()
        .map(|(j, s)| {
            let (o, g) = (py.beta[base_k + j], fitted.beta[base_k + j]);
            let ratio = if o == 0.0 { f64::NAN } else { g / o };
            format!("{}: oracle {o:+.6} rust {g:+.6} ratio {ratio:+.2}x", s.name)
        })
        .collect::<Vec<_>>()
        .join(" | ");

    // ---- the objective control, ONE-SIDED ----
    let reference = r.objective_at(&py);
    let slack = equation_tolerance(CONTRACT, "regressor_fitted_objective_slack");
    let delta = info.objective - reference;
    println!(
        "  {stem} rung 4: f_rust(fit) {:.4} vs f_rust(oracle MAP) {reference:.4} \
         (delta {delta:+.4}, bar +{slack}); {} rounds, {} iters, status {}\n    \
         identifiability: {table}",
        info.objective, info.rounds, info.iterations, info.status
    );
    assert!(
        info.objective <= reference + slack,
        "{stem}: the Rust fit landed at {:.6} against the objective at the oracle's MAP \
         {reference:.6}, i.e. {delta:+.6}, over the contract's ONE-SIDED slack {slack}. A \
         LOWER objective is a BETTER MAP and passes; only EXCEEDING the reference is a \
         defect. Coefficients at this point, for the record and NOT as a bar — {table}",
        info.objective
    );

    // D-09 diagnostics, observable on the spliced design exactly as on the base one.
    assert!(
        info.rounds <= rounds,
        "{stem}: {} restart rounds exceeds the contract's max_rounds {rounds}",
        info.rounds
    );
    assert!(
        !info.budget_hit,
        "{stem}: the cooperative fit budget was crossed at a round boundary — a \
         fixture-sized fit on {} columns must never reach it",
        r.design.k
    );
    assert!(
        info.status == "Stalled" || info.status == "Converged",
        "{stem}: L-BFGS terminal status {:?} is not one the D-09 recipe accepts",
        info.status
    );
}

#[test]
fn retail_regressors_fit_objective_and_identifiability() {
    fit_objective_and_identifiability("retail_regressors");
}

#[test]
fn retail_regressors_holidays_fit_objective_and_identifiability() {
    fit_objective_and_identifiability("retail_regressors_holidays");
}

// GSD_REG_PARITY-GATE_REGION_END
//
// EVERYTHING ABOVE THIS LINE IS THE SCANNED REGION. Everything below is the gate itself,
// its literal set and its case table — all three of which necessarily CONTAIN the numbers
// they hunt, which is exactly why the scan stops here. The tail is not unguarded: the gate
// asserts it holds no test attribute beyond its own two, so no rung can be hidden below the
// line to escape the scan.

/// The no-literal region gate (D-15, T-06.1-06) and the case table that proves it
/// discriminates.
///
/// # Why this is Rust and not a shell grep
///
/// The prior-phase gate over `prophet.rs::parity` is a bash grep inside a plan's `<verify>`.
/// CLAUDE.md Verification Discipline rule 7 requires a guard regex to ship a
/// must-match / must-not-match case table — and a Rust test cannot exercise a bash regex,
/// so a shell-only gate plus a Rust case table would be TWO artifacts free to drift apart.
/// One implementation, one case table, both under `--lib`, both inside CI's existing sweep.
/// The plan's `<verify>` runs these two tests rather than re-implementing the scan.
///
/// The repo's guard patterns have been wrong five times, and every one was caught by a case
/// table rather than by review. Re-run the table; do not re-read the pattern.
#[cfg(test)]
mod no_literal {
    /// Every tolerance `contracts/prophet-parity-v1.yaml` owns and this ladder asserts, in
    /// both the bare and the `1.0`-prefixed spellings Rust accepts.
    ///
    /// `1e-15` is in this set deliberately: it is `regressor_predict_path_rel_yscale`, the
    /// tightest bar in the phase and SC1's own number, and the first draft of this pattern
    /// omitted it — a gate blind to the tolerance most worth protecting. `3.33e-15` is the
    /// MEASURED design-cell residual, which belongs in prose and never in code.
    const BANNED: [&str; 10] = [
        "1e-14", "1.0e-14", "1e-15", "1.0e-15", "1e-10", "1.0e-10", "0.5", "0.02", "0.0079",
        "3.33e-15",
    ];

    /// The boundary marker, assembled at run time so the full string appears in the source
    /// exactly ONCE — at the boundary. A `const` holding it literally would appear twice and
    /// the "exactly once" assertion below would be vacuous.
    fn marker() -> String {
        format!("{}-{}", "GSD_REG_PARITY", "GATE_REGION_END")
    }

    /// Remove line comments (`//`, `///`, `//!`) AND block comments (`/* … */`, nested,
    /// multi-line), keeping string literals, which ARE code.
    ///
    /// Stripping only line comments — as the first draft of this gate did — turns a block
    /// comment explaining why a bar is what it is into a CI failure, and this module's
    /// rationale is full of exactly those numbers.
    ///
    /// SCOPE, stated rather than assumed: character literals are not tracked, so a literal
    /// double-quote character would flip the string state and mask the rest of the region.
    /// That is the one blind spot, and the gate ASSERTS the region contains no such literal
    /// rather than hoping. Lifetimes are unaffected precisely BECAUSE char literals are not
    /// tracked.
    fn strip_comments(src: &str) -> String {
        let b: Vec<char> = src.chars().collect();
        let mut out = String::with_capacity(src.len());
        let (mut i, mut depth) = (0_usize, 0_usize);
        let (mut in_line, mut in_str) = (false, false);
        while i < b.len() {
            let c = b[i];
            let n = if i + 1 < b.len() { b[i + 1] } else { '\0' };
            if in_line {
                if c == '\n' {
                    in_line = false;
                    out.push('\n');
                }
                i += 1;
            } else if depth > 0 {
                if c == '/' && n == '*' {
                    depth += 1;
                    i += 2;
                } else if c == '*' && n == '/' {
                    depth -= 1;
                    i += 2;
                } else {
                    if c == '\n' {
                        out.push('\n');
                    }
                    i += 1;
                }
            } else if in_str {
                if c == '\\' {
                    i += 2;
                } else {
                    if c == '"' {
                        in_str = false;
                    }
                    out.push(c);
                    i += 1;
                }
            } else if c == '/' && n == '/' {
                in_line = true;
                i += 2;
            } else if c == '/' && n == '*' {
                depth = 1;
                i += 2;
            } else {
                if c == '"' {
                    in_str = true;
                }
                out.push(c);
                i += 1;
            }
        }
        out
    }

    /// The first banned tolerance appearing as a COMPLETE numeric token, or `None`.
    ///
    /// The token boundary is what separates a bar from an innocent number: a bare substring
    /// search flags `10.5` for containing the objective-slack bar, and a gate that cries
    /// wolf gets deleted. A match is rejected when the character immediately before or after
    /// is a digit or a dot — which is why `10.5`, `0.05` and a version string are all clean
    /// while a bare assignment of the same number is not.
    fn first_banned(code: &str) -> Option<&'static str> {
        let bytes = code.as_bytes();
        let edge = |o: Option<&u8>| o.is_some_and(|c| c.is_ascii_digit() || *c == b'.');
        for pat in BANNED {
            let mut from = 0;
            while let Some(rel) = code[from..].find(pat) {
                let at = from + rel;
                let before = at.checked_sub(1).map(|p| &bytes[p]);
                let after = bytes.get(at + pat.len());
                if !edge(before) && !edge(after) {
                    return Some(pat);
                }
                from = at + 1;
            }
        }
        None
    }

    /// Does `src` open a RAW string literal (`r"…"`, `r#"…"#`)?
    ///
    /// The identifier boundary is the whole content of this predicate, and the first draft
    /// did not have it: a bare search for the two characters `r` then a quote fires on
    /// `"weather",` — the last letter of a perfectly ordinary string followed by its own
    /// closing quote — which made the gate red against clean code. A raw string opener is
    /// an `r` (optionally followed by hashes) that is NOT part of a longer identifier.
    fn has_raw_string(src: &str) -> bool {
        let b = src.as_bytes();
        for (i, &c) in b.iter().enumerate() {
            if c != b'r' {
                continue;
            }
            let prev_is_ident = i
                .checked_sub(1)
                .is_some_and(|p| b[p].is_ascii_alphanumeric() || b[p] == b'_');
            if prev_is_ident {
                continue;
            }
            let mut j = i + 1;
            while b.get(j) == Some(&b'#') {
                j += 1;
            }
            if b.get(j) == Some(&b'"') {
                return true;
            }
        }
        false
    }

    /// The gate: no contract-owned tolerance literal lives in the ladder's non-comment code.
    #[test]
    fn tolerance_literals_live_in_the_contract_not_in_this_module() {
        let src = include_str!("reg_parity.rs");
        let m = marker();
        assert_eq!(
            src.matches(m.as_str()).count(),
            1,
            "the region boundary marker must appear EXACTLY once; a second copy makes the \
             scanned region ambiguous and zero copies make it empty"
        );
        let at = src.find(m.as_str()).expect("the marker is present");
        let region = &src[..at];
        assert!(
            region.len() > 10_000,
            "the scanned region is {} bytes, which means the marker moved and this gate is \
             now scanning almost nothing",
            region.len()
        );
        // The stripper's one blind spot, asserted absent rather than assumed absent.
        assert!(
            !region.contains("'\"'"),
            "the region contains a literal double-quote CHARACTER, the one construct \
             `strip_comments` cannot track: it would flip the string state and mask \
             everything after it"
        );
        assert!(
            !has_raw_string(region),
            "the region contains a raw string literal, which `strip_comments` does not model"
        );

        let code = strip_comments(region);
        assert!(
            first_banned(&code).is_none(),
            "a contract-owned tolerance literal ({:?}) appears in the ladder's non-comment \
             code. Every bar must be read at test time through `equation_tolerance`: a bar \
             written here can be loosened without the contract ever noticing, which makes \
             every tolerance in `contracts/prophet-parity-v1.yaml` decorative. Put the \
             NUMBER in the contract and the RATIONALE in a comment — comments are stripped \
             before this search precisely so the rationale can stay in the file",
            first_banned(&code)
        );

        // ANTI-VACUITY for the tail. The scan stops at the marker, so the one way to evade
        // it is to put a rung BELOW the line. Exactly two tests live there: this one and the
        // case table. The needle is assembled at run time so it does not match itself.
        let needle = format!("#[{}]", "test");
        assert_eq!(
            src[at..].matches(needle.as_str()).count(),
            2,
            "exactly two tests may live below the region marker — this gate and its case \
             table. A rung below the line would be UNSCANNED, which is the only way to \
             hardcode a bar in this module and stay green"
        );
    }

    /// The must-match / must-not-match case table (CLAUDE.md Verification Discipline #7).
    ///
    /// The `0.5` and `0.02` entries in [`BANNED`] are known traps for a legitimate midpoint
    /// or half-width written in code. They stay, because both ARE contract-owned tolerances
    /// here and dropping them would let the objective-slack bar be hardcoded. The trap is
    /// pinned by the must-not-match rows instead, so the next person meets it as a DECISION
    /// rather than as a surprise.
    #[test]
    fn case_table_proves_the_pattern_discriminates() {
        let cases: [(&str, bool); 15] = [
            // ---- MUST MATCH: a bar written into code ----
            ("let tol = 1e-15;", true),
            ("assert!(d < 1.0e-14)", true),
            ("if x > 0.5 {", true),
            ("let b = 1e-10;", true),
            ("let rel = 0.02;", true),
            ("let band = 0.0079;", true),
            ("let measured = 3.33e-15;", true),
            ("let t = 1.0e-15_f64;", true),
            // ---- MUST NOT MATCH: comments, contract reads, near-miss numbers ----
            ("// the measured residual is 3.33e-15", false),
            ("/* 1e-14 has 3x headroom over the measured value */", false),
            ("equation_tolerance(\"regressor_design_cells_abs\")", false),
            ("const V: &str = \"1.5.0\";", false),
            ("let v = x[15];", false),
            ("let p = 0.05;", false),
            ("let q = 10.5;", false),
        ];
        for (src, want) in cases {
            let hit = first_banned(&strip_comments(src)).is_some();
            assert_eq!(
                hit, want,
                "case table row {src:?}: expected match = {want}, got {hit}. Re-run this \
                 table rather than re-reading the pattern — every wrong guard regex in this \
                 repo was caught by a table and none by review"
            );
        }
        // A table that exercised only one verdict would prove one half of nothing.
        assert!(
            cases.iter().any(|(_, w)| *w) && cases.iter().any(|(_, w)| !*w),
            "the case table must exercise BOTH verdicts"
        );

        // SECOND PATTERN, SECOND TABLE. `has_raw_string` is a guard regex in its own right
        // and gets its own cases, because its first draft was wrong in exactly the way rule
        // 7 predicts: it fired on `"weather",`, where the `r` is the last letter of the
        // word and the quote is the string's own terminator.
        let raw_cases: [(&str, bool); 6] = [
            ("let s = r\"x\";", true),
            ("let s = r#\"x\"#;", true),
            ("let s = r##\"x\"##;", true),
            ("            \"weather\",", false),
            ("let four = \"fourier_order\";", false),
            ("let n = \"a number\";", false),
        ];
        for (src, want) in raw_cases {
            assert_eq!(
                has_raw_string(src),
                want,
                "raw-string case table row {src:?}: expected {want}"
            );
        }
    }
}
