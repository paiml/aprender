//! The tracer (SC1): one request carrying four external regressors goes in through the
//! single stateless `forecast` door on `model: prophet`, reaches the Prophet design as
//! appended columns, and comes back carrying `extra_regressors_additive` — proven against
//! the committed Python Prophet 1.4.0 oracle at 24 columns.
//!
//! # No tolerance literal lives in this file (D-15)
//!
//! Every bar is read from `contracts/prophet-parity-v1.yaml` through
//! `test_support::equation_tolerance`. A test that hardcodes a tolerance can be loosened
//! without the contract ever noticing, which makes the contract decorative.

use super::{splice, standardize_one, RegressorChannel, RegressorSpec, Standardized};
use crate::prophet::{make_design, predict, Mode, Params, Seasonality, Spec};
use crate::test_support::{equation_tolerance, load_json};
use crate::types::{ForecastArgs, RegressorArg};

const FIXTURE: &str = "retail_regressors_prophet140.json";
const CONTRACT: &str = "prophet-parity-v1";
const SEED: u64 = 42;

fn f64s(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .expect("a JSON array")
        .iter()
        .map(|x| x.as_f64().expect("a JSON number"))
        .collect()
}

/// Read a scalar that the oracle may store either bare or as a ONE-ELEMENT ARRAY.
///
/// Not defensive coding: the spike-011 regressor fixture writes `params.k` as `[0.5029636]`
/// while the seven older spike-001/003 fixtures write it bare, so `prophet::parity`'s
/// `python_params` (which uses `as_f64()` directly) would panic on this one. Accepting both
/// keeps the fixture byte-identical to the spike original, which the `cmp` verify requires.
fn scalar(v: &serde_json::Value, what: &str) -> f64 {
    v.as_f64()
        .or_else(|| {
            v.as_array()
                .and_then(|a| a.first())
                .and_then(|x| x.as_f64())
        })
        .unwrap_or_else(|| panic!("{what} must be a number or a one-element array, got {v}"))
}

// D-17: date handling lives in ONE module. `parse_ymd` is the crate's trusted-input
// parser and is what `prophet::parity::days` already uses; a second byte-slicing path
// here panics with an opaque index message where the shared helper names the field.
// Measured: every `ds` in this fixture is a bare `YYYY-MM-DD` of length 10.
fn day(s: &str) -> i64 {
    crate::dates::parse_ymd(s)
}

/// The one tracer, in four parts: the door, the design, the standardisation constants and
/// the predict path.
#[test]
fn a_four_regressor_request_matches_the_python_oracle_at_24_columns() {
    let fx = load_json(FIXTURE);
    let exact = equation_tolerance(CONTRACT, "data_prep_exact");
    let std_bar = equation_tolerance(CONTRACT, "regressor_standardization_abs");
    let path_bar = equation_tolerance(CONTRACT, "regressor_predict_path_rel_yscale");

    let hist_ds: Vec<String> = fx["history"]["ds"]
        .as_array()
        .expect("history.ds")
        .iter()
        .map(|v| v.as_str().expect("a ds string")[..10].to_string())
        .collect();
    let hist_y = f64s(&fx["history"]["y"]);
    let horizon = usize::try_from(fx["horizon"].as_u64().expect("horizon")).expect("fits usize");
    let n_hist = hist_ds.len();
    let names: Vec<String> = fx["regressors_requested"]
        .as_array()
        .expect("regressors_requested")
        .iter()
        .map(|r| r["name"].as_str().expect("name").to_string())
        .collect();

    // ---- (a) THE DOOR: one public call, four regressors, on the prophet arm ----
    let regressors: Vec<RegressorArg> = fx["regressors_requested"]
        .as_array()
        .expect("regressors_requested")
        .iter()
        .map(|r| {
            let name = r["name"].as_str().expect("name").to_string();
            RegressorArg {
                values: f64s(&fx["regressor_values"][&name]),
                name,
                mode: Some(r["mode"].as_str().expect("mode").to_string()),
                prior_scale: Some(r["prior_scale"].as_f64().expect("prior_scale")),
                // The oracle records the literal string "auto"; on this wire auto is the
                // ABSENCE of the field. All four oracle regressors are "auto".
                standardize: None,
            }
        })
        .collect();
    let args = ForecastArgs {
        ds: hist_ds.clone(),
        y: hist_y.clone(),
        horizon,
        freq: Some("MS".into()),
        seed: Some(SEED),
        regressors: Some(regressors),
        ..ForecastArgs::default()
    };
    let resp = crate::forecast::forecast(&args)
        .expect("a four-regressor prophet request must be accepted");
    for key in [
        "promo",
        "price",
        "discount",
        "weather",
        "yearly",
        "extra_regressors_additive",
        "extra_regressors_multiplicative",
        "additive_terms",
        "multiplicative_terms",
    ] {
        assert!(
            resp.components.contains_key(key),
            "the response must carry component {key:?}; got {:?}",
            resp.components.keys().collect::<Vec<_>>()
        );
    }
    assert!(
        resp.yhat.iter().all(|v| v.is_finite()),
        "every yhat must be finite"
    );

    // ---- (b) THE DESIGN: 24 columns, in the oracle's order, with the oracle's scales ----
    let hist_days: Vec<i64> = hist_ds.iter().map(|s| day(s)).collect();
    let seas = fx["seasonalities"]["yearly"].clone();
    let spec = Spec::default_linear(vec![Seasonality {
        name: "yearly".into(),
        period: seas["period"].as_f64().expect("period"),
        order: usize::try_from(seas["fourier_order"].as_u64().expect("order")).expect("usize"),
        prior_scale: seas["prior_scale"].as_f64().expect("prior_scale"),
        mode: Mode::Additive,
    }]);
    let mut design = make_design(&hist_days, &hist_y, &spec);

    let specs: Vec<RegressorSpec> = fx["regressors_requested"]
        .as_array()
        .expect("regressors_requested")
        .iter()
        .map(|r| RegressorSpec {
            name: r["name"].as_str().expect("name").into(),
            mode: match r["mode"].as_str().expect("mode") {
                "multiplicative" => Mode::Multiplicative,
                _ => Mode::Additive,
            },
            prior_scale: r["prior_scale"].as_f64().expect("prior_scale"),
            standardize: None,
        })
        .collect();
    let full: Vec<Vec<f64>> = names
        .iter()
        .map(|n| f64s(&fx["regressor_values"][n]))
        .collect();
    let hist_vals: Vec<Vec<f64>> = full.iter().map(|v| v[..n_hist].to_vec()).collect();
    let std_regs: Vec<Standardized> = specs
        .iter()
        .zip(&hist_vals)
        .map(|(s, h)| standardize_one(s, h))
        .collect();
    splice(&mut design, &std_regs, &hist_vals);

    let want_cols: Vec<String> = fx["columns"]
        .as_array()
        .expect("columns")
        .iter()
        .map(|c| c.as_str().expect("a column name").to_string())
        .collect();
    assert_eq!(want_cols.len(), 24, "the oracle fixture has 24 columns");
    assert_eq!(
        design.k, 24,
        "the spliced design must have exactly 24 columns, got {}",
        design.k
    );
    let got_cols: Vec<String> = design.cols.iter().map(|c| c.name.clone()).collect();
    assert_eq!(
        got_cols, want_cols,
        "column NAMES must match the oracle element-for-element; Prophet's order is \
         seasonalities, then name-sorted holidays, then extra regressors in INSERTION order"
    );

    for (label, got, want) in [
        (
            "prior_scales",
            &design.prior_scales,
            &f64s(&fx["prior_scales"]),
        ),
        ("s_a", &design.s_a, &f64s(&fx["s_a"])),
        ("s_m", &design.s_m, &f64s(&fx["s_m"])),
    ] {
        assert_eq!(got.len(), want.len(), "{label} length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= exact,
                "{label}[{i}]: got {g}, oracle {w}, differ by {} (bar data_prep_exact = {exact})",
                (g - w).abs()
            );
        }
    }

    // ---- (c) THE STANDARDISATION CONSTANTS: ddof = 1, over history rows only ----
    for s in &std_regs {
        let want = &fx["extra_regressors"][&s.name];
        let want_mu = want["mu"].as_f64().expect("mu");
        let want_std = want["std"].as_f64().expect("std");
        assert!(
            (s.mu - want_mu).abs() <= std_bar,
            "{}: mu {} vs oracle {want_mu}, differ by {} (bar {std_bar})",
            s.name,
            s.mu,
            (s.mu - want_mu).abs()
        );
        assert!(
            (s.std - want_std).abs() <= std_bar,
            "{}: std {} vs oracle {want_std}, differ by {} (bar {std_bar}). A population \
             divisor (ddof = 0) gives 6.500320 where the oracle stores 6.511441581253706 \
             for `price`",
            s.name,
            s.std,
            (s.std - want_std).abs()
        );
    }

    // ---- (d) THE PREDICT PATH: the oracle's OWN MAP parameters through our `predict` ----
    //
    // Fit-independent by construction: any optimiser disagreement is excluded, so what is
    // measured here is the regressor-aware predict arithmetic alone.
    let fc_ds: Vec<i64> = fx["forecast"]["ds"]
        .as_array()
        .expect("forecast.ds")
        .iter()
        .map(|v| day(v.as_str().expect("a ds string")))
        .collect();
    assert_eq!(
        fc_ds.len(),
        n_hist + horizon,
        "the oracle grid is history + horizon"
    );
    let params = Params {
        k: scalar(&fx["params"]["k"], "params.k"),
        m: scalar(&fx["params"]["m"], "params.m"),
        delta: f64s(&fx["params"]["delta"]),
        beta: f64s(&fx["params"]["beta"]),
        sigma_obs: scalar(&fx["params"]["sigma_obs"], "params.sigma_obs"),
    };
    let fc = predict(
        &design,
        &params,
        &fc_ds,
        SEED,
        &RegressorChannel {
            specs: &std_regs,
            values: &full,
        },
    )
    .expect("the full 305-row value channel matches the 305-row grid");

    let y_scale = fx["y_scale"].as_f64().expect("y_scale");
    // The bar binds RELATIVE to y_scale. On retail_sales an ABSOLUTE 1e-10 bar passes with
    // only ~3x headroom at a y_scale of 518253 and would start failing for arithmetic
    // reasons rather than for a defect.
    let abs_bar = path_bar * y_scale;
    for (label, got, want) in [
        ("yhat", &fc.yhat, f64s(&fx["forecast"]["yhat"])),
        ("trend", &fc.trend, f64s(&fx["forecast"]["trend"])),
    ] {
        assert_eq!(got.len(), want.len(), "{label} length");
        let worst = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst <= abs_bar,
            "{label}: worst absolute difference {worst:e} exceeds \
             regressor_predict_path_rel_yscale ({path_bar:e}) x y_scale ({y_scale}) = \
             {abs_bar:e}"
        );
    }

    // ---- (d2) THE ROLL-UPS BY VALUE, not merely by key presence ----
    //
    // Added beyond the plan's four parts, deliberately: parts (a)-(d) prove the two roll-up
    // KEYS exist and that `yhat` matches, but `yhat` is built from `additive_terms` and
    // never reads `extra_regressors_additive`. A roll-up that selected the wrong columns, or
    // forgot the `y_scale` factor on the additive arm, would pass every assertion above
    // while being exactly the defect SC1 is about. The oracle publishes all nine component
    // series over the full 305-row grid, so there is no reason to assert less than that.
    let want_components = fx["forecast"]["components"]
        .as_object()
        .expect("forecast.components");
    let mut checked = 0;
    for (name, want) in want_components {
        let want = f64s(want);
        let got = fc
            .components
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.clone())
            .unwrap_or_else(|| {
                panic!(
                    "the oracle publishes component {name:?}; we produced {:?}",
                    fc.components.iter().map(|(n, _)| n).collect::<Vec<_>>()
                )
            });
        assert_eq!(got.len(), want.len(), "component {name} length");
        let worst = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst <= abs_bar,
            "component {name}: worst absolute difference {worst:e} exceeds {abs_bar:e}"
        );
        checked += 1;
    }
    assert_eq!(
        checked, 9,
        "the oracle publishes 9 components and every one must be compared; a shrinking \
         loop is how a value comparison silently stops comparing"
    );
}
