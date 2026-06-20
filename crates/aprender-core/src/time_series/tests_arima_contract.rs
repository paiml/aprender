// =========================================================================
// FALSIFY-ARIMA: arima-v1.yaml contract (aprender ARIMA)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-ARIMA-* tests
//   Why 2: ARIMA tests only in tests/contracts/, not near implementation
//   Why 3: no mapping from arima-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: ARIMA was "obviously correct" (Box-Jenkins methodology)
//
// References:
//   - provable-contracts/contracts/arima-v1.yaml
//   - Box, Jenkins, Reinsel (2015) "Time Series Analysis"
// =========================================================================

use super::*;
use crate::primitives::Vector;

/// FALSIFY-ARIMA-001: Forecast length matches requested periods
#[test]
fn falsify_arima_001_forecast_length() {
    let data = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

    let mut arima = ARIMA::new(1, 0, 0);
    arima.fit(&data).expect("fit");

    let forecast = arima.forecast(5).expect("forecast");
    assert_eq!(
        forecast.len(),
        5,
        "FALSIFIED ARIMA-001: {} forecasts for 5 requested",
        forecast.len()
    );
}

/// FALSIFY-ARIMA-002: Forecasts are finite
#[test]
fn falsify_arima_002_finite_forecasts() {
    let data = Vector::from_slice(&[10.0, 12.0, 11.0, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5, 16.0]);

    let mut arima = ARIMA::new(1, 0, 0);
    arima.fit(&data).expect("fit");

    let forecast = arima.forecast(3).expect("forecast");
    for (i, &v) in forecast.as_slice().iter().enumerate() {
        assert!(
            v.is_finite(),
            "FALSIFIED ARIMA-002: forecast[{i}] = {v} is not finite"
        );
    }
}

/// FALSIFY-ARIMA-003: Deterministic forecasts
#[test]
fn falsify_arima_003_deterministic() {
    let data = Vector::from_slice(&[1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0, 5.0, 7.0]);

    let mut arima = ARIMA::new(1, 0, 0);
    arima.fit(&data).expect("fit");

    let f1 = arima.forecast(3).expect("forecast 1");
    let f2 = arima.forecast(3).expect("forecast 2");
    for i in 0..3 {
        assert_eq!(
            f1[i], f2[i],
            "FALSIFIED ARIMA-003: forecast differs at index {i}"
        );
    }
}

/// FALSIFY-ARIMA-004: Order is preserved
#[test]
fn falsify_arima_004_order_preserved() {
    let arima = ARIMA::new(2, 1, 1);
    let (p, d, q) = arima.order();
    assert_eq!(p, 2, "FALSIFIED ARIMA-004: p={p}, expected 2");
    assert_eq!(d, 1, "FALSIFIED ARIMA-004: d={d}, expected 1");
    assert_eq!(q, 1, "FALSIFIED ARIMA-004: q={q}, expected 1");
}

mod arima_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    // FALSIFY-ARIMA-001-prop: Forecast length matches for random horizons
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn falsify_arima_001_prop_forecast_length(
            h in 1..=10usize,
            seed in 0..200u32,
        ) {
            let data: Vec<f64> = (0..20)
                .map(|i| ((i as f64 + seed as f64) * 0.37).sin() * 10.0 + 20.0)
                .collect();
            let v = Vector::from_vec(data);

            let mut arima = ARIMA::new(1, 0, 0);
            arima.fit(&v).expect("fit");

            let forecast = arima.forecast(h).expect("forecast");
            prop_assert_eq!(
                forecast.len(),
                h,
                "FALSIFIED ARIMA-001-prop: {} forecasts for {} requested",
                forecast.len(), h
            );
        }
    }

    // FALSIFY-ARIMA-002-prop: Forecasts are finite for random data
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn falsify_arima_002_prop_finite_forecasts(
            seed in 0..200u32,
        ) {
            let data: Vec<f64> = (0..20)
                .map(|i| ((i as f64 + seed as f64) * 0.37).sin() * 10.0 + 20.0)
                .collect();
            let v = Vector::from_vec(data);

            let mut arima = ARIMA::new(1, 0, 0);
            arima.fit(&v).expect("fit");

            let forecast = arima.forecast(5).expect("forecast");
            for (i, &val) in forecast.as_slice().iter().enumerate() {
                prop_assert!(
                    val.is_finite(),
                    "FALSIFIED ARIMA-002-prop: forecast[{}]={} not finite",
                    i, val
                );
            }
        }
    }
}

/// FALSIFY-ARIMA-INTEGRATE-D2 (PMAT-834): reverse-differencing for d >= 2 must seed each
/// un-differencing pass with the last value of the corresponding INTERMEDIATE difference, not
/// y[n] every time. On a perfectly quadratic series y_t (constant 2nd difference) the 1-step
/// forecast must continue the parabola; the prior code re-seeded every pass with y[n], so for
/// d == 2 the forecast overshoots badly. Series [10,20,35,55,80] (1st diffs 10,15,20,25; 2nd
/// diff constant 5) continues to 110. Pre-fix (both seeds = y[n] = 80) overshoots to ~165.
#[test]
fn falsify_arima_integrate_d2_seeds_intermediate_difference() {
    let data = Vector::from_slice(&[10.0, 20.0, 35.0, 55.0, 80.0]);
    let mut arima = ARIMA::new(0, 2, 0);
    arima.fit(&data).expect("fit");
    let f = arima.forecast(1).expect("forecast");
    // Correct continuation of the parabola is 110 (next 1st diff 30, plus y[n]=80).
    // RED pre-fix: ~165 (double-counts y[n]). GREEN post-fix: ~110.
    assert!(
        f[0] < 135.0,
        "ARIMA(0,2,0) forecast {} overshot — reverse-differencing re-seeded with y[n] instead of the last 1st-difference (expected ~110, pre-fix ~165)",
        f[0]
    );
}

/// FALSIFY-ARIMA-AR-CENTERING (PMAT-862, contract arima-ar-centering-v1):
/// `ARIMA(p,0,q)` must fit the MEAN-CENTERED Box-Jenkins model
/// `y_t - mu = sum phi_k (y_{t-k} - mu) + e_t`. The buggy code estimated AR coefficients on the
/// UNCENTERED levels: `phi = sum y_i*y_{i-1-lag} / sum y_{i-1-lag}^2`. For a stationary series with
/// nonzero mean mu, both sums are dominated by `n*mu^2`, so every coefficient collapsed to ~1.0
/// regardless of the true autocorrelation, and `ARIMA(1,0,0)` 1-step forecasts diverged to ~2x the
/// series level.
///
/// Series: a deterministic stationary AR(1) with mu ~= 49.84, true phi = 0.5 (mild positive
/// autocorrelation), range ~= 4.74 (generated with statsmodels, seed 7).
///   RED  (uncentered): coef ~= 1.0000, 1-step forecast ~= 100.93 (~2x level).
///   GREEN (centered) : coef ~= 0.370 (demeaned OLS phi_hat), 1-step forecast ~= 50.30 (~level).
/// statsmodels `ARIMA(order=(1,0,0))` reference on the same series: ar.L1 = 0.364, forecast = 50.30.
#[test]
fn falsify_arima_ar_centering_nonzero_mean_ar1() {
    // Deterministic stationary AR(1), mu ~= 50, phi = 0.5 (statsmodels seed 7).
    let series = [
        50.0, 49.5341, 49.7999, 50.3074, 49.3648, 49.6845, 49.8413, 48.1659, 50.1006, 50.6508,
        49.7, 49.6784, 50.3445, 49.9109, 49.7127, 48.4031, 49.7561, 50.0019, 50.2754, 48.6112,
        50.9563, 50.6325, 49.9291, 51.9936, 50.9514, 49.025, 49.1073, 47.2653, 49.6821, 49.4246,
        48.9697, 50.5573, 48.6276, 49.8492, 47.8602, 48.2679, 47.9297, 50.4269, 51.9796, 50.6604,
        51.1709, 50.4055, 50.7708, 49.6326, 48.1079, 47.2509, 49.0086, 51.7519, 51.1453, 50.0481,
        51.9361, 51.2053, 50.7041, 50.6046, 50.1699, 49.7755, 48.4528, 49.728, 49.7692, 51.0777,
    ];
    let data = Vector::from_slice(&series);

    let n = series.len() as f64;
    let mean: f64 = series.iter().sum::<f64>() / n;
    let range = series.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - series.iter().copied().fold(f64::INFINITY, f64::min);

    let mut arima = ARIMA::new(1, 0, 0);
    arima.fit(&data).expect("fit");

    // The estimated AR(1) coefficient must be the true (centered) phi, NOT ~1.0.
    let phi = arima.ar_coefficients().expect("ar coef present").as_slice()[0];
    assert!(
        (phi - 0.37).abs() < 0.05,
        "FALSIFIED ARIMA-AR-CENTERING: AR(1) coef {phi} (expected demeaned phi_hat ~0.37, statsmodels 0.364). \
         coef ~= 1.0 means AR was estimated on UNCENTERED levels."
    );
    assert!(
        phi < 0.9,
        "FALSIFIED ARIMA-AR-CENTERING: AR(1) coef {phi} collapsed toward 1.0 — AR estimated on uncentered data."
    );

    // The 1-step forecast must sit near the series level, NOT ~2x it.
    let fc = arima.forecast(1).expect("forecast")[0];
    assert!(
        (fc - mean).abs() < 0.5 * range,
        "FALSIFIED ARIMA-AR-CENTERING: 1-step forecast {fc} diverged from level (mean {mean}, range {range}); \
         band |fc - mean| < 0.5*range = {}. Pre-fix forecast ~100.93 (~2x level).",
        0.5 * range
    );
    // statsmodels ARIMA(1,0,0) reference forecast on this series is 50.30.
    assert!(
        (fc - 50.30).abs() < 0.5,
        "FALSIFIED ARIMA-AR-CENTERING: 1-step forecast {fc} != statsmodels reference 50.30 (+/-0.5)."
    );
}
