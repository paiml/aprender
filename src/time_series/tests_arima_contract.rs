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
