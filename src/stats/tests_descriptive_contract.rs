// =========================================================================
// FALSIFY-DS: descriptive statistics contract (aprender stats)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-DS-* tests for descriptive stats
//   Why 2: descriptive tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no YAML contract for descriptive statistics yet
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: Mean/median/quantile was "obviously correct" (basic arithmetic)
//
// References:
//   - Hyndman & Fan (1996) "Sample Quantiles in Statistical Packages"
// =========================================================================

use super::*;

/// FALSIFY-DS-001: Median of sorted data is correct
#[test]
fn falsify_ds_001_median_correct() {
    let data = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let stats = DescriptiveStats::new(&data);
    let median = stats.quantile(0.5).expect("valid quantile");

    assert!(
        (median - 3.0).abs() < 1e-5,
        "FALSIFIED DS-001: median={median}, expected 3.0"
    );
}

/// FALSIFY-DS-002: Min quantile (0.0) returns minimum
#[test]
fn falsify_ds_002_min_quantile() {
    let data = Vector::from_slice(&[5.0, 1.0, 3.0, 2.0, 4.0]);
    let stats = DescriptiveStats::new(&data);
    let min = stats.quantile(0.0).expect("valid quantile");

    assert!(
        (min - 1.0).abs() < 1e-5,
        "FALSIFIED DS-002: min quantile={min}, expected 1.0"
    );
}

/// FALSIFY-DS-003: Max quantile (1.0) returns maximum
#[test]
fn falsify_ds_003_max_quantile() {
    let data = Vector::from_slice(&[5.0, 1.0, 3.0, 2.0, 4.0]);
    let stats = DescriptiveStats::new(&data);
    let max = stats.quantile(1.0).expect("valid quantile");

    assert!(
        (max - 5.0).abs() < 1e-5,
        "FALSIFIED DS-003: max quantile={max}, expected 5.0"
    );
}

/// FALSIFY-DS-004: Quantile ordering: Q1 <= median <= Q3
#[test]
fn falsify_ds_004_quantile_ordering() {
    let data = Vector::from_slice(&[10.0, 2.0, 8.0, 4.0, 6.0, 1.0, 9.0, 3.0, 7.0, 5.0]);
    let stats = DescriptiveStats::new(&data);

    let q1 = stats.quantile(0.25).expect("Q1");
    let median = stats.quantile(0.50).expect("median");
    let q3 = stats.quantile(0.75).expect("Q3");

    assert!(
        q1 <= median && median <= q3,
        "FALSIFIED DS-004: Q1={q1}, median={median}, Q3={q3} — ordering violated"
    );
}
