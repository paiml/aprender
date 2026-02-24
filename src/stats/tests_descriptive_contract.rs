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

mod ds_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-DS-004-prop: Quantile ordering Q1 <= median <= Q3 for random data
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn falsify_ds_004_prop_quantile_ordering(
            seed in 0..1000u32,
            n in 5..=30usize,
        ) {
            let data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 100.0)
                .collect();
            let v = Vector::from_vec(data);
            let stats = DescriptiveStats::new(&v);

            let q1 = stats.quantile(0.25).expect("Q1");
            let median = stats.quantile(0.50).expect("median");
            let q3 = stats.quantile(0.75).expect("Q3");

            prop_assert!(
                q1 <= median + 1e-5 && median <= q3 + 1e-5,
                "FALSIFIED DS-004-prop: Q1={}, med={}, Q3={} — ordering violated (n={}, seed={})",
                q1, median, q3, n, seed
            );
        }
    }

    /// FALSIFY-DS-001-prop: Median in [min, max] for random data
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn falsify_ds_001_prop_median_bounded(
            seed in 0..1000u32,
            n in 3..=30usize,
        ) {
            let data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 50.0)
                .collect();
            let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            let v = Vector::from_vec(data);
            let stats = DescriptiveStats::new(&v);
            let median = stats.quantile(0.5).expect("median");

            prop_assert!(
                median >= min_val - 1e-5 && median <= max_val + 1e-5,
                "FALSIFIED DS-001-prop: median={} not in [{}, {}]",
                median, min_val, max_val
            );
        }
    }
}
