// =========================================================================
// FALSIFY-SE: shannon-entropy-v1.yaml contract (aprender spectral_entropy)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had spectral_entropy tests but zero FALSIFY-SE-* tests
//   Why 2: unit tests verify integration, not mathematical invariants
//   Why 3: no mapping from shannon-entropy-v1.yaml to aprender test names
//   Why 4: aprender predates the provable-contracts YAML convention
//   Why 5: entropy was "obviously correct" (standard -Σ p log p)
//
// References:
//   - provable-contracts/contracts/shannon-entropy-v1.yaml
//   - Shannon (1948) "A Mathematical Theory of Communication"
// =========================================================================

use super::isolation::spectral_entropy;

/// FALSIFY-SE-001: Range bound — normalized entropy ∈ [0, 1]
///
/// spectral_entropy normalizes by max_entropy = ln(N), so output ∈ [0, 1].
#[test]
fn falsify_se_001_range_bound() {
    let test_cases: Vec<Vec<f32>> = vec![
        vec![1.0, 1.0, 1.0, 1.0],         // uniform → max entropy
        vec![100.0, 0.001, 0.001, 0.001], // peaked → low entropy
        vec![1.0, 2.0, 3.0, 4.0, 5.0],    // varying
        vec![0.1; 100],                   // large uniform
        vec![1e-6, 1.0, 1e-6],            // near-singular
    ];

    for (i, mags) in test_cases.iter().enumerate() {
        let h = spectral_entropy(mags);
        assert!(
            (0.0..=1.001).contains(&h),
            "FALSIFIED SE-001 case {i}: entropy = {h}, expected ∈ [0, 1]"
        );
    }
}

/// FALSIFY-SE-002: Zero entropy — constant input yields max entropy (uniform)
///
/// All-equal magnitudes form a uniform distribution → entropy = ln(N)/ln(N) = 1.
#[test]
fn falsify_se_002_uniform_max_entropy() {
    for &n in &[2_usize, 4, 8, 16, 100] {
        let uniform = vec![1.0; n];
        let h = spectral_entropy(&uniform);
        assert!(
            (h - 1.0).abs() < 1e-4,
            "FALSIFIED SE-002: uniform({n}) entropy = {h}, expected ≈ 1.0"
        );
    }
}

/// FALSIFY-SE-002b: Zero/empty input → entropy = 0
#[test]
fn falsify_se_002b_zero_input() {
    assert!((spectral_entropy(&[]) - 0.0).abs() < 1e-6, "empty → 0");
    assert!(
        (spectral_entropy(&[0.0, 0.0, 0.0]) - 0.0).abs() < 1e-6,
        "all-zero → 0"
    );
}

/// FALSIFY-SE-003: Monotonicity — peaked distribution has lower entropy than uniform
///
/// A distribution concentrated on one bin has entropy < uniform entropy.
#[test]
fn falsify_se_003_peaked_lower_than_uniform() {
    let n = 8;
    let uniform = vec![1.0; n];
    let h_uniform = spectral_entropy(&uniform);

    // Peaked: one bin dominates
    let mut peaked = vec![0.01; n];
    peaked[0] = 100.0;
    let h_peaked = spectral_entropy(&peaked);

    assert!(
        h_peaked < h_uniform,
        "FALSIFIED SE-003: peaked entropy ({h_peaked}) not < uniform entropy ({h_uniform})"
    );
}

mod se_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-SE-001-prop: Range bound [0, 1] for random magnitudes
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_se_001_prop_range_bound(
            seed in 0..500u32,
        ) {
            let n = (seed % 20 + 2) as usize;
            let mags: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin().abs() * 10.0 + 0.01)
                .collect();

            let h = spectral_entropy(&mags);
            prop_assert!(
                (0.0..=1.001).contains(&h),
                "FALSIFIED SE-001-prop: entropy={} not in [0,1] for n={} seed={}",
                h, n, seed
            );
        }
    }

    /// FALSIFY-SE-003-prop: Peaked < uniform for random sizes
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        #[test]
        fn falsify_se_003_prop_peaked_lower(
            n in 3..=30usize,
        ) {
            let uniform = vec![1.0f32; n];
            let h_uniform = spectral_entropy(&uniform);

            let mut peaked = vec![0.01f32; n];
            peaked[0] = 100.0;
            let h_peaked = spectral_entropy(&peaked);

            prop_assert!(
                h_peaked < h_uniform,
                "FALSIFIED SE-003-prop: peaked({}) >= uniform({}) for n={}",
                h_peaked, h_uniform, n
            );
        }
    }
}
