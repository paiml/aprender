// =========================================================================
// FALSIFY-MCL: metrics-clustering-v1.yaml contract (aprender clustering metrics)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-MCL-* tests for clustering metrics
//   Why 2: metric tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no mapping from metrics-clustering-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: Silhouette was "obviously correct" (standard formula)
//
// References:
//   - provable-contracts/contracts/metrics-clustering-v1.yaml
//   - Rousseeuw (1987) "Silhouettes: a graphical aid"
// =========================================================================

use super::*;
use crate::primitives::Matrix;

/// FALSIFY-MCL-001: Silhouette score ∈ [-1, 1]
#[test]
fn falsify_mcl_001_silhouette_bounded() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 10.0, 10.0, 10.1, 10.1, 10.2, 10.2,
        ],
    )
    .expect("valid");
    let labels = vec![0_usize, 0, 0, 1, 1, 1];

    let score = silhouette_score(&data, &labels);
    assert!(
        (-1.0..=1.0).contains(&score),
        "FALSIFIED MCL-001: silhouette_score={score} not in [-1, 1]"
    );
}

/// FALSIFY-MCL-002: Well-separated clusters → high silhouette
#[test]
fn falsify_mcl_002_high_silhouette_for_separated() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, 0.01, 0.01, 0.02, 0.02, 100.0, 100.0, 100.01, 100.01, 100.02, 100.02,
        ],
    )
    .expect("valid");
    let labels = vec![0_usize, 0, 0, 1, 1, 1];

    let score = silhouette_score(&data, &labels);
    assert!(
        score > 0.9,
        "FALSIFIED MCL-002: silhouette={score} <= 0.9 for well-separated clusters"
    );
}

/// FALSIFY-MCL-003: Silhouette is deterministic
#[test]
fn falsify_mcl_003_silhouette_deterministic() {
    let data = Matrix::from_vec(4, 2, vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0]).expect("valid");
    let labels = vec![0_usize, 0, 1, 1];

    let s1 = silhouette_score(&data, &labels);
    let s2 = silhouette_score(&data, &labels);
    assert_eq!(
        s1, s2,
        "FALSIFIED MCL-003: silhouette scores differ on same input"
    );
}

mod mcl_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-MCL-001-prop: Silhouette score in [-1, 1] for random data
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn falsify_mcl_001_prop_silhouette_bounded(
            seed in 0..200u32,
        ) {
            let n = 6;
            let data: Vec<f32> = (0..n * 2)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                .collect();
            let matrix = Matrix::from_vec(n, 2, data).expect("valid");
            let labels: Vec<usize> = (0..n).map(|i| i % 2).collect();

            let score = silhouette_score(&matrix, &labels);
            prop_assert!(
                (-1.0..=1.0001).contains(&score),
                "FALSIFIED MCL-001-prop: silhouette={} not in [-1,1]",
                score
            );
        }
    }

    /// FALSIFY-MCL-003-prop: Silhouette is deterministic
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn falsify_mcl_003_prop_deterministic(
            seed in 0..200u32,
        ) {
            let n = 6;
            let data: Vec<f32> = (0..n * 2)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                .collect();
            let matrix = Matrix::from_vec(n, 2, data).expect("valid");
            let labels: Vec<usize> = (0..n).map(|i| i % 2).collect();

            let s1 = silhouette_score(&matrix, &labels);
            let s2 = silhouette_score(&matrix, &labels);
            prop_assert_eq!(
                s1.to_bits(), s2.to_bits(),
                "FALSIFIED MCL-003-prop: silhouette differs on same input"
            );
        }
    }
}
