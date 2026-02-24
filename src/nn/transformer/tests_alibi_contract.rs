// =========================================================================
// FALSIFY-AL: alibi-kernel-v1.yaml contract (aprender ALiBi)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had 7 ALiBi unit tests but zero FALSIFY-AL-* tests
//   Why 2: unit tests verify shapes/integration, not mathematical invariants
//   Why 3: no mapping from alibi-kernel-v1.yaml to aprender test names
//   Why 4: aprender predates the provable-contracts YAML convention
//   Why 5: ALiBi was "obviously correct" (simple linear bias formula)
//
// References:
//   - provable-contracts/contracts/alibi-kernel-v1.yaml
//   - Press et al. (2022) "Train Short, Test Long"
// =========================================================================

use super::*;

/// FALSIFY-AL-001: Negative bias — all biases must be <= 0
///
/// Contract: -m_h * |i - j| <= 0 for all i, j, h
#[test]
fn falsify_al_001_negative_bias() {
    for &num_heads in &[1, 2, 4, 8, 16] {
        let alibi = ALiBi::new(num_heads);
        let bias = alibi.compute_bias(10);
        let data = bias.data();

        for (idx, &val) in data.iter().enumerate() {
            assert!(
                val <= 0.0,
                "FALSIFIED AL-001: bias[{idx}] = {val} > 0 (num_heads={num_heads})"
            );
        }
    }
}

/// FALSIFY-AL-002: Slope positivity — all slopes must be > 0
///
/// Contract: m_h = 2^(-8h/H) > 0 for all h
#[test]
fn falsify_al_002_slope_positivity() {
    for &num_heads in &[1, 2, 3, 4, 6, 8, 12, 16, 32] {
        let alibi = ALiBi::new(num_heads);
        for (h, &slope) in alibi.slopes().iter().enumerate() {
            assert!(
                slope > 0.0,
                "FALSIFIED AL-002: slope[{h}] = {slope} not > 0 (num_heads={num_heads})"
            );
        }
    }
}

/// FALSIFY-AL-003: Self-position zero — bias[i][i] = 0 (no penalty for self)
///
/// Contract: -m_h * |i - i| = 0
#[test]
fn falsify_al_003_self_position_zero() {
    let alibi = ALiBi::new(8);
    let seq_len = 10;
    let bias = alibi.compute_bias(seq_len);
    let data = bias.data();

    for h in 0..8 {
        for i in 0..seq_len {
            let idx = h * seq_len * seq_len + i * seq_len + i;
            assert!(
                data[idx].abs() < 1e-7,
                "FALSIFIED AL-003: bias[{h}][{i}][{i}] = {}, expected 0",
                data[idx]
            );
        }
    }
}

/// FALSIFY-AL-004: Head-monotonic slopes — slopes decrease with head index
///
/// For power-of-2 heads: m_0 > m_1 > ... > m_{H-1}
#[test]
fn falsify_al_004_head_monotonic_slopes() {
    for &num_heads in &[2, 4, 8, 16, 32] {
        let alibi = ALiBi::new(num_heads);
        let slopes = alibi.slopes();
        for i in 1..slopes.len() {
            assert!(
                slopes[i] < slopes[i - 1],
                "FALSIFIED AL-004: slope[{i}]={} not < slope[{}]={} (num_heads={num_heads})",
                slopes[i],
                i - 1,
                slopes[i - 1]
            );
        }
    }
}

/// FALSIFY-AL-001b: Bias linearity — bias decreases linearly with distance
///
/// Contract: bias[h][i][j] = -slope_h * |i - j|
#[test]
fn falsify_al_001b_bias_linearity() {
    let alibi = ALiBi::new(4);
    let seq_len = 8;
    let bias = alibi.compute_bias(seq_len);
    let data = bias.data();

    for h in 0..4 {
        let slope = alibi.slopes()[h];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let idx = h * seq_len * seq_len + i * seq_len + j;
                let expected = -slope * (i as f32 - j as f32).abs();
                let diff = (data[idx] - expected).abs();
                assert!(
                    diff < 1e-5,
                    "FALSIFIED AL-001: bias[{h}][{i}][{j}] = {}, expected {expected}",
                    data[idx]
                );
            }
        }
    }
}

mod alibi_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-AL-001-prop: Negative bias for random head counts
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_al_001_prop_negative_bias(
            num_heads in prop::sample::select(vec![1usize, 2, 4, 8, 16, 32]),
            seq_len in 2..=20usize,
        ) {
            let alibi = ALiBi::new(num_heads);
            let bias = alibi.compute_bias(seq_len);
            for (idx, &val) in bias.data().iter().enumerate() {
                prop_assert!(
                    val <= 0.0,
                    "FALSIFIED AL-001-prop: bias[{}]={} > 0 (heads={}, seq={})",
                    idx, val, num_heads, seq_len
                );
            }
        }
    }

    /// FALSIFY-AL-002-prop: Slope positivity for random head counts
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_al_002_prop_slope_positivity(
            num_heads in prop::sample::select(vec![1usize, 2, 3, 4, 6, 8, 12, 16, 32]),
        ) {
            let alibi = ALiBi::new(num_heads);
            for (h, &slope) in alibi.slopes().iter().enumerate() {
                prop_assert!(
                    slope > 0.0,
                    "FALSIFIED AL-002-prop: slope[{}]={} <= 0 (heads={})",
                    h, slope, num_heads
                );
            }
        }
    }
}
