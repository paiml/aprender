// =========================================================================
// FALSIFY-REXT: rope-extrapolation-v1.yaml contract (aprender RoPE freqs)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had FALSIFY-RO-* tests but zero FALSIFY-REXT-* tests
//   Why 2: RO tests verify apply() output, not base frequency invariants
//   Why 3: no mapping from rope-extrapolation-v1.yaml to aprender test names
//   Why 4: rope-extrapolation-v1 was created after RO to isolate frequency claims
//   Why 5: frequency computation was "obviously correct" (1/θ^(2i/d))
//
// References:
//   - provable-contracts/contracts/rope-extrapolation-v1.yaml
//   - Su et al. (2021) "RoFormer: Enhanced Transformer with Rotary Position Embedding"
// =========================================================================

use super::*;

/// FALSIFY-REXT-001: Base frequencies positive and decreasing
///
/// freq_i = 1 / base^(2i/d) must be positive and strictly decreasing in i.
#[test]
fn falsify_rext_001_frequencies_positive_decreasing() {
    for &(head_dim, base) in &[
        (8, 10000.0f32),
        (32, 10000.0),
        (64, 10000.0),
        (128, 1_000_000.0),
    ] {
        let half_dim = head_dim / 2;
        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        // All positive
        for (i, &f) in freqs.iter().enumerate() {
            assert!(
                f > 0.0,
                "FALSIFIED REXT-001: freq[{i}] = {f} <= 0 (head_dim={head_dim}, base={base})"
            );
        }

        // Strictly decreasing
        for i in 1..freqs.len() {
            assert!(
                freqs[i] < freqs[i - 1],
                "FALSIFIED REXT-001: freq[{i}]={} not < freq[{}]={} (not decreasing)",
                freqs[i],
                i - 1,
                freqs[i - 1]
            );
        }
    }
}

/// FALSIFY-REXT-002: freq_0 = 1.0 — first frequency is always 1
///
/// At i=0: freq_0 = 1 / base^(0/d) = 1 / 1 = 1.0
#[test]
fn falsify_rext_002_freq_zero_is_one() {
    for &base in &[10_000.0f32, 500_000.0, 1_000_000.0] {
        for &head_dim in &[4, 8, 32, 64, 128] {
            let freq_0 = 1.0 / base.powf(0.0 / head_dim as f32);
            assert!(
                (freq_0 - 1.0).abs() < 1e-6,
                "FALSIFIED REXT-002: freq_0 = {freq_0} != 1.0 (base={base}, d={head_dim})"
            );
        }
    }
}

/// FALSIFY-REXT-003: Higher base → lower frequencies (for i > 0)
///
/// Increasing theta lowers all non-zero frequencies, enabling longer context.
#[test]
fn falsify_rext_003_higher_base_lower_freq() {
    let head_dim = 64;
    let half_dim = head_dim / 2;
    let base_low = 10_000.0f32;
    let base_high = 1_000_000.0f32;

    let freqs_low: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / base_low.powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let freqs_high: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / base_high.powf(2.0 * i as f32 / head_dim as f32))
        .collect();

    // For i > 0, higher base must give lower frequencies
    for i in 1..half_dim {
        assert!(
            freqs_high[i] < freqs_low[i],
            "FALSIFIED REXT-003: freq_high[{i}]={} not < freq_low[{i}]={} (base {base_high} vs {base_low})",
            freqs_high[i],
            freqs_low[i]
        );
    }
}

/// FALSIFY-REXT-004: Rotation at position 0 is identity
///
/// R(0, i) = I: cos(0) = 1, sin(0) = 0 → rotation is identity.
#[test]
fn falsify_rext_004_position_zero_identity() {
    let rope = RotaryPositionEmbedding::new(8, 100);

    let half_dim = 4;
    // cos_cache at pos=0 should be all 1.0
    for i in 0..half_dim {
        assert!(
            (rope.cos_cache[i] - 1.0).abs() < 1e-6,
            "FALSIFIED REXT-004: cos_cache[0][{i}] = {} != 1.0",
            rope.cos_cache[i]
        );
    }
    // sin_cache at pos=0 should be all 0.0
    for i in 0..half_dim {
        assert!(
            rope.sin_cache[i].abs() < 1e-6,
            "FALSIFIED REXT-004: sin_cache[0][{i}] = {} != 0.0",
            rope.sin_cache[i]
        );
    }
}

/// FALSIFY-REXT-005: with_base changes behavior vs default base
///
/// RoPE with base=10000 must produce different output than base=1000000.
#[test]
fn falsify_rext_005_different_base_different_output() {
    let rope_default = RotaryPositionEmbedding::new(8, 100);
    let rope_custom = RotaryPositionEmbedding::with_base(8, 100, 1_000_000.0);

    let x = Tensor::new(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[1, 1, 1, 8]);

    let out_default = rope_default.apply(&x, &[5]); // position 5
    let out_custom = rope_custom.apply(&x, &[5]);

    let diff: f32 = out_default
        .data()
        .iter()
        .zip(out_custom.data().iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum();

    assert!(
        diff > 1e-4,
        "FALSIFIED REXT-005: different bases produce same output at pos=5 (diff={diff})"
    );
}
