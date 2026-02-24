// =========================================================================
// FALSIFY-RP: rope-kernel-v1.yaml contract (aprender RoPE)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had 8+ RoPE unit tests but zero FALSIFY-RP-* tests
//   Why 2: unit tests verify shapes/cache, not mathematical invariants
//   Why 3: no mapping from rope-kernel-v1.yaml to aprender test names
//   Why 4: aprender predates the provable-contracts YAML convention
//   Why 5: RoPE math was "obviously correct" (cos/sin rotation)
//
// References:
//   - provable-contracts/contracts/rope-kernel-v1.yaml
//   - Su et al. (2021) "RoFormer: Enhanced Transformer with Rotary Position Embedding"
// =========================================================================

use super::*;

/// FALSIFY-RP-001: Norm preservation — ‖RoPE(x, m)‖ ≈ ‖x‖
///
/// Contract: RoPE applies 2D rotations to pairs, which preserve norm.
#[test]
fn falsify_rp_001_norm_preservation() {
    let rope = RotaryPositionEmbedding::new(8, 128);

    let data: Vec<f32> = (0..8).map(|i| (i as f32 * 0.37).sin() + 0.5).collect();
    let x = Tensor::new(&data, &[1, 1, 1, 8]); // [batch=1, seq=1, heads=1, dim=8]
    let input_norm: f32 = data.iter().map(|v| v * v).sum::<f32>().sqrt();

    for pos in [0, 1, 10, 50, 100] {
        let y = rope.apply(&x, &[pos]);
        let output_norm: f32 = y.data().iter().map(|v| v * v).sum::<f32>().sqrt();

        let diff = (output_norm - input_norm).abs();
        assert!(
            diff < 1e-4,
            "FALSIFIED RP-001: ‖RoPE(x, {pos})‖ = {output_norm}, ‖x‖ = {input_norm}, diff = {diff}"
        );
    }
}

/// FALSIFY-RP-002: Relative position — dot(RoPE(q,m), RoPE(k,n)) depends only on q,k,m-n
///
/// Contract: ⟨RoPE(q,m), RoPE(k,n)⟩ = ⟨RoPE(q,0), RoPE(k,n-m)⟩
#[test]
fn falsify_rp_002_relative_position() {
    let rope = RotaryPositionEmbedding::new(8, 128);

    let q_data: Vec<f32> = (0..8).map(|i| (i as f32 * 0.37).sin()).collect();
    let k_data: Vec<f32> = (0..8).map(|i| (i as f32 * 0.73).cos()).collect();

    let q = Tensor::new(&q_data, &[1, 1, 1, 8]);
    let k = Tensor::new(&k_data, &[1, 1, 1, 8]);

    // Test several (m, n) pairs with same relative offset
    let offsets = [(10, 15), (20, 25), (50, 55)]; // All have n-m = 5

    let mut dots = Vec::new();
    for &(m, n) in &offsets {
        let q_rot = rope.apply(&q, &[m]);
        let k_rot = rope.apply(&k, &[n]);

        let dot: f32 = q_rot
            .data()
            .iter()
            .zip(k_rot.data().iter())
            .map(|(&a, &b)| a * b)
            .sum();
        dots.push(dot);
    }

    // All dots should be approximately equal (same relative offset)
    for i in 1..dots.len() {
        let diff = (dots[i] - dots[0]).abs();
        assert!(
            diff < 1e-3,
            "FALSIFIED RP-002: dot products for same relative offset differ: {:?}",
            dots
        );
    }
}

/// FALSIFY-RP-004: Zero position — RoPE(x, 0) = x (identity at position 0)
///
/// Contract: At position 0, all angles are 0, so cos=1, sin=0 → identity
#[test]
fn falsify_rp_004_zero_position_identity() {
    let rope = RotaryPositionEmbedding::new(8, 128);

    let data: Vec<f32> = vec![1.0, -2.0, 3.0, -0.5, 4.0, -1.0, 2.5, -3.0];
    let x = Tensor::new(&data, &[1, 1, 1, 8]);
    let y = rope.apply(&x, &[0]);

    for (i, (&orig, &rotated)) in data.iter().zip(y.data().iter()).enumerate() {
        let diff = (orig - rotated).abs();
        assert!(
            diff < 1e-5,
            "FALSIFIED RP-004: RoPE(x, 0)[{i}] = {rotated}, expected {orig} (diff = {diff})"
        );
    }
}

/// FALSIFY-RP-001b: Norm preservation with multi-head input
#[test]
fn falsify_rp_001_multihead_norm_preservation() {
    let rope = RotaryPositionEmbedding::new(8, 128);

    // [batch=1, seq=1, heads=4, dim=8]
    let data: Vec<f32> = (0..32).map(|i| (i as f32 * 0.23).sin()).collect();
    let x = Tensor::new(&data, &[1, 1, 4, 8]);

    let y = rope.apply(&x, &[42]);

    // Check norm per head
    for h in 0..4 {
        let start = h * 8;
        let end = start + 8;
        let input_norm: f32 = data[start..end].iter().map(|v| v * v).sum::<f32>().sqrt();
        let output_norm: f32 = y.data()[start..end]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();

        let diff = (output_norm - input_norm).abs();
        assert!(
            diff < 1e-4,
            "FALSIFIED RP-001: head {h} norm changed: {input_norm} → {output_norm}"
        );
    }
}
