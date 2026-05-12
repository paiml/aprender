// SHIP-TWO-001 — `flash-attention-v1` algorithm-level PARTIAL
// discharge for FALSIFY-FA-001..004 (closes 4/4 sweep).
//
// Contract: `contracts/flash-attention-v1.yaml`.
// Spec: Dao et al. (2022) FlashAttention — IO-aware exact attention
// with online-softmax tiling.

// ===========================================================================
// FA-001 — Equivalence: |FlashAttn(Q,K,V) - StdAttn(Q,K,V)| < 1e-5
// ===========================================================================

pub const AC_FA_001_TOLERANCE: f32 = 1.0e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fa001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_std_attention_equivalence(
    flash_out: &[f32],
    std_out: &[f32],
) -> Fa001Verdict {
    if flash_out.is_empty() || std_out.is_empty() { return Fa001Verdict::Fail; }
    if flash_out.len() != std_out.len() { return Fa001Verdict::Fail; }
    for (&f, &s) in flash_out.iter().zip(std_out.iter()) {
        if !f.is_finite() || !s.is_finite() { return Fa001Verdict::Fail; }
        if (f - s).abs() > AC_FA_001_TOLERANCE { return Fa001Verdict::Fail; }
    }
    Fa001Verdict::Pass
}

// ===========================================================================
// FA-002 — Online softmax matches full softmax (tiled === non-tiled)
// ===========================================================================

pub const AC_FA_002_TOLERANCE: f32 = 1.0e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fa002Verdict { Pass, Fail }

/// Reference numerically-stable softmax over a 1D score vector.
#[must_use]
pub fn softmax(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() { return vec![]; }
    let m = scores.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    if !m.is_finite() { return vec![]; }
    let exps: Vec<f32> = scores.iter().map(|&x| (x - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    if s == 0.0 || !s.is_finite() { return vec![]; }
    exps.iter().map(|&e| e / s).collect()
}

/// Reference online softmax: process scores in tiles of size `tile_size`,
/// maintaining running max + denom across tiles. Should produce the same
/// result as `softmax(scores)` within tolerance.
#[must_use]
pub fn online_softmax(scores: &[f32], tile_size: usize) -> Vec<f32> {
    if scores.is_empty() || tile_size == 0 { return vec![]; }
    let n = scores.len();
    let mut running_max = f32::NEG_INFINITY;
    let mut running_sum = 0.0_f32;
    // First pass: compute running max + sum.
    let mut idx = 0;
    while idx < n {
        let end = (idx + tile_size).min(n);
        let tile = &scores[idx..end];
        let tile_max = tile.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        if !tile_max.is_finite() { return vec![]; }
        let new_max = running_max.max(tile_max);
        if running_max.is_finite() {
            running_sum *= (running_max - new_max).exp();
        } else {
            running_sum = 0.0;
        }
        for &x in tile {
            running_sum += (x - new_max).exp();
        }
        running_max = new_max;
        idx = end;
    }
    // Second pass: emit normalized values.
    if running_sum == 0.0 || !running_sum.is_finite() { return vec![]; }
    scores.iter().map(|&x| (x - running_max).exp() / running_sum).collect()
}

#[must_use]
pub fn verdict_from_online_softmax_match(scores: &[f32], tile_size: usize) -> Fa002Verdict {
    if scores.is_empty() || tile_size == 0 { return Fa002Verdict::Fail; }
    if !scores.iter().all(|v| v.is_finite()) { return Fa002Verdict::Fail; }
    let full = softmax(scores);
    let tiled = online_softmax(scores, tile_size);
    if full.is_empty() || tiled.is_empty() || full.len() != tiled.len() {
        return Fa002Verdict::Fail;
    }
    for (&a, &b) in full.iter().zip(tiled.iter()) {
        if !a.is_finite() || !b.is_finite() { return Fa002Verdict::Fail; }
        if (a - b).abs() > AC_FA_002_TOLERANCE { return Fa002Verdict::Fail; }
    }
    Fa002Verdict::Pass
}

// ===========================================================================
// FA-003 — Weight normalization: each weight row sums to 1.0
// ===========================================================================

pub const AC_FA_003_TOLERANCE: f32 = 1.0e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fa003Verdict { Pass, Fail }

/// Pass iff every row of `weights` (n×m row-major) sums to 1.0 within tolerance.
#[must_use]
pub fn verdict_from_weight_normalization(weights: &[f32], n: usize, m: usize) -> Fa003Verdict {
    if weights.is_empty() || n == 0 || m == 0 { return Fa003Verdict::Fail; }
    if weights.len() != n * m { return Fa003Verdict::Fail; }
    for row in 0..n {
        let mut sum = 0.0_f32;
        for col in 0..m {
            let v = weights[row * m + col];
            if !v.is_finite() { return Fa003Verdict::Fail; }
            sum += v;
        }
        if (sum - 1.0).abs() > AC_FA_003_TOLERANCE { return Fa003Verdict::Fail; }
    }
    Fa003Verdict::Pass
}

// ===========================================================================
// FA-004 — Single tile: when N ≤ tile_size, matches standard attention exactly
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fa004Verdict { Pass, Fail }

/// When `n_seq ≤ tile_size`, online softmax should reduce to a single
/// tile and produce the SAME bits as the full softmax. This catches
/// edge-case regressions in tile-loop bounds.
#[must_use]
pub fn verdict_from_single_tile_exactness(scores: &[f32], tile_size: usize) -> Fa004Verdict {
    if scores.is_empty() || tile_size == 0 { return Fa004Verdict::Fail; }
    if scores.len() > tile_size { return Fa004Verdict::Fail; } // out of single-tile regime
    let full = softmax(scores);
    let tiled = online_softmax(scores, tile_size);
    if full.is_empty() || tiled.is_empty() || full.len() != tiled.len() {
        return Fa004Verdict::Fail;
    }
    // Within a single tile the results should match byte-exactly (same
    // arithmetic, no cross-tile rescaling).
    for (&f, &t) in full.iter().zip(tiled.iter()) {
        if f.to_bits() != t.to_bits() { return Fa004Verdict::Fail; }
    }
    Fa004Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // FA-001 (equivalence)
    #[test] fn fa001_pass_identical() {
        let a = vec![0.1_f32, 0.5, 0.3];
        assert_eq!(verdict_from_std_attention_equivalence(&a, &a), Fa001Verdict::Pass);
    }
    #[test] fn fa001_pass_within_tolerance() {
        let f = vec![1.0_f32];
        let s = vec![1.0_f32 + 5e-6];
        assert_eq!(verdict_from_std_attention_equivalence(&f, &s), Fa001Verdict::Pass);
    }
    #[test] fn fa001_fail_above_tolerance() {
        let f = vec![1.0_f32];
        let s = vec![1.001_f32];
        assert_eq!(verdict_from_std_attention_equivalence(&f, &s), Fa001Verdict::Fail);
    }
    #[test] fn fa001_fail_length() {
        let f = vec![1.0_f32];
        let s = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_std_attention_equivalence(&f, &s), Fa001Verdict::Fail);
    }
    #[test] fn fa001_fail_nan() {
        let f = vec![f32::NAN];
        let s = vec![1.0_f32];
        assert_eq!(verdict_from_std_attention_equivalence(&f, &s), Fa001Verdict::Fail);
    }

    // FA-002 (online softmax)
    #[test] fn fa002_pass_matches_full() {
        let scores = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(verdict_from_online_softmax_match(&scores, 2), Fa002Verdict::Pass);
    }
    #[test] fn fa002_pass_tile_size_1() {
        let scores = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_online_softmax_match(&scores, 1), Fa002Verdict::Pass);
    }
    #[test] fn fa002_pass_tile_size_eq_n() {
        let scores = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_online_softmax_match(&scores, 3), Fa002Verdict::Pass);
    }
    #[test] fn fa002_pass_extreme_values() {
        // Online softmax must remain numerically stable across tiles.
        let scores = vec![100.0_f32, -100.0, 50.0, -50.0];
        assert_eq!(verdict_from_online_softmax_match(&scores, 2), Fa002Verdict::Pass);
    }
    #[test] fn fa002_fail_empty() {
        assert_eq!(verdict_from_online_softmax_match(&[], 2), Fa002Verdict::Fail);
    }
    #[test] fn fa002_fail_zero_tile() {
        let scores = vec![1.0_f32];
        assert_eq!(verdict_from_online_softmax_match(&scores, 0), Fa002Verdict::Fail);
    }
    #[test] fn fa002_fail_nan() {
        let scores = vec![1.0_f32, f32::NAN];
        assert_eq!(verdict_from_online_softmax_match(&scores, 2), Fa002Verdict::Fail);
    }

    // FA-003 (weight normalization)
    #[test] fn fa003_pass_canonical() {
        // 2 rows × 3 cols, each row sums to 1.0
        let w = vec![0.2_f32, 0.5, 0.3, 0.1, 0.4, 0.5];
        assert_eq!(verdict_from_weight_normalization(&w, 2, 3), Fa003Verdict::Pass);
    }
    #[test] fn fa003_fail_undersum() {
        let w = vec![0.2_f32, 0.3, 0.4]; // sums to 0.9
        assert_eq!(verdict_from_weight_normalization(&w, 1, 3), Fa003Verdict::Fail);
    }
    #[test] fn fa003_fail_oversum() {
        let w = vec![0.5_f32, 0.5, 0.5]; // sums to 1.5
        assert_eq!(verdict_from_weight_normalization(&w, 1, 3), Fa003Verdict::Fail);
    }
    #[test] fn fa003_fail_dim_mismatch() {
        let w = vec![0.2_f32, 0.5, 0.3];
        assert_eq!(verdict_from_weight_normalization(&w, 2, 3), Fa003Verdict::Fail);
    }

    // FA-004 (single tile exactness)
    #[test] fn fa004_pass_within_single_tile() {
        let scores = vec![1.0_f32, 2.0, 3.0]; // n=3 ≤ tile_size=8
        assert_eq!(verdict_from_single_tile_exactness(&scores, 8), Fa004Verdict::Pass);
    }
    #[test] fn fa004_pass_n_eq_tile() {
        let scores = vec![1.0_f32, 2.0]; // n=2 == tile_size=2
        assert_eq!(verdict_from_single_tile_exactness(&scores, 2), Fa004Verdict::Pass);
    }
    #[test] fn fa004_fail_n_above_tile() {
        // The exactness predicate only applies when n ≤ tile_size; out of
        // domain → Fail.
        let scores = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_single_tile_exactness(&scores, 2), Fa004Verdict::Fail);
    }
    #[test] fn fa004_fail_zero_tile() {
        let scores = vec![1.0_f32];
        assert_eq!(verdict_from_single_tile_exactness(&scores, 0), Fa004Verdict::Fail);
    }

    // Helper sanity
    #[test] fn softmax_uniform_input() {
        let s = softmax(&[1.0_f32, 1.0, 1.0]);
        for &v in &s {
            assert!((v - 1.0 / 3.0).abs() < 1e-6);
        }
    }
    #[test] fn online_softmax_matches_full_at_canonical() {
        let scores = vec![0.5_f32, 1.0, 1.5, 2.0];
        let full = softmax(&scores);
        let tiled = online_softmax(&scores, 2);
        for (&a, &b) in full.iter().zip(tiled.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_FA_001_TOLERANCE - 1e-5).abs() < 1e-12);
        assert!((AC_FA_002_TOLERANCE - 1e-5).abs() < 1e-12);
        assert!((AC_FA_003_TOLERANCE - 1e-5).abs() < 1e-12);
    }
}
