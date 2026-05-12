// SHIP-TWO-001 — `sliding-window-attention-v1` algorithm-level PARTIAL
// discharge for FALSIFY-SWA-001..007 (closes 7/7 sweep).
//
// Contract: `contracts/sliding-window-attention-v1.yaml`.

// ===========================================================================
// Reference sliding-window mask (causal and non-causal variants)
// ===========================================================================

/// Non-causal symmetric window: mask[i][j] = 1 iff |i - j| < W.
#[must_use]
pub fn swa_mask_symmetric(seq_len: usize, w: usize) -> Vec<Vec<u8>> {
    let mut m = vec![vec![0_u8; seq_len]; seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let d = i.abs_diff(j);
            if d < w { m[i][j] = 1; }
        }
    }
    m
}

/// Causal sliding window: mask[i][j] = 1 iff j <= i AND i - j < W.
#[must_use]
pub fn swa_mask_causal(seq_len: usize, w: usize) -> Vec<Vec<u8>> {
    let mut m = vec![vec![0_u8; seq_len]; seq_len];
    for i in 0..seq_len {
        for j in 0..=i {
            if i - j < w { m[i][j] = 1; }
        }
    }
    m
}

/// Apply mask to logits; -inf for masked positions, raw value otherwise.
/// Then softmax row-wise. Returns Some(rows) or None on empty/dim mismatch.
#[must_use]
pub fn windowed_softmax(logits: &[Vec<f32>], mask: &[Vec<u8>]) -> Option<Vec<Vec<f64>>> {
    if logits.is_empty() || logits.len() != mask.len() { return None; }
    let n = logits.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if logits[i].len() != n || mask[i].len() != n { return None; }
        let mut max = f32::NEG_INFINITY;
        for j in 0..n { if mask[i][j] == 1 && logits[i][j] > max { max = logits[i][j]; } }
        if !max.is_finite() { return None; }
        let mut sum = 0.0_f64;
        let mut exps = vec![0.0_f64; n];
        for j in 0..n {
            if mask[i][j] == 1 {
                let e = ((logits[i][j] - max) as f64).exp();
                exps[j] = e;
                sum += e;
            }
        }
        if sum == 0.0 { return None; }
        let row: Vec<f64> = exps.iter().map(|e| e / sum).collect();
        out.push(row);
    }
    Some(out)
}

// ===========================================================================
// SWA-001 — Symmetry: mask(i,j) == mask(j,i) for non-causal window
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Swa001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_window_symmetry(seq_len: usize, w: usize) -> Swa001Verdict {
    if seq_len == 0 || w == 0 { return Swa001Verdict::Fail; }
    let m = swa_mask_symmetric(seq_len, w);
    for i in 0..seq_len {
        for j in 0..seq_len {
            if m[i][j] != m[j][i] { return Swa001Verdict::Fail; }
        }
    }
    Swa001Verdict::Pass
}

// ===========================================================================
// SWA-002 — Causal masking: mask(i,j) == 0 for j > i
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Swa002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_causal_constraint(seq_len: usize, w: usize) -> Swa002Verdict {
    if seq_len == 0 || w == 0 { return Swa002Verdict::Fail; }
    let m = swa_mask_causal(seq_len, w);
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            if m[i][j] != 0 { return Swa002Verdict::Fail; }
        }
    }
    Swa002Verdict::Pass
}

// ===========================================================================
// SWA-003 — Effective context: ctx(i) = min(i+1, W) for causal mask
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Swa003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_effective_context(seq_len: usize, w: usize) -> Swa003Verdict {
    if seq_len == 0 || w == 0 { return Swa003Verdict::Fail; }
    let m = swa_mask_causal(seq_len, w);
    for i in 0..seq_len {
        let count: usize = m[i].iter().map(|x| *x as usize).sum();
        let expected = (i + 1).min(w);
        if count != expected { return Swa003Verdict::Fail; }
    }
    Swa003Verdict::Pass
}

// ===========================================================================
// SWA-004 — Dense degeneration: W >= seq_len ⇒ full causal mask
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Swa004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_dense_degeneration(seq_len: usize) -> Swa004Verdict {
    if seq_len == 0 { return Swa004Verdict::Fail; }
    let m = swa_mask_causal(seq_len, seq_len);
    for i in 0..seq_len {
        for j in 0..seq_len {
            let expected: u8 = u8::from(j <= i);
            if m[i][j] != expected { return Swa004Verdict::Fail; }
        }
    }
    Swa004Verdict::Pass
}

// ===========================================================================
// SWA-005 — Multi-layer receptive field: 1 + L*(W-1)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Swa005Verdict { Pass, Fail }

#[must_use]
pub const fn receptive_field(num_layers: u32, w: u32) -> u32 {
    if w == 0 || num_layers == 0 { return 0; }
    1 + num_layers * (w - 1)
}

#[must_use]
pub const fn verdict_from_receptive_field(
    num_layers: u32,
    w: u32,
    observed: u32,
) -> Swa005Verdict {
    if num_layers == 0 || w == 0 { return Swa005Verdict::Fail; }
    if receptive_field(num_layers, w) == observed { Swa005Verdict::Pass } else { Swa005Verdict::Fail }
}

// ===========================================================================
// SWA-006 — Windowed softmax sums to 1
// ===========================================================================

pub const AC_SWA_006_TOLERANCE: f64 = 1e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Swa006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_windowed_softmax_normalized(
    logits: &[Vec<f32>],
    mask: &[Vec<u8>],
) -> Swa006Verdict {
    let probs = match windowed_softmax(logits, mask) { Some(v) => v, None => return Swa006Verdict::Fail };
    for row in probs {
        let s: f64 = row.iter().sum();
        if (s - 1.0).abs() > AC_SWA_006_TOLERANCE { return Swa006Verdict::Fail; }
    }
    Swa006Verdict::Pass
}

// ===========================================================================
// SWA-007 — Attention count bounded: count(mask(i,:) == 1) <= W
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Swa007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_count_bound(seq_len: usize, w: usize) -> Swa007Verdict {
    if seq_len == 0 || w == 0 { return Swa007Verdict::Fail; }
    let m = swa_mask_causal(seq_len, w);
    for row in m {
        let count: usize = row.iter().map(|x| *x as usize).sum();
        if count > w { return Swa007Verdict::Fail; }
    }
    Swa007Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference impl spot checks
    #[test] fn ref_symmetric_w3() {
        let m = swa_mask_symmetric(5, 3);
        // Expected: mask[i][j] = 1 iff |i-j| < 3.
        for i in 0..5 {
            for j in 0..5 {
                let d = (i as i32 - j as i32).unsigned_abs() as usize;
                let expected: u8 = u8::from(d < 3);
                assert_eq!(m[i][j], expected);
            }
        }
    }

    #[test] fn ref_causal_w3_seq4() {
        let m = swa_mask_causal(4, 3);
        // Row 0: only [0]; row 1: [0, 1]; row 2: [0, 1, 2]; row 3: [1, 2, 3].
        assert_eq!(m[0], vec![1, 0, 0, 0]);
        assert_eq!(m[1], vec![1, 1, 0, 0]);
        assert_eq!(m[2], vec![1, 1, 1, 0]);
        assert_eq!(m[3], vec![0, 1, 1, 1]);
    }

    // SWA-001
    #[test] fn swa001_pass_w3() { assert_eq!(verdict_from_window_symmetry(8, 3), Swa001Verdict::Pass); }
    #[test] fn swa001_pass_w_eq_seq() { assert_eq!(verdict_from_window_symmetry(8, 8), Swa001Verdict::Pass); }
    #[test] fn swa001_fail_zero_w() { assert_eq!(verdict_from_window_symmetry(8, 0), Swa001Verdict::Fail); }
    #[test] fn swa001_fail_zero_seq() { assert_eq!(verdict_from_window_symmetry(0, 3), Swa001Verdict::Fail); }

    // SWA-002
    #[test] fn swa002_pass_w3() { assert_eq!(verdict_from_causal_constraint(8, 3), Swa002Verdict::Pass); }
    #[test] fn swa002_pass_dense() { assert_eq!(verdict_from_causal_constraint(8, 8), Swa002Verdict::Pass); }

    // SWA-003
    #[test] fn swa003_pass_canonical() { assert_eq!(verdict_from_effective_context(10, 4), Swa003Verdict::Pass); }
    #[test] fn swa003_pass_w_gt_seq() { assert_eq!(verdict_from_effective_context(5, 100), Swa003Verdict::Pass); }

    // SWA-004
    #[test] fn swa004_pass_seq8() { assert_eq!(verdict_from_dense_degeneration(8), Swa004Verdict::Pass); }
    #[test] fn swa004_pass_seq1() { assert_eq!(verdict_from_dense_degeneration(1), Swa004Verdict::Pass); }
    #[test] fn swa004_fail_zero() { assert_eq!(verdict_from_dense_degeneration(0), Swa004Verdict::Fail); }

    // SWA-005
    #[test] fn swa005_pass_canonical() {
        // L=4, W=8 → 1 + 4*7 = 29.
        assert_eq!(verdict_from_receptive_field(4, 8, 29), Swa005Verdict::Pass);
    }
    #[test] fn swa005_pass_l1() {
        // L=1 → just W.
        assert_eq!(verdict_from_receptive_field(1, 8, 8), Swa005Verdict::Pass);
    }
    #[test] fn swa005_fail_off_by_one() {
        assert_eq!(verdict_from_receptive_field(4, 8, 30), Swa005Verdict::Fail);
    }
    #[test] fn swa005_fail_zero_layers() {
        assert_eq!(verdict_from_receptive_field(0, 8, 1), Swa005Verdict::Fail);
    }

    // SWA-006
    #[test] fn swa006_pass_uniform_logits() {
        let n = 4;
        let logits = vec![vec![1.0_f32; n]; n];
        let mask = swa_mask_causal(n, 3);
        assert_eq!(verdict_from_windowed_softmax_normalized(&logits, &mask), Swa006Verdict::Pass);
    }
    #[test] fn swa006_pass_random_like() {
        let n = 6;
        let logits: Vec<Vec<f32>> = (0..n).map(|i| {
            (0..n).map(|j| ((i + j) as f32) * 0.3).collect()
        }).collect();
        let mask = swa_mask_causal(n, 3);
        assert_eq!(verdict_from_windowed_softmax_normalized(&logits, &mask), Swa006Verdict::Pass);
    }
    #[test] fn swa006_fail_dim_mismatch() {
        let logits = vec![vec![1.0_f32, 2.0]];
        let mask = vec![vec![1_u8, 1, 1]];
        assert_eq!(verdict_from_windowed_softmax_normalized(&logits, &mask), Swa006Verdict::Fail);
    }

    // SWA-007
    #[test] fn swa007_pass_w3_seq8() { assert_eq!(verdict_from_count_bound(8, 3), Swa007Verdict::Pass); }
    #[test] fn swa007_pass_w_gt_seq() { assert_eq!(verdict_from_count_bound(8, 32), Swa007Verdict::Pass); }
    #[test] fn swa007_fail_zero_w() { assert_eq!(verdict_from_count_bound(8, 0), Swa007Verdict::Fail); }

    // Provenance pin
    #[test] fn provenance_tolerance() {
        assert!((AC_SWA_006_TOLERANCE - 1e-9).abs() < 1e-15);
    }

    // Receptive-field helper standalone test
    #[test] fn receptive_field_canonical() {
        assert_eq!(receptive_field(1, 8), 8);
        assert_eq!(receptive_field(2, 8), 15);
        assert_eq!(receptive_field(4, 8), 29);
        assert_eq!(receptive_field(0, 8), 0);
        assert_eq!(receptive_field(4, 0), 0);
    }
}
