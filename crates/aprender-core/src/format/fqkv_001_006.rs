// SHIP-TWO-001 — `fused-qkv-projection-v1` algorithm-level PARTIAL
// discharge for FALSIFY-FQKV-001..006 (closes 6/6 sweep).
//
// Contract: `contracts/fused-qkv-projection-v1.yaml`.
// Spec: Fused QKV projection — concatenated weight matrix for single
// matvec attention projection (Vaswani 2017 + Whisper decoder; PMAT-054A
// shared Q8_1 quantization).

// ===========================================================================
// FQKV-001 — Fused matches separate Q+K+V projections within 1e-6
// ===========================================================================

pub const AC_FQKV_001_TOLERANCE: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fqkv001Verdict { Pass, Fail }

/// `fused_qkv` is a concatenated [q; k; v] buffer of length 3*d_model.
/// `q`, `k`, `v` are the separately-computed reference projections each
/// of length d_model. Verdict checks element-wise equivalence.
#[must_use]
pub fn verdict_from_fused_separate_equivalence(
    fused_qkv: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> Fqkv001Verdict {
    if q.is_empty() || k.is_empty() || v.is_empty() { return Fqkv001Verdict::Fail; }
    if q.len() != k.len() || k.len() != v.len() { return Fqkv001Verdict::Fail; }
    let d = q.len();
    if fused_qkv.len() != 3 * d { return Fqkv001Verdict::Fail; }
    for i in 0..d {
        if !q[i].is_finite() || !fused_qkv[i].is_finite() { return Fqkv001Verdict::Fail; }
        if (fused_qkv[i] - q[i]).abs() > AC_FQKV_001_TOLERANCE { return Fqkv001Verdict::Fail; }
    }
    for i in 0..d {
        if !k[i].is_finite() || !fused_qkv[d + i].is_finite() { return Fqkv001Verdict::Fail; }
        if (fused_qkv[d + i] - k[i]).abs() > AC_FQKV_001_TOLERANCE { return Fqkv001Verdict::Fail; }
    }
    for i in 0..d {
        if !v[i].is_finite() || !fused_qkv[2 * d + i].is_finite() { return Fqkv001Verdict::Fail; }
        if (fused_qkv[2 * d + i] - v[i]).abs() > AC_FQKV_001_TOLERANCE { return Fqkv001Verdict::Fail; }
    }
    Fqkv001Verdict::Pass
}

// ===========================================================================
// FQKV-002 — Weight layout: W_qkv[i*d..(i+1)*d, :] = W_i for i ∈ {q, k, v}
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fqkv002Verdict { Pass, Fail }

/// Verdict that fused W_qkv (shape 3d × d, row-major flattened) has
/// W_q rows in [0..d), W_k rows in [d..2d), W_v rows in [2d..3d) — all
/// byte-exact to the corresponding W_q/W_k/W_v inputs.
#[must_use]
pub fn verdict_from_weight_layout(
    w_qkv: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    d_model: usize,
) -> Fqkv002Verdict {
    if d_model == 0 { return Fqkv002Verdict::Fail; }
    let block = d_model * d_model;
    if w_qkv.len() != 3 * block { return Fqkv002Verdict::Fail; }
    if w_q.len() != block || w_k.len() != block || w_v.len() != block {
        return Fqkv002Verdict::Fail;
    }
    for i in 0..block {
        if w_qkv[i].to_bits() != w_q[i].to_bits() { return Fqkv002Verdict::Fail; }
    }
    for i in 0..block {
        if w_qkv[block + i].to_bits() != w_k[i].to_bits() { return Fqkv002Verdict::Fail; }
    }
    for i in 0..block {
        if w_qkv[2 * block + i].to_bits() != w_v[i].to_bits() { return Fqkv002Verdict::Fail; }
    }
    Fqkv002Verdict::Pass
}

// ===========================================================================
// FQKV-003 — Bias layout: b_qkv[i*d..(i+1)*d] = b_i for i ∈ {q, k, v}
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fqkv003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_bias_layout(
    b_qkv: &[f32],
    b_q: &[f32],
    b_k: &[f32],
    b_v: &[f32],
) -> Fqkv003Verdict {
    if b_q.is_empty() || b_k.is_empty() || b_v.is_empty() { return Fqkv003Verdict::Fail; }
    if b_q.len() != b_k.len() || b_k.len() != b_v.len() { return Fqkv003Verdict::Fail; }
    let d = b_q.len();
    if b_qkv.len() != 3 * d { return Fqkv003Verdict::Fail; }
    for i in 0..d {
        if b_qkv[i].to_bits() != b_q[i].to_bits() { return Fqkv003Verdict::Fail; }
    }
    for i in 0..d {
        if b_qkv[d + i].to_bits() != b_k[i].to_bits() { return Fqkv003Verdict::Fail; }
    }
    for i in 0..d {
        if b_qkv[2 * d + i].to_bits() != b_v[i].to_bits() { return Fqkv003Verdict::Fail; }
    }
    Fqkv003Verdict::Pass
}

// ===========================================================================
// FQKV-004 — Whisper dimensions: d_model ∈ {384, 512, 768, 1024, 1280}
// ===========================================================================

pub const AC_FQKV_004_WHISPER_DIMS: [u64; 5] = [384, 512, 768, 1024, 1280];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fqkv004Verdict { Pass, Fail }

/// Pass iff `d_model` is one of the canonical Whisper sizes (tiny, base,
/// small, medium, large) AND fused output length == 3 × d_model.
#[must_use]
pub fn verdict_from_whisper_dims(d_model: u64, fused_output_len: u64) -> Fqkv004Verdict {
    if d_model == 0 { return Fqkv004Verdict::Fail; }
    if !AC_FQKV_004_WHISPER_DIMS.contains(&d_model) { return Fqkv004Verdict::Fail; }
    if fused_output_len != 3 * d_model { return Fqkv004Verdict::Fail; }
    Fqkv004Verdict::Pass
}

// ===========================================================================
// FQKV-005 — Shared Q8_1 matches separate quantization (PMAT-054A) — exact
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fqkv005Verdict { Pass, Fail }

/// Pass iff `shared_qkv` is byte-exactly equal to the concatenation of
/// `separate_q`, `separate_k`, `separate_v` (the contract specifies
/// tolerance=0.0 for shared Q8 path because Q8Quantize is deterministic).
#[must_use]
pub fn verdict_from_shared_q8_equivalence(
    shared_qkv: &[f32],
    separate_q: &[f32],
    separate_k: &[f32],
    separate_v: &[f32],
) -> Fqkv005Verdict {
    if separate_q.is_empty() || separate_k.is_empty() || separate_v.is_empty() {
        return Fqkv005Verdict::Fail;
    }
    if separate_q.len() != separate_k.len() || separate_k.len() != separate_v.len() {
        return Fqkv005Verdict::Fail;
    }
    let d = separate_q.len();
    if shared_qkv.len() != 3 * d { return Fqkv005Verdict::Fail; }
    for i in 0..d {
        if shared_qkv[i].to_bits() != separate_q[i].to_bits() { return Fqkv005Verdict::Fail; }
    }
    for i in 0..d {
        if shared_qkv[d + i].to_bits() != separate_k[i].to_bits() { return Fqkv005Verdict::Fail; }
    }
    for i in 0..d {
        if shared_qkv[2 * d + i].to_bits() != separate_v[i].to_bits() { return Fqkv005Verdict::Fail; }
    }
    Fqkv005Verdict::Pass
}

// ===========================================================================
// FQKV-006 — Shared Q8_1 buffer not clobbered between GEMV launches
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fqkv006Verdict { Pass, Fail }

/// Pass iff Q8 buffer snapshots taken before each GEMV launch are
/// byte-identical (read-only invariant: GEMV must NOT write to its
/// quantized input buffer). 3 snapshots: before q-gemv, before k-gemv,
/// before v-gemv.
#[must_use]
pub fn verdict_from_q8_buffer_readonly(
    snapshot_before_q: &[u8],
    snapshot_before_k: &[u8],
    snapshot_before_v: &[u8],
) -> Fqkv006Verdict {
    if snapshot_before_q.is_empty() { return Fqkv006Verdict::Fail; }
    if snapshot_before_q.len() != snapshot_before_k.len() { return Fqkv006Verdict::Fail; }
    if snapshot_before_q.len() != snapshot_before_v.len() { return Fqkv006Verdict::Fail; }
    if snapshot_before_q != snapshot_before_k { return Fqkv006Verdict::Fail; }
    if snapshot_before_q != snapshot_before_v { return Fqkv006Verdict::Fail; }
    Fqkv006Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // FQKV-001 (fused vs separate)
    #[test] fn fqkv001_pass_canonical() {
        let q = vec![1.0_f32, 2.0];
        let k = vec![3.0_f32, 4.0];
        let v = vec![5.0_f32, 6.0];
        let fused = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(verdict_from_fused_separate_equivalence(&fused, &q, &k, &v), Fqkv001Verdict::Pass);
    }
    #[test] fn fqkv001_pass_within_tol() {
        let q = vec![1.0_f32];
        let k = vec![2.0_f32];
        let v = vec![3.0_f32];
        let fused = vec![1.0_f32 + 5e-7, 2.0, 3.0];
        assert_eq!(verdict_from_fused_separate_equivalence(&fused, &q, &k, &v), Fqkv001Verdict::Pass);
    }
    #[test] fn fqkv001_fail_q_drift() {
        let q = vec![1.0_f32];
        let k = vec![2.0_f32];
        let v = vec![3.0_f32];
        let fused = vec![1.5_f32, 2.0, 3.0]; // q slot drifted
        assert_eq!(verdict_from_fused_separate_equivalence(&fused, &q, &k, &v), Fqkv001Verdict::Fail);
    }
    #[test] fn fqkv001_fail_v_drift() {
        let q = vec![1.0_f32];
        let k = vec![2.0_f32];
        let v = vec![3.0_f32];
        let fused = vec![1.0_f32, 2.0, 99.0]; // v slot drifted
        assert_eq!(verdict_from_fused_separate_equivalence(&fused, &q, &k, &v), Fqkv001Verdict::Fail);
    }
    #[test] fn fqkv001_fail_length() {
        let q = vec![1.0_f32];
        let k = vec![2.0_f32];
        let v = vec![3.0_f32];
        let fused = vec![1.0_f32, 2.0, 3.0, 99.0]; // wrong total length
        assert_eq!(verdict_from_fused_separate_equivalence(&fused, &q, &k, &v), Fqkv001Verdict::Fail);
    }

    // FQKV-002 (weight layout)
    #[test] fn fqkv002_pass_canonical() {
        let w_q = vec![1.0_f32, 2.0, 3.0, 4.0]; // 2x2
        let w_k = vec![5.0_f32, 6.0, 7.0, 8.0];
        let w_v = vec![9.0_f32, 10.0, 11.0, 12.0];
        let mut w_qkv = w_q.clone();
        w_qkv.extend_from_slice(&w_k);
        w_qkv.extend_from_slice(&w_v);
        assert_eq!(verdict_from_weight_layout(&w_qkv, &w_q, &w_k, &w_v, 2), Fqkv002Verdict::Pass);
    }
    #[test] fn fqkv002_fail_swapped_q_k() {
        // Common error: row-major confusion swaps Q and K blocks.
        let w_q = vec![1.0_f32, 2.0, 3.0, 4.0];
        let w_k = vec![5.0_f32, 6.0, 7.0, 8.0];
        let w_v = vec![9.0_f32, 10.0, 11.0, 12.0];
        let mut w_qkv = w_k.clone(); // swapped!
        w_qkv.extend_from_slice(&w_q);
        w_qkv.extend_from_slice(&w_v);
        assert_eq!(verdict_from_weight_layout(&w_qkv, &w_q, &w_k, &w_v, 2), Fqkv002Verdict::Fail);
    }
    #[test] fn fqkv002_fail_size() {
        let w_q = vec![1.0_f32];
        let w_k = vec![1.0_f32];
        let w_v = vec![1.0_f32];
        let w_qkv = vec![1.0_f32, 1.0]; // wrong size
        assert_eq!(verdict_from_weight_layout(&w_qkv, &w_q, &w_k, &w_v, 1), Fqkv002Verdict::Fail);
    }

    // FQKV-003 (bias layout)
    #[test] fn fqkv003_pass_canonical() {
        let b_q = vec![0.1_f32, 0.2];
        let b_k = vec![0.3_f32, 0.4];
        let b_v = vec![0.5_f32, 0.6];
        let mut b_qkv = b_q.clone();
        b_qkv.extend_from_slice(&b_k);
        b_qkv.extend_from_slice(&b_v);
        assert_eq!(verdict_from_bias_layout(&b_qkv, &b_q, &b_k, &b_v), Fqkv003Verdict::Pass);
    }
    #[test] fn fqkv003_fail_drift() {
        let b_q = vec![0.1_f32];
        let b_k = vec![0.2_f32];
        let b_v = vec![0.3_f32];
        let b_qkv = vec![0.1_f32, 0.999, 0.3]; // k slot off
        assert_eq!(verdict_from_bias_layout(&b_qkv, &b_q, &b_k, &b_v), Fqkv003Verdict::Fail);
    }
    #[test] fn fqkv003_fail_length() {
        let b_q = vec![0.1_f32];
        let b_k = vec![0.2_f32];
        let b_v = vec![0.3_f32];
        let b_qkv = vec![0.1_f32, 0.2]; // wrong total
        assert_eq!(verdict_from_bias_layout(&b_qkv, &b_q, &b_k, &b_v), Fqkv003Verdict::Fail);
    }

    // FQKV-004 (Whisper dims)
    #[test] fn fqkv004_pass_tiny() {
        // Whisper-tiny: d_model = 384.
        assert_eq!(verdict_from_whisper_dims(384, 1152), Fqkv004Verdict::Pass);
    }
    #[test] fn fqkv004_pass_large() {
        // Whisper-large: d_model = 1280.
        assert_eq!(verdict_from_whisper_dims(1280, 3840), Fqkv004Verdict::Pass);
    }
    #[test] fn fqkv004_pass_all_canonical() {
        for &d in &AC_FQKV_004_WHISPER_DIMS {
            assert_eq!(verdict_from_whisper_dims(d, 3 * d), Fqkv004Verdict::Pass);
        }
    }
    #[test] fn fqkv004_fail_non_whisper_dim() {
        // 4096 is Qwen-class, not Whisper.
        assert_eq!(verdict_from_whisper_dims(4096, 12288), Fqkv004Verdict::Fail);
    }
    #[test] fn fqkv004_fail_wrong_output_len() {
        // d_model=384 but output is not 3*384.
        assert_eq!(verdict_from_whisper_dims(384, 1024), Fqkv004Verdict::Fail);
    }
    #[test] fn fqkv004_fail_zero() {
        assert_eq!(verdict_from_whisper_dims(0, 0), Fqkv004Verdict::Fail);
    }

    // FQKV-005 (shared Q8_1 equivalence)
    #[test] fn fqkv005_pass_byte_exact() {
        let q = vec![1.0_f32, 2.0];
        let k = vec![3.0_f32, 4.0];
        let v = vec![5.0_f32, 6.0];
        let mut shared = q.clone();
        shared.extend_from_slice(&k);
        shared.extend_from_slice(&v);
        assert_eq!(verdict_from_shared_q8_equivalence(&shared, &q, &k, &v), Fqkv005Verdict::Pass);
    }
    #[test] fn fqkv005_fail_one_ulp() {
        // tolerance=0.0 — even 1-ULP drift fails.
        let q = vec![1.0_f32];
        let k = vec![2.0_f32];
        let v = vec![3.0_f32];
        let shared = vec![f32::from_bits(1.0_f32.to_bits() + 1), 2.0, 3.0];
        assert_eq!(verdict_from_shared_q8_equivalence(&shared, &q, &k, &v), Fqkv005Verdict::Fail);
    }
    #[test] fn fqkv005_fail_length() {
        let q = vec![1.0_f32];
        let k = vec![2.0_f32];
        let v = vec![3.0_f32];
        let shared = vec![1.0_f32, 2.0]; // missing v slot
        assert_eq!(verdict_from_shared_q8_equivalence(&shared, &q, &k, &v), Fqkv005Verdict::Fail);
    }

    // FQKV-006 (Q8 buffer read-only)
    #[test] fn fqkv006_pass_buffer_unchanged() {
        let snap = vec![1_u8, 2, 3, 4, 5];
        assert_eq!(verdict_from_q8_buffer_readonly(&snap, &snap, &snap), Fqkv006Verdict::Pass);
    }
    #[test] fn fqkv006_fail_clobbered_after_q() {
        // Q-GEMV wrote to the Q8 buffer.
        let snap_q = vec![1_u8, 2, 3];
        let snap_k = vec![1_u8, 2, 99]; // corrupted
        let snap_v = vec![1_u8, 2, 99];
        assert_eq!(verdict_from_q8_buffer_readonly(&snap_q, &snap_k, &snap_v), Fqkv006Verdict::Fail);
    }
    #[test] fn fqkv006_fail_clobbered_after_k() {
        let snap_q = vec![1_u8, 2, 3];
        let snap_k = vec![1_u8, 2, 3];
        let snap_v = vec![1_u8, 2, 99]; // corrupted between K and V launches
        assert_eq!(verdict_from_q8_buffer_readonly(&snap_q, &snap_k, &snap_v), Fqkv006Verdict::Fail);
    }
    #[test] fn fqkv006_fail_length_mismatch() {
        let snap_q = vec![1_u8, 2, 3];
        let snap_k = vec![1_u8, 2];
        let snap_v = vec![1_u8, 2, 3];
        assert_eq!(verdict_from_q8_buffer_readonly(&snap_q, &snap_k, &snap_v), Fqkv006Verdict::Fail);
    }
    #[test] fn fqkv006_fail_empty() {
        assert_eq!(verdict_from_q8_buffer_readonly(&[], &[], &[]), Fqkv006Verdict::Fail);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_FQKV_001_TOLERANCE - 1e-6).abs() < 1e-12);
        assert_eq!(AC_FQKV_004_WHISPER_DIMS, [384, 512, 768, 1024, 1280]);
    }
}
