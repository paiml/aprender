// SHIP-TWO-001 — `kv-cache-equivalence-v1` algorithm-level PARTIAL
// discharge for FALSIFY-KCE-001..006 (closes 6/6 sweep).
//
// Contract: `contracts/kv-cache-equivalence-v1.yaml`.
// Spec: KV cache equivalence, two-phase generation, fused kernel
// correctness; PagedAttention page shape and frame conditions
// (Qwen2.5-Coder Showcase §14, FlashAttention).

// ===========================================================================
// KCE-001 — Prefill/incremental equivalence: |cached - full| < 1e-5
// ===========================================================================

pub const AC_KCE_001_TOLERANCE: f32 = 1.0e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kce001Verdict { Pass, Fail }

/// Pass iff `cached_last_token` matches `full_forward[n]` element-wise
/// within `AC_KCE_001_TOLERANCE` for the last token of an n-token sequence.
#[must_use]
pub fn verdict_from_prefill_incremental(cached: &[f32], full_last_token: &[f32]) -> Kce001Verdict {
    if cached.is_empty() || full_last_token.is_empty() { return Kce001Verdict::Fail; }
    if cached.len() != full_last_token.len() { return Kce001Verdict::Fail; }
    for (&a, &b) in cached.iter().zip(full_last_token.iter()) {
        if !a.is_finite() || !b.is_finite() { return Kce001Verdict::Fail; }
        if (a - b).abs() > AC_KCE_001_TOLERANCE { return Kce001Verdict::Fail; }
    }
    Kce001Verdict::Pass
}

// ===========================================================================
// KCE-002 — Page shape: page_elements == block_size * n_kv * d_k
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kce002Verdict { Pass, Fail }

#[must_use]
pub const fn page_elements(block_size: u64, n_kv: u64, d_k: u64) -> u64 {
    block_size.saturating_mul(n_kv).saturating_mul(d_k)
}

#[must_use]
pub const fn verdict_from_page_shape(
    block_size: u64,
    n_kv: u64,
    d_k: u64,
    observed: u64,
) -> Kce002Verdict {
    if block_size == 0 || n_kv == 0 || d_k == 0 { return Kce002Verdict::Fail; }
    let expected = page_elements(block_size, n_kv, d_k);
    if observed == expected { Kce002Verdict::Pass } else { Kce002Verdict::Fail }
}

// ===========================================================================
// KCE-003 — Batched/serial equivalence: |batched - serial| < 1e-5
// ===========================================================================

pub const AC_KCE_003_TOLERANCE: f32 = 1.0e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kce003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_batched_serial(batched: &[f32], serial: &[f32]) -> Kce003Verdict {
    if batched.is_empty() || serial.is_empty() { return Kce003Verdict::Fail; }
    if batched.len() != serial.len() { return Kce003Verdict::Fail; }
    for (&b, &s) in batched.iter().zip(serial.iter()) {
        if !b.is_finite() || !s.is_finite() { return Kce003Verdict::Fail; }
        if (b - s).abs() > AC_KCE_003_TOLERANCE { return Kce003Verdict::Fail; }
    }
    Kce003Verdict::Pass
}

// ===========================================================================
// KCE-004 — Fused kernel equivalence: tolerance depends on quantization
// ===========================================================================

pub const AC_KCE_004_Q4K_TOLERANCE: f32 = 1.0e-3;
pub const AC_KCE_004_F16_TOLERANCE: f32 = 1.0e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedDtype { Q4K, F16, F32 }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kce004Verdict { Pass, Fail }

#[must_use]
pub const fn fused_tolerance_for(dtype: FusedDtype) -> f32 {
    match dtype {
        FusedDtype::Q4K => AC_KCE_004_Q4K_TOLERANCE,
        FusedDtype::F16 | FusedDtype::F32 => AC_KCE_004_F16_TOLERANCE,
    }
}

#[must_use]
pub fn verdict_from_fused_kernel(
    fused: &[f32],
    decomposed: &[f32],
    dtype: FusedDtype,
) -> Kce004Verdict {
    if fused.is_empty() || decomposed.is_empty() { return Kce004Verdict::Fail; }
    if fused.len() != decomposed.len() { return Kce004Verdict::Fail; }
    let tol = fused_tolerance_for(dtype);
    for (&a, &b) in fused.iter().zip(decomposed.iter()) {
        if !a.is_finite() || !b.is_finite() { return Kce004Verdict::Fail; }
        if (a - b).abs() > tol { return Kce004Verdict::Fail; }
    }
    Kce004Verdict::Pass
}

// ===========================================================================
// KCE-005 — Frame condition: cache append preserves entries [0..old_len]
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kce005Verdict { Pass, Fail }

/// Pass iff `cache_after[..old_len]` is byte-identical to `cache_before`.
#[must_use]
pub fn verdict_from_frame_preservation(cache_before: &[f32], cache_after: &[f32]) -> Kce005Verdict {
    if cache_before.is_empty() || cache_after.is_empty() { return Kce005Verdict::Fail; }
    if cache_after.len() < cache_before.len() { return Kce005Verdict::Fail; }
    for (i, &b) in cache_before.iter().enumerate() {
        // Byte-exact via to_bits — frame condition is a "modifies" predicate
        // and any drift, even in the lowest ULP, is a regression.
        if cache_after[i].to_bits() != b.to_bits() { return Kce005Verdict::Fail; }
    }
    Kce005Verdict::Pass
}

// ===========================================================================
// KCE-006 — Old state: cache.len after == cache.len before + new_token_count
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kce006Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_length_growth(
    old_len: u64,
    new_len: u64,
    new_token_count: u64,
) -> Kce006Verdict {
    // saturating in case of overflow on 32-bit hosts (u64 won't, but defensive).
    let expected = old_len.saturating_add(new_token_count);
    if new_len == expected { Kce006Verdict::Pass } else { Kce006Verdict::Fail }
}

#[cfg(test)]
mod tests {
    use super::*;

    // KCE-001 (prefill/incremental)
    #[test] fn kce001_pass_identical() {
        let a = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_prefill_incremental(&a, &a), Kce001Verdict::Pass);
    }
    #[test] fn kce001_pass_within_tolerance() {
        let a = vec![1.0_f32, 2.0];
        let b = vec![1.0_f32 + 1e-6, 2.0 - 1e-6];
        assert_eq!(verdict_from_prefill_incremental(&a, &b), Kce001Verdict::Pass);
    }
    #[test] fn kce001_fail_above_tolerance() {
        let a = vec![1.0_f32];
        let b = vec![1.001_f32]; // delta = 0.001 > 1e-5
        assert_eq!(verdict_from_prefill_incremental(&a, &b), Kce001Verdict::Fail);
    }
    #[test] fn kce001_fail_length_mismatch() {
        let a = vec![1.0_f32];
        let b = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_prefill_incremental(&a, &b), Kce001Verdict::Fail);
    }
    #[test] fn kce001_fail_nan() {
        let a = vec![f32::NAN];
        let b = vec![1.0_f32];
        assert_eq!(verdict_from_prefill_incremental(&a, &b), Kce001Verdict::Fail);
    }

    // KCE-002 (page shape)
    #[test] fn kce002_pass_canonical() {
        // Block=16, n_kv=4, d_k=128 → 8192 elements/page.
        assert_eq!(page_elements(16, 4, 128), 8192);
        assert_eq!(verdict_from_page_shape(16, 4, 128, 8192), Kce002Verdict::Pass);
    }
    #[test] fn kce002_fail_off_by_factor() {
        // PagedAttention common bug: forgot a dimension → halved page size.
        assert_eq!(verdict_from_page_shape(16, 4, 128, 4096), Kce002Verdict::Fail);
    }
    #[test] fn kce002_fail_zero() {
        assert_eq!(verdict_from_page_shape(0, 4, 128, 0), Kce002Verdict::Fail);
        assert_eq!(verdict_from_page_shape(16, 0, 128, 0), Kce002Verdict::Fail);
    }

    // KCE-003 (batched/serial)
    #[test] fn kce003_pass_identical() {
        let a = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_batched_serial(&a, &a), Kce003Verdict::Pass);
    }
    #[test] fn kce003_fail_drift() {
        let a = vec![1.0_f32];
        let b = vec![1.5_f32];
        assert_eq!(verdict_from_batched_serial(&a, &b), Kce003Verdict::Fail);
    }
    #[test] fn kce003_fail_length() {
        let a = vec![1.0_f32];
        let b = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_batched_serial(&a, &b), Kce003Verdict::Fail);
    }

    // KCE-004 (fused kernel) — band depends on dtype
    #[test] fn kce004_pass_q4k_within_band() {
        let fused = vec![1.0_f32, 2.0];
        let decomposed = vec![1.0005_f32, 2.0005]; // ~5e-4 < 1e-3 Q4K tol
        assert_eq!(verdict_from_fused_kernel(&fused, &decomposed, FusedDtype::Q4K), Kce004Verdict::Pass);
    }
    #[test] fn kce004_fail_q4k_above_band() {
        let fused = vec![1.0_f32];
        let decomposed = vec![1.01_f32]; // 1e-2 > 1e-3 Q4K tol
        assert_eq!(verdict_from_fused_kernel(&fused, &decomposed, FusedDtype::Q4K), Kce004Verdict::Fail);
    }
    #[test] fn kce004_pass_f16_within_band() {
        let fused = vec![1.0_f32];
        let decomposed = vec![1.0_f32 + 5e-6]; // < 1e-5 F16 tol
        assert_eq!(verdict_from_fused_kernel(&fused, &decomposed, FusedDtype::F16), Kce004Verdict::Pass);
    }
    #[test] fn kce004_fail_f16_above_band() {
        let fused = vec![1.0_f32];
        let decomposed = vec![1.0_f32 + 1e-3]; // > 1e-5 F16 tol
        assert_eq!(verdict_from_fused_kernel(&fused, &decomposed, FusedDtype::F16), Kce004Verdict::Fail);
    }
    #[test] fn kce004_dtype_band_separation() {
        // Same delta passes Q4K but fails F16 — the band SEPARATION is what
        // makes this gate non-trivial.
        let fused = vec![1.0_f32];
        let decomposed = vec![1.0_f32 + 5e-4];
        assert_eq!(verdict_from_fused_kernel(&fused, &decomposed, FusedDtype::Q4K), Kce004Verdict::Pass);
        assert_eq!(verdict_from_fused_kernel(&fused, &decomposed, FusedDtype::F16), Kce004Verdict::Fail);
    }

    // KCE-005 (frame condition)
    #[test] fn kce005_pass_append_preserves_old() {
        let before = vec![1.0_f32, 2.0, 3.0];
        let mut after = before.clone();
        after.extend_from_slice(&[4.0, 5.0]);
        assert_eq!(verdict_from_frame_preservation(&before, &after), Kce005Verdict::Pass);
    }
    #[test] fn kce005_fail_byte_drift() {
        let before = vec![1.0_f32, 2.0, 3.0];
        let mut after = before.clone();
        // True 1-ULP perturbation at magnitude 2.0 (f32::EPSILON is the
        // 1-ULP at magnitude 1.0, which rounds away when added at 2.0).
        after[1] = f32::from_bits(2.0_f32.to_bits() + 1);
        after.extend_from_slice(&[4.0]);
        assert_eq!(verdict_from_frame_preservation(&before, &after), Kce005Verdict::Fail);
    }
    #[test] fn kce005_fail_overwritten() {
        // The exact regression class: append wrote into existing slot.
        let before = vec![1.0_f32, 2.0, 3.0];
        let mut after = vec![1.0_f32, 99.0, 3.0]; // slot 1 corrupted
        after.extend_from_slice(&[4.0]);
        assert_eq!(verdict_from_frame_preservation(&before, &after), Kce005Verdict::Fail);
    }
    #[test] fn kce005_fail_after_shorter() {
        let before = vec![1.0_f32, 2.0, 3.0];
        let after = vec![1.0_f32, 2.0]; // shorter than before — impossible
        assert_eq!(verdict_from_frame_preservation(&before, &after), Kce005Verdict::Fail);
    }

    // KCE-006 (length growth)
    #[test] fn kce006_pass_canonical() {
        assert_eq!(verdict_from_length_growth(100, 105, 5), Kce006Verdict::Pass);
    }
    #[test] fn kce006_pass_no_growth() {
        assert_eq!(verdict_from_length_growth(50, 50, 0), Kce006Verdict::Pass);
    }
    #[test] fn kce006_fail_off_by_one() {
        // Append wrote 5 entries but len only grew by 4 — the regression class.
        assert_eq!(verdict_from_length_growth(100, 104, 5), Kce006Verdict::Fail);
    }
    #[test] fn kce006_fail_phantom_growth() {
        // Append wrote phantom slot — len grew by more than n_new.
        assert_eq!(verdict_from_length_growth(100, 110, 5), Kce006Verdict::Fail);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_KCE_001_TOLERANCE - 1e-5).abs() < 1e-12);
        assert!((AC_KCE_003_TOLERANCE - 1e-5).abs() < 1e-12);
        assert!((AC_KCE_004_Q4K_TOLERANCE - 1e-3).abs() < 1e-9);
        assert!((AC_KCE_004_F16_TOLERANCE - 1e-5).abs() < 1e-12);
    }
}
