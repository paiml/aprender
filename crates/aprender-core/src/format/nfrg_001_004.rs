// SHIP-TWO-001 — `nf4-fused-rmsnorm-gemv-v1` algorithm-level PARTIAL
// discharge for FALSIFY-NF4-RMS-001..004 (closes 4/4 sweep).
//
// Contract: `contracts/nf4-fused-rmsnorm-gemv-v1.yaml`.
// Spec: Fused RMSNorm + NF4 GEMV — entrenar QLoRA training kernel
// (PMAT-475; replicates QWEN-009 Q4K fusion for NF4 dtype).

// ===========================================================================
// NF4-RMS-001 — Fused matches separate RMSNorm + NF4 GEMV within 1e-4
// ===========================================================================

pub const AC_NFRG_001_TOLERANCE: f32 = 1.0e-4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nfrg001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_fused_equivalence(fused: &[f32], separate: &[f32]) -> Nfrg001Verdict {
    if fused.is_empty() || separate.is_empty() { return Nfrg001Verdict::Fail; }
    if fused.len() != separate.len() { return Nfrg001Verdict::Fail; }
    for (&a, &b) in fused.iter().zip(separate.iter()) {
        if !a.is_finite() || !b.is_finite() { return Nfrg001Verdict::Fail; }
        if (a - b).abs() > AC_NFRG_001_TOLERANCE { return Nfrg001Verdict::Fail; }
    }
    Nfrg001Verdict::Pass
}

// ===========================================================================
// NF4-RMS-002 — Kernel count reduction: fused == 1, separate == 2; no
// intermediate writes
// ===========================================================================

pub const AC_NFRG_002_FUSED_KERNELS: u64 = 1;
pub const AC_NFRG_002_SEPARATE_KERNELS: u64 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nfrg002Verdict { Pass, Fail }

/// Pass iff:
/// 1. fused dispatches exactly 1 kernel
/// 2. separate dispatches exactly 2 kernels (RMSNorm + GEMV)
/// 3. fused intermediate writes == 0 (no DRAM roundtrip for normed)
/// 4. separate intermediate writes >= 1 (RMSNorm always writes normed)
#[must_use]
pub const fn verdict_from_kernel_count(
    fused_kernel_count: u64,
    separate_kernel_count: u64,
    fused_intermediate_writes: u64,
    separate_intermediate_writes: u64,
) -> Nfrg002Verdict {
    if fused_kernel_count != AC_NFRG_002_FUSED_KERNELS { return Nfrg002Verdict::Fail; }
    if separate_kernel_count != AC_NFRG_002_SEPARATE_KERNELS { return Nfrg002Verdict::Fail; }
    if fused_intermediate_writes != 0 { return Nfrg002Verdict::Fail; }
    if separate_intermediate_writes == 0 { return Nfrg002Verdict::Fail; }
    Nfrg002Verdict::Pass
}

// ===========================================================================
// NF4-RMS-003 — Throughput improvement ≥ 5%
// ===========================================================================

pub const AC_NFRG_003_MIN_GAIN: f32 = 0.05;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nfrg003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_throughput_gain(fused_tps: f32, separate_tps: f32) -> Nfrg003Verdict {
    if !fused_tps.is_finite() || !separate_tps.is_finite() { return Nfrg003Verdict::Fail; }
    if fused_tps <= 0.0 || separate_tps <= 0.0 { return Nfrg003Verdict::Fail; }
    let gain = (fused_tps / separate_tps) - 1.0;
    if !gain.is_finite() { return Nfrg003Verdict::Fail; }
    if gain < AC_NFRG_003_MIN_GAIN { return Nfrg003Verdict::Fail; }
    Nfrg003Verdict::Pass
}

// ===========================================================================
// NF4-RMS-004 — NF4 dequant identity AND rectangular GQA dims supported
// ===========================================================================

pub const AC_NFRG_004_QWEN_HIDDEN: u64 = 1536;
pub const AC_NFRG_004_QWEN_KV_DIM: u64 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nfrg004Verdict { Pass, Fail }

/// Pass iff:
/// 1. dequant_fused == dequant_standalone byte-exactly (NF4 LUT must
///    be numerically identical between fused and standalone paths)
/// 2. weight matrix shape (out_dim, hidden_size) accepts both square
///    (Q: 1536→1536) and rectangular (K/V: 1536→256) configurations
#[must_use]
pub fn verdict_from_dequant_and_gqa_dims(
    dequant_fused: &[f32],
    dequant_standalone: &[f32],
    out_dim: u64,
    hidden_size: u64,
) -> Nfrg004Verdict {
    if dequant_fused.is_empty() || dequant_standalone.is_empty() { return Nfrg004Verdict::Fail; }
    if dequant_fused.len() != dequant_standalone.len() { return Nfrg004Verdict::Fail; }
    if out_dim == 0 || hidden_size == 0 { return Nfrg004Verdict::Fail; }
    // NF4 block alignment requires hidden_size % 64 == 0 (per contract).
    if !hidden_size.is_multiple_of(64) { return Nfrg004Verdict::Fail; }
    // Byte-exact dequant identity.
    for (&a, &b) in dequant_fused.iter().zip(dequant_standalone.iter()) {
        if a.to_bits() != b.to_bits() { return Nfrg004Verdict::Fail; }
    }
    Nfrg004Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // NF4-RMS-001 (fused equivalence)
    #[test] fn nfrg_001_pass_identical() {
        let a = vec![0.5_f32, 1.0, -0.3];
        assert_eq!(verdict_from_fused_equivalence(&a, &a), Nfrg001Verdict::Pass);
    }
    #[test] fn nfrg_001_pass_within_tolerance() {
        let a = vec![1.0_f32];
        let b = vec![1.0_f32 + 5e-5]; // < 1e-4
        assert_eq!(verdict_from_fused_equivalence(&a, &b), Nfrg001Verdict::Pass);
    }
    #[test] fn nfrg_001_fail_above_tolerance() {
        let a = vec![1.0_f32];
        let b = vec![1.001_f32]; // > 1e-4
        assert_eq!(verdict_from_fused_equivalence(&a, &b), Nfrg001Verdict::Fail);
    }
    #[test] fn nfrg_001_fail_length() {
        let a = vec![1.0_f32];
        let b = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_fused_equivalence(&a, &b), Nfrg001Verdict::Fail);
    }
    #[test] fn nfrg_001_fail_nan() {
        let a = vec![f32::NAN];
        let b = vec![1.0_f32];
        assert_eq!(verdict_from_fused_equivalence(&a, &b), Nfrg001Verdict::Fail);
    }

    // NF4-RMS-002 (kernel count)
    #[test] fn nfrg_002_pass_canonical() {
        // Fused: 1 kernel, 0 writes. Separate: 2 kernels, 1 write.
        assert_eq!(verdict_from_kernel_count(1, 2, 0, 1), Nfrg002Verdict::Pass);
    }
    #[test] fn nfrg_002_pass_separate_with_extra_writes() {
        // Separate may write more than 1 (e.g., debug telemetry); only the
        // floor matters.
        assert_eq!(verdict_from_kernel_count(1, 2, 0, 5), Nfrg002Verdict::Pass);
    }
    #[test] fn nfrg_002_fail_fused_writes_nonzero() {
        // Fused must NOT write the intermediate normed buffer.
        assert_eq!(verdict_from_kernel_count(1, 2, 1, 1), Nfrg002Verdict::Fail);
    }
    #[test] fn nfrg_002_fail_separate_no_writes() {
        // The contract requires separate to have AT LEAST 1 intermediate
        // write (RMSNorm output) — zero means our model of "separate"
        // is wrong.
        assert_eq!(verdict_from_kernel_count(1, 2, 0, 0), Nfrg002Verdict::Fail);
    }
    #[test] fn nfrg_002_fail_wrong_fused_count() {
        assert_eq!(verdict_from_kernel_count(2, 2, 0, 1), Nfrg002Verdict::Fail);
    }
    #[test] fn nfrg_002_fail_wrong_separate_count() {
        assert_eq!(verdict_from_kernel_count(1, 3, 0, 1), Nfrg002Verdict::Fail);
    }

    // NF4-RMS-003 (throughput)
    #[test] fn nfrg_003_pass_canonical() {
        // 10% gain — well above 5%.
        assert_eq!(verdict_from_throughput_gain(1100.0, 1000.0), Nfrg003Verdict::Pass);
    }
    #[test] fn nfrg_003_pass_at_higher_gain() {
        // 20% gain.
        assert_eq!(verdict_from_throughput_gain(1200.0, 1000.0), Nfrg003Verdict::Pass);
    }
    #[test] fn nfrg_003_fail_below_5_percent() {
        // 4% gain.
        assert_eq!(verdict_from_throughput_gain(1040.0, 1000.0), Nfrg003Verdict::Fail);
    }
    #[test] fn nfrg_003_fail_no_gain() {
        assert_eq!(verdict_from_throughput_gain(1000.0, 1000.0), Nfrg003Verdict::Fail);
    }
    #[test] fn nfrg_003_fail_regression() {
        // Fused slower than separate.
        assert_eq!(verdict_from_throughput_gain(900.0, 1000.0), Nfrg003Verdict::Fail);
    }
    #[test] fn nfrg_003_fail_zero_separate() {
        assert_eq!(verdict_from_throughput_gain(1100.0, 0.0), Nfrg003Verdict::Fail);
    }
    #[test] fn nfrg_003_fail_nan() {
        assert_eq!(verdict_from_throughput_gain(f32::NAN, 1000.0), Nfrg003Verdict::Fail);
    }

    // NF4-RMS-004 (dequant + GQA dims)
    #[test] fn nfrg_004_pass_qwen_q_square() {
        // Qwen 1.5B Q: 1536→1536.
        let a = vec![1.0_f32; 1024];
        let b = a.clone();
        assert_eq!(
            verdict_from_dequant_and_gqa_dims(&a, &b, 1536, 1536),
            Nfrg004Verdict::Pass
        );
    }
    #[test] fn nfrg_004_pass_qwen_kv_rectangular() {
        // Qwen 1.5B K/V: 1536→256 (GQA).
        let a = vec![0.5_f32; 256];
        let b = a.clone();
        assert_eq!(
            verdict_from_dequant_and_gqa_dims(&a, &b, 256, 1536),
            Nfrg004Verdict::Pass
        );
    }
    #[test] fn nfrg_004_fail_dequant_byte_drift() {
        // 1-ULP perturbation in dequant output is a regression
        // (NF4 LUT must be byte-identical between fused/standalone).
        let a = vec![1.0_f32];
        let b = vec![f32::from_bits(1.0_f32.to_bits() + 1)];
        assert_eq!(
            verdict_from_dequant_and_gqa_dims(&a, &b, 1536, 1536),
            Nfrg004Verdict::Fail
        );
    }
    #[test] fn nfrg_004_fail_hidden_not_aligned() {
        // hidden_size must be divisible by 64 (NF4 block alignment).
        let a = vec![1.0_f32];
        let b = vec![1.0_f32];
        assert_eq!(
            verdict_from_dequant_and_gqa_dims(&a, &b, 1536, 1500),
            Nfrg004Verdict::Fail
        );
    }
    #[test] fn nfrg_004_fail_zero_dim() {
        let a = vec![1.0_f32];
        let b = vec![1.0_f32];
        assert_eq!(
            verdict_from_dequant_and_gqa_dims(&a, &b, 0, 1536),
            Nfrg004Verdict::Fail
        );
        assert_eq!(
            verdict_from_dequant_and_gqa_dims(&a, &b, 1536, 0),
            Nfrg004Verdict::Fail
        );
    }
    #[test] fn nfrg_004_fail_length_mismatch() {
        let a = vec![1.0_f32, 2.0];
        let b = vec![1.0_f32];
        assert_eq!(
            verdict_from_dequant_and_gqa_dims(&a, &b, 1536, 1536),
            Nfrg004Verdict::Fail
        );
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_NFRG_001_TOLERANCE - 1e-4).abs() < 1e-9);
        assert_eq!(AC_NFRG_002_FUSED_KERNELS, 1);
        assert_eq!(AC_NFRG_002_SEPARATE_KERNELS, 2);
        assert!((AC_NFRG_003_MIN_GAIN - 0.05).abs() < 1e-9);
        assert_eq!(AC_NFRG_004_QWEN_HIDDEN, 1536);
        assert_eq!(AC_NFRG_004_QWEN_KV_DIM, 256);
    }
}
