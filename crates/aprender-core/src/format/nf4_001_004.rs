// SHIP-TWO-001 — `nf4-fused-gate-up-swiglu-v1` algorithm-level PARTIAL
// discharge for FALSIFY-NF4-FFN-001..004 (closes 4/4 sweep).
//
// Contract: `contracts/nf4-fused-gate-up-swiglu-v1.yaml`.
// Spec: Fused RMSNorm + Gate + Up + SwiGLU for NF4 quantized weights —
// entrenar QLoRA training kernel (PMAT-475; replicates QWEN-009 Q4K
// fusion for NF4 dtype).

// ===========================================================================
// NF4-FFN-001 — Fused FFN matches separate kernels within 1e-4
// ===========================================================================

pub const AC_NF4_001_TOLERANCE: f32 = 1.0e-4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nf4001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_fused_equivalence(fused: &[f32], separate: &[f32]) -> Nf4001Verdict {
    if fused.is_empty() || separate.is_empty() { return Nf4001Verdict::Fail; }
    if fused.len() != separate.len() { return Nf4001Verdict::Fail; }
    for (&a, &b) in fused.iter().zip(separate.iter()) {
        if !a.is_finite() || !b.is_finite() { return Nf4001Verdict::Fail; }
        if (a - b).abs() > AC_NF4_001_TOLERANCE { return Nf4001Verdict::Fail; }
    }
    Nf4001Verdict::Pass
}

// ===========================================================================
// NF4-FFN-002 — Kernel count: fused == 1, separate == 4; throughput gain ≥ 15%
// ===========================================================================

pub const AC_NF4_002_MIN_THROUGHPUT_GAIN: f32 = 0.15;
pub const AC_NF4_002_FUSED_KERNELS: u64 = 1;
pub const AC_NF4_002_SEPARATE_KERNELS: u64 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nf4002Verdict { Pass, Fail }

/// Pass iff:
/// 1. fused dispatches exactly 1 kernel per FFN block
/// 2. separate dispatches exactly 4 kernels per FFN block
/// 3. observed throughput gain (fused/separate - 1) ≥ 0.15
#[must_use]
pub fn verdict_from_kernel_count_and_throughput(
    fused_kernel_count: u64,
    separate_kernel_count: u64,
    fused_tps: f32,
    separate_tps: f32,
) -> Nf4002Verdict {
    if fused_kernel_count != AC_NF4_002_FUSED_KERNELS { return Nf4002Verdict::Fail; }
    if separate_kernel_count != AC_NF4_002_SEPARATE_KERNELS { return Nf4002Verdict::Fail; }
    if !fused_tps.is_finite() || !separate_tps.is_finite() { return Nf4002Verdict::Fail; }
    if fused_tps <= 0.0 || separate_tps <= 0.0 { return Nf4002Verdict::Fail; }
    let gain = (fused_tps / separate_tps) - 1.0;
    if !gain.is_finite() { return Nf4002Verdict::Fail; }
    if gain < AC_NF4_002_MIN_THROUGHPUT_GAIN { return Nf4002Verdict::Fail; }
    Nf4002Verdict::Pass
}

// ===========================================================================
// NF4-FFN-003 — SwiGLU numerical stability for gate ∈ [-100, 100]
// ===========================================================================

pub const AC_NF4_003_GATE_BOUND: f32 = 100.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nf4003Verdict { Pass, Fail }

/// Numerically stable SiLU: x * sigmoid(x). Sigmoid uses positive/negative
/// branches to avoid exp overflow.
#[must_use]
pub fn stable_silu(x: f32) -> f32 {
    if !x.is_finite() { return f32::NAN; }
    let s = if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    };
    x * s
}

/// Pass iff for every (gate, up) pair in `[-100, 100]`, the SwiGLU output
/// `silu(gate) * up` is finite.
#[must_use]
pub fn verdict_from_swiglu_stability(gates: &[f32], ups: &[f32]) -> Nf4003Verdict {
    if gates.is_empty() || ups.is_empty() { return Nf4003Verdict::Fail; }
    if gates.len() != ups.len() { return Nf4003Verdict::Fail; }
    for (&g, &u) in gates.iter().zip(ups.iter()) {
        if !g.is_finite() || !u.is_finite() { return Nf4003Verdict::Fail; }
        if g.abs() > AC_NF4_003_GATE_BOUND { return Nf4003Verdict::Fail; } // OOB
        let silu_g = stable_silu(g);
        if !silu_g.is_finite() { return Nf4003Verdict::Fail; }
        let out = silu_g * u;
        if !out.is_finite() { return Nf4003Verdict::Fail; }
    }
    Nf4003Verdict::Pass
}

// ===========================================================================
// NF4-FFN-004 — Bandwidth savings ≥ 100 KB at Qwen 1.5B dimensions
// ===========================================================================

pub const AC_NF4_004_QWEN_HIDDEN: u64 = 1536;
pub const AC_NF4_004_QWEN_INTERMEDIATE: u64 = 8960;
pub const AC_NF4_004_MIN_SAVINGS_BYTES: u64 = 100 * 1024; // 102_400

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nf4004Verdict { Pass, Fail }

#[must_use]
pub const fn separate_bandwidth_bytes(hidden: u64, intermediate: u64) -> u64 {
    // separate_bw = hidden * 12 + intermediate * 16 (per contract formula)
    hidden.saturating_mul(12).saturating_add(intermediate.saturating_mul(16))
}

#[must_use]
pub const fn fused_bandwidth_bytes(hidden: u64, intermediate: u64) -> u64 {
    // fused_bw = hidden * 4 + intermediate * 4
    hidden.saturating_mul(4).saturating_add(intermediate.saturating_mul(4))
}

#[must_use]
pub const fn bandwidth_savings_bytes(hidden: u64, intermediate: u64) -> u64 {
    let s = separate_bandwidth_bytes(hidden, intermediate);
    let f = fused_bandwidth_bytes(hidden, intermediate);
    s.saturating_sub(f)
}

#[must_use]
pub const fn verdict_from_bandwidth_savings(hidden: u64, intermediate: u64) -> Nf4004Verdict {
    if hidden == 0 || intermediate == 0 { return Nf4004Verdict::Fail; }
    let savings = bandwidth_savings_bytes(hidden, intermediate);
    if savings >= AC_NF4_004_MIN_SAVINGS_BYTES { Nf4004Verdict::Pass } else { Nf4004Verdict::Fail }
}

#[cfg(test)]
mod tests {
    use super::*;

    // NF4-FFN-001 (fused equivalence)
    #[test] fn nf4_001_pass_identical() {
        let a = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_fused_equivalence(&a, &a), Nf4001Verdict::Pass);
    }
    #[test] fn nf4_001_pass_within_tolerance() {
        let a = vec![1.0_f32];
        let b = vec![1.0_f32 + 5e-5]; // < 1e-4
        assert_eq!(verdict_from_fused_equivalence(&a, &b), Nf4001Verdict::Pass);
    }
    #[test] fn nf4_001_fail_above_tolerance() {
        let a = vec![1.0_f32];
        let b = vec![1.001_f32]; // > 1e-4
        assert_eq!(verdict_from_fused_equivalence(&a, &b), Nf4001Verdict::Fail);
    }
    #[test] fn nf4_001_fail_length() {
        let a = vec![1.0_f32];
        let b = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_fused_equivalence(&a, &b), Nf4001Verdict::Fail);
    }
    #[test] fn nf4_001_fail_nan() {
        let a = vec![f32::NAN];
        let b = vec![1.0_f32];
        assert_eq!(verdict_from_fused_equivalence(&a, &b), Nf4001Verdict::Fail);
    }

    // NF4-FFN-002 (kernel count + throughput)
    #[test] fn nf4_002_pass_canonical() {
        // Fused 1 kernel, separate 4 kernels, fused 20% faster.
        assert_eq!(
            verdict_from_kernel_count_and_throughput(1, 4, 1200.0, 1000.0),
            Nf4002Verdict::Pass
        );
    }
    #[test] fn nf4_002_pass_just_above_15_percent() {
        // 15.1% gain — clearly above the strict-≥ threshold (1150/1000 - 1
        // is not exactly 0.15 in f32, so we probe just above).
        assert_eq!(
            verdict_from_kernel_count_and_throughput(1, 4, 1151.0, 1000.0),
            Nf4002Verdict::Pass
        );
    }
    #[test] fn nf4_002_fail_below_15_percent() {
        // 14% gain — below threshold.
        assert_eq!(
            verdict_from_kernel_count_and_throughput(1, 4, 1140.0, 1000.0),
            Nf4002Verdict::Fail
        );
    }
    #[test] fn nf4_002_fail_wrong_fused_count() {
        // Fused must dispatch exactly 1 kernel.
        assert_eq!(
            verdict_from_kernel_count_and_throughput(2, 4, 1200.0, 1000.0),
            Nf4002Verdict::Fail
        );
    }
    #[test] fn nf4_002_fail_wrong_separate_count() {
        assert_eq!(
            verdict_from_kernel_count_and_throughput(1, 3, 1200.0, 1000.0),
            Nf4002Verdict::Fail
        );
    }
    #[test] fn nf4_002_fail_zero_separate_tps() {
        assert_eq!(
            verdict_from_kernel_count_and_throughput(1, 4, 1200.0, 0.0),
            Nf4002Verdict::Fail
        );
    }
    #[test] fn nf4_002_fail_nan_tps() {
        assert_eq!(
            verdict_from_kernel_count_and_throughput(1, 4, f32::NAN, 1000.0),
            Nf4002Verdict::Fail
        );
    }

    // NF4-FFN-003 (SwiGLU stability)
    #[test] fn nf4_003_pass_canonical_range() {
        // Sweep gate values in [-90, 90] with random ups.
        let gates: Vec<f32> = (-9..=9).map(|i| i as f32 * 10.0).collect();
        let ups: Vec<f32> = gates.iter().map(|&g| g * 0.5 + 1.0).collect();
        assert_eq!(verdict_from_swiglu_stability(&gates, &ups), Nf4003Verdict::Pass);
    }
    #[test] fn nf4_003_pass_edge_cases() {
        // Contract specifies gate=0, -88, 88 must be stable.
        let gates = vec![0.0_f32, -88.0, 88.0];
        let ups = vec![1.0_f32, 1.0, 1.0];
        assert_eq!(verdict_from_swiglu_stability(&gates, &ups), Nf4003Verdict::Pass);
    }
    #[test] fn nf4_003_fail_gate_oob() {
        // gate=200 exceeds [-100, 100] domain.
        let gates = vec![200.0_f32];
        let ups = vec![1.0_f32];
        assert_eq!(verdict_from_swiglu_stability(&gates, &ups), Nf4003Verdict::Fail);
    }
    #[test] fn nf4_003_fail_nan() {
        let gates = vec![f32::NAN];
        let ups = vec![1.0_f32];
        assert_eq!(verdict_from_swiglu_stability(&gates, &ups), Nf4003Verdict::Fail);
    }
    #[test] fn nf4_003_fail_length_mismatch() {
        let gates = vec![1.0_f32, 2.0];
        let ups = vec![1.0_f32];
        assert_eq!(verdict_from_swiglu_stability(&gates, &ups), Nf4003Verdict::Fail);
    }
    #[test] fn nf4_003_fail_empty() {
        assert_eq!(verdict_from_swiglu_stability(&[], &[]), Nf4003Verdict::Fail);
    }
    #[test] fn stable_silu_at_zero() {
        assert!((stable_silu(0.0) - 0.0).abs() < 1e-7);
    }
    #[test] fn stable_silu_at_minus_88_finite() {
        // Naive sigmoid would underflow exp(-88) only if we computed
        // exp(-(-88)) = exp(88), but the negative branch handles this.
        let s = stable_silu(-88.0);
        assert!(s.is_finite());
    }

    // NF4-FFN-004 (bandwidth savings)
    #[test] fn nf4_004_pass_qwen_1_5b() {
        // Qwen 1.5B: hidden=1536, intermediate=8960.
        // savings = 1536 * 8 + 8960 * 12 = 12288 + 107520 = 119808 ≥ 102400.
        assert_eq!(
            verdict_from_bandwidth_savings(1536, 8960),
            Nf4004Verdict::Pass
        );
        let savings = bandwidth_savings_bytes(1536, 8960);
        assert_eq!(savings, 119_808);
    }
    #[test] fn nf4_004_pass_larger_model() {
        // Qwen 7B: hidden=3584, intermediate=18944 — savings should grow linearly.
        assert_eq!(
            verdict_from_bandwidth_savings(3584, 18944),
            Nf4004Verdict::Pass
        );
    }
    #[test] fn nf4_004_fail_too_small() {
        // hidden=64, intermediate=128 — savings = 64*8 + 128*12 = 2048
        // which is < 102400.
        assert_eq!(
            verdict_from_bandwidth_savings(64, 128),
            Nf4004Verdict::Fail
        );
    }
    #[test] fn nf4_004_fail_zero() {
        assert_eq!(verdict_from_bandwidth_savings(0, 8960), Nf4004Verdict::Fail);
        assert_eq!(verdict_from_bandwidth_savings(1536, 0), Nf4004Verdict::Fail);
    }
    #[test] fn separate_bw_canonical() {
        // separate_bw = hidden * 12 + intermediate * 16
        assert_eq!(separate_bandwidth_bytes(1536, 8960), 1536 * 12 + 8960 * 16);
    }
    #[test] fn fused_bw_canonical() {
        // fused_bw = hidden * 4 + intermediate * 4
        assert_eq!(fused_bandwidth_bytes(1536, 8960), 1536 * 4 + 8960 * 4);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_NF4_001_TOLERANCE - 1e-4).abs() < 1e-9);
        assert!((AC_NF4_002_MIN_THROUGHPUT_GAIN - 0.15).abs() < 1e-9);
        assert_eq!(AC_NF4_002_FUSED_KERNELS, 1);
        assert_eq!(AC_NF4_002_SEPARATE_KERNELS, 4);
        assert!((AC_NF4_003_GATE_BOUND - 100.0).abs() < 1e-9);
        assert_eq!(AC_NF4_004_QWEN_HIDDEN, 1536);
        assert_eq!(AC_NF4_004_QWEN_INTERMEDIATE, 8960);
        assert_eq!(AC_NF4_004_MIN_SAVINGS_BYTES, 102_400);
    }
}
