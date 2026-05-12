// SHIP-TWO-001 — `nf4-tensor-core-gemm-v1` (3 gates) +
// `nf4-backward-tensor-core-gemm-v1` (2 gates) algorithm-level
// PARTIAL discharge for FALSIFY-NF4-TC-001..003 AND
// FALSIFY-NF4-BTC-001..002 (closes 5/5 across both contracts).
//
// Contracts:
// - `contracts/nf4-tensor-core-gemm-v1.yaml`
// - `contracts/nf4-backward-tensor-core-gemm-v1.yaml`
// Spec: PMAT-479 — WMMA 16×16×16 NF4 tensor-core GEMM (forward +
// backward), Dettmers QLoRA 2023.

// ===========================================================================
// NF4-TC-001 — TC GEMM matches naive NF4 GEMM within 1e-3
// ===========================================================================

pub const AC_NFTC_001_TOLERANCE: f32 = 1.0e-3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nftc001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_tc_gemm_equivalence(tc_out: &[f32], naive_out: &[f32]) -> Nftc001Verdict {
    if tc_out.is_empty() || naive_out.is_empty() { return Nftc001Verdict::Fail; }
    if tc_out.len() != naive_out.len() { return Nftc001Verdict::Fail; }
    for (&a, &b) in tc_out.iter().zip(naive_out.iter()) {
        if !a.is_finite() || !b.is_finite() { return Nftc001Verdict::Fail; }
        if (a - b).abs() > AC_NFTC_001_TOLERANCE { return Nftc001Verdict::Fail; }
    }
    Nftc001Verdict::Pass
}

// ===========================================================================
// NF4-TC-002 — Throughput improvement ≥ 5× naive
// ===========================================================================

pub const AC_NFTC_002_MIN_SPEEDUP: f32 = 5.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nftc002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_tc_speedup(tc_tps: f32, naive_tps: f32) -> Nftc002Verdict {
    if !tc_tps.is_finite() || !naive_tps.is_finite() { return Nftc002Verdict::Fail; }
    if tc_tps <= 0.0 || naive_tps <= 0.0 { return Nftc002Verdict::Fail; }
    let speedup = tc_tps / naive_tps;
    if !speedup.is_finite() { return Nftc002Verdict::Fail; }
    if speedup < AC_NFTC_002_MIN_SPEEDUP { return Nftc002Verdict::Fail; }
    Nftc002Verdict::Pass
}

// ===========================================================================
// NF4-TC-003 — NF4 dequant to FP16 in shared memory matches CPU ref ≤ 1e-3
// ===========================================================================

pub const AC_NFTC_003_TOLERANCE: f32 = 1.0e-3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nftc003Verdict { Pass, Fail }

/// Pass iff GPU shared-memory dequant matches CPU reference within
/// `AC_NFTC_003_TOLERANCE`. Caller dumps shared memory contents
/// after the dequant phase and the verdict compares to the CPU path.
#[must_use]
pub fn verdict_from_dequant_match(gpu_shared_mem: &[f32], cpu_reference: &[f32]) -> Nftc003Verdict {
    if gpu_shared_mem.is_empty() || cpu_reference.is_empty() { return Nftc003Verdict::Fail; }
    if gpu_shared_mem.len() != cpu_reference.len() { return Nftc003Verdict::Fail; }
    for (&g, &c) in gpu_shared_mem.iter().zip(cpu_reference.iter()) {
        if !g.is_finite() || !c.is_finite() { return Nftc003Verdict::Fail; }
        if (g - c).abs() > AC_NFTC_003_TOLERANCE { return Nftc003Verdict::Fail; }
    }
    Nftc003Verdict::Pass
}

// ===========================================================================
// NF4-BTC-001 — Backward gradient: |nf4_grad - fp32_grad| < 1e-3 ∀ param
// ===========================================================================

pub const AC_NFBTC_001_TOLERANCE: f32 = 1.0e-3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nfbtc001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_backward_gradient_parity(
    nf4_grad: &[f32],
    fp32_grad: &[f32],
) -> Nfbtc001Verdict {
    if nf4_grad.is_empty() || fp32_grad.is_empty() { return Nfbtc001Verdict::Fail; }
    if nf4_grad.len() != fp32_grad.len() { return Nfbtc001Verdict::Fail; }
    for (&n, &f) in nf4_grad.iter().zip(fp32_grad.iter()) {
        if !n.is_finite() || !f.is_finite() { return Nfbtc001Verdict::Fail; }
        if (n - f).abs() > AC_NFBTC_001_TOLERANCE { return Nfbtc001Verdict::Fail; }
    }
    Nfbtc001Verdict::Pass
}

// ===========================================================================
// NF4-BTC-002 — Memory saving: peak_vram(nf4) < 0.5 × peak_vram(fp16)
// ===========================================================================

pub const AC_NFBTC_002_VRAM_RATIO_MAX: f32 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nfbtc002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_memory_saving(nf4_peak_vram: u64, fp16_peak_vram: u64) -> Nfbtc002Verdict {
    if nf4_peak_vram == 0 || fp16_peak_vram == 0 { return Nfbtc002Verdict::Fail; }
    let ratio = (nf4_peak_vram as f64) / (fp16_peak_vram as f64);
    if !ratio.is_finite() { return Nfbtc002Verdict::Fail; }
    if ratio >= AC_NFBTC_002_VRAM_RATIO_MAX as f64 { return Nfbtc002Verdict::Fail; }
    Nfbtc002Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // NF4-TC-001 (TC GEMM equivalence)
    #[test] fn nftc001_pass_identical() {
        let a = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_tc_gemm_equivalence(&a, &a), Nftc001Verdict::Pass);
    }
    #[test] fn nftc001_pass_within_tol() {
        let a = vec![1.0_f32];
        let b = vec![1.0_f32 + 5e-4]; // < 1e-3
        assert_eq!(verdict_from_tc_gemm_equivalence(&a, &b), Nftc001Verdict::Pass);
    }
    #[test] fn nftc001_fail_above_tol() {
        let a = vec![1.0_f32];
        let b = vec![1.01_f32]; // > 1e-3
        assert_eq!(verdict_from_tc_gemm_equivalence(&a, &b), Nftc001Verdict::Fail);
    }
    #[test] fn nftc001_fail_length() {
        let a = vec![1.0_f32];
        let b = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_tc_gemm_equivalence(&a, &b), Nftc001Verdict::Fail);
    }
    #[test] fn nftc001_fail_nan() {
        let a = vec![f32::NAN];
        let b = vec![1.0_f32];
        assert_eq!(verdict_from_tc_gemm_equivalence(&a, &b), Nftc001Verdict::Fail);
    }

    // NF4-TC-002 (speedup)
    #[test] fn nftc002_pass_5x() {
        // Exactly 5× speedup — at the boundary.
        assert_eq!(verdict_from_tc_speedup(5000.0, 1000.0), Nftc002Verdict::Pass);
    }
    #[test] fn nftc002_pass_higher_speedup() {
        // 40× — typical tensor core gain.
        assert_eq!(verdict_from_tc_speedup(40_000.0, 1000.0), Nftc002Verdict::Pass);
    }
    #[test] fn nftc002_fail_below_5x() {
        // 4× speedup — below threshold.
        assert_eq!(verdict_from_tc_speedup(4000.0, 1000.0), Nftc002Verdict::Fail);
    }
    #[test] fn nftc002_fail_no_gain() {
        assert_eq!(verdict_from_tc_speedup(1000.0, 1000.0), Nftc002Verdict::Fail);
    }
    #[test] fn nftc002_fail_zero_naive() {
        assert_eq!(verdict_from_tc_speedup(5000.0, 0.0), Nftc002Verdict::Fail);
    }
    #[test] fn nftc002_fail_nan() {
        assert_eq!(verdict_from_tc_speedup(f32::NAN, 1000.0), Nftc002Verdict::Fail);
    }

    // NF4-TC-003 (dequant to FP16 in shared memory)
    #[test] fn nftc003_pass_identical() {
        let a = vec![0.5_f32, -0.3, 0.0];
        assert_eq!(verdict_from_dequant_match(&a, &a), Nftc003Verdict::Pass);
    }
    #[test] fn nftc003_pass_within_tol() {
        // FP16 quantization noise typical at this band.
        let gpu = vec![0.5_f32 + 5e-4];
        let cpu = vec![0.5_f32];
        assert_eq!(verdict_from_dequant_match(&gpu, &cpu), Nftc003Verdict::Pass);
    }
    #[test] fn nftc003_fail_above_tol() {
        // Bank conflict in shared memory could corrupt to this magnitude.
        let gpu = vec![0.5_f32 + 0.01];
        let cpu = vec![0.5_f32];
        assert_eq!(verdict_from_dequant_match(&gpu, &cpu), Nftc003Verdict::Fail);
    }
    #[test] fn nftc003_fail_length() {
        let gpu = vec![0.5_f32];
        let cpu = vec![0.5_f32, 0.3];
        assert_eq!(verdict_from_dequant_match(&gpu, &cpu), Nftc003Verdict::Fail);
    }

    // NF4-BTC-001 (backward gradient parity)
    #[test] fn nfbtc001_pass_identical() {
        let a = vec![0.1_f32, -0.2, 0.3];
        assert_eq!(verdict_from_backward_gradient_parity(&a, &a), Nfbtc001Verdict::Pass);
    }
    #[test] fn nfbtc001_pass_within_tol() {
        let nf4 = vec![0.1_f32];
        let fp32 = vec![0.1_f32 + 5e-4];
        assert_eq!(verdict_from_backward_gradient_parity(&nf4, &fp32), Nfbtc001Verdict::Pass);
    }
    #[test] fn nfbtc001_fail_above_tol() {
        let nf4 = vec![0.1_f32];
        let fp32 = vec![0.5_f32];
        assert_eq!(verdict_from_backward_gradient_parity(&nf4, &fp32), Nfbtc001Verdict::Fail);
    }
    #[test] fn nfbtc001_fail_length() {
        let nf4 = vec![0.1_f32];
        let fp32 = vec![0.1_f32, 0.2];
        assert_eq!(verdict_from_backward_gradient_parity(&nf4, &fp32), Nfbtc001Verdict::Fail);
    }
    #[test] fn nfbtc001_fail_nan() {
        let nf4 = vec![f32::NAN];
        let fp32 = vec![0.1_f32];
        assert_eq!(verdict_from_backward_gradient_parity(&nf4, &fp32), Nfbtc001Verdict::Fail);
    }

    // NF4-BTC-002 (memory saving)
    #[test] fn nfbtc002_pass_canonical() {
        // NF4 uses 4 bits/weight, FP16 uses 16 bits/weight → 4x compression.
        // Even with activations + gradients, NF4 should be < 0.5x FP16 VRAM.
        // 4 GB NF4 vs 10 GB FP16 → ratio 0.4 < 0.5.
        assert_eq!(
            verdict_from_memory_saving(4 * 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024),
            Nfbtc002Verdict::Pass
        );
    }
    #[test] fn nfbtc002_pass_25_percent() {
        // 1/4 ratio is well below 0.5.
        assert_eq!(
            verdict_from_memory_saving(2_000_000, 8_000_000),
            Nfbtc002Verdict::Pass
        );
    }
    #[test] fn nfbtc002_fail_at_boundary() {
        // Ratio = 0.5 — strictly less is required (the contract says
        // "< 0.5 × peak_vram(fp16)" — the strict < fails at equality).
        assert_eq!(
            verdict_from_memory_saving(5_000_000, 10_000_000),
            Nfbtc002Verdict::Fail
        );
    }
    #[test] fn nfbtc002_fail_above_50_percent() {
        // NF4 used MORE than half FP16 — quant savings broken.
        assert_eq!(
            verdict_from_memory_saving(7_000_000, 10_000_000),
            Nfbtc002Verdict::Fail
        );
    }
    #[test] fn nfbtc002_fail_zero() {
        assert_eq!(verdict_from_memory_saving(0, 10_000_000), Nfbtc002Verdict::Fail);
        assert_eq!(verdict_from_memory_saving(5_000_000, 0), Nfbtc002Verdict::Fail);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_NFTC_001_TOLERANCE - 1e-3).abs() < 1e-9);
        assert!((AC_NFTC_002_MIN_SPEEDUP - 5.0).abs() < 1e-9);
        assert!((AC_NFTC_003_TOLERANCE - 1e-3).abs() < 1e-9);
        assert!((AC_NFBTC_001_TOLERANCE - 1e-3).abs() < 1e-9);
        assert!((AC_NFBTC_002_VRAM_RATIO_MAX - 0.5).abs() < 1e-9);
    }
}
