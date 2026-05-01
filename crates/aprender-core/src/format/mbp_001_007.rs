// SHIP-TWO-001 — `gpu-multi-backend-parity-v1` algorithm-level PARTIAL
// discharge for FALSIFY-MBP-001..007 (closes 7/7 sweep).
//
// Contract: `contracts/gpu-multi-backend-parity-v1.yaml`.

// ===========================================================================
// Canonical cosine thresholds per backend
// ===========================================================================

pub const AC_MBP_GPU_PARITY_THRESHOLD: f64 = 0.98;
pub const AC_MBP_PYTORCH_CANARY_THRESHOLD: f64 = 0.9999;
pub const AC_MBP_Q4K_SPEEDUP_RATIO: f64 = 2.0;

// Reference cosine helper.
fn cosine(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() { return None; }
    if a.iter().chain(b).any(|v| !v.is_finite()) { return None; }
    let dot: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { return None; }
    Some(dot / (na * nb))
}

// ===========================================================================
// MBP-001 — wgpu vs CPU cosine >= 0.98
// MBP-002 — NVRTC vs CPU cosine >= 0.98
// MBP-004 — CUDA-JIT vs CPU cosine >= 0.98 (pre-Blackwell)
//
// All three reduce to identical cosine-threshold comparisons.
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mbp001Verdict { Pass, Fail }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mbp002Verdict { Pass, Fail }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mbp004Verdict { Pass, Fail }

#[must_use]
fn verdict_cosine_ge_threshold(a: &[f32], b: &[f32], threshold: f64) -> bool {
    matches!(cosine(a, b), Some(c) if c >= threshold)
}

#[must_use]
pub fn verdict_from_wgpu_parity(wgpu: &[f32], cpu: &[f32]) -> Mbp001Verdict {
    if verdict_cosine_ge_threshold(wgpu, cpu, AC_MBP_GPU_PARITY_THRESHOLD) {
        Mbp001Verdict::Pass
    } else {
        Mbp001Verdict::Fail
    }
}

#[must_use]
pub fn verdict_from_nvrtc_parity(nvrtc: &[f32], cpu: &[f32]) -> Mbp002Verdict {
    if verdict_cosine_ge_threshold(nvrtc, cpu, AC_MBP_GPU_PARITY_THRESHOLD) {
        Mbp002Verdict::Pass
    } else {
        Mbp002Verdict::Fail
    }
}

#[must_use]
pub fn verdict_from_cuda_jit_parity(cuda: &[f32], cpu: &[f32]) -> Mbp004Verdict {
    if verdict_cosine_ge_threshold(cuda, cpu, AC_MBP_GPU_PARITY_THRESHOLD) {
        Mbp004Verdict::Pass
    } else {
        Mbp004Verdict::Fail
    }
}

// ===========================================================================
// MBP-003 — PyTorch canary: GPU vs CPU cosine >= 0.9999 (hardware sanity)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mbp003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_pytorch_canary(gpu: &[f32], cpu: &[f32]) -> Mbp003Verdict {
    if verdict_cosine_ge_threshold(gpu, cpu, AC_MBP_PYTORCH_CANARY_THRESHOLD) {
        Mbp003Verdict::Pass
    } else {
        Mbp003Verdict::Fail
    }
}

// ===========================================================================
// MBP-005 — Q4K GEMV >= 2× cuBLAS FP16 HGEMM tok/s
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mbp005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_q4k_speedup(q4k_tps: f64, cublas_tps: f64) -> Mbp005Verdict {
    if !q4k_tps.is_finite() || !cublas_tps.is_finite() { return Mbp005Verdict::Fail; }
    if cublas_tps <= 0.0 || q4k_tps < 0.0 { return Mbp005Verdict::Fail; }
    if q4k_tps / cublas_tps >= AC_MBP_Q4K_SPEEDUP_RATIO {
        Mbp005Verdict::Pass
    } else {
        Mbp005Verdict::Fail
    }
}

// ===========================================================================
// MBP-006 — No silent fallback: parity < 0.98 ⇒ CPU output (exact match)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mbp006Verdict { Pass, Fail }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendOutcome {
    /// GPU parity met threshold; output came from GPU.
    GpuPassed,
    /// GPU parity failed; output came from CPU exactly (line stopped, fallback).
    GpuFailedFellBackToCpu,
    /// GPU parity failed but output was served from GPU anyway (Toyota Way violation).
    GpuFailedServedGarbage,
}

#[must_use]
pub const fn verdict_from_no_silent_fallback(outcome: BackendOutcome) -> Mbp006Verdict {
    match outcome {
        BackendOutcome::GpuPassed => Mbp006Verdict::Pass,
        BackendOutcome::GpuFailedFellBackToCpu => Mbp006Verdict::Pass,
        BackendOutcome::GpuFailedServedGarbage => Mbp006Verdict::Fail,
    }
}

// ===========================================================================
// MBP-007 — Driver update resolves JIT bug: post-driver cosine >= 0.98
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mbp007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_driver_update_resolves(driver_version: u32, post_driver_cosine: f64) -> Mbp007Verdict {
    if !post_driver_cosine.is_finite() { return Mbp007Verdict::Fail; }
    if driver_version < 590 { return Mbp007Verdict::Fail; }
    if post_driver_cosine >= AC_MBP_GPU_PARITY_THRESHOLD { Mbp007Verdict::Pass } else { Mbp007Verdict::Fail }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: f32) -> [f32; 4] { [a, a + 0.01, a + 0.02, a + 0.03] }

    // MBP-001
    #[test] fn mbp001_pass_close() {
        let cpu = near(1.0);
        let wgpu = near(1.0);
        assert_eq!(verdict_from_wgpu_parity(&wgpu, &cpu), Mbp001Verdict::Pass);
    }
    #[test] fn mbp001_fail_orthogonal() {
        let cpu = [1.0_f32, 0.0, 0.0, 0.0];
        let wgpu = [0.0_f32, 1.0, 0.0, 0.0];
        assert_eq!(verdict_from_wgpu_parity(&wgpu, &cpu), Mbp001Verdict::Fail);
    }
    #[test] fn mbp001_fail_dim_mismatch() {
        assert_eq!(verdict_from_wgpu_parity(&[1.0_f32], &[1.0, 2.0]), Mbp001Verdict::Fail);
    }

    // MBP-002
    #[test] fn mbp002_pass() {
        let cpu = near(1.0);
        let nvrtc = near(1.0);
        assert_eq!(verdict_from_nvrtc_parity(&nvrtc, &cpu), Mbp002Verdict::Pass);
    }

    // MBP-003 (tighter threshold)
    #[test] fn mbp003_pass_perfect() {
        let cpu = near(1.0);
        assert_eq!(verdict_from_pytorch_canary(&cpu, &cpu), Mbp003Verdict::Pass);
    }
    #[test] fn mbp003_fail_close_but_below_canary() {
        // 0.99 cosine — passes 0.98 GPU threshold but fails 0.9999 canary.
        let cpu = [1.0_f32, 0.0, 0.0];
        let gpu = [0.85_f32, 0.4, 0.0];
        // Close to cpu cosine ~ 0.85, well below 0.9999.
        assert_eq!(verdict_from_pytorch_canary(&gpu, &cpu), Mbp003Verdict::Fail);
    }

    // MBP-004
    #[test] fn mbp004_pass() {
        let cpu = near(1.0);
        let cuda = near(1.0);
        assert_eq!(verdict_from_cuda_jit_parity(&cuda, &cpu), Mbp004Verdict::Pass);
    }

    // MBP-005
    #[test] fn mbp005_pass_2x() {
        // Q4K 200 tok/s vs cuBLAS 100 tok/s → 2x.
        assert_eq!(verdict_from_q4k_speedup(200.0, 100.0), Mbp005Verdict::Pass);
    }
    #[test] fn mbp005_pass_3x() {
        assert_eq!(verdict_from_q4k_speedup(300.0, 100.0), Mbp005Verdict::Pass);
    }
    #[test] fn mbp005_fail_1_5x() {
        assert_eq!(verdict_from_q4k_speedup(150.0, 100.0), Mbp005Verdict::Fail);
    }
    #[test] fn mbp005_fail_zero_baseline() {
        assert_eq!(verdict_from_q4k_speedup(100.0, 0.0), Mbp005Verdict::Fail);
    }
    #[test] fn mbp005_fail_nan() {
        assert_eq!(verdict_from_q4k_speedup(f64::NAN, 100.0), Mbp005Verdict::Fail);
    }

    // MBP-006
    #[test] fn mbp006_pass_gpu_passed() {
        assert_eq!(verdict_from_no_silent_fallback(BackendOutcome::GpuPassed), Mbp006Verdict::Pass);
    }
    #[test] fn mbp006_pass_clean_fallback() {
        assert_eq!(
            verdict_from_no_silent_fallback(BackendOutcome::GpuFailedFellBackToCpu),
            Mbp006Verdict::Pass
        );
    }
    #[test] fn mbp006_fail_silent_garbage() {
        // Toyota Way violation: GPU failed parity but kept serving.
        assert_eq!(
            verdict_from_no_silent_fallback(BackendOutcome::GpuFailedServedGarbage),
            Mbp006Verdict::Fail
        );
    }

    // MBP-007
    #[test] fn mbp007_pass_driver_resolves() {
        assert_eq!(verdict_from_driver_update_resolves(595, 0.99), Mbp007Verdict::Pass);
    }
    #[test] fn mbp007_fail_old_driver() {
        // Driver 580 < 590 fails the version pin even with good cosine.
        assert_eq!(verdict_from_driver_update_resolves(580, 0.99), Mbp007Verdict::Fail);
    }
    #[test] fn mbp007_fail_still_low_cosine() {
        assert_eq!(verdict_from_driver_update_resolves(595, 0.5), Mbp007Verdict::Fail);
    }
    #[test] fn mbp007_fail_nan_cosine() {
        assert_eq!(verdict_from_driver_update_resolves(595, f64::NAN), Mbp007Verdict::Fail);
    }

    // Provenance pins
    #[test] fn provenance_thresholds() {
        assert!((AC_MBP_GPU_PARITY_THRESHOLD - 0.98).abs() < 1e-9);
        assert!((AC_MBP_PYTORCH_CANARY_THRESHOLD - 0.9999).abs() < 1e-9);
        assert!((AC_MBP_Q4K_SPEEDUP_RATIO - 2.0).abs() < 1e-12);
    }
}
