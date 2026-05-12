// SHIP-TWO-001 — `apr-cli-operations-v1` algorithm-level PARTIAL
// discharge for FALSIFY-OPS-002.
//
// Contract: `contracts/apr-cli-operations-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What FALSIFY-OPS-002 says
//
//   rule: No resource leaks after command exit
//   prediction: GPU memory returns to baseline after inference error
//   test: Force inference OOM, measure GPU memory after, compare
//         to baseline
//   if_fails: GPU memory leak on error path
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given (`baseline_bytes`, `post_error_bytes`,
// `tolerance_bytes`), Pass iff:
//
//   baseline_bytes is in [0, AC_OPS_002_MAX_PLAUSIBLE_GPU_BYTES] AND
//   post_error_bytes is in same range AND
//   |post_error_bytes - baseline_bytes| <= tolerance_bytes AND
//   tolerance_bytes <= AC_OPS_002_MAX_TOLERANCE_BYTES (256 MiB)
//
// Bounded-delta verdict — different from byte-equality (which
// would be too strict; CUDA driver retains some metadata pages).
// Tolerance must itself be bounded — otherwise a contributor
// could pass `tolerance = u64::MAX` and silently disable the gate.

/// Maximum plausible GPU memory in bytes (1 TiB).
///
/// Catches counter corruption / unsigned-overflow regressions.
/// 1 TiB exceeds any consumer or datacenter GPU.
pub const AC_OPS_002_MAX_PLAUSIBLE_GPU_BYTES: u64 = 1_024 * 1_024 * 1_024 * 1_024; // 1 TiB

/// Maximum legal tolerance band, in bytes (256 MiB).
///
/// CUDA driver typically retains tens of MiB of metadata after
/// freeing user allocations. 256 MiB gives generous slack while
/// still rejecting tolerance values that would mask real leaks
/// (e.g., tolerance = 8 GiB would silently accept a per-error
/// leak of an entire 7B model's weights).
pub const AC_OPS_002_MAX_TOLERANCE_BYTES: u64 = 256 * 1_024 * 1_024; // 256 MiB

/// Binary verdict for `FALSIFY-OPS-002`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ops002Verdict {
    /// `|post_error - baseline| <= tolerance` AND all values are
    /// within their domain bounds.
    Pass,
    /// One or more of:
    /// - Either memory reading exceeds 1 TiB (corruption).
    /// - `tolerance > 256 MiB` (caller error — would mask leaks).
    /// - `|post_error - baseline| > tolerance` (memory leak).
    Fail,
}

/// Pure verdict function for `FALSIFY-OPS-002`.
///
/// Inputs:
/// - `baseline_bytes`: GPU memory occupied before the inference
///   error (typically the model's working set).
/// - `post_error_bytes`: GPU memory occupied after the failing
///   inference returned (should be back to baseline).
/// - `tolerance_bytes`: allowable absolute delta. Capped at
///   256 MiB.
///
/// Pass iff:
/// 1. `baseline_bytes <= 1 TiB`,
/// 2. `post_error_bytes <= 1 TiB`,
/// 3. `tolerance_bytes <= 256 MiB`,
/// 4. `|post_error_bytes - baseline_bytes| <= tolerance_bytes`.
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// 6 GiB baseline, 6 GiB after error, 64 MiB tolerance — `Pass`:
/// ```
/// use aprender::format::ops_002::{
///     verdict_from_gpu_memory_leak_delta, Ops002Verdict,
/// };
/// const GIB: u64 = 1_073_741_824;
/// const MIB: u64 = 1_048_576;
/// let v = verdict_from_gpu_memory_leak_delta(6 * GIB, 6 * GIB, 64 * MIB);
/// assert_eq!(v, Ops002Verdict::Pass);
/// ```
///
/// 6 GiB baseline, 8 GiB after error (2 GiB leak), 64 MiB tolerance —
/// `Fail`:
/// ```
/// use aprender::format::ops_002::{
///     verdict_from_gpu_memory_leak_delta, Ops002Verdict,
/// };
/// const GIB: u64 = 1_073_741_824;
/// const MIB: u64 = 1_048_576;
/// let v = verdict_from_gpu_memory_leak_delta(6 * GIB, 8 * GIB, 64 * MIB);
/// assert_eq!(v, Ops002Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_gpu_memory_leak_delta(
    baseline_bytes: u64,
    post_error_bytes: u64,
    tolerance_bytes: u64,
) -> Ops002Verdict {
    if baseline_bytes > AC_OPS_002_MAX_PLAUSIBLE_GPU_BYTES {
        return Ops002Verdict::Fail;
    }
    if post_error_bytes > AC_OPS_002_MAX_PLAUSIBLE_GPU_BYTES {
        return Ops002Verdict::Fail;
    }
    if tolerance_bytes > AC_OPS_002_MAX_TOLERANCE_BYTES {
        return Ops002Verdict::Fail;
    }
    let delta = post_error_bytes.abs_diff(baseline_bytes);
    if delta <= tolerance_bytes {
        Ops002Verdict::Pass
    } else {
        Ops002Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KIB: u64 = 1_024;
    const MIB: u64 = 1_024 * KIB;
    const GIB: u64 = 1_024 * MIB;
    const TIB: u64 = 1_024 * GIB;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — domain caps.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_max_gpu_bytes_is_1_tib() {
        assert_eq!(AC_OPS_002_MAX_PLAUSIBLE_GPU_BYTES, TIB);
    }

    #[test]
    fn provenance_max_tolerance_is_256_mib() {
        assert_eq!(AC_OPS_002_MAX_TOLERANCE_BYTES, 256 * MIB);
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — clean memory return at canonical scales.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_exact_baseline_match() {
        let v = verdict_from_gpu_memory_leak_delta(6 * GIB, 6 * GIB, 64 * MIB);
        assert_eq!(v, Ops002Verdict::Pass);
    }

    #[test]
    fn pass_within_tolerance_above() {
        // 6 GiB → 6 GiB + 32 MiB; within 64 MiB tolerance.
        let v = verdict_from_gpu_memory_leak_delta(6 * GIB, 6 * GIB + 32 * MIB, 64 * MIB);
        assert_eq!(v, Ops002Verdict::Pass);
    }

    #[test]
    fn pass_within_tolerance_below() {
        // 6 GiB → 6 GiB - 32 MiB; within 64 MiB tolerance.
        let v = verdict_from_gpu_memory_leak_delta(6 * GIB, 6 * GIB - 32 * MIB, 64 * MIB);
        assert_eq!(v, Ops002Verdict::Pass);
    }

    #[test]
    fn pass_at_exact_tolerance_boundary() {
        // delta = tolerance (inclusive).
        let v = verdict_from_gpu_memory_leak_delta(6 * GIB, 6 * GIB + 64 * MIB, 64 * MIB);
        assert_eq!(v, Ops002Verdict::Pass);
    }

    #[test]
    fn pass_zero_baseline_zero_after() {
        let v = verdict_from_gpu_memory_leak_delta(0, 0, MIB);
        assert_eq!(v, Ops002Verdict::Pass);
    }

    #[test]
    fn pass_at_max_tolerance_256_mib() {
        let v = verdict_from_gpu_memory_leak_delta(6 * GIB, 6 * GIB + 200 * MIB, 256 * MIB);
        assert_eq!(v, Ops002Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — leak above tolerance.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_just_above_tolerance() {
        // delta = 64 MiB + 1 byte; tolerance = 64 MiB.
        let v =
            verdict_from_gpu_memory_leak_delta(6 * GIB, 6 * GIB + 64 * MIB + 1, 64 * MIB);
        assert_eq!(
            v,
            Ops002Verdict::Fail,
            "1-byte over tolerance must Fail"
        );
    }

    #[test]
    fn fail_2_gib_leak() {
        let v = verdict_from_gpu_memory_leak_delta(6 * GIB, 8 * GIB, 64 * MIB);
        assert_eq!(v, Ops002Verdict::Fail);
    }

    #[test]
    fn fail_full_model_leak() {
        // Per-error leak of an entire 7B model's working set: 4 GiB.
        let v = verdict_from_gpu_memory_leak_delta(6 * GIB, 10 * GIB, 64 * MIB);
        assert_eq!(
            v,
            Ops002Verdict::Fail,
            "model-sized leak must Fail (catastrophic)"
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — domain violations (memory readings).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_baseline_above_1_tib() {
        let v = verdict_from_gpu_memory_leak_delta(TIB + 1, 6 * GIB, 64 * MIB);
        assert_eq!(v, Ops002Verdict::Fail);
    }

    #[test]
    fn fail_post_error_above_1_tib() {
        let v = verdict_from_gpu_memory_leak_delta(6 * GIB, TIB + 1, 64 * MIB);
        assert_eq!(v, Ops002Verdict::Fail);
    }

    #[test]
    fn fail_baseline_at_u64_max() {
        let v = verdict_from_gpu_memory_leak_delta(u64::MAX, 6 * GIB, 64 * MIB);
        assert_eq!(v, Ops002Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Fail band — caller error (tolerance too large).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_tolerance_just_above_256_mib() {
        let v = verdict_from_gpu_memory_leak_delta(6 * GIB, 6 * GIB, 256 * MIB + 1);
        assert_eq!(
            v,
            Ops002Verdict::Fail,
            "tolerance > 256 MiB must Fail (would mask leaks)"
        );
    }

    #[test]
    fn fail_tolerance_8_gib_would_mask_model_leak() {
        // Adversarial: tolerance = 8 GiB would silently accept a
        // 7B-model-sized leak.
        let v = verdict_from_gpu_memory_leak_delta(6 * GIB, 6 * GIB, 8 * GIB);
        assert_eq!(v, Ops002Verdict::Fail);
    }

    #[test]
    fn fail_tolerance_at_u64_max() {
        let v = verdict_from_gpu_memory_leak_delta(6 * GIB, 6 * GIB, u64::MAX);
        assert_eq!(v, Ops002Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Boundary sweep — delta around tolerance band.
    // -------------------------------------------------------------------------
    #[test]
    fn delta_sweep_around_64_mib_tolerance() {
        let baseline = 6 * GIB;
        let tolerance = 64 * MIB;
        let probes: Vec<(u64, Ops002Verdict)> = vec![
            (baseline, Ops002Verdict::Pass),
            (baseline + MIB, Ops002Verdict::Pass),
            (baseline + 32 * MIB, Ops002Verdict::Pass),
            (baseline + tolerance - 1, Ops002Verdict::Pass),
            (baseline + tolerance, Ops002Verdict::Pass), // inclusive
            (baseline + tolerance + 1, Ops002Verdict::Fail),
            (baseline + 100 * MIB, Ops002Verdict::Fail),
            (baseline + 1 * GIB, Ops002Verdict::Fail),
        ];
        for (post, expected) in probes {
            let v = verdict_from_gpu_memory_leak_delta(baseline, post, tolerance);
            assert_eq!(
                v, expected,
                "post={post} expected {expected:?} (baseline=6 GiB, tol=64 MiB)"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — typical RTX 4090 / A100 baselines.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_rtx_4090_clean_inference_error() {
        // RTX 4090 24 GiB; Qwen2.5-7B Q4_K = ~5 GiB; 64 MiB tolerance
        // covers driver metadata.
        let v = verdict_from_gpu_memory_leak_delta(5 * GIB, 5 * GIB + 12 * MIB, 64 * MIB);
        assert_eq!(v, Ops002Verdict::Pass);
    }

    #[test]
    fn pass_a100_80gb_clean_inference_error() {
        // A100 80 GiB; Qwen2.5-72B Q4_K = ~38 GiB.
        let v = verdict_from_gpu_memory_leak_delta(38 * GIB, 38 * GIB + 8 * MIB, 64 * MIB);
        assert_eq!(v, Ops002Verdict::Pass);
    }

    #[test]
    fn fail_per_request_kv_cache_leak() {
        // Realistic regression: KV cache for a failed request not
        // freed; each failure adds ~256 MiB.
        let v =
            verdict_from_gpu_memory_leak_delta(5 * GIB, 5 * GIB + 256 * MIB + MIB, 64 * MIB);
        assert_eq!(
            v,
            Ops002Verdict::Fail,
            "KV cache leak must Fail"
        );
    }
}
