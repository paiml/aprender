// `backend-dispatch-v1` algorithm-level PARTIAL discharge for the 5
// backend-dispatch falsifiers (GPU threshold monotonicity, garbage
// oracle, QK norm score bound, BPE roundtrip, SIMD dispatch
// equivalence).
//
// Contract: `contracts/backend-dispatch-v1.yaml`.

/// GPU dispatch threshold per `equations.gpu_threshold`.
pub const AC_BD_GPU_THRESHOLD: u32 = 100_000;

/// SIMD-only threshold (no threading) per `equations.simd_only_threshold`.
pub const AC_BD_SIMD_ONLY_THRESHOLD: u32 = 1_000;

/// Garbage-oracle repetition ratio threshold (>0.3 = garbage).
pub const AC_BD_GARBAGE_REPETITION_THRESHOLD: f32 = 0.3;

/// Garbage-oracle minimum unique character count.
pub const AC_BD_GARBAGE_UNIQUE_CHARS_MIN: usize = 10;

/// SIMD vs scalar tolerance (contract: 0.0 — exact equality).
pub const AC_BD_SIMD_TOLERANCE: f32 = 0.0;

// =============================================================================
// FALSIFY-BD-001 — GPU dispatch monotonic
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDispatchVerdict {
    /// dispatch_chooses_gpu(n) ⇔ n >= 100_000.
    /// Monotonic: if n1 >= threshold AND n2 > n1 then n2 >= threshold.
    Pass,
    /// Non-monotonic dispatch decision.
    Fail,
}

#[must_use]
pub fn verdict_from_gpu_dispatch(n: u32, chose_gpu: bool) -> GpuDispatchVerdict {
    let should_be_gpu = n >= AC_BD_GPU_THRESHOLD;
    if chose_gpu == should_be_gpu {
        GpuDispatchVerdict::Pass
    } else {
        GpuDispatchVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-BD-002 — garbage oracle
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GarbageOracleVerdict {
    /// `is_garbage` matches predicate (rep_ratio > 0.3 OR unique_chars < 10).
    Pass,
    /// Oracle missed a degenerate string OR false-flagged a healthy one.
    Fail,
}

#[must_use]
pub fn verdict_from_garbage_oracle(
    repetition_ratio: f32,
    unique_chars: usize,
    classified_as_garbage: bool,
) -> GarbageOracleVerdict {
    if !repetition_ratio.is_finite() {
        return GarbageOracleVerdict::Fail;
    }
    let should_be_garbage = repetition_ratio > AC_BD_GARBAGE_REPETITION_THRESHOLD
        || unique_chars < AC_BD_GARBAGE_UNIQUE_CHARS_MIN;
    if classified_as_garbage == should_be_garbage {
        GarbageOracleVerdict::Pass
    } else {
        GarbageOracleVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-BD-003 — QK norm score bound
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QkNormBoundVerdict {
    /// |pre_softmax_score| ≤ sqrt(head_dim).
    Pass,
    /// L2 normalization not applied — score exploded.
    Fail,
}

#[must_use]
pub fn verdict_from_qk_norm_bound(score: f32, head_dim: u32) -> QkNormBoundVerdict {
    if head_dim == 0 {
        return QkNormBoundVerdict::Fail;
    }
    if !score.is_finite() {
        return QkNormBoundVerdict::Fail;
    }
    let bound = (head_dim as f32).sqrt();
    if score.abs() <= bound + 1e-5 {
        QkNormBoundVerdict::Pass
    } else {
        QkNormBoundVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-BD-004 — BPE roundtrip
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BpeRoundtripVerdict {
    /// encode(decode(tokens)) == tokens (lossless).
    Pass,
    /// Tokens diverge — lossy tokenization.
    Fail,
}

#[must_use]
pub fn verdict_from_bpe_roundtrip(original_tokens: &[u32], reencoded_tokens: &[u32]) -> BpeRoundtripVerdict {
    if original_tokens.is_empty() {
        return BpeRoundtripVerdict::Fail;
    }
    if original_tokens != reencoded_tokens {
        return BpeRoundtripVerdict::Fail;
    }
    BpeRoundtripVerdict::Pass
}

// =============================================================================
// FALSIFY-BD-005 — SIMD dispatch equivalence (exact)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdEquivalenceVerdict {
    /// SIMD output bit-identical to scalar output (tolerance 0.0).
    Pass,
    /// Diverged — SIMD dispatch produced different result.
    Fail,
}

#[must_use]
pub fn verdict_from_simd_equivalence(simd: &[f32], scalar: &[f32]) -> SimdEquivalenceVerdict {
    if simd.len() != scalar.len() {
        return SimdEquivalenceVerdict::Fail;
    }
    if simd.is_empty() {
        return SimdEquivalenceVerdict::Fail;
    }
    for (a, b) in simd.iter().zip(scalar.iter()) {
        if a.is_nan() != b.is_nan() {
            return SimdEquivalenceVerdict::Fail;
        }
        if !a.is_nan() && a != b {
            return SimdEquivalenceVerdict::Fail;
        }
    }
    SimdEquivalenceVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_gpu_threshold_100k() {
        assert_eq!(AC_BD_GPU_THRESHOLD, 100_000);
    }

    #[test]
    fn provenance_simd_only_threshold_1k() {
        assert_eq!(AC_BD_SIMD_ONLY_THRESHOLD, 1_000);
    }

    #[test]
    fn provenance_garbage_threshold_03() {
        assert!((AC_BD_GARBAGE_REPETITION_THRESHOLD - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn provenance_garbage_unique_min_10() {
        assert_eq!(AC_BD_GARBAGE_UNIQUE_CHARS_MIN, 10);
    }

    // -------------------------------------------------------------------------
    // Section 2: BD-001 GPU threshold.
    // -------------------------------------------------------------------------
    #[test]
    fn fbd001_pass_above_threshold_gpu() {
        assert_eq!(
            verdict_from_gpu_dispatch(150_000, true),
            GpuDispatchVerdict::Pass
        );
    }

    #[test]
    fn fbd001_pass_at_threshold_gpu() {
        assert_eq!(
            verdict_from_gpu_dispatch(100_000, true),
            GpuDispatchVerdict::Pass
        );
    }

    #[test]
    fn fbd001_pass_below_threshold_cpu() {
        assert_eq!(
            verdict_from_gpu_dispatch(50_000, false),
            GpuDispatchVerdict::Pass
        );
    }

    #[test]
    fn fbd001_fail_below_threshold_chose_gpu() {
        assert_eq!(
            verdict_from_gpu_dispatch(50_000, true),
            GpuDispatchVerdict::Fail
        );
    }

    #[test]
    fn fbd001_fail_above_threshold_chose_cpu() {
        assert_eq!(
            verdict_from_gpu_dispatch(200_000, false),
            GpuDispatchVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: BD-002 garbage oracle.
    // -------------------------------------------------------------------------
    #[test]
    fn fbd002_pass_clean_text_classified_clean() {
        // 0.1 rep ratio + 26 chars = healthy.
        assert_eq!(
            verdict_from_garbage_oracle(0.1, 26, false),
            GarbageOracleVerdict::Pass
        );
    }

    #[test]
    fn fbd002_pass_high_rep_classified_garbage() {
        assert_eq!(
            verdict_from_garbage_oracle(0.5, 30, true),
            GarbageOracleVerdict::Pass
        );
    }

    #[test]
    fn fbd002_pass_low_diversity_classified_garbage() {
        assert_eq!(
            verdict_from_garbage_oracle(0.1, 5, true),
            GarbageOracleVerdict::Pass
        );
    }

    #[test]
    fn fbd002_fail_high_rep_classified_clean() {
        // Oracle missed degenerate text.
        assert_eq!(
            verdict_from_garbage_oracle(0.5, 30, false),
            GarbageOracleVerdict::Fail
        );
    }

    #[test]
    fn fbd002_fail_clean_classified_garbage() {
        assert_eq!(
            verdict_from_garbage_oracle(0.05, 50, true),
            GarbageOracleVerdict::Fail
        );
    }

    #[test]
    fn fbd002_fail_nan_repetition() {
        assert_eq!(
            verdict_from_garbage_oracle(f32::NAN, 30, false),
            GarbageOracleVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: BD-003 QK norm bound.
    // -------------------------------------------------------------------------
    #[test]
    fn fbd003_pass_at_bound() {
        // head_dim=64 → bound = 8.
        assert_eq!(
            verdict_from_qk_norm_bound(8.0, 64),
            QkNormBoundVerdict::Pass
        );
    }

    #[test]
    fn fbd003_pass_within_bound() {
        assert_eq!(
            verdict_from_qk_norm_bound(5.0, 64),
            QkNormBoundVerdict::Pass
        );
    }

    #[test]
    fn fbd003_pass_negative_within_bound() {
        assert_eq!(
            verdict_from_qk_norm_bound(-7.5, 64),
            QkNormBoundVerdict::Pass
        );
    }

    #[test]
    fn fbd003_fail_above_bound() {
        // 100 > sqrt(64)=8.
        assert_eq!(
            verdict_from_qk_norm_bound(100.0, 64),
            QkNormBoundVerdict::Fail
        );
    }

    #[test]
    fn fbd003_fail_zero_head_dim() {
        assert_eq!(
            verdict_from_qk_norm_bound(0.0, 0),
            QkNormBoundVerdict::Fail
        );
    }

    #[test]
    fn fbd003_fail_nan_score() {
        assert_eq!(
            verdict_from_qk_norm_bound(f32::NAN, 64),
            QkNormBoundVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: BD-004 BPE roundtrip.
    // -------------------------------------------------------------------------
    #[test]
    fn fbd004_pass_lossless_roundtrip() {
        let t = vec![1u32, 2, 3, 4, 5];
        assert_eq!(
            verdict_from_bpe_roundtrip(&t, &t),
            BpeRoundtripVerdict::Pass
        );
    }

    #[test]
    fn fbd004_fail_token_drift() {
        let original = vec![1u32, 2, 3];
        let reencoded = vec![1u32, 2, 4];
        assert_eq!(
            verdict_from_bpe_roundtrip(&original, &reencoded),
            BpeRoundtripVerdict::Fail
        );
    }

    #[test]
    fn fbd004_fail_length_change() {
        let original = vec![1u32, 2];
        let reencoded = vec![1u32, 2, 3];
        assert_eq!(
            verdict_from_bpe_roundtrip(&original, &reencoded),
            BpeRoundtripVerdict::Fail
        );
    }

    #[test]
    fn fbd004_fail_empty() {
        assert_eq!(
            verdict_from_bpe_roundtrip(&[], &[]),
            BpeRoundtripVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: BD-005 SIMD dispatch equivalence (exact).
    // -------------------------------------------------------------------------
    #[test]
    fn fbd005_pass_bit_identical() {
        let v = vec![1.5_f32, 2.5];
        assert_eq!(
            verdict_from_simd_equivalence(&v, &v),
            SimdEquivalenceVerdict::Pass
        );
    }

    #[test]
    fn fbd005_fail_tiny_drift() {
        // Contract: tolerance == 0.0 — even 1 ULP fails.
        let simd = vec![1.5_f32 + f32::EPSILON];
        let scalar = vec![1.5_f32];
        assert_eq!(
            verdict_from_simd_equivalence(&simd, &scalar),
            SimdEquivalenceVerdict::Fail
        );
    }

    #[test]
    fn fbd005_fail_length_mismatch() {
        assert_eq!(
            verdict_from_simd_equivalence(&[1.0], &[1.0, 2.0]),
            SimdEquivalenceVerdict::Fail
        );
    }

    #[test]
    fn fbd005_fail_empty() {
        assert_eq!(
            verdict_from_simd_equivalence(&[], &[]),
            SimdEquivalenceVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — full healthy backend dispatch passes all 5.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_dispatch_passes_all_5() {
        // 7B model GEMM ≈ 1M elements → GPU.
        assert_eq!(
            verdict_from_gpu_dispatch(1_000_000, true),
            GpuDispatchVerdict::Pass
        );
        // Healthy English text.
        assert_eq!(
            verdict_from_garbage_oracle(0.05, 35, false),
            GarbageOracleVerdict::Pass
        );
        // Qwen2 head_dim=128 → bound = sqrt(128) ≈ 11.3.
        assert_eq!(
            verdict_from_qk_norm_bound(11.0, 128),
            QkNormBoundVerdict::Pass
        );
        // Lossless BPE roundtrip.
        let tokens = vec![100u32, 200, 300];
        assert_eq!(
            verdict_from_bpe_roundtrip(&tokens, &tokens),
            BpeRoundtripVerdict::Pass
        );
        // SIMD bit-identical.
        let v = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(
            verdict_from_simd_equivalence(&v, &v),
            SimdEquivalenceVerdict::Pass
        );
    }

    #[test]
    fn realistic_pre_fix_all_5_failures() {
        // 001: chose CPU on 1M elements.
        assert_eq!(
            verdict_from_gpu_dispatch(1_000_000, false),
            GpuDispatchVerdict::Fail
        );
        // 002: oracle missed "aaaaaaa".
        assert_eq!(
            verdict_from_garbage_oracle(0.95, 1, false),
            GarbageOracleVerdict::Fail
        );
        // 003: L2 normalization missing — score 200 on head_dim=64.
        assert_eq!(
            verdict_from_qk_norm_bound(200.0, 64),
            QkNormBoundVerdict::Fail
        );
        // 004: tokenizer dropped a token.
        let original = vec![1u32, 2, 3];
        let reencoded = vec![1u32, 3];
        assert_eq!(
            verdict_from_bpe_roundtrip(&original, &reencoded),
            BpeRoundtripVerdict::Fail
        );
        // 005: SIMD reduction order leaked an FMA.
        let simd = vec![1.5_f32];
        let scalar = vec![1.5_f32 + f32::EPSILON];
        assert_eq!(
            verdict_from_simd_equivalence(&simd, &scalar),
            SimdEquivalenceVerdict::Fail
        );
    }
}
