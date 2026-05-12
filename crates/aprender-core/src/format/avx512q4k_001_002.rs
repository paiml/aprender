// `avx512-q4k-v1` algorithm-level PARTIAL discharge for the 2 AVX-512
// Q4_K GEMV falsifiers (scalar equivalence within 1e-3, ≥1.5x AVX2
// throughput).
//
// Contract: `contracts/avx512-q4k-v1.yaml`.
// Refs: GPTQ (Frantar et al., 2023), QuIP# (Chee et al., 2023).
//
// ## Disambiguation
//
// `avx2-fma-dot-v1.yaml` (task #402) covers the AVX2 8-wide f32 dot
// product. This contract — avx512-q4k-v1 — covers the AVX-512 16-wide
// Q4_K-quantized GEMV (different SIMD width, different operand format).
// Module suffix `avx512q4k_` disambiguates from `dot_*`.

/// Tolerance budget for AVX-512 vs scalar reference per
/// `equations.C-AVX512-Q4K-001`.
pub const AC_AVX512_Q4K_TOLERANCE: f32 = 1.0e-3;

/// Minimum throughput speedup over AVX2 per `equations.C-AVX512-Q4K-002`.
pub const AC_AVX512_Q4K_MIN_SPEEDUP: f64 = 1.5;

/// Minimum input dimension for the speedup gate to apply.
pub const AC_AVX512_Q4K_SPEEDUP_MIN_DIM: usize = 1024;

/// Q4_K super-block layout per the contract description.
pub const AC_AVX512_Q4K_BLOCK_ELEMENTS: usize = 256;
pub const AC_AVX512_Q4K_BLOCK_BYTES: usize = 144;

// =============================================================================
// FALSIFY-AVX512-Q4K-001 — AVX-512 output matches scalar within 1e-3
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Avx512Q4kEquivalenceVerdict {
    /// max_i |avx512[i] - scalar[i]| < 1e-3.
    Pass,
    /// At least one element exceeds tolerance.
    Fail,
}

/// `avx512_output` and `scalar_output` are the SIMD and scalar reference
/// dequant+matmul outputs for the same Q4_K weight + f32 input.
#[must_use]
pub fn verdict_from_avx512_q4k_equivalence(
    avx512_output: &[f32],
    scalar_output: &[f32],
) -> Avx512Q4kEquivalenceVerdict {
    if avx512_output.len() != scalar_output.len() {
        return Avx512Q4kEquivalenceVerdict::Fail;
    }
    if avx512_output.is_empty() {
        return Avx512Q4kEquivalenceVerdict::Fail;
    }
    for (a, b) in avx512_output.iter().zip(scalar_output.iter()) {
        if (a - b).abs() >= AC_AVX512_Q4K_TOLERANCE {
            return Avx512Q4kEquivalenceVerdict::Fail;
        }
    }
    Avx512Q4kEquivalenceVerdict::Pass
}

// =============================================================================
// FALSIFY-AVX512-Q4K-002 — ≥1.5x AVX2 throughput at in_dim ≥ 1024
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Avx512Q4kSpeedupVerdict {
    /// in_dim ≥ 1024 ⇒ throughput(avx512) ≥ 1.5 * throughput(avx2).
    /// in_dim < 1024 ⇒ vacuous Pass (gate doesn't apply).
    Pass,
    /// At in_dim ≥ 1024 the speedup is below 1.5x — overhead regression.
    Fail,
}

#[must_use]
pub fn verdict_from_avx512_q4k_speedup(
    in_dim: usize,
    avx2_throughput: f64,
    avx512_throughput: f64,
) -> Avx512Q4kSpeedupVerdict {
    if in_dim < AC_AVX512_Q4K_SPEEDUP_MIN_DIM {
        return Avx512Q4kSpeedupVerdict::Pass;
    }
    if avx2_throughput <= 0.0 {
        // Can't compute speedup; treat as harness defect (Fail).
        return Avx512Q4kSpeedupVerdict::Fail;
    }
    let speedup = avx512_throughput / avx2_throughput;
    if speedup >= AC_AVX512_Q4K_MIN_SPEEDUP {
        Avx512Q4kSpeedupVerdict::Pass
    } else {
        Avx512Q4kSpeedupVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_tolerance_1e_neg3() {
        assert!((AC_AVX512_Q4K_TOLERANCE - 1.0e-3).abs() < f32::EPSILON);
    }

    #[test]
    fn provenance_min_speedup_1_5() {
        assert!((AC_AVX512_Q4K_MIN_SPEEDUP - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn provenance_speedup_min_dim_1024() {
        assert_eq!(AC_AVX512_Q4K_SPEEDUP_MIN_DIM, 1024);
    }

    #[test]
    fn provenance_super_block_256_elements() {
        assert_eq!(AC_AVX512_Q4K_BLOCK_ELEMENTS, 256);
    }

    #[test]
    fn provenance_super_block_144_bytes() {
        assert_eq!(AC_AVX512_Q4K_BLOCK_BYTES, 144);
    }

    // -------------------------------------------------------------------------
    // Section 2: AVX512-Q4K-001 equivalence.
    // -------------------------------------------------------------------------
    #[test]
    fn fa001_pass_exact_match() {
        let a = vec![1.0, 2.0, 3.0];
        let s = vec![1.0, 2.0, 3.0];
        assert_eq!(
            verdict_from_avx512_q4k_equivalence(&a, &s),
            Avx512Q4kEquivalenceVerdict::Pass
        );
    }

    #[test]
    fn fa001_pass_within_tolerance() {
        let a = vec![1.0001, 2.0002, 3.0003];
        let s = vec![1.0, 2.0, 3.0];
        assert_eq!(
            verdict_from_avx512_q4k_equivalence(&a, &s),
            Avx512Q4kEquivalenceVerdict::Pass
        );
    }

    #[test]
    fn fa001_fail_above_tolerance() {
        let a = vec![1.0, 2.0, 3.5]; // 0.5 > 1e-3
        let s = vec![1.0, 2.0, 3.0];
        assert_eq!(
            verdict_from_avx512_q4k_equivalence(&a, &s),
            Avx512Q4kEquivalenceVerdict::Fail
        );
    }

    #[test]
    fn fa001_fail_at_threshold() {
        // Strict less-than.
        let a = vec![1.001];
        let s = vec![1.0];
        assert_eq!(
            verdict_from_avx512_q4k_equivalence(&a, &s),
            Avx512Q4kEquivalenceVerdict::Fail
        );
    }

    #[test]
    fn fa001_fail_length_mismatch() {
        let a = vec![1.0, 2.0];
        let s = vec![1.0];
        assert_eq!(
            verdict_from_avx512_q4k_equivalence(&a, &s),
            Avx512Q4kEquivalenceVerdict::Fail
        );
    }

    #[test]
    fn fa001_fail_empty() {
        let a: Vec<f32> = vec![];
        let s: Vec<f32> = vec![];
        assert_eq!(
            verdict_from_avx512_q4k_equivalence(&a, &s),
            Avx512Q4kEquivalenceVerdict::Fail
        );
    }

    #[test]
    fn fa001_pass_super_block_256() {
        // 256-element super-block: all match within tolerance.
        let a: Vec<f32> = (0..256).map(|i| i as f32 + 1e-5).collect();
        let s: Vec<f32> = (0..256).map(|i| i as f32).collect();
        assert_eq!(
            verdict_from_avx512_q4k_equivalence(&a, &s),
            Avx512Q4kEquivalenceVerdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: AVX512-Q4K-002 throughput.
    // -------------------------------------------------------------------------
    #[test]
    fn fa002_pass_speedup_2x() {
        // AVX2 = 100, AVX-512 = 200 ⇒ 2x.
        assert_eq!(
            verdict_from_avx512_q4k_speedup(2048, 100.0, 200.0),
            Avx512Q4kSpeedupVerdict::Pass
        );
    }

    #[test]
    fn fa002_pass_speedup_at_1_5x() {
        assert_eq!(
            verdict_from_avx512_q4k_speedup(2048, 100.0, 150.0),
            Avx512Q4kSpeedupVerdict::Pass
        );
    }

    #[test]
    fn fa002_fail_speedup_below_1_5x() {
        // 1.4x — below threshold.
        assert_eq!(
            verdict_from_avx512_q4k_speedup(2048, 100.0, 140.0),
            Avx512Q4kSpeedupVerdict::Fail
        );
    }

    #[test]
    fn fa002_fail_no_speedup() {
        // Same throughput.
        assert_eq!(
            verdict_from_avx512_q4k_speedup(2048, 100.0, 100.0),
            Avx512Q4kSpeedupVerdict::Fail
        );
    }

    #[test]
    fn fa002_pass_small_dim_vacuous() {
        // in_dim < 1024 ⇒ gate doesn't apply.
        assert_eq!(
            verdict_from_avx512_q4k_speedup(512, 100.0, 100.0),
            Avx512Q4kSpeedupVerdict::Pass
        );
    }

    #[test]
    fn fa002_pass_at_min_dim() {
        // in_dim == 1024 with 1.5x.
        assert_eq!(
            verdict_from_avx512_q4k_speedup(1024, 100.0, 150.0),
            Avx512Q4kSpeedupVerdict::Pass
        );
    }

    #[test]
    fn fa002_fail_zero_avx2_throughput() {
        // Harness defect: AVX2 reported 0 tps.
        assert_eq!(
            verdict_from_avx512_q4k_speedup(2048, 0.0, 200.0),
            Avx512Q4kSpeedupVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: Realistic — full healthy AVX-512 Q4K kernel.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_kernel_passes_both() {
        // 256-element output, all within 1e-4 of scalar; speedup 1.7x at 2048.
        let a: Vec<f32> = (0..256).map(|i| i as f32 + 1e-4).collect();
        let s: Vec<f32> = (0..256).map(|i| i as f32).collect();
        assert_eq!(
            verdict_from_avx512_q4k_equivalence(&a, &s),
            Avx512Q4kEquivalenceVerdict::Pass
        );
        assert_eq!(
            verdict_from_avx512_q4k_speedup(2048, 100.0, 170.0),
            Avx512Q4kSpeedupVerdict::Pass
        );
    }

    #[test]
    fn realistic_pre_fix_both_failures() {
        // Equivalence: lane-mishandled element at index 64.
        let mut a: Vec<f32> = (0..256).map(|i| i as f32).collect();
        a[64] += 5.0; // wrong by 5.0
        let s: Vec<f32> = (0..256).map(|i| i as f32).collect();
        assert_eq!(
            verdict_from_avx512_q4k_equivalence(&a, &s),
            Avx512Q4kEquivalenceVerdict::Fail
        );
        // Speedup: zmm overhead regression — only 1.2x.
        assert_eq!(
            verdict_from_avx512_q4k_speedup(2048, 100.0, 120.0),
            Avx512Q4kSpeedupVerdict::Fail
        );
    }
}
