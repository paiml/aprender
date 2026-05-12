// SHIP-TWO-001 — `roofline-model-v1` algorithm-level PARTIAL
// discharge for FALSIFY-RM-001..005 (closes 5/5 sweep).
//
// Contract: `contracts/roofline-model-v1.yaml`.
// Spec: Roofline performance bound analysis for LLM inference
// (Williams et al. 2009; Qwen3 Performance Parity throughput analysis).
//
// NOTE: This roofline contract uses RM-* gate prefix; the
// metrics-regression-v1 contract (in earlier session bundle) uses
// the same prefix. Verdict function names disambiguate.

// ===========================================================================
// RM-001 — Positive ceilings: bw_ceiling > 0 AND compute_ceiling > 0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rm001Verdict { Pass, Fail }

#[must_use]
pub fn bandwidth_ceiling_tps(effective_bandwidth_gb_s: f64, model_bytes: u64) -> Option<f64> {
    if effective_bandwidth_gb_s <= 0.0 || !effective_bandwidth_gb_s.is_finite() {
        return None;
    }
    if model_bytes == 0 { return None; }
    let bytes_gb = (model_bytes as f64) / 1.0e9;
    if bytes_gb == 0.0 { return None; }
    let ceiling = effective_bandwidth_gb_s / bytes_gb;
    if !ceiling.is_finite() || ceiling <= 0.0 { return None; }
    Some(ceiling)
}

#[must_use]
pub fn compute_ceiling_tps(effective_gflops: f64, ops_per_token: u64) -> Option<f64> {
    if effective_gflops <= 0.0 || !effective_gflops.is_finite() { return None; }
    if ops_per_token == 0 { return None; }
    let ceiling = effective_gflops * 1.0e9 / (ops_per_token as f64);
    if !ceiling.is_finite() || ceiling <= 0.0 { return None; }
    Some(ceiling)
}

#[must_use]
pub fn verdict_from_positive_ceilings(
    effective_bandwidth_gb_s: f64,
    model_bytes: u64,
    effective_gflops: f64,
    ops_per_token: u64,
) -> Rm001Verdict {
    match (
        bandwidth_ceiling_tps(effective_bandwidth_gb_s, model_bytes),
        compute_ceiling_tps(effective_gflops, ops_per_token),
    ) {
        (Some(_), Some(_)) => Rm001Verdict::Pass,
        _ => Rm001Verdict::Fail,
    }
}

// ===========================================================================
// RM-002 — Memory-bound classification: bw_ceiling < compute_ceiling
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rm002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_memory_bound_classification(
    bw_ceiling: f64,
    compute_ceiling: f64,
    claimed_memory_bound: bool,
) -> Rm002Verdict {
    if !bw_ceiling.is_finite() || !compute_ceiling.is_finite() { return Rm002Verdict::Fail; }
    if bw_ceiling <= 0.0 || compute_ceiling <= 0.0 { return Rm002Verdict::Fail; }
    let actually_memory_bound = bw_ceiling < compute_ceiling;
    if actually_memory_bound == claimed_memory_bound { Rm002Verdict::Pass } else { Rm002Verdict::Fail }
}

// ===========================================================================
// RM-003 — Throughput bounded: throughput ≤ min(bw_ceiling, compute_ceiling)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rm003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_throughput_bounded(
    observed_throughput: f64,
    bw_ceiling: f64,
    compute_ceiling: f64,
) -> Rm003Verdict {
    if !observed_throughput.is_finite() { return Rm003Verdict::Fail; }
    if !bw_ceiling.is_finite() || !compute_ceiling.is_finite() { return Rm003Verdict::Fail; }
    if observed_throughput < 0.0 { return Rm003Verdict::Fail; }
    if bw_ceiling <= 0.0 || compute_ceiling <= 0.0 { return Rm003Verdict::Fail; }
    let upper = bw_ceiling.min(compute_ceiling);
    if observed_throughput <= upper { Rm003Verdict::Pass } else { Rm003Verdict::Fail }
}

// ===========================================================================
// RM-004 — Monotonicity: more params (same quant) → fewer bytes → lower ceiling
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rm004Verdict { Pass, Fail }

/// Pass iff `params_a < params_b` ⟹ `ceiling(a) > ceiling(b)` (with the
/// same bandwidth and bits-per-weight, doubling params halves ceiling).
#[must_use]
pub fn verdict_from_monotonicity(
    params_a: u64,
    ceiling_a: f64,
    params_b: u64,
    ceiling_b: f64,
) -> Rm004Verdict {
    if params_a == 0 || params_b == 0 { return Rm004Verdict::Fail; }
    if !ceiling_a.is_finite() || !ceiling_b.is_finite() { return Rm004Verdict::Fail; }
    if ceiling_a <= 0.0 || ceiling_b <= 0.0 { return Rm004Verdict::Fail; }
    let monotone = match params_a.cmp(&params_b) {
        std::cmp::Ordering::Less => ceiling_a > ceiling_b,
        std::cmp::Ordering::Greater => ceiling_a < ceiling_b,
        std::cmp::Ordering::Equal => (ceiling_a - ceiling_b).abs() < 1.0e-9,
    };
    if monotone { Rm004Verdict::Pass } else { Rm004Verdict::Fail }
}

// ===========================================================================
// RM-005 — SIMD roofline parity: byte-exact (contract tolerance=0.0)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rm005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_simd_roofline_parity(
    scalar_ceiling: f64,
    simd_ceiling: f64,
) -> Rm005Verdict {
    if !scalar_ceiling.is_finite() || !simd_ceiling.is_finite() { return Rm005Verdict::Fail; }
    if scalar_ceiling.to_bits() == simd_ceiling.to_bits() { Rm005Verdict::Pass } else { Rm005Verdict::Fail }
}

// ===========================================================================
// Helpers (in-module)
// ===========================================================================

#[must_use]
pub const fn model_bytes(total_params: u64, bits_per_weight: u8) -> u64 {
    if bits_per_weight == 0 { return 0; }
    total_params.saturating_mul(bits_per_weight as u64) / 8
}

#[cfg(test)]
mod tests {
    use super::*;

    // RM-001 (positive ceilings)
    #[test] fn rm001_pass_canonical() {
        // Qwen 1.5B Q4K: ~1GB model, 50 GB/s DDR5 bandwidth, ~3 GFLOP/token.
        assert_eq!(
            verdict_from_positive_ceilings(50.0, 1_000_000_000, 100.0, 3_000_000_000),
            Rm001Verdict::Pass
        );
    }
    #[test] fn rm001_fail_zero_bandwidth() {
        // Contract's stated falsifier: bandwidth = 0 → division by zero.
        assert_eq!(
            verdict_from_positive_ceilings(0.0, 1_000_000_000, 100.0, 3_000_000_000),
            Rm001Verdict::Fail
        );
    }
    #[test] fn rm001_fail_zero_model() {
        assert_eq!(
            verdict_from_positive_ceilings(50.0, 0, 100.0, 3_000_000_000),
            Rm001Verdict::Fail
        );
    }
    #[test] fn rm001_fail_zero_gflops() {
        assert_eq!(
            verdict_from_positive_ceilings(50.0, 1_000_000_000, 0.0, 3_000_000_000),
            Rm001Verdict::Fail
        );
    }
    #[test] fn rm001_fail_zero_ops() {
        assert_eq!(
            verdict_from_positive_ceilings(50.0, 1_000_000_000, 100.0, 0),
            Rm001Verdict::Fail
        );
    }
    #[test] fn rm001_fail_nan() {
        assert_eq!(
            verdict_from_positive_ceilings(f64::NAN, 1_000_000_000, 100.0, 3_000_000_000),
            Rm001Verdict::Fail
        );
    }

    // RM-002 (memory-bound classification)
    #[test] fn rm002_pass_memory_bound() {
        // bw=50, compute=200 → memory-bound, claimed=true.
        assert_eq!(verdict_from_memory_bound_classification(50.0, 200.0, true), Rm002Verdict::Pass);
    }
    #[test] fn rm002_pass_compute_bound() {
        // bw=200, compute=50 → compute-bound, claimed=false.
        assert_eq!(verdict_from_memory_bound_classification(200.0, 50.0, false), Rm002Verdict::Pass);
    }
    #[test] fn rm002_fail_misclassified() {
        // bw=50 < compute=200 means memory-bound; claimed=false is wrong.
        assert_eq!(verdict_from_memory_bound_classification(50.0, 200.0, false), Rm002Verdict::Fail);
    }
    #[test] fn rm002_fail_zero_ceiling() {
        assert_eq!(verdict_from_memory_bound_classification(0.0, 200.0, true), Rm002Verdict::Fail);
    }
    #[test] fn rm002_fail_nan() {
        assert_eq!(
            verdict_from_memory_bound_classification(f64::NAN, 200.0, true),
            Rm002Verdict::Fail
        );
    }

    // RM-003 (throughput bounded)
    #[test] fn rm003_pass_below_min() {
        // observed=80, bw=100, compute=200 → 80 ≤ min(100, 200) = 100.
        assert_eq!(verdict_from_throughput_bounded(80.0, 100.0, 200.0), Rm003Verdict::Pass);
    }
    #[test] fn rm003_pass_at_min() {
        assert_eq!(verdict_from_throughput_bounded(100.0, 100.0, 200.0), Rm003Verdict::Pass);
    }
    #[test] fn rm003_fail_above_min() {
        // observed=150 > min(100, 200) = 100 — violates bound.
        assert_eq!(verdict_from_throughput_bounded(150.0, 100.0, 200.0), Rm003Verdict::Fail);
    }
    #[test] fn rm003_fail_negative_throughput() {
        assert_eq!(verdict_from_throughput_bounded(-10.0, 100.0, 200.0), Rm003Verdict::Fail);
    }
    #[test] fn rm003_fail_nan() {
        assert_eq!(
            verdict_from_throughput_bounded(f64::NAN, 100.0, 200.0),
            Rm003Verdict::Fail
        );
    }

    // RM-004 (monotonicity)
    #[test] fn rm004_pass_double_params_half_ceiling() {
        // 1B params → 100 tok/s; 2B params → 50 tok/s.
        assert_eq!(verdict_from_monotonicity(1_000_000_000, 100.0, 2_000_000_000, 50.0), Rm004Verdict::Pass);
    }
    #[test] fn rm004_pass_smaller_higher() {
        // Same comparison, reversed args.
        assert_eq!(verdict_from_monotonicity(2_000_000_000, 50.0, 1_000_000_000, 100.0), Rm004Verdict::Pass);
    }
    #[test] fn rm004_pass_equal() {
        assert_eq!(verdict_from_monotonicity(1_000_000_000, 100.0, 1_000_000_000, 100.0), Rm004Verdict::Pass);
    }
    #[test] fn rm004_fail_non_monotonic() {
        // Bigger model claims HIGHER ceiling → physically impossible at same bandwidth.
        assert_eq!(verdict_from_monotonicity(1_000_000_000, 50.0, 2_000_000_000, 100.0), Rm004Verdict::Fail);
    }
    #[test] fn rm004_fail_zero_params() {
        assert_eq!(verdict_from_monotonicity(0, 100.0, 1_000_000_000, 50.0), Rm004Verdict::Fail);
    }

    // RM-005 (SIMD parity)
    #[test] fn rm005_pass_identical() {
        assert_eq!(verdict_from_simd_roofline_parity(100.5, 100.5), Rm005Verdict::Pass);
    }
    #[test] fn rm005_fail_one_ulp() {
        let s = 100.0_f64;
        let v = f64::from_bits(s.to_bits() + 1);
        assert_eq!(verdict_from_simd_roofline_parity(s, v), Rm005Verdict::Fail);
    }
    #[test] fn rm005_fail_nan() {
        assert_eq!(verdict_from_simd_roofline_parity(f64::NAN, 100.0), Rm005Verdict::Fail);
    }

    // Helper sanity
    #[test] fn model_bytes_canonical() {
        // 1B params Q4 = 0.5 GB.
        assert_eq!(model_bytes(1_000_000_000, 4), 500_000_000);
    }
    #[test] fn model_bytes_f16() {
        // 1B params F16 = 2 GB.
        assert_eq!(model_bytes(1_000_000_000, 16), 2_000_000_000);
    }
    #[test] fn model_bytes_zero_bits() {
        assert_eq!(model_bytes(1_000_000_000, 0), 0);
    }
    #[test] fn bandwidth_ceiling_canonical() {
        // 50 GB/s on 1 GB model → 50 tok/s.
        let c = bandwidth_ceiling_tps(50.0, 1_000_000_000).unwrap();
        assert!((c - 50.0).abs() < 1e-6);
    }
    #[test] fn compute_ceiling_canonical() {
        // 100 GFLOPS / 1 GFLOP per token = 100 tok/s.
        let c = compute_ceiling_tps(100.0, 1_000_000_000).unwrap();
        assert!((c - 100.0).abs() < 1e-6);
    }
}
