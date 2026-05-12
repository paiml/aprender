// `gpu-weight-residency-v1` algorithm-level PARTIAL discharge for
// FALSIFY-GWR-001..005.
//
// Contract: `contracts/gpu-weight-residency-v1.yaml`.
//
// Pure-Rust verdicts for the 5 falsification gates:
//   GWR-001: nvidia-smi shows model_bytes resident at server startup
//   GWR-002: ≥ 180 tok/s on Qwen2.5-1.5B Q4K (RTX 4090)
//   GWR-003: zero `cudaMemcpyHtoD` calls during steady-state inference
//   GWR-004: GPU output matches CPU within tolerance (token-id parity)
//   GWR-005: Grace Blackwell uses CU_MEM_ATTACH_GLOBAL eager allocation

/// Throughput floor for GWR-002 (RTX 4090, Qwen2.5-1.5B Q4_K_M).
pub const AC_GWR_MIN_TPS_RTX4090: f32 = 180.0;
/// Allowed slack between actual and declared model bytes (GWR-001),
/// to absorb CUDA context overhead. nvidia-smi can report slightly more
/// than the model's footprint due to kernels and runtime structures.
pub const AC_GWR_RESIDENCY_TOLERANCE_PCT: f32 = 10.0;
/// GWR-003 — zero per-inference HtoD transfers in steady state.
pub const AC_GWR_MAX_HTOD_PER_INFERENCE: u32 = 0;
/// GWR-005 — flag the contract requires for unified-memory eager alloc.
pub const AC_GWR_GRACE_ALLOC_FLAG: &str = "CU_MEM_ATTACH_GLOBAL";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GwrVerdict {
    Pass,
    Fail,
}

/// GWR-001: server startup residency.
///
/// Pass iff `observed_bytes >= model_bytes` AND
/// `observed_bytes <= model_bytes * (1 + AC_GWR_RESIDENCY_TOLERANCE_PCT/100)`.
#[must_use]
pub fn verdict_from_weight_residency(model_bytes: u64, observed_bytes: u64) -> GwrVerdict {
    if model_bytes == 0 {
        return GwrVerdict::Fail;
    }
    if observed_bytes < model_bytes {
        return GwrVerdict::Fail;
    }
    let upper = model_bytes
        .saturating_mul((100.0 + AC_GWR_RESIDENCY_TOLERANCE_PCT) as u64)
        / 100;
    if observed_bytes > upper {
        return GwrVerdict::Fail;
    }
    GwrVerdict::Pass
}

/// GWR-002: throughput floor (RTX 4090).
#[must_use]
pub fn verdict_from_throughput(observed_tps: f32) -> GwrVerdict {
    if !observed_tps.is_finite() || observed_tps <= 0.0 {
        return GwrVerdict::Fail;
    }
    if observed_tps >= AC_GWR_MIN_TPS_RTX4090 {
        GwrVerdict::Pass
    } else {
        GwrVerdict::Fail
    }
}

/// GWR-003: zero `cudaMemcpyHtoD` per inference in steady state.
#[must_use]
pub fn verdict_from_no_per_inference_htod(htod_count: u32) -> GwrVerdict {
    // Contract pins AC_GWR_MAX_HTOD_PER_INFERENCE = 0; equality check is
    // load-bearing, but the spec's "≤" is preserved in the constant name.
    if htod_count == AC_GWR_MAX_HTOD_PER_INFERENCE {
        GwrVerdict::Pass
    } else {
        GwrVerdict::Fail
    }
}

/// GWR-004: GPU output matches CPU output (token-id parity).
///
/// Pass iff slices have equal length, are non-empty, and match exactly.
#[must_use]
pub fn verdict_from_gpu_cpu_parity(gpu: &[u32], cpu: &[u32]) -> GwrVerdict {
    if gpu.is_empty() || cpu.is_empty() || gpu.len() != cpu.len() {
        return GwrVerdict::Fail;
    }
    if gpu == cpu {
        GwrVerdict::Pass
    } else {
        GwrVerdict::Fail
    }
}

/// GWR-005: Grace Blackwell eager-allocation flag.
#[must_use]
pub fn verdict_from_grace_alloc_flag(flag: &str) -> GwrVerdict {
    if flag == AC_GWR_GRACE_ALLOC_FLAG {
        GwrVerdict::Pass
    } else {
        GwrVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: Provenance pin.
    // -----------------------------------------------------------------
    #[test]
    fn provenance_min_tps_180() {
        assert_eq!(AC_GWR_MIN_TPS_RTX4090, 180.0);
    }

    #[test]
    fn provenance_residency_tolerance_10pct() {
        assert_eq!(AC_GWR_RESIDENCY_TOLERANCE_PCT, 10.0);
    }

    #[test]
    fn provenance_max_htod_zero() {
        assert_eq!(AC_GWR_MAX_HTOD_PER_INFERENCE, 0);
    }

    #[test]
    fn provenance_grace_flag_string() {
        assert_eq!(AC_GWR_GRACE_ALLOC_FLAG, "CU_MEM_ATTACH_GLOBAL");
    }

    // -----------------------------------------------------------------
    // Section 2: GWR-001 weight residency.
    // -----------------------------------------------------------------
    #[test]
    fn fgwr001_pass_exact_match() {
        let v = verdict_from_weight_residency(1_073_741_824, 1_073_741_824);
        assert_eq!(v, GwrVerdict::Pass);
    }

    #[test]
    fn fgwr001_pass_within_tolerance() {
        // model 1GB, observed 1.05GB (5% over) — OK
        let v = verdict_from_weight_residency(1_073_741_824, 1_127_428_915);
        assert_eq!(v, GwrVerdict::Pass);
    }

    #[test]
    fn fgwr001_fail_below_model_bytes() {
        // Under-allocation → Fail
        let v = verdict_from_weight_residency(1_073_741_824, 500_000_000);
        assert_eq!(v, GwrVerdict::Fail);
    }

    #[test]
    fn fgwr001_fail_over_tolerance() {
        // observed = 1.20× → > 10% tolerance
        let v = verdict_from_weight_residency(1_073_741_824, 1_288_490_188);
        assert_eq!(v, GwrVerdict::Fail);
    }

    #[test]
    fn fgwr001_fail_zero_model_bytes() {
        let v = verdict_from_weight_residency(0, 100);
        assert_eq!(v, GwrVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: GWR-002 throughput.
    // -----------------------------------------------------------------
    #[test]
    fn fgwr002_pass_at_threshold() {
        let v = verdict_from_throughput(180.0);
        assert_eq!(v, GwrVerdict::Pass);
    }

    #[test]
    fn fgwr002_pass_well_above() {
        let v = verdict_from_throughput(440.0);
        assert_eq!(v, GwrVerdict::Pass);
    }

    #[test]
    fn fgwr002_fail_just_under() {
        let v = verdict_from_throughput(179.9);
        assert_eq!(v, GwrVerdict::Fail);
    }

    #[test]
    fn fgwr002_fail_zero_or_negative() {
        let v = verdict_from_throughput(0.0);
        assert_eq!(v, GwrVerdict::Fail);
        let v = verdict_from_throughput(-100.0);
        assert_eq!(v, GwrVerdict::Fail);
    }

    #[test]
    fn fgwr002_fail_nan() {
        let v = verdict_from_throughput(f32::NAN);
        assert_eq!(v, GwrVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: GWR-003 zero HtoD per inference.
    // -----------------------------------------------------------------
    #[test]
    fn fgwr003_pass_zero_htod() {
        let v = verdict_from_no_per_inference_htod(0);
        assert_eq!(v, GwrVerdict::Pass);
    }

    #[test]
    fn fgwr003_fail_one_htod() {
        let v = verdict_from_no_per_inference_htod(1);
        assert_eq!(v, GwrVerdict::Fail);
    }

    #[test]
    fn fgwr003_fail_many_htod() {
        let v = verdict_from_no_per_inference_htod(64);
        assert_eq!(v, GwrVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 5: GWR-004 GPU/CPU parity.
    // -----------------------------------------------------------------
    #[test]
    fn fgwr004_pass_exact_match() {
        let v = verdict_from_gpu_cpu_parity(&[1, 2, 3, 4, 5], &[1, 2, 3, 4, 5]);
        assert_eq!(v, GwrVerdict::Pass);
    }

    #[test]
    fn fgwr004_fail_one_token_drift() {
        let v = verdict_from_gpu_cpu_parity(&[1, 2, 3, 4, 5], &[1, 2, 9, 4, 5]);
        assert_eq!(v, GwrVerdict::Fail);
    }

    #[test]
    fn fgwr004_fail_length_mismatch() {
        let v = verdict_from_gpu_cpu_parity(&[1, 2, 3], &[1, 2, 3, 4]);
        assert_eq!(v, GwrVerdict::Fail);
    }

    #[test]
    fn fgwr004_fail_empty() {
        let v = verdict_from_gpu_cpu_parity(&[], &[]);
        assert_eq!(v, GwrVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 6: GWR-005 Grace flag.
    // -----------------------------------------------------------------
    #[test]
    fn fgwr005_pass_correct_flag() {
        let v = verdict_from_grace_alloc_flag("CU_MEM_ATTACH_GLOBAL");
        assert_eq!(v, GwrVerdict::Pass);
    }

    #[test]
    fn fgwr005_fail_lazy_flag() {
        let v = verdict_from_grace_alloc_flag("CU_MEM_ATTACH_HOST");
        assert_eq!(v, GwrVerdict::Fail);
    }

    #[test]
    fn fgwr005_fail_empty() {
        let v = verdict_from_grace_alloc_flag("");
        assert_eq!(v, GwrVerdict::Fail);
    }

    #[test]
    fn fgwr005_fail_case_mismatch() {
        // Case-sensitive — exact CUDA flag string match.
        let v = verdict_from_grace_alloc_flag("cu_mem_attach_global");
        assert_eq!(v, GwrVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 7: Mutation survey + realistic.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_002_tps_around_threshold() {
        for tps_x10 in [1799_u32, 1800, 1801, 2000, 4400] {
            let tps = tps_x10 as f32 / 10.0;
            let v = verdict_from_throughput(tps);
            let want = if tps >= AC_GWR_MIN_TPS_RTX4090 {
                GwrVerdict::Pass
            } else {
                GwrVerdict::Fail
            };
            assert_eq!(v, want, "tps={tps}");
        }
    }

    #[test]
    fn mutation_survey_001_residency_tolerance_band() {
        let model = 1_000_000_u64;
        for pct in [0_u32, 5, 10, 11, 20, 50, 200] {
            let observed = model * (100 + pct as u64) / 100;
            let v = verdict_from_weight_residency(model, observed);
            let want = if pct <= 10 {
                GwrVerdict::Pass
            } else {
                GwrVerdict::Fail
            };
            assert_eq!(v, want, "pct={pct}");
        }
    }

    #[test]
    fn realistic_healthy_gpu_serve_passes_all_5() {
        // Qwen2.5-1.5B Q4K, RTX 4090 — apr serve --gpu canonical run.
        let model_bytes: u64 = 1_073_741_824; // 1GB approx
        let v1 = verdict_from_weight_residency(model_bytes, model_bytes + 50_000_000);
        let v2 = verdict_from_throughput(440.0); // measured 440.4 tok/s
        let v3 = verdict_from_no_per_inference_htod(0);
        let v4 = verdict_from_gpu_cpu_parity(&[1, 2, 3, 4, 5], &[1, 2, 3, 4, 5]);
        let v5 = verdict_from_grace_alloc_flag("CU_MEM_ATTACH_GLOBAL");
        assert_eq!(v1, GwrVerdict::Pass);
        assert_eq!(v2, GwrVerdict::Pass);
        assert_eq!(v3, GwrVerdict::Pass);
        assert_eq!(v4, GwrVerdict::Pass);
        assert_eq!(v5, GwrVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_5_failures() {
        // Pre-fix regressions:
        //   1: weights loaded on-demand → 0 bytes after startup
        //   2: 50 tok/s baseline before any residency fix (PCIe-bound)
        //   3: 64 HtoD per inference (re-uploading weights)
        //   4: GPU sampler drifted vs CPU greedy
        //   5: lazy unified-memory flag on Grace
        let v1 = verdict_from_weight_residency(1_000_000_000, 0);
        let v2 = verdict_from_throughput(50.0);
        let v3 = verdict_from_no_per_inference_htod(64);
        let v4 = verdict_from_gpu_cpu_parity(&[1, 2, 3, 4, 5], &[1, 2, 9, 4, 5]);
        let v5 = verdict_from_grace_alloc_flag("CU_MEM_ATTACH_HOST");
        assert_eq!(v1, GwrVerdict::Fail);
        assert_eq!(v2, GwrVerdict::Fail);
        assert_eq!(v3, GwrVerdict::Fail);
        assert_eq!(v4, GwrVerdict::Fail);
        assert_eq!(v5, GwrVerdict::Fail);
    }
}
