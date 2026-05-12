// SHIP-TWO-001 — `kernel-fusion-v1` algorithm-level PARTIAL discharge
// for FALSIFY-FUSION-001..003 (closes 3/3 sweep).
//
// Contract: `contracts/kernel-fusion-v1.yaml`.
// Spec: Kernel fusion decision contract with Poka-Yoke enforcement
// (PAR-077 lesson: Fused RMSNorm+Gate+Up+SwiGLU was 3× slower —
// shared memory overhead exceeded gain on small input vectors).

// ===========================================================================
// FUSION-001 — Registry completeness: every fused kernel has a contract entry
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionStatus { Active, Blocked, Planned }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fusion001Verdict { Pass, Fail }

/// Pass iff every fused kernel name in `kernel_files` has a corresponding
/// entry in `registry_kernel_names` (the contract's `fusion_decisions` map).
#[must_use]
pub fn verdict_from_registry_completeness(
    kernel_files: &[&str],
    registry_kernel_names: &[&str],
) -> Fusion001Verdict {
    if kernel_files.is_empty() { return Fusion001Verdict::Fail; }
    if registry_kernel_names.is_empty() { return Fusion001Verdict::Fail; }
    for &k in kernel_files {
        if !registry_kernel_names.contains(&k) {
            return Fusion001Verdict::Fail; // orphaned kernel
        }
    }
    Fusion001Verdict::Pass
}

// ===========================================================================
// FUSION-002 — BLOCKED entries: must have non-null unfused AND fused tok/s
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fusion002Verdict { Pass, Fail }

/// Pass iff for the given entry status, benchmark data is complete:
/// - BLOCKED: BOTH unfused_tps AND fused_tps must be Some(positive)
/// - ACTIVE: optional (may have unfused_tps=None for new fusions)
/// - PLANNED: no benchmark required yet
#[must_use]
pub fn verdict_from_blocked_benchmark_completeness(
    status: FusionStatus,
    unfused_tps: Option<f32>,
    fused_tps: Option<f32>,
) -> Fusion002Verdict {
    match status {
        FusionStatus::Blocked => match (unfused_tps, fused_tps) {
            (Some(u), Some(f)) if u > 0.0 && f > 0.0 && u.is_finite() && f.is_finite() => {
                Fusion002Verdict::Pass
            }
            _ => Fusion002Verdict::Fail,
        },
        FusionStatus::Active | FusionStatus::Planned => Fusion002Verdict::Pass,
    }
}

// ===========================================================================
// FUSION-003 — Performance gate: BLOCKED iff fused < unfused * 0.9
// ===========================================================================

pub const AC_FUSION_003_PERF_RATIO: f32 = 0.9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fusion003Verdict { Pass, Fail }

/// Decision rule: ACTIVE iff fused_tps ≥ unfused_tps * 0.9; BLOCKED otherwise.
/// Pass iff the contract's claimed status matches what the perf data implies.
#[must_use]
pub fn verdict_from_performance_gate(
    claimed_status: FusionStatus,
    unfused_tps: f32,
    fused_tps: f32,
) -> Fusion003Verdict {
    if !unfused_tps.is_finite() || !fused_tps.is_finite() { return Fusion003Verdict::Fail; }
    if unfused_tps <= 0.0 || fused_tps <= 0.0 { return Fusion003Verdict::Fail; }
    let threshold = unfused_tps * AC_FUSION_003_PERF_RATIO;
    let derived_status = if fused_tps >= threshold {
        FusionStatus::Active
    } else {
        FusionStatus::Blocked
    };
    match (claimed_status, derived_status) {
        (FusionStatus::Active, FusionStatus::Active) => Fusion003Verdict::Pass,
        (FusionStatus::Blocked, FusionStatus::Blocked) => Fusion003Verdict::Pass,
        // PLANNED entries don't have measured perf data yet, so this gate
        // only applies to ACTIVE/BLOCKED comparisons.
        (FusionStatus::Planned, _) => Fusion003Verdict::Fail,
        _ => Fusion003Verdict::Fail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // FUSION-001 (registry completeness)
    #[test] fn fusion001_pass_canonical() {
        let kernels = ["FusedSwigluKernel", "BatchedSwigluKernel"];
        let registry = ["FusedSwigluKernel", "BatchedSwigluKernel", "FusedRmsNormGateUpSwigluQ4KKernel"];
        assert_eq!(
            verdict_from_registry_completeness(&kernels, &registry),
            Fusion001Verdict::Pass
        );
    }
    #[test] fn fusion001_fail_orphaned_kernel() {
        // Kernel exists but contract entry missing.
        let kernels = ["FusedSwigluKernel", "OrphanedFusedKernel"];
        let registry = ["FusedSwigluKernel"];
        assert_eq!(
            verdict_from_registry_completeness(&kernels, &registry),
            Fusion001Verdict::Fail
        );
    }
    #[test] fn fusion001_fail_empty_kernels() {
        let registry = ["FusedSwigluKernel"];
        assert_eq!(
            verdict_from_registry_completeness(&[], &registry),
            Fusion001Verdict::Fail
        );
    }
    #[test] fn fusion001_fail_empty_registry() {
        let kernels = ["FusedSwigluKernel"];
        assert_eq!(
            verdict_from_registry_completeness(&kernels, &[]),
            Fusion001Verdict::Fail
        );
    }

    // FUSION-002 (BLOCKED benchmark completeness)
    #[test] fn fusion002_pass_blocked_complete() {
        // FUSION-003: unfused=80.6, fused=26.9.
        assert_eq!(
            verdict_from_blocked_benchmark_completeness(
                FusionStatus::Blocked,
                Some(80.6),
                Some(26.9),
            ),
            Fusion002Verdict::Pass
        );
    }
    #[test] fn fusion002_pass_active_partial() {
        // ACTIVE: benchmark optional (FUSION-001 has unfused=null in YAML).
        assert_eq!(
            verdict_from_blocked_benchmark_completeness(
                FusionStatus::Active,
                None,
                None,
            ),
            Fusion002Verdict::Pass
        );
    }
    #[test] fn fusion002_pass_planned_no_data() {
        // PLANNED: no benchmark yet (FUSION-004 dp4a is PLANNED with no data).
        assert_eq!(
            verdict_from_blocked_benchmark_completeness(
                FusionStatus::Planned,
                None,
                None,
            ),
            Fusion002Verdict::Pass
        );
    }
    #[test] fn fusion002_fail_blocked_missing_unfused() {
        assert_eq!(
            verdict_from_blocked_benchmark_completeness(
                FusionStatus::Blocked,
                None,
                Some(26.9),
            ),
            Fusion002Verdict::Fail
        );
    }
    #[test] fn fusion002_fail_blocked_missing_fused() {
        assert_eq!(
            verdict_from_blocked_benchmark_completeness(
                FusionStatus::Blocked,
                Some(80.6),
                None,
            ),
            Fusion002Verdict::Fail
        );
    }
    #[test] fn fusion002_fail_blocked_zero_tps() {
        assert_eq!(
            verdict_from_blocked_benchmark_completeness(
                FusionStatus::Blocked,
                Some(80.6),
                Some(0.0),
            ),
            Fusion002Verdict::Fail
        );
    }
    #[test] fn fusion002_fail_blocked_nan_tps() {
        assert_eq!(
            verdict_from_blocked_benchmark_completeness(
                FusionStatus::Blocked,
                Some(80.6),
                Some(f32::NAN),
            ),
            Fusion002Verdict::Fail
        );
    }

    // FUSION-003 (performance gate)
    #[test] fn fusion003_pass_blocked_3x_slower() {
        // FUSION-003 canonical: unfused=80.6, fused=26.9 → ratio 0.334 < 0.9.
        // Claimed BLOCKED matches derived BLOCKED → Pass.
        assert_eq!(
            verdict_from_performance_gate(FusionStatus::Blocked, 80.6, 26.9),
            Fusion003Verdict::Pass
        );
    }
    #[test] fn fusion003_pass_active_within_10_percent() {
        // 95% of unfused → ACTIVE.
        assert_eq!(
            verdict_from_performance_gate(FusionStatus::Active, 100.0, 95.0),
            Fusion003Verdict::Pass
        );
    }
    #[test] fn fusion003_pass_active_faster_than_unfused() {
        // 110% of unfused (real fusion gain) → ACTIVE.
        assert_eq!(
            verdict_from_performance_gate(FusionStatus::Active, 100.0, 110.0),
            Fusion003Verdict::Pass
        );
    }
    #[test] fn fusion003_fail_misclassified_as_active() {
        // 50% of unfused (much slower) but claimed ACTIVE → derived BLOCKED.
        assert_eq!(
            verdict_from_performance_gate(FusionStatus::Active, 100.0, 50.0),
            Fusion003Verdict::Fail
        );
    }
    #[test] fn fusion003_fail_misclassified_as_blocked() {
        // 95% of unfused (within 10%) but claimed BLOCKED → derived ACTIVE.
        assert_eq!(
            verdict_from_performance_gate(FusionStatus::Blocked, 100.0, 95.0),
            Fusion003Verdict::Fail
        );
    }
    #[test] fn fusion003_fail_planned_status() {
        // PLANNED entries can't be classified by perf data.
        assert_eq!(
            verdict_from_performance_gate(FusionStatus::Planned, 80.6, 26.9),
            Fusion003Verdict::Fail
        );
    }
    #[test] fn fusion003_fail_zero_unfused() {
        assert_eq!(
            verdict_from_performance_gate(FusionStatus::Active, 0.0, 100.0),
            Fusion003Verdict::Fail
        );
    }
    #[test] fn fusion003_fail_nan() {
        assert_eq!(
            verdict_from_performance_gate(FusionStatus::Active, f32::NAN, 100.0),
            Fusion003Verdict::Fail
        );
    }
    #[test] fn fusion003_pass_at_threshold_boundary() {
        // 90% of unfused (exactly at boundary) → ACTIVE.
        assert_eq!(
            verdict_from_performance_gate(FusionStatus::Active, 100.0, 90.0),
            Fusion003Verdict::Pass
        );
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_FUSION_003_PERF_RATIO - 0.9).abs() < 1e-9);
    }
}
