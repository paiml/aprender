// `apr-qa-chaos-v1` algorithm-level PARTIAL discharge for the 5
// chaos-engineering falsifiers (memory budget, graceful OOM, signal
// handling, batch-overwrite protection, disk exhaustion).
//
// Contract: `contracts/apr-qa-chaos-v1.yaml`.
// Refs: GH-434 (OOM on 57 GB models), GH-352 (apr pull RAM), GH-471
// (GPU hangs on MoE), GH-478 (per-layer dequant OOM 32B).
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Five pure decision predicates over inputs derived from a chaos run
// (peak RSS, exit code under ulimit, exit code under SIGINT, disk-full
// behavior). Live discharge is `/usr/bin/time -v` + `ulimit` + `kill
// -INT` shell harnesses; this module pins the predicates so future
// emergency-path rewrites cannot drift on the resource-bound semantics.

/// Memory budget multiplier per F-CHAOS-001:
/// `peak_rss_bytes < 3 * model_size_bytes + 512 MB overhead`.
pub const AC_CHAOS_RSS_MULTIPLIER: f64 = 3.0;

/// Memory overhead constant (512 MB).
pub const AC_CHAOS_RSS_OVERHEAD_BYTES: u64 = 512 * 1024 * 1024;

/// SIGINT exit code per Unix convention (128 + 2).
pub const AC_CHAOS_SIGINT_EXIT_CODE: i32 = 130;

/// SIGSEGV exit code (128 + 11) — the regression class for F-CHAOS-002.
pub const AC_CHAOS_SIGSEGV_EXIT_CODE: i32 = 139;

/// Maximum acceptable SIGINT response time per F-CHAOS-003.
pub const AC_CHAOS_SIGINT_MAX_RESPONSE_MS: u64 = 2_000;

/// Keywords that satisfy F-CHAOS-002's stderr requirement.
pub const AC_CHAOS_OOM_KEYWORDS: [&str; 3] = ["memory", "oom", "allocation"];

/// Keywords that satisfy F-CHAOS-005's stderr requirement.
pub const AC_CHAOS_DISK_KEYWORDS: [&str; 3] = ["disk", "space", "write"];

// =============================================================================
// F-CHAOS-001 — memory budget
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryBudgetVerdict {
    /// `peak_rss < 3 * model_size + 512 MB`.
    Pass,
    /// Peak RSS exceeds budget — unbounded allocation regression.
    Fail,
}

#[must_use]
pub fn verdict_from_memory_budget(peak_rss_bytes: u64, model_size_bytes: u64) -> MemoryBudgetVerdict {
    let budget = (model_size_bytes as f64 * AC_CHAOS_RSS_MULTIPLIER) as u64
        + AC_CHAOS_RSS_OVERHEAD_BYTES;
    if peak_rss_bytes < budget {
        MemoryBudgetVerdict::Pass
    } else {
        MemoryBudgetVerdict::Fail
    }
}

// =============================================================================
// F-CHAOS-002 — graceful OOM
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GracefulOomVerdict {
    /// Exit code is non-zero AND not SIGSEGV (139), AND stderr mentions
    /// memory/OOM/allocation.
    Pass,
    /// SIGSEGV crash, exit 0 (silent failure), or stderr without OOM signal.
    Fail,
}

#[must_use]
pub fn verdict_from_graceful_oom(exit_code: i32, stderr: &str) -> GracefulOomVerdict {
    if exit_code == AC_CHAOS_SIGSEGV_EXIT_CODE {
        return GracefulOomVerdict::Fail;
    }
    if exit_code == 0 {
        // Silent success on memory-limited run is a worse regression than
        // a segfault — partial output emitted as valid.
        return GracefulOomVerdict::Fail;
    }
    let lower = stderr.to_lowercase();
    for kw in AC_CHAOS_OOM_KEYWORDS {
        if lower.contains(kw) {
            return GracefulOomVerdict::Pass;
        }
    }
    GracefulOomVerdict::Fail
}

// =============================================================================
// F-CHAOS-003 — signal handling
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalHandlingVerdict {
    /// Exit 130 (SIGINT) AND response time < 2s AND no corrupt cache.
    Pass,
    /// Wrong exit code, slow response, or corrupt cache state.
    Fail,
}

#[must_use]
pub fn verdict_from_signal_handling(
    exit_code: i32,
    response_time_ms: u64,
    cache_corrupt: bool,
) -> SignalHandlingVerdict {
    if exit_code != AC_CHAOS_SIGINT_EXIT_CODE {
        return SignalHandlingVerdict::Fail;
    }
    if response_time_ms >= AC_CHAOS_SIGINT_MAX_RESPONSE_MS {
        return SignalHandlingVerdict::Fail;
    }
    if cache_corrupt {
        return SignalHandlingVerdict::Fail;
    }
    SignalHandlingVerdict::Pass
}

// =============================================================================
// F-CHAOS-004 — batch overwrite protection
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverwriteProtectionVerdict {
    /// `apr <cmd> -o <existing_file>` without `--force` exits non-zero.
    Pass,
    /// Silently overwrote (exit 0) — the regression class.
    Fail,
}

#[must_use]
pub fn verdict_from_overwrite_protection(
    output_file_existed: bool,
    force_flag: bool,
    exit_code: i32,
) -> OverwriteProtectionVerdict {
    if !output_file_existed {
        // No conflict possible.
        return OverwriteProtectionVerdict::Pass;
    }
    if force_flag {
        // User explicitly authorized overwrite.
        return OverwriteProtectionVerdict::Pass;
    }
    if exit_code == 0 {
        OverwriteProtectionVerdict::Fail
    } else {
        OverwriteProtectionVerdict::Pass
    }
}

// =============================================================================
// F-CHAOS-005 — disk exhaustion
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiskExhaustionVerdict {
    /// Full-disk write produces non-zero exit AND mentions disk/space/write
    /// in stderr.
    Pass,
    /// Silent success or exit without disk-related stderr.
    Fail,
}

#[must_use]
pub fn verdict_from_disk_exhaustion(exit_code: i32, stderr: &str) -> DiskExhaustionVerdict {
    if exit_code == 0 {
        return DiskExhaustionVerdict::Fail;
    }
    let lower = stderr.to_lowercase();
    for kw in AC_CHAOS_DISK_KEYWORDS {
        if lower.contains(kw) {
            return DiskExhaustionVerdict::Pass;
        }
    }
    DiskExhaustionVerdict::Fail
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONE_GB: u64 = 1024 * 1024 * 1024;
    const ONE_MB: u64 = 1024 * 1024;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_rss_multiplier_3() {
        assert!((AC_CHAOS_RSS_MULTIPLIER - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn provenance_overhead_512_mb() {
        assert_eq!(AC_CHAOS_RSS_OVERHEAD_BYTES, 512 * 1024 * 1024);
    }

    #[test]
    fn provenance_sigint_exit_130() {
        assert_eq!(AC_CHAOS_SIGINT_EXIT_CODE, 130);
    }

    #[test]
    fn provenance_sigsegv_exit_139() {
        assert_eq!(AC_CHAOS_SIGSEGV_EXIT_CODE, 139);
    }

    #[test]
    fn provenance_sigint_response_ms_2000() {
        assert_eq!(AC_CHAOS_SIGINT_MAX_RESPONSE_MS, 2_000);
    }

    // -------------------------------------------------------------------------
    // Section 2: F-CHAOS-001 memory budget.
    // -------------------------------------------------------------------------
    #[test]
    fn fc001_pass_well_under_budget() {
        // 1 GB model, used 2 GB peak (under 3 GB + 512 MB).
        assert_eq!(
            verdict_from_memory_budget(2 * ONE_GB, ONE_GB),
            MemoryBudgetVerdict::Pass
        );
    }

    #[test]
    fn fc001_pass_at_budget_minus_one() {
        // 1 GB model: budget = 3 GB + 512 MB. Just under.
        let budget = 3 * ONE_GB + 512 * ONE_MB;
        assert_eq!(
            verdict_from_memory_budget(budget - 1, ONE_GB),
            MemoryBudgetVerdict::Pass
        );
    }

    #[test]
    fn fc001_fail_at_budget() {
        // Strict less-than: equality fails.
        let budget = 3 * ONE_GB + 512 * ONE_MB;
        assert_eq!(
            verdict_from_memory_budget(budget, ONE_GB),
            MemoryBudgetVerdict::Fail
        );
    }

    #[test]
    fn fc001_fail_unbounded() {
        // 100x over budget — clearly leak.
        assert_eq!(
            verdict_from_memory_budget(50 * ONE_GB, ONE_GB),
            MemoryBudgetVerdict::Fail
        );
    }

    #[test]
    fn fc001_pass_7b_under_15gb() {
        // Contract example: `apr run 7B.gguf` uses < 15 GB RSS.
        let model_7b = 5 * ONE_GB; // q4_k_m roughly 5 GB
        let peak_15gb = 15 * ONE_GB - 1;
        assert_eq!(
            verdict_from_memory_budget(peak_15gb, model_7b),
            MemoryBudgetVerdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: F-CHAOS-002 graceful OOM.
    // -------------------------------------------------------------------------
    #[test]
    fn fc002_pass_oom_with_memory_word() {
        let v = verdict_from_graceful_oom(1, "Error: out of memory");
        assert_eq!(v, GracefulOomVerdict::Pass);
    }

    #[test]
    fn fc002_pass_oom_with_allocation_word() {
        let v = verdict_from_graceful_oom(1, "Error: failed allocation");
        assert_eq!(v, GracefulOomVerdict::Pass);
    }

    #[test]
    fn fc002_pass_oom_uppercase() {
        let v = verdict_from_graceful_oom(2, "OOM detected");
        assert_eq!(v, GracefulOomVerdict::Pass);
    }

    #[test]
    fn fc002_fail_sigsegv() {
        let v = verdict_from_graceful_oom(139, "any stderr");
        assert_eq!(v, GracefulOomVerdict::Fail);
    }

    #[test]
    fn fc002_fail_silent_zero_exit() {
        let v = verdict_from_graceful_oom(0, "memory exhausted");
        assert_eq!(v, GracefulOomVerdict::Fail);
    }

    #[test]
    fn fc002_fail_nonzero_no_oom_word() {
        let v = verdict_from_graceful_oom(1, "Error: file not found");
        assert_eq!(v, GracefulOomVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: F-CHAOS-003 signal handling.
    // -------------------------------------------------------------------------
    #[test]
    fn fc003_pass_clean_sigint() {
        let v = verdict_from_signal_handling(130, 500, false);
        assert_eq!(v, SignalHandlingVerdict::Pass);
    }

    #[test]
    fn fc003_pass_sigint_at_limit() {
        // Just under 2s.
        let v = verdict_from_signal_handling(130, 1_999, false);
        assert_eq!(v, SignalHandlingVerdict::Pass);
    }

    #[test]
    fn fc003_fail_wrong_exit_code() {
        let v = verdict_from_signal_handling(0, 500, false);
        assert_eq!(v, SignalHandlingVerdict::Fail);
    }

    #[test]
    fn fc003_fail_slow_response() {
        let v = verdict_from_signal_handling(130, 2_000, false);
        assert_eq!(v, SignalHandlingVerdict::Fail);
    }

    #[test]
    fn fc003_fail_corrupt_cache() {
        let v = verdict_from_signal_handling(130, 500, true);
        assert_eq!(v, SignalHandlingVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: F-CHAOS-004 batch overwrite protection.
    // -------------------------------------------------------------------------
    #[test]
    fn fc004_pass_no_existing_file() {
        let v = verdict_from_overwrite_protection(false, false, 0);
        assert_eq!(v, OverwriteProtectionVerdict::Pass);
    }

    #[test]
    fn fc004_pass_force_flag_set() {
        // User explicitly opted in.
        let v = verdict_from_overwrite_protection(true, true, 0);
        assert_eq!(v, OverwriteProtectionVerdict::Pass);
    }

    #[test]
    fn fc004_pass_existing_file_command_refused() {
        let v = verdict_from_overwrite_protection(true, false, 1);
        assert_eq!(v, OverwriteProtectionVerdict::Pass);
    }

    #[test]
    fn fc004_fail_silent_overwrite() {
        // The regression: file existed, no --force, but exit 0 = overwrite.
        let v = verdict_from_overwrite_protection(true, false, 0);
        assert_eq!(v, OverwriteProtectionVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: F-CHAOS-005 disk exhaustion.
    // -------------------------------------------------------------------------
    #[test]
    fn fc005_pass_disk_word() {
        let v = verdict_from_disk_exhaustion(1, "Error: no disk space");
        assert_eq!(v, DiskExhaustionVerdict::Pass);
    }

    #[test]
    fn fc005_pass_space_word() {
        let v = verdict_from_disk_exhaustion(1, "Error: no space left on device");
        assert_eq!(v, DiskExhaustionVerdict::Pass);
    }

    #[test]
    fn fc005_pass_write_word() {
        let v = verdict_from_disk_exhaustion(28, "Error: write failed");
        assert_eq!(v, DiskExhaustionVerdict::Pass);
    }

    #[test]
    fn fc005_fail_silent_zero_exit() {
        // The regression: partial file written silently.
        let v = verdict_from_disk_exhaustion(0, "Error: no space left");
        assert_eq!(v, DiskExhaustionVerdict::Fail);
    }

    #[test]
    fn fc005_fail_nonzero_no_disk_word() {
        let v = verdict_from_disk_exhaustion(1, "Error: tensor mismatch");
        assert_eq!(v, DiskExhaustionVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — full chaos run passes all 5.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_chaos_run_passes_all_5() {
        // 5 GB model under 16 GB RSS budget; OOM mentions memory; SIGINT
        // exit 130 in 100ms; no overwrite without --force; full-disk
        // produces "no space" error.
        assert_eq!(
            verdict_from_memory_budget(13 * ONE_GB, 5 * ONE_GB),
            MemoryBudgetVerdict::Pass
        );
        assert_eq!(
            verdict_from_graceful_oom(1, "out of memory"),
            GracefulOomVerdict::Pass
        );
        assert_eq!(
            verdict_from_signal_handling(130, 100, false),
            SignalHandlingVerdict::Pass
        );
        assert_eq!(
            verdict_from_overwrite_protection(true, false, 1),
            OverwriteProtectionVerdict::Pass
        );
        assert_eq!(
            verdict_from_disk_exhaustion(1, "no space left on device"),
            DiskExhaustionVerdict::Pass
        );
    }

    #[test]
    fn realistic_pre_fix_all_5_failures() {
        // Each gate's regression class.
        // 001: 100x over budget (memory leak).
        assert_eq!(
            verdict_from_memory_budget(500 * ONE_GB, 5 * ONE_GB),
            MemoryBudgetVerdict::Fail
        );
        // 002: SIGSEGV under ulimit (the contract test's failure mode).
        assert_eq!(
            verdict_from_graceful_oom(139, "ulimit reached"),
            GracefulOomVerdict::Fail
        );
        // 003: wrong exit (e.g., 0 — process kept running through SIGINT).
        assert_eq!(
            verdict_from_signal_handling(0, 500, false),
            SignalHandlingVerdict::Fail
        );
        // 004: silent overwrite.
        assert_eq!(
            verdict_from_overwrite_protection(true, false, 0),
            OverwriteProtectionVerdict::Fail
        );
        // 005: silent partial file on full disk.
        assert_eq!(
            verdict_from_disk_exhaustion(0, "wrote 0 bytes"),
            DiskExhaustionVerdict::Fail
        );
    }
}
