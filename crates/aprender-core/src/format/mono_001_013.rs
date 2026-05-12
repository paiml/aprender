// `cgp-monorepo-consolidation-v1` algorithm-level PARTIAL discharge for
// the 13 APR-MONO migration falsifiers (build perf, CI time, merge
// conflicts, publish time, broken-publish incidents, clone time, history,
// shim re-exports, version compat, registry compliance, single binary,
// flat layout, contract per subcommand).
//
// Contract: `contracts/cgp-monorepo-consolidation-v1.yaml`.
// Refs: Potvin & Levenberg (CACM 2016), Brousse (ICSE 2019), Brito et al.
// (MSR 2023), Rastogi et al. (ICSME 2023).
//
// ## Disambiguation
//
// `cgp-monorepo-build-v1.yaml` (task #417) is a sibling contract covering
// 7 BUILD invariants (workspace shape, naming, layout). This contract —
// cgp-monorepo-consolidation-v1 — covers 13 MIGRATION/RUNTIME invariants
// (perf budgets, history preservation, contract enforcement). Module
// suffix `mono_` disambiguates from the existing `monorepo_` module.

use std::collections::HashSet;

/// Incremental compile budget per FALSIFY-MONO-001 (3× regression bound).
pub const AC_MONO_INCR_COMPILE_REGRESSION_FACTOR: f64 = 3.0;

/// Pre-migration incremental compile baseline (5s).
pub const AC_MONO_INCR_COMPILE_BASELINE_S: f64 = 5.0;

/// CI time budget per FALSIFY-MONO-002 (10 min).
pub const AC_MONO_CI_TIME_BUDGET_S: u32 = 600;

/// Merge-conflict rolling-average threshold per FALSIFY-MONO-003.
pub const AC_MONO_MERGE_CONFLICT_THRESHOLD_PER_WEEK: u32 = 2;

/// Publish time budget per FALSIFY-MONO-004 (5 min).
pub const AC_MONO_PUBLISH_TIME_BUDGET_S: u32 = 300;

/// Broken-publish-incidents budget per FALSIFY-MONO-005 (90-day window).
pub const AC_MONO_BROKEN_PUBLISH_BUDGET: u32 = 2;

/// Clone-time budget per FALSIFY-MONO-006 (30 sec).
pub const AC_MONO_CLONE_TIME_BUDGET_S: u32 = 30;

// =============================================================================
// FALSIFY-MONO-001 — incremental compile within 3× of baseline
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonoIncrCompileVerdict {
    Pass,
    Fail,
}

#[must_use]
pub fn verdict_from_mono_incr_compile(measured_seconds: f64) -> MonoIncrCompileVerdict {
    if !measured_seconds.is_finite() || measured_seconds < 0.0 {
        return MonoIncrCompileVerdict::Fail;
    }
    let budget = AC_MONO_INCR_COMPILE_BASELINE_S * AC_MONO_INCR_COMPILE_REGRESSION_FACTOR;
    if measured_seconds <= budget {
        MonoIncrCompileVerdict::Pass
    } else {
        MonoIncrCompileVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-MONO-002 — CI time within 10 min
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonoCiTimeVerdict { Pass, Fail }

#[must_use]
pub fn verdict_from_mono_ci_time(seconds: u32) -> MonoCiTimeVerdict {
    if seconds <= AC_MONO_CI_TIME_BUDGET_S { MonoCiTimeVerdict::Pass } else { MonoCiTimeVerdict::Fail }
}

// =============================================================================
// FALSIFY-MONO-003 — merge conflicts ≤ 2/week rolling avg
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonoMergeConflictVerdict { Pass, Fail }

#[must_use]
pub fn verdict_from_mono_merge_conflict(conflicts_per_week: u32) -> MonoMergeConflictVerdict {
    if conflicts_per_week <= AC_MONO_MERGE_CONFLICT_THRESHOLD_PER_WEEK {
        MonoMergeConflictVerdict::Pass
    } else {
        MonoMergeConflictVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-MONO-004 — daily publish time ≤ 5 min
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonoPublishTimeVerdict { Pass, Fail }

#[must_use]
pub fn verdict_from_mono_publish_time(seconds: u32) -> MonoPublishTimeVerdict {
    if seconds <= AC_MONO_PUBLISH_TIME_BUDGET_S {
        MonoPublishTimeVerdict::Pass
    } else {
        MonoPublishTimeVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-MONO-005 — broken publishes ≤ 2 in 90 days
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonoBrokenPublishVerdict { Pass, Fail }

#[must_use]
pub fn verdict_from_mono_broken_publish(incidents_in_90d: u32) -> MonoBrokenPublishVerdict {
    if incidents_in_90d <= AC_MONO_BROKEN_PUBLISH_BUDGET {
        MonoBrokenPublishVerdict::Pass
    } else {
        MonoBrokenPublishVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-MONO-006 — clone time ≤ 30s
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonoCloneTimeVerdict { Pass, Fail }

#[must_use]
pub fn verdict_from_mono_clone_time(seconds: u32) -> MonoCloneTimeVerdict {
    if seconds <= AC_MONO_CLONE_TIME_BUDGET_S {
        MonoCloneTimeVerdict::Pass
    } else {
        MonoCloneTimeVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-MONO-007 — git subtree preserves history
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonoHistoryVerdict { Pass, Fail }

#[must_use]
pub fn verdict_from_mono_history(commits_from_original_repo: u32) -> MonoHistoryVerdict {
    if commits_from_original_repo > 0 {
        MonoHistoryVerdict::Pass
    } else {
        MonoHistoryVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-MONO-008 — shim crates re-export correctly
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonoShimReexportVerdict { Pass, Fail }

#[must_use]
pub fn verdict_from_mono_shim_reexport(
    shim_compiles: bool,
    test_results_match_legacy: bool,
) -> MonoShimReexportVerdict {
    if shim_compiles && test_results_match_legacy {
        MonoShimReexportVerdict::Pass
    } else {
        MonoShimReexportVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-MONO-009 — workspace version bump compat
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonoVersionBumpVerdict { Pass, Fail }

#[must_use]
pub fn verdict_from_mono_version_bump(
    is_major_bump: bool,
    breaking_api_changes: u32,
) -> MonoVersionBumpVerdict {
    if breaking_api_changes == 0 {
        return MonoVersionBumpVerdict::Pass;
    }
    if is_major_bump {
        MonoVersionBumpVerdict::Pass
    } else {
        MonoVersionBumpVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-MONO-010 — every crate name is in registry
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonoRegistryComplianceVerdict { Pass, Fail }

#[must_use]
pub fn verdict_from_mono_registry_compliance(
    workspace_names: &[&str],
    registered_names: &[&str],
) -> MonoRegistryComplianceVerdict {
    if workspace_names.is_empty() {
        return MonoRegistryComplianceVerdict::Fail;
    }
    let registry: HashSet<&&str> = registered_names.iter().collect();
    for name in workspace_names {
        if !registry.contains(name) {
            return MonoRegistryComplianceVerdict::Fail;
        }
    }
    MonoRegistryComplianceVerdict::Pass
}

// =============================================================================
// FALSIFY-MONO-011 — only apr-cli has [[bin]]
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonoSingleBinaryVerdict { Pass, Fail }

/// Allowed crates that may have [[bin]] sections (apr-cli + build tooling).
pub const AC_MONO_BIN_ALLOWED: [&str; 2] = ["apr-cli", "aprender-contracts-cli"];

#[must_use]
pub fn verdict_from_mono_single_binary(crates_with_bins: &[&str]) -> MonoSingleBinaryVerdict {
    for c in crates_with_bins {
        if !AC_MONO_BIN_ALLOWED.contains(c) {
            return MonoSingleBinaryVerdict::Fail;
        }
    }
    MonoSingleBinaryVerdict::Pass
}

// =============================================================================
// FALSIFY-MONO-012 — flat layout (depth ≤ 2 from repo root)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonoFlatLayoutVerdict { Pass, Fail }

#[must_use]
pub fn verdict_from_mono_flat_layout(member_paths: &[&str]) -> MonoFlatLayoutVerdict {
    if member_paths.is_empty() {
        return MonoFlatLayoutVerdict::Fail;
    }
    for path in member_paths {
        // Expected: `crates/<name>` (1 separator after "crates/").
        if !path.starts_with("crates/") {
            return MonoFlatLayoutVerdict::Fail;
        }
        let suffix = &path["crates/".len()..];
        if suffix.contains('/') {
            return MonoFlatLayoutVerdict::Fail;
        }
        if suffix.is_empty() {
            return MonoFlatLayoutVerdict::Fail;
        }
    }
    MonoFlatLayoutVerdict::Pass
}

// =============================================================================
// FALSIFY-MONO-013 — every apr subcommand has a contract
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonoSubcommandContractVerdict { Pass, Fail }

#[must_use]
pub fn verdict_from_mono_subcommand_contract(
    subcommands: &[&str],
    contract_names: &[&str],
) -> MonoSubcommandContractVerdict {
    if subcommands.is_empty() {
        return MonoSubcommandContractVerdict::Fail;
    }
    let contracts: HashSet<&&str> = contract_names.iter().collect();
    for cmd in subcommands {
        if !contracts.contains(cmd) {
            return MonoSubcommandContractVerdict::Fail;
        }
    }
    MonoSubcommandContractVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_incr_factor_3() {
        assert!((AC_MONO_INCR_COMPILE_REGRESSION_FACTOR - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn provenance_ci_budget_600s() {
        assert_eq!(AC_MONO_CI_TIME_BUDGET_S, 600);
    }

    #[test]
    fn provenance_publish_budget_300s() {
        assert_eq!(AC_MONO_PUBLISH_TIME_BUDGET_S, 300);
    }

    #[test]
    fn provenance_clone_budget_30s() {
        assert_eq!(AC_MONO_CLONE_TIME_BUDGET_S, 30);
    }

    #[test]
    fn provenance_bin_allowed_2() {
        assert_eq!(AC_MONO_BIN_ALLOWED.len(), 2);
    }

    // -------------------------------------------------------------------------
    // Section 2: MONO-001 incr compile.
    // -------------------------------------------------------------------------
    #[test]
    fn fm001_pass_under_budget() {
        assert_eq!(verdict_from_mono_incr_compile(10.0), MonoIncrCompileVerdict::Pass);
    }

    #[test]
    fn fm001_pass_at_budget() {
        assert_eq!(verdict_from_mono_incr_compile(15.0), MonoIncrCompileVerdict::Pass);
    }

    #[test]
    fn fm001_fail_over_budget() {
        assert_eq!(verdict_from_mono_incr_compile(20.0), MonoIncrCompileVerdict::Fail);
    }

    #[test]
    fn fm001_fail_nan() {
        assert_eq!(verdict_from_mono_incr_compile(f64::NAN), MonoIncrCompileVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: MONO-002..006 simple budget gates.
    // -------------------------------------------------------------------------
    #[test]
    fn fm002_pass_under_ci_budget() {
        assert_eq!(verdict_from_mono_ci_time(180), MonoCiTimeVerdict::Pass);
    }

    #[test]
    fn fm002_fail_over_ci_budget() {
        assert_eq!(verdict_from_mono_ci_time(700), MonoCiTimeVerdict::Fail);
    }

    #[test]
    fn fm003_pass_low_conflicts() {
        assert_eq!(verdict_from_mono_merge_conflict(1), MonoMergeConflictVerdict::Pass);
    }

    #[test]
    fn fm003_fail_high_conflicts() {
        assert_eq!(verdict_from_mono_merge_conflict(5), MonoMergeConflictVerdict::Fail);
    }

    #[test]
    fn fm004_pass_fast_publish() {
        assert_eq!(verdict_from_mono_publish_time(120), MonoPublishTimeVerdict::Pass);
    }

    #[test]
    fn fm004_fail_slow_publish() {
        assert_eq!(verdict_from_mono_publish_time(600), MonoPublishTimeVerdict::Fail);
    }

    #[test]
    fn fm005_pass_zero_incidents() {
        assert_eq!(verdict_from_mono_broken_publish(0), MonoBrokenPublishVerdict::Pass);
    }

    #[test]
    fn fm005_fail_many_incidents() {
        assert_eq!(verdict_from_mono_broken_publish(5), MonoBrokenPublishVerdict::Fail);
    }

    #[test]
    fn fm006_pass_fast_clone() {
        assert_eq!(verdict_from_mono_clone_time(15), MonoCloneTimeVerdict::Pass);
    }

    #[test]
    fn fm006_fail_slow_clone() {
        assert_eq!(verdict_from_mono_clone_time(60), MonoCloneTimeVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: MONO-007 history preservation.
    // -------------------------------------------------------------------------
    #[test]
    fn fm007_pass_history_preserved() {
        assert_eq!(verdict_from_mono_history(100), MonoHistoryVerdict::Pass);
    }

    #[test]
    fn fm007_fail_no_history() {
        assert_eq!(verdict_from_mono_history(0), MonoHistoryVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: MONO-008 shim re-exports.
    // -------------------------------------------------------------------------
    #[test]
    fn fm008_pass_shim_works() {
        assert_eq!(
            verdict_from_mono_shim_reexport(true, true),
            MonoShimReexportVerdict::Pass
        );
    }

    #[test]
    fn fm008_fail_shim_compile_error() {
        assert_eq!(
            verdict_from_mono_shim_reexport(false, true),
            MonoShimReexportVerdict::Fail
        );
    }

    #[test]
    fn fm008_fail_results_diverge() {
        assert_eq!(
            verdict_from_mono_shim_reexport(true, false),
            MonoShimReexportVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: MONO-009 version bump compat.
    // -------------------------------------------------------------------------
    #[test]
    fn fm009_pass_no_breaking_changes() {
        assert_eq!(
            verdict_from_mono_version_bump(false, 0),
            MonoVersionBumpVerdict::Pass
        );
    }

    #[test]
    fn fm009_pass_breaking_in_major_bump() {
        assert_eq!(
            verdict_from_mono_version_bump(true, 5),
            MonoVersionBumpVerdict::Pass
        );
    }

    #[test]
    fn fm009_fail_breaking_in_minor_bump() {
        assert_eq!(
            verdict_from_mono_version_bump(false, 1),
            MonoVersionBumpVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: MONO-010 registry compliance.
    // -------------------------------------------------------------------------
    #[test]
    fn fm010_pass_all_registered() {
        let workspace = ["aprender-core", "aprender-train"];
        let registry = ["aprender-core", "aprender-train", "aprender-extra"];
        assert_eq!(
            verdict_from_mono_registry_compliance(&workspace, &registry),
            MonoRegistryComplianceVerdict::Pass
        );
    }

    #[test]
    fn fm010_fail_unregistered() {
        let workspace = ["aprender-core", "rogue-crate"];
        let registry = ["aprender-core"];
        assert_eq!(
            verdict_from_mono_registry_compliance(&workspace, &registry),
            MonoRegistryComplianceVerdict::Fail
        );
    }

    #[test]
    fn fm010_fail_empty_workspace() {
        let registry = ["aprender-core"];
        assert_eq!(
            verdict_from_mono_registry_compliance(&[], &registry),
            MonoRegistryComplianceVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 8: MONO-011 single binary.
    // -------------------------------------------------------------------------
    #[test]
    fn fm011_pass_only_apr_cli_has_bin() {
        let bins = ["apr-cli"];
        assert_eq!(
            verdict_from_mono_single_binary(&bins),
            MonoSingleBinaryVerdict::Pass
        );
    }

    #[test]
    fn fm011_pass_apr_cli_plus_contracts_tooling() {
        let bins = ["apr-cli", "aprender-contracts-cli"];
        assert_eq!(
            verdict_from_mono_single_binary(&bins),
            MonoSingleBinaryVerdict::Pass
        );
    }

    #[test]
    fn fm011_fail_unauthorized_binary() {
        let bins = ["apr-cli", "rogue-bin"];
        assert_eq!(
            verdict_from_mono_single_binary(&bins),
            MonoSingleBinaryVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 9: MONO-012 flat layout.
    // -------------------------------------------------------------------------
    #[test]
    fn fm012_pass_flat() {
        let p = ["crates/aprender-core", "crates/aprender-train"];
        assert_eq!(verdict_from_mono_flat_layout(&p), MonoFlatLayoutVerdict::Pass);
    }

    #[test]
    fn fm012_fail_nested() {
        let p = ["crates/aprender-core", "crates/parent/child"];
        assert_eq!(verdict_from_mono_flat_layout(&p), MonoFlatLayoutVerdict::Fail);
    }

    #[test]
    fn fm012_fail_outside_crates() {
        let p = ["src/lib"];
        assert_eq!(verdict_from_mono_flat_layout(&p), MonoFlatLayoutVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 10: MONO-013 contract per subcommand.
    // -------------------------------------------------------------------------
    #[test]
    fn fm013_pass_all_contracts_present() {
        let cmds = ["run", "serve", "chat"];
        let contracts = ["run", "serve", "chat", "extra"];
        assert_eq!(
            verdict_from_mono_subcommand_contract(&cmds, &contracts),
            MonoSubcommandContractVerdict::Pass
        );
    }

    #[test]
    fn fm013_fail_missing_contract() {
        let cmds = ["run", "new-cmd"];
        let contracts = ["run"];
        assert_eq!(
            verdict_from_mono_subcommand_contract(&cmds, &contracts),
            MonoSubcommandContractVerdict::Fail
        );
    }

    #[test]
    fn fm013_fail_empty_subcommands() {
        let contracts = ["run"];
        assert_eq!(
            verdict_from_mono_subcommand_contract(&[], &contracts),
            MonoSubcommandContractVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 11: Realistic — full healthy migration passes all 13.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_migration_passes_all_13() {
        assert_eq!(verdict_from_mono_incr_compile(8.0), MonoIncrCompileVerdict::Pass);
        assert_eq!(verdict_from_mono_ci_time(240), MonoCiTimeVerdict::Pass);
        assert_eq!(verdict_from_mono_merge_conflict(1), MonoMergeConflictVerdict::Pass);
        assert_eq!(verdict_from_mono_publish_time(150), MonoPublishTimeVerdict::Pass);
        assert_eq!(verdict_from_mono_broken_publish(0), MonoBrokenPublishVerdict::Pass);
        assert_eq!(verdict_from_mono_clone_time(20), MonoCloneTimeVerdict::Pass);
        assert_eq!(verdict_from_mono_history(50), MonoHistoryVerdict::Pass);
        assert_eq!(
            verdict_from_mono_shim_reexport(true, true),
            MonoShimReexportVerdict::Pass
        );
        assert_eq!(
            verdict_from_mono_version_bump(false, 0),
            MonoVersionBumpVerdict::Pass
        );
        let workspace = ["aprender-core"];
        let registry = ["aprender-core", "aprender-train"];
        assert_eq!(
            verdict_from_mono_registry_compliance(&workspace, &registry),
            MonoRegistryComplianceVerdict::Pass
        );
        assert_eq!(
            verdict_from_mono_single_binary(&["apr-cli"]),
            MonoSingleBinaryVerdict::Pass
        );
        let p = ["crates/aprender-core"];
        assert_eq!(verdict_from_mono_flat_layout(&p), MonoFlatLayoutVerdict::Pass);
        assert_eq!(
            verdict_from_mono_subcommand_contract(&["run"], &["run", "serve"]),
            MonoSubcommandContractVerdict::Pass
        );
    }

    #[test]
    fn realistic_pre_fix_all_13_failures() {
        assert_eq!(verdict_from_mono_incr_compile(60.0), MonoIncrCompileVerdict::Fail);
        assert_eq!(verdict_from_mono_ci_time(900), MonoCiTimeVerdict::Fail);
        assert_eq!(verdict_from_mono_merge_conflict(10), MonoMergeConflictVerdict::Fail);
        assert_eq!(verdict_from_mono_publish_time(600), MonoPublishTimeVerdict::Fail);
        assert_eq!(verdict_from_mono_broken_publish(20), MonoBrokenPublishVerdict::Fail);
        assert_eq!(verdict_from_mono_clone_time(120), MonoCloneTimeVerdict::Fail);
        assert_eq!(verdict_from_mono_history(0), MonoHistoryVerdict::Fail);
        assert_eq!(
            verdict_from_mono_shim_reexport(false, false),
            MonoShimReexportVerdict::Fail
        );
        assert_eq!(
            verdict_from_mono_version_bump(false, 5),
            MonoVersionBumpVerdict::Fail
        );
        assert_eq!(
            verdict_from_mono_registry_compliance(&["rogue"], &["aprender-core"]),
            MonoRegistryComplianceVerdict::Fail
        );
        assert_eq!(
            verdict_from_mono_single_binary(&["rogue-bin"]),
            MonoSingleBinaryVerdict::Fail
        );
        assert_eq!(
            verdict_from_mono_flat_layout(&["crates/parent/child"]),
            MonoFlatLayoutVerdict::Fail
        );
        assert_eq!(
            verdict_from_mono_subcommand_contract(&["new-cmd"], &["run"]),
            MonoSubcommandContractVerdict::Fail
        );
    }
}
