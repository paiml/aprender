// `ci-infra-v1` algorithm-level PARTIAL discharge for the 6 CI-
// infrastructure falsifiers (RC1-RC6 root-cause prevention).
//
// Contract: `contracts/ci-infra-v1.yaml`.
// Refs: docs/specifications/components/ci-infrastructure.md, RC1-RC6
// root-cause analyses.

/// Maximum acceptable test --exclude flags in ci.yml (per F-INFRA-005).
pub const AC_CIINFRA_MAX_EXCLUSIONS: u32 = 2;

// =============================================================================
// F-INFRA-001 — workflow_trigger_explicit
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CiTriggerExplicitVerdict {
    /// All push-triggered workflows include `branches:` filter.
    Pass,
    /// At least one bare `on: push:` (no filter) — RC1 phantom failures.
    Fail,
}

#[must_use]
pub fn verdict_from_ci_trigger_explicit(bare_push_workflow_count: u32) -> CiTriggerExplicitVerdict {
    if bare_push_workflow_count == 0 {
        CiTriggerExplicitVerdict::Pass
    } else {
        CiTriggerExplicitVerdict::Fail
    }
}

// =============================================================================
// F-INFRA-002 — platform_deps_gated
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CiPlatformDepsVerdict {
    /// All platform-specific deps under `[target.'cfg(...)'.dependencies]`.
    Pass,
    /// At least one platform dep in unconditional `[dependencies]` —
    /// RC2 nightly Windows failure.
    Fail,
}

#[must_use]
pub fn verdict_from_ci_platform_deps(unguarded_platform_dep_count: u32) -> CiPlatformDepsVerdict {
    if unguarded_platform_dep_count == 0 {
        CiPlatformDepsVerdict::Pass
    } else {
        CiPlatformDepsVerdict::Fail
    }
}

// =============================================================================
// F-INFRA-003 — external_repos_pinned
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CiExternalRepoPinVerdict {
    /// All external-repo checkouts pinned to tag (v*) or 40-char SHA.
    Pass,
    /// At least one floating reference (main/master/HEAD) — RC3
    /// cross-repo breakage.
    Fail,
}

#[must_use]
pub fn verdict_from_ci_external_repo_pin(floating_external_count: u32) -> CiExternalRepoPinVerdict {
    if floating_external_count == 0 {
        CiExternalRepoPinVerdict::Pass
    } else {
        CiExternalRepoPinVerdict::Fail
    }
}

// =============================================================================
// F-INFRA-004 — patches_tracked
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CiPatchesTrackedVerdict {
    /// Every `[patch.crates-io]` entry has GH-/issue/expiration comment.
    Pass,
    /// At least one untracked patch — RC4 floating cc dependency.
    Fail,
}

#[must_use]
pub fn verdict_from_ci_patches_tracked(untracked_patch_count: u32) -> CiPatchesTrackedVerdict {
    if untracked_patch_count == 0 {
        CiPatchesTrackedVerdict::Pass
    } else {
        CiPatchesTrackedVerdict::Fail
    }
}

// =============================================================================
// F-INFRA-005 — no_untested_exclusions
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CiNoExclusionsVerdict {
    /// `--exclude` flag count in ci.yml ≤ 2 (GPU-only acceptable).
    Pass,
    /// > 2 exclusions — RC5 untested code paths.
    Fail,
}

#[must_use]
pub fn verdict_from_ci_no_exclusions(exclude_flag_count: u32) -> CiNoExclusionsVerdict {
    if exclude_flag_count <= AC_CIINFRA_MAX_EXCLUSIONS {
        CiNoExclusionsVerdict::Pass
    } else {
        CiNoExclusionsVerdict::Fail
    }
}

// =============================================================================
// F-INFRA-006 — rustflags_consistent
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CiRustflagsConsistentVerdict {
    /// All workflows that set RUSTFLAGS use the same value (or only 0/1
    /// workflows set it).
    Pass,
    /// Multiple distinct RUSTFLAGS values — RC6 cascading fix cycles.
    Fail,
}

#[must_use]
pub fn verdict_from_ci_rustflags_consistent(distinct_rustflags_value_count: u32) -> CiRustflagsConsistentVerdict {
    if distinct_rustflags_value_count <= 1 {
        CiRustflagsConsistentVerdict::Pass
    } else {
        CiRustflagsConsistentVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_max_exclusions_2() {
        assert_eq!(AC_CIINFRA_MAX_EXCLUSIONS, 2);
    }

    // -------------------------------------------------------------------------
    // Section 2: F-INFRA-001 trigger explicit.
    // -------------------------------------------------------------------------
    #[test]
    fn fci001_pass_no_bare_push() {
        assert_eq!(
            verdict_from_ci_trigger_explicit(0),
            CiTriggerExplicitVerdict::Pass
        );
    }

    #[test]
    fn fci001_fail_one_bare_push() {
        assert_eq!(
            verdict_from_ci_trigger_explicit(1),
            CiTriggerExplicitVerdict::Fail
        );
    }

    #[test]
    fn fci001_fail_many_bare_push() {
        assert_eq!(
            verdict_from_ci_trigger_explicit(5),
            CiTriggerExplicitVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: F-INFRA-002 platform deps.
    // -------------------------------------------------------------------------
    #[test]
    fn fci002_pass_all_gated() {
        assert_eq!(
            verdict_from_ci_platform_deps(0),
            CiPlatformDepsVerdict::Pass
        );
    }

    #[test]
    fn fci002_fail_one_unguarded() {
        assert_eq!(
            verdict_from_ci_platform_deps(1),
            CiPlatformDepsVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: F-INFRA-003 external repos.
    // -------------------------------------------------------------------------
    #[test]
    fn fci003_pass_all_pinned() {
        assert_eq!(
            verdict_from_ci_external_repo_pin(0),
            CiExternalRepoPinVerdict::Pass
        );
    }

    #[test]
    fn fci003_fail_floating_main() {
        assert_eq!(
            verdict_from_ci_external_repo_pin(1),
            CiExternalRepoPinVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: F-INFRA-004 patches tracked.
    // -------------------------------------------------------------------------
    #[test]
    fn fci004_pass_no_untracked() {
        assert_eq!(
            verdict_from_ci_patches_tracked(0),
            CiPatchesTrackedVerdict::Pass
        );
    }

    #[test]
    fn fci004_fail_one_untracked() {
        assert_eq!(
            verdict_from_ci_patches_tracked(1),
            CiPatchesTrackedVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: F-INFRA-005 no untested exclusions.
    // -------------------------------------------------------------------------
    #[test]
    fn fci005_pass_zero_exclusions() {
        assert_eq!(
            verdict_from_ci_no_exclusions(0),
            CiNoExclusionsVerdict::Pass
        );
    }

    #[test]
    fn fci005_pass_at_threshold() {
        assert_eq!(
            verdict_from_ci_no_exclusions(2),
            CiNoExclusionsVerdict::Pass
        );
    }

    #[test]
    fn fci005_fail_three_exclusions() {
        assert_eq!(
            verdict_from_ci_no_exclusions(3),
            CiNoExclusionsVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: F-INFRA-006 RUSTFLAGS consistent.
    // -------------------------------------------------------------------------
    #[test]
    fn fci006_pass_zero_workflows_set_rustflags() {
        assert_eq!(
            verdict_from_ci_rustflags_consistent(0),
            CiRustflagsConsistentVerdict::Pass
        );
    }

    #[test]
    fn fci006_pass_one_consistent_value() {
        assert_eq!(
            verdict_from_ci_rustflags_consistent(1),
            CiRustflagsConsistentVerdict::Pass
        );
    }

    #[test]
    fn fci006_fail_two_distinct_values() {
        assert_eq!(
            verdict_from_ci_rustflags_consistent(2),
            CiRustflagsConsistentVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 8: Realistic — full healthy CI passes all 6.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_ci_passes_all_6() {
        // All RC1-RC6 root causes eliminated.
        assert_eq!(verdict_from_ci_trigger_explicit(0), CiTriggerExplicitVerdict::Pass);
        assert_eq!(verdict_from_ci_platform_deps(0), CiPlatformDepsVerdict::Pass);
        assert_eq!(verdict_from_ci_external_repo_pin(0), CiExternalRepoPinVerdict::Pass);
        assert_eq!(verdict_from_ci_patches_tracked(0), CiPatchesTrackedVerdict::Pass);
        assert_eq!(verdict_from_ci_no_exclusions(2), CiNoExclusionsVerdict::Pass);
        assert_eq!(
            verdict_from_ci_rustflags_consistent(1),
            CiRustflagsConsistentVerdict::Pass
        );
    }

    #[test]
    fn realistic_pre_fix_all_6_failures() {
        // RC1: bare push trigger.
        assert_eq!(verdict_from_ci_trigger_explicit(3), CiTriggerExplicitVerdict::Fail);
        // RC2: nix in unconditional [dependencies].
        assert_eq!(verdict_from_ci_platform_deps(1), CiPlatformDepsVerdict::Fail);
        // RC3: paiml/sibling pinned to main.
        assert_eq!(
            verdict_from_ci_external_repo_pin(2),
            CiExternalRepoPinVerdict::Fail
        );
        // RC4: untracked cc patch.
        assert_eq!(verdict_from_ci_patches_tracked(1), CiPatchesTrackedVerdict::Fail);
        // RC5: 5 crates excluded from CI.
        assert_eq!(verdict_from_ci_no_exclusions(5), CiNoExclusionsVerdict::Fail);
        // RC6: 3 distinct RUSTFLAGS across workflows.
        assert_eq!(
            verdict_from_ci_rustflags_consistent(3),
            CiRustflagsConsistentVerdict::Fail
        );
    }
}
