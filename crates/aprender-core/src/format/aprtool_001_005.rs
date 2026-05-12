// `apr-tool-*` family algorithm-level PARTIAL discharge for the 5
// falsification conditions shared by every `apr-tool-*-v1.yaml` contract.
//
// Contracts: `contracts/apr-tool-*.yaml` (21 contracts × 5 conditions =
// 105 falsifier instantiations).
//
// All apr-tool-* contracts (bashrs, ccpo, cohete, copia, decy, depyler,
// etc.) share an identical falsification schema. Each declares 5
// conditions describing a "healthy active-tool repo":
//
//   1. "gh repo view paiml/<tool> returns archived=true" (P0 — reclassify)
//   2. "README.md missing or empty" (P0 — create_readme)
//   3. "No CI workflow in .github/workflows/" (P1 — add_ci)
//   4. "cargo build fails (Rust) or equivalent build fails" (P0 — fix_build)
//   5. "Zero tests in repository" (P1 — add_tests)
//
// One verdict module satisfies the algorithm-level pin for all 21
// apr-tool contracts. Live discharge is `gh repo view` + filesystem
// inspection + per-language build. This module pins the predicates so
// future tool consolidations cannot drift on the shape of any of the 5
// invariants.
//
// Local IDs FALSIFY-APRTOOL-001..005 are synthesized to give each
// unkeyed YAML entry a stable handle.

/// Minimum CI-workflow files in `.github/workflows/`.
pub const AC_APRTOOL_MIN_CI_WORKFLOWS: u32 = 1;

/// Minimum test count per FALSIFY-APRTOOL-005.
pub const AC_APRTOOL_MIN_TESTS: u32 = 1;

// =============================================================================
// FALSIFY-APRTOOL-001 — repo is NOT archived
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotArchivedVerdict {
    /// Repo is live (archived == false).
    Pass,
    /// Repo is archived but still classified as active-tool (P0 reclassify).
    Fail,
}

#[must_use]
pub fn verdict_from_not_archived(archived: bool) -> NotArchivedVerdict {
    if archived {
        NotArchivedVerdict::Fail
    } else {
        NotArchivedVerdict::Pass
    }
}

// =============================================================================
// FALSIFY-APRTOOL-002 — README.md present and non-empty
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadmeVerdict {
    /// README.md exists AND has content.
    Pass,
    /// Missing OR empty (P0 create_readme).
    Fail,
}

#[must_use]
pub fn verdict_from_readme(readme_exists: bool, readme_size_bytes: u64) -> ReadmeVerdict {
    if !readme_exists {
        return ReadmeVerdict::Fail;
    }
    if readme_size_bytes == 0 {
        return ReadmeVerdict::Fail;
    }
    ReadmeVerdict::Pass
}

// =============================================================================
// FALSIFY-APRTOOL-003 — at least 1 CI workflow file
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CiWorkflowVerdict {
    /// .github/workflows/ contains ≥ 1 .yml/.yaml workflow.
    Pass,
    /// Empty or missing (P1 add_ci).
    Fail,
}

#[must_use]
pub fn verdict_from_ci_workflow(workflow_count: u32) -> CiWorkflowVerdict {
    if workflow_count >= AC_APRTOOL_MIN_CI_WORKFLOWS {
        CiWorkflowVerdict::Pass
    } else {
        CiWorkflowVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-APRTOOL-004 — build passes
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildPassesVerdict {
    /// `cargo build` (or per-language equivalent) exits 0.
    Pass,
    /// Build failed (P0 fix_build).
    Fail,
}

#[must_use]
pub fn verdict_from_build_passes(build_exit_code: i32) -> BuildPassesVerdict {
    if build_exit_code == 0 {
        BuildPassesVerdict::Pass
    } else {
        BuildPassesVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-APRTOOL-005 — at least 1 test
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HasTestsVerdict {
    /// Repository has ≥ 1 test (anywhere — `tests/`, `src/**/*::tests`, etc.).
    Pass,
    /// Zero tests (P1 add_tests).
    Fail,
}

#[must_use]
pub fn verdict_from_has_tests(test_count: u32) -> HasTestsVerdict {
    if test_count >= AC_APRTOOL_MIN_TESTS {
        HasTestsVerdict::Pass
    } else {
        HasTestsVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_min_ci_workflows_is_1() {
        assert_eq!(AC_APRTOOL_MIN_CI_WORKFLOWS, 1);
    }

    #[test]
    fn provenance_min_tests_is_1() {
        assert_eq!(AC_APRTOOL_MIN_TESTS, 1);
    }

    // -------------------------------------------------------------------------
    // Section 2: APRTOOL-001 not archived.
    // -------------------------------------------------------------------------
    #[test]
    fn at001_pass_repo_live() {
        assert_eq!(verdict_from_not_archived(false), NotArchivedVerdict::Pass);
    }

    #[test]
    fn at001_fail_repo_archived() {
        assert_eq!(verdict_from_not_archived(true), NotArchivedVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: APRTOOL-002 README.
    // -------------------------------------------------------------------------
    #[test]
    fn at002_pass_readme_present() {
        assert_eq!(verdict_from_readme(true, 1024), ReadmeVerdict::Pass);
    }

    #[test]
    fn at002_pass_readme_minimal() {
        // 1 byte still counts.
        assert_eq!(verdict_from_readme(true, 1), ReadmeVerdict::Pass);
    }

    #[test]
    fn at002_fail_no_readme() {
        assert_eq!(verdict_from_readme(false, 0), ReadmeVerdict::Fail);
    }

    #[test]
    fn at002_fail_empty_readme() {
        assert_eq!(verdict_from_readme(true, 0), ReadmeVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: APRTOOL-003 CI workflow.
    // -------------------------------------------------------------------------
    #[test]
    fn at003_pass_one_workflow() {
        assert_eq!(verdict_from_ci_workflow(1), CiWorkflowVerdict::Pass);
    }

    #[test]
    fn at003_pass_many_workflows() {
        assert_eq!(verdict_from_ci_workflow(10), CiWorkflowVerdict::Pass);
    }

    #[test]
    fn at003_fail_zero_workflows() {
        assert_eq!(verdict_from_ci_workflow(0), CiWorkflowVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: APRTOOL-004 build passes.
    // -------------------------------------------------------------------------
    #[test]
    fn at004_pass_build_clean() {
        assert_eq!(verdict_from_build_passes(0), BuildPassesVerdict::Pass);
    }

    #[test]
    fn at004_fail_build_compile_error() {
        assert_eq!(verdict_from_build_passes(1), BuildPassesVerdict::Fail);
    }

    #[test]
    fn at004_fail_build_panic() {
        assert_eq!(verdict_from_build_passes(101), BuildPassesVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: APRTOOL-005 has tests.
    // -------------------------------------------------------------------------
    #[test]
    fn at005_pass_one_test() {
        assert_eq!(verdict_from_has_tests(1), HasTestsVerdict::Pass);
    }

    #[test]
    fn at005_pass_many_tests() {
        assert_eq!(verdict_from_has_tests(500), HasTestsVerdict::Pass);
    }

    #[test]
    fn at005_fail_zero_tests() {
        assert_eq!(verdict_from_has_tests(0), HasTestsVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — full healthy active-tool repo passes all 5.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_tool_passes_all_5() {
        // bashrs / cohete / copia state: live, has README, CI present, builds
        // clean, has tests.
        assert_eq!(verdict_from_not_archived(false), NotArchivedVerdict::Pass);
        assert_eq!(verdict_from_readme(true, 4096), ReadmeVerdict::Pass);
        assert_eq!(verdict_from_ci_workflow(3), CiWorkflowVerdict::Pass);
        assert_eq!(verdict_from_build_passes(0), BuildPassesVerdict::Pass);
        assert_eq!(verdict_from_has_tests(125), HasTestsVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_5_failures() {
        // Each gate's regression class.
        assert_eq!(verdict_from_not_archived(true), NotArchivedVerdict::Fail);
        assert_eq!(verdict_from_readme(false, 0), ReadmeVerdict::Fail);
        assert_eq!(verdict_from_ci_workflow(0), CiWorkflowVerdict::Fail);
        assert_eq!(verdict_from_build_passes(1), BuildPassesVerdict::Fail);
        assert_eq!(verdict_from_has_tests(0), HasTestsVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 8: Family coverage — all 21 apr-tool-* contracts.
    // -------------------------------------------------------------------------
    #[test]
    fn family_coverage_uniform_schema() {
        // The 21 apr-tool-* contracts share this falsification schema.
        // Verdicts work regardless of the tool's identity or implementation language.
        let tools = [
            "bashrs", "ccpo", "cohete", "copia", "decy", "depyler",
            "forjar", "fueguia", "ggsa", "kit", "labrador", "letrar",
            "linguamatic", "lupa", "manodura", "molino", "pmat",
            "ramificar", "ruchy", "sembrar", "tomb",
        ];
        assert_eq!(tools.len(), 21);
        for t in tools {
            assert_eq!(
                verdict_from_not_archived(false),
                NotArchivedVerdict::Pass,
                "tool {t} live"
            );
        }
    }
}
