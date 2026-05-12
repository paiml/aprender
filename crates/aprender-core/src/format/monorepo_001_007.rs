// `cgp-monorepo-build-v1` algorithm-level PARTIAL discharge for the 7
// monorepo build-verification falsifiers (no duplicates, version
// consistency, ≥60 members, no [patch.crates-io], all dirs have
// Cargo.toml, aprender-* naming, flat layout).
//
// Contract: `contracts/cgp-monorepo-build-v1.yaml`.
// Refs: Potvin & Levenberg (CACM 2016), Rastogi et al. (ICSME 2023),
// Burn/Nushell flat-layout monorepos.

use std::collections::HashSet;

/// Minimum workspace member count per FALSIFY-BUILD-003.
pub const AC_MONOREPO_MIN_MEMBERS: usize = 60;

/// Workspace version per metadata (0.29.0 at contract creation,
/// pinned for the FALSIFY-BUILD-002 gate).
pub const AC_MONOREPO_WORKSPACE_VERSION: &str = "0.29.0";

/// Crate names exempt from the aprender-* naming convention.
/// `aprender` is the root facade (cargo install aprender → apr binary);
/// `apr-cli` is the internal CLI logic crate per the contract description.
pub const AC_MONOREPO_NAME_EXEMPTIONS: [&str; 2] = ["aprender", "apr-cli"];

/// Legacy crate names that MUST NOT appear as workspace [package] names.
pub const AC_MONOREPO_FORBIDDEN_NAMES: [&str; 4] =
    ["trueno", "realizar", "entrenar", "batuta"];

// =============================================================================
// FALSIFY-BUILD-001 — no duplicate workspace members
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoDuplicatesVerdict {
    /// All [package] names are unique.
    Pass,
    /// At least two members share a name — merge conflict.
    Fail,
}

#[must_use]
pub fn verdict_from_no_duplicates(member_names: &[&str]) -> NoDuplicatesVerdict {
    if member_names.is_empty() {
        return NoDuplicatesVerdict::Fail;
    }
    let mut seen: HashSet<&&str> = HashSet::new();
    for n in member_names {
        if !seen.insert(n) {
            return NoDuplicatesVerdict::Fail;
        }
    }
    NoDuplicatesVerdict::Pass
}

// =============================================================================
// FALSIFY-BUILD-002 — workspace version consistency
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionConsistencyVerdict {
    /// All members using `version.workspace = true` resolve to 0.29.0.
    Pass,
    /// Workspace version mismatch — root Cargo.toml drift.
    Fail,
}

#[must_use]
pub fn verdict_from_version_consistency(resolved_workspace_version: &str) -> VersionConsistencyVerdict {
    if resolved_workspace_version == AC_MONOREPO_WORKSPACE_VERSION {
        VersionConsistencyVerdict::Pass
    } else {
        VersionConsistencyVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-BUILD-003 — workspace member count ≥ 60
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinMemberCountVerdict {
    /// `cargo metadata --workspace` reports ≥ 60 members.
    Pass,
    /// Below threshold — crates accidentally excluded.
    Fail,
}

#[must_use]
pub fn verdict_from_min_member_count(actual_count: usize) -> MinMemberCountVerdict {
    if actual_count >= AC_MONOREPO_MIN_MEMBERS {
        MinMemberCountVerdict::Pass
    } else {
        MinMemberCountVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-BUILD-004 — no [patch.crates-io] in root Cargo.toml
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoPatchVerdict {
    /// Root Cargo.toml does not contain `[patch.crates-io]` section.
    Pass,
    /// Patch section leaked into root.
    Fail,
}

#[must_use]
pub fn verdict_from_no_patch(root_cargo_toml: &str) -> NoPatchVerdict {
    if root_cargo_toml.contains("[patch.crates-io]") {
        NoPatchVerdict::Fail
    } else {
        NoPatchVerdict::Pass
    }
}

// =============================================================================
// FALSIFY-BUILD-005 — all workspace dirs have Cargo.toml
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllHaveCargoTomlVerdict {
    /// Every workspace member dir contains a Cargo.toml file.
    Pass,
    /// At least one is missing — broken subtree merge.
    Fail,
}

/// `(dir_name, has_cargo_toml)` per workspace member directory.
#[must_use]
pub fn verdict_from_all_have_cargo_toml(member_dirs: &[(&str, bool)]) -> AllHaveCargoTomlVerdict {
    if member_dirs.is_empty() {
        return AllHaveCargoTomlVerdict::Fail;
    }
    for (_dir, has_cargo) in member_dirs {
        if !*has_cargo {
            return AllHaveCargoTomlVerdict::Fail;
        }
    }
    AllHaveCargoTomlVerdict::Pass
}

// =============================================================================
// FALSIFY-BUILD-006 — all crate names use aprender-* prefix (or apr-cli)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AprPrefixVerdict {
    /// Every package name is "apr-cli" OR starts with "aprender-",
    /// AND no member uses a forbidden legacy name.
    Pass,
    /// Some package violates the naming convention.
    Fail,
}

#[must_use]
pub fn verdict_from_apr_prefix(member_names: &[&str]) -> AprPrefixVerdict {
    if member_names.is_empty() {
        return AprPrefixVerdict::Fail;
    }
    for name in member_names {
        if AC_MONOREPO_FORBIDDEN_NAMES.contains(name) {
            return AprPrefixVerdict::Fail;
        }
        if AC_MONOREPO_NAME_EXEMPTIONS.contains(name) {
            continue;
        }
        if !name.starts_with("aprender-") {
            return AprPrefixVerdict::Fail;
        }
    }
    AprPrefixVerdict::Pass
}

// =============================================================================
// FALSIFY-BUILD-007 — flat layout (no nested crates)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlatLayoutVerdict {
    /// Every Cargo.toml path is `crates/<name>/Cargo.toml` (depth 2 from
    /// crates/, not deeper).
    Pass,
    /// Nested Cargo.toml found — flat-layout invariant violated.
    Fail,
}

#[must_use]
pub fn verdict_from_flat_layout(cargo_toml_paths: &[&str]) -> FlatLayoutVerdict {
    if cargo_toml_paths.is_empty() {
        return FlatLayoutVerdict::Fail;
    }
    for path in cargo_toml_paths {
        if !path.starts_with("crates/") {
            return FlatLayoutVerdict::Fail;
        }
        let relative = &path["crates/".len()..];
        // Expected: `<name>/Cargo.toml` (one slash). Anything deeper = nested.
        let slashes = relative.matches('/').count();
        if slashes != 1 {
            return FlatLayoutVerdict::Fail;
        }
        if !path.ends_with("/Cargo.toml") {
            return FlatLayoutVerdict::Fail;
        }
    }
    FlatLayoutVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_min_members_60() {
        assert_eq!(AC_MONOREPO_MIN_MEMBERS, 60);
    }

    #[test]
    fn provenance_workspace_version_0_29() {
        assert_eq!(AC_MONOREPO_WORKSPACE_VERSION, "0.29.0");
    }

    #[test]
    fn provenance_naming_exemptions_contains_apr_cli() {
        assert!(AC_MONOREPO_NAME_EXEMPTIONS.contains(&"apr-cli"));
    }

    #[test]
    fn provenance_forbidden_names_count_4() {
        assert_eq!(AC_MONOREPO_FORBIDDEN_NAMES.len(), 4);
    }

    // -------------------------------------------------------------------------
    // Section 2: BUILD-001 no duplicates.
    // -------------------------------------------------------------------------
    #[test]
    fn fb001_pass_unique_members() {
        let m = ["aprender", "aprender-core", "apr-cli"];
        assert_eq!(verdict_from_no_duplicates(&m), NoDuplicatesVerdict::Pass);
    }

    #[test]
    fn fb001_fail_duplicate() {
        let m = ["aprender-core", "aprender-train", "aprender-core"];
        assert_eq!(verdict_from_no_duplicates(&m), NoDuplicatesVerdict::Fail);
    }

    #[test]
    fn fb001_fail_empty() {
        assert_eq!(
            verdict_from_no_duplicates(&[]),
            NoDuplicatesVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: BUILD-002 version consistency.
    // -------------------------------------------------------------------------
    #[test]
    fn fb002_pass_canonical_version() {
        assert_eq!(
            verdict_from_version_consistency("0.29.0"),
            VersionConsistencyVerdict::Pass
        );
    }

    #[test]
    fn fb002_fail_old_version() {
        assert_eq!(
            verdict_from_version_consistency("0.28.0"),
            VersionConsistencyVerdict::Fail
        );
    }

    #[test]
    fn fb002_fail_future_version() {
        assert_eq!(
            verdict_from_version_consistency("0.30.0"),
            VersionConsistencyVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: BUILD-003 minimum member count.
    // -------------------------------------------------------------------------
    #[test]
    fn fb003_pass_at_threshold() {
        assert_eq!(verdict_from_min_member_count(60), MinMemberCountVerdict::Pass);
    }

    #[test]
    fn fb003_pass_above_threshold() {
        assert_eq!(verdict_from_min_member_count(70), MinMemberCountVerdict::Pass);
    }

    #[test]
    fn fb003_fail_below_threshold() {
        assert_eq!(verdict_from_min_member_count(59), MinMemberCountVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: BUILD-004 no [patch.crates-io].
    // -------------------------------------------------------------------------
    #[test]
    fn fb004_pass_clean_root() {
        let toml = "[workspace]\nmembers = [\"crates/*\"]";
        assert_eq!(verdict_from_no_patch(toml), NoPatchVerdict::Pass);
    }

    #[test]
    fn fb004_fail_patch_present() {
        let toml = "[workspace]\nmembers = [\"crates/*\"]\n[patch.crates-io]\nfoo = { path = \"local\" }";
        assert_eq!(verdict_from_no_patch(toml), NoPatchVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: BUILD-005 all dirs have Cargo.toml.
    // -------------------------------------------------------------------------
    #[test]
    fn fb005_pass_all_present() {
        let dirs = [("aprender-core", true), ("apr-cli", true)];
        assert_eq!(
            verdict_from_all_have_cargo_toml(&dirs),
            AllHaveCargoTomlVerdict::Pass
        );
    }

    #[test]
    fn fb005_fail_missing_cargo_toml() {
        let dirs = [("aprender-core", true), ("aprender-train", false)];
        assert_eq!(
            verdict_from_all_have_cargo_toml(&dirs),
            AllHaveCargoTomlVerdict::Fail
        );
    }

    #[test]
    fn fb005_fail_empty() {
        assert_eq!(
            verdict_from_all_have_cargo_toml(&[]),
            AllHaveCargoTomlVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: BUILD-006 aprender-* naming.
    // -------------------------------------------------------------------------
    #[test]
    fn fb006_pass_canonical_names() {
        let m = ["aprender", "aprender-core", "aprender-train", "apr-cli"];
        assert_eq!(verdict_from_apr_prefix(&m), AprPrefixVerdict::Pass);
    }

    #[test]
    fn fb006_fail_legacy_name() {
        let m = ["aprender-core", "trueno"];
        assert_eq!(verdict_from_apr_prefix(&m), AprPrefixVerdict::Fail);
    }

    #[test]
    fn fb006_fail_non_aprender_prefix() {
        let m = ["aprender-core", "random-crate"];
        assert_eq!(verdict_from_apr_prefix(&m), AprPrefixVerdict::Fail);
    }

    #[test]
    fn fb006_fail_each_legacy_individually() {
        for legacy in AC_MONOREPO_FORBIDDEN_NAMES {
            let m = vec!["aprender-core", legacy];
            assert_eq!(
                verdict_from_apr_prefix(&m),
                AprPrefixVerdict::Fail,
                "legacy name {legacy} must Fail"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 8: BUILD-007 flat layout.
    // -------------------------------------------------------------------------
    #[test]
    fn fb007_pass_all_flat() {
        let p = [
            "crates/aprender-core/Cargo.toml",
            "crates/aprender-train/Cargo.toml",
        ];
        assert_eq!(verdict_from_flat_layout(&p), FlatLayoutVerdict::Pass);
    }

    #[test]
    fn fb007_fail_nested_crate() {
        let p = [
            "crates/aprender-core/Cargo.toml",
            "crates/aprender-core/sub-crate/Cargo.toml",
        ];
        assert_eq!(verdict_from_flat_layout(&p), FlatLayoutVerdict::Fail);
    }

    #[test]
    fn fb007_fail_outside_crates_dir() {
        let p = ["src/main.rs/Cargo.toml"];
        assert_eq!(verdict_from_flat_layout(&p), FlatLayoutVerdict::Fail);
    }

    #[test]
    fn fb007_fail_empty() {
        assert_eq!(verdict_from_flat_layout(&[]), FlatLayoutVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 9: Realistic — full healthy monorepo passes all 7.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_monorepo_passes_all_7() {
        // 001
        let names = ["aprender", "aprender-core", "aprender-train", "apr-cli"];
        assert_eq!(verdict_from_no_duplicates(&names), NoDuplicatesVerdict::Pass);
        // 002
        assert_eq!(
            verdict_from_version_consistency("0.29.0"),
            VersionConsistencyVerdict::Pass
        );
        // 003
        assert_eq!(verdict_from_min_member_count(70), MinMemberCountVerdict::Pass);
        // 004
        let toml = "[workspace]\nmembers = [\"crates/*\"]\nresolver = \"2\"";
        assert_eq!(verdict_from_no_patch(toml), NoPatchVerdict::Pass);
        // 005
        let dirs = [("aprender-core", true), ("apr-cli", true)];
        assert_eq!(
            verdict_from_all_have_cargo_toml(&dirs),
            AllHaveCargoTomlVerdict::Pass
        );
        // 006
        assert_eq!(verdict_from_apr_prefix(&names), AprPrefixVerdict::Pass);
        // 007
        let paths = [
            "crates/aprender/Cargo.toml",
            "crates/aprender-core/Cargo.toml",
        ];
        assert_eq!(verdict_from_flat_layout(&paths), FlatLayoutVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_7_failures() {
        // 001: subtree merge created two `trueno` entries (one renamed, one not).
        assert_eq!(
            verdict_from_no_duplicates(&["aprender-core", "aprender-core"]),
            NoDuplicatesVerdict::Fail
        );
        // 002: stale workspace version.
        assert_eq!(
            verdict_from_version_consistency("0.20.0"),
            VersionConsistencyVerdict::Fail
        );
        // 003: workspace member missing — only 30.
        assert_eq!(verdict_from_min_member_count(30), MinMemberCountVerdict::Fail);
        // 004: dev override leaked into root Cargo.toml.
        let bad_toml = "[patch.crates-io]\ntrueno = { path = \"../trueno\" }";
        assert_eq!(verdict_from_no_patch(bad_toml), NoPatchVerdict::Fail);
        // 005: subtree merge dropped Cargo.toml.
        assert_eq!(
            verdict_from_all_have_cargo_toml(&[("aprender-core", false)]),
            AllHaveCargoTomlVerdict::Fail
        );
        // 006: legacy name `realizar` still in workspace.
        assert_eq!(
            verdict_from_apr_prefix(&["aprender-core", "realizar"]),
            AprPrefixVerdict::Fail
        );
        // 007: nested crate at crates/foo/bar/Cargo.toml.
        assert_eq!(
            verdict_from_flat_layout(&["crates/foo/bar/Cargo.toml"]),
            FlatLayoutVerdict::Fail
        );
    }
}
