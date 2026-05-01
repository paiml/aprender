// SHIP-TWO-001 — `apr-cli-dep-migration-v1` algorithm-level
// PARTIAL discharge for FALSIFY-DEPMIG-001 + 002 (closes 2/2).
//
// Contract: `contracts/apr-cli-dep-migration-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// (APR-MONO consolidation per `feedback_monorepo_single_source_of_truth`).

// ===========================================================================
// DEPMIG-001 — zero old crate names in apr-cli/Cargo.toml deps
// ===========================================================================

/// Forbidden old crate names that must NOT appear as `^<name> ` in
/// `crates/apr-cli/Cargo.toml`.
///
/// Per APR-MONO consolidation, all listed crates were absorbed into
/// the aprender monorepo as `aprender-*` workspace crates. Their
/// crates.io packages still exist as historical artifacts but
/// MUST NOT be pulled by the published `apr-cli` crate.
pub const AC_DEPMIG_001_FORBIDDEN_OLD_NAMES: &[&[u8]] = &[
    b"batuta",
    b"realizar",
    b"trueno",
    b"entrenar",
    b"alimentar",
    b"renacer",
    b"certeza",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Depmig001Verdict {
    /// `forbidden_match_count == 0`.
    Pass,
    /// At least one old crate name appears in the deps section.
    Fail,
}

/// Pure verdict function for `FALSIFY-DEPMIG-001`.
///
/// Pass iff `forbidden_match_count == 0`.
#[must_use]
pub fn verdict_from_old_dep_count(forbidden_match_count: u64) -> Depmig001Verdict {
    if forbidden_match_count == 0 {
        Depmig001Verdict::Pass
    } else {
        Depmig001Verdict::Fail
    }
}

// ===========================================================================
// DEPMIG-002 — cargo install aprender succeeds (Replacing|Installed)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Depmig002Verdict {
    /// stdout/stderr non-empty AND contains `Replacing` OR `Installed`.
    Pass,
    /// Empty output OR neither token present (cargo install
    /// failed silently or with unrecognized message).
    Fail,
}

/// Pure verdict function for `FALSIFY-DEPMIG-002`.
///
/// Pass iff combined output contains `Replacing` OR `Installed`
/// substring.
#[must_use]
pub fn verdict_from_cargo_install_output(output: &[u8]) -> Depmig002Verdict {
    if output.is_empty() {
        return Depmig002Verdict::Fail;
    }
    if contains_subseq(output, b"Replacing") || contains_subseq(output, b"Installed") {
        Depmig002Verdict::Pass
    } else {
        Depmig002Verdict::Fail
    }
}

// ===========================================================================
// Shared primitive
// ===========================================================================

#[must_use]
fn contains_subseq(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.len() > haystack.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // DEPMIG-001
    // -------------------------------------------------------------------------
    #[test]
    fn d001_provenance_seven_forbidden_names() {
        assert_eq!(AC_DEPMIG_001_FORBIDDEN_OLD_NAMES.len(), 7);
    }

    #[test]
    fn d001_provenance_canonical_old_names() {
        let names: Vec<&[u8]> = AC_DEPMIG_001_FORBIDDEN_OLD_NAMES.to_vec();
        assert!(names.contains(&b"batuta".as_slice()));
        assert!(names.contains(&b"realizar".as_slice()));
        assert!(names.contains(&b"trueno".as_slice()));
        assert!(names.contains(&b"entrenar".as_slice()));
        assert!(names.contains(&b"alimentar".as_slice()));
        assert!(names.contains(&b"renacer".as_slice()));
        assert!(names.contains(&b"certeza".as_slice()));
    }

    #[test]
    fn d001_pass_zero_old_deps() {
        let v = verdict_from_old_dep_count(0);
        assert_eq!(v, Depmig001Verdict::Pass);
    }

    #[test]
    fn d001_fail_one_old_dep() {
        let v = verdict_from_old_dep_count(1);
        assert_eq!(v, Depmig001Verdict::Fail);
    }

    #[test]
    fn d001_fail_all_seven_old_deps() {
        let v = verdict_from_old_dep_count(7);
        assert_eq!(v, Depmig001Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // DEPMIG-002
    // -------------------------------------------------------------------------
    #[test]
    fn d002_pass_replacing_message() {
        let output = b"  Replacing aprender v0.31.1 with aprender v0.31.2";
        let v = verdict_from_cargo_install_output(output);
        assert_eq!(v, Depmig002Verdict::Pass);
    }

    #[test]
    fn d002_pass_installed_message() {
        let output = b"  Installed package `aprender v0.31.2` (executable `apr`)";
        let v = verdict_from_cargo_install_output(output);
        assert_eq!(v, Depmig002Verdict::Pass);
    }

    #[test]
    fn d002_pass_realistic_full_output() {
        let output = b"\
    Updating crates.io index
  Downloaded aprender v0.31.2
   Compiling aprender v0.31.2
    Finished `release` profile [optimized] target(s)
   Replacing aprender v0.31.1 with aprender v0.31.2
";
        let v = verdict_from_cargo_install_output(output);
        assert_eq!(v, Depmig002Verdict::Pass);
    }

    #[test]
    fn d002_fail_empty_output() {
        let v = verdict_from_cargo_install_output(&[]);
        assert_eq!(v, Depmig002Verdict::Fail);
    }

    #[test]
    fn d002_fail_unrelated_output() {
        let v = verdict_from_cargo_install_output(b"some unrelated cargo output");
        assert_eq!(v, Depmig002Verdict::Fail);
    }

    #[test]
    fn d002_fail_compile_error() {
        let output = b"error[E0432]: unresolved import `realizar::Model`";
        let v = verdict_from_cargo_install_output(output);
        assert_eq!(
            v,
            Depmig002Verdict::Fail,
            "compile error must Fail (no Replacing/Installed)"
        );
    }
}
