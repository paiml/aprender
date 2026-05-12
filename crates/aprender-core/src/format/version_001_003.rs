// `apr-version-traceability-v1` algorithm-level PARTIAL discharge for the
// 3 version-traceability falsifiers (no `(unknown)` sentinel, no
// alternative sentinels, version string present).
//
// Contract: `contracts/apr-version-traceability-v1.yaml`.
// Refs: paiml/aprender#597 (version string shows `(unknown)` instead of
// git hash on cargo install builds).

/// Forbidden sentinel forms per FALSIFY-VERSION-002.
pub const AC_VERSION_FORBIDDEN_SENTINELS: [&str; 4] = [
    "(unknown)",
    "(0000000)",
    "(<empty>)",
    "(null)",
];

/// Regex-shaped predicate for FALSIFY-VERSION-003: version output must
/// contain a `0.X.Y` token (CARGO_PKG_VERSION semver).
pub const AC_VERSION_PKG_PREFIX: &str = "0.";

// =============================================================================
// FALSIFY-VERSION-001 — output does NOT contain `(unknown)`
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoUnknownSentinelVerdict {
    /// Output does not contain the literal `(unknown)`.
    Pass,
    /// Output contains `(unknown)` — sentinel leaked.
    Fail,
}

#[must_use]
pub fn verdict_from_no_unknown_sentinel(version_output: &str) -> NoUnknownSentinelVerdict {
    if version_output.contains("(unknown)") {
        NoUnknownSentinelVerdict::Fail
    } else {
        NoUnknownSentinelVerdict::Pass
    }
}

// =============================================================================
// FALSIFY-VERSION-002 — no sentinel from forbidden set
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoForbiddenSentinelVerdict {
    /// None of `(unknown)`, `(0000000)`, `(<empty>)`, `(null)` present.
    Pass,
    /// At least one forbidden sentinel form leaked.
    Fail,
}

#[must_use]
pub fn verdict_from_no_forbidden_sentinel(version_output: &str) -> NoForbiddenSentinelVerdict {
    for sentinel in AC_VERSION_FORBIDDEN_SENTINELS {
        if version_output.contains(sentinel) {
            return NoForbiddenSentinelVerdict::Fail;
        }
    }
    NoForbiddenSentinelVerdict::Pass
}

// =============================================================================
// FALSIFY-VERSION-003 — output contains a CARGO_PKG_VERSION (0.X.Y)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionPresentVerdict {
    /// Output contains a 0.X.Y semver token.
    Pass,
    /// Output missing version string entirely (e.g., empty `apr --version`).
    Fail,
}

#[must_use]
pub fn verdict_from_version_present(version_output: &str) -> VersionPresentVerdict {
    // Look for "0." followed by a digit-or-dot run consistent with semver.
    // Scan once char-by-char to keep this dependency-free.
    let bytes = version_output.as_bytes();
    let prefix = AC_VERSION_PKG_PREFIX.as_bytes();
    let n = bytes.len();
    if n < prefix.len() + 3 {
        return VersionPresentVerdict::Fail;
    }
    for start in 0..=n - prefix.len() {
        if &bytes[start..start + prefix.len()] != prefix {
            continue;
        }
        // After "0." need at least one digit AND another dot AND another digit
        // (matching `0.X.Y` minimum).
        let mut i = start + prefix.len();
        let major_start = i;
        while i < n && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == major_start {
            continue;
        }
        if i >= n || bytes[i] != b'.' {
            continue;
        }
        i += 1; // skip the second dot
        let minor_start = i;
        while i < n && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == minor_start {
            continue;
        }
        return VersionPresentVerdict::Pass;
    }
    VersionPresentVerdict::Fail
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_forbidden_sentinels_count_4() {
        assert_eq!(AC_VERSION_FORBIDDEN_SENTINELS.len(), 4);
    }

    #[test]
    fn provenance_unknown_sentinel_in_forbidden() {
        assert!(AC_VERSION_FORBIDDEN_SENTINELS.contains(&"(unknown)"));
    }

    #[test]
    fn provenance_pkg_prefix_is_0_dot() {
        assert_eq!(AC_VERSION_PKG_PREFIX, "0.");
    }

    // -------------------------------------------------------------------------
    // Section 2: VERSION-001 no (unknown).
    // -------------------------------------------------------------------------
    #[test]
    fn fv001_pass_clean_version_with_hash() {
        let v = "apr 0.31.2 (b8c5d809)";
        assert_eq!(
            verdict_from_no_unknown_sentinel(v),
            NoUnknownSentinelVerdict::Pass
        );
    }

    #[test]
    fn fv001_pass_no_git_fallback() {
        let v = "apr 0.31.2 (v0.31.2+no-git)";
        assert_eq!(
            verdict_from_no_unknown_sentinel(v),
            NoUnknownSentinelVerdict::Pass
        );
    }

    #[test]
    fn fv001_fail_unknown_leaked() {
        let v = "apr 0.4.13 (unknown)";
        assert_eq!(
            verdict_from_no_unknown_sentinel(v),
            NoUnknownSentinelVerdict::Fail
        );
    }

    #[test]
    fn fv001_fail_unknown_in_other_position() {
        let v = "apr (unknown) version";
        assert_eq!(
            verdict_from_no_unknown_sentinel(v),
            NoUnknownSentinelVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: VERSION-002 no forbidden sentinels.
    // -------------------------------------------------------------------------
    #[test]
    fn fv002_pass_clean_version() {
        let v = "apr 0.31.2 (b8c5d809)";
        assert_eq!(
            verdict_from_no_forbidden_sentinel(v),
            NoForbiddenSentinelVerdict::Pass
        );
    }

    #[test]
    fn fv002_fail_unknown() {
        assert_eq!(
            verdict_from_no_forbidden_sentinel("apr 0.4.13 (unknown)"),
            NoForbiddenSentinelVerdict::Fail
        );
    }

    #[test]
    fn fv002_fail_zero_hash() {
        assert_eq!(
            verdict_from_no_forbidden_sentinel("apr 0.4.13 (0000000)"),
            NoForbiddenSentinelVerdict::Fail
        );
    }

    #[test]
    fn fv002_fail_empty_marker() {
        assert_eq!(
            verdict_from_no_forbidden_sentinel("apr 0.4.13 (<empty>)"),
            NoForbiddenSentinelVerdict::Fail
        );
    }

    #[test]
    fn fv002_fail_null_marker() {
        assert_eq!(
            verdict_from_no_forbidden_sentinel("apr 0.4.13 (null)"),
            NoForbiddenSentinelVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: VERSION-003 version-string present.
    // -------------------------------------------------------------------------
    #[test]
    fn fv003_pass_canonical_format() {
        assert_eq!(
            verdict_from_version_present("apr 0.31.2 (b8c5d809)"),
            VersionPresentVerdict::Pass
        );
    }

    #[test]
    fn fv003_pass_old_version() {
        assert_eq!(
            verdict_from_version_present("apr 0.4.13 (unknown)"),
            VersionPresentVerdict::Pass
        );
    }

    #[test]
    fn fv003_pass_version_with_no_git_suffix() {
        assert_eq!(
            verdict_from_version_present("apr 0.31.2 (v0.31.2+no-git)"),
            VersionPresentVerdict::Pass
        );
    }

    #[test]
    fn fv003_fail_no_version_at_all() {
        assert_eq!(
            verdict_from_version_present("apr (unknown)"),
            VersionPresentVerdict::Fail
        );
    }

    #[test]
    fn fv003_fail_empty_output() {
        assert_eq!(verdict_from_version_present(""), VersionPresentVerdict::Fail);
    }

    #[test]
    fn fv003_fail_only_major_dot() {
        // "0." alone (no minor/patch) doesn't satisfy 0.X.Y.
        assert_eq!(
            verdict_from_version_present("apr 0."),
            VersionPresentVerdict::Fail
        );
    }

    #[test]
    fn fv003_pass_two_dot_pattern_in_path() {
        // Path "/usr/lib/0.so" embeds "0.so" — must NOT trigger Pass.
        // (no 0.X.Y semver match because "so" is not digits.)
        assert_eq!(
            verdict_from_version_present("loaded /usr/lib/0.so"),
            VersionPresentVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: Realistic — full healthy version output passes all 3.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_clean_release_passes_all_3() {
        let v = "apr 0.31.2 (b8c5d809) — built 2026-05-02";
        assert_eq!(verdict_from_no_unknown_sentinel(v), NoUnknownSentinelVerdict::Pass);
        assert_eq!(verdict_from_no_forbidden_sentinel(v), NoForbiddenSentinelVerdict::Pass);
        assert_eq!(verdict_from_version_present(v), VersionPresentVerdict::Pass);
    }

    #[test]
    fn realistic_no_git_fallback_passes_all_3() {
        // The exact "cargo install from crates.io" healthy fallback.
        let v = "apr 0.31.2 (v0.31.2+no-git)";
        assert_eq!(verdict_from_no_unknown_sentinel(v), NoUnknownSentinelVerdict::Pass);
        assert_eq!(verdict_from_no_forbidden_sentinel(v), NoForbiddenSentinelVerdict::Pass);
        assert_eq!(verdict_from_version_present(v), VersionPresentVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_001_002_fail_003_passes() {
        // The exact regression class from GH-597: "apr 0.4.13 (unknown)"
        // — has version string (003 passes), but leaked sentinel (001/002 fail).
        let v = "apr 0.4.13 (unknown)";
        assert_eq!(verdict_from_no_unknown_sentinel(v), NoUnknownSentinelVerdict::Fail);
        assert_eq!(verdict_from_no_forbidden_sentinel(v), NoForbiddenSentinelVerdict::Fail);
        assert_eq!(verdict_from_version_present(v), VersionPresentVerdict::Pass);
    }
}
