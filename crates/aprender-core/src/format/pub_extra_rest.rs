// SHIP-TWO-001 — `apr-cli-publish-extra-v1` algorithm-level
// PARTIAL discharge for the remaining 6 PUB-EXTRA gates
// (003 + 004 + 005 + 008 + 009 + 010). Closes 10/10 sweep.
//
// Contract: `contracts/apr-cli-publish-extra-v1.yaml`.

// ===========================================================================
// PUB-EXTRA-003 — --extra-file roundtrip byte-identity (sha256)
// ===========================================================================

pub const AC_PUB_EXTRA_003_SHA256_BYTES: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PubExtra003Verdict {
    Pass,
    Fail,
}

/// Pass iff sidecar file's pre-publish sha256 matches post-pull sha256.
#[must_use]
pub fn verdict_from_extra_file_roundtrip(probe_sha256: &[u8], pulled_sha256: &[u8]) -> PubExtra003Verdict {
    if probe_sha256.len() != AC_PUB_EXTRA_003_SHA256_BYTES {
        return PubExtra003Verdict::Fail;
    }
    if pulled_sha256.len() != AC_PUB_EXTRA_003_SHA256_BYTES {
        return PubExtra003Verdict::Fail;
    }
    if probe_sha256 == pulled_sha256 {
        PubExtra003Verdict::Pass
    } else {
        PubExtra003Verdict::Fail
    }
}

// ===========================================================================
// PUB-EXTRA-004 — no-manifest backward compat (no manifest.yaml in dry-run)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PubExtra004Verdict {
    Pass,
    Fail,
}

/// Pass iff stdout is non-empty AND does NOT contain `manifest.yaml`.
#[must_use]
pub fn verdict_from_no_manifest_backward_compat(stdout: &[u8]) -> PubExtra004Verdict {
    if stdout.is_empty() {
        return PubExtra004Verdict::Fail;
    }
    if contains_subseq(stdout, b"manifest.yaml") {
        PubExtra004Verdict::Fail
    } else {
        PubExtra004Verdict::Pass
    }
}

// ===========================================================================
// PUB-EXTRA-005 + 008 — script does NOT invoke Python uploaders
// (zero-tolerance grep-output count)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PubExtraScriptPurityVerdict {
    Pass,
    Fail,
}

/// Pass iff `python_invocation_match_count == 0`.
///
/// Same shape used for both PUB-EXTRA-005 (ex-04-upload-hf.sh) and
/// PUB-EXTRA-008 (ex-05-verify-manifest.sh).
#[must_use]
pub fn verdict_from_script_python_purity(python_invocation_match_count: u64) -> PubExtraScriptPurityVerdict {
    if python_invocation_match_count == 0 {
        PubExtraScriptPurityVerdict::Pass
    } else {
        PubExtraScriptPurityVerdict::Fail
    }
}

// ===========================================================================
// PUB-EXTRA-009 — sha256-mismatch preflight aborts with exit 2 + zero uploads
// ===========================================================================

/// Required exit code per contract: "exit code 2".
pub const AC_PUB_EXTRA_009_REQUIRED_EXIT: i32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PubExtra009Verdict {
    Pass,
    Fail,
}

/// Pass iff `exit_code == 2 AND hf_api_call_count == 0`.
#[must_use]
pub fn verdict_from_preflight_abort(exit_code: i32, hf_api_call_count: u64) -> PubExtra009Verdict {
    if exit_code != AC_PUB_EXTRA_009_REQUIRED_EXIT {
        return PubExtra009Verdict::Fail;
    }
    if hf_api_call_count != 0 {
        return PubExtra009Verdict::Fail;
    }
    PubExtra009Verdict::Pass
}

// ===========================================================================
// PUB-EXTRA-010 — preflight_validate_manifest appears >= 4 times
// ===========================================================================

pub const AC_PUB_EXTRA_010_MIN_OCCURRENCES: u64 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PubExtra010Verdict {
    Pass,
    Fail,
}

/// Pass iff `occurrence_count >= 4` (1 definition + 3 invocations).
#[must_use]
pub fn verdict_from_preflight_invocation_count(occurrence_count: u64) -> PubExtra010Verdict {
    if occurrence_count >= AC_PUB_EXTRA_010_MIN_OCCURRENCES {
        PubExtra010Verdict::Pass
    } else {
        PubExtra010Verdict::Fail
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

    // PUB-EXTRA-003 ------------------------------------------------------------
    #[test]
    fn p003_pass_byte_identical() {
        let d = [0xab_u8; 32];
        assert_eq!(verdict_from_extra_file_roundtrip(&d, &d), PubExtra003Verdict::Pass);
    }

    #[test]
    fn p003_fail_corruption() {
        let probe = [0xab_u8; 32];
        let mut pulled = [0xab_u8; 32];
        pulled[0] = 0xac;
        assert_eq!(verdict_from_extra_file_roundtrip(&probe, &pulled), PubExtra003Verdict::Fail);
    }

    #[test]
    fn p003_fail_wrong_length() {
        assert_eq!(verdict_from_extra_file_roundtrip(&[0u8; 16], &[0u8; 16]), PubExtra003Verdict::Fail);
    }

    #[test]
    fn p003_fail_empty() {
        assert_eq!(verdict_from_extra_file_roundtrip(&[], &[]), PubExtra003Verdict::Fail);
    }

    // PUB-EXTRA-004 ------------------------------------------------------------
    #[test]
    fn p004_pass_no_manifest_in_output() {
        let stdout = b"DRY RUN: would upload README.md, model.apr, tokenizer.json";
        assert_eq!(verdict_from_no_manifest_backward_compat(stdout), PubExtra004Verdict::Pass);
    }

    #[test]
    fn p004_fail_manifest_yaml_in_output() {
        let stdout = b"DRY RUN: would upload manifest.yaml, model.apr";
        assert_eq!(verdict_from_no_manifest_backward_compat(stdout), PubExtra004Verdict::Fail);
    }

    #[test]
    fn p004_fail_empty_stdout() {
        assert_eq!(verdict_from_no_manifest_backward_compat(&[]), PubExtra004Verdict::Fail);
    }

    // PUB-EXTRA-005/008 --------------------------------------------------------
    #[test]
    fn pscript_pass_zero_python_matches() {
        assert_eq!(verdict_from_script_python_purity(0), PubExtraScriptPurityVerdict::Pass);
    }

    #[test]
    fn pscript_fail_one_python_match() {
        assert_eq!(verdict_from_script_python_purity(1), PubExtraScriptPurityVerdict::Fail);
    }

    #[test]
    fn pscript_fail_many_matches() {
        assert_eq!(verdict_from_script_python_purity(100), PubExtraScriptPurityVerdict::Fail);
    }

    // PUB-EXTRA-009 ------------------------------------------------------------
    #[test]
    fn p009_provenance_required_exit_is_2() {
        assert_eq!(AC_PUB_EXTRA_009_REQUIRED_EXIT, 2);
    }

    #[test]
    fn p009_pass_exit_2_zero_calls() {
        assert_eq!(verdict_from_preflight_abort(2, 0), PubExtra009Verdict::Pass);
    }

    #[test]
    fn p009_fail_exit_zero() {
        assert_eq!(verdict_from_preflight_abort(0, 0), PubExtra009Verdict::Fail);
    }

    #[test]
    fn p009_fail_exit_one() {
        // Contract requires exactly 2, not just non-zero.
        assert_eq!(verdict_from_preflight_abort(1, 0), PubExtra009Verdict::Fail);
    }

    #[test]
    fn p009_fail_exit_2_with_uploads() {
        assert_eq!(verdict_from_preflight_abort(2, 1), PubExtra009Verdict::Fail);
    }

    // PUB-EXTRA-010 ------------------------------------------------------------
    #[test]
    fn p010_provenance_min_occurrences_is_4() {
        assert_eq!(AC_PUB_EXTRA_010_MIN_OCCURRENCES, 4);
    }

    #[test]
    fn p010_pass_at_exact_floor() {
        assert_eq!(verdict_from_preflight_invocation_count(4), PubExtra010Verdict::Pass);
    }

    #[test]
    fn p010_pass_above_floor() {
        assert_eq!(verdict_from_preflight_invocation_count(10), PubExtra010Verdict::Pass);
    }

    #[test]
    fn p010_fail_just_below_floor() {
        assert_eq!(verdict_from_preflight_invocation_count(3), PubExtra010Verdict::Fail);
    }

    #[test]
    fn p010_fail_zero() {
        assert_eq!(verdict_from_preflight_invocation_count(0), PubExtra010Verdict::Fail);
    }

    // Shared primitive ---------------------------------------------------------
    #[test]
    fn contains_subseq_basic() {
        assert!(contains_subseq(b"hello world", b"world"));
        assert!(!contains_subseq(b"hello world", b"xyz"));
    }
}
