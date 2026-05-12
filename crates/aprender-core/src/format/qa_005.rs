// SHIP-TWO-001 — `apr-cli-qa-v1` algorithm-level PARTIAL discharge
// for FALSIFY-QA-005.
//
// Contract: `contracts/apr-cli-qa-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// (apr CLI QA gates).
//
// ## What FALSIFY-QA-005 says
//
//   rule: version matches HEAD
//   prediction: "apr --version contains current git hash"
//   test: "apr --version | grep -q $(git rev-parse --short HEAD)"
//   if_fails: "stale binary installed"
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given (`version_output`, `git_short_hash`), Pass iff:
//
//   version_output is non-empty AND
//   git_short_hash is non-empty AND
//   git_short_hash length is in canonical 7..=12 byte range AND
//   git_short_hash is hex-only (lowercase) AND
//   version_output contains git_short_hash as a substring
//
// Composes substring containment with hex-shape and length sanity
// of the git hash. Catches:
// - Stale binary: version_output has old hash, doesn't contain
//   current short hash.
// - Garbled inputs: hash with non-hex chars (corruption).
// - Wrong-length hash: a regression that pads/truncates.

/// Minimum length of a git short hash.
///
/// `git rev-parse --short HEAD` defaults to 7 chars but auto-grows
/// for larger repos. 7 is the historical minimum.
pub const AC_QA_005_MIN_HASH_LEN: usize = 7;

/// Maximum length of a canonical git short hash.
///
/// Long-form sha-1 is 40, but `--short` typically caps at 12. We
/// accept up to 12 for sanity; full 40-char hashes are technically
/// valid but unusual for `--short`.
pub const AC_QA_005_MAX_HASH_LEN: usize = 12;

/// Binary verdict for `FALSIFY-QA-005`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qa005Verdict {
    /// Both inputs valid AND `version_output` contains `git_short_hash`
    /// as a substring.
    Pass,
    /// One or more of:
    /// - `version_output.is_empty()` (caller error — apr --version
    ///   silent).
    /// - `git_short_hash` is empty / wrong length / non-hex.
    /// - `version_output` does NOT contain `git_short_hash`
    ///   (stale binary regression).
    Fail,
}

/// Pure verdict function for `FALSIFY-QA-005`.
///
/// Inputs:
/// - `version_output`: stdout from `apr --version`.
/// - `git_short_hash`: result of `git rev-parse --short HEAD`.
///
/// Pass iff:
/// 1. `!version_output.is_empty()`,
/// 2. `git_short_hash.len() >= 7 AND <= 12`,
/// 3. All bytes of `git_short_hash` are lowercase hex (`0-9` or `a-f`),
/// 4. `version_output` contains `git_short_hash` as a substring.
///
/// Otherwise `Fail`.
#[must_use]
pub fn verdict_from_version_git_hash(
    version_output: &[u8],
    git_short_hash: &[u8],
) -> Qa005Verdict {
    if version_output.is_empty() {
        return Qa005Verdict::Fail;
    }
    if git_short_hash.len() < AC_QA_005_MIN_HASH_LEN
        || git_short_hash.len() > AC_QA_005_MAX_HASH_LEN
    {
        return Qa005Verdict::Fail;
    }
    if !git_short_hash
        .iter()
        .all(|&b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Qa005Verdict::Fail;
    }
    if contains_subsequence(version_output, git_short_hash) {
        Qa005Verdict::Pass
    } else {
        Qa005Verdict::Fail
    }
}

/// Returns `true` iff `needle` appears as a contiguous subsequence
/// of `haystack`. Same primitive used in `pull_dataset_001/005` and
/// `pub_cli_001`.
#[must_use]
fn contains_subsequence(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.len() > haystack.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — git short-hash length range.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_min_hash_len_is_7() {
        assert_eq!(AC_QA_005_MIN_HASH_LEN, 7);
    }

    #[test]
    fn provenance_max_hash_len_is_12() {
        assert_eq!(AC_QA_005_MAX_HASH_LEN, 12);
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — canonical short hashes embedded in version.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_canonical_7char_hash() {
        let v = verdict_from_version_git_hash(
            b"apr 0.31.2 (b7bf4b0)",
            b"b7bf4b0",
        );
        assert_eq!(v, Qa005Verdict::Pass);
    }

    #[test]
    fn pass_8char_hash() {
        let v = verdict_from_version_git_hash(
            b"apr 0.31.2 (b7bf4b07)",
            b"b7bf4b07",
        );
        assert_eq!(v, Qa005Verdict::Pass);
    }

    #[test]
    fn pass_12char_hash_at_max_len() {
        let v = verdict_from_version_git_hash(
            b"apr 0.31.2 (b7bf4b07a000)",
            b"b7bf4b07a000",
        );
        assert_eq!(v, Qa005Verdict::Pass);
    }

    #[test]
    fn pass_realistic_long_version_output() {
        let v = verdict_from_version_git_hash(
            b"apr 0.31.2\nbuild: release\ncommit: b7bf4b07a\nrustc: 1.84.0",
            b"b7bf4b07a",
        );
        assert_eq!(v, Qa005Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — stale binary (hash mismatch).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_old_hash_in_version() {
        // Binary built from old commit; version shows old hash.
        let v = verdict_from_version_git_hash(
            b"apr 0.31.0 (deadbee)",
            b"b7bf4b0",
        );
        assert_eq!(
            v,
            Qa005Verdict::Fail,
            "stale binary must Fail (current hash not in version)"
        );
    }

    #[test]
    fn fail_no_hash_in_version() {
        let v = verdict_from_version_git_hash(
            b"apr 0.31.2",
            b"b7bf4b0",
        );
        assert_eq!(v, Qa005Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — empty / wrong-length hash.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_empty_hash() {
        let v = verdict_from_version_git_hash(b"apr 0.31.2 (b7bf4b0)", &[]);
        assert_eq!(v, Qa005Verdict::Fail);
    }

    #[test]
    fn fail_hash_too_short() {
        // 6 chars is below the 7-char minimum.
        let v = verdict_from_version_git_hash(
            b"apr 0.31.2 (b7bf4b)",
            b"b7bf4b",
        );
        assert_eq!(v, Qa005Verdict::Fail);
    }

    #[test]
    fn fail_hash_too_long() {
        // 13 chars exceeds 12-char cap.
        let v = verdict_from_version_git_hash(
            b"apr 0.31.2 (b7bf4b07a0000)",
            b"b7bf4b07a0000",
        );
        assert_eq!(v, Qa005Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Fail band — non-hex hash (corruption).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_uppercase_hex() {
        // git --short emits lowercase; uppercase is corruption.
        let v = verdict_from_version_git_hash(
            b"apr 0.31.2 (B7BF4B0)",
            b"B7BF4B0",
        );
        assert_eq!(v, Qa005Verdict::Fail);
    }

    #[test]
    fn fail_non_hex_chars() {
        // Has `g` which is invalid hex.
        let v = verdict_from_version_git_hash(
            b"apr 0.31.2 (b7g4b0a)",
            b"b7g4b0a",
        );
        assert_eq!(v, Qa005Verdict::Fail);
    }

    #[test]
    fn fail_hash_with_dash() {
        let v = verdict_from_version_git_hash(
            b"apr 0.31.2 (abc-def)",
            b"abc-def",
        );
        assert_eq!(v, Qa005Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Fail band — empty version output.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_empty_version_output() {
        let v = verdict_from_version_git_hash(&[], b"b7bf4b0");
        assert_eq!(v, Qa005Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — apr release tag scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_v0_31_release() {
        let v = verdict_from_version_git_hash(
            b"apr 0.31.0 (62893da)",
            b"62893da",
        );
        assert_eq!(v, Qa005Verdict::Pass);
    }

    #[test]
    fn fail_release_binary_dirty_workspace() {
        // User has uncommitted changes; current HEAD diverges from
        // the binary's committed hash.
        let v = verdict_from_version_git_hash(
            b"apr 0.31.0 (62893da)",
            b"a8bb681", // current HEAD differs from binary
        );
        assert_eq!(v, Qa005Verdict::Fail);
    }
}
