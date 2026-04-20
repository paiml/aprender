//! Resume-sidecar path resolver + etag comparator for `apr pull --resume`
//! (CRUX-A-05).
//!
//! Contract: `contracts/crux-A-05-v1.yaml`.
//!
//! Pure classifier — takes the user-visible `--out` target path (or the
//! live ETag values) and returns deterministic paths / decisions the
//! resume machinery will use. No I/O, no filesystem access.
//!
//! The actual HTTP Range requests, the advisory file lock, the bytes on
//! disk, and the byte-identical-after-resume sha256 check are all
//! discharged by separate network/strace-gated harnesses (follow-up).

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// Suffix appended to the target path for the advisory lock file.
/// Matches the canonical shell-test convention `$TARGET.lock`.
pub const LOCK_SUFFIX: &str = ".lock";

/// Suffix appended to the target path for the ETag sidecar.
/// Matches the canonical shell-test convention `$TARGET.etag`.
pub const ETAG_SUFFIX: &str = ".etag";

/// Suffix appended to the target path for the in-progress partial file.
/// `foo.bin.partial` lives alongside `foo.bin` until resume completes.
pub const PARTIAL_SUFFIX: &str = ".partial";

/// Return the advisory lock file path for a given target. Deterministic
/// append of `LOCK_SUFFIX`.
///
/// CRUX-A-05 ALGO-004 sub-claim of FALSIFY-004: two `apr pull --resume`
/// invocations targeting the same path compute the SAME lock path and
/// therefore contend for the same advisory flock.
pub fn lock_path(target: &Path) -> PathBuf {
    append_suffix(target, LOCK_SUFFIX)
}

/// Return the ETag sidecar path for a given target. Deterministic append
/// of `ETAG_SUFFIX`.
///
/// CRUX-A-05 ALGO-003 sub-claim of FALSIFY-003: the stale-partial
/// detector reads from a fixed `$TARGET.etag` path so a pre-seeded
/// "wrong-etag-value" file is always observable.
pub fn etag_path(target: &Path) -> PathBuf {
    append_suffix(target, ETAG_SUFFIX)
}

/// Return the in-progress partial path for a given target. Deterministic
/// append of `PARTIAL_SUFFIX`.
pub fn partial_path(target: &Path) -> PathBuf {
    append_suffix(target, PARTIAL_SUFFIX)
}

/// Classification of a recorded-vs-served ETag pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EtagDecision {
    /// No recorded ETag on disk → treat any existing partial as stale.
    MissingRecorded,
    /// Both present and byte-equal → the partial is resumable.
    Match,
    /// Both present but differ → the recorded partial is stale and MUST
    /// be discarded with a warning. FALSIFY-003.
    Mismatch,
}

/// Compare a recorded ETag (from the sidecar) against the live server
/// ETag. Whitespace is trimmed on both sides (HTTP header parsers
/// sometimes leave trailing newlines on disk).
///
/// CRUX-A-05 ALGO-003 sub-claim of FALSIFY-003: a mismatched recorded
/// ETag deterministically classifies as `Mismatch`, which is the
/// algorithm-level precondition for the "discard stale partial +
/// emit warning" integration.
pub fn compare_etag(recorded: Option<&str>, served: &str) -> EtagDecision {
    match recorded {
        None => EtagDecision::MissingRecorded,
        Some(r) if r.trim().is_empty() => EtagDecision::MissingRecorded,
        Some(r) if r.trim() == served.trim() => EtagDecision::Match,
        Some(_) => EtagDecision::Mismatch,
    }
}

/// True iff the decision calls for discarding the partial.
/// Convenience helper for caller symmetry with other CRUX classifiers.
pub fn should_discard_partial(decision: &EtagDecision) -> bool {
    matches!(
        decision,
        EtagDecision::Mismatch | EtagDecision::MissingRecorded
    )
}

fn append_suffix(target: &Path, suffix: &str) -> PathBuf {
    let mut name: OsString = target.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_path_appends_dot_lock() {
        let p = lock_path(Path::new("/tmp/apr-resume-test.bin"));
        assert_eq!(p, PathBuf::from("/tmp/apr-resume-test.bin.lock"));
    }

    #[test]
    fn etag_path_appends_dot_etag() {
        let p = etag_path(Path::new("/tmp/apr-resume-test.bin"));
        assert_eq!(p, PathBuf::from("/tmp/apr-resume-test.bin.etag"));
    }

    #[test]
    fn partial_path_appends_dot_partial() {
        let p = partial_path(Path::new("/tmp/apr-resume-test.bin"));
        assert_eq!(p, PathBuf::from("/tmp/apr-resume-test.bin.partial"));
    }

    #[test]
    fn sidecar_paths_are_deterministic() {
        // CRUX-A-05 ALGO-004: same target → same lock path. Two
        // processes racing on the same `--out` MUST contend.
        let a = lock_path(Path::new("/tmp/x"));
        let b = lock_path(Path::new("/tmp/x"));
        assert_eq!(a, b);
        assert_eq!(etag_path(Path::new("/tmp/x")), etag_path(Path::new("/tmp/x")));
    }

    #[test]
    fn sidecar_paths_are_distinct_by_target() {
        // Two different targets MUST produce two different lock paths
        // (else unrelated pulls would contend).
        assert_ne!(
            lock_path(Path::new("/tmp/a")),
            lock_path(Path::new("/tmp/b"))
        );
    }

    #[test]
    fn sidecar_paths_preserve_directory() {
        let p = lock_path(Path::new("/var/cache/apr/model.bin"));
        assert_eq!(p.parent(), Some(Path::new("/var/cache/apr")));
    }

    #[test]
    fn sidecar_paths_work_on_relative_paths() {
        let p = lock_path(Path::new("model.bin"));
        assert_eq!(p, PathBuf::from("model.bin.lock"));
    }

    #[test]
    fn sidecar_paths_work_on_paths_with_many_dots() {
        // Sanity: we do not strip extensions. `foo.tar.gz` gets
        // `foo.tar.gz.lock`, not `foo.tar.lock`.
        let p = lock_path(Path::new("/tmp/foo.tar.gz"));
        assert_eq!(p, PathBuf::from("/tmp/foo.tar.gz.lock"));
    }

    #[test]
    fn compare_etag_returns_missing_recorded_on_none() {
        assert_eq!(
            compare_etag(None, "W/\"abc\""),
            EtagDecision::MissingRecorded
        );
    }

    #[test]
    fn compare_etag_returns_missing_recorded_on_empty_string() {
        assert_eq!(
            compare_etag(Some(""), "W/\"abc\""),
            EtagDecision::MissingRecorded
        );
        assert_eq!(
            compare_etag(Some("   "), "W/\"abc\""),
            EtagDecision::MissingRecorded
        );
    }

    #[test]
    fn compare_etag_returns_match_on_equal() {
        assert_eq!(
            compare_etag(Some("W/\"abc\""), "W/\"abc\""),
            EtagDecision::Match
        );
    }

    #[test]
    fn compare_etag_trims_trailing_whitespace() {
        // HTTP header writes often leave trailing newlines on disk.
        assert_eq!(
            compare_etag(Some("W/\"abc\"\n"), "W/\"abc\""),
            EtagDecision::Match
        );
    }

    #[test]
    fn compare_etag_returns_mismatch_on_diff() {
        assert_eq!(
            compare_etag(Some("wrong-etag-value"), "W/\"served-etag\""),
            EtagDecision::Mismatch
        );
    }

    #[test]
    fn should_discard_partial_agrees_with_decision() {
        assert!(should_discard_partial(&EtagDecision::Mismatch));
        assert!(should_discard_partial(&EtagDecision::MissingRecorded));
        assert!(!should_discard_partial(&EtagDecision::Match));
    }

    #[test]
    fn compare_etag_is_deterministic() {
        let a = compare_etag(Some("x"), "y");
        let b = compare_etag(Some("x"), "y");
        assert_eq!(a, b);
    }

    #[test]
    fn etag_mismatch_sub_claim_falsify_003() {
        // CRUX-A-05 ALGO-003 sub-claim of FALSIFY-003: a pre-seeded
        // stale etag "wrong-etag-value" on disk against a real served
        // etag MUST classify as Mismatch, which is the precondition for
        // the "discard + warn" integration path.
        let decision = compare_etag(
            Some("wrong-etag-value"),
            "\"d41d8cd98f00b204e9800998ecf8427e\"",
        );
        assert_eq!(decision, EtagDecision::Mismatch);
        assert!(should_discard_partial(&decision));
    }

    #[test]
    fn sidecar_suffixes_are_stable() {
        // Downstream shell tests grep for these exact suffixes.
        assert_eq!(LOCK_SUFFIX, ".lock");
        assert_eq!(ETAG_SUFFIX, ".etag");
        assert_eq!(PARTIAL_SUFFIX, ".partial");
    }
}
