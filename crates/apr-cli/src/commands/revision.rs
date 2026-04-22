//! Revision specifier parsing for `apr pull --revision` (CRUX-A-03).
//!
//! Contract: `contracts/crux-A-03-v1.yaml`.
//!
//! This module implements the LOCAL classification of a revision spec that
//! a user passes to `apr pull --revision <REV>`. It does NOT resolve the
//! revision against any remote — that requires hitting the HuggingFace Hub
//! API (`GET /api/models/<repo>/revision/<REV>`) and is out of scope for
//! offline falsification. The classifier's purpose is to reject obviously
//! malformed revision specs before any network call is attempted, and to
//! echo the accepted form in `--dry-run` output so callers can confirm
//! what will be pinned.
//!
//! Accepted forms (mirrored from huggingface_hub):
//!   - "main" / any git ref name (branch, tag) — arbitrary non-empty UTF-8
//!   - full SHA — exactly 40 lowercase hex chars
//!   - short SHA — 7..=39 lowercase hex chars
//!
//! Rejected forms:
//!   - empty string
//!   - leading/trailing whitespace, or interior whitespace
//!   - contains "://" (callers passed a URL by mistake)
//!   - contains NUL or control characters

/// Classification of a user-supplied revision specifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RevisionKind {
    /// Exactly 40 lowercase hex characters. Immutable pin.
    FullSha,
    /// 7..=39 lowercase hex characters. Ambiguous pin — remote may resolve
    /// to a unique commit or fail with "ambiguous".
    ShortSha,
    /// Arbitrary git ref name (branch, tag, alias like "main"). Mutable —
    /// remote will resolve to the tip of that ref at pull time.
    RefName,
}

/// Default revision used when `--revision` is omitted. Mirrors the
/// huggingface_hub default (`main`).
pub const DEFAULT_REVISION: &str = "main";

/// Classify a user-supplied revision spec. Returns `Err(reason)` for
/// malformed input. All checks are offline and deterministic — no network,
/// no filesystem.
pub fn classify_revision(rev: &str) -> Result<RevisionKind, &'static str> {
    if rev.is_empty() {
        return Err("revision must not be empty");
    }
    if rev.contains("://") {
        return Err("revision must not contain '://' (pass a ref name or SHA, not a URL)");
    }
    if rev.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return Err("revision must not contain whitespace or control characters");
    }

    let is_hex = rev.chars().all(|c| c.is_ascii_digit() || matches!(c, 'a'..='f'));
    if is_hex {
        match rev.len() {
            40 => return Ok(RevisionKind::FullSha),
            7..=39 => return Ok(RevisionKind::ShortSha),
            _ => {} // fall through to RefName (e.g. a 6-char tag that happens to be hex)
        }
    }

    Ok(RevisionKind::RefName)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_sha_classified() {
        let sha = "0123456789abcdef0123456789abcdef01234567"; // 40 hex
        assert_eq!(classify_revision(sha), Ok(RevisionKind::FullSha));
    }

    #[test]
    fn short_sha_classified() {
        assert_eq!(classify_revision("abc1234"), Ok(RevisionKind::ShortSha));
        assert_eq!(
            classify_revision("0123456789abcdef0123456789abcdef0123456"), // 39 hex
            Ok(RevisionKind::ShortSha)
        );
    }

    #[test]
    fn refname_classified() {
        assert_eq!(classify_revision("main"), Ok(RevisionKind::RefName));
        assert_eq!(classify_revision("v1.0"), Ok(RevisionKind::RefName));
        assert_eq!(
            classify_revision("release/2026"),
            Ok(RevisionKind::RefName)
        );
    }

    #[test]
    fn hex_too_short_is_refname() {
        // 6 chars of hex is not a SHA (short SHA starts at 7 per git convention)
        // but remains a plausible ref name.
        assert_eq!(classify_revision("abc123"), Ok(RevisionKind::RefName));
    }

    #[test]
    fn hex_too_long_is_refname() {
        // 41+ hex chars can't be a SHA and is a valid (if strange) ref name.
        let long = "0123456789abcdef0123456789abcdef012345678"; // 41 hex
        assert_eq!(classify_revision(long), Ok(RevisionKind::RefName));
    }

    #[test]
    fn empty_rejected() {
        assert!(classify_revision("").is_err());
    }

    #[test]
    fn url_rejected() {
        assert!(classify_revision("https://example.com/x").is_err());
        assert!(classify_revision("hf://repo").is_err());
    }

    #[test]
    fn whitespace_rejected() {
        assert!(classify_revision(" main").is_err());
        assert!(classify_revision("main ").is_err());
        assert!(classify_revision("main\n").is_err());
        assert!(classify_revision("has space").is_err());
    }

    #[test]
    fn uppercase_hex_is_refname_not_sha() {
        // HF API lowercases SHAs; treat uppercase hex as a ref name, not a SHA.
        let up = "0123456789ABCDEF0123456789ABCDEF01234567";
        assert_eq!(classify_revision(up), Ok(RevisionKind::RefName));
    }

    #[test]
    fn classification_is_deterministic() {
        for input in ["main", "abc1234", "0123456789abcdef0123456789abcdef01234567"] {
            assert_eq!(classify_revision(input), classify_revision(input));
        }
    }
}
