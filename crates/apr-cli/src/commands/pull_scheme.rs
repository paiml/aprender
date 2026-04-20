//! Pull-source URL scheme classifier for `apr pull` (CRUX-A-15).
//!
//! Contract: `contracts/crux-A-15-v1.yaml`.
//!
//! Pure classifier — takes the raw string passed to `apr pull` and returns
//! the transport category the caller should use. No I/O, no filesystem,
//! no network.
//!
//! The actual filesystem walk for `file://` paths, the HF cache lookup for
//! `hf://` with `HF_HUB_OFFLINE=1`, and the byte-identical copy are all
//! discharged by separate network/strace-gated harnesses (follow-up).
//!
//! The algorithm-level sub-claim we DO discharge here: a `file://` URL is
//! classified as `Local(path)` and therefore will not be sent to any HTTP
//! transport. This is a necessary (not sufficient) condition for the full
//! "zero outbound TCP" invariant of FALSIFY-CRUX-A-15-001.

use std::path::PathBuf;

/// Classification of a user-supplied pull source string.
///
/// The variants map 1:1 to the transport that `apr pull` should dispatch:
/// `Local` → filesystem copy; `HfHub` → HuggingFace Hub HTTPS; `Https` →
/// direct HTTPS; `BareOrgRepo` / `ShortName` → resolved via the alias map
/// before dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PullScheme {
    /// `file:///path/to/dir` → pure filesystem copy, never touches network.
    Local(PathBuf),
    /// `hf://org/repo[@rev]` → HuggingFace Hub transport.
    HfHub(String),
    /// `https://host/...` → direct HTTPS download.
    Https(String),
    /// `org/repo` (no scheme, contains slash) → treat as `hf://org/repo`.
    BareOrgRepo(String),
    /// Single token, no scheme, no slash → alias lookup required.
    ShortName(String),
}

/// Classify a raw pull-source string.
///
/// Returns `Err` when the scheme is obviously malformed (empty string,
/// unsupported scheme, `file:` without the `//` separator).
///
/// No I/O — this is a pure function. It does NOT check whether the
/// `file://` path exists on disk; that is the caller's concern.
pub fn classify_pull_source(src: &str) -> Result<PullScheme, String> {
    let trimmed = src.trim();
    if trimmed.is_empty() {
        return Err("empty pull source".to_string());
    }

    if let Some(rest) = trimmed.strip_prefix("file://") {
        return Ok(PullScheme::Local(PathBuf::from(rest)));
    }
    if let Some(rest) = trimmed.strip_prefix("hf://") {
        if rest.is_empty() {
            return Err("hf:// with empty repo".to_string());
        }
        return Ok(PullScheme::HfHub(rest.to_string()));
    }
    if trimmed.starts_with("https://") || trimmed.starts_with("http://") {
        return Ok(PullScheme::Https(trimmed.to_string()));
    }

    // Scheme-bearing but unsupported? Any `scheme:` prefix that isn't one
    // of the known ones is an error. We detect this via `://` rather than
    // `:` alone so `org/repo` with a colon in the repo name still flows
    // through. (HF repo names can contain colons in tags but not here.)
    if let Some(idx) = trimmed.find("://") {
        let scheme = &trimmed[..idx];
        return Err(format!("unsupported URL scheme: {scheme:?}"));
    }

    // `file:` without the `//` separator is almost certainly a typo.
    if trimmed.starts_with("file:") {
        return Err("file: scheme requires file:// prefix".to_string());
    }

    // No scheme, has a slash → bare `org/repo`.
    if trimmed.contains('/') {
        return Ok(PullScheme::BareOrgRepo(trimmed.to_string()));
    }

    // No scheme, no slash → short alias.
    Ok(PullScheme::ShortName(trimmed.to_string()))
}

/// True iff the classification corresponds to a filesystem-only transport.
/// Callers use this to skip any network-layer setup for local pulls.
///
/// The CRUX-A-15 ALGO-001 sub-claim of FALSIFY-CRUX-A-15-001 is exactly:
/// "for any `src` starting with `file://`, `is_local_transport(classify)`
/// returns true". This is a necessary condition for zero outbound TCP.
pub fn is_local_transport(scheme: &PullScheme) -> bool {
    matches!(scheme, PullScheme::Local(_))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_scheme_absolute_path() {
        let r = classify_pull_source("file:///tmp/models/a").unwrap();
        assert_eq!(r, PullScheme::Local(PathBuf::from("/tmp/models/a")));
        assert!(is_local_transport(&r));
    }

    #[test]
    fn file_scheme_relative_path() {
        let r = classify_pull_source("file://models/a").unwrap();
        assert_eq!(r, PullScheme::Local(PathBuf::from("models/a")));
        assert!(is_local_transport(&r));
    }

    #[test]
    fn hf_scheme_classifies_as_hf_hub() {
        let r = classify_pull_source("hf://Qwen/Qwen2.5-Coder-1.5B").unwrap();
        assert_eq!(r, PullScheme::HfHub("Qwen/Qwen2.5-Coder-1.5B".to_string()));
        assert!(!is_local_transport(&r));
    }

    #[test]
    fn https_scheme_classifies_as_https() {
        let r =
            classify_pull_source("https://huggingface.co/org/repo/resolve/main/config.json")
                .unwrap();
        assert!(matches!(r, PullScheme::Https(_)));
        assert!(!is_local_transport(&r));
    }

    #[test]
    fn bare_org_slash_repo_classifies_as_bare() {
        let r = classify_pull_source("openai-community/gpt2").unwrap();
        assert_eq!(r, PullScheme::BareOrgRepo("openai-community/gpt2".to_string()));
        assert!(!is_local_transport(&r));
    }

    #[test]
    fn short_name_classifies_as_short_name() {
        let r = classify_pull_source("qwen2.5-coder").unwrap();
        assert_eq!(r, PullScheme::ShortName("qwen2.5-coder".to_string()));
        assert!(!is_local_transport(&r));
    }

    #[test]
    fn empty_input_is_error() {
        assert!(classify_pull_source("").is_err());
        assert!(classify_pull_source("   ").is_err());
    }

    #[test]
    fn unsupported_scheme_is_error() {
        let err = classify_pull_source("ftp://example.com/x").unwrap_err();
        assert!(err.contains("ftp"), "error should name the scheme: {err}");
        let err = classify_pull_source("s3://bucket/key").unwrap_err();
        assert!(err.contains("s3"), "error should name the scheme: {err}");
    }

    #[test]
    fn file_without_slash_slash_is_error() {
        let err = classify_pull_source("file:/tmp/x").unwrap_err();
        assert!(
            err.contains("file://"),
            "error should hint at correct prefix: {err}"
        );
    }

    #[test]
    fn hf_with_empty_repo_is_error() {
        assert!(classify_pull_source("hf://").is_err());
    }

    #[test]
    fn leading_and_trailing_whitespace_ignored() {
        let r = classify_pull_source("  file:///tmp/a  ").unwrap();
        assert_eq!(r, PullScheme::Local(PathBuf::from("/tmp/a")));
    }

    #[test]
    fn is_local_transport_rejects_non_local() {
        for src in [
            "hf://a/b",
            "https://x/y",
            "openai/gpt2",
            "qwen2.5-coder",
        ] {
            let r = classify_pull_source(src).unwrap();
            assert!(
                !is_local_transport(&r),
                "{src:?} must not be classified as Local transport"
            );
        }
    }

    #[test]
    fn file_scheme_is_always_local_transport() {
        // CRUX-A-15 ALGO-001 sub-claim of FALSIFY-001: any input starting
        // with `file://` must classify as Local and therefore route to the
        // filesystem transport, not HTTP. This is a necessary condition
        // for "zero outbound TCP connect()".
        for src in [
            "file:///",
            "file:///a",
            "file:///tmp/models/a",
            "file://relative/path",
            "file:///path/with/many/segments/model.apr",
        ] {
            let r = classify_pull_source(src).unwrap();
            assert!(
                is_local_transport(&r),
                "{src:?} must be classified as Local transport"
            );
        }
    }

    #[test]
    fn is_deterministic() {
        let a = classify_pull_source("file:///tmp/x").unwrap();
        let b = classify_pull_source("file:///tmp/x").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn classification_is_total_over_all_strings() {
        // Every non-empty classification either returns Ok or a specific
        // Err — never panics. Exercise a small bag of weird inputs.
        for src in [
            "🎉",
            "x",
            "a/b/c",
            "https://",
            "hf://",
            "file://",
            "arbitrary-nonsense::with::colons",
        ] {
            let _ = classify_pull_source(src); // must not panic
        }
    }
}
