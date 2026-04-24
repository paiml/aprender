//! `HF_ENDPOINT` custom-mirror classifier for `apr pull` (CRUX-A-08).
//!
//! Contract: `contracts/crux-A-08-v1.yaml`.
//!
//! Pure classifier pair:
//!   - `resolve_endpoint(env)` — read `HF_ENDPOINT`, validate scheme,
//!     normalize trailing slash, return canonical base URL.
//!   - `pull_url(endpoint, repo, revision, file)` — construct the
//!     deterministic `{endpoint}/{repo}/resolve/{rev}/{file}` URL.
//!
//! Both functions are pure: no I/O, no std::env reads, no network.
//! The network-level sub-claim (strace shows only mirror IPs) is
//! discharged by a separate network-gated harness (follow-up).

/// Env var that overrides the default HuggingFace hub base URL. HF's
/// own `huggingface_hub.constants.ENDPOINT` reads this same name.
pub const HF_ENDPOINT_ENV: &str = "HF_ENDPOINT";

/// Default HuggingFace Hub base URL when `HF_ENDPOINT` is unset.
/// Must be byte-equal to `huggingface_hub.constants.ENDPOINT` default.
pub const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";

/// Errors returned when the user-supplied `HF_ENDPOINT` is malformed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EndpointError {
    /// Scheme is not http or https (ftp://, file://, etc.). Contract
    /// requires exit code 2 for this class of error.
    InvalidScheme(String),
    /// Value is empty or whitespace-only.
    Empty,
    /// Value does not contain `://` — not a URL at all.
    NotAUrl(String),
}

impl std::fmt::Display for EndpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EndpointError::InvalidScheme(s) => write!(
                f,
                "HF_ENDPOINT has invalid scheme (must be http or https): {s:?}"
            ),
            EndpointError::Empty => {
                write!(f, "HF_ENDPOINT is empty or whitespace")
            }
            EndpointError::NotAUrl(s) => {
                write!(f, "HF_ENDPOINT is not a URL (missing '://'): {s:?}")
            }
        }
    }
}

impl std::error::Error for EndpointError {}

/// Resolve the effective HuggingFace hub endpoint from an env snapshot.
/// Returns the default when `HF_ENDPOINT` is unset. Trailing slash is
/// stripped so downstream URL construction can unconditionally use
/// `{endpoint}/{repo}/...`.
///
/// CRUX-A-08 ALGO-001 sub-claim of FALSIFY-001: when `HF_ENDPOINT` is
/// set to a mirror, the resolved base URL points at the mirror, not
/// at `huggingface.co`. Algorithm-level precondition for the
/// integration-level strace check.
pub fn resolve_endpoint<'a, I>(env: I) -> Result<String, EndpointError>
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    for (k, v) in env {
        if k == HF_ENDPOINT_ENV {
            return parse_endpoint(v);
        }
    }
    Ok(DEFAULT_HF_ENDPOINT.to_string())
}

/// Parse + validate a raw endpoint value. Public so CLI overrides
/// (e.g. `--endpoint FOO`) can share the exact same rules.
pub fn parse_endpoint(raw: &str) -> Result<String, EndpointError> {
    let v = raw.trim();
    if v.is_empty() {
        return Err(EndpointError::Empty);
    }

    // Must have a scheme delimiter.
    let scheme_end = v
        .find("://")
        .ok_or_else(|| EndpointError::NotAUrl(v.to_string()))?;
    let scheme = &v[..scheme_end].to_ascii_lowercase();
    if scheme != "http" && scheme != "https" {
        return Err(EndpointError::InvalidScheme(v.to_string()));
    }

    // Strip trailing slashes (including repeated).
    let normalized = v.trim_end_matches('/').to_string();
    if normalized.len() < scheme_end + 3 {
        // e.g. "http://" with nothing after the scheme — not a valid base.
        return Err(EndpointError::NotAUrl(v.to_string()));
    }
    Ok(normalized)
}

/// Build the canonical pull URL for a repo/revision/file against an
/// already-resolved endpoint. Deterministic, pure.
///
/// Formula from contract `endpoint_override`:
///   `{endpoint.rstrip('/')}/{repo}/resolve/{revision}/{file}`
pub fn pull_url(endpoint: &str, repo: &str, revision: &str, file: &str) -> String {
    let base = endpoint.trim_end_matches('/');
    let file = file.trim_start_matches('/');
    format!("{base}/{repo}/resolve/{revision}/{file}")
}

/// Return true iff the URL's host+scheme portion points at the given
/// endpoint. Used by FALSIFY-001's algorithm sub-claim to assert
/// "URL does not leak to huggingface.co".
pub fn url_targets_endpoint(url: &str, endpoint: &str) -> bool {
    let base = endpoint.trim_end_matches('/');
    url.starts_with(&format!("{base}/"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_endpoint_when_env_unset() {
        let got = resolve_endpoint(std::iter::empty::<(&str, &str)>()).unwrap();
        assert_eq!(got, "https://huggingface.co");
    }

    #[test]
    fn hf_endpoint_override_wins() {
        let got = resolve_endpoint([("HF_ENDPOINT", "https://mirror.local")]).unwrap();
        assert_eq!(got, "https://mirror.local");
    }

    #[test]
    fn trailing_slash_stripped() {
        let got = resolve_endpoint([("HF_ENDPOINT", "https://mirror.local/")]).unwrap();
        assert_eq!(got, "https://mirror.local");
    }

    #[test]
    fn multiple_trailing_slashes_stripped() {
        let got = resolve_endpoint([("HF_ENDPOINT", "https://mirror.local///")]).unwrap();
        assert_eq!(got, "https://mirror.local");
    }

    #[test]
    fn http_scheme_accepted() {
        let got = resolve_endpoint([("HF_ENDPOINT", "http://127.0.0.1:18080")]).unwrap();
        assert_eq!(got, "http://127.0.0.1:18080");
    }

    #[test]
    fn ftp_scheme_rejected() {
        let err = resolve_endpoint([("HF_ENDPOINT", "ftp://mirror.local")]).unwrap_err();
        match err {
            EndpointError::InvalidScheme(_) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn file_scheme_rejected() {
        // Contract requires `file://` rejection at parse time.
        assert!(resolve_endpoint([("HF_ENDPOINT", "file:///etc/passwd")]).is_err());
    }

    #[test]
    fn empty_hf_endpoint_rejected() {
        assert!(resolve_endpoint([("HF_ENDPOINT", "")]).is_err());
        assert!(resolve_endpoint([("HF_ENDPOINT", "   ")]).is_err());
    }

    #[test]
    fn garbage_rejected_as_not_url() {
        assert!(resolve_endpoint([("HF_ENDPOINT", "not a url")]).is_err());
    }

    #[test]
    fn scheme_only_rejected() {
        // `https://` with no host is not a usable endpoint.
        let err = resolve_endpoint([("HF_ENDPOINT", "https://")]).unwrap_err();
        match err {
            EndpointError::NotAUrl(_) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn scheme_case_is_normalized_for_validation() {
        // Scheme check is case-insensitive (HTTP/HTTPS both legal).
        assert!(resolve_endpoint([("HF_ENDPOINT", "HTTPS://mirror.local")]).is_ok());
        assert!(resolve_endpoint([("HF_ENDPOINT", "HTTP://mirror.local")]).is_ok());
    }

    #[test]
    fn unrelated_env_var_ignored() {
        let got = resolve_endpoint([("SOME_OTHER_VAR", "ftp://bad")]).unwrap();
        assert_eq!(got, "https://huggingface.co");
    }

    #[test]
    fn resolve_is_deterministic() {
        let a = resolve_endpoint([("HF_ENDPOINT", "https://mirror")]).unwrap();
        let b = resolve_endpoint([("HF_ENDPOINT", "https://mirror")]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn pull_url_default_matches_hf_formula() {
        let url = pull_url(
            DEFAULT_HF_ENDPOINT,
            "bert-base-uncased",
            "main",
            "config.json",
        );
        assert_eq!(
            url,
            "https://huggingface.co/bert-base-uncased/resolve/main/config.json"
        );
    }

    #[test]
    fn pull_url_mirror_matches_hf_formula() {
        let url = pull_url(
            "http://127.0.0.1:18080",
            "bert-base-uncased",
            "main",
            "config.json",
        );
        assert_eq!(
            url,
            "http://127.0.0.1:18080/bert-base-uncased/resolve/main/config.json"
        );
    }

    #[test]
    fn pull_url_strips_trailing_slash_on_endpoint() {
        // Double-trailing-slash safety: mirror provided as-is and also
        // through a reverse-proxy stripper should produce the same URL.
        let a = pull_url("http://127.0.0.1:18080/", "repo", "main", "file");
        let b = pull_url("http://127.0.0.1:18080", "repo", "main", "file");
        assert_eq!(a, b);
    }

    #[test]
    fn pull_url_strips_leading_slash_on_file() {
        // If the caller accidentally prefixes the file path, we don't
        // emit a double slash.
        let a = pull_url(DEFAULT_HF_ENDPOINT, "r", "main", "/file");
        let b = pull_url(DEFAULT_HF_ENDPOINT, "r", "main", "file");
        assert_eq!(a, b);
    }

    #[test]
    fn pull_url_is_deterministic() {
        let a = pull_url(DEFAULT_HF_ENDPOINT, "r", "main", "file");
        let b = pull_url(DEFAULT_HF_ENDPOINT, "r", "main", "file");
        assert_eq!(a, b);
    }

    #[test]
    fn url_targets_endpoint_agreement() {
        let mirror = "http://127.0.0.1:18080";
        let url = pull_url(mirror, "r", "main", "f");
        assert!(url_targets_endpoint(&url, mirror));
        assert!(!url_targets_endpoint(&url, DEFAULT_HF_ENDPOINT));
    }

    #[test]
    fn falsify_001_sub_claim_mirror_keeps_url_off_huggingface_co() {
        // CRUX-A-08 ALGO-001 sub-claim of FALSIFY-001: with
        // HF_ENDPOINT=http://127.0.0.1:18080 set, the constructed
        // pull URL MUST target the mirror and MUST NOT contain
        // "huggingface.co" anywhere. Algorithm-level precondition
        // for the integration-level strace "zero leaks to canonical
        // host" assertion.
        let mirror = "http://127.0.0.1:18080";
        let endpoint = resolve_endpoint([("HF_ENDPOINT", mirror)]).unwrap();
        assert_eq!(endpoint, mirror);
        let url = pull_url(&endpoint, "bert-base-uncased", "main", "config.json");
        assert!(
            !url.contains("huggingface.co"),
            "URL leaked to huggingface.co despite HF_ENDPOINT override: {url}",
        );
        assert!(
            url.contains("127.0.0.1:18080"),
            "URL does not target mirror: {url}"
        );
        assert!(url_targets_endpoint(&url, mirror));
    }

    #[test]
    fn falsify_001_sub_claim_unset_uses_canonical_host() {
        // Converse: with HF_ENDPOINT UNSET, the URL MUST target
        // huggingface.co — so we know the override machinery is
        // gate-sensitive, not just "always matches mirror".
        let endpoint = resolve_endpoint(std::iter::empty::<(&str, &str)>()).unwrap();
        assert_eq!(endpoint, "https://huggingface.co");
        let url = pull_url(&endpoint, "r", "main", "f");
        assert!(url.contains("huggingface.co"), "unexpected default: {url}");
    }

    #[test]
    fn hf_endpoint_env_name_stable() {
        // Downstream tests + docs depend on this exact name.
        assert_eq!(HF_ENDPOINT_ENV, "HF_ENDPOINT");
    }

    #[test]
    fn default_endpoint_stable() {
        assert_eq!(DEFAULT_HF_ENDPOINT, "https://huggingface.co");
    }
}
