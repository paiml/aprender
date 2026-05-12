//! Pull-source URL scheme classifier for `apr pull` (CRUX-A-14, CRUX-A-15).
//!
//! Contracts: `contracts/crux-A-14-v1.yaml` (multi-cloud object-store URIs) +
//! `contracts/crux-A-15-v1.yaml` (file://, hf://, https:// transports).
//!
//! Pure classifier — takes the raw string passed to `apr pull` and returns
//! the transport category the caller should use. No I/O, no filesystem,
//! no network.
//!
//! CRUX-A-14 algorithm-level sub-claim discharged here: `s3://`, `gs://`,
//! and `az://` are classified as their respective object-store transport
//! (never routed to HTTP/HF). The paired
//! `required_credential_env_for_scheme` helper names the standard credential
//! env var so downstream transport code can honor the invariant "no
//! aprender-specific auth". Actual network fetch + sha256 parity is gated by
//! a MinIO/GCS/Azure live harness (follow-up).
//!
//! CRUX-A-15 algorithm-level sub-claim discharged here: a `file://` URL is
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
    /// `s3://bucket/key` → AWS S3 (or S3-compatible: MinIO, Cloudflare R2).
    S3 { bucket: String, key: String },
    /// `gs://bucket/object` → Google Cloud Storage.
    Gs { bucket: String, object: String },
    /// `az://container/blob` → Azure Blob Storage.
    Az { container: String, blob: String },
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
    if let Some(rest) = trimmed.strip_prefix("s3://") {
        let (bucket, key) = split_bucket_key(rest, "s3")?;
        return Ok(PullScheme::S3 { bucket, key });
    }
    if let Some(rest) = trimmed.strip_prefix("gs://") {
        let (bucket, object) = split_bucket_key(rest, "gs")?;
        return Ok(PullScheme::Gs { bucket, object });
    }
    if let Some(rest) = trimmed.strip_prefix("az://") {
        let (container, blob) = split_bucket_key(rest, "az")?;
        return Ok(PullScheme::Az { container, blob });
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

/// True iff the classification corresponds to a cloud object-store transport
/// (S3/GCS/Azure). Callers use this to dispatch to the right SDK client.
///
/// CRUX-A-14 ALGO-001 sub-claim of FALSIFY-CRUX-A-14-001/002: any
/// well-formed `s3://`, `gs://`, or `az://` URI classifies as the
/// corresponding object-store variant (never HTTPS, never HF, never local).
/// Any *other* URL scheme (`ftp://`, `ssh://`, etc.) must error — a
/// necessary condition for the "exit 2 on unsupported scheme" gate.
pub fn is_object_store_transport(scheme: &PullScheme) -> bool {
    matches!(
        scheme,
        PullScheme::S3 { .. } | PullScheme::Gs { .. } | PullScheme::Az { .. }
    )
}

/// Name of the standard environment variable that downstream transport
/// code should look up credentials from, for the given scheme. Returns
/// `None` for transports that don't require credentials (local, https) or
/// whose credentials are HF-specific (`HF_TOKEN`, handled elsewhere).
///
/// CRUX-A-14 INV-A-14-002: "credentials are pulled from standard env/config
/// — no aprender-specific auth". This helper encodes that mapping so no
/// downstream code ever invents an `APR_S3_KEY` or similar.
pub fn required_credential_env_for_scheme(scheme: &PullScheme) -> Option<&'static str> {
    match scheme {
        PullScheme::S3 { .. } => Some("AWS_ACCESS_KEY_ID"),
        PullScheme::Gs { .. } => Some("GOOGLE_APPLICATION_CREDENTIALS"),
        PullScheme::Az { .. } => Some("AZURE_STORAGE_KEY"),
        _ => None,
    }
}

/// Split a bucket/key (or container/blob) tail into its two parts,
/// erroring if the bucket or key is empty. Factored out so S3/GS/Azure
/// share the same parser and error shape.
fn split_bucket_key(rest: &str, scheme: &str) -> Result<(String, String), String> {
    match rest.split_once('/') {
        Some((b, k)) if !b.is_empty() && !k.is_empty() => Ok((b.to_string(), k.to_string())),
        _ => Err(format!("{scheme}:// requires bucket/key form")),
    }
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
        let r = classify_pull_source("https://huggingface.co/org/repo/resolve/main/config.json")
            .unwrap();
        assert!(matches!(r, PullScheme::Https(_)));
        assert!(!is_local_transport(&r));
    }

    #[test]
    fn bare_org_slash_repo_classifies_as_bare() {
        let r = classify_pull_source("openai-community/gpt2").unwrap();
        assert_eq!(
            r,
            PullScheme::BareOrgRepo("openai-community/gpt2".to_string())
        );
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
        for src in ["ftp://example.com/x", "ssh://host/x", "rsync://x/y"] {
            let err = classify_pull_source(src).unwrap_err();
            assert!(err.contains("unsupported"), "expect rejection: {err}");
        }
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
            "s3://b/k",
            "gs://b/o",
            "az://c/b",
        ] {
            let r = classify_pull_source(src).unwrap();
            assert!(
                !is_local_transport(&r),
                "{src:?} must not be classified as Local transport"
            );
        }
    }

    // ===== CRUX-A-14 object-store classifier tests =====

    #[test]
    fn s3_scheme_classifies_as_s3() {
        let r = classify_pull_source("s3://aprtest/model.gguf").unwrap();
        assert_eq!(
            r,
            PullScheme::S3 {
                bucket: "aprtest".to_string(),
                key: "model.gguf".to_string()
            }
        );
        assert!(is_object_store_transport(&r));
        assert!(!is_local_transport(&r));
        assert_eq!(
            required_credential_env_for_scheme(&r),
            Some("AWS_ACCESS_KEY_ID")
        );
    }

    #[test]
    fn gs_scheme_classifies_as_gcs() {
        let r = classify_pull_source("gs://my-bucket/path/to/model.safetensors").unwrap();
        assert_eq!(
            r,
            PullScheme::Gs {
                bucket: "my-bucket".to_string(),
                object: "path/to/model.safetensors".to_string()
            }
        );
        assert!(is_object_store_transport(&r));
        assert_eq!(
            required_credential_env_for_scheme(&r),
            Some("GOOGLE_APPLICATION_CREDENTIALS")
        );
    }

    #[test]
    fn az_scheme_classifies_as_azure() {
        let r = classify_pull_source("az://models-container/qwen.gguf").unwrap();
        assert_eq!(
            r,
            PullScheme::Az {
                container: "models-container".to_string(),
                blob: "qwen.gguf".to_string()
            }
        );
        assert!(is_object_store_transport(&r));
        assert_eq!(
            required_credential_env_for_scheme(&r),
            Some("AZURE_STORAGE_KEY")
        );
    }

    #[test]
    fn object_store_requires_bucket_and_key() {
        for src in [
            "s3://",
            "s3://bucket",
            "s3://bucket/",
            "s3:///key",
            "gs://",
            "gs://bucket",
            "az://",
            "az://container",
        ] {
            let err = classify_pull_source(src).unwrap_err();
            assert!(
                err.contains("bucket/key"),
                "expect bucket/key-shape rejection for {src:?}: {err}"
            );
        }
    }

    #[test]
    fn object_store_transports_never_local_or_hf() {
        for src in ["s3://b/k", "gs://b/o", "az://c/b"] {
            let r = classify_pull_source(src).unwrap();
            assert!(!is_local_transport(&r), "{src} must not be local");
            assert!(
                !matches!(r, PullScheme::HfHub(_)),
                "{src} must not be HF Hub"
            );
            assert!(
                !matches!(r, PullScheme::Https(_)),
                "{src} must not be HTTPS"
            );
        }
    }

    #[test]
    fn is_object_store_transport_rejects_non_object_store() {
        for src in [
            "file:///tmp/x",
            "hf://a/b",
            "https://x/y",
            "openai/gpt2",
            "qwen",
        ] {
            let r = classify_pull_source(src).unwrap();
            assert!(
                !is_object_store_transport(&r),
                "{src:?} must not be object-store transport"
            );
            assert_eq!(
                required_credential_env_for_scheme(&r),
                None,
                "{src:?} must not require object-store credential"
            );
        }
    }

    #[test]
    fn credential_env_names_are_standard_not_aprender_specific() {
        // CRUX-A-14 INV-A-14-002: no aprender-specific auth env vars.
        for src in ["s3://b/k", "gs://b/o", "az://c/b"] {
            let r = classify_pull_source(src).unwrap();
            let env = required_credential_env_for_scheme(&r).unwrap();
            assert!(
                !env.starts_with("APR_") && !env.starts_with("APRENDER_"),
                "{src:?} -> {env}: credential must be a standard SDK env var"
            );
        }
    }

    #[test]
    fn object_store_keys_preserve_path_separators() {
        let r = classify_pull_source("s3://bucket/a/b/c/model.gguf").unwrap();
        match r {
            PullScheme::S3 { key, .. } => assert_eq!(key, "a/b/c/model.gguf"),
            other => panic!("expected S3, got {other:?}"),
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
