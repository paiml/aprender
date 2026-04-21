//! Shared model-cache root + blob-dedup-key classifier for `APR_MODELS`
//! (CRUX-A-21). Parity with Ollama's `OLLAMA_MODELS` for multi-user /
//! systemd-managed deployments.
//!
//! Contract: `contracts/crux-A-21-v1.yaml`.
//!
//! Three pure algorithm-level sub-claims live here:
//!
//! 1. `resolve_registry_root(env, home)` is deterministic: if `APR_MODELS`
//!    is set non-empty it wins; otherwise `$HOME/.apr/models`. No I/O —
//!    the caller is responsible for creating the directory.
//!
//! 2. `blob_path_for(root, sha256_hex)` produces `{root}/blobs/sha256-<hex>`.
//!    Dedup across users is guaranteed *by construction*: two processes
//!    pulling the same content-addressed blob write to the same path, so
//!    the filesystem collapses to one inode — the necessary condition
//!    for FALSIFY-CRUX-A-21-001.
//!
//! 3. `classify_pull_permission_outcome(io_err_kind)` maps an
//!    I/O-error variant onto the contract-defined exit code (13 for
//!    permission-denied, with an actionable daemon-user hint). This is
//!    the necessary condition for FALSIFY-CRUX-A-21-002: the pipeline
//!    must exit 13 on EACCES, not silently fall back to `$HOME`.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

/// Well-known env var names. Matches Ollama's convention for parity.
pub const APR_MODELS_ENV: &str = "APR_MODELS";
/// Default subdirectory under the user's home when `APR_MODELS` is unset.
pub const DEFAULT_REGISTRY_SUBDIR: &str = ".apr/models";
/// Subdirectory inside the registry root where content-addressed blobs live.
pub const BLOBS_SUBDIR: &str = "blobs";

/// Resolve the registry root for `apr pull`.
///
/// Precedence:
///   1. `apr_models_env` if `Some(...)` and non-empty after trim
///   2. `home/.apr/models`
///
/// Returns `Err` if `home` is empty (no fallback available).
pub fn resolve_registry_root(
    apr_models_env: Option<&str>,
    home: &Path,
) -> Result<PathBuf, RegistryRootError> {
    if let Some(env) = apr_models_env {
        let trimmed = env.trim();
        if !trimmed.is_empty() {
            return Ok(PathBuf::from(trimmed));
        }
    }
    if home.as_os_str().is_empty() {
        return Err(RegistryRootError::MissingHome);
    }
    Ok(home.join(DEFAULT_REGISTRY_SUBDIR))
}

/// Error cases for `resolve_registry_root`.
#[derive(Debug, PartialEq, Eq)]
pub enum RegistryRootError {
    /// Neither `APR_MODELS` nor `home` was usable. Callers MUST exit with
    /// a configuration error rather than silently picking `/tmp` or similar.
    MissingHome,
}

/// Compute the on-disk path for a content-addressed blob.
///
/// The `sha256_hex` must be the lowercase 64-character hex digest of the
/// blob contents. Returns `Err` for empty / malformed digests because
/// collapsing a malformed digest into a shared blob dir would silently
/// lose dedup correctness.
pub fn blob_path_for(root: &Path, sha256_hex: &str) -> Result<PathBuf, BlobPathError> {
    if sha256_hex.len() != 64 {
        return Err(BlobPathError::WrongLength(sha256_hex.len()));
    }
    if !sha256_hex
        .bytes()
        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(BlobPathError::NonHexLower);
    }
    let filename = format!("sha256-{sha256_hex}");
    Ok(root.join(BLOBS_SUBDIR).join(filename))
}

/// Error cases for `blob_path_for`.
#[derive(Debug, PartialEq, Eq)]
pub enum BlobPathError {
    /// sha256 hex must be exactly 64 characters.
    WrongLength(usize),
    /// Must be lowercase ASCII hex; uppercase or non-hex bytes are rejected.
    NonHexLower,
}

/// Contract-defined exit outcome for an `apr pull` permission scenario.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PullPermissionOutcome {
    /// All good — proceed with the fetch.
    Ok,
    /// EACCES on the registry root → exit 13 with a daemon-user hint.
    /// The hint is the fixed string returned here so callers emit a
    /// consistent user-facing message across platforms.
    PermissionDenied { exit_code: i32, hint: &'static str },
    /// Registry root is missing and cannot be created.
    NotFound { exit_code: i32 },
    /// Any other I/O failure — generic exit 1.
    Other { exit_code: i32, kind: ErrorKind },
}

/// Map an `std::io::ErrorKind` onto the contract's exit-code table.
///
/// Contract FALSIFY-CRUX-A-21-002: an unprivileged user must see exit
/// code 13 (Unix convention for EACCES) with a hint pointing at the
/// daemon user, not a silent fall-through that writes to `$HOME`.
pub fn classify_pull_permission_outcome(kind: ErrorKind) -> PullPermissionOutcome {
    match kind {
        ErrorKind::PermissionDenied => PullPermissionOutcome::PermissionDenied {
            exit_code: 13,
            hint: "permission denied on APR_MODELS — run as daemon user or adjust mode",
        },
        ErrorKind::NotFound => PullPermissionOutcome::NotFound { exit_code: 1 },
        other => PullPermissionOutcome::Other {
            exit_code: 1,
            kind: other,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== resolve_registry_root =====

    #[test]
    fn env_wins_when_set() {
        let got =
            resolve_registry_root(Some("/var/lib/apr/models"), Path::new("/home/user")).unwrap();
        assert_eq!(got, PathBuf::from("/var/lib/apr/models"));
    }

    #[test]
    fn env_whitespace_trimmed() {
        let got = resolve_registry_root(Some("  /srv/apr  "), Path::new("/home/u")).unwrap();
        assert_eq!(got, PathBuf::from("/srv/apr"));
    }

    #[test]
    fn empty_env_falls_back_to_home() {
        for env in [Some(""), Some("   "), None] {
            let got = resolve_registry_root(env, Path::new("/home/u")).unwrap();
            assert_eq!(got, PathBuf::from("/home/u/.apr/models"));
        }
    }

    #[test]
    fn missing_home_without_env_errors() {
        let err = resolve_registry_root(None, Path::new("")).unwrap_err();
        assert_eq!(err, RegistryRootError::MissingHome);
    }

    #[test]
    fn env_alone_sufficient_even_without_home() {
        // If APR_MODELS is set, we do NOT need $HOME at all — systemd
        // deploys often have a fictitious daemon $HOME.
        let got = resolve_registry_root(Some("/var/apr"), Path::new("")).unwrap();
        assert_eq!(got, PathBuf::from("/var/apr"));
    }

    #[test]
    fn resolve_is_deterministic() {
        let a = resolve_registry_root(Some("/x"), Path::new("/h")).unwrap();
        let b = resolve_registry_root(Some("/x"), Path::new("/h")).unwrap();
        assert_eq!(a, b);
    }

    // ===== blob_path_for =====

    #[test]
    fn blob_path_uses_sha256_prefix() {
        let hex = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let p = blob_path_for(Path::new("/var/apr"), hex).unwrap();
        assert_eq!(
            p,
            PathBuf::from(
                "/var/apr/blobs/sha256-e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            )
        );
    }

    #[test]
    fn blob_path_dedup_is_by_construction() {
        // FALSIFY-CRUX-A-21-001 sub-claim: identical sha256 under the
        // same root produces identical path. The filesystem collapses
        // two concurrent pulls to one inode automatically.
        let hex = "a".repeat(64);
        let a = blob_path_for(Path::new("/shared"), &hex).unwrap();
        let b = blob_path_for(Path::new("/shared"), &hex).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn blob_path_different_hash_yields_different_path() {
        let a = blob_path_for(Path::new("/r"), &"a".repeat(64)).unwrap();
        let b = blob_path_for(Path::new("/r"), &"b".repeat(64)).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn blob_path_rejects_wrong_length() {
        for hex in ["", "abc", &"a".repeat(63), &"a".repeat(65)] {
            let err = blob_path_for(Path::new("/r"), hex).unwrap_err();
            assert!(matches!(err, BlobPathError::WrongLength(_)));
        }
    }

    #[test]
    fn blob_path_rejects_uppercase_hex() {
        // Mixed case would break dedup (two different paths for the
        // same content). Reject uppercase outright.
        let hex = "A".repeat(64);
        assert_eq!(
            blob_path_for(Path::new("/r"), &hex).unwrap_err(),
            BlobPathError::NonHexLower
        );
    }

    #[test]
    fn blob_path_rejects_non_hex_bytes() {
        let mut hex = "a".repeat(63);
        hex.push('z');
        assert_eq!(
            blob_path_for(Path::new("/r"), &hex).unwrap_err(),
            BlobPathError::NonHexLower
        );
    }

    #[test]
    fn blob_dir_is_always_under_root() {
        let hex = "0".repeat(64);
        let root = Path::new("/some/registry/root");
        let p = blob_path_for(root, &hex).unwrap();
        assert!(p.starts_with(root));
        assert_eq!(p.parent().unwrap(), root.join(BLOBS_SUBDIR));
    }

    // ===== classify_pull_permission_outcome =====

    #[test]
    fn permission_denied_yields_exit_13_with_hint() {
        // FALSIFY-CRUX-A-21-002: EACCES must exit 13 with actionable hint.
        let r = classify_pull_permission_outcome(ErrorKind::PermissionDenied);
        match r {
            PullPermissionOutcome::PermissionDenied { exit_code, hint } => {
                assert_eq!(exit_code, 13);
                assert!(
                    hint.to_lowercase().contains("permission"),
                    "hint should mention permission: {hint}"
                );
                assert!(
                    hint.to_lowercase().contains("daemon")
                        || hint.to_lowercase().contains("mode"),
                    "hint should be actionable (daemon/mode): {hint}"
                );
            }
            other => panic!("expected PermissionDenied, got {other:?}"),
        }
    }

    #[test]
    fn not_found_yields_exit_1() {
        let r = classify_pull_permission_outcome(ErrorKind::NotFound);
        assert_eq!(r, PullPermissionOutcome::NotFound { exit_code: 1 });
    }

    #[test]
    fn generic_io_errors_yield_exit_1() {
        for k in [
            ErrorKind::UnexpectedEof,
            ErrorKind::Interrupted,
            ErrorKind::TimedOut,
        ] {
            match classify_pull_permission_outcome(k) {
                PullPermissionOutcome::Other { exit_code, kind } => {
                    assert_eq!(exit_code, 1);
                    assert_eq!(kind, k);
                }
                other => panic!("expected Other for {k:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn permission_classifier_is_deterministic() {
        let a = classify_pull_permission_outcome(ErrorKind::PermissionDenied);
        let b = classify_pull_permission_outcome(ErrorKind::PermissionDenied);
        assert_eq!(a, b);
    }

    #[test]
    fn permission_classifier_never_returns_ok_on_eacces() {
        // Strengthens FALSIFY-CRUX-A-21-002: no path through the
        // classifier returns `Ok` when `ErrorKind::PermissionDenied` is
        // passed. The silent-fall-through bug that motivated the gate
        // is structurally impossible here.
        let r = classify_pull_permission_outcome(ErrorKind::PermissionDenied);
        assert!(!matches!(r, PullPermissionOutcome::Ok));
    }
}
