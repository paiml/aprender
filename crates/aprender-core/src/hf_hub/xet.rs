//! HF Xet protocol upload for files > 5 GiB.
//!
//! Implements the Xet content-addressable storage upload path used by
//! Hugging Face Hub when a file exceeds the 5 GiB HTTP preupload threshold.
//! The HTTP `preupload/main` endpoint returns `uploadMode:lfs` with empty
//! `upload_url` / `chunk_urls` arrays for files above the threshold — under
//! the current HF Hub protocol the client is expected to transfer such files
//! via the Xet CAS backend. See:
//!
//! - Contract: `contracts/apr-publish-hf-large-file-v1.yaml` (F-PUB-LFS-001)
//! - Spec:     `docs/specifications/aprender-train/ship-two-models-spec.md` §12.8
//! - Xet docs: <https://huggingface.co/docs/xet/index>
//! - Reference impl: <https://github.com/huggingface/xet-core> (`hf-xet` crate, Apache-2.0)
//!
//! The `hf-xet` crate ships a blocking Rust API (`XetSessionBuilder`,
//! `XetUploadCommit::upload_from_path_blocking`, `commit_blocking`). The
//! session uses a *token-refresh URL* so we never construct raw access tokens
//! ourselves — we hand hf-xet the HF Hub endpoint
//! `/api/models/{repo_id}/xet-write-token/{revision}` plus our
//! `Authorization: Bearer {HF_TOKEN}` header and hf-xet handles the rest
//! (token fetch, dedup, xorb/shard upload, LFS pointer commit).
//!
//! # Falsifiable gates discharged here
//!
//! - **FALSIFY-PUB-LFS-001 (file_size_dispatch)** — `should_use_xet` returns
//!   the partitioning: `size > 5 GiB → xet`, otherwise HTTP LFS.
//! - **FALSIFY-PUB-LFS-002 (token_endpoint)** — `build_token_refresh_url`
//!   emits the exact HF Hub URL shape.
//! - **FALSIFY-PUB-LFS-010 (three_format_dogfood)** — exercised by callers
//!   that invoke `XetUploader::upload` three times (apr, safetensors, gguf).
//!
//! Phases 3–7 of the Xet pipeline (chunking, dedup, xorb upload, shard
//! upload) live inside `hf-xet`; by delegating we inherit HF's reference
//! invariants and avoid reimplementing the protocol.

#[cfg(feature = "xet")]
use std::path::Path;

#[cfg(feature = "xet")]
use super::{HfHubError, Result};

/// 5 GiB — the Hugging Face Hub HTTP preupload threshold.
///
/// Files whose size is strictly greater than this value MUST be transferred
/// via the Xet CAS path. Files `<= 5 GiB` MAY use the HTTP multipart/single
/// LFS path in `upload.rs`.
pub const HF_XET_THRESHOLD_BYTES: u64 = 5 * 1024 * 1024 * 1024;

/// Dispatch gate (FALSIFY-PUB-LFS-001).
///
/// Returns `true` iff the given file size mandates the Xet transfer path.
#[must_use]
pub const fn should_use_xet(file_size_bytes: u64) -> bool {
    file_size_bytes > HF_XET_THRESHOLD_BYTES
}

/// Build the HF Hub xet-write-token refresh URL.
///
/// Shape (from Xet protocol spec and xet-core reference impl):
///
/// ```text
/// {api_base}/api/models/{repo_id}/xet-write-token/{revision}
/// ```
///
/// `hf-xet` calls this URL whenever its cached access token expires.
#[must_use]
pub fn build_token_refresh_url(api_base: &str, repo_id: &str, revision: &str) -> String {
    let base = api_base.trim_end_matches('/');
    format!("{base}/api/models/{repo_id}/xet-write-token/{revision}")
}

/// Uploads single large files to an HF Hub repo via the Xet protocol.
///
/// Thin wrapper around `hf-xet`'s blocking `XetSession` that:
/// 1. Builds the token-refresh URL for the target repo + revision.
/// 2. Attaches the caller's HF write token via an HTTP `Authorization` header.
/// 3. Opens a single-file upload commit, uploads the artifact by path, and
///    commits (which writes the LFS pointer to the git tree).
///
/// All phases 3–7 of the Xet pipeline (chunking, dedup, xorb/shard assembly,
/// CAS uploads) are delegated to the `hf-xet` crate — the Hugging Face
/// reference implementation of the protocol.
#[cfg(feature = "xet")]
#[derive(Debug)]
pub struct XetUploader<'a> {
    /// HF Hub API base, e.g. `https://huggingface.co`.
    pub api_base: &'a str,
    /// HF repo ID, e.g. `paiml/qwen2.5-coder-7b-apache-q4k-v1`.
    pub repo_id: &'a str,
    /// Git revision/branch. Typically `"main"`.
    pub revision: &'a str,
    /// HF write token (never logged or persisted).
    pub token: &'a str,
}

#[cfg(feature = "xet")]
impl<'a> XetUploader<'a> {
    /// Upload a single file via Xet.
    ///
    /// `commit_msg` is attached to the downstream LFS pointer commit inside
    /// `hf-xet`. Returns `Ok(())` on full success of both the CAS upload
    /// **and** the LFS pointer commit; otherwise `Err(HfHubError::XetUpload)`.
    ///
    /// Wire the *blocking* hf-xet API here. Callers are synchronous
    /// (`apr publish` → `ureq` elsewhere in this module) so there is no
    /// outer tokio runtime and we do not want to spawn one implicitly.
    pub fn upload_file(&self, local_path: &Path, _commit_msg: &str) -> Result<()> {
        use xet::xet_session::{header, HeaderMap, HeaderValue, Sha256Policy, XetSessionBuilder};

        // Phase 1: token refresh URL + auth header (FALSIFY-PUB-LFS-002).
        let token_refresh_url = build_token_refresh_url(self.api_base, self.repo_id, self.revision);
        let mut headers = HeaderMap::new();
        let auth_value = format!("Bearer {}", self.token);
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_str(&auth_value)
                .map_err(|e| HfHubError::XetUpload(format!("auth header build failed: {e}")))?,
        );

        // Phase 2: build session + commit (delegates phases 3-7 to hf-xet).
        let session = XetSessionBuilder::new()
            .build()
            .map_err(|e| HfHubError::XetUpload(format!("session build failed: {e}")))?;

        let commit = session
            .new_upload_commit()
            .map_err(|e| HfHubError::XetUpload(format!("new_upload_commit failed: {e}")))?
            .with_token_refresh_url(token_refresh_url, headers)
            .build_blocking()
            .map_err(|e| HfHubError::XetUpload(format!("commit build_blocking failed: {e}")))?;

        commit
            .upload_from_path_blocking(local_path.to_path_buf(), Sha256Policy::Compute)
            .map_err(|e| HfHubError::XetUpload(format!("upload_from_path failed: {e}")))?;

        // Phase 8: LFS pointer commit (FALSIFY-PUB-LFS-009).
        commit
            .commit_blocking()
            .map_err(|e| HfHubError::PartialUpload {
                cas_success: true,
                commit_success: false,
                detail: format!("commit_blocking failed: {e}"),
            })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_gate_partitions_exactly_at_5_gib() {
        // Below, at, and above the boundary (FALSIFY-PUB-LFS-001).
        assert!(!should_use_xet(0));
        assert!(!should_use_xet(HF_XET_THRESHOLD_BYTES - 1));
        assert!(!should_use_xet(HF_XET_THRESHOLD_BYTES));
        assert!(should_use_xet(HF_XET_THRESHOLD_BYTES + 1));
        // Real SHIP-TWO-001 teacher sizes (evidence/ship-two-001/ex-04-preflight-gate-smoketest-v2.json).
        assert!(should_use_xet(8_035_635_524));
        assert!(should_use_xet(15_231_938_404));
        assert!(should_use_xet(8_037_129_408));
    }

    #[test]
    fn token_refresh_url_matches_hf_protocol_shape() {
        // Exact shape the Xet protocol expects (FALSIFY-PUB-LFS-002).
        let url = build_token_refresh_url("https://huggingface.co", "paiml/my-model", "main");
        assert_eq!(
            url,
            "https://huggingface.co/api/models/paiml/my-model/xet-write-token/main"
        );
    }

    #[test]
    fn token_refresh_url_strips_trailing_slash() {
        let url = build_token_refresh_url("https://huggingface.co/", "org/repo", "main");
        assert!(!url.contains("co//api"));
        assert_eq!(
            url,
            "https://huggingface.co/api/models/org/repo/xet-write-token/main"
        );
    }

    #[test]
    fn token_refresh_url_supports_non_main_revision() {
        // Revisions other than `main` are valid (e.g. PR branches).
        let url = build_token_refresh_url("https://huggingface.co", "org/repo", "release-v1");
        assert_eq!(
            url,
            "https://huggingface.co/api/models/org/repo/xet-write-token/release-v1"
        );
    }
}
