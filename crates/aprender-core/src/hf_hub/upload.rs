use super::super::repo_api::{CommitOp, LfsBody, PreuploadFile, RepoApi};
use super::super::{HfHubClient, HfHubError, ModelCard, Result};

/// 5GB chunk size for S3 multipart upload.
const LFS_CHUNK_SIZE: usize = 5 * 1024 * 1024 * 1024;

impl HfHubClient {
    /// Send preupload request to HuggingFace API and return parsed file info.
    ///
    /// The request goes through [`RepoApi::preupload`], the one HF upload path
    /// (#4961).
    #[cfg(feature = "hf-hub-integration")]
    pub(crate) fn send_preupload_request(
        &self,
        repo_id: &str,
        filename: &str,
        data: &[u8],
        sha256: &str,
        token: &str,
    ) -> Result<serde_json::Value> {
        eprintln!(
            "[LFS] Step 1: Requesting upload URLs for {filename} (size={}, sha256={}...)",
            data.len(),
            sha256.get(..16).unwrap_or(sha256)
        );
        let file = PreuploadFile {
            path: filename,
            size: data.len() as u64,
            sample: data,
        };
        let files = RepoApi::new(&self.api_base, repo_id, token)
            .preupload("main", &[file])
            .inspect_err(|e| eprintln!("[LFS] ERROR: Preupload failed: {e}"))?;
        eprintln!(
            "[LFS] Preupload response: {}",
            serde_json::to_string_pretty(&files).unwrap_or_default()
        );
        files.into_iter().next().ok_or_else(|| {
            eprintln!("[LFS] ERROR: Empty files array in preupload response");
            HfHubError::NetworkError("No file info returned".to_string())
        })
    }

    /// Upload data via chunked/multipart presigned URLs.
    #[cfg(feature = "hf-hub-integration")]
    fn upload_chunks(
        data: &[u8],
        urls: &[serde_json::Value],
        file_info: &serde_json::Value,
        token: &str,
    ) -> Result<()> {
        use std::time::Instant;

        eprintln!(
            "[LFS] Step 2: Multipart upload with {} presigned URLs",
            urls.len()
        );
        let file_size = data.len();

        for (i, url_value) in urls.iter().enumerate() {
            let chunk_url = url_value.as_str().ok_or_else(|| {
                HfHubError::NetworkError(format!("Invalid chunk URL at index {}", i))
            })?;
            let chunk_start = i * LFS_CHUNK_SIZE;
            let chunk_end = ((i + 1) * LFS_CHUNK_SIZE).min(file_size);
            let chunk_data = &data[chunk_start..chunk_end];

            eprintln!(
                "[LFS] Uploading chunk {}/{}: bytes {}-{} ({:.1} MB)",
                i + 1,
                urls.len(),
                chunk_start,
                chunk_end,
                chunk_data.len() as f64 / 1_000_000.0
            );

            let t = Instant::now();
            let resp = ureq::put(chunk_url)
                .set("Content-Type", "application/octet-stream")
                .timeout(std::time::Duration::from_hours(2))
                .send_bytes(chunk_data)
                .map_err(|e| {
                    eprintln!("[LFS] ERROR: Chunk {} upload failed: {}", i + 1, e);
                    HfHubError::NetworkError(format!("Chunk upload failed: {e}"))
                })?;

            let status = resp.status();
            eprintln!(
                "[LFS] Chunk {}/{} uploaded: status={}, elapsed={:.1}s",
                i + 1,
                urls.len(),
                status,
                t.elapsed().as_secs_f64()
            );
            if !(200..300).contains(&status) {
                return Err(HfHubError::NetworkError(format!(
                    "Chunk upload failed with status {}",
                    status
                )));
            }
        }

        if let Some(completion_url) = file_info.get("completionUrl").and_then(|v| v.as_str()) {
            eprintln!("[LFS] Calling completion URL: {}", completion_url);
            let _ = ureq::post(completion_url)
                .set("Authorization", &format!("Bearer {token}"))
                .set("Content-Type", "application/json")
                .send_json(serde_json::json!({}));
        }
        Ok(())
    }

    /// Upload data to a single presigned URL.
    #[cfg(feature = "hf-hub-integration")]
    fn upload_single(data: &[u8], url: &str, file_info: &serde_json::Value) -> Result<()> {
        use std::time::Instant;

        eprintln!(
            "[LFS] Step 2: Single URL upload to {}",
            &url[..url.len().min(100)]
        );
        let upload_start = Instant::now();
        let headers = file_info.get("uploadHeader").and_then(|v| v.as_object());

        let mut request = ureq::put(url)
            .set("Content-Type", "application/octet-stream")
            .timeout(std::time::Duration::from_hours(2));

        if let Some(hdrs) = headers {
            for (key, value) in hdrs {
                if let Some(v) = value.as_str() {
                    eprintln!("[LFS] Adding header: {}: {}...", key, &v[..v.len().min(20)]);
                    request = request.set(key, v);
                }
            }
        }

        let resp = request.send_bytes(data).map_err(|e| {
            eprintln!("[LFS] ERROR: Upload failed: {}", e);
            HfHubError::NetworkError(format!("Upload failed: {e}"))
        })?;

        let status = resp.status();
        eprintln!(
            "[LFS] Upload complete: status={}, elapsed={:.1}s, speed={:.1} MB/s",
            status,
            upload_start.elapsed().as_secs_f64(),
            (data.len() as f64 / 1_000_000.0) / upload_start.elapsed().as_secs_f64()
        );

        if !(200..300).contains(&status) {
            let body = resp.into_string().unwrap_or_default();
            return Err(HfHubError::NetworkError(format!(
                "Upload failed (HTTP {}): {}",
                status, body
            )));
        }
        Ok(())
    }

    /// Upload data via the standard LFS batch API.
    ///
    /// PMAT-690 P3-C-prep defect 5 (2026-05-17): HuggingFace's `preupload`
    /// endpoint returns `uploadMode: "lfs"` with no inline URLs for files in
    /// the 5MB-5GB band, expecting the client to obtain the presigned S3 URL
    /// via the LFS Batch API (RFC: <https://github.com/git-lfs/git-lfs/blob/main/docs/api/batch.md>).
    /// The endpoint for HuggingFace models is
    /// `https://huggingface.co/{repo}.git/info/lfs/objects/batch` (no
    /// `/datasets/` prefix — that's the dataset path used in
    /// `aprender-data`).
    ///
    /// [`RepoApi::upload_lfs`] (#4961) runs the batch request, skips the PUT
    /// when the Hub already holds the object, and otherwise PUTs the bytes to
    /// the presigned URL (no auth header — the URL itself is the credential).
    ///
    /// Caller (`upload_via_lfs`) handles step 4 (commit LFS pointer).
    #[cfg(feature = "hf-hub-integration")]
    fn upload_via_lfs_batch(
        &self,
        repo_id: &str,
        filename: &str,
        data: &[u8],
        sha256: &str,
        token: &str,
    ) -> Result<()> {
        eprintln!(
            "[LFS-BATCH] Step 2: LFS batch upload of {filename} ({:.1} MB)",
            data.len() as f64 / 1_000_000.0
        );
        let put_start = std::time::Instant::now();
        let sent = RepoApi::new(&self.api_base, repo_id, token)
            .upload_lfs(sha256, data.len() as u64, LfsBody::Bytes(data))
            .inspect_err(|e| eprintln!("[LFS-BATCH] ERROR: {e}"))?;
        if sent {
            let secs = put_start.elapsed().as_secs_f64();
            eprintln!(
                "[LFS-BATCH] PUT complete: elapsed={secs:.1}s, speed={:.1} MB/s",
                (data.len() as f64 / 1_000_000.0) / secs
            );
        } else {
            eprintln!(
                "[LFS-BATCH] Object already exists on HF storage — skipping PUT, \
                 proceeding to pointer commit"
            );
        }
        Ok(())
    }

    /// Commit an LFS pointer to the HuggingFace Hub.
    ///
    /// PMAT-690 P3-C-prep defect 5 — memory rule
    /// `feedback_hf_commit_ndjson_load_bearing.md` (2026-04-18): the commit
    /// must be an NDJSON `lfsFile` line, which [`RepoApi::commit`] (#4961)
    /// writes; a JSON `addOrUpdate` body returns 200 and drops the file.
    #[cfg(feature = "hf-hub-integration")]
    fn commit_lfs_pointer(
        &self,
        repo_id: &str,
        filename: &str,
        sha256: &str,
        file_size: usize,
        commit_msg: &str,
        token: &str,
    ) -> Result<()> {
        eprintln!("[LFS] Step 3: Committing LFS pointer {filename} -> sha256:{sha256} ({file_size} bytes)");
        let op = CommitOp::Lfs {
            path: filename.to_string(),
            sha256: sha256.to_string(),
            size: file_size as u64,
        };
        let oid = RepoApi::new(&self.api_base, repo_id, token)
            .commit("main", commit_msg, &[op])
            .inspect_err(|e| eprintln!("[LFS] ERROR: Commit failed: {e}"))?;
        eprintln!("[LFS] Commit successful: {oid}");
        Ok(())
    }

    /// PMAT-690 P3-C-prep defect 6 (2026-05-18): public LFS-alias commit.
    ///
    /// Emits an NDJSON commit that adds a new filename pointing at an
    /// already-uploaded LFS object identified by `sha256` + `file_size`.
    /// HF deduplicates LFS blobs by OID — the same bytes back both paths
    /// at zero storage cost. Used by `apr publish` to auto-emit a
    /// `model.safetensors` alias next to a descriptive-named SafeTensors
    /// export so HF Transformers `AutoModelForCausalLM.from_pretrained`
    /// can auto-discover the weights.
    ///
    /// # Errors
    ///
    /// Returns `HfHubError::MissingToken` when `HF_TOKEN` is not set.
    /// Propagates the underlying NDJSON commit error otherwise.
    #[cfg(feature = "hf-hub-integration")]
    pub fn commit_lfs_alias(
        &self,
        repo_id: &str,
        alias_filename: &str,
        sha256: &str,
        file_size: usize,
        commit_msg: &str,
    ) -> Result<()> {
        let token = self.token.as_ref().ok_or(HfHubError::MissingToken)?;
        self.commit_lfs_pointer(
            repo_id,
            alias_filename,
            sha256,
            file_size,
            commit_msg,
            token,
        )
    }

    /// Stub when feature is disabled.
    #[cfg(not(feature = "hf-hub-integration"))]
    pub fn commit_lfs_alias(
        &self,
        _repo_id: &str,
        _alias_filename: &str,
        _sha256: &str,
        _file_size: usize,
        _commit_msg: &str,
    ) -> Result<()> {
        Err(HfHubError::NetworkError(
            "commit_lfs_alias requires the hf-hub-integration feature".to_string(),
        ))
    }

    /// Abort with a precise error when the Xet transfer path is needed but
    /// not compiled in.
    ///
    /// Applies only to builds WITHOUT `--features xet`. Under the default
    /// `hf-hub-integration` feature, files > 5 GiB cannot be uploaded through
    /// HF Hub's HTTP preupload API (HF returns `uploadMode:lfs` with empty
    /// URLs for that size class). The dogfood path is to rebuild with
    /// `--features xet`, which wires in `hf-xet` for the Xet CAS protocol.
    #[cfg(all(feature = "hf-hub-integration", not(feature = "xet")))]
    fn reject_needs_xet_feature(filename: &str, file_size: usize) -> Result<()> {
        let gib = file_size as f64 / (1024.0 * 1024.0 * 1024.0);
        eprintln!(
            "[LFS] ERROR: File {filename} ({gib:.2} GiB) exceeds HF Hub's 5 GiB HTTP threshold"
        );
        eprintln!("[LFS] HF Hub returned uploadMode=lfs with no presigned URLs, which means");
        eprintln!("[LFS] the file must transfer via the Xet content-addressable protocol.");
        eprintln!("[LFS] Rebuild apr with Xet support:");
        eprintln!("[LFS]   cargo build --release --features cuda,apr-cli/xet");
        eprintln!("[LFS] (See contracts/apr-publish-hf-large-file-v1.yaml and");
        eprintln!("[LFS]  docs/specifications/aprender-train/ship-two-models-spec.md §12.8.)");
        Err(HfHubError::NetworkError(format!(
            "File {filename} ({gib:.2} GiB) exceeds HF Hub's 5 GiB HTTP threshold; \
             rebuild with `--features xet` to enable the Xet upload path."
        )))
    }

    /// Upload a large file via the Xet CAS protocol (F-PUB-LFS-001).
    ///
    /// Writes `data` to a tempfile and hands it to `XetUploader`, which
    /// delegates chunking, dedup, xorb/shard upload, and the LFS pointer
    /// commit to the `hf-xet` crate (HF's reference implementation).
    ///
    /// The tempfile round-trip is a short-term accommodation: callers today
    /// pass `model_data: &[u8]` already resident in memory (see
    /// `client_impl.rs::push_to_hub`). A future refactor will thread a
    /// `&Path` through the whole upload stack to skip this copy.
    #[cfg(feature = "xet")]
    fn upload_via_xet(
        &self,
        repo_id: &str,
        filename: &str,
        data: &[u8],
        commit_msg: &str,
        token: &str,
    ) -> Result<()> {
        use std::io::Write;

        eprintln!(
            "[XET] Dispatching {} ({:.2} GiB) via hf-xet (>5 GiB path)",
            filename,
            data.len() as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        // Materialize bytes to a tempfile so hf-xet can stream the contents.
        // (hf-xet's `upload_from_path_blocking` reads from disk.)
        let mut tmp = tempfile::NamedTempFile::new()
            .map_err(|e| HfHubError::XetUpload(format!("tempfile create failed: {e}")))?;
        tmp.write_all(data)
            .map_err(|e| HfHubError::XetUpload(format!("tempfile write failed: {e}")))?;
        tmp.flush()
            .map_err(|e| HfHubError::XetUpload(format!("tempfile flush failed: {e}")))?;

        let uploader = super::super::xet::XetUploader {
            api_base: &self.api_base,
            repo_id,
            revision: "main",
            token,
        };
        uploader.upload_file(tmp.path(), commit_msg)?;

        eprintln!("[XET] Xet upload + LFS pointer commit succeeded for {filename}");
        Ok(())
    }

    /// Upload large file via HuggingFace Hub multipart upload (APR-PUB-001)
    ///
    /// HuggingFace multipart upload flow for files > 5GB:
    /// 1. POST to /api/models/{repo}/preupload/main with SHA256 to get presigned URLs
    /// 2. Upload file parts to presigned URLs (5GB chunks)
    /// 3. POST completion to finalize upload
    /// 4. POST commit with LFS pointer
    ///
    /// **OBS-003/OBS-004**: Full verbose logging for diagnostics
    #[cfg(feature = "hf-hub-integration")]
    #[allow(clippy::disallowed_methods)]
    pub(crate) fn upload_via_lfs(
        &self,
        repo_id: &str,
        filename: &str,
        data: &[u8],
        commit_msg: &str,
        token: &str,
    ) -> Result<()> {
        use sha2::{Digest, Sha256};
        use std::time::Instant;

        let start = Instant::now();
        let file_size = data.len();

        eprintln!(
            "[LFS] Calculating SHA256 for {} ({:.1} MB)...",
            filename,
            file_size as f64 / 1_000_000.0
        );
        let mut hasher = Sha256::new();
        hasher.update(data);
        let sha256 = format!("{:x}", hasher.finalize());
        eprintln!("[LFS] SHA256: {}", sha256);
        eprintln!("[LFS] Using token: {}...", &token[..12.min(token.len())]);

        let num_chunks = (file_size + LFS_CHUNK_SIZE - 1) / LFS_CHUNK_SIZE;
        eprintln!(
            "[LFS] File size: {} bytes, will upload in {} chunk(s)",
            file_size, num_chunks
        );

        let file_info = self.send_preupload_request(repo_id, filename, data, &sha256, token)?;

        let upload_mode = file_info
            .get("uploadMode")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        eprintln!("[LFS] Upload mode: {}", upload_mode);

        let upload_url = file_info
            .get("uploadUrl")
            .or_else(|| file_info.get("upload_url"))
            .and_then(|v| v.as_str());
        let chunk_urls = file_info
            .get("chunkUrls")
            .or_else(|| file_info.get("chunk_urls"))
            .or_else(|| file_info.get("urls"))
            .and_then(|v| v.as_array());

        eprintln!("[LFS] Upload URL present: {}", upload_url.is_some());
        eprintln!("[LFS] Chunk URLs present: {}", chunk_urls.is_some());

        // FALSIFY-PUB-LFS-001: dispatch files > 5 GiB to the Xet path
        // when HF Hub has responded with `uploadMode:lfs` and no URLs.
        if upload_url.is_none()
            && chunk_urls.is_none()
            && upload_mode == "lfs"
            && super::super::xet::should_use_xet(file_size as u64)
        {
            #[cfg(feature = "xet")]
            {
                return self.upload_via_xet(repo_id, filename, data, commit_msg, token);
            }
            #[cfg(not(feature = "xet"))]
            {
                return Self::reject_needs_xet_feature(filename, file_size);
            }
        }

        if let Some(urls) = chunk_urls {
            Self::upload_chunks(data, urls, &file_info, token)?;
        } else if let Some(url) = upload_url {
            Self::upload_single(data, url, &file_info)?;
        } else {
            // PMAT-690 P3-C-prep defect 5 (2026-05-17): when preupload returns
            // `uploadMode:lfs` with no inline upload URL, files in the 5MB-5GB
            // band must use the standard LFS batch API to obtain a presigned
            // S3 URL. Previously we skipped this and went straight to commit,
            // landing orphaned LFS pointers (paiml/albor-370m-v1 first
            // observed). The Xet branch above handles >5 GiB; this branch
            // handles the gap below it.
            eprintln!(
                "[LFS] No inline upload URL — falling back to LFS batch API \
                 (PMAT-690 defect 5)"
            );
            self.upload_via_lfs_batch(repo_id, filename, data, &sha256, token)?;
        }

        self.commit_lfs_pointer(repo_id, filename, &sha256, file_size, commit_msg, token)?;
        eprintln!(
            "[LFS] Total upload time: {:.1}s",
            start.elapsed().as_secs_f64()
        );
        Ok(())
    }

    /// Generate a model card from model metadata
    ///
    /// Creates a ModelCard with auto-populated fields from training info.
    #[must_use]
    pub fn auto_generate_card(repo_id: &str, model_type: &str, version: &str) -> ModelCard {
        ModelCard::new(repo_id, version)
            .with_name(repo_id.split('/').next_back().unwrap_or(repo_id))
            .with_architecture(model_type)
            .with_description(format!("{model_type} model trained with aprender"))
    }
}
