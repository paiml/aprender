impl HfHubClient {

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
    #[allow(clippy::disallowed_methods)] // serde_json::json! macro internally uses unwrap()
    fn upload_via_lfs(
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

        // Calculate SHA256 hash for LFS
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

        // For files > 5GB, we need to use HuggingFace's multipart upload API
        // The preupload endpoint returns presigned URLs for S3 multipart upload
        const CHUNK_SIZE: usize = 5 * 1024 * 1024 * 1024; // 5GB chunks (S3 limit)
        let num_chunks = (file_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
        eprintln!(
            "[LFS] File size: {} bytes, will upload in {} chunk(s)",
            file_size, num_chunks
        );

        // Step 1: Request presigned upload URLs via preupload endpoint
        let preupload_url = format!("{}/api/models/{}/preupload/main", self.api_base, repo_id);
        eprintln!(
            "[LFS] Step 1: Requesting upload URLs from {}",
            preupload_url
        );

        // Include SHA256 in the request to indicate we want multipart upload
        let preupload_body = serde_json::json!({
            "files": [{
                "path": filename,
                "size": file_size,
                "sample": base64_encode(&data[..data.len().min(512)])
            }]
        });
        eprintln!(
            "[LFS] Preupload request (size={}, sha256={}...)",
            file_size,
            sha256.get(..16).unwrap_or(&sha256)
        );

        let preupload_resp = match ureq::post(&preupload_url)
            .set("Authorization", &format!("Bearer {token}"))
            .set("Content-Type", "application/json")
            .send_json(&preupload_body)
        {
            Ok(resp) => resp,
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp
                    .into_string()
                    .unwrap_or_else(|_| "unable to read body".to_string());
                eprintln!(
                    "[LFS] ERROR: Preupload failed with status {}: {}",
                    code, body
                );
                return Err(HfHubError::NetworkError(format!(
                    "Preupload failed (HTTP {}): {}",
                    code, body
                )));
            }
            Err(e) => {
                eprintln!("[LFS] ERROR: Preupload request failed: {}", e);
                return Err(HfHubError::NetworkError(format!("Preupload failed: {e}")));
            }
        };

        eprintln!(
            "[LFS] Preupload response status: {}",
            preupload_resp.status()
        );

        let preupload_data: serde_json::Value = preupload_resp.into_json().map_err(|e| {
            eprintln!("[LFS] ERROR: Failed to parse preupload response: {}", e);
            HfHubError::NetworkError(format!("Preupload parse failed: {e}"))
        })?;

        eprintln!(
            "[LFS] Preupload response: {}",
            serde_json::to_string_pretty(&preupload_data).unwrap_or_default()
        );

        // Extract upload info from response
        let files = preupload_data["files"].as_array().ok_or_else(|| {
            eprintln!("[LFS] ERROR: Invalid preupload response - no 'files' array");
            HfHubError::NetworkError("Invalid preupload response".to_string())
        })?;

        if files.is_empty() {
            eprintln!("[LFS] ERROR: Empty files array in preupload response");
            return Err(HfHubError::NetworkError(
                "No file info returned".to_string(),
            ));
        }

        let file_info = &files[0];
        let upload_mode = file_info
            .get("uploadMode")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        eprintln!("[LFS] Upload mode: {}", upload_mode);

        // Check for various URL fields that HF might return
        let upload_url = file_info
            .get("uploadUrl")
            .or_else(|| file_info.get("upload_url"))
            .and_then(|v| v.as_str());

        // Also check for chunked/multipart URLs
        let chunk_urls = file_info
            .get("chunkUrls")
            .or_else(|| file_info.get("chunk_urls"))
            .or_else(|| file_info.get("urls"))
            .and_then(|v| v.as_array());

        eprintln!("[LFS] Upload URL present: {}", upload_url.is_some());
        eprintln!("[LFS] Chunk URLs present: {}", chunk_urls.is_some());

        // For files > 5GB without upload URL, HuggingFace requires multipart upload
        // which is only available via their git LFS transfer agent (huggingface-cli)
        // See: https://huggingface.co/docs/hub/en/storage-limits
        const FIVE_GB: usize = 5 * 1024 * 1024 * 1024;
        if upload_url.is_none()
            && chunk_urls.is_none()
            && upload_mode == "lfs"
            && file_size > FIVE_GB
        {
            eprintln!("[LFS] ERROR: File {} ({:.1} GB) exceeds 5GB HuggingFace Hub limit for HTTP API uploads",
                filename, file_size as f64 / 1_000_000_000.0);
            eprintln!("[LFS] ");
            eprintln!("[LFS] Files > 5GB require HuggingFace's multipart transfer agent.");
            eprintln!("[LFS] Options:");
            eprintln!("[LFS]   1. Split model into shards < 5GB each (recommended)");
            eprintln!("[LFS]      Use: apr export --max-shard-size 4GB");
            eprintln!("[LFS]   2. Use huggingface-cli with lfs-enable-largefiles:");
            eprintln!("[LFS]      git clone https://huggingface.co/{}", repo_id);
            eprintln!("[LFS]      huggingface-cli lfs-enable-largefiles ./");
            eprintln!("[LFS]      cp {} ./", filename);
            eprintln!("[LFS]      git add . && git commit -m 'Add model' && git push");
            eprintln!("[LFS] ");
            return Err(HfHubError::NetworkError(format!(
                "File {} ({:.1} GB) exceeds 5GB limit. \
                 HuggingFace Hub requires multipart LFS for files > 5GB. \
                 Split into smaller shards or use huggingface-cli.",
                filename,
                file_size as f64 / 1_000_000_000.0
            )));
        }

        if let Some(urls) = chunk_urls {
            // Multipart upload with presigned URLs
            eprintln!(
                "[LFS] Step 2: Multipart upload with {} presigned URLs",
                urls.len()
            );

            for (i, url_value) in urls.iter().enumerate() {
                let chunk_url = url_value.as_str().ok_or_else(|| {
                    HfHubError::NetworkError(format!("Invalid chunk URL at index {}", i))
                })?;

                let chunk_start = i * CHUNK_SIZE;
                let chunk_end = ((i + 1) * CHUNK_SIZE).min(file_size);
                let chunk_data = &data[chunk_start..chunk_end];

                eprintln!(
                    "[LFS] Uploading chunk {}/{}: bytes {}-{} ({:.1} MB)",
                    i + 1,
                    urls.len(),
                    chunk_start,
                    chunk_end,
                    chunk_data.len() as f64 / 1_000_000.0
                );

                let chunk_start_time = Instant::now();
                let resp = ureq::put(chunk_url)
                    .set("Content-Type", "application/octet-stream")
                    .timeout(std::time::Duration::from_secs(7200))
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
                    chunk_start_time.elapsed().as_secs_f64()
                );

                if !(200..300).contains(&status) {
                    return Err(HfHubError::NetworkError(format!(
                        "Chunk upload failed with status {}",
                        status
                    )));
                }
            }

            // Call completion endpoint if provided
            if let Some(completion_url) = file_info.get("completionUrl").and_then(|v| v.as_str()) {
                eprintln!("[LFS] Calling completion URL: {}", completion_url);
                let _ = ureq::post(completion_url)
                    .set("Authorization", &format!("Bearer {token}"))
                    .set("Content-Type", "application/json")
                    .send_json(serde_json::json!({}));
            }
        } else if let Some(url) = upload_url {
            // Single URL upload (for smaller LFS files)
            eprintln!(
                "[LFS] Step 2: Single URL upload to {}",
                &url[..url.len().min(100)]
            );

            let upload_start = Instant::now();
            let headers = file_info.get("uploadHeader").and_then(|v| v.as_object());

            let mut request = ureq::put(url)
                .set("Content-Type", "application/octet-stream")
                .timeout(std::time::Duration::from_secs(7200));

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
                (file_size as f64 / 1_000_000.0) / upload_start.elapsed().as_secs_f64()
            );

            if !(200..300).contains(&status) {
                let body = resp.into_string().unwrap_or_default();
                return Err(HfHubError::NetworkError(format!(
                    "Upload failed (HTTP {}): {}",
                    status, body
                )));
            }
        } else {
            // No upload URL - the file may already exist or we need to commit the LFS pointer
            eprintln!("[LFS] No upload URL returned - proceeding to commit LFS pointer");
            eprintln!("[LFS] (This may mean the file content already exists on HF's LFS storage)");
        }

        // Step 3: Commit with LFS pointer
        eprintln!("[LFS] Step 3: Committing LFS pointer");
        let lfs_pointer = format!(
            "version https://git-lfs.github.com/spec/v1\noid sha256:{}\nsize {}\n",
            sha256, file_size
        );
        eprintln!("[LFS] Pointer content:\n{}", lfs_pointer);

        let commit_url = format!("{}/api/models/{}/commit/main", self.api_base, repo_id);
        eprintln!("[LFS] Commit URL: {}", commit_url);

        let commit_body = serde_json::json!({
            "summary": commit_msg,
            "operations": [{
                "op": "addOrUpdate",
                "path": filename,
                "content": base64_encode(lfs_pointer.as_bytes()),
                "encoding": "base64",
                "lfs": {
                    "sha256": sha256,
                    "size": file_size
                }
            }]
        });

        let commit_resp = ureq::post(&commit_url)
            .set("Authorization", &format!("Bearer {token}"))
            .set("Content-Type", "application/json")
            .send_json(&commit_body);

        match commit_resp {
            Ok(resp) if resp.status() >= 200 && resp.status() < 300 => {
                let body = resp.into_string().unwrap_or_default();
                eprintln!("[LFS] Commit successful: {}", &body[..body.len().min(200)]);
                eprintln!(
                    "[LFS] Total upload time: {:.1}s",
                    start.elapsed().as_secs_f64()
                );
                Ok(())
            }
            Ok(resp) => {
                let status = resp.status();
                let body = resp.into_string().unwrap_or_default();
                eprintln!(
                    "[LFS] ERROR: Commit failed with status {}: {}",
                    status,
                    &body[..body.len().min(500)]
                );
                Err(HfHubError::NetworkError(format!(
                    "Commit failed (HTTP {}): {}",
                    status, body
                )))
            }
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                eprintln!(
                    "[LFS] ERROR: Commit failed with status {}: {}",
                    code,
                    &body[..body.len().min(500)]
                );
                Err(HfHubError::NetworkError(format!(
                    "Commit failed (HTTP {code}): {body}"
                )))
            }
            Err(e) => {
                eprintln!("[LFS] ERROR: Network error during commit: {}", e);
                Err(HfHubError::NetworkError(format!("Network error: {e}")))
            }
        }
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
