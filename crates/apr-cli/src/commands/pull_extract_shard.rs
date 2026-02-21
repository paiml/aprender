
/// Extract a shard filename from a "key": "value" pair.
fn extract_shard_filename(kv_pair: &str) -> Option<String> {
    let colon_pos = kv_pair.rfind(':')?;
    let value = kv_pair[colon_pos + 1..].trim();
    let filename = value.trim_matches(|c: char| c == '"' || c.is_whitespace());
    if filename.ends_with(".safetensors") && !filename.is_empty() {
        Some(filename.to_string())
    } else {
        None
    }
}

fn extract_shard_files_from_index(json: &str) -> Vec<String> {
    let Some(weight_map_start) = json.find("\"weight_map\"") else {
        return Vec::new();
    };
    let Some(entries) = find_brace_content(&json[weight_map_start..]) else {
        return Vec::new();
    };
    let mut sorted: Vec<String> = entries
        .split(',')
        .filter_map(extract_shard_filename)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    sorted.sort();
    sorted
}

fn home_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(std::path::PathBuf::from)
}

/// GH-229: Resolve HuggingFace auth token for gated models.
///
/// Priority: HF_TOKEN env var → ~/.huggingface/token file → ~/.cache/huggingface/token
fn resolve_hf_token() -> Option<String> {
    // Priority 1: Environment variable
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.is_empty() {
            return Some(token);
        }
    }
    // Priority 2: HuggingFace CLI token file
    if let Some(home) = home_dir() {
        for path in [
            home.join(".huggingface/token"),
            home.join(".cache/huggingface/token"),
        ] {
            if let Ok(token) = std::fs::read_to_string(&path) {
                let token = token.trim().to_string();
                if !token.is_empty() {
                    return Some(token);
                }
            }
        }
    }
    None
}

/// Build an authenticated ureq request if HF token is available.
fn hf_get(url: &str) -> ureq::Request {
    let req = ureq::get(url);
    if let Some(token) = resolve_hf_token() {
        req.set("Authorization", &format!("Bearer {token}"))
    } else {
        req
    }
}

fn download_file(url: &str, path: &Path) -> Result<()> {
    let response = hf_get(url)
        .call()
        .map_err(|e| CliError::NetworkError(format!("Download failed: {e}")))?;

    let mut file = std::fs::File::create(path)?;
    let mut reader = response.into_reader();
    std::io::copy(&mut reader, &mut file)?;

    Ok(())
}

/// GH-213: Download a file with progress, computing BLAKE3 hash incrementally.
///
/// Returns a `FileChecksum` with the downloaded size and BLAKE3 hash.
/// Verifies that downloaded bytes match Content-Length when available.
fn download_file_with_progress(url: &str, path: &Path) -> Result<FileChecksum> {
    let response = hf_get(url)
        .call()
        .map_err(|e| CliError::NetworkError(format!("Download failed: {e}")))?;

    let total = response
        .header("Content-Length")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);

    let mut file = std::fs::File::create(path)?;
    let mut reader = response.into_reader();
    let mut hasher = blake3::Hasher::new();
    let mut downloaded: u64 = 0;
    let mut buf = vec![0u8; 64 * 1024];
    let mut last_pct: u64 = 0;

    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| CliError::NetworkError(format!("Read failed: {e}")))?;
        if n == 0 {
            break;
        }
        let chunk = &buf[..n];
        io::Write::write_all(&mut file, chunk)?;
        hasher.update(chunk);
        downloaded += n as u64;

        if total > 0 {
            let pct = downloaded * 100 / total;
            if pct / 10 > last_pct / 10 {
                print!(" {}%", pct);
                io::stdout().flush().ok();
                last_pct = pct;
            }
        }
    }

    // GH-213: Verify Content-Length match (catches incomplete transfers)
    if total > 0 && downloaded != total {
        // Remove the partial file
        let _ = std::fs::remove_file(path);
        return Err(CliError::NetworkError(format!(
            "Download incomplete for '{}': expected {} bytes, got {} bytes",
            path.display(),
            total,
            downloaded
        )));
    }

    Ok(FileChecksum {
        size: downloaded,
        blake3: hasher.finalize().to_hex().to_string(),
    })
}

/// Backward-compatible wrapper: resolve URI to string (for existing callers that expect String)
#[allow(dead_code)]
pub fn resolve_hf_uri(uri: &str) -> Result<String> {
    match resolve_hf_model(uri)? {
        ResolvedModel::SingleFile(s) => Ok(s),
        ResolvedModel::Sharded { org, repo, .. } => {
            // Return the index.json URI for backward compatibility
            Ok(format!(
                "hf://{}/{}/model.safetensors.index.json",
                org, repo
            ))
        }
    }
}
