
/// Remove a model from cache
pub fn remove(model_ref: &str) -> Result<()> {
    println!("{}", "=== APR Remove ===".cyan().bold());
    println!();
    println!("Model: {}", model_ref.cyan());

    let mut fetcher = ModelFetcher::new().map_err(|e| {
        CliError::ValidationFailed(format!("Failed to initialize model fetcher: {e}"))
    })?;

    let removed = fetcher
        .remove(model_ref)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to remove model: {e}")))?;

    if removed {
        println!("{} Model removed from cache", "✓".green());
    } else {
        println!("{} Model not found in cache", "⚠".yellow());
    }

    Ok(())
}

/// Resolve a model reference to a local path (for run/serve commands)
/// Downloads if not cached and auto_pull is enabled
#[allow(dead_code)]
pub fn resolve_model_path(model_ref: &str) -> Result<std::path::PathBuf> {
    // If it's already a local file path, use it directly
    let path = std::path::Path::new(model_ref);
    if path.exists() && path.is_file() {
        return Ok(path.to_path_buf());
    }

    // Try to resolve via pacha
    let mut fetcher = ModelFetcher::with_config(FetchConfig::default()).map_err(|e| {
        CliError::ValidationFailed(format!("Failed to initialize model fetcher: {e}"))
    })?;

    // Pull (uses cache if available)
    let result = fetcher
        .pull(model_ref, |progress| {
            if progress.total_bytes > 0 {
                let pct = progress.percent();
                eprint!(
                    "\rPulling model... [{:30}] {:5.1}%",
                    "=".repeat((pct / 3.33) as usize),
                    pct
                );
                io::stderr().flush().ok();
            }
        })
        .map_err(|e| {
            // Not a pacha model ref, check if file exists
            CliError::ValidationFailed(format!(
                "Model '{}' not found. Not a local file and could not resolve via registry: {}",
                model_ref, e
            ))
        })?;

    if !result.cache_hit {
        eprintln!(); // Newline after progress
    }

    Ok(result.path)
}

/// Format bytes to human-readable string
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// GH-198 + GAP-UX-002: Download companion files (tokenizer.json, config.json) for SafeTensors models.
///
/// SafeTensors format stores weights only — unlike GGUF which embeds tokenizer and config.
/// The realizar inference engine expects these as sibling files.
///
/// GAP-UX-002: Store companions with model hash prefix to prevent cross-model conflicts.
/// Example: `d71534cb.safetensors` → `d71534cb.config.json`, `d71534cb.tokenizer.json`
fn fetch_safetensors_companions(model_path: &Path, resolved_uri: &str) -> Result<()> {
    // Extract HF repo from resolved URI: "hf://org/repo/file.safetensors" → "org/repo"
    let Some(repo_id) = extract_hf_repo(resolved_uri) else {
        // Not an HF URI — can't fetch companions (local file or unknown source)
        return Ok(());
    };

    // GAP-UX-002: Extract model stem (hash) for prefixing companion files
    // Model: d71534cb948e32eb.safetensors → stem: d71534cb948e32eb
    let model_stem = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");

    let companions = ["tokenizer.json", "config.json"];
    let cache_dir = model_path
        .parent()
        .ok_or_else(|| CliError::ValidationFailed("Model path has no parent directory".into()))?;

    for filename in &companions {
        // GAP-UX-002: Use hash-prefixed filename (e.g., "d71534cb.config.json")
        let prefixed_filename = format!("{}.{}", model_stem, filename);
        let sibling_path = cache_dir.join(&prefixed_filename);

        if sibling_path.exists() {
            println!(
                "  {} {} (already exists)",
                "✓".green(),
                prefixed_filename.dimmed()
            );
            continue;
        }

        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo_id, filename
        );

        match ureq::get(&url).call() {
            Ok(response) => {
                let mut body = Vec::new();
                response.into_reader().read_to_end(&mut body).map_err(|e| {
                    CliError::NetworkError(format!("Failed to read {filename}: {e}"))
                })?;
                std::fs::write(&sibling_path, &body).map_err(|e| {
                    CliError::ValidationFailed(format!(
                        "Failed to write {}: {e}",
                        sibling_path.display()
                    ))
                })?;
                println!(
                    "  {} {} ({})",
                    "✓".green(),
                    prefixed_filename,
                    format_bytes(body.len() as u64).dimmed()
                );
            }
            Err(ureq::Error::Status(404, _)) => {
                // File doesn't exist in repo — not fatal
                println!(
                    "  {} {} (not found in repo)",
                    "⚠".yellow(),
                    prefixed_filename.dimmed()
                );
            }
            Err(e) => {
                // Network error — warn but don't block the pull
                eprintln!(
                    "  {} Failed to download {}: {}",
                    "⚠".yellow(),
                    prefixed_filename,
                    e
                );
            }
        }
    }

    Ok(())
}

/// GH-211: Convert a SafeTensors model to APR and GGUF formats for full QA qualification.
///
/// After download, produces sibling `.apr` and `.gguf` files so that all MVP
/// test matrix cells (SafeTensors/APR/GGUF × run/chat/serve) can execute.
fn convert_safetensors_formats(safetensors_path: &Path) -> Result<()> {
    println!();
    println!("{}", "Converting formats...".yellow());

    // Derive output paths from the SafeTensors path
    let apr_path = safetensors_path.with_extension("apr");
    let gguf_path = safetensors_path.with_extension("gguf");

    // Step 1: SafeTensors → APR
    if apr_path.exists() {
        println!(
            "  {} {} (already exists)",
            "✓".green(),
            apr_path.file_name().unwrap_or_default().to_string_lossy()
        );
    } else {
        let source = safetensors_path.to_string_lossy().to_string();
        let options = ImportOptions {
            architecture: aprender::format::Architecture::Auto,
            validation: ValidationConfig::Basic,
            quantize: None,
            compress: None,
            strict: false,
            cache: false,
            tokenizer_path: None,
            allow_no_config: true,
        };
        match apr_import(&source, &apr_path, options) {
            Ok(_report) => {
                println!(
                    "  {} {} (SafeTensors → APR)",
                    "✓".green(),
                    apr_path.file_name().unwrap_or_default().to_string_lossy()
                );
            }
            Err(e) => {
                eprintln!(
                    "  {} APR conversion failed: {} (non-fatal)",
                    "⚠".yellow(),
                    e
                );
            }
        }
    }

    // Step 2: APR → GGUF (requires APR to exist)
    if gguf_path.exists() {
        println!(
            "  {} {} (already exists)",
            "✓".green(),
            gguf_path.file_name().unwrap_or_default().to_string_lossy()
        );
    } else if apr_path.exists() {
        let options = ExportOptions {
            format: ExportFormat::Gguf,
            quantize: None,
            ..Default::default()
        };
        match apr_export(&apr_path, &gguf_path, options) {
            Ok(_report) => {
                println!(
                    "  {} {} (APR → GGUF)",
                    "✓".green(),
                    gguf_path.file_name().unwrap_or_default().to_string_lossy()
                );
            }
            Err(e) => {
                eprintln!(
                    "  {} GGUF conversion failed: {} (non-fatal)",
                    "⚠".yellow(),
                    e
                );
            }
        }
    } else {
        eprintln!(
            "  {} GGUF conversion skipped (APR not available)",
            "⚠".yellow()
        );
    }

    Ok(())
}

/// Extract HuggingFace repo ID from a resolved URI.
///
/// Examples:
///   "hf://Qwen/Qwen2.5-Coder-0.5B-Instruct/model.safetensors" → Some("Qwen/Qwen2.5-Coder-0.5B-Instruct")
///   "hf://Qwen/Qwen2.5-Coder-0.5B-Instruct" → Some("Qwen/Qwen2.5-Coder-0.5B-Instruct")
///   "/local/path/model.safetensors" → None
fn extract_hf_repo(uri: &str) -> Option<String> {
    let path = uri.strip_prefix("hf://")?;
    let parts: Vec<&str> = path.split('/').collect();
    if parts.len() >= 2 {
        Some(format!("{}/{}", parts[0], parts[1]))
    } else {
        None
    }
}

/// PMAT-108 + GH-213: Resolve HuggingFace model reference to a downloadable target.
///
/// Returns `SingleFile` for:
/// - Non-HF URIs (local paths, URLs)
/// - URIs with explicit file extension (`.gguf`, `.safetensors`, etc.)
/// - Repos with a single `model.safetensors`
/// - GGUF repos (auto-detects best quantization)
///
/// Returns `Sharded` for:
/// - Repos with `model.safetensors.index.json` (sharded SafeTensors, typically 3B+ models)
///
/// Priority for GGUF auto-detection: Q4_K_M > Q4_K_S > Q4_0 > Q8_0 > any
/// GH-213: Normalize bare "org/repo" to "hf://org/repo".
fn normalize_hf_uri(uri: &str) -> String {
    if !uri.contains("://") && !uri.starts_with('/') && !uri.starts_with('.') {
        let parts: Vec<&str> = uri.split('/').collect();
        if parts.len() >= 2 && !parts[0].is_empty() && !parts[1].is_empty() {
            return format!("hf://{uri}");
        }
    }
    uri.to_string()
}

/// Select best GGUF file by quantization priority (Q4_K_M > Q4_K_S > Q4_0 > Q8_0 > first).
fn select_best_gguf(gguf_files: &[&str], org: &str, repo: &str) -> ResolvedModel {
    let quantization_priority = ["q4_k_m", "q4_k_s", "q4_0", "q8_0"];
    for quant in quantization_priority {
        if let Some(file) = gguf_files.iter().find(|f| f.to_lowercase().contains(quant)) {
            return ResolvedModel::SingleFile(format!("hf://{org}/{repo}/{file}"));
        }
    }
    ResolvedModel::SingleFile(format!("hf://{org}/{repo}/{}", gguf_files[0]))
}

/// Download and parse sharded SafeTensors index, returning shard filenames.
fn resolve_sharded_safetensors(org: &str, repo: &str) -> Result<ResolvedModel> {
    let index_url =
        format!("https://huggingface.co/{org}/{repo}/resolve/main/model.safetensors.index.json");
    let index_response = hf_get(&index_url)
        .call()
        .map_err(|e| CliError::NetworkError(format!("Failed to download model index: {e}")))?;

    let mut index_body = Vec::new();
    index_response
        .into_reader()
        .read_to_end(&mut index_body)
        .map_err(|e| CliError::NetworkError(format!("Failed to read model index: {e}")))?;

    let index_json = String::from_utf8_lossy(&index_body);
    let shard_files = extract_shard_files_from_index(&index_json);

    if shard_files.is_empty() {
        return Err(CliError::ValidationFailed(format!(
            "Sharded model index for {org}/{repo} contains no shard files"
        )));
    }

    Ok(ResolvedModel::Sharded {
        org: org.to_string(),
        repo: repo.to_string(),
        shard_files,
    })
}

/// Find a SafeTensors file in the repo file list, returning it as a resolved model.
fn find_safetensors_file(filenames: &[&str], org: &str, repo: &str) -> Option<ResolvedModel> {
    if filenames
        .iter()
        .any(|f| f.to_lowercase() == "model.safetensors")
    {
        return Some(ResolvedModel::SingleFile(format!(
            "hf://{org}/{repo}/model.safetensors"
        )));
    }
    filenames
        .iter()
        .find(|f| f.to_lowercase().ends_with(".safetensors"))
        .map(|file| ResolvedModel::SingleFile(format!("hf://{org}/{repo}/{file}")))
}

fn resolve_hf_model(uri: &str) -> Result<ResolvedModel> {
    let uri = normalize_hf_uri(uri);
    let uri = uri.as_str();

    if !uri.starts_with("hf://") {
        return Ok(ResolvedModel::SingleFile(uri.to_string()));
    }

    let has_model_ext = std::path::Path::new(uri).extension().is_some_and(|ext| {
        ext.eq_ignore_ascii_case("gguf")
            || ext.eq_ignore_ascii_case("safetensors")
            || ext.eq_ignore_ascii_case("apr")
            || ext.eq_ignore_ascii_case("pt")
    });
    if has_model_ext {
        return Ok(ResolvedModel::SingleFile(uri.to_string()));
    }

    let path = uri.strip_prefix("hf://").unwrap_or(uri);
    let parts: Vec<&str> = path.split('/').collect();

    if parts.len() < 2 {
        return Err(CliError::ValidationFailed(format!(
            "Invalid HuggingFace URI: {uri}. Expected hf://org/repo or hf://org/repo/file.gguf"
        )));
    }

    let org = parts[0];
    let repo = parts[1];

    let api_url = format!("https://huggingface.co/api/models/{org}/{repo}");
    let response = hf_get(&api_url)
        .call()
        .map_err(|e| CliError::NetworkError(format!("Failed to query HuggingFace API: {e}")))?;

    let body: serde_json::Value = response.into_json().map_err(|e| {
        CliError::ValidationFailed(format!("Failed to parse HuggingFace response: {e}"))
    })?;

    let siblings = body["siblings"]
        .as_array()
        .ok_or_else(|| CliError::ValidationFailed("No files found in repository".to_string()))?;

    let filenames: Vec<&str> = siblings
        .iter()
        .filter_map(|s| s["rfilename"].as_str())
        .collect();

    let gguf_files: Vec<&str> = filenames
        .iter()
        .copied()
        .filter(|f| f.to_lowercase().ends_with(".gguf"))
        .collect();

    if !gguf_files.is_empty() {
        return Ok(select_best_gguf(&gguf_files, org, repo));
    }

    if filenames.contains(&"model.safetensors.index.json") {
        return resolve_sharded_safetensors(org, repo);
    }

    if let Some(model) = find_safetensors_file(&filenames, org, repo) {
        return Ok(model);
    }

    Err(CliError::ValidationFailed(format!(
        "No .gguf or .safetensors files found in {org}/{repo}"
    )))
}

/// GH-213: Extract unique shard filenames from index.json weight_map, sorted for deterministic order.
///
/// Format: `{"metadata": {...}, "weight_map": {"tensor.name": "model-00001-of-00006.safetensors", ...}}`
/// Find the content of a brace-delimited section, handling nesting.
fn find_brace_content(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let content = &text[start + 1..];
    let mut depth = 1usize;
    for (i, c) in content.char_indices() {
        match c {
            '{' => depth += 1,
            '}' if depth == 1 => return Some(&content[..i]),
            '}' => depth -= 1,
            _ => {}
        }
    }
    None
}
