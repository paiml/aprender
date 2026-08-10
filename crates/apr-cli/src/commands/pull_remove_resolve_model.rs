
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
        Ok(())
    } else {
        // GH-601: rm of nonexistent model must exit non-zero (like unix rm).
        println!("{} Model not found in cache", "⚠".yellow());
        Err(CliError::FileNotFound(std::path::PathBuf::from(model_ref)))
    }
}

/// Resolve a model reference to a local path (for run/serve commands)
/// Downloads if not cached and auto_pull is enabled
#[allow(dead_code)]
pub fn resolve_model_path(model_ref: &str) -> Result<std::path::PathBuf> {
    contract_pre_model_path_resolution!();
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

    contract_post_model_path_resolution!(&());
    Ok(result.path)
}

/// Format bytes to human-readable string
fn format_bytes(bytes: u64) -> String {
    batuta_common::fmt::format_bytes(bytes)
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

    // GH-356: tokenizer.json is optional — some models only have tokenizer.model (SentencePiece)
    let companions = [
        "tokenizer.json",
        "config.json",
        "tokenizer_config.json",
        "tokenizer.model",
    ];
    let cache_dir = model_path
        .parent()
        .ok_or_else(|| CliError::ValidationFailed("Model path has no parent directory".into()))?;

    for filename in &companions {
        // GAP-UX-002: Use hash-prefixed filename (e.g., "d71534cb.config.json")
        let prefixed_filename = format!("{model_stem}.{filename}");
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

        // GH-355: Use hf_get() for auth — ureq::get() bypassed gated model tokens
        match hf_get(&url)?.call() {
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
                // File doesn't exist in repo — not fatal for any companion
                println!(
                    "  {} {} (not found in repo)",
                    "⚠".yellow(),
                    prefixed_filename.dimmed()
                );
            }
            Err(ureq::Error::Status(401, _)) => {
                eprintln!(
                    "  {} {} (access denied — set HF_TOKEN for gated models)",
                    "⚠".yellow(),
                    prefixed_filename,
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

    // GH-356: Post-condition — at least one tokenizer file must exist.
    // Same contract as download_companion_files (sharded path). Without this,
    // inference fails late with a cryptic "tokenizer not found" instead of failing fast here.
    let tokenizer_prefixes = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"];
    let has_tokenizer = tokenizer_prefixes
        .iter()
        .any(|f| cache_dir.join(format!("{model_stem}.{f}")).exists());
    if !has_tokenizer {
        return Err(CliError::ValidationFailed(format!(
            "No tokenizer found for this model. Tried: {}.\n\
             The model may require a custom tokenizer not hosted in the repository.",
            tokenizer_prefixes.join(", ")
        )));
    }

    Ok(())
}

/// GH-352: Print a hint about format conversion instead of doing it eagerly.
///
/// Previously (GH-211), this function ran `apr_import()` + `apr_export()` to produce
/// sibling `.apr` and `.gguf` files. This loaded the ENTIRE model into memory — twice —
/// causing 55+ GB RSS for large models like Qwen3-30B-A3B.
///
/// Root cause (five-whys): pull should download only. Conversion is `apr convert`'s job.
/// The realizar inference engine reads SafeTensors directly — no conversion needed to run.
fn convert_safetensors_formats(safetensors_path: &Path) -> Result<()> {
    let apr_path = safetensors_path.with_extension("apr");
    let gguf_path = safetensors_path.with_extension("gguf");

    // If both already exist (from a previous pull), just note it
    if apr_path.exists() && gguf_path.exists() {
        println!();
        println!(
            "  {} APR and GGUF formats available",
            "✓".green(),
        );
        return Ok(());
    }

    // GH-352: Hint instead of eagerly converting (which loads entire model into RAM)
    println!();
    println!(
        "  {} To convert formats, run:",
        "ℹ".cyan(),
    );
    if !apr_path.exists() {
        println!(
            "    apr convert {} --format apr",
            safetensors_path.display()
        );
    }
    if !gguf_path.exists() {
        println!(
            "    apr convert {} --format gguf",
            safetensors_path.display()
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
        let matches: Vec<_> = gguf_files.iter().filter(|f| f.to_lowercase().contains(quant)).collect();
        if matches.len() == 1 {
            return ResolvedModel::SingleFile(format!("hf://{org}/{repo}/{}", matches[0]));
        } else if matches.len() > 1 {
            // For now, if there are multiple parts (sharded GGUF), just pick the first one 
            // (Note: full sharded GGUF support in `apr pull` might require more work, but this avoids random picking).
            // Usually the first shard contains metadata, but it's not a full model.
            // A better fix for sharded GGUF is to download all parts, but pacha single-file streaming doesn't support that yet.
            // We will just pick the first part so it doesn't crash on `find`.
            if let Some(first_part) = matches.iter().find(|f| f.contains("-00001-of-")) {
                return ResolvedModel::SingleFile(format!("hf://{org}/{repo}/{}", first_part));
            }
            return ResolvedModel::SingleFile(format!("hf://{org}/{repo}/{}", matches[0]));
        }
    }
    ResolvedModel::SingleFile(format!("hf://{org}/{repo}/{}", gguf_files[0]))
}

/// Download and parse sharded SafeTensors index, returning shard filenames.
fn resolve_sharded_safetensors(org: &str, repo: &str) -> Result<ResolvedModel> {
    let index_url =
        format!("https://huggingface.co/{org}/{repo}/resolve/main/model.safetensors.index.json");
    let index_response = hf_get(&index_url)?
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

/// Check if a URI already has a known model file extension.
fn has_known_model_extension(uri: &str) -> bool {
    std::path::Path::new(uri).extension().is_some_and(|ext| {
        ext.eq_ignore_ascii_case("gguf")
            || ext.eq_ignore_ascii_case("safetensors")
            || ext.eq_ignore_ascii_case("apr")
            || ext.eq_ignore_ascii_case("pt")
    })
}

pub(crate) fn resolve_hf_model(uri: &str) -> Result<ResolvedModel> {
    let uri = normalize_hf_uri(uri);
    let uri = uri.as_str();

    if !uri.starts_with("hf://") {
        return Ok(ResolvedModel::SingleFile(uri.to_string()));
    }

    if has_known_model_extension(uri) {
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
    let response = hf_get(&api_url)?.call().map_err(|e| match &e {
        ureq::Error::Status(401, _) => {
            CliError::NetworkError(format_gated_model_error(&api_url))
        }
        _ => CliError::NetworkError(format!("Failed to query HuggingFace API: {e}")),
    })?;

    let body: serde_json::Value = {
        let text = response.into_string().map_err(|e| {
            CliError::ValidationFailed(format!("Failed to read HuggingFace response: {e}"))
        })?;
        serde_json::from_str(&text).map_err(|e| {
            CliError::ValidationFailed(format!("Failed to parse HuggingFace response: {e}"))
        })?
    };

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
        // #1893: a complete sharded-GGUF set (`model-NNNNN-of-MMMMM.gguf`, no
        // index.json) must download ALL parts — not a single `select_best_gguf`
        // pick, which would silently grab one part and produce a broken model.
        if let Some(shard_files) = detect_gguf_shards(&gguf_files) {
            return Ok(ResolvedModel::Sharded {
                org: org.to_string(),
                repo: repo.to_string(),
                shard_files,
            });
        }
        return Ok(select_best_gguf(&gguf_files, org, repo));
    }

    if filenames.contains(&"model.safetensors.index.json") {
        return resolve_sharded_safetensors(org, repo);
    }

    if let Some(model) = find_safetensors_file(&filenames, org, repo) {
        return Ok(model);
    }

    resolve_hf_model_fallback(&filenames, org, repo)
}

/// GH-357: Handle repos with no GGUF/SafeTensors — detect PyTorch-only repos.
fn resolve_hf_model_fallback(filenames: &[&str], org: &str, repo: &str) -> Result<ResolvedModel> {
    let has_bin_files = filenames
        .iter()
        .any(|f| f.to_lowercase().ends_with(".bin"));
    if has_bin_files {
        return Err(CliError::ValidationFailed(format!(
            "{org}/{repo} only has PyTorch .bin weights (no SafeTensors or GGUF).\n\
             Convert first with:\n  \
             python -c \"from transformers import AutoModelForCausalLM; \
             m = AutoModelForCausalLM.from_pretrained('{org}/{repo}'); \
             m.save_pretrained('{repo}-st', safe_serialization=True)\"\n\
             Or request SafeTensors on the model page."
        )));
    }

    Err(CliError::ValidationFailed(format!(
        "No .gguf or .safetensors files found in {org}/{repo}"
    )))
}

/// #1893: Parse a sharded-GGUF filename `<prefix>-NNNNN-of-MMMMM.gguf` into
/// `(prefix_lowercased, part_no, total)`. Returns `None` for any name that
/// isn't a shard part (single-file GGUF, multi-quant repos, etc.).
///
/// The `.gguf` suffix and prefix are matched case-insensitively for grouping;
/// the caller keeps the original-case filename for download.
fn parse_gguf_shard_name(name: &str) -> Option<(String, u32, u32)> {
    let lower = name.to_lowercase();
    let stem = lower.strip_suffix(".gguf")?;
    // "<prefix>-NNNNN" + "-of-" + "MMMMM"
    let (prefix_and_part, total_str) = stem.rsplit_once("-of-")?;
    let total: u32 = total_str.parse().ok()?;
    let (prefix, part_str) = prefix_and_part.rsplit_once('-')?;
    let part: u32 = part_str.parse().ok()?;
    Some((prefix.to_string(), part, total))
}

/// #1893: Detect a COMPLETE sharded-GGUF set among a repo's `.gguf` files.
///
/// Modern 7B+ GGUFs ship split as `<prefix>-NNNNN-of-MMMMM.gguf` (zero-padded,
/// 1-indexed) with NO `index.json` (unlike sharded SafeTensors). Returns the
/// part filenames sorted by part number IFF a single prefix has a complete set
/// (`total >= 2` and all parts `1..=total` present). Returns `None` for a
/// single-file GGUF, unrelated multi-quant GGUFs, or an incomplete set — so the
/// caller falls back to single-file selection (`select_best_gguf`).
fn detect_gguf_shards(gguf_files: &[&str]) -> Option<Vec<String>> {
    use std::collections::BTreeMap;
    // (prefix, total) -> { part_no -> original_filename }
    let mut groups: BTreeMap<(String, u32), BTreeMap<u32, String>> = BTreeMap::new();
    for &f in gguf_files {
        if let Some((prefix, part, total)) = parse_gguf_shard_name(f) {
            groups
                .entry((prefix, total))
                .or_default()
                .insert(part, f.to_string());
        }
    }
    for ((_, total), parts) in groups {
        if total >= 2 && parts.len() as u32 == total && (1..=total).all(|n| parts.contains_key(&n)) {
            // BTreeMap iterates by ascending part number → correctly ordered.
            return Some(parts.into_values().collect());
        }
    }
    None
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

#[cfg(test)]
mod sharded_gguf_tests {
    use super::{detect_gguf_shards, parse_gguf_shard_name};

    // ===== #1893: sharded-GGUF detection (contract sharded-gguf-pull-v1) =====

    /// FT-SHGGUF-001: a complete 2-part set is detected and returned in part order.
    #[test]
    fn detects_complete_two_part_set() {
        let files = ["m-00002-of-00002.gguf", "m-00001-of-00002.gguf"];
        let got = detect_gguf_shards(&files).expect("complete set must be detected");
        assert_eq!(
            got,
            vec!["m-00001-of-00002.gguf".to_string(), "m-00002-of-00002.gguf".to_string()],
            "FT-SHGGUF-001: parts returned sorted by part number regardless of input order"
        );
    }

    /// FT-SHGGUF-002: a complete 3-part set with a realistic prefix.
    #[test]
    fn detects_three_part_with_quant_prefix() {
        let files = [
            "Qwen2.5-7B-Instruct-Q4_K_M-00001-of-00003.gguf",
            "Qwen2.5-7B-Instruct-Q4_K_M-00002-of-00003.gguf",
            "Qwen2.5-7B-Instruct-Q4_K_M-00003-of-00003.gguf",
        ];
        let got = detect_gguf_shards(&files).expect("3-part set");
        assert_eq!(got.len(), 3);
        assert_eq!(got[0], "Qwen2.5-7B-Instruct-Q4_K_M-00001-of-00003.gguf");
    }

    /// FT-SHGGUF-003: a single non-sharded GGUF is NOT treated as sharded
    /// (caller falls back to select_best_gguf).
    #[test]
    fn single_file_is_not_sharded() {
        assert_eq!(detect_gguf_shards(&["model.gguf"]), None);
        assert_eq!(detect_gguf_shards(&["qwen2.5-coder-1.5b-q4_k_m.gguf"]), None);
    }

    /// FT-SHGGUF-004: unrelated multi-quant GGUFs (not a shard set) → None.
    #[test]
    fn multi_quant_not_sharded() {
        let files = ["model-Q4_K_M.gguf", "model-Q8_0.gguf", "model-f16.gguf"];
        assert_eq!(detect_gguf_shards(&files), None);
    }

    /// FT-SHGGUF-005: an INCOMPLETE set (missing a part) → None, so we never
    /// claim a model is downloadable when a part is absent.
    #[test]
    fn incomplete_set_rejected() {
        let files = ["m-00001-of-00003.gguf", "m-00002-of-00003.gguf"]; // missing 00003
        assert_eq!(detect_gguf_shards(&files), None);
    }

    /// FT-SHGGUF-006: filename parser handles case + rejects non-shard names.
    #[test]
    fn parser_edge_cases() {
        assert_eq!(
            parse_gguf_shard_name("M-00001-OF-00004.GGUF"),
            Some(("m".to_string(), 1, 4))
        );
        assert_eq!(parse_gguf_shard_name("model.gguf"), None);
        assert_eq!(parse_gguf_shard_name("foo-bar.gguf"), None);
        assert_eq!(parse_gguf_shard_name("model.safetensors"), None);
    }
}
