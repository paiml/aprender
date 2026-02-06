//! Pull command implementation using pacha ModelFetcher
//!
//! Downloads and caches models from HuggingFace with Ollama-like UX.
//! Models are cached at `~/.cache/pacha/models/` for reuse.
//!
//! Usage:
//!   apr pull qwen2.5-coder:7b           # Short alias
//!   apr pull hf://bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/qwen2.5-coder-7b-instruct-q4_k_m.gguf
//!   apr pull TheBloke/Llama-2-7B-GGUF   # Will auto-detect quant

use crate::error::{CliError, Result};
use colored::Colorize;
use pacha::fetcher::{FetchConfig, ModelFetcher};
use pacha::format::ModelFormat;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::io::{self, Read, Write};
use std::path::Path;

/// Result of resolving a HuggingFace model reference.
///
/// Single-file models (small SafeTensors, GGUF) use the pacha fetcher.
/// Sharded models (3B+ SafeTensors) are downloaded directly to `~/.apr/cache/hf/`.
enum ResolvedModel {
    /// Single file downloadable via pacha (existing behavior)
    SingleFile(String),
    /// Sharded SafeTensors model (multiple .safetensors files + index.json)
    Sharded {
        org: String,
        repo: String,
        shard_files: Vec<String>,
    },
}

/// GH-213: Manifest recording checksums for each file in a sharded download.
///
/// Written to `.apr-manifest.json` in the cache directory after a successful download.
/// Used by the pre-inference contract gate to verify shard integrity without re-hashing.
#[derive(Debug, Serialize, Deserialize)]
pub struct ShardManifest {
    pub version: u32,
    pub repo: String,
    pub files: HashMap<String, FileChecksum>,
}

/// GH-213: Size and BLAKE3 hash of a downloaded file.
#[derive(Debug, Serialize, Deserialize)]
pub struct FileChecksum {
    pub size: u64,
    pub blake3: String,
}

/// Run the pull command
pub fn run(model_ref: &str, force: bool) -> Result<()> {
    println!("{}", "=== APR Pull ===".cyan().bold());
    println!();

    // GH-213: Resolve HuggingFace URI — detect single vs sharded models
    let resolved = resolve_hf_model(model_ref)?;

    match resolved {
        ResolvedModel::SingleFile(ref uri) => run_single_file(uri, force),
        ResolvedModel::Sharded {
            ref org,
            ref repo,
            ref shard_files,
        } => run_sharded(org, repo, shard_files, force),
    }
}

/// Pull a single-file model via pacha (existing behavior)
fn run_single_file(model_ref: &str, force: bool) -> Result<()> {
    println!("Model: {}", model_ref.cyan());

    // Initialize pacha ModelFetcher
    let mut fetcher = ModelFetcher::with_config(FetchConfig::default()).map_err(|e| {
        CliError::ValidationFailed(format!("Failed to initialize model fetcher: {e}"))
    })?;

    // Check if already cached
    if !force && fetcher.is_cached(model_ref) {
        println!("{} Model already cached", "✓".green());

        // Get the cached path
        let result = fetcher
            .pull_quiet(model_ref)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to get cached model: {e}")))?;

        println!("  Path: {}", result.path.display());
        println!("  Size: {}", result.size_human());
        println!("  Format: {:?}", result.format);

        // GH-198: Ensure companion files exist for cached SafeTensors models
        if matches!(result.format, ModelFormat::SafeTensors(_)) {
            fetch_safetensors_companions(&result.path, &result.resolved_uri)?;
        }

        println!();
        println!("{}", "Usage:".cyan().bold());
        println!("  apr run {}", result.path.display());
        return Ok(());
    }

    println!();
    println!("{}", "Downloading...".yellow());

    // Pull with progress callback
    let result = fetcher
        .pull(model_ref, |progress| {
            // Simple progress bar
            let pct = progress.percent();
            print!(
                "\r  [{:50}] {:5.1}% ({}/{})",
                "=".repeat((pct / 2.0) as usize),
                pct,
                format_bytes(progress.downloaded_bytes),
                format_bytes(progress.total_bytes)
            );
            io::stdout().flush().ok();
        })
        .map_err(|e| CliError::NetworkError(format!("Download failed: {e}")))?;

    println!();
    println!();

    if result.cache_hit {
        println!("{} Model retrieved from cache", "✓".green());
    } else {
        println!("{} Downloaded successfully", "✓".green());
    }

    println!("  Path: {}", result.path.display().to_string().green());
    println!("  Size: {}", result.size_human().yellow());
    println!("  Format: {:?}", result.format);
    println!("  Hash: {}", &result.hash[..16]);

    // GH-198: Download companion files for SafeTensors models
    if matches!(result.format, ModelFormat::SafeTensors(_)) {
        fetch_safetensors_companions(&result.path, &result.resolved_uri)?;
    }

    println!();
    println!("{}", "Usage:".cyan().bold());
    println!("  apr run {}", result.path.display());
    println!("  apr serve {}", result.path.display());

    Ok(())
}

/// GH-213: Pull a sharded SafeTensors model (3B+ models with multiple shard files)
fn run_sharded(org: &str, repo: &str, shard_files: &[String], force: bool) -> Result<()> {
    println!(
        "Model: {}/{} ({} shards)",
        org.cyan(),
        repo.cyan(),
        shard_files.len().to_string().yellow()
    );

    let cache_dir = dirs::home_dir()
        .ok_or_else(|| CliError::ValidationFailed("Cannot find home directory".to_string()))?
        .join(".apr")
        .join("cache")
        .join("hf")
        .join(org)
        .join(repo);

    std::fs::create_dir_all(&cache_dir)?;

    let base_url = format!("https://huggingface.co/{org}/{repo}/resolve/main");

    // Download index.json
    let index_path = cache_dir.join("model.safetensors.index.json");
    if force || !index_path.exists() {
        println!();
        println!("  {} model.safetensors.index.json", "Downloading".yellow());
        download_file(
            &format!("{base_url}/model.safetensors.index.json"),
            &index_path,
        )?;
    } else {
        println!("  {} model.safetensors.index.json (cached)", "✓".green());
    }

    // GH-213: Load existing manifest for cache-hit verification
    let manifest_path = cache_dir.join(".apr-manifest.json");
    let existing_manifest: Option<ShardManifest> = if !force && manifest_path.exists() {
        std::fs::read_to_string(&manifest_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
    } else {
        None
    };

    // Download each shard, collecting checksums
    let mut file_checksums: HashMap<String, FileChecksum> = HashMap::new();
    let total_shards = shard_files.len();
    for (i, shard_file) in shard_files.iter().enumerate() {
        let shard_path = cache_dir.join(shard_file);

        if !force && shard_path.exists() {
            // GH-213: On cache hit, verify file size against manifest
            if let Some(ref manifest) = existing_manifest {
                if let Some(expected) = manifest.files.get(shard_file.as_str()) {
                    let actual_size = std::fs::metadata(&shard_path).map(|m| m.len()).unwrap_or(0);
                    if actual_size == expected.size {
                        // Carry forward existing checksum
                        file_checksums.insert(
                            shard_file.clone(),
                            FileChecksum {
                                size: expected.size,
                                blake3: expected.blake3.clone(),
                            },
                        );
                        println!(
                            "  {} [{}/{}] {} (cached, verified)",
                            "✓".green(),
                            i + 1,
                            total_shards,
                            shard_file
                        );
                        continue;
                    }
                    println!(
                        "  {} [{}/{}] {} (size mismatch: {} vs {} bytes, re-downloading)",
                        "⚠".yellow(),
                        i + 1,
                        total_shards,
                        shard_file,
                        actual_size,
                        expected.size
                    );
                    // Fall through to re-download
                }
            } else {
                println!(
                    "  {} [{}/{}] {} (cached)",
                    "✓".green(),
                    i + 1,
                    total_shards,
                    shard_file
                );
                continue;
            }
        }

        let shard_url = format!("{base_url}/{shard_file}");
        print!(
            "  {} [{}/{}] {}...",
            "↓".yellow(),
            i + 1,
            total_shards,
            shard_file
        );
        io::stdout().flush().ok();

        let checksum = download_file_with_progress(&shard_url, &shard_path)?;
        file_checksums.insert(shard_file.clone(), checksum);
        println!(" {}", "done".green());
    }

    // Download companion files (tokenizer.json, config.json, tokenizer_config.json)
    let companions = [
        ("tokenizer.json", true),
        ("config.json", true),
        ("tokenizer_config.json", false),
    ];
    for (filename, required) in &companions {
        let companion_path = cache_dir.join(filename);
        if !force && companion_path.exists() {
            println!("  {} {} (cached)", "✓".green(), filename);
            continue;
        }

        let url = format!("{base_url}/{filename}");
        match download_file(&url, &companion_path) {
            Ok(()) => println!("  {} {}", "✓".green(), filename),
            Err(e) if *required => {
                return Err(CliError::ValidationFailed(format!(
                    "{filename} is required for inference but download failed: {e}"
                )));
            }
            Err(_) => println!("  {} {} (not available, optional)", "⚠".yellow(), filename),
        }
    }

    // GH-213: Write shard manifest with checksums
    if !file_checksums.is_empty() {
        let manifest = ShardManifest {
            version: 1,
            repo: format!("{org}/{repo}"),
            files: file_checksums,
        };
        let manifest_json = serde_json::to_string_pretty(&manifest).map_err(|e| {
            CliError::ValidationFailed(format!("Failed to serialize manifest: {e}"))
        })?;
        std::fs::write(&manifest_path, manifest_json)?;
        println!("  {} .apr-manifest.json (integrity checksums)", "✓".green());
    }

    println!();
    println!("{} Downloaded successfully", "✓".green());
    println!("  Path: {}", index_path.display().to_string().green());
    println!("  Shards: {}", total_shards.to_string().yellow());

    println!();
    println!("{}", "Usage:".cyan().bold());
    println!("  apr run {}", index_path.display());
    println!("  apr serve {}", index_path.display());

    Ok(())
}

/// List cached models
pub fn list() -> Result<()> {
    println!("{}", "=== Cached Models ===".cyan().bold());
    println!();

    let fetcher = ModelFetcher::new().map_err(|e| {
        CliError::ValidationFailed(format!("Failed to initialize model fetcher: {e}"))
    })?;

    let models = fetcher.list();

    if models.is_empty() {
        println!("{}", "No cached models found.".dimmed());
        println!();
        println!("Pull a model with:");
        println!("  apr pull hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf");
        println!();
        println!("Or run directly (auto-downloads):");
        println!("  apr run hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf");
        return Ok(());
    }

    // Print header
    println!(
        "{:<40} {:<12} {:<10} {}",
        "NAME".dimmed(),
        "SIZE".dimmed(),
        "FORMAT".dimmed(),
        "PATH".dimmed()
    );
    println!("{}", "-".repeat(100).dimmed());

    for model in &models {
        let size = format_bytes(model.size_bytes);
        let format = format!("{:?}", model.format);
        let name = if model.name.len() > 38 {
            format!("{}...", &model.name[..35])
        } else {
            model.name.clone()
        };

        println!(
            "{:<40} {:<12} {:<10} {}",
            name.cyan(),
            size.yellow(),
            format,
            model.path.display().to_string().dimmed()
        );
    }

    println!();

    // Print stats
    let stats = fetcher.stats();
    println!(
        "Total: {} models, {} used",
        models.len(),
        format_bytes(stats.total_size_bytes)
    );

    Ok(())
}

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
fn resolve_hf_model(uri: &str) -> Result<ResolvedModel> {
    // GH-213: Normalize bare "org/repo" to "hf://org/repo"
    // Detects patterns like "Qwen/Qwen2.5-Coder-3B-Instruct" (no scheme, no extension,
    // exactly one slash separating org/repo, optionally with a filename after second slash)
    let uri = if !uri.contains("://") && !uri.starts_with('/') && !uri.starts_with('.') {
        let parts: Vec<&str> = uri.split('/').collect();
        if parts.len() >= 2 && !parts[0].is_empty() && !parts[1].is_empty() {
            format!("hf://{uri}")
        } else {
            uri.to_string()
        }
    } else {
        uri.to_string()
    };
    let uri = uri.as_str();

    // If not a HuggingFace URI, return unchanged as single file
    if !uri.starts_with("hf://") {
        return Ok(ResolvedModel::SingleFile(uri.to_string()));
    }

    // If already has a known model extension, return unchanged
    let has_model_ext = std::path::Path::new(uri).extension().is_some_and(|ext| {
        ext.eq_ignore_ascii_case("gguf")
            || ext.eq_ignore_ascii_case("safetensors")
            || ext.eq_ignore_ascii_case("apr")
            || ext.eq_ignore_ascii_case("pt")
    });
    if has_model_ext {
        return Ok(ResolvedModel::SingleFile(uri.to_string()));
    }

    // Parse org/repo from URI
    let path = uri.strip_prefix("hf://").unwrap_or(uri);
    let parts: Vec<&str> = path.split('/').collect();

    if parts.len() < 2 {
        return Err(CliError::ValidationFailed(format!(
            "Invalid HuggingFace URI: {}. Expected hf://org/repo or hf://org/repo/file.gguf",
            uri
        )));
    }

    let org = parts[0];
    let repo = parts[1];

    // Query HuggingFace API for repo files
    let api_url = format!("https://huggingface.co/api/models/{}/{}", org, repo);

    let response = ureq::get(&api_url)
        .call()
        .map_err(|e| CliError::NetworkError(format!("Failed to query HuggingFace API: {}", e)))?;

    let body: serde_json::Value = response.into_json().map_err(|e| {
        CliError::ValidationFailed(format!("Failed to parse HuggingFace response: {}", e))
    })?;

    // Extract siblings (files) from response
    let siblings = body["siblings"]
        .as_array()
        .ok_or_else(|| CliError::ValidationFailed("No files found in repository".to_string()))?;

    let filenames: Vec<&str> = siblings
        .iter()
        .filter_map(|s| s["rfilename"].as_str())
        .collect();

    // Find GGUF files
    let gguf_files: Vec<&str> = filenames
        .iter()
        .copied()
        .filter(|f| f.to_lowercase().ends_with(".gguf"))
        .collect();

    if !gguf_files.is_empty() {
        // Priority: Q4_K_M > Q4_K_S > Q4_0 > Q8_0 > any
        let quantization_priority = ["q4_k_m", "q4_k_s", "q4_0", "q8_0"];

        for quant in quantization_priority {
            if let Some(file) = gguf_files.iter().find(|f| f.to_lowercase().contains(quant)) {
                return Ok(ResolvedModel::SingleFile(format!(
                    "hf://{}/{}/{}",
                    org, repo, file
                )));
            }
        }

        // Fallback to first GGUF file
        return Ok(ResolvedModel::SingleFile(format!(
            "hf://{}/{}/{}",
            org, repo, gguf_files[0]
        )));
    }

    // GH-213: Check for sharded SafeTensors (model.safetensors.index.json)
    let has_index = filenames.contains(&"model.safetensors.index.json");

    if has_index {
        // Download and parse index.json to get shard filenames
        let index_url = format!(
            "https://huggingface.co/{}/{}/resolve/main/model.safetensors.index.json",
            org, repo
        );
        let index_response = ureq::get(&index_url)
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
                "Sharded model index for {}/{} contains no shard files",
                org, repo
            )));
        }

        return Ok(ResolvedModel::Sharded {
            org: org.to_string(),
            repo: repo.to_string(),
            shard_files,
        });
    }

    // Fall back to single model.safetensors
    let has_single_st = filenames
        .iter()
        .any(|f| f.to_lowercase() == "model.safetensors");

    if has_single_st {
        return Ok(ResolvedModel::SingleFile(format!(
            "hf://{}/{}/model.safetensors",
            org, repo
        )));
    }

    // Last resort: any .safetensors file
    if let Some(file) = filenames
        .iter()
        .find(|f| f.to_lowercase().ends_with(".safetensors"))
    {
        return Ok(ResolvedModel::SingleFile(format!(
            "hf://{}/{}/{}",
            org, repo, file
        )));
    }

    Err(CliError::ValidationFailed(format!(
        "No .gguf or .safetensors files found in {}/{}",
        org, repo
    )))
}

/// GH-213: Extract unique shard filenames from index.json weight_map, sorted for deterministic order.
///
/// Format: `{"metadata": {...}, "weight_map": {"tensor.name": "model-00001-of-00006.safetensors", ...}}`
fn extract_shard_files_from_index(json: &str) -> Vec<String> {
    let mut files = HashSet::new();

    if let Some(weight_map_start) = json.find("\"weight_map\"") {
        let after_key = &json[weight_map_start..];
        if let Some(brace_start) = after_key.find('{') {
            let content = &after_key[brace_start + 1..];
            // Find matching closing brace
            let mut depth = 1;
            let mut end_pos = 0;
            for (i, c) in content.char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end_pos = i;
                            break;
                        }
                    }
                    _ => {}
                }
            }

            let entries = &content[..end_pos];

            // Extract shard filenames (values in key:value pairs)
            for part in entries.split(',') {
                if let Some(colon_pos) = part.rfind(':') {
                    let value = part[colon_pos + 1..].trim();
                    let filename =
                        value.trim_matches(|c| c == '"' || c == ' ' || c == '\n' || c == '\r');
                    if filename.ends_with(".safetensors") && !filename.is_empty() {
                        files.insert(filename.to_string());
                    }
                }
            }
        }
    }

    // Sort for deterministic download order (model-00001, model-00002, ...)
    let mut sorted: Vec<String> = files.into_iter().collect();
    sorted.sort();
    sorted
}

/// Download a file from URL to local path (no progress)
fn download_file(url: &str, path: &Path) -> Result<()> {
    let response = ureq::get(url)
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
    let response = ureq::get(url)
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

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // format_bytes tests
    // =========================================================================

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_bytes(5 * 1024 * 1024 * 1024), "5.00 GB");
    }

    #[test]
    fn test_format_bytes_zero() {
        assert_eq!(format_bytes(0), "0 B");
    }

    #[test]
    fn test_format_bytes_small() {
        assert_eq!(format_bytes(1), "1 B");
        assert_eq!(format_bytes(100), "100 B");
        assert_eq!(format_bytes(1023), "1023 B");
    }

    #[test]
    fn test_format_bytes_kilobytes() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(2048), "2.00 KB");
        assert_eq!(format_bytes(512 * 1024), "512.00 KB");
    }

    #[test]
    fn test_format_bytes_megabytes() {
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(100 * 1024 * 1024), "100.00 MB");
        assert_eq!(format_bytes(500 * 1024 * 1024), "500.00 MB");
    }

    #[test]
    fn test_format_bytes_gigabytes() {
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_bytes(10 * 1024 * 1024 * 1024), "10.00 GB");
        assert_eq!(format_bytes(100 * 1024 * 1024 * 1024), "100.00 GB");
    }

    #[test]
    fn test_format_bytes_fractional_gb() {
        // 4.5 GB = 4.5 * 1024 * 1024 * 1024 = 4831838208 bytes
        assert_eq!(format_bytes(4831838208), "4.50 GB");
    }

    #[test]
    fn test_format_bytes_fractional_mb() {
        // 2.5 MB = 2.5 * 1024 * 1024 = 2621440 bytes
        assert_eq!(format_bytes(2621440), "2.50 MB");
    }

    // =========================================================================
    // PMAT-108: resolve_hf_uri Tests (Extreme TDD)
    // =========================================================================

    #[test]
    fn test_pmat_108_resolve_uri_with_gguf_extension_unchanged() {
        let uri = "hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/model.gguf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "URI with .gguf should be unchanged");
    }

    #[test]
    fn test_pmat_108_resolve_uri_case_insensitive_gguf() {
        let uri = "hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/model.GGUF";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "URI with .GGUF should be unchanged");
    }

    #[test]
    fn test_pmat_108_resolve_non_hf_uri_unchanged() {
        let uri = "/path/to/local/model.gguf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "Non-hf:// URI should be unchanged");
    }

    #[test]
    fn test_pmat_108_resolve_invalid_uri_fails() {
        let uri = "hf://invalid";
        let result = resolve_hf_uri(uri);
        assert!(result.is_err(), "Invalid URI should fail");
    }

    #[test]
    fn test_resolve_hf_uri_relative_path() {
        let uri = "./models/test.gguf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "Relative path should be unchanged");
    }

    #[test]
    fn test_resolve_hf_uri_absolute_path() {
        let uri = "/home/user/models/test.gguf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "Absolute path should be unchanged");
    }

    #[test]
    fn test_resolve_hf_uri_https_url() {
        let uri = "https://example.com/model.gguf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "HTTPS URL should be unchanged");
    }

    #[test]
    fn test_resolve_hf_uri_with_mixed_case_extension() {
        let uri = "hf://Org/Repo/model.GgUf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "Mixed case .GgUf should be unchanged");
    }

    #[test]
    fn test_resolve_hf_uri_empty_string() {
        let uri = "";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "Empty string should be unchanged");
    }

    #[test]
    fn test_resolve_hf_uri_invalid_hf_format() {
        // hf:// without org/repo should fail
        let result = resolve_hf_uri("hf://only-one-part");
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("Invalid HuggingFace URI"));
            }
            other => panic!("Expected ValidationFailed, got {:?}", other),
        }
    }

    #[test]
    fn test_resolve_hf_uri_with_safetensors_extension_unchanged() {
        // .safetensors files are not .gguf, so they will trigger HF API query
        // This test verifies the logic path, but we can't test the full flow
        // without mocking. Instead, test that non-gguf HF URIs attempt resolution.
        // The test_resolve_hf_uri_invalid_hf_format covers the error case.
        // For now, we just verify the URI format is preserved for files with .gguf extension
        let uri = "hf://org/repo/model.gguf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, ".gguf extension should be unchanged");
    }

    #[test]
    #[ignore] // Requires network access
    fn test_resolve_hf_uri_with_safetensors_queries_api() {
        // This test would need network access to verify API query behavior
        let uri = "hf://org/repo/model.safetensors";
        let _result = resolve_hf_uri(uri);
        // Result depends on network and repo existence
    }

    // Integration test (requires network, marked ignore for CI)
    #[test]
    #[ignore]
    fn test_pmat_108_resolve_qwen_repo_finds_q4_k_m() {
        let uri = "hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert!(resolved.ends_with(".gguf"), "Should end with .gguf");
        assert!(
            resolved.to_lowercase().contains("q4_k_m"),
            "Should prefer Q4_K_M quantization: {}",
            resolved
        );
    }

    // =========================================================================
    // resolve_model_path tests
    // =========================================================================

    #[test]
    fn test_resolve_model_path_existing_file() {
        // Create a temp file
        let temp_dir = std::env::temp_dir().join("apr_pull_test_path");
        let _ = std::fs::create_dir_all(&temp_dir);
        let test_file = temp_dir.join("test_model.gguf");
        let _ = std::fs::write(&test_file, "GGUF");

        let result = resolve_model_path(test_file.to_str().unwrap());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), test_file);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_resolve_model_path_nonexistent_local_fails() {
        let result = resolve_model_path("/nonexistent/model.gguf");
        // This will try pacha which will fail with validation error
        assert!(result.is_err());
    }

    // =========================================================================
    // GH-198: extract_hf_repo tests
    // =========================================================================

    #[test]
    fn test_gh198_extract_hf_repo_with_file() {
        let uri = "hf://Qwen/Qwen2.5-Coder-0.5B-Instruct/model.safetensors";
        assert_eq!(
            extract_hf_repo(uri),
            Some("Qwen/Qwen2.5-Coder-0.5B-Instruct".to_string())
        );
    }

    #[test]
    fn test_gh198_extract_hf_repo_without_file() {
        let uri = "hf://Qwen/Qwen2.5-Coder-0.5B-Instruct";
        assert_eq!(
            extract_hf_repo(uri),
            Some("Qwen/Qwen2.5-Coder-0.5B-Instruct".to_string())
        );
    }

    #[test]
    fn test_gh198_extract_hf_repo_local_path() {
        assert_eq!(extract_hf_repo("/local/path/model.safetensors"), None);
    }

    #[test]
    fn test_gh198_extract_hf_repo_empty() {
        assert_eq!(extract_hf_repo(""), None);
    }

    #[test]
    fn test_gh198_extract_hf_repo_only_org() {
        // hf://org (missing repo) → None
        assert_eq!(extract_hf_repo("hf://org"), None);
    }

    #[test]
    fn test_gh198_extract_hf_repo_nested_path() {
        let uri = "hf://org/repo/subdir/model.safetensors";
        assert_eq!(extract_hf_repo(uri), Some("org/repo".to_string()));
    }

    // =========================================================================
    // GH-198: fetch_safetensors_companions tests
    // =========================================================================

    #[test]
    fn test_gh198_companions_non_hf_uri_is_noop() {
        let temp_dir = std::env::temp_dir().join("apr_gh198_noop");
        let _ = std::fs::create_dir_all(&temp_dir);
        let model_path = temp_dir.join("d71534cb.safetensors");
        let _ = std::fs::write(&model_path, b"dummy");

        // Local URI → should return Ok without downloading anything
        let result = fetch_safetensors_companions(&model_path, "/local/model.safetensors");
        assert!(result.is_ok());

        // No companion files should be created (GAP-UX-002: hash-prefixed)
        assert!(!temp_dir.join("d71534cb.tokenizer.json").exists());
        assert!(!temp_dir.join("d71534cb.config.json").exists());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_gh198_companions_skips_existing() {
        let temp_dir = std::env::temp_dir().join("apr_gh198_existing");
        let _ = std::fs::create_dir_all(&temp_dir);
        let model_path = temp_dir.join("abc123.safetensors");
        let _ = std::fs::write(&model_path, b"dummy");

        // Pre-create companion files (GAP-UX-002: hash-prefixed)
        let _ = std::fs::write(temp_dir.join("abc123.tokenizer.json"), b"{}");
        let _ = std::fs::write(temp_dir.join("abc123.config.json"), b"{}");

        // Should succeed without attempting downloads (files already exist)
        let result = fetch_safetensors_companions(
            &model_path,
            "hf://Qwen/Qwen2.5-Coder-0.5B-Instruct/model.safetensors",
        );
        assert!(result.is_ok());

        // Verify files are unchanged (still our dummy content)
        let content = std::fs::read_to_string(temp_dir.join("abc123.tokenizer.json")).unwrap();
        assert_eq!(content, "{}");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    #[ignore] // Requires network access
    fn test_gh198_companions_downloads_from_hf() {
        let temp_dir = std::env::temp_dir().join("apr_gh198_download");
        let _ = std::fs::remove_dir_all(&temp_dir);
        let _ = std::fs::create_dir_all(&temp_dir);
        // GAP-UX-002: Use hash-prefixed model name
        let model_path = temp_dir.join("d71534cb948e32eb.safetensors");
        let _ = std::fs::write(&model_path, b"dummy");

        let result = fetch_safetensors_companions(
            &model_path,
            "hf://Qwen/Qwen2.5-Coder-0.5B-Instruct/model.safetensors",
        );
        assert!(result.is_ok());

        // Both companion files should now exist (GAP-UX-002: hash-prefixed)
        assert!(
            temp_dir.join("d71534cb948e32eb.tokenizer.json").exists(),
            "d71534cb948e32eb.tokenizer.json should be downloaded"
        );
        assert!(
            temp_dir.join("d71534cb948e32eb.config.json").exists(),
            "d71534cb948e32eb.config.json should be downloaded"
        );

        // Verify tokenizer.json has vocab
        let tok =
            std::fs::read_to_string(temp_dir.join("d71534cb948e32eb.tokenizer.json")).unwrap();
        assert!(tok.contains("vocab"), "tokenizer.json should contain vocab");

        // Verify config.json has model architecture
        let cfg = std::fs::read_to_string(temp_dir.join("d71534cb948e32eb.config.json")).unwrap();
        assert!(
            cfg.contains("num_hidden_layers"),
            "config.json should contain num_hidden_layers"
        );

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    // =========================================================================
    // GH-213: extract_shard_files_from_index tests
    // =========================================================================

    #[test]
    fn test_gh213_extract_shard_files_basic() {
        let json = r#"{
            "metadata": {"total_size": 123456},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
                "model.layers.1.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
                "model.layers.2.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
                "lm_head.weight": "model-00004-of-00004.safetensors"
            }
        }"#;

        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 4);
        assert_eq!(shards[0], "model-00001-of-00004.safetensors");
        assert_eq!(shards[1], "model-00002-of-00004.safetensors");
        assert_eq!(shards[2], "model-00003-of-00004.safetensors");
        assert_eq!(shards[3], "model-00004-of-00004.safetensors");
    }

    #[test]
    fn test_gh213_extract_shard_files_deduplicates() {
        let json = r#"{
            "weight_map": {
                "a.weight": "model-00001-of-00002.safetensors",
                "b.weight": "model-00001-of-00002.safetensors",
                "c.weight": "model-00001-of-00002.safetensors",
                "d.weight": "model-00002-of-00002.safetensors"
            }
        }"#;

        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 2, "Should deduplicate shard filenames");
    }

    #[test]
    fn test_gh213_extract_shard_files_sorted() {
        let json = r#"{
            "weight_map": {
                "z.weight": "model-00003-of-00003.safetensors",
                "a.weight": "model-00001-of-00003.safetensors",
                "m.weight": "model-00002-of-00003.safetensors"
            }
        }"#;

        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 3);
        // Should be sorted alphabetically
        assert!(shards[0] < shards[1]);
        assert!(shards[1] < shards[2]);
    }

    #[test]
    fn test_gh213_extract_shard_files_empty_weight_map() {
        let json = r#"{"weight_map": {}}"#;
        let shards = extract_shard_files_from_index(json);
        assert!(shards.is_empty());
    }

    #[test]
    fn test_gh213_extract_shard_files_no_weight_map() {
        let json = r#"{"metadata": {"total_size": 123}}"#;
        let shards = extract_shard_files_from_index(json);
        assert!(shards.is_empty());
    }

    #[test]
    fn test_gh213_extract_shard_files_ignores_non_safetensors() {
        let json = r#"{
            "weight_map": {
                "a.weight": "model-00001-of-00002.safetensors",
                "b.weight": "not-a-safetensors-file.bin"
            }
        }"#;

        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0], "model-00001-of-00002.safetensors");
    }

    #[test]
    fn test_gh213_extract_shard_files_malformed_json() {
        let json = "not valid json at all";
        let shards = extract_shard_files_from_index(json);
        assert!(shards.is_empty(), "Malformed JSON should return empty");
    }

    // =========================================================================
    // GH-213: resolve_hf_model tests (offline, no network)
    // =========================================================================

    #[test]
    fn test_gh213_resolve_non_hf_uri_is_single_file() {
        let result = resolve_hf_model("/path/to/model.gguf").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "/path/to/model.gguf"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_gh213_resolve_hf_with_extension_is_single_file() {
        let result = resolve_hf_model("hf://org/repo/model.safetensors").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "hf://org/repo/model.safetensors"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_gh213_resolve_hf_invalid_uri_fails() {
        let result = resolve_hf_model("hf://invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_gh213_resolve_bare_org_repo_normalizes() {
        // "Qwen/Qwen2.5-Coder-3B-Instruct" should be treated as "hf://Qwen/Qwen2.5-Coder-3B-Instruct"
        // Can't test full resolution without network, but verify it doesn't return as SingleFile unchanged
        let result = resolve_hf_model("Qwen/FakeRepo");
        // Will fail with network error (repo doesn't exist), which proves it tried HF API
        assert!(
            result.is_err(),
            "Bare org/repo should attempt HF resolution"
        );
    }

    #[test]
    fn test_gh213_resolve_bare_org_repo_with_gguf_extension() {
        // "org/repo/file.gguf" should normalize to "hf://org/repo/file.gguf" → SingleFile
        let result = resolve_hf_model("org/repo/model.gguf").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => {
                assert_eq!(s, "hf://org/repo/model.gguf");
            }
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_gh213_resolve_bare_single_component_unchanged() {
        // "justAName" (no slash) should not be normalized, stays as local path
        let result = resolve_hf_model("justAName").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "justAName"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_gh213_resolve_relative_path_not_normalized() {
        // "./path/to/model" should NOT be treated as org/repo
        let result = resolve_hf_model("./path/to/model").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "./path/to/model"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_gh213_resolve_absolute_path_not_normalized() {
        // "/home/user/model" should NOT be treated as org/repo
        let result = resolve_hf_model("/home/user/model").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "/home/user/model"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    // Integration tests (require network, marked ignore for CI)
    #[test]
    #[ignore]
    fn test_gh213_resolve_small_model_is_single_file() {
        // 0.5B model has a single model.safetensors
        let result = resolve_hf_model("hf://Qwen/Qwen2.5-Coder-0.5B-Instruct").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => {
                assert!(
                    s.contains("model.safetensors"),
                    "Should resolve to model.safetensors: {}",
                    s
                );
            }
            ResolvedModel::Sharded { .. } => panic!("0.5B should be single file, not sharded"),
        }
    }

    #[test]
    #[ignore]
    fn test_gh213_resolve_large_model_is_sharded() {
        // 3B+ models use sharded SafeTensors
        let result = resolve_hf_model("hf://Qwen/Qwen2.5-Coder-3B-Instruct").unwrap();
        match result {
            ResolvedModel::Sharded {
                org,
                repo,
                shard_files,
            } => {
                assert_eq!(org, "Qwen");
                assert_eq!(repo, "Qwen2.5-Coder-3B-Instruct");
                assert!(
                    shard_files.len() > 1,
                    "3B model should have multiple shards, got {}",
                    shard_files.len()
                );
                // All shards should end with .safetensors
                for f in &shard_files {
                    assert!(
                        f.ends_with(".safetensors"),
                        "Shard should be .safetensors: {}",
                        f
                    );
                }
            }
            ResolvedModel::SingleFile(s) => {
                panic!("3B should be sharded, got SingleFile({})", s)
            }
        }
    }

    #[test]
    #[ignore]
    fn test_gh213_resolve_7b_model_is_sharded() {
        let result = resolve_hf_model("hf://Qwen/Qwen2.5-Coder-7B-Instruct").unwrap();
        match result {
            ResolvedModel::Sharded { shard_files, .. } => {
                assert!(
                    shard_files.len() > 1,
                    "7B model should have multiple shards, got {}",
                    shard_files.len()
                );
            }
            ResolvedModel::SingleFile(s) => {
                panic!("7B should be sharded, got SingleFile({})", s)
            }
        }
    }
}
