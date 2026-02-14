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
use aprender::format::{
    apr_export, apr_import, ExportFormat, ExportOptions, ImportOptions, ValidationConfig,
};
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
#[derive(Debug)]
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
            // GH-211: Convert to APR + GGUF for full QA qualification
            convert_safetensors_formats(&result.path)?;
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
        // GH-211: Convert to APR + GGUF for full QA qualification
        convert_safetensors_formats(&result.path)?;
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
        download_or_verify_shard(
            &cache_dir,
            &base_url,
            shard_file,
            i,
            total_shards,
            force,
            existing_manifest.as_ref(),
            &mut file_checksums,
        )?;
    }

    download_companion_files(&cache_dir, &base_url, force)?;
    write_shard_manifest(&manifest_path, org, repo, file_checksums)?;

    println!();
    println!("{} Downloaded successfully", "✓".green());
    println!("  Path: {}", index_path.display().to_string().green());
    println!("  Shards: {}", total_shards.to_string().yellow());

    // GH-211: Convert sharded SafeTensors to APR + GGUF for full QA qualification
    convert_safetensors_formats(&index_path)?;

    println!();
    println!("{}", "Usage:".cyan().bold());
    println!("  apr run {}", index_path.display());
    println!("  apr serve {}", index_path.display());

    Ok(())
}

/// Download or verify a single shard file, updating the checksum map.
fn download_or_verify_shard(
    cache_dir: &Path,
    base_url: &str,
    shard_file: &str,
    index: usize,
    total: usize,
    force: bool,
    existing_manifest: Option<&ShardManifest>,
    checksums: &mut HashMap<String, FileChecksum>,
) -> Result<()> {
    let shard_path = cache_dir.join(shard_file);

    if !force && shard_path.exists() {
        if let Some(manifest) = existing_manifest {
            if let Some(expected) = manifest.files.get(shard_file) {
                let actual_size = std::fs::metadata(&shard_path).map(|m| m.len()).unwrap_or(0);
                if actual_size == expected.size {
                    checksums.insert(
                        shard_file.to_string(),
                        FileChecksum {
                            size: expected.size,
                            blake3: expected.blake3.clone(),
                        },
                    );
                    println!(
                        "  {} [{}/{}] {} (cached, verified)",
                        "✓".green(),
                        index + 1,
                        total,
                        shard_file
                    );
                    return Ok(());
                }
                println!(
                    "  {} [{}/{}] {} (size mismatch: {} vs {} bytes, re-downloading)",
                    "⚠".yellow(),
                    index + 1,
                    total,
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
                index + 1,
                total,
                shard_file
            );
            return Ok(());
        }
    }

    let shard_url = format!("{base_url}/{shard_file}");
    print!(
        "  {} [{}/{}] {}...",
        "↓".yellow(),
        index + 1,
        total,
        shard_file
    );
    io::stdout().flush().ok();

    let checksum = download_file_with_progress(&shard_url, &shard_path)?;
    checksums.insert(shard_file.to_string(), checksum);
    println!(" {}", "done".green());
    Ok(())
}

/// Download companion files (tokenizer.json, config.json, tokenizer_config.json) for sharded models.
fn download_companion_files(cache_dir: &Path, base_url: &str, force: bool) -> Result<()> {
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
    Ok(())
}

/// Write shard manifest with BLAKE3 checksums for integrity verification.
fn write_shard_manifest(
    manifest_path: &Path,
    org: &str,
    repo: &str,
    file_checksums: HashMap<String, FileChecksum>,
) -> Result<()> {
    if file_checksums.is_empty() {
        return Ok(());
    }
    let manifest = ShardManifest {
        version: 1,
        repo: format!("{org}/{repo}"),
        files: file_checksums,
    };
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to serialize manifest: {e}")))?;
    std::fs::write(manifest_path, manifest_json)?;
    println!("  {} .apr-manifest.json (integrity checksums)", "✓".green());
    Ok(())
}

/// List cached models
// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
pub fn list(json: bool) -> Result<()> {
    let fetcher = ModelFetcher::new().map_err(|e| {
        CliError::ValidationFailed(format!("Failed to initialize model fetcher: {e}"))
    })?;

    let models = fetcher.list();

    // GH-248: JSON output mode
    if json {
        let models_json: Vec<serde_json::Value> = models
            .iter()
            .map(|m| {
                serde_json::json!({
                    "name": m.name,
                    "size_bytes": m.size_bytes,
                    "format": format!("{:?}", m.format),
                    "path": m.path.display().to_string(),
                })
            })
            .collect();
        let stats = fetcher.stats();
        let output = serde_json::json!({
            "models": models_json,
            "total": models.len(),
            "total_size_bytes": stats.total_size_bytes,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&output).unwrap_or_default()
        );
        return Ok(());
    }

    println!("{}", "=== Cached Models ===".cyan().bold());
    println!();

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

    // =========================================================================
    // format_bytes: exhaustive boundary tests
    // =========================================================================

    #[test]
    fn test_format_bytes_boundary_just_below_kb() {
        assert_eq!(format_bytes(1023), "1023 B");
    }

    #[test]
    fn test_format_bytes_boundary_exact_kb() {
        assert_eq!(format_bytes(1024), "1.00 KB");
    }

    #[test]
    fn test_format_bytes_boundary_just_above_kb() {
        assert_eq!(format_bytes(1025), "1.00 KB");
    }

    #[test]
    fn test_format_bytes_boundary_just_below_mb() {
        // 1 MB - 1 byte = 1048575 bytes → KB range
        assert_eq!(format_bytes(1_048_575), "1024.00 KB");
    }

    #[test]
    fn test_format_bytes_boundary_exact_mb() {
        assert_eq!(format_bytes(1_048_576), "1.00 MB");
    }

    #[test]
    fn test_format_bytes_boundary_just_below_gb() {
        // 1 GB - 1 byte = 1073741823 bytes → MB range
        assert_eq!(format_bytes(1_073_741_823), "1024.00 MB");
    }

    #[test]
    fn test_format_bytes_boundary_exact_gb() {
        assert_eq!(format_bytes(1_073_741_824), "1.00 GB");
    }

    #[test]
    fn test_format_bytes_large_gb() {
        // 100 GB
        assert_eq!(format_bytes(107_374_182_400), "100.00 GB");
    }

    #[test]
    fn test_format_bytes_u64_max() {
        // u64::MAX should not panic, gives some large GB value
        let result = format_bytes(u64::MAX);
        assert!(
            result.contains("GB"),
            "u64::MAX should be in GB range: {}",
            result
        );
    }

    #[test]
    fn test_format_bytes_fractional_kb() {
        // 1.5 KB = 1536 bytes
        assert_eq!(format_bytes(1536), "1.50 KB");
    }

    #[test]
    fn test_format_bytes_7b_model_size() {
        // ~4.1 GB typical for 7B Q4_K_M
        assert_eq!(format_bytes(4_402_341_888), "4.10 GB");
    }

    // =========================================================================
    // extract_hf_repo: comprehensive edge cases
    // =========================================================================

    #[test]
    fn test_extract_hf_repo_just_prefix() {
        // "hf://" with nothing after → parts = [""], len < 2
        assert_eq!(extract_hf_repo("hf://"), None);
    }

    #[test]
    fn test_extract_hf_repo_single_slash_after_prefix() {
        // "hf://org/" → parts = ["org", ""], but parts[1] is empty
        // Still returns Some because len >= 2 and format just joins
        let result = extract_hf_repo("hf://org/");
        assert_eq!(result, Some("org/".to_string()));
    }

    #[test]
    fn test_extract_hf_repo_with_multiple_nested_paths() {
        // Deep nesting: only org/repo extracted
        let uri = "hf://org/repo/subdir1/subdir2/model.safetensors";
        assert_eq!(extract_hf_repo(uri), Some("org/repo".to_string()));
    }

    #[test]
    fn test_extract_hf_repo_wrong_scheme() {
        assert_eq!(extract_hf_repo("https://huggingface.co/org/repo"), None);
    }

    #[test]
    fn test_extract_hf_repo_no_scheme() {
        assert_eq!(extract_hf_repo("org/repo/model.gguf"), None);
    }

    #[test]
    fn test_extract_hf_repo_hf_prefix_case_sensitive() {
        // "HF://" (uppercase) should not match
        assert_eq!(extract_hf_repo("HF://org/repo"), None);
    }

    #[test]
    fn test_extract_hf_repo_special_chars_in_name() {
        let uri = "hf://TheBloke/Llama-2-7B-GGUF/model.gguf";
        assert_eq!(
            extract_hf_repo(uri),
            Some("TheBloke/Llama-2-7B-GGUF".to_string())
        );
    }

    #[test]
    fn test_extract_hf_repo_dots_in_name() {
        let uri = "hf://org/model.name.v2/file.safetensors";
        assert_eq!(extract_hf_repo(uri), Some("org/model.name.v2".to_string()));
    }

    // =========================================================================
    // extract_shard_files_from_index: comprehensive edge cases
    // =========================================================================

    #[test]
    fn test_extract_shard_files_single_shard() {
        let json = r#"{"weight_map": {"a.weight": "model-00001-of-00001.safetensors"}}"#;
        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0], "model-00001-of-00001.safetensors");
    }

    #[test]
    fn test_extract_shard_files_many_shards() {
        // 6 shards with heavy deduplication
        let json = r#"{
            "weight_map": {
                "a": "model-00001-of-00006.safetensors",
                "b": "model-00001-of-00006.safetensors",
                "c": "model-00002-of-00006.safetensors",
                "d": "model-00003-of-00006.safetensors",
                "e": "model-00004-of-00006.safetensors",
                "f": "model-00005-of-00006.safetensors",
                "g": "model-00005-of-00006.safetensors",
                "h": "model-00006-of-00006.safetensors"
            }
        }"#;
        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 6);
        assert_eq!(shards[0], "model-00001-of-00006.safetensors");
        assert_eq!(shards[5], "model-00006-of-00006.safetensors");
    }

    #[test]
    fn test_extract_shard_files_empty_string() {
        let shards = extract_shard_files_from_index("");
        assert!(shards.is_empty());
    }

    #[test]
    fn test_extract_shard_files_no_weight_map_key() {
        let json = r#"{"other_key": {"a": "file.safetensors"}}"#;
        let shards = extract_shard_files_from_index(json);
        assert!(shards.is_empty());
    }

    #[test]
    fn test_extract_shard_files_weight_map_not_object() {
        // weight_map is a string, not object — should not crash
        let json = r#"{"weight_map": "not an object"}"#;
        let shards = extract_shard_files_from_index(json);
        assert!(shards.is_empty());
    }

    #[test]
    fn test_extract_shard_files_mixed_extensions() {
        // Only .safetensors files should be included
        let json = r#"{
            "weight_map": {
                "a": "model-00001.safetensors",
                "b": "model-00002.bin",
                "c": "model-00003.pt",
                "d": "model-00004.safetensors"
            }
        }"#;
        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 2);
        assert!(shards.contains(&"model-00001.safetensors".to_string()));
        assert!(shards.contains(&"model-00004.safetensors".to_string()));
    }

    #[test]
    fn test_extract_shard_files_nested_braces() {
        // JSON with nested braces in metadata before weight_map
        let json = r#"{
            "metadata": {"nested": {"deep": "value"}},
            "weight_map": {
                "a.weight": "shard-001.safetensors",
                "b.weight": "shard-002.safetensors"
            }
        }"#;
        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 2);
    }

    #[test]
    fn test_extract_shard_files_whitespace_in_values() {
        // Extra whitespace and newlines around values
        let json = r#"{
            "weight_map": {
                "a.weight":   "  model-00001.safetensors  "  ,
                "b.weight":
                    "model-00002.safetensors"
            }
        }"#;
        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 2);
    }

    #[test]
    fn test_extract_shard_files_values_with_path_separators() {
        // Filenames shouldn't have path separators, but test robustness
        let json = r#"{"weight_map": {"a": "subdir/model.safetensors"}}"#;
        let shards = extract_shard_files_from_index(json);
        // Contains "/" so not matching simple pattern, but the function does string trim
        // It checks ends_with(".safetensors")
        assert_eq!(shards.len(), 1);
    }

    #[test]
    fn test_extract_shard_files_real_qwen_format() {
        // Realistic index.json fragment from Qwen2.5-Coder-3B-Instruct
        let json = r#"{
  "metadata": {
    "total_size": 6534782976
  },
  "weight_map": {
    "lm_head.weight": "model-00002-of-00002.safetensors",
    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.input_layernorm.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.mlp.down_proj.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.mlp.gate_proj.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.mlp.up_proj.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.post_attention_layernorm.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.self_attn.o_proj.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.self_attn.v_proj.weight": "model-00001-of-00002.safetensors",
    "model.layers.35.self_attn.v_proj.weight": "model-00002-of-00002.safetensors",
    "model.norm.weight": "model-00002-of-00002.safetensors"
  }
}"#;
        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0], "model-00001-of-00002.safetensors");
        assert_eq!(shards[1], "model-00002-of-00002.safetensors");
    }

    // =========================================================================
    // resolve_hf_model: offline URI normalization & extension detection
    // =========================================================================

    #[test]
    fn test_resolve_hf_model_with_apr_extension() {
        let result = resolve_hf_model("hf://org/repo/model.apr").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "hf://org/repo/model.apr"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile for .apr"),
        }
    }

    #[test]
    fn test_resolve_hf_model_with_pt_extension() {
        let result = resolve_hf_model("hf://org/repo/model.pt").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "hf://org/repo/model.pt"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile for .pt"),
        }
    }

    #[test]
    fn test_resolve_hf_model_case_insensitive_safetensors() {
        let result = resolve_hf_model("hf://org/repo/model.SafeTensors").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "hf://org/repo/model.SafeTensors"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_with_mixed_case_apr() {
        let result = resolve_hf_model("hf://org/repo/model.APR").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "hf://org/repo/model.APR"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile for .APR"),
        }
    }

    #[test]
    fn test_resolve_hf_model_bare_org_repo_with_safetensors() {
        // "org/repo/model.safetensors" → "hf://org/repo/model.safetensors" → SingleFile
        let result = resolve_hf_model("org/repo/model.safetensors").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "hf://org/repo/model.safetensors"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_bare_org_repo_with_apr() {
        let result = resolve_hf_model("org/repo/model.apr").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "hf://org/repo/model.apr"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_relative_path_with_dots() {
        // ../path should NOT be normalized to hf://
        let result = resolve_hf_model("../models/test.gguf").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "../models/test.gguf"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_http_url() {
        let result = resolve_hf_model("http://example.com/model.gguf").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "http://example.com/model.gguf"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_ftp_url() {
        let result = resolve_hf_model("ftp://example.com/model.gguf").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "ftp://example.com/model.gguf"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_empty_org_fails() {
        // "hf:///repo" → parts = ["", "repo"] → len >= 2 but first is empty
        // The function proceeds with empty org, which goes to API call → fails
        let result = resolve_hf_model("hf:///repo");
        // This triggers a network call with empty org, which will fail
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_hf_model_hf_with_single_part_fails() {
        let result = resolve_hf_model("hf://onlyorg");
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("Invalid HuggingFace URI"));
            }
            Err(other) => panic!("Expected ValidationFailed, got: {}", other),
            Ok(_) => panic!("Expected error, got Ok"),
        }
    }

    #[test]
    fn test_resolve_hf_model_bare_empty_parts() {
        // "/" alone → parts = ["", ""], both empty → does NOT normalize
        let result = resolve_hf_model("/").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "/"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_bare_single_slash() {
        // "a/" → parts = ["a", ""], parts[1].is_empty() → no normalization
        let result = resolve_hf_model("a/").unwrap();
        // parts[1] is empty, so bare org/repo normalization skipped
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "a/"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    // =========================================================================
    // ShardManifest serialization/deserialization tests
    // =========================================================================

    #[test]
    fn test_shard_manifest_serialize_deserialize() {
        let mut files = HashMap::new();
        files.insert(
            "model-00001-of-00002.safetensors".to_string(),
            FileChecksum {
                size: 5_000_000_000,
                blake3: "abc123def456".to_string(),
            },
        );
        files.insert(
            "model-00002-of-00002.safetensors".to_string(),
            FileChecksum {
                size: 3_000_000_000,
                blake3: "789xyz000111".to_string(),
            },
        );

        let manifest = ShardManifest {
            version: 1,
            repo: "Qwen/Qwen2.5-Coder-3B-Instruct".to_string(),
            files,
        };

        let json = serde_json::to_string_pretty(&manifest).expect("serialize");
        let deserialized: ShardManifest = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.version, 1);
        assert_eq!(deserialized.repo, "Qwen/Qwen2.5-Coder-3B-Instruct");
        assert_eq!(deserialized.files.len(), 2);

        let shard1 = deserialized
            .files
            .get("model-00001-of-00002.safetensors")
            .expect("shard1");
        assert_eq!(shard1.size, 5_000_000_000);
        assert_eq!(shard1.blake3, "abc123def456");

        let shard2 = deserialized
            .files
            .get("model-00002-of-00002.safetensors")
            .expect("shard2");
        assert_eq!(shard2.size, 3_000_000_000);
        assert_eq!(shard2.blake3, "789xyz000111");
    }

    #[test]
    fn test_shard_manifest_empty_files() {
        let manifest = ShardManifest {
            version: 1,
            repo: "org/repo".to_string(),
            files: HashMap::new(),
        };

        let json = serde_json::to_string(&manifest).expect("serialize");
        let deserialized: ShardManifest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.version, 1);
        assert!(deserialized.files.is_empty());
    }

    #[test]
    fn test_file_checksum_serialize_deserialize() {
        let checksum = FileChecksum {
            size: 1_234_567_890,
            blake3: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
        };

        let json = serde_json::to_string(&checksum).expect("serialize");
        let deserialized: FileChecksum = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.size, 1_234_567_890);
        assert_eq!(
            deserialized.blake3,
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        );
    }

    #[test]
    fn test_shard_manifest_version_zero() {
        let manifest = ShardManifest {
            version: 0,
            repo: "test/repo".to_string(),
            files: HashMap::new(),
        };
        let json = serde_json::to_string(&manifest).expect("serialize");
        assert!(json.contains("\"version\":0"));
    }

    #[test]
    fn test_shard_manifest_large_version() {
        let manifest = ShardManifest {
            version: u32::MAX,
            repo: "test/repo".to_string(),
            files: HashMap::new(),
        };
        let json = serde_json::to_string(&manifest).expect("serialize");
        let deserialized: ShardManifest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.version, u32::MAX);
    }

    #[test]
    fn test_file_checksum_zero_size() {
        let checksum = FileChecksum {
            size: 0,
            blake3: "empty".to_string(),
        };
        let json = serde_json::to_string(&checksum).expect("serialize");
        let deserialized: FileChecksum = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.size, 0);
    }

    #[test]
    fn test_file_checksum_max_u64_size() {
        let checksum = FileChecksum {
            size: u64::MAX,
            blake3: "huge".to_string(),
        };
        let json = serde_json::to_string(&checksum).expect("serialize");
        let deserialized: FileChecksum = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.size, u64::MAX);
    }

    // =========================================================================
    // resolve_hf_uri: backward-compat wrapper edge cases
    // =========================================================================

    #[test]
    fn test_resolve_hf_uri_with_apr_extension() {
        let uri = "hf://org/repo/model.apr";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri);
    }

    #[test]
    fn test_resolve_hf_uri_with_pt_extension() {
        let uri = "hf://org/repo/model.pt";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri);
    }

    #[test]
    fn test_resolve_hf_uri_bare_org_repo_gguf() {
        // "org/repo/file.gguf" → normalizes to "hf://org/repo/file.gguf"
        let resolved = resolve_hf_uri("org/repo/file.gguf").unwrap();
        assert_eq!(resolved, "hf://org/repo/file.gguf");
    }

    #[test]
    fn test_resolve_hf_uri_dot_relative_path() {
        let uri = "./some/dir/model.gguf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "dot-relative path should not be normalized");
    }

    #[test]
    fn test_resolve_hf_uri_dot_dot_relative_path() {
        let uri = "../parent/model.gguf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(
            resolved, uri,
            "parent-relative path should not be normalized"
        );
    }

    #[test]
    fn test_resolve_hf_uri_just_a_word() {
        // Single word with no slashes: not normalized, returned as SingleFile
        let resolved = resolve_hf_uri("model").unwrap();
        assert_eq!(resolved, "model");
    }

    // =========================================================================
    // fetch_safetensors_companions: path edge cases (offline)
    // =========================================================================

    #[test]
    fn test_fetch_companions_empty_uri() {
        let temp_dir = std::env::temp_dir().join("apr_companion_empty_uri");
        let _ = std::fs::create_dir_all(&temp_dir);
        let model_path = temp_dir.join("hash123.safetensors");
        let _ = std::fs::write(&model_path, b"dummy");

        // Empty URI → not hf:// → noop
        let result = fetch_safetensors_companions(&model_path, "");
        assert!(result.is_ok());
        assert!(!temp_dir.join("hash123.tokenizer.json").exists());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_fetch_companions_model_stem_extraction() {
        let temp_dir = std::env::temp_dir().join("apr_companion_stem");
        let _ = std::fs::create_dir_all(&temp_dir);

        // Model with complex hash stem
        let model_path = temp_dir.join("e910cab26ae116eb.converted.safetensors");
        let _ = std::fs::write(&model_path, b"dummy");

        // Pre-create companion files with the full stem (without .safetensors)
        // The stem is "e910cab26ae116eb.converted"
        let _ = std::fs::write(
            temp_dir.join("e910cab26ae116eb.converted.tokenizer.json"),
            b"{}",
        );
        let _ = std::fs::write(
            temp_dir.join("e910cab26ae116eb.converted.config.json"),
            b"{}",
        );

        // Should succeed — files already exist
        let result = fetch_safetensors_companions(&model_path, "hf://org/repo/model.safetensors");
        assert!(result.is_ok());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_fetch_companions_https_uri_noop() {
        let temp_dir = std::env::temp_dir().join("apr_companion_https");
        let _ = std::fs::create_dir_all(&temp_dir);
        let model_path = temp_dir.join("model.safetensors");
        let _ = std::fs::write(&model_path, b"dummy");

        // https:// URI — extract_hf_repo returns None → noop
        let result =
            fetch_safetensors_companions(&model_path, "https://example.com/model.safetensors");
        assert!(result.is_ok());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    // =========================================================================
    // resolve_hf_model: URI normalization edge cases
    // =========================================================================

    #[test]
    fn test_resolve_hf_model_double_slash_bare_path() {
        // "a//b" → parts = ["a", "", "b"], parts[1].is_empty() → no normalization
        let result = resolve_hf_model("a//b").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "a//b"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_with_query_string() {
        // URI with query params in extension — extension check uses Path
        // Path::extension sees "gguf?rev=main" which doesn't match .gguf
        // So it falls through to the HF API query path which fails with network error
        let result = resolve_hf_model("hf://org/repo/model.gguf?rev=main");
        // The result should be an error since the extension is not recognized
        // and the API query for "org/repo" will fail (doesn't exist)
        match result {
            Ok(ResolvedModel::SingleFile(_)) => {
                // Extension was detected somehow — acceptable
            }
            Err(CliError::NetworkError(_)) => {
                // Expected: API query failed since "org/repo" doesn't exist
            }
            Err(CliError::ValidationFailed(_)) => {
                // Also acceptable: no files found
            }
            other => panic!("Unexpected result: {:?}", other),
        }
    }

    #[test]
    fn test_resolve_hf_model_unicode_in_path() {
        // Unicode org/repo should be normalized
        // Will fail at API call since repo doesn't exist
        let result = resolve_hf_model("org-\u{00e9}/repo-\u{00fc}/model.gguf").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => {
                assert!(s.starts_with("hf://"), "Should be normalized: {}", s);
                assert!(s.ends_with("model.gguf"));
            }
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_spaces_in_path() {
        // Spaces in path — should not crash
        let result = resolve_hf_model("org name/repo name/model.gguf").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => {
                assert!(s.starts_with("hf://"));
                assert!(s.ends_with("model.gguf"));
            }
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    // =========================================================================
    // NEW: format_bytes additional edge cases
    // =========================================================================

    #[test]
    fn test_format_bytes_one_byte() {
        assert_eq!(format_bytes(1), "1 B");
    }

    #[test]
    fn test_format_bytes_exactly_two_kb() {
        assert_eq!(format_bytes(2 * 1024), "2.00 KB");
    }

    #[test]
    fn test_format_bytes_just_above_mb() {
        // 1 MB + 1 byte
        assert_eq!(format_bytes(1_048_577), "1.00 MB");
    }

    #[test]
    fn test_format_bytes_just_above_gb() {
        // 1 GB + 1 byte
        assert_eq!(format_bytes(1_073_741_825), "1.00 GB");
    }

    #[test]
    fn test_format_bytes_terabyte_range() {
        // 1 TB = 1024 GB
        let tb = 1024_u64 * 1024 * 1024 * 1024;
        let result = format_bytes(tb);
        assert_eq!(result, "1024.00 GB");
    }

    #[test]
    fn test_format_bytes_10_tb() {
        let ten_tb = 10 * 1024_u64 * 1024 * 1024 * 1024;
        let result = format_bytes(ten_tb);
        assert_eq!(result, "10240.00 GB");
    }

    #[test]
    fn test_format_bytes_exact_256_mb() {
        assert_eq!(format_bytes(256 * 1024 * 1024), "256.00 MB");
    }

    #[test]
    fn test_format_bytes_1b_model_size() {
        // ~600 MB typical for 1B Q4_K_M
        assert_eq!(format_bytes(629_145_600), "600.00 MB");
    }

    #[test]
    fn test_format_bytes_13b_model_size() {
        // ~7.4 GB typical for 13B Q4_K_M
        assert_eq!(format_bytes(7_945_689_498), "7.40 GB");
    }

    #[test]
    fn test_format_bytes_half_kb() {
        assert_eq!(format_bytes(512), "512 B");
    }

    // =========================================================================
    // NEW: extract_hf_repo additional edge cases
    // =========================================================================

    #[test]
    fn test_extract_hf_repo_trailing_slash_after_repo() {
        // "hf://org/repo/" → parts = ["org", "repo", ""], len >= 2 → Some("org/repo")
        assert_eq!(
            extract_hf_repo("hf://org/repo/"),
            Some("org/repo".to_string())
        );
    }

    #[test]
    fn test_extract_hf_repo_with_multiple_trailing_slashes() {
        let uri = "hf://org/repo///";
        assert_eq!(extract_hf_repo(uri), Some("org/repo".to_string()));
    }

    #[test]
    fn test_extract_hf_repo_just_hf_no_colon_slash() {
        // "hf" without "://" → strip_prefix fails → None
        assert_eq!(extract_hf_repo("hf"), None);
    }

    #[test]
    fn test_extract_hf_repo_hf_colon_no_slashes() {
        assert_eq!(extract_hf_repo("hf:org/repo"), None);
    }

    #[test]
    fn test_extract_hf_repo_hf_single_slash() {
        assert_eq!(extract_hf_repo("hf:/org/repo"), None);
    }

    #[test]
    fn test_extract_hf_repo_numeric_org_and_repo() {
        let uri = "hf://12345/67890/model.safetensors";
        assert_eq!(extract_hf_repo(uri), Some("12345/67890".to_string()));
    }

    #[test]
    fn test_extract_hf_repo_hyphenated_names() {
        let uri = "hf://my-org/my-awesome-model-v2/weights.safetensors";
        assert_eq!(
            extract_hf_repo(uri),
            Some("my-org/my-awesome-model-v2".to_string())
        );
    }

    #[test]
    fn test_extract_hf_repo_underscored_names() {
        let uri = "hf://my_org/my_model_v2";
        assert_eq!(extract_hf_repo(uri), Some("my_org/my_model_v2".to_string()));
    }

    #[test]
    fn test_extract_hf_repo_very_long_names() {
        let long_org = "a".repeat(100);
        let long_repo = "b".repeat(200);
        let uri = format!("hf://{}/{}/model.safetensors", long_org, long_repo);
        assert_eq!(
            extract_hf_repo(&uri),
            Some(format!("{}/{}", long_org, long_repo))
        );
    }

    #[test]
    fn test_extract_hf_repo_with_at_symbol() {
        // Some HF repos use @ for versions
        let uri = "hf://org/repo@main/model.safetensors";
        assert_eq!(extract_hf_repo(uri), Some("org/repo@main".to_string()));
    }

    #[test]
    fn test_extract_hf_repo_empty_after_prefix() {
        // "hf://" → path = "" → parts = [""], len < 2 → None
        assert_eq!(extract_hf_repo("hf://"), None);
    }

    #[test]
    fn test_extract_hf_repo_single_char_org_and_repo() {
        assert_eq!(extract_hf_repo("hf://a/b"), Some("a/b".to_string()));
    }

    // =========================================================================
    // NEW: extract_shard_files_from_index additional edge cases
    // =========================================================================

    #[test]
    fn test_extract_shard_files_truncated_json() {
        // JSON that's cut off mid-stream
        let json = r#"{"weight_map": {"a.weight": "model-00001.safe"#;
        let shards = extract_shard_files_from_index(json);
        assert!(
            shards.is_empty(),
            "Truncated JSON should produce no results"
        );
    }

    #[test]
    fn test_extract_shard_files_unicode_tensor_names() {
        let json = r#"{
            "weight_map": {
                "模型.层.0.权重": "model-00001-of-00001.safetensors"
            }
        }"#;
        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0], "model-00001-of-00001.safetensors");
    }

    #[test]
    fn test_extract_shard_files_colons_in_tensor_names() {
        // Tensor names with colons (like "model:layers:0:weight") use rfind(':')
        // so the last colon determines the split point
        let json = r#"{
            "weight_map": {
                "model:layers:0:weight": "shard-001.safetensors",
                "model:layers:1:weight": "shard-002.safetensors"
            }
        }"#;
        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 2);
    }

    #[test]
    fn test_extract_shard_files_empty_json_object() {
        let json = "{}";
        let shards = extract_shard_files_from_index(json);
        assert!(shards.is_empty());
    }

    #[test]
    fn test_extract_shard_files_null_json() {
        let json = "null";
        let shards = extract_shard_files_from_index(json);
        assert!(shards.is_empty());
    }

    #[test]
    fn test_extract_shard_files_array_instead_of_object() {
        let json = r#"[{"weight_map": {"a": "model.safetensors"}}]"#;
        // weight_map is inside an array element — the string search still finds it
        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 1);
    }

    #[test]
    fn test_extract_shard_files_weight_map_with_empty_value() {
        let json = r#"{"weight_map": {"a.weight": ""}}"#;
        let shards = extract_shard_files_from_index(json);
        assert!(shards.is_empty(), "Empty filename should be excluded");
    }

    #[test]
    fn test_extract_shard_files_weight_map_value_not_safetensors() {
        let json = r#"{"weight_map": {"a.weight": "model.gguf"}}"#;
        let shards = extract_shard_files_from_index(json);
        assert!(
            shards.is_empty(),
            "Non-safetensors files should be excluded"
        );
    }

    #[test]
    fn test_extract_shard_files_large_model_40_shards() {
        // Simulate a very large model with 40 shards
        let mut entries = Vec::new();
        for i in 1..=200 {
            let shard_num = (i % 40) + 1;
            entries.push(format!(
                "\"tensor_{}\": \"model-{:05}-of-00040.safetensors\"",
                i, shard_num
            ));
        }
        let json = format!("{{\"weight_map\": {{{}}}}}", entries.join(",\n"));
        let shards = extract_shard_files_from_index(&json);
        assert_eq!(shards.len(), 40);
        assert_eq!(shards[0], "model-00001-of-00040.safetensors");
        assert_eq!(shards[39], "model-00040-of-00040.safetensors");
    }

    #[test]
    fn test_extract_shard_files_weight_map_appears_in_metadata() {
        // "weight_map" string appears in metadata as well — should find the right one
        let json = r#"{
            "metadata": {"description": "This model has a weight_map section"},
            "weight_map": {
                "a.weight": "actual-shard.safetensors"
            }
        }"#;
        let shards = extract_shard_files_from_index(json);
        // The first occurrence of "weight_map" is in metadata description (as string content),
        // but the parser looks for `"weight_map"` and then the next `{`
        // The first `"weight_map"` found is inside the metadata string value, and the next `{` after it
        // would be the actual weight_map object. This is a known edge case of string-based parsing.
        // The result depends on the exact JSON layout.
        assert!(
            !shards.is_empty(),
            "Should find shards from the actual weight_map"
        );
    }

    #[test]
    fn test_extract_shard_files_sorted_alphanumerically() {
        let json = r#"{
            "weight_map": {
                "z": "shard-c.safetensors",
                "y": "shard-a.safetensors",
                "x": "shard-b.safetensors"
            }
        }"#;
        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 3);
        assert_eq!(shards[0], "shard-a.safetensors");
        assert_eq!(shards[1], "shard-b.safetensors");
        assert_eq!(shards[2], "shard-c.safetensors");
    }

    #[test]
    fn test_extract_shard_files_weight_map_no_opening_brace() {
        // "weight_map" key exists but is followed by a string, not object
        let json = r#"{"weight_map": "just a string, no object here"}"#;
        let shards = extract_shard_files_from_index(json);
        assert!(shards.is_empty());
    }

    #[test]
    fn test_extract_shard_files_multiple_weight_map_keys() {
        // Technically invalid JSON (duplicate keys), but tests parser resilience
        // The string search finds the first "weight_map"
        let json = r#"{
            "weight_map": {"a": "first.safetensors"},
            "weight_map": {"b": "second.safetensors"}
        }"#;
        let shards = extract_shard_files_from_index(json);
        // First weight_map is found; second is ignored
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0], "first.safetensors");
    }

    // =========================================================================
    // NEW: ResolvedModel enum tests
    // =========================================================================

    #[test]
    fn test_resolved_model_single_file_debug() {
        let model = ResolvedModel::SingleFile("test.gguf".to_string());
        let debug = format!("{:?}", model);
        assert!(debug.contains("SingleFile"));
        assert!(debug.contains("test.gguf"));
    }

    #[test]
    fn test_resolved_model_sharded_debug() {
        let model = ResolvedModel::Sharded {
            org: "Qwen".to_string(),
            repo: "Qwen2.5-Coder-3B".to_string(),
            shard_files: vec!["shard-001.safetensors".to_string()],
        };
        let debug = format!("{:?}", model);
        assert!(debug.contains("Sharded"));
        assert!(debug.contains("Qwen"));
        assert!(debug.contains("Qwen2.5-Coder-3B"));
        assert!(debug.contains("shard-001.safetensors"));
    }

    #[test]
    fn test_resolved_model_sharded_empty_shard_files() {
        let model = ResolvedModel::Sharded {
            org: "org".to_string(),
            repo: "repo".to_string(),
            shard_files: vec![],
        };
        match model {
            ResolvedModel::Sharded { shard_files, .. } => {
                assert!(shard_files.is_empty());
            }
            _ => panic!("Expected Sharded"),
        }
    }

    // =========================================================================
    // NEW: ShardManifest/FileChecksum additional tests
    // =========================================================================

    #[test]
    fn test_shard_manifest_deserialize_unknown_fields() {
        // Forward compatibility: extra fields should be ignored (serde default)
        let json = r#"{"version": 2, "repo": "org/repo", "files": {}, "extra_field": "ignored"}"#;
        let manifest: ShardManifest =
            serde_json::from_str(json).expect("unknown fields should be ignored by default");
        assert_eq!(manifest.version, 2);
    }

    #[test]
    fn test_shard_manifest_missing_required_field() {
        // Missing "repo" field should fail deserialization
        let json = r#"{"version": 1, "files": {}}"#;
        assert!(serde_json::from_str::<ShardManifest>(json).is_err());
    }

    #[test]
    fn test_file_checksum_missing_blake3_field() {
        // Missing "blake3" should fail
        let json = r#"{"size": 100}"#;
        assert!(serde_json::from_str::<FileChecksum>(json).is_err());
    }

    #[test]
    fn test_file_checksum_missing_size_field() {
        // Missing "size" should fail
        let json = r#"{"blake3": "abc123"}"#;
        assert!(serde_json::from_str::<FileChecksum>(json).is_err());
    }

    #[test]
    fn test_shard_manifest_special_chars_in_repo() {
        let manifest = ShardManifest {
            version: 1,
            repo: "org/repo-with.dots_and-dashes".to_string(),
            files: HashMap::new(),
        };
        let json = serde_json::to_string(&manifest).expect("serialize");
        let deserialized: ShardManifest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.repo, "org/repo-with.dots_and-dashes");
    }

    #[test]
    fn test_shard_manifest_many_files() {
        let mut files = HashMap::new();
        for i in 0..100 {
            files.insert(
                format!("model-{:05}-of-00100.safetensors", i + 1),
                FileChecksum {
                    size: 5_000_000_000 + i as u64,
                    blake3: format!("hash_{:05}", i),
                },
            );
        }
        let manifest = ShardManifest {
            version: 1,
            repo: "big/model".to_string(),
            files,
        };
        let json = serde_json::to_string(&manifest).expect("serialize");
        let deserialized: ShardManifest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.files.len(), 100);
    }

    #[test]
    fn test_file_checksum_unicode_blake3() {
        // blake3 field is a string — should handle any string content
        let checksum = FileChecksum {
            size: 42,
            blake3: "hash_with_\u{00e9}moji".to_string(),
        };
        let json = serde_json::to_string(&checksum).expect("serialize");
        let deserialized: FileChecksum = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.blake3, "hash_with_\u{00e9}moji");
    }

    // =========================================================================
    // NEW: resolve_hf_model additional URI edge cases
    // =========================================================================

    #[test]
    fn test_resolve_hf_model_bare_org_repo_with_pt_extension() {
        // "org/repo/model.pt" → normalizes to "hf://org/repo/model.pt" → SingleFile
        let result = resolve_hf_model("org/repo/model.pt").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "hf://org/repo/model.pt"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_file_url_scheme() {
        // "file:///path/to/model" has "://" so NOT normalized
        let result = resolve_hf_model("file:///path/to/model.gguf").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "file:///path/to/model.gguf"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_s3_url_scheme() {
        // "s3://bucket/key" has "://" → not normalized, not hf:// → SingleFile
        let result = resolve_hf_model("s3://bucket/model.gguf").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "s3://bucket/model.gguf"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_just_dot() {
        // "." starts with '.' → NOT normalized
        let result = resolve_hf_model(".").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "."),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_empty_string() {
        let result = resolve_hf_model("").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, ""),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_gguf_mixed_case_extension() {
        let result = resolve_hf_model("hf://org/repo/model.GGuF").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "hf://org/repo/model.GGuF"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile for .GGuF"),
        }
    }

    #[test]
    fn test_resolve_hf_model_pt_mixed_case() {
        let result = resolve_hf_model("hf://org/repo/model.PT").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "hf://org/repo/model.PT"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    // =========================================================================
    // NEW: resolve_hf_uri backward-compat wrapper additional tests
    // =========================================================================

    #[test]
    fn test_resolve_hf_uri_single_file_with_safetensors() {
        // .safetensors has a known extension → SingleFile passthrough
        let uri = "hf://org/repo/model.safetensors";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri);
    }

    #[test]
    fn test_resolve_hf_uri_with_file_scheme() {
        let uri = "file:///home/user/model.gguf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri);
    }

    #[test]
    fn test_resolve_hf_uri_s3_scheme() {
        let uri = "s3://bucket/key/model.safetensors";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri);
    }

    // =========================================================================
    // NEW: fetch_safetensors_companions additional offline tests
    // =========================================================================

    #[test]
    fn test_fetch_companions_hf_single_slash_uri_noop() {
        // "hf:/onlyorg" → extract_hf_repo returns None → noop
        let temp_dir = std::env::temp_dir().join("apr_companion_hf_single_slash");
        let _ = std::fs::create_dir_all(&temp_dir);
        let model_path = temp_dir.join("model.safetensors");
        let _ = std::fs::write(&model_path, b"dummy");

        let result = fetch_safetensors_companions(&model_path, "hf:/onlyorg");
        assert!(result.is_ok());
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_fetch_companions_uppercase_hf_noop() {
        // "HF://org/repo" — extract_hf_repo uses strip_prefix("hf://") which is case-sensitive
        let temp_dir = std::env::temp_dir().join("apr_companion_uppercase_hf");
        let _ = std::fs::create_dir_all(&temp_dir);
        let model_path = temp_dir.join("model.safetensors");
        let _ = std::fs::write(&model_path, b"dummy");

        let result = fetch_safetensors_companions(&model_path, "HF://Org/Repo");
        assert!(result.is_ok());
        // No companions created (not a valid hf:// URI)
        assert!(!temp_dir.join("model.tokenizer.json").exists());
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_fetch_companions_model_without_extension() {
        // Model file with no extension → file_stem returns the whole name
        let temp_dir = std::env::temp_dir().join("apr_companion_no_ext");
        let _ = std::fs::create_dir_all(&temp_dir);
        let model_path = temp_dir.join("model_no_ext");
        let _ = std::fs::write(&model_path, b"dummy");

        // Pre-create companion files with the stem prefix
        let _ = std::fs::write(temp_dir.join("model_no_ext.tokenizer.json"), b"{}");
        let _ = std::fs::write(temp_dir.join("model_no_ext.config.json"), b"{}");

        let result = fetch_safetensors_companions(&model_path, "hf://org/repo/model.safetensors");
        assert!(result.is_ok());
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
