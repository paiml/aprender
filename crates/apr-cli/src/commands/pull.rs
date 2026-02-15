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

include!("pull_part_02.rs");
include!("pull_part_03.rs");
include!("pull_part_04.rs");
