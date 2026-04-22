//! Pull command: download and cache models from HuggingFace (`~/.cache/pacha/models/`).

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
#[provable_contracts_macros::contract(
    "apr-cli-operations-v1",
    equation = "mutating_output_contract"
)]
pub fn run(
    model_ref: &str,
    force: bool,
    dry_run: bool,
    revision: Option<&str>,
    offline: bool,
) -> Result<()> {
    contract_pre_pull_cache_integrity!();
    println!("{}", "=== APR Pull ===".cyan().bold());
    println!();

    // CRUX-A-01 FALSIFY-CRUX-A-01-001: --dry-run resolves short name to
    // canonical URL and exits with zero network I/O.
    // CRUX-A-03 ALGO-001..003: --dry-run echoes the revision spec the user
    // supplied (or the default "main") and validates its local form.
    // CRUX-A-20 ALGO-001..005: --dry-run also echoes the resolved offline
    // mode so callers can assert CLI-flag / env-var equivalence offline.
    if dry_run {
        return run_dry_run(model_ref, revision, offline);
    }

    // GH-213: Resolve HuggingFace URI — detect single vs sharded models
    let resolved = resolve_hf_model(model_ref)?;

    let result = match resolved {
        ResolvedModel::SingleFile(ref uri) => run_single_file(uri, force),
        ResolvedModel::Sharded {
            ref org,
            ref repo,
            ref shard_files,
        } => run_sharded(org, repo, shard_files, force),
    };
    if let Ok(ref r) = result {
        contract_post_pull_cache_integrity!(r);
    }
    result
}

/// Pull a single-file model.
///
/// GH-352: For HuggingFace URIs, streams directly to disk instead of buffering
/// the entire file in memory through pacha's resolver. For non-HF URIs (pacha
/// aliases), falls back to the pacha fetcher.
fn run_single_file(model_ref: &str, force: bool) -> Result<()> {
    println!("Model: {}", model_ref.cyan());

    // GH-352: HuggingFace URIs bypass pacha to avoid O(model_size) memory buffering
    if model_ref.starts_with("hf://") {
        return run_single_file_streaming(model_ref, force);
    }

    let mut fetcher = ModelFetcher::with_config(FetchConfig::default()).map_err(|e| {
        CliError::ValidationFailed(format!("Failed to initialize model fetcher: {e}"))
    })?;

    if !force && fetcher.is_cached(model_ref) {
        return handle_cached_model(&mut fetcher, model_ref);
    }

    let result = download_single_model(&mut fetcher, model_ref)?;
    ensure_safetensors_companions(&result)?;
    print_pull_usage(&result.path, true);
    Ok(())
}

/// GH-352: Stream a single HuggingFace file directly to disk.
///
/// Uses O(64KB) memory instead of O(model_size). The pacha fetcher's
/// `resolver.resolve()` buffers the entire response via `response.bytes()`,
/// which consumed ~4.5 GB for a 7B GGUF. This function streams with a 64KB
/// chunked read, computes BLAKE3 incrementally, and saves to the pacha cache.
fn run_single_file_streaming(model_ref: &str, force: bool) -> Result<()> {
    let (org, repo, filename) = parse_hf_single_uri(model_ref)?;
    let url = format!("https://huggingface.co/{org}/{repo}/resolve/main/{filename}");

    let cache_dir = get_pacha_cache_dir()?;
    std::fs::create_dir_all(&cache_dir)?;
    let (extension, cache_path) = build_single_cache_path(&cache_dir, model_ref, &filename);

    if !force && cache_path.exists() {
        return report_cached_single(&cache_path);
    }

    stream_and_post_process(&url, &cache_path, model_ref, &extension)?;
    print_pull_usage(&cache_path, true);
    Ok(())
}

fn stream_and_post_process(
    url: &str,
    cache_path: &std::path::Path,
    model_ref: &str,
    extension: &str,
) -> Result<()> {
    println!();
    println!("{}", "Downloading (streaming)...".yellow());
    let checksum = download_file_with_progress(url, cache_path)?;
    report_downloaded_single(cache_path, &checksum);

    if extension == "safetensors" {
        fetch_safetensors_companions(cache_path, model_ref)?;
        convert_safetensors_formats(cache_path)?;
    }
    Ok(())
}

fn parse_hf_single_uri(model_ref: &str) -> Result<(String, String, String)> {
    let path = model_ref.strip_prefix("hf://").unwrap_or(model_ref);
    let parts: Vec<&str> = path.split('/').collect();
    if parts.len() < 3 {
        return Err(CliError::ValidationFailed(format!(
            "HuggingFace URI must include a filename: {model_ref}"
        )));
    }
    Ok((
        parts[0].to_string(),
        parts[1].to_string(),
        parts[2..].join("/"),
    ))
}

fn build_single_cache_path(
    cache_dir: &std::path::Path,
    model_ref: &str,
    filename: &str,
) -> (String, std::path::PathBuf) {
    let uri_hash = blake3::hash(model_ref.as_bytes()).to_hex().to_string();
    let extension = std::path::Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("bin")
        .to_string();
    let cache_filename = format!("{}.{extension}", &uri_hash[..16]);
    let cache_path = cache_dir.join(&cache_filename);
    (extension, cache_path)
}

fn report_cached_single(cache_path: &std::path::Path) -> Result<()> {
    let metadata = std::fs::metadata(cache_path)?;
    println!("{} Model already cached", "✓".green());
    println!("  Path: {}", cache_path.display());
    println!("  Size: {}", format_bytes(metadata.len()));
    print_pull_usage(cache_path, true);
    Ok(())
}

fn report_downloaded_single(cache_path: &std::path::Path, checksum: &FileChecksum) {
    println!();
    println!("{} Downloaded successfully", "✓".green());
    println!("  Path: {}", cache_path.display().to_string().green());
    println!("  Size: {}", format_bytes(checksum.size).yellow());
    println!("  Hash: {}", &checksum.blake3[..16]);
}

/// Get the pacha model cache directory.
fn get_pacha_cache_dir() -> Result<std::path::PathBuf> {
    if let Ok(cache_home) = std::env::var("XDG_CACHE_HOME") {
        return Ok(std::path::PathBuf::from(cache_home)
            .join("pacha")
            .join("models"));
    }
    Ok(dirs::home_dir()
        .ok_or_else(|| CliError::ValidationFailed("Cannot find home directory".to_string()))?
        .join(".cache")
        .join("pacha")
        .join("models"))
}

/// Handle a model that is already cached in pacha.
fn handle_cached_model(fetcher: &mut ModelFetcher, model_ref: &str) -> Result<()> {
    println!("{} Model already cached", "✓".green());
    let result = fetcher
        .pull_quiet(model_ref)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to get cached model: {e}")))?;

    println!("  Path: {}", result.path.display());
    println!("  Size: {}", result.size_human());
    println!("  Format: {}", result.format.name());

    ensure_safetensors_companions(&result)?;
    print_pull_usage(&result.path, false);
    Ok(())
}

/// Download a single model with progress bar.
fn download_single_model(
    fetcher: &mut ModelFetcher,
    model_ref: &str,
) -> Result<pacha::fetcher::FetchResult> {
    println!();
    println!("{}", "Downloading...".yellow());

    let result = fetcher
        .pull(model_ref, |progress| {
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
    println!("  Format: {}", result.format.name());
    println!("  Hash: {}", &result.hash[..16]);
    Ok(result)
}

/// Ensure companion files exist for SafeTensors models (GH-198, GH-211).
fn ensure_safetensors_companions(result: &pacha::fetcher::FetchResult) -> Result<()> {
    if matches!(result.format, ModelFormat::SafeTensors(_)) {
        fetch_safetensors_companions(&result.path, &result.resolved_uri)?;
        convert_safetensors_formats(&result.path)?;
    }
    Ok(())
}

/// Print usage instructions after a successful pull.
fn print_pull_usage(path: &Path, show_serve: bool) {
    println!();
    println!("{}", "Usage:".cyan().bold());
    println!("  apr run {}", path.display());
    if show_serve {
        println!("  apr serve {}", path.display());
    }
}

include!("pull_sharded.rs");

/// CRUX-A-01 FALSIFY-CRUX-A-01-001: `--dry-run` resolver.
///
/// Emits the resolved canonical URL on stdout and returns `Ok(())` with zero
/// network I/O. Short names are resolved via the embedded alias map
/// (`configs/aliases.yaml`); scheme-qualified inputs (`hf://…`,
/// `https://…`) and bare `org/repo` inputs echo as their canonical forms.
///
/// CRUX-A-01 FALSIFY-CRUX-A-01-003: unknown short names (no scheme, no `/`)
/// return an error that includes a Levenshtein ≤ 2 "did you mean …" hint.
/// CRUX-A-03 ALGO-001..003: `--revision` is classified locally and echoed
/// in the dry-run output. Malformed revisions (empty, whitespace, URL)
/// fail fast without touching the network.
///
/// CRUX-A-20 ALGO-001..005: the effective offline signal (CLI flag OR
/// `APR_OFFLINE` OR `HF_HUB_OFFLINE` truthy) is echoed too.
fn run_dry_run(model_ref: &str, revision: Option<&str>, offline_flag: bool) -> Result<()> {
    use super::aliases;
    use super::offline;
    use super::revision as rev;

    let resolved = if let Some(url) = aliases::resolve_short_name(model_ref) {
        url
    } else if !model_ref.contains("://") && model_ref.contains('/') {
        format!("hf://{model_ref}")
    } else {
        return Err(unknown_short_name_error(model_ref));
    };

    let rev_spec = revision.unwrap_or(rev::DEFAULT_REVISION);
    let rev_kind = rev::classify_revision(rev_spec).map_err(|msg| {
        CliError::ValidationFailed(format!("CRUX-A-03: invalid --revision {rev_spec:?}: {msg}"))
    })?;

    // CRUX-A-20: resolve offline signal from CLI flag + env vars.
    let env = offline::read_offline_env();
    let env_borrowed: Vec<(&str, &str)> =
        env.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    let is_offline = offline::is_offline(offline_flag, env_borrowed.iter().copied());

    println!("Model:    {}", model_ref.cyan());
    println!("Resolved: {}", resolved.green());
    println!("Revision: {} ({:?})", rev_spec.green(), rev_kind);
    println!(
        "Offline:  {}",
        if is_offline {
            "true".green()
        } else {
            "false".yellow()
        }
    );
    println!("Mode:     {} (no network I/O)", "dry-run".yellow());
    Ok(())
}

/// CRUX-A-01 FALSIFY-CRUX-A-01-003: build an error carrying a did-you-mean
/// hint derived from Levenshtein ≤ 2 matches against the alias map.
fn unknown_short_name_error(name: &str) -> CliError {
    use super::aliases;

    let suggestions = aliases::did_you_mean(name, 2);
    let hint = if suggestions.is_empty() {
        "Run `apr registry aliases --json` to list known short names.".to_string()
    } else {
        format!(
            "did you mean {}? (run `apr registry aliases --json` for the full list)",
            suggestions
                .iter()
                .map(|s| format!("`{s}`"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    CliError::ValidationFailed(format!(
        "CRUX-A-01: unknown short name '{name}' and not a fully-qualified URI. {hint}"
    ))
}

include!("pull_list.rs");
include!("pull_remove_resolve_model.rs");
include!("pull_extract_shard.rs");
include!("pull_04.rs");
