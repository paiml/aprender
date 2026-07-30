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
pub(crate) enum ResolvedModel {
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

pub(crate) fn build_single_cache_path(
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
pub(crate) fn get_pacha_cache_dir() -> Result<std::path::PathBuf> {
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

/// GH-213: Pull a sharded SafeTensors model (3B+ models with multiple shard files)
fn run_sharded(org: &str, repo: &str, shard_files: &[String], force: bool) -> Result<()> {
    println!(
        "Model: {}/{} ({} shards)",
        org.cyan(),
        repo.cyan(),
        shard_files.len().to_string().yellow()
    );

    let cache_dir = resolve_shard_cache_dir(org, repo)?;
    std::fs::create_dir_all(&cache_dir)?;

    let base_url = format!("https://huggingface.co/{org}/{repo}/resolve/main");

    // #1893: sharded GGUF (`model-NNNNN-of-MMMMM.gguf`) has NO index.json and
    // needs no SafeTensors conversion — download all parts and point usage at
    // the first part (GGUF split loaders find the rest via split.* metadata).
    if shard_files
        .iter()
        .all(|f| f.to_lowercase().ends_with(".gguf"))
    {
        return run_sharded_gguf(org, repo, &cache_dir, &base_url, shard_files, force);
    }

    let index_path = cache_dir.join("model.safetensors.index.json");

    download_index_if_needed(&base_url, &index_path, force)?;

    let manifest_path = cache_dir.join(".apr-manifest.json");
    let existing_manifest = load_existing_manifest(&manifest_path, force);

    let file_checksums = download_all_shards(
        &cache_dir,
        &base_url,
        shard_files,
        force,
        existing_manifest.as_ref(),
    )?;

    download_companion_files(&cache_dir, &base_url, force)?;
    write_shard_manifest(&manifest_path, org, repo, file_checksums)?;

    println!();
    println!("{} Downloaded successfully", "✓".green());
    println!("  Path: {}", index_path.display().to_string().green());
    println!("  Shards: {}", shard_files.len().to_string().yellow());

    convert_safetensors_formats(&index_path)?;

    println!();
    println!("{}", "Usage:".cyan().bold());
    println!("  apr run {}", index_path.display());
    println!("  apr serve {}", index_path.display());
    Ok(())
}

/// #1893: Download a sharded GGUF model. Unlike sharded SafeTensors there is no
/// `index.json` and no format conversion — fetch all `-of-` parts and point
/// usage at the first part (GGUF split loaders open part 1 and discover the
/// siblings via the `split.count` / `split.no` metadata keys).
fn run_sharded_gguf(
    org: &str,
    repo: &str,
    cache_dir: &Path,
    base_url: &str,
    shard_files: &[String],
    force: bool,
) -> Result<()> {
    let manifest_path = cache_dir.join(".apr-manifest.json");
    let existing_manifest = load_existing_manifest(&manifest_path, force);

    let file_checksums = download_all_shards(
        cache_dir,
        base_url,
        shard_files,
        force,
        existing_manifest.as_ref(),
    )?;

    download_companion_files(cache_dir, base_url, force)?;
    write_shard_manifest(&manifest_path, org, repo, file_checksums)?;

    println!();
    println!(
        "{} Downloaded {} GGUF shards",
        "✓".green(),
        shard_files.len().to_string().yellow()
    );

    // #1893 criterion 2: merge the parts into one loadable GGUF so the existing
    // single-file loader runs the model unchanged ("without manual
    // pre-stitching").
    let part_paths: Vec<std::path::PathBuf> =
        shard_files.iter().map(|f| cache_dir.join(f)).collect();
    let merged_path = cache_dir.join("model.gguf");
    match aprender::format::gguf::merge_gguf_shards(&part_paths, &merged_path) {
        Ok(()) => {
            // The merged model supersedes the parts — delete them so the model
            // doesn't occupy ~2× its size on disk indefinitely.
            for part in &part_paths {
                if let Err(e) = std::fs::remove_file(part) {
                    eprintln!(
                        "  {} could not remove shard {} ({e})",
                        "!".yellow(),
                        part.display()
                    );
                }
            }
            println!(
                "  {} merged {} parts → model.gguf",
                "✓".green(),
                shard_files.len().to_string().yellow()
            );
            println!("  Path: {}", merged_path.display().to_string().green());
            println!();
            println!("{}", "Usage:".cyan().bold());
            println!("  apr run {}", merged_path.display());
            println!("  apr serve {}", merged_path.display());
        }
        Err(e) => {
            // Honest failure: the individual parts are NOT independently
            // runnable, so do not point `apr run` at one of them.
            eprintln!("  {} could not assemble the sharded model: {e}", "✗".red());
            eprintln!(
                "  The {} parts were downloaded to {} but cannot be run \
                 individually. Please file an issue (#1893) with the model name.",
                shard_files.len(),
                cache_dir.display()
            );
            return Err(CliError::ValidationFailed(format!(
                "sharded GGUF merge failed: {e}"
            )));
        }
    }
    Ok(())
}

/// Resolve the on-disk cache directory for a model reference, without any
/// network I/O.
///
/// Used by `apr pull --verify`, which inspects an already-downloaded model.
/// Accepts `hf://org/repo`, `org/repo`, or a bare path to a cache directory.
pub(crate) fn resolve_cache_dir_for_ref(model_ref: &str) -> Result<std::path::PathBuf> {
    // An explicit directory wins - lets the operator verify any cache layout.
    let as_path = std::path::Path::new(model_ref);
    if as_path.is_dir() {
        return Ok(as_path.to_path_buf());
    }
    let trimmed = model_ref
        .trim_start_matches("hf://")
        .trim_start_matches("https://huggingface.co/")
        .trim_matches('/');
    let mut parts = trimmed.splitn(2, '/');
    match (parts.next(), parts.next()) {
        (Some(org), Some(repo)) if !org.is_empty() && !repo.is_empty() => {
            resolve_shard_cache_dir(org, repo)
        }
        _ => Err(CliError::ValidationFailed(format!(
            "Cannot resolve a cache directory from '{model_ref}'. \
             Expected `org/repo`, `hf://org/repo`, or an existing directory."
        ))),
    }
}

/// Resolve the cache directory for a sharded model.
fn resolve_shard_cache_dir(org: &str, repo: &str) -> Result<std::path::PathBuf> {
    Ok(dirs::home_dir()
        .ok_or_else(|| CliError::ValidationFailed("Cannot find home directory".to_string()))?
        .join(".apr")
        .join("cache")
        .join("hf")
        .join(org)
        .join(repo))
}

/// Download the SafeTensors index.json if not already cached.
fn download_index_if_needed(base_url: &str, index_path: &Path, force: bool) -> Result<()> {
    if force || !index_path.exists() {
        println!();
        println!("  {} model.safetensors.index.json", "Downloading".yellow());
        download_file(
            &format!("{base_url}/model.safetensors.index.json"),
            index_path,
        )?;
    } else {
        println!("  {} model.safetensors.index.json (cached)", "✓".green());
    }
    Ok(())
}

/// Load existing shard manifest for cache-hit verification (GH-213).
fn load_existing_manifest(manifest_path: &Path, force: bool) -> Option<ShardManifest> {
    if force || !manifest_path.exists() {
        return None;
    }
    std::fs::read_to_string(manifest_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
}

/// Download all shards, collecting checksums for the manifest.
fn download_all_shards(
    cache_dir: &Path,
    base_url: &str,
    shard_files: &[String],
    force: bool,
    existing_manifest: Option<&ShardManifest>,
) -> Result<HashMap<String, FileChecksum>> {
    let mut file_checksums: HashMap<String, FileChecksum> = HashMap::new();
    let total = shard_files.len();
    for (i, shard_file) in shard_files.iter().enumerate() {
        download_or_verify_shard(
            cache_dir,
            base_url,
            shard_file,
            i,
            total,
            force,
            existing_manifest,
            &mut file_checksums,
        )?;
    }
    Ok(file_checksums)
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

/// Download companion files (tokenizer, config) for sharded models.
///
/// GH-356: tokenizer.json is optional — some models only have tokenizer.model (SentencePiece)
/// or tokenizer_config.json. We validate that at least ONE tokenizer file was obtained.
fn download_companion_files(cache_dir: &Path, base_url: &str, force: bool) -> Result<()> {
    // (filename, is_required) — tokenizer files are individually optional but collectively required
    let companions = [
        ("tokenizer.json", false),
        ("config.json", true),
        ("tokenizer_config.json", false),
        ("tokenizer.model", false),
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
            Err(CliError::HttpNotFound(_)) if *required => {
                return Err(CliError::ValidationFailed(format!(
                    "{filename} is required for inference but was not found (HTTP 404) at {url}"
                )));
            }
            Err(CliError::HttpNotFound(_)) => {
                println!("  {} {} (not found in repo)", "⚠".yellow(), filename);
            }
            Err(e) if *required => {
                return Err(CliError::ValidationFailed(format!(
                    "{filename} is required for inference but download failed: {e}"
                )));
            }
            Err(_) => println!("  {} {} (not available, optional)", "⚠".yellow(), filename),
        }
    }

    // GH-356: Validate at least one tokenizer file exists
    let tokenizer_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"];
    let has_tokenizer = tokenizer_files.iter().any(|f| cache_dir.join(f).exists());
    if !has_tokenizer {
        return Err(CliError::ValidationFailed(format!(
            "No tokenizer found for this model. Tried: {}.\n\
             The model may require a custom tokenizer not hosted in the repository.",
            tokenizer_files.join(", ")
        )));
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
include!("pull_dataset.rs");

#[cfg(all(test, feature = "inference"))]
mod sharded_gguf_interop_tests {
    use aprender::format::gguf::{
        export_tensors_to_gguf, merge_gguf_shards, GgmlType, GgufTensor, GgufValue,
    };
    use std::path::Path;

    fn write_part(path: &Path, tensors: &[GgufTensor], meta: &[(String, GgufValue)]) {
        let mut buf = Vec::new();
        export_tensors_to_gguf(&mut buf, tensors, meta).expect("export part");
        std::fs::write(path, &buf).expect("write part");
    }

    /// FT-MERGE-006: the merged sharded GGUF is accepted by realizar's OWN GGUF
    /// parser (`GGUFModel::from_bytes`) — the actual inference loader, not just
    /// aprender-core's reader. Closes the cross-parser verification gap: a merge
    /// validated only by the writer's sibling reader could still be rejected by
    /// the loader it exists to feed.
    #[test]
    fn merged_sharded_gguf_loads_in_realizar() {
        let dir = std::env::temp_dir().join(format!("apr-merge-interop-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let p0 = dir.join("model-00001-of-00002.gguf");
        let p1 = dir.join("model-00002-of-00002.gguf");
        let merged = dir.join("model.gguf");

        let tensor = |name: &str, fill: u8| GgufTensor {
            name: name.into(),
            shape: vec![4],
            dtype: GgmlType::F32,
            data: vec![fill; 16],
        };
        write_part(
            &p0,
            &[tensor("blk.0.weight", 1)],
            &[
                (
                    "general.architecture".into(),
                    GgufValue::String("gemma".into()),
                ),
                ("gemma.embedding_length".into(), GgufValue::Uint32(2048)),
                ("gemma.block_count".into(), GgufValue::Uint32(18)),
                ("split.no".into(), GgufValue::Uint16(0)),
                ("split.count".into(), GgufValue::Uint16(2)),
            ],
        );
        write_part(
            &p1,
            &[tensor("blk.1.weight", 2)],
            &[("split.no".into(), GgufValue::Uint16(1))],
        );

        merge_gguf_shards(&[p0, p1], &merged).expect("merge");
        let bytes = std::fs::read(&merged).expect("read merged");

        let parsed = realizar::gguf::GGUFModel::from_bytes(&bytes);
        assert!(
            parsed.is_ok(),
            "realizar's GGUF loader must accept the merged sharded file: {:?}",
            parsed.err()
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
