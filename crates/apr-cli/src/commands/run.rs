//! Run command implementation
//!
//! Implements APR-SPEC §4.15: Run Command
//!
//! Run models directly with zero configuration. Supports:
//! - Local files: `apr run model.apr --input audio.wav`
//! - HuggingFace: `apr run hf://openai/whisper-tiny --input audio.wav`
//! - URLs: `apr run https://example.com/model.apr --input audio.wav`
//!
//! Features automatic:
//! - Model caching (~/.apr/cache/)
//! - mmap loading for models >50MB
//! - Backend selection (AVX2/GPU/WASM via trueno)
//!
//! # Architecture (APR-CLI-DELEGATE-001)
//!
//! All inference now delegates to `realizar::run_inference()` via `execute_with_realizar()`.
//!
//! ## Deprecated Legacy Code (kept for reference)
//!
//! The following functions are deprecated and no longer called in the main path:
//! - `execute_apr_inference()` - Superseded by realizar
//! - `execute_safetensors_inference()` - Superseded by realizar
//! - `execute_gguf_inference()` - Superseded by realizar
//! - `run_safetensors_generation()` - Superseded by realizar
//! - `run_gguf_generate()` - Superseded by realizar
//!
//! These remain in the codebase for:
//! 1. Historical reference (how inference was done pre-realizar)
//! 2. Potential fallback if realizar has regressions (compile with different features)
//!
//! See PMAT-SHOWCASE-BRICK-001 for cleanup tracking.

// Allow dead code during development - legacy functions pending removal
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(clippy::needless_return)]
#![allow(clippy::format_push_string)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::disallowed_methods)]

use crate::error::{CliError, Result};
use colored::Colorize;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Model source types
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ModelSource {
    /// Local file path
    Local(PathBuf),
    /// HuggingFace Hub (hf://org/repo or hf://org/repo/filename.gguf)
    HuggingFace {
        org: String,
        repo: String,
        /// Optional specific file within the repo (e.g., "model-q4_k_m.gguf")
        file: Option<String>,
    },
    /// Direct URL
    Url(String),
}

impl ModelSource {
    /// Parse a source string into a ModelSource
    pub(crate) fn parse(source: &str) -> Result<Self> {
        if source.starts_with("hf://") {
            let path = source.strip_prefix("hf://").unwrap_or(source);
            let parts: Vec<&str> = path.split('/').collect();
            if parts.len() >= 2 {
                // Check if there's a filename (3rd part with extension)
                let file = if parts.len() >= 3 && parts[2].contains('.') {
                    Some(parts[2..].join("/"))
                } else {
                    None
                };
                Ok(Self::HuggingFace {
                    org: parts[0].to_string(),
                    repo: parts[1].to_string(),
                    file,
                })
            } else {
                Err(CliError::InvalidFormat(format!(
                    "Invalid HuggingFace source: {source}. Expected hf://org/repo"
                )))
            }
        } else if source.starts_with("http://") || source.starts_with("https://") {
            Ok(Self::Url(source.to_string()))
        } else {
            Ok(Self::Local(PathBuf::from(source)))
        }
    }

    /// Get cache path for this source
    pub(crate) fn cache_path(&self) -> PathBuf {
        let cache_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".apr")
            .join("cache");

        match self {
            Self::Local(path) => path.clone(),
            Self::HuggingFace { org, repo, .. } => cache_dir.join("hf").join(org).join(repo),
            Self::Url(url) => {
                // Hash URL for cache key
                let hash = format!("{:x}", md5_hash(url.as_bytes()));
                cache_dir.join("urls").join(&hash[..16])
            }
        }
    }
}

/// Simple MD5-like hash for cache keys (not cryptographic)
fn md5_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in data {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Run options
#[derive(Debug, Clone)]
pub(crate) struct RunOptions {
    /// Input file (audio, text, etc.)
    pub input: Option<PathBuf>,
    /// Text prompt for LLM generation
    pub prompt: Option<String>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Output format (text, json, srt, vtt)
    pub output_format: String,
    /// Force re-download (bypass cache)
    pub force: bool,
    /// Disable GPU acceleration
    pub no_gpu: bool,
    /// Offline mode: refuse any network access
    pub offline: bool,
    /// Benchmark mode: output performance metrics
    pub benchmark: bool,
    /// Verbose mode: show loading/backend metadata (NOISY-GUARD F-UX-27)
    pub verbose: bool,
    /// Enable inference tracing (APR-TRACE-001)
    pub trace: bool,
    /// Trace specific steps only
    pub trace_steps: Option<Vec<String>>,
    /// Verbose tracing (show tensor values)
    pub trace_verbose: bool,
    /// Save trace output to JSON file
    pub trace_output: Option<PathBuf>,
    /// Trace detail level: none, basic, layer, payload (PMAT-480)
    pub trace_level: String,
    /// Enable inline Roofline profiling (PMAT-480)
    pub profile: bool,
    /// Sampling temperature (0.0 = greedy argmax, >0 = stochastic)
    /// PMAT-496 / F-CLIPARITY-01
    pub temperature: f32,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Top-p nucleus sampling (None = disabled)
    pub top_p: Option<f32>,
    /// RNG seed for deterministic sampling
    pub seed: u64,
    /// Repetition penalty (1.0 = no penalty)
    pub repeat_penalty: f32,
    /// Context window for repetition penalty
    pub repeat_last_n: usize,
    /// Process prompt tokens one-by-one (debug prefill)
    pub split_prompt: bool,
    /// `--stream`: emit one NDJSON event per generated token.
    ///
    /// Known here (not only at the print site) because streaming is the one
    /// mode that needs per-token decoded text, and resolving the tokenizer for
    /// it costs a second open of the model file.
    pub stream: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            input: None,
            prompt: None,
            max_tokens: 32,
            output_format: "text".to_string(),
            force: false,
            no_gpu: false,
            offline: false,
            benchmark: false,
            verbose: false,
            trace: false,
            trace_steps: None,
            trace_verbose: false,
            trace_output: None,
            trace_level: "basic".to_string(),
            profile: false,
            temperature: 0.0,
            top_k: 1,
            top_p: None,
            seed: 299_792_458,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            split_prompt: false,
            stream: false,
        }
    }
}

/// Run result
#[derive(Debug, Clone)]
pub(crate) struct RunResult {
    /// Output text
    pub text: String,
    /// Processing time in seconds
    pub duration_secs: f64,
    /// Whether model was cached
    pub cached: bool,
    /// Number of tokens generated (for benchmark mode)
    pub tokens_generated: Option<usize>,
    /// Tokens per second from inference engine (GH-250)
    pub tok_per_sec: Option<f64>,
    /// Whether GPU was used (GH-250)
    pub used_gpu: Option<bool>,
    /// GH-250: Generated token IDs for parity checking
    pub generated_tokens: Option<Vec<u32>>,
    /// Per-token decoded text, positionally aligned with `generated_tokens`.
    ///
    /// `Some` only in `--stream` mode and only when a tokenizer for the model
    /// could be resolved. `--stream` used to emit `"text":""` for every token
    /// because nothing ever decoded the ids one at a time.
    pub token_texts: Option<Vec<String>>,
}

/// Run the model on input
#[provable_contracts_macros::contract("apr-cli-operations-v1", equation = "long_running_graceful")]
pub(crate) fn run_model(source: &str, options: &RunOptions) -> Result<RunResult> {
    let start = Instant::now();

    // Resolve alias if applicable
    let resolved_source =
        crate::commands::aliases::resolve_short_name(source).unwrap_or_else(|| source.to_string());

    // Convert to canonical HF if it looks like org/repo (but isn't a local path)
    let hf_uri = if !resolved_source.contains("://")
        && resolved_source.contains('/')
        && !std::path::Path::new(&resolved_source).exists()
        && !resolved_source.starts_with('/')
    {
        format!("hf://{resolved_source}")
    } else {
        resolved_source
    };

    // GH-213: Actually query HF to find the best GGUF file
    let fully_resolved_source = match crate::commands::pull::resolve_hf_model(&hf_uri) {
        Ok(crate::commands::pull::ResolvedModel::SingleFile(uri)) => uri,
        _ => hf_uri,
    };

    // Parse source
    let model_source = ModelSource::parse(&fully_resolved_source)?;

    // Resolve model path (download if needed, respecting offline mode)
    let model_path = resolve_model(&model_source, options.force, options.offline)?;

    // Validate model exists
    if !model_path.exists() {
        return Err(CliError::FileNotFound(model_path));
    }

    // Check if input is required
    let input_path = options.input.as_ref();

    // Load and run model
    // BUG-RUN-001 FIX: Now receives InferenceOutput with actual token count
    let output = execute_inference(&model_path, input_path, options)?;

    let duration = start.elapsed();

    // BUG-RUN-001 FIX: Use actual token count from inference engine if available,
    // otherwise fall back to word count approximation
    let tokens_generated = output
        .tokens_generated
        .or_else(|| Some(output.text.split_whitespace().count()));

    Ok(RunResult {
        text: output.text,
        duration_secs: duration.as_secs_f64(),
        cached: matches!(model_source, ModelSource::Local(_)) || model_source.cache_path().exists(),
        tokens_generated,
        tok_per_sec: output.tok_per_sec,
        used_gpu: output.used_gpu,
        generated_tokens: output.generated_tokens,
        token_texts: output.token_texts,
    })
}

/// Resolve model source to local path
///
/// When `offline` is true, this function enforces strict network isolation:
/// - Local files are allowed
/// - Cached models are allowed (no network required)
/// - Non-cached remote sources are rejected with a clear error
///
/// Per Section 9.2 (Sovereign AI): "apr run --offline is mandatory for production"
pub(crate) fn resolve_model(source: &ModelSource, force: bool, offline: bool) -> Result<PathBuf> {
    match source {
        ModelSource::Local(path) => Ok(path.clone()),
        ModelSource::HuggingFace { org, repo, file } => {
            // Check multiple cache locations for the model
            // GH-528: Skip cache when --force is set to re-download
            if !force {
                if let Some(path) = find_cached_model(org, repo, file.as_deref()) {
                    return Ok(path);
                }
            }

            if offline {
                // OFFLINE MODE: Reject any network access attempt
                return Err(CliError::ValidationFailed(format!(
                    "OFFLINE MODE: Model hf://{org}/{repo} not cached. \
                     Network access is disabled. Cache the model first with: \
                     apr import hf://{org}/{repo}"
                )));
            }

            // Auto-download like ollama
            eprintln!("{}", format!("Downloading hf://{org}/{repo}...").yellow());
            download_hf_model(org, repo, file.as_deref())
        }
        ModelSource::Url(url) => {
            let cache_path = source.cache_path();
            // GH-528: Skip cache when --force is set
            if !force && cache_path.exists() {
                // Cached URLs are allowed even in offline mode
                find_model_in_dir(&cache_path)
            } else if offline {
                // OFFLINE MODE: Reject any network access attempt
                Err(CliError::ValidationFailed(format!(
                    "OFFLINE MODE: URL {url} not cached. \
                     Network access is disabled. Download and cache the model first."
                )))
            } else {
                // Auto-download from URL
                eprintln!("{}", format!("Downloading {url}...").yellow());
                download_url_model(url)
            }
        }
    }
}

/// Find model in various cache locations (HF cache, apr cache)
///
/// If `file` is specified, look for that exact file. Otherwise, search for common model files.
/// Search for a target file (or common model filenames) in a directory.
fn find_model_file_in_dir(dir: &Path, file: Option<&str>) -> Option<PathBuf> {
    if let Some(filename) = file {
        let path = dir.join(filename);
        if path.exists() {
            return Some(path);
        }
    } else {
        // Sharded model indices are preferred if present
        let index_path = dir.join("model.safetensors.index.json");
        if index_path.exists() {
            return Some(index_path);
        }

        for name in &["model.safetensors", "pytorch_model.bin", "model.apr"] {
            let path = dir.join(name);
            if path.exists() {
                return Some(path);
            }
        }
    }
    None
}

/// Search HuggingFace hub cache snapshots for a model.
fn find_in_hf_cache(org: &str, repo: &str, file: Option<&str>) -> Option<PathBuf> {
    let hf_cache = dirs::home_dir()?
        .join(".cache")
        .join("huggingface")
        .join("hub");
    let snapshots_dir = hf_cache
        .join(format!("models--{org}--{repo}"))
        .join("snapshots");
    let entries = std::fs::read_dir(&snapshots_dir).ok()?;
    for entry in entries.flatten() {
        if let Some(found) = find_model_file_in_dir(&entry.path(), file) {
            return Some(found);
        }
    }
    None
}

/// Search APR cache for a model.
fn find_in_apr_cache(org: &str, repo: &str, file: Option<&str>) -> Option<PathBuf> {
    let apr_cache = dirs::home_dir()?
        .join(".apr")
        .join("cache")
        .join("hf")
        .join(org)
        .join(repo);
    if !apr_cache.exists() {
        return None;
    }
    if let Some(filename) = file {
        let path = apr_cache.join(filename);
        if path.exists() {
            return Some(path);
        }
    } else {
        // Sharded model indices are preferred if present
        let index_path = apr_cache.join("model.safetensors.index.json");
        if index_path.exists() {
            return Some(index_path);
        }

        for ext in &["apr", "safetensors", "gguf"] {
            let path = apr_cache.join(format!("model.{ext}"));
            if path.exists() {
                return Some(path);
            }
        }
    }
    None
}

fn find_cached_model(org: &str, repo: &str, file: Option<&str>) -> Option<PathBuf> {
    // 1. Check HF Cache
    if let Some(path) = find_in_hf_cache(org, repo, file) {
        return Some(path);
    }

    // 2. Check APR Cache
    if let Some(path) = find_in_apr_cache(org, repo, file) {
        return Some(path);
    }

    // 3. Check Pacha Cache (used by `apr pull` for HF single-file streaming)
    if let Some(filename) = file {
        if let Ok(pacha_dir) = crate::commands::pull::get_pacha_cache_dir() {
            // Reconstruct the original model_ref that `apr pull` would have hashed
            let model_ref = format!("hf://{org}/{repo}/{filename}");
            let (_, pacha_path) =
                crate::commands::pull::build_single_cache_path(&pacha_dir, &model_ref, filename);
            if pacha_path.exists() {
                return Some(pacha_path);
            }
        }
    }

    None
}

/// Download model from HuggingFace and cache it
///
/// If `file` is specified (e.g., "model-q4_k_m.gguf"), download that specific file.
/// Otherwise, download model.safetensors (or sharded version).
pub(crate) fn download_hf_model(org: &str, repo: &str, file: Option<&str>) -> Result<PathBuf> {
    let cache_dir = hf_cache_dir(org, repo)?;
    let base_url = format!("https://huggingface.co/{org}/{repo}/resolve/main");

    if let Some(filename) = file {
        return download_single_hf_file(&cache_dir, &base_url, filename);
    }

    let model_path = download_main_safetensors(&cache_dir, &base_url)?;
    let config_path =
        download_required_companion(&base_url, &cache_dir, "config.json", &[&model_path])?;
    download_required_companion(
        &base_url,
        &cache_dir,
        "tokenizer.json",
        &[&model_path, &config_path],
    )?;
    download_optional_companion(&base_url, &cache_dir, "tokenizer_config.json");

    eprintln!("{}", "  Download complete!".green());
    Ok(model_path)
}

fn hf_cache_dir(org: &str, repo: &str) -> Result<PathBuf> {
    let dir = dirs::home_dir()
        .ok_or_else(|| CliError::ValidationFailed("Cannot find home directory".to_string()))?
        .join(".apr")
        .join("cache")
        .join("hf")
        .join(org)
        .join(repo);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn download_single_hf_file(cache_dir: &Path, base_url: &str, filename: &str) -> Result<PathBuf> {
    let model_url = format!("{base_url}/{filename}");
    let model_path = cache_dir.join(filename);

    eprintln!("  Downloading {}...", filename);
    download_file(&model_url, &model_path)?;
    eprintln!("{}", "  Download complete!".green());
    Ok(model_path)
}

/// Download SafeTensors weights. GH-127: prefer sharded (index.json) over single file.
fn download_main_safetensors(cache_dir: &Path, base_url: &str) -> Result<PathBuf> {
    let index_url = format!("{base_url}/model.safetensors.index.json");
    let index_path = cache_dir.join("model.safetensors.index.json");

    if download_file(&index_url, &index_path).is_ok() {
        eprintln!("  Detected sharded model (multi-tensor)");
        return download_sharded_model(cache_dir, &index_path, base_url);
    }

    let model_url = format!("{base_url}/model.safetensors");
    let model_path = cache_dir.join("model.safetensors");
    eprintln!("  Downloading model.safetensors...");
    download_file(&model_url, &model_path)?;
    Ok(model_path)
}

/// Download a required companion file. On failure, delete `cleanup_paths` and return a
/// contextual error. Used for config.json / tokenizer.json (GH-150).
fn download_required_companion(
    base_url: &str,
    cache_dir: &Path,
    filename: &str,
    cleanup_paths: &[&Path],
) -> Result<PathBuf> {
    let url = format!("{base_url}/{filename}");
    let path = cache_dir.join(filename);
    eprintln!("  Downloading {}...", filename);
    download_file(&url, &path).map_err(|e| {
        for p in cleanup_paths {
            let _ = std::fs::remove_file(p);
        }
        CliError::ValidationFailed(format!(
            "{filename} is required for inference but download failed: {e}\n\
             Ensure the HuggingFace repo contains {filename}"
        ))
    })?;
    Ok(path)
}

fn download_optional_companion(base_url: &str, cache_dir: &Path, filename: &str) {
    let url = format!("{base_url}/{filename}");
    let path = cache_dir.join(filename);
    eprintln!("  Downloading {}...", filename);
    if let Err(e) = download_file(&url, &path) {
        eprintln!("  Note: {filename} not available (optional): {e}");
    }
}

include!("inference_output.rs");
include!("run_resolve_tokenizer.rs");
include!("safetensors.rs");
include!("gguf_generate_result.rs");
include!("run_entry.rs");
include!("run_07.rs");
