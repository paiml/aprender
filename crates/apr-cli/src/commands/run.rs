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
//! The following legacy functions are DEAD CODE scheduled for removal:
//! - `execute_apr_inference()` (lines 603-881)
//! - `execute_safetensors_inference()` (lines 927-1208)
//! - `execute_gguf_inference()` (lines 1317-1488)
//! - `run_safetensors_generation()` (lines 1235-1309)
//! - `run_gguf_generate()` (lines 1500-1633)
//!
//! TODO(SHOWCASE-BRICK-001): Remove dead code after realizar refactor is stable.

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
}

/// Run the model on input
pub(crate) fn run_model(source: &str, options: &RunOptions) -> Result<RunResult> {
    let start = Instant::now();

    // Parse source
    let model_source = ModelSource::parse(source)?;

    // Resolve model path (download if needed, respecting offline mode)
    let model_path = resolve_model(&model_source, options.force, options.offline)?;

    // Validate model exists
    if !model_path.exists() {
        return Err(CliError::FileNotFound(model_path));
    }

    // Check if input is required
    let input_path = options.input.as_ref();

    // Load and run model
    let text = execute_inference(&model_path, input_path, options)?;

    let duration = start.elapsed();

    // Estimate tokens generated from output text (word count approximation)
    let tokens_generated = Some(text.split_whitespace().count());

    Ok(RunResult {
        text,
        duration_secs: duration.as_secs_f64(),
        cached: matches!(model_source, ModelSource::Local(_)) || model_source.cache_path().exists(),
        tokens_generated,
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
fn resolve_model(source: &ModelSource, _force: bool, offline: bool) -> Result<PathBuf> {
    match source {
        ModelSource::Local(path) => Ok(path.clone()),
        ModelSource::HuggingFace { org, repo, file } => {
            // Check multiple cache locations for the model
            if let Some(path) = find_cached_model(org, repo, file.as_deref()) {
                return Ok(path);
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
            if cache_path.exists() {
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
fn find_cached_model(org: &str, repo: &str, file: Option<&str>) -> Option<PathBuf> {
    // Check HuggingFace hub cache first (standard location)
    let hf_cache = dirs::home_dir().map(|h| h.join(".cache").join("huggingface").join("hub"))?;

    let hf_model_dir = hf_cache.join(format!("models--{org}--{repo}"));
    if hf_model_dir.exists() {
        // Find the latest snapshot
        let snapshots_dir = hf_model_dir.join("snapshots");
        if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
            for entry in entries.flatten() {
                let snapshot_dir = entry.path();
                // If specific file requested, look for it
                if let Some(filename) = file {
                    let model_path = snapshot_dir.join(filename);
                    if model_path.exists() {
                        return Some(model_path);
                    }
                } else {
                    // Look for model.safetensors or model files
                    for filename in &["model.safetensors", "pytorch_model.bin", "model.apr"] {
                        let model_path = snapshot_dir.join(filename);
                        if model_path.exists() {
                            return Some(model_path);
                        }
                    }
                }
            }
        }
    }

    // Check apr cache
    let apr_cache =
        dirs::home_dir().map(|h| h.join(".apr").join("cache").join("hf").join(org).join(repo))?;
    if apr_cache.exists() {
        // If specific file requested, look for it
        if let Some(filename) = file {
            let model_path = apr_cache.join(filename);
            if model_path.exists() {
                return Some(model_path);
            }
        } else {
            for ext in &["apr", "safetensors", "gguf"] {
                let pattern = apr_cache.join(format!("model.{ext}"));
                if pattern.exists() {
                    return Some(pattern);
                }
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
    let cache_dir = dirs::home_dir()
        .ok_or_else(|| CliError::ValidationFailed("Cannot find home directory".to_string()))?
        .join(".apr")
        .join("cache")
        .join("hf")
        .join(org)
        .join(repo);

    std::fs::create_dir_all(&cache_dir)?;

    let base_url = format!("https://huggingface.co/{org}/{repo}/resolve/main");

    // If specific file requested (e.g., GGUF), download just that file
    if let Some(filename) = file {
        let model_url = format!("{base_url}/{filename}");
        let model_path = cache_dir.join(filename);

        eprintln!("  Downloading {}...", filename);
        download_file(&model_url, &model_path)?;
        eprintln!("{}", "  Download complete!".green());
        return Ok(model_path);
    }

    // Default path: download SafeTensors model with config/tokenizer
    let tokenizer_url = format!("{base_url}/tokenizer.json");
    let config_url = format!("{base_url}/config.json");

    let tokenizer_path = cache_dir.join("tokenizer.json");
    let config_path = cache_dir.join("config.json");

    // GH-127: Check for sharded model first (index.json indicates multi-tensor model)
    let index_url = format!("{base_url}/model.safetensors.index.json");
    let index_path = cache_dir.join("model.safetensors.index.json");

    let model_path = if download_file(&index_url, &index_path).is_ok() {
        // Sharded model - download all shards listed in index
        eprintln!("  Detected sharded model (multi-tensor)");
        download_sharded_model(&cache_dir, &index_path, &base_url)?
    } else {
        // Single model.safetensors file
        let model_url = format!("{base_url}/model.safetensors");
        let model_path = cache_dir.join("model.safetensors");

        eprintln!("  Downloading model.safetensors...");
        download_file(&model_url, &model_path)?;
        model_path
    };

    // Download config.json (REQUIRED for SafeTensors inference) - GH-150
    eprintln!("  Downloading config.json...");
    download_file(&config_url, &config_path).map_err(|e| {
        // Clean up partial download on failure
        let _ = std::fs::remove_file(&model_path);
        CliError::ValidationFailed(format!(
            "config.json is required for inference but download failed: {e}\n\
             Ensure the HuggingFace repo contains config.json"
        ))
    })?;

    // Download tokenizer.json (REQUIRED for text encoding/decoding)
    eprintln!("  Downloading tokenizer.json...");
    download_file(&tokenizer_url, &tokenizer_path).map_err(|e| {
        // Clean up partial download on failure
        let _ = std::fs::remove_file(&model_path);
        let _ = std::fs::remove_file(&config_path);
        CliError::ValidationFailed(format!(
            "tokenizer.json is required for inference but download failed: {e}\n\
             Ensure the HuggingFace repo contains tokenizer.json"
        ))
    })?;

    // Download tokenizer_config.json (optional but recommended)
    let tokenizer_config_url = format!("{base_url}/tokenizer_config.json");
    let tokenizer_config_path = cache_dir.join("tokenizer_config.json");
    eprintln!("  Downloading tokenizer_config.json...");
    if let Err(e) = download_file(&tokenizer_config_url, &tokenizer_config_path) {
        eprintln!("  Note: tokenizer_config.json not available (optional): {e}");
    }

    eprintln!("{}", "  Download complete!".green());

    Ok(model_path)
}

/// Download sharded model files from HuggingFace (GH-127)
///
/// Parses the index.json to get list of shard files and downloads each one.
/// Returns path to the index file which can be used to locate all shards.
fn download_sharded_model(cache_dir: &Path, index_path: &Path, base_url: &str) -> Result<PathBuf> {
    // Read and parse index file
    let index_content = std::fs::read_to_string(index_path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to read index file: {e}")))?;

    // Parse weight_map to get unique shard filenames
    // Format: {"metadata": {...}, "weight_map": {"tensor.name": "model-00001-of-00006.safetensors", ...}}
    let shard_files: HashSet<String> = extract_shard_files(&index_content);

    if shard_files.is_empty() {
        return Err(CliError::ValidationFailed(
            "Sharded model index contains no shard files".to_string(),
        ));
    }

    let total_shards = shard_files.len();
    eprintln!("  Found {} shard files to download", total_shards);

    // Download each shard
    for (i, shard_file) in shard_files.iter().enumerate() {
        let shard_url = format!("{base_url}/{shard_file}");
        let shard_path = cache_dir.join(shard_file);

        // Skip if already cached
        if shard_path.exists() {
            eprintln!("  [{}/{}] {} (cached)", i + 1, total_shards, shard_file);
            continue;
        }

        eprintln!(
            "  [{}/{}] Downloading {}...",
            i + 1,
            total_shards,
            shard_file
        );
        download_file(&shard_url, &shard_path)?;
    }

    // Return path to index file (caller uses this to locate shards)
    Ok(index_path.to_path_buf())
}

/// Extract unique shard filenames from index.json weight_map
fn extract_shard_files(json: &str) -> HashSet<String> {
    let mut files = HashSet::new();

    // Simple parsing - find "weight_map" section and extract shard filenames
    // Shard files look like: "model-00001-of-00006.safetensors"
    if let Some(weight_map_start) = json.find("\"weight_map\"") {
        let after_key = &json[weight_map_start..];
        if let Some(brace_start) = after_key.find('{') {
            let content = &after_key[brace_start + 1..];
            // Find matching closing brace (handle nested braces)
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
                // Format: "tensor.name": "shard-file.safetensors"
                if let Some(colon_pos) = part.rfind(':') {
                    let value = part[colon_pos + 1..].trim();
                    // Remove quotes and extract filename
                    let filename =
                        value.trim_matches(|c| c == '"' || c == ' ' || c == '\n' || c == '\r');
                    if filename.ends_with(".safetensors") && !filename.is_empty() {
                        files.insert(filename.to_string());
                    }
                }
            }
        }
    }

    files
}

/// Download model from arbitrary URL
///
/// Caches to ~/.apr/cache/url/<hash>/<filename>
fn download_url_model(url: &str) -> Result<PathBuf> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Hash URL for cache directory
    let mut hasher = DefaultHasher::new();
    url.hash(&mut hasher);
    let url_hash = format!("{:016x}", hasher.finish());

    // Extract filename from URL or use default
    let filename = url
        .rsplit('/')
        .next()
        .filter(|s| !s.is_empty() && s.contains('.'))
        .unwrap_or("model.safetensors");

    let cache_dir = dirs::home_dir()
        .ok_or_else(|| CliError::ValidationFailed("Cannot find home directory".to_string()))?
        .join(".apr")
        .join("cache")
        .join("url")
        .join(&url_hash);

    std::fs::create_dir_all(&cache_dir)?;

    let model_path = cache_dir.join(filename);

    // Download model
    eprintln!("  Downloading {}...", filename);
    download_file(url, &model_path)?;

    eprintln!("{}", "  Download complete!".green());

    Ok(model_path)
}

/// Download a file from URL to local path
fn download_file(url: &str, path: &Path) -> Result<()> {
    use std::io::Write;

    // Use ureq for simple HTTP requests (already a dependency via hf-hub)
    let response = ureq::get(url)
        .call()
        .map_err(|e| CliError::ValidationFailed(format!("Download failed: {e}")))?;

    if response.status() != 200 {
        return Err(CliError::ValidationFailed(format!(
            "Download failed with status {}: {}",
            response.status(),
            url
        )));
    }

    let mut file = std::fs::File::create(path)?;
    let mut reader = response.into_reader();
    std::io::copy(&mut reader, &mut file)?;

    Ok(())
}

/// Find model file in directory
#[allow(clippy::unnecessary_wraps)] // Consistent with error-returning callers
fn find_model_in_dir(dir: &Path) -> Result<PathBuf> {
    for ext in &["apr", "safetensors", "gguf"] {
        let pattern = dir.join(format!("*.{ext}"));
        if let Some(path) = glob_first(&pattern) {
            return Ok(path);
        }
    }
    // Return directory itself if no model found
    Ok(dir.to_path_buf())
}

/// Get first match from glob pattern
fn glob_first(pattern: &Path) -> Option<PathBuf> {
    glob::glob(pattern.to_str()?).ok()?.next()?.ok()
}

/// Execute inference on model
fn execute_inference(
    model_path: &Path,
    input_path: Option<&PathBuf>,
    options: &RunOptions,
) -> Result<String> {
    // Check model file size for mmap decision
    let metadata = std::fs::metadata(model_path)?;
    let use_mmap = metadata.len() > 50 * 1024 * 1024; // 50MB threshold

    // F-UX-26: Only show mmap info in verbose mode (NOISY-GUARD)
    if use_mmap && options.verbose {
        eprintln!(
            "{}",
            format!("Using mmap for {}MB model", metadata.len() / 1024 / 1024).dimmed()
        );
    }

    // Try realizar inference if feature enabled
    #[cfg(feature = "inference")]
    {
        return execute_with_realizar(model_path, input_path, options, use_mmap);
    }

    // Fallback: placeholder when realizar not available
    #[cfg(not(feature = "inference"))]
    {
        let input_desc =
            input_path.map_or_else(|| "stdin".to_string(), |p| p.display().to_string());

        Ok(format!(
            "[Inference requires --features inference]\nModel: {}\nInput: {}\nFormat: {}\nGPU: {}",
            model_path.display(),
            input_desc,
            options.output_format,
            if options.no_gpu { "disabled" } else { "auto" }
        ))
    }
}

/// Execute inference using realizar engine
///
/// Per spec APR-CLI-DELEGATE-001: All inference delegates to realizar's
/// high-level API. This eliminates ~1500 lines of duplicated code.
#[cfg(feature = "inference")]
fn execute_with_realizar(
    model_path: &Path,
    input_path: Option<&PathBuf>,
    options: &RunOptions,
    _use_mmap: bool,
) -> Result<String> {
    use realizar::{run_inference, InferenceConfig};

    // Get prompt from options or input file
    let prompt = if let Some(ref p) = options.prompt {
        Some(p.clone())
    } else if let Some(path) = input_path {
        Some(std::fs::read_to_string(path)?)
    } else {
        None
    };

    // Build inference config
    let mut config = InferenceConfig::new(model_path);
    if let Some(ref p) = prompt {
        config = config.with_prompt(p);
    }
    config = config
        .with_max_tokens(options.max_tokens)
        .with_verbose(options.verbose); // NOISY-GUARD F-UX-27: explicit --verbose flag

    if options.no_gpu {
        config = config.without_gpu();
    }

    if options.trace {
        config = config.with_trace(true);
    }

    // Run inference via realizar
    let result = run_inference(&config)
        .map_err(|e| CliError::InferenceFailed(format!("Inference failed: {e}")))?;

    // Report performance if benchmarking
    if options.benchmark {
        eprintln!(
            "{}",
            format!(
                "Generated {} tokens in {:.1}ms ({:.1} tok/s)",
                result.generated_token_count, result.inference_ms, result.tok_per_sec
            )
            .green()
        );
    }

    Ok(result.text)
}

/// Execute APR model inference (APR v2 format)
///
/// APR v2 now supports transformer inference with forward() and generate() methods.
/// For transformer models with proper metadata, runs autoregressive generation.
/// Supports GPU acceleration via `AprV2ModelCuda` when `--gpu` is specified.
#[cfg(feature = "inference")]
fn execute_apr_inference(
    model_path: &Path,
    input_path: Option<&PathBuf>,
    options: &RunOptions,
) -> Result<String> {
    use realizar::apr::AprModel;
    use std::time::Instant;

    // Check if GPU should be used
    #[cfg(feature = "cuda")]
    let use_gpu = !options.no_gpu && realizar::apr::AprV2ModelCuda::is_available();
    #[cfg(not(feature = "cuda"))]
    let use_gpu = false;

    // Load the APR v2 model
    let start = Instant::now();
    let model = AprModel::load(model_path)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load APR model: {e}")))?;
    let load_time = start.elapsed();

    // Display model info
    let model_type = model
        .metadata()
        .model_type
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let architecture = model
        .metadata()
        .architecture
        .clone()
        .unwrap_or_else(|| "unknown".to_string());

    eprintln!(
        "{}",
        format!(
            "Loaded {} model (arch: {}, {} tensors, ~{} parameters) in {:.2}ms",
            model_type,
            architecture,
            model.tensor_count(),
            model.estimated_parameters(),
            load_time.as_secs_f64() * 1000.0
        )
        .dimmed()
    );

    // Check if this is a transformer model
    if model.metadata().is_transformer() {
        eprintln!("{}", "Running transformer generation...".cyan());

        // Try to load tokenizer from sibling file
        let tokenizer_info = AprModel::load_tokenizer_from_sibling(model_path);
        let (vocab, _bos_id, eos_id) = match &tokenizer_info {
            Some((v, b, e)) => {
                eprintln!(
                    "{}",
                    format!("Loaded tokenizer ({} tokens)", v.len()).dimmed()
                );
                (Some(v.as_slice()), *b, *e)
            }
            None => {
                eprintln!(
                    "{}",
                    "No tokenizer.json found. Using token IDs only.".yellow()
                );
                (None, None, Some(2u32)) // Default EOS
            }
        };

        // Get input tokens
        let input_tokens = if let Some(ref prompt) = options.prompt {
            // Parse comma-separated token IDs or encode text
            if prompt.contains(',') || prompt.chars().all(|c| c.is_ascii_digit() || c == ',') {
                parse_token_ids(prompt)?
            } else {
                // Text prompt - encode using tokenizer
                if let Some(tokens) = AprModel::encode_text(model_path, prompt) {
                    eprintln!(
                        "{}",
                        format!("Encoded {} chars to {} tokens", prompt.len(), tokens.len())
                            .dimmed()
                    );
                    tokens
                } else {
                    eprintln!(
                        "{}",
                        "Warning: No tokenizer found. Using BOS token.".yellow()
                    );
                    vec![1u32] // BOS token fallback
                }
            }
        } else if let Some(path) = input_path {
            let content = std::fs::read_to_string(path)?;
            parse_token_ids(&content)?
        } else {
            // Default: just use token ID 1 (BOS)
            vec![1u32]
        };

        let max_new_tokens = options.max_tokens;

        // Capture vocab_size before potential model move
        let vocab_size = model.metadata().vocab_size.unwrap_or(0);

        // Report backend selection
        let backend_label = if use_gpu { "GPU" } else { "CPU" };
        eprintln!(
            "{}",
            format!(
                "Generating {} tokens from {} input tokens ({} backend)...",
                max_new_tokens,
                input_tokens.len(),
                backend_label
            )
            .dimmed()
        );

        // Setup tracing if enabled (APR-TRACE-001)
        let mut tracer = if options.trace {
            use realizar::{InferenceTracer, ModelInfo, TraceConfig};

            let mut trace_config = TraceConfig::enabled();
            trace_config.verbose = options.trace_verbose;
            options.trace_output.clone_into(&mut trace_config.output);
            if let Some(ref steps) = options.trace_steps {
                trace_config.steps = TraceConfig::parse_steps(&steps.join(","));
            }

            let mut t = InferenceTracer::new(trace_config);
            t.set_model_info(ModelInfo {
                name: format!("APR Model ({})", architecture),
                num_layers: model.metadata().num_layers.unwrap_or(0),
                hidden_dim: model.metadata().hidden_size.unwrap_or(0),
                vocab_size,
                num_heads: model.metadata().num_heads.unwrap_or(0),
                quant_type: None,
            });

            // Trace ENCODE step
            t.start_step(realizar::TraceStep::Tokenize);
            t.trace_encode(
                options.prompt.as_deref().unwrap_or(""),
                &input_tokens,
                vocab_size,
            );

            Some(t)
        } else {
            None
        };

        // Run generation (GPU or CPU path)
        let infer_start = Instant::now();
        let output_tokens = if use_gpu {
            // GPU path via AprV2ModelCuda
            #[cfg(feature = "cuda")]
            {
                let mut cuda_model = realizar::apr::AprV2ModelCuda::new(model, 0)
                    .map_err(|e| CliError::ModelLoadFailed(format!("CUDA init failed: {e}")))?;
                eprintln!(
                    "{}",
                    format!(
                        "Using GPU: {} ({} MB VRAM)",
                        cuda_model.device_name(),
                        cuda_model.vram_mb()
                    )
                    .green()
                );
                cuda_model
                    .generate_cuda(&input_tokens, max_new_tokens, eos_id.unwrap_or(2))
                    .map_err(|e| CliError::InferenceFailed(format!("GPU generation failed: {e}")))?
            }
            #[cfg(not(feature = "cuda"))]
            {
                // Fallback to CPU (should not reach here due to use_gpu check)
                model
                    .generate(&input_tokens, max_new_tokens, eos_id)
                    .map_err(|e| CliError::InferenceFailed(format!("Generation failed: {e}")))?
            }
        } else {
            // CPU path
            model
                .generate(&input_tokens, max_new_tokens, eos_id)
                .map_err(|e| CliError::InferenceFailed(format!("Generation failed: {e}")))?
        };
        let infer_time = infer_start.elapsed();

        // Trace DECODE step for each generated token (APR-TRACE-001)
        if let Some(ref mut t) = tracer {
            let generated = &output_tokens[input_tokens.len()..];
            for (i, &token_id) in generated.iter().enumerate() {
                t.start_step(realizar::TraceStep::Decode);
                let decoded = vocab
                    .map(|v| realizar::apr::AprModel::decode_tokens(v, &[token_id]))
                    .unwrap_or_else(|| format!("<token_{}>", token_id));
                t.trace_decode(i + 1, token_id, &decoded, vocab_size);
            }

            // Output trace
            if let Err(e) = t.write_output() {
                eprintln!("Warning: Failed to write trace output: {e}");
            }
        }

        let new_tokens = output_tokens.len() - input_tokens.len();
        let tokens_per_sec = if infer_time.as_secs_f64() > 0.0 {
            new_tokens as f64 / infer_time.as_secs_f64()
        } else {
            0.0
        };

        let mut output = format!(
            "APR v2 Transformer Generation\n\
             Architecture: {}\n\
             Vocab size: {}\n\
             Input tokens: {:?}\n\
             Generated tokens: {} new ({} total)\n\
             Generation time: {:.2}ms ({:.1} tok/s)\n\n",
            architecture,
            vocab_size,
            input_tokens,
            new_tokens,
            output_tokens.len(),
            infer_time.as_secs_f64() * 1000.0,
            tokens_per_sec
        );

        // Decode and show output text if tokenizer available
        if let Some(v) = vocab {
            let generated_tokens = &output_tokens[input_tokens.len()..];
            let decoded_text = AprModel::decode_tokens(v, generated_tokens);
            output.push_str("Generated text:\n");
            output.push_str(&format!("  {}\n\n", decoded_text));
        }

        output.push_str("Output tokens:\n");
        output.push_str(&format!("  {:?}\n", output_tokens));

        // Show which tokens were generated (new ones)
        if new_tokens > 0 {
            output.push_str("\nGenerated token IDs:\n  ");
            for (i, &tok) in output_tokens.iter().skip(input_tokens.len()).enumerate() {
                if i > 0 {
                    output.push_str(", ");
                }
                output.push_str(&format!("{}", tok));
            }
            output.push('\n');
        }

        return Ok(output);
    }

    // Fallback: display metadata for non-transformer models
    let tensor_names = model.tensor_names();
    let mut output = format!(
        "APR v2 Model: {}\n\
         Architecture: {}\n\
         Tensors: {}\n\
         Load time: {:.2}ms\n\n",
        model_type,
        architecture,
        model.tensor_count(),
        load_time.as_secs_f64() * 1000.0
    );

    output.push_str("Available tensors:\n");
    for name in tensor_names.iter().take(20) {
        output.push_str(&format!("  - {name}\n"));
    }
    if tensor_names.len() > 20 {
        output.push_str(&format!("  ... and {} more\n", tensor_names.len() - 20));
    }

    output.push_str(
        "\nNote: Model missing transformer config. Add hidden_size, num_layers, num_heads, vocab_size to metadata.",
    );

    Ok(output)
}

/// Parse token IDs from input string (JSON array or comma-separated)
#[cfg(feature = "inference")]
fn parse_token_ids(input: &str) -> Result<Vec<u32>> {
    let trimmed = input.trim();
    if trimmed.starts_with('[') {
        // JSON array
        serde_json::from_str(trimmed)
            .map_err(|e| CliError::InvalidFormat(format!("Failed to parse token IDs: {e}")))
    } else {
        // Comma or space separated
        trimmed
            .split([',', ' ', '\n', '\t'])
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.trim()
                    .parse::<u32>()
                    .map_err(|e| CliError::InvalidFormat(format!("Invalid token ID: {s} - {e}")))
            })
            .collect()
    }
}

/// Clean model output by stripping ChatML markers and extra tokens
#[cfg(feature = "inference")]
fn clean_model_output(raw: &str) -> String {
    let mut cleaned = raw.to_string();
    // Strip ChatML markers commonly present in instruct model output
    let markers = [
        "<|im_start|>assistant\n",
        "<|im_start|>assistant",
        "<|im_end|>",
        "<|im_start|>",
        "<|endoftext|>",
    ];
    for marker in markers {
        cleaned = cleaned.replace(marker, "");
    }
    cleaned.trim().to_string()
}

/// Execute SafeTensors model inference
///
/// Uses realizar's safetensors support for transformer model generation with tracing.
/// Loads sibling config.json and tokenizer.json for full inference capability.
#[cfg(feature = "inference")]
fn execute_safetensors_inference(
    model_path: &Path,
    input_path: Option<&PathBuf>,
    options: &RunOptions,
) -> Result<String> {
    use realizar::apr::AprModel;
    use realizar::safetensors::{SafetensorsConfig, SafetensorsModel};
    use std::time::Instant;

    // Load SafeTensors file
    let start = Instant::now();
    let data = std::fs::read(model_path)?;
    let st_model = SafetensorsModel::from_bytes(&data)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load SafeTensors: {e}")))?;

    let tensor_count = st_model.tensors.len();
    let load_time = start.elapsed();

    eprintln!(
        "{}",
        format!(
            "Loaded SafeTensors model with {} tensors in {:.2}ms",
            tensor_count,
            load_time.as_secs_f64() * 1000.0
        )
        .dimmed()
    );

    // Try to load config.json for model architecture
    let config = SafetensorsConfig::load_from_sibling(model_path);
    let (hidden_size, num_layers, vocab_size, num_heads) = if let Some(ref cfg) = config {
        let architecture = cfg.architecture();
        eprintln!(
            "{}",
            format!(
                "Loaded config.json: {} (hidden={}, layers={}, vocab={})",
                architecture,
                cfg.hidden_size.unwrap_or(0),
                cfg.num_hidden_layers.unwrap_or(0),
                cfg.vocab_size.unwrap_or(0)
            )
            .dimmed()
        );
        (
            cfg.hidden_size.unwrap_or(0),
            cfg.num_hidden_layers.unwrap_or(0),
            cfg.vocab_size.unwrap_or(0),
            cfg.num_attention_heads.unwrap_or(0),
        )
    } else {
        eprintln!("{}", "No config.json found. Metadata-only mode.".yellow());
        (0, 0, 0, 0)
    };

    // Try to load tokenizer from sibling file (same as APR)
    let tokenizer_info = AprModel::load_tokenizer_from_sibling(model_path);
    let (vocab, eos_id) = match &tokenizer_info {
        Some((v, _bos, e)) => {
            eprintln!(
                "{}",
                format!("Loaded tokenizer.json ({} tokens)", v.len()).dimmed()
            );
            (Some(v.as_slice()), *e)
        }
        None => {
            eprintln!(
                "{}",
                "No tokenizer.json found. Using token IDs only.".yellow()
            );
            (None, Some(2u32)) // Default EOS
        }
    };

    // Setup tracing if enabled (APR-TRACE-001)
    let mut tracer = if options.trace {
        use realizar::{InferenceTracer, ModelInfo, TraceConfig};

        let mut trace_config = TraceConfig::enabled();
        trace_config.verbose = options.trace_verbose;
        options.trace_output.clone_into(&mut trace_config.output);
        if let Some(ref steps) = options.trace_steps {
            trace_config.steps = TraceConfig::parse_steps(&steps.join(","));
        }

        let mut t = InferenceTracer::new(trace_config);
        t.set_model_info(ModelInfo {
            name: format!(
                "SafeTensors Model ({})",
                config
                    .as_ref()
                    .map(realizar::SafetensorsConfig::architecture)
                    .unwrap_or_else(|| "unknown".to_string())
            ),
            num_layers,
            hidden_dim: hidden_size,
            vocab_size,
            num_heads,
            quant_type: None,
        });
        Some(t)
    } else {
        None
    };

    // Get input tokens
    let input_tokens = if let Some(ref prompt) = options.prompt {
        // Parse comma-separated token IDs or encode text
        if prompt.contains(',') || prompt.chars().all(|c| c.is_ascii_digit() || c == ',') {
            parse_token_ids(prompt)?
        } else {
            // Text prompt - encode using tokenizer
            if let Some(tokens) = AprModel::encode_text(model_path, prompt) {
                eprintln!(
                    "{}",
                    format!("Encoded {} chars to {} tokens", prompt.len(), tokens.len()).dimmed()
                );

                // Trace ENCODE step
                if let Some(ref mut t) = tracer {
                    t.start_step(realizar::TraceStep::Tokenize);
                    t.trace_encode(prompt, &tokens, vocab_size);
                }

                tokens
            } else {
                eprintln!(
                    "{}",
                    "Warning: No tokenizer found. Using BOS token.".yellow()
                );
                vec![1u32] // BOS token fallback
            }
        }
    } else if let Some(path) = input_path {
        let content = std::fs::read_to_string(path)?;
        parse_token_ids(&content)?
    } else {
        // Default: just use token ID 1 (BOS)
        vec![1u32]
    };

    // Check if we have config for inference
    if config.is_none() || vocab.is_none() {
        // Fallback: display metadata only (no generation without config/tokenizer)
        let tensor_names: Vec<&str> = st_model.tensor_names();
        let mut output = format!(
            "SafeTensors Model (metadata only)\n\
             Model: {}\n\
             Tensors: {}\n\
             Load time: {:.2}ms\n\n",
            model_path.display(),
            tensor_count,
            load_time.as_secs_f64() * 1000.0
        );

        if config.is_none() {
            output.push_str("Note: No config.json found - cannot run inference.\n");
            output.push_str("      Place config.json in the same directory as the model.\n\n");
            // F-JID-01 + F-AWS-05: Emit ExecutionFailed event with error and cause
            if let Some(ref mut t) = tracer {
                t.record_execution_failed("Initialization Failure", "Missing config.json");
            }
        }
        if vocab.is_none() {
            output.push_str("Note: No tokenizer.json found - cannot encode/decode text.\n");
            output.push_str("      Place tokenizer.json in the same directory as the model.\n\n");
        }

        output.push_str("Tensor names (first 15):\n");
        for (i, name) in tensor_names.iter().take(15).enumerate() {
            if let Some(info) = st_model.get_tensor_info(name) {
                output.push_str(&format!(
                    "  {}. {} ({:?}, {:?})\n",
                    i + 1,
                    name,
                    info.dtype,
                    info.shape
                ));
            }
        }

        if tensor_names.len() > 15 {
            output.push_str(&format!("  ... and {} more\n", tensor_names.len() - 15));
        }

        // Output trace if enabled
        if let Some(ref mut t) = tracer {
            if let Err(e) = t.write_output() {
                eprintln!("Warning: Failed to write trace output: {e}");
            }
        }

        return Ok(output);
    }

    // We have both config and tokenizer - run generation
    let cfg = config.expect("config verified above");
    let v = vocab.expect("vocab verified above");

    eprintln!("{}", "Running SafeTensors transformer generation...".cyan());
    eprintln!(
        "{}",
        format!(
            "Generating {} tokens from {} input tokens...",
            options.max_tokens,
            input_tokens.len()
        )
        .dimmed()
    );

    // Trace EMBED step
    if let Some(ref mut t) = tracer {
        t.start_step(realizar::TraceStep::Embed);
        t.trace_embed(input_tokens.len(), hidden_size, None);
    }

    // Run simplified generation (single forward pass for demonstration)
    // Full transformer inference would require implementing the full forward pass
    let infer_start = Instant::now();

    // For now, generate a simple output based on model inspection
    // Full generation would require: embedding lookup, attention, FFN, etc.
    let generated_tokens = run_safetensors_generation(
        &st_model,
        &cfg,
        &input_tokens,
        options.max_tokens,
        eos_id,
        &mut tracer,
    );

    let infer_time = infer_start.elapsed();

    // Trace DECODE step for generated tokens
    if let Some(ref mut t) = tracer {
        for (i, &token_id) in generated_tokens.iter().enumerate() {
            t.start_step(realizar::TraceStep::Decode);
            let decoded = AprModel::decode_tokens(v, &[token_id]);
            t.trace_decode(i + 1, token_id, &decoded, vocab_size);
        }

        // Output trace
        if let Err(e) = t.write_output() {
            eprintln!("Warning: Failed to write trace output: {e}");
        }
    }

    let new_tokens = generated_tokens.len();
    let tokens_per_sec = if infer_time.as_secs_f64() > 0.0 {
        new_tokens as f64 / infer_time.as_secs_f64()
    } else {
        0.0
    };

    let mut output = format!(
        "SafeTensors Transformer Generation\n\
         Architecture: {}\n\
         Hidden: {}, Layers: {}, Vocab: {}\n\
         Input tokens: {:?}\n\
         Generated tokens: {} ({:.1} tok/s)\n\
         Generation time: {:.2}ms\n\n",
        cfg.architecture(),
        hidden_size,
        num_layers,
        vocab_size,
        input_tokens,
        new_tokens,
        tokens_per_sec,
        infer_time.as_secs_f64() * 1000.0
    );

    // Decode and show output text
    let decoded_text = AprModel::decode_tokens(v, &generated_tokens);
    output.push_str("Generated text:\n");
    output.push_str(&format!("  {}\n\n", clean_model_output(&decoded_text)));

    output.push_str("Generated token IDs:\n");
    output.push_str(&format!("  {:?}\n", generated_tokens));

    Ok(output)
}

/// Find embedding tensor name in SafeTensors model
#[cfg(feature = "inference")]
fn find_embedding_tensor(model: &realizar::safetensors::SafetensorsModel) -> Option<&str> {
    // Common embedding tensor names across different model architectures
    let candidates = [
        "model.embed_tokens.weight",
        "transformer.wte.weight",
        "embeddings.word_embeddings.weight",
        "embed_tokens.weight",
        "wte.weight",
        "token_embedding.weight",
    ];

    candidates
        .into_iter()
        .find(|&name| model.has_tensor(name))
        .map(|v| v as _)
}

/// Run simplified SafeTensors generation
///
/// This is a placeholder that demonstrates the tracing flow.
/// Full inference would require implementing the complete transformer forward pass.
#[cfg(feature = "inference")]
fn run_safetensors_generation(
    model: &realizar::safetensors::SafetensorsModel,
    config: &realizar::safetensors::SafetensorsConfig,
    input_tokens: &[u32],
    max_tokens: usize,
    eos_id: Option<u32>,
    tracer: &mut Option<realizar::InferenceTracer>,
) -> Vec<u32> {
    let vocab_size = config.vocab_size.unwrap_or(32000);
    let num_layers = config.num_hidden_layers.unwrap_or(0);
    let hidden_size = config.hidden_size.unwrap_or(0);

    let mut generated = Vec::new();
    let eos = eos_id.unwrap_or(2);

    // Create placeholder logits for tracing (in real impl, would be computed)
    let placeholder_logits: Vec<f32> = vec![0.0; vocab_size];

    // For demonstration: trace transformer layers and generate placeholder tokens
    // In a real implementation, this would run the actual transformer forward pass
    for i in 0..max_tokens.min(16) {
        // Trace TRANSFORMER step (simulated)
        if let Some(ref mut t) = tracer {
            t.start_step(realizar::TraceStep::TransformerBlock);
            t.trace_layer(
                num_layers.saturating_sub(1), // Last layer
                i,
                None, // No actual hidden state values
                1,    // seq_len
                hidden_size,
            );
        }

        // Generate token (placeholder - real impl would use logits)
        // For now, copy input pattern or generate based on tensor inspection
        let token = if i < input_tokens.len() {
            // Echo input during "prefill" phase
            input_tokens[i]
        } else {
            // Placeholder generation
            let last_input = input_tokens.last().copied().unwrap_or(1);
            // Simple pattern: increment token ID (bounded by vocab)
            (last_input.wrapping_add(i as u32)) % (vocab_size as u32)
        };

        // Trace LM_HEAD step
        if let Some(ref mut t) = tracer {
            t.start_step(realizar::TraceStep::LmHead);
            t.trace_lm_head(i, &placeholder_logits, vocab_size);
        }

        // Trace SAMPLE step
        if let Some(ref mut t) = tracer {
            t.start_step(realizar::TraceStep::Sample);
            t.trace_sample(i, &placeholder_logits, token, 0.0, 1);
        }

        // Check for EOS
        if token == eos {
            break;
        }

        generated.push(token);
    }

    // Add note that this is demo output
    if !model.tensors.is_empty() && tracer.is_some() {
        eprintln!(
            "{}",
            "Note: SafeTensors generation is in demo mode (tracing enabled).".yellow()
        );
    }

    generated
}

/// Execute GGUF model inspection
///
/// Execute GGUF model inference using realizar's optimized OwnedQuantizedModel.
///
/// Uses quantized compute for better performance than naive dequantize-then-compute.
#[cfg(feature = "inference")]
fn execute_gguf_inference(
    model_path: &Path,
    input_path: Option<&PathBuf>,
    options: &RunOptions,
) -> Result<String> {
    use realizar::chat_template::{format_messages, ChatMessage};
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
    use std::time::Instant;

    // Load GGUF model via memory mapping
    let start = Instant::now();
    let mapped_model = MappedGGUFModel::from_path(model_path)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load GGUF model: {e}")))?;
    let mmap_time = start.elapsed();

    // Pre-fault all mmap pages to avoid page faults during model load/inference (PAR-200: B4 CPU perf fix)
    // Without this, OwnedQuantizedModel::from_mapped() triggers ~9M minor page faults = 2.5s overhead!
    {
        let prefault_start = Instant::now();
        let data = mapped_model.data();
        let page_size = 4096;
        let mut checksum: u8 = 0;
        // Touch one byte per page to force kernel to fault in the page
        for i in (0..data.len()).step_by(page_size) {
            checksum = checksum.wrapping_add(data[i]);
        }
        // Use checksum to prevent dead code elimination
        std::hint::black_box(checksum);
        // Debug timing (can be removed in production)
        let _ = (data.len().div_ceil(page_size), prefault_start.elapsed());
    }

    // Try to create optimized quantized model
    let load_start = Instant::now();
    let model_result = OwnedQuantizedModel::from_mapped(&mapped_model);
    let _load_time = load_start.elapsed();

    match model_result {
        Ok(model) => {
            // Get input tokens - use GGUF's embedded tokenizer
            let input_tokens = if let Some(ref prompt) = options.prompt {
                // Parse comma-separated token IDs or encode text
                if prompt.contains(',') || prompt.chars().all(|c| c.is_ascii_digit() || c == ',') {
                    parse_token_ids(prompt)?
                } else {
                    // Text prompt - apply chat template for instruct models, then encode
                    // Detect instruct model from filename (e.g., "qwen2-0_5b-instruct-q4_0.gguf")
                    let model_name = model_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("");
                    let is_instruct = model_name.to_lowercase().contains("instruct");

                    let formatted_prompt = if is_instruct {
                        // Apply ChatML template for instruct models
                        let messages = vec![ChatMessage::user(prompt)];
                        format_messages(&messages, Some(model_name))
                            .unwrap_or_else(|_| prompt.clone())
                    } else {
                        prompt.clone()
                    };

                    // F-UX-40: Debug output only in trace/verbose mode (NOISY-GUARD)
                    if options.trace || options.verbose {
                        eprintln!(
                            "[APR-TRACE] Model: {} (instruct={})",
                            model_name, is_instruct
                        );
                        eprintln!("[APR-TRACE] Formatted prompt: {:?}", formatted_prompt);
                    }

                    let tokens = mapped_model.model.encode(&formatted_prompt);
                    // F-UX-40: Debug output only in trace/verbose mode
                    if options.trace || options.verbose {
                        eprintln!(
                            "[APR-TRACE] encode returned: {:?}",
                            tokens.as_ref().map(std::vec::Vec::len)
                        );
                    }
                    tokens.unwrap_or_else(|| vec![1u32])
                }
            } else if let Some(path) = input_path {
                let content = std::fs::read_to_string(path)?;
                parse_token_ids(&content)?
            } else {
                vec![1u32] // BOS token
            };

            let max_new_tokens = options.max_tokens;
            // PAR-200: Use greedy sampling for GPU argmax path (faster + deterministic)
            let gen_config = QuantizedGenerateConfig {
                max_tokens: max_new_tokens.min(128),
                temperature: 0.0,     // Greedy sampling for GPU argmax
                top_k: 1,             // Force argmax path
                trace: options.trace, // PMAT-TRACE-GGUF-001: Pass trace flag
                ..Default::default()
            };

            // Create decode function for tracing (APR-TRACE-001)
            let decode_fn = |token_id: u32| -> String { mapped_model.model.decode(&[token_id]) };

            // PAR-200: Use GPU-resident path for 20x faster inference (116 tok/s vs 5.7 tok/s)
            // APR-TRACE-001: Pass trace options for traced generation when --trace is enabled
            let trace_opts = if options.trace { Some(options) } else { None };
            let gen_result = run_gguf_generate(
                model,
                &input_tokens,
                &gen_config,
                options.no_gpu,
                options.benchmark,
                trace_opts,
                Some(&decode_fn),
            )?;

            // Show inference-only performance (excludes loading time)
            if options.benchmark {
                let new_tokens = gen_result.tokens.len().saturating_sub(input_tokens.len());
                let tok_per_sec = if gen_result.inference_ms > 0.0 {
                    new_tokens as f64 / (gen_result.inference_ms / 1000.0)
                } else {
                    0.0
                };
                eprintln!(
                    "Inference: {} tokens in {:.1}ms ({:.1} tok/s)",
                    new_tokens, gen_result.inference_ms, tok_per_sec
                );
            }

            // Decode output using GGUF's embedded tokenizer - only new tokens
            let generated_tokens = &gen_result.tokens[input_tokens.len()..];
            let decoded_text = mapped_model.model.decode(generated_tokens);

            // Clean output: strip ChatML markers for instruct models
            let cleaned = clean_model_output(&decoded_text);
            Ok(cleaned)
        }
        Err(e) => {
            // Fallback to metadata display
            let model = &mapped_model.model;
            let mut output = format!(
                "GGUF Model (quantized inference unavailable)\n\
                 Model: {}\n\
                 Load error: {}\n\
                 GGUF Version: {}\n\
                 Tensors: {}\n\
                 Metadata entries: {}\n\n",
                model_path.display(),
                e,
                model.header.version,
                model.tensors.len(),
                model.metadata.len()
            );

            output.push_str("Metadata (first 10):\n");
            for (i, (key, _)) in model.metadata.iter().take(10).enumerate() {
                output.push_str(&format!("  {}. {}\n", i + 1, key));
            }
            if model.metadata.len() > 10 {
                output.push_str(&format!("  ... and {} more\n", model.metadata.len() - 10));
            }

            output.push_str("\nTensors (first 10):\n");
            for (i, tensor) in model.tensors.iter().take(10).enumerate() {
                output.push_str(&format!(
                    "  {}. {} (type: {}, dims: {:?})\n",
                    i + 1,
                    tensor.name,
                    tensor.qtype,
                    tensor.dims
                ));
            }
            if model.tensors.len() > 10 {
                output.push_str(&format!("  ... and {} more\n", model.tensors.len() - 10));
            }

            Ok(output)
        }
    }
}

/// Result from GGUF generation including timing
#[cfg(feature = "inference")]
struct GgufGenerateResult {
    tokens: Vec<u32>,
    inference_ms: f64,
}

/// Run GGUF generation with GPU-resident path for optimal performance (PAR-200)
/// Supports inference tracing when `trace_options` is provided (APR-TRACE-001)
#[cfg(feature = "inference")]
#[allow(clippy::too_many_arguments)]
fn run_gguf_generate(
    model: realizar::gguf::OwnedQuantizedModel,
    input_tokens: &[u32],
    gen_config: &realizar::gguf::QuantizedGenerateConfig,
    no_gpu: bool,
    benchmark: bool,
    trace_options: Option<&RunOptions>,
    decode_fn: Option<&dyn Fn(u32) -> String>,
) -> Result<GgufGenerateResult> {
    #[cfg(feature = "cuda")]
    if !no_gpu {
        use realizar::gguf::OwnedQuantizedModelCuda;
        // F-UX-40/F-UX-26: Only show CUDA init in verbose/benchmark mode (NOISY-GUARD)
        let verbose = trace_options.is_some_and(|o| o.verbose);
        if verbose || benchmark {
            eprintln!("Initializing CUDA GPU 0 (GPU-resident mode)...");
        }
        let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0)
            .map_err(|e| CliError::InferenceFailed(format!("CUDA init failed: {e}")))?;

        // Warmup for CUDA graphs (critical for accurate timing)
        if benchmark {
            eprintln!("Warmup (3 iterations)...");
            for _ in 0..3 {
                let _ = cuda_model.generate_gpu_resident(input_tokens, gen_config);
            }
        }

        // Check if tracing is enabled (APR-TRACE-001)
        let trace_enabled = trace_options.is_some_and(|o| o.trace);

        // Measure inference time separately from loading
        let infer_start = Instant::now();

        let tokens = if trace_enabled {
            // GPU path with tracing (APR-TRACE-001: F-HW-04-B CUDA Graph parity)
            use realizar::{InferenceTracer, ModelInfo, TraceConfig};

            let opts = trace_options.expect("trace_options must be Some when trace_enabled");
            let mut trace_config = TraceConfig::enabled();
            trace_config.verbose = opts.trace_verbose;
            trace_config.output = opts.trace_output.clone();
            if let Some(ref steps) = opts.trace_steps {
                trace_config.steps = TraceConfig::parse_steps(&steps.join(","));
            }

            let mut tracer = InferenceTracer::new(trace_config);
            tracer.set_model_info(ModelInfo {
                name: "GGUF Model (GPU)".to_string(),
                num_layers: cuda_model.model().config.num_layers,
                hidden_dim: cuda_model.model().config.hidden_dim,
                vocab_size: cuda_model.model().config.vocab_size,
                num_heads: cuda_model.model().config.num_heads,
                quant_type: None,
            });

            // PMAT-TRACE-GGUF-001: GPU tracing via gen_config.trace flag
            // The gen_config.trace flag is already set for [TRACE-CACHE] output
            let result = cuda_model
                .generate_gpu_resident(input_tokens, gen_config)
                .map_err(|e| CliError::InferenceFailed(format!("GPU generation failed: {e}")))?;

            // Write InferenceTracer output (model info summary)
            if let Err(e) = tracer.write_output() {
                eprintln!("Warning: Failed to write trace output: {e}");
            }

            result
        } else {
            // GPU path without tracing (fast path)
            cuda_model
                .generate_gpu_resident(input_tokens, gen_config)
                .map_err(|e| CliError::InferenceFailed(format!("GPU generation failed: {e}")))?
        };

        let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

        return Ok(GgufGenerateResult {
            tokens,
            inference_ms,
        });
    }

    // CPU fallback - with optional tracing (APR-TRACE-001)
    #[allow(unused_variables)]
    let _ = benchmark; // Used only in CUDA path for warmup
    let infer_start = Instant::now();
    let cpu_model = model;

    // Check if tracing is enabled
    let trace_enabled = trace_options.is_some_and(|o| o.trace);

    let tokens = if trace_enabled {
        // Use traced generation path (APR-TRACE-001)
        use realizar::{InferenceTracer, ModelInfo, TraceConfig};

        let opts = trace_options.expect("trace_options must be Some when trace_enabled");
        let mut trace_config = TraceConfig::enabled();
        trace_config.verbose = opts.trace_verbose;
        trace_config.output = opts.trace_output.clone();
        if let Some(ref steps) = opts.trace_steps {
            trace_config.steps = TraceConfig::parse_steps(&steps.join(","));
        }

        let mut tracer = InferenceTracer::new(trace_config);
        tracer.set_model_info(ModelInfo {
            name: "GGUF Model".to_string(),
            num_layers: cpu_model.config.num_layers,
            hidden_dim: cpu_model.config.hidden_dim,
            vocab_size: cpu_model.config.vocab_size,
            num_heads: cpu_model.config.num_heads,
            quant_type: None,
        });

        // PMAT-TRACE-GGUF-001: CPU traced generation now implemented in realizar
        // The gen_config.trace flag is already set, so generate_with_cache outputs [TRACE-CACHE] messages
        let result = cpu_model
            .generate_with_cache(input_tokens, gen_config)
            .map_err(|e| CliError::InferenceFailed(format!("CPU generation failed: {e}")))?;

        // Write InferenceTracer output (model info summary)
        if let Err(e) = tracer.write_output() {
            eprintln!("Warning: Failed to write trace output: {e}");
        }

        result
    } else {
        cpu_model
            .generate_with_cache(input_tokens, gen_config)
            .map_err(|e| CliError::InferenceFailed(format!("Generation failed: {e}")))?
    };

    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

    Ok(GgufGenerateResult {
        tokens,
        inference_ms,
    })
}

/// Parse input features from file or stdin
#[cfg(feature = "inference")]
fn parse_input_features(input_path: Option<&PathBuf>) -> Result<Vec<f32>> {
    let input_text = if let Some(path) = input_path {
        std::fs::read_to_string(path)?
    } else {
        // Read from stdin
        use std::io::Read;
        let mut buffer = String::new();
        std::io::stdin().read_to_string(&mut buffer)?;
        buffer
    };

    // Parse as JSON array or comma-separated values
    if input_text.trim().starts_with('[') {
        // JSON array
        serde_json::from_str(&input_text)
            .map_err(|e| CliError::InvalidFormat(format!("Failed to parse JSON input: {e}")))
    } else {
        // CSV or space-separated
        input_text
            .split([',', ' ', '\n', '\t'])
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.trim()
                    .parse::<f32>()
                    .map_err(|e| CliError::InvalidFormat(format!("Invalid float: {s} - {e}")))
            })
            .collect()
    }
}

/// Format prediction output based on options
#[cfg(feature = "inference")]
fn format_prediction_output(
    output: &[f32],
    inference_time: std::time::Duration,
    options: &RunOptions,
) -> Result<String> {
    let inference_ms = inference_time.as_secs_f64() * 1000.0;

    match options.output_format.as_str() {
        "json" => {
            let result = serde_json::json!({
                "predictions": output,
                "inference_time_ms": inference_ms
            });
            serde_json::to_string_pretty(&result)
                .map_err(|e| CliError::InvalidFormat(format!("JSON serialization failed: {e}")))
        }
        _ => {
            // Default text format
            let mut result = String::new();
            result.push_str("Predictions:\n");
            for (i, &val) in output.iter().enumerate() {
                result.push_str(&format!("  [{}]: {:.6}\n", i, val));
            }
            result.push_str(&format!("\nInference time: {:.2}ms", inference_ms));
            Ok(result)
        }
    }
}

/// Run command entry point
///
/// Per Section 9.2 (Sovereign AI), the `offline` flag enforces strict network isolation:
/// - When `true`, all network access is blocked at the type level
/// - Production deployments MUST use `--offline` mode
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    source: &str,
    input: Option<&Path>,
    prompt: Option<&str>,
    max_tokens: usize,
    stream: bool,
    _language: Option<&str>,
    _task: Option<&str>,
    output_format: &str,
    no_gpu: bool,
    offline: bool,
    benchmark: bool,
    verbose: bool,
    trace: bool,
    trace_steps: Option<&[String]>,
    trace_verbose: bool,
    trace_output: Option<PathBuf>,
) -> Result<()> {
    if offline {
        println!("{}", "=== APR Run (OFFLINE MODE) ===".cyan().bold());
        eprintln!(
            "{}",
            "Network access disabled. Only local/cached models allowed.".yellow()
        );
    } else {
        println!("{}", "=== APR Run ===".cyan().bold());
    }
    println!();
    println!("Source: {source}");

    // Setup trace config if tracing enabled (APR-TRACE-001)
    if trace {
        eprintln!("{}", "Inference tracing enabled (APR-TRACE-001)".cyan());
        if let Some(steps) = trace_steps {
            eprintln!("  Trace steps: {}", steps.join(", "));
        }
        if trace_verbose {
            eprintln!("  Verbose mode enabled");
        }
        if let Some(ref path) = trace_output {
            eprintln!("  Output: {}", path.display());
        }
    }

    let options = RunOptions {
        input: input.map(Path::to_path_buf),
        prompt: prompt.map(String::from),
        max_tokens,
        output_format: output_format.to_string(),
        force: false,
        no_gpu,
        offline,
        benchmark,
        verbose,
        trace,
        trace_steps: trace_steps.map(<[std::string::String]>::to_vec),
        trace_verbose,
        trace_output,
    };

    let result = run_model(source, &options)?;

    if benchmark {
        // Benchmark mode - output performance metrics
        let tokens_generated = result.tokens_generated.unwrap_or(max_tokens);
        let tok_per_sec = if result.duration_secs > 0.0 {
            tokens_generated as f64 / result.duration_secs
        } else {
            0.0
        };

        println!();
        println!("{}", "=== Benchmark Results ===".cyan().bold());
        println!("tok/s: {:.1}", tok_per_sec);
        println!("tokens: {}", tokens_generated);
        println!("latency: {:.2}ms", result.duration_secs * 1000.0);
        println!("model: {}", source);
        println!();

        // Clean output for parsing
        if output_format == "json" {
            println!(
                r#"{{"tok_s": {:.1}, "tokens": {}, "latency_ms": {:.2}}}"#,
                tok_per_sec,
                tokens_generated,
                result.duration_secs * 1000.0
            );
        }
    } else if stream {
        // Streaming mode - output token by token
        for word in result.text.split_whitespace() {
            print!("{word} ");
            std::io::Write::flush(&mut std::io::stdout())?;
        }
        println!();
    } else {
        // Batch mode - output all at once
        println!();
        println!("{}", "Output:".green().bold());
        println!("{}", result.text);
    }

    if !benchmark {
        println!();
        println!(
            "Completed in {:.2}s {}",
            result.duration_secs,
            if result.cached {
                "(cached)".dimmed()
            } else {
                "(downloaded)".dimmed()
            }
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== ModelSource Tests ====================

    #[test]
    fn test_parse_local_path() {
        let source = ModelSource::parse("model.apr").unwrap();
        assert_eq!(source, ModelSource::Local(PathBuf::from("model.apr")));
    }

    #[test]
    fn test_parse_absolute_path() {
        let source = ModelSource::parse("/path/to/model.apr").unwrap();
        assert_eq!(
            source,
            ModelSource::Local(PathBuf::from("/path/to/model.apr"))
        );
    }

    #[test]
    fn test_parse_huggingface_source() {
        let source = ModelSource::parse("hf://openai/whisper-tiny").unwrap();
        assert_eq!(
            source,
            ModelSource::HuggingFace {
                org: "openai".to_string(),
                repo: "whisper-tiny".to_string(),
                file: None,
            }
        );
    }

    #[test]
    fn test_parse_huggingface_with_file() {
        let source =
            ModelSource::parse("hf://Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/model-q4_k_m.gguf")
                .unwrap();
        assert_eq!(
            source,
            ModelSource::HuggingFace {
                org: "Qwen".to_string(),
                repo: "Qwen2.5-Coder-0.5B-Instruct-GGUF".to_string(),
                file: Some("model-q4_k_m.gguf".to_string()),
            }
        );
    }

    #[test]
    fn test_parse_huggingface_invalid() {
        let result = ModelSource::parse("hf://invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_https_url() {
        let source = ModelSource::parse("https://example.com/model.apr").unwrap();
        assert_eq!(
            source,
            ModelSource::Url("https://example.com/model.apr".to_string())
        );
    }

    #[test]
    fn test_parse_http_url() {
        let source = ModelSource::parse("http://example.com/model.apr").unwrap();
        assert_eq!(
            source,
            ModelSource::Url("http://example.com/model.apr".to_string())
        );
    }

    // ==================== Cache Path Tests ====================

    #[test]
    fn test_cache_path_local() {
        let source = ModelSource::Local(PathBuf::from("/tmp/model.apr"));
        assert_eq!(source.cache_path(), PathBuf::from("/tmp/model.apr"));
    }

    #[test]
    fn test_cache_path_huggingface() {
        let source = ModelSource::HuggingFace {
            org: "openai".to_string(),
            repo: "whisper-tiny".to_string(),
            file: None,
        };
        let cache = source.cache_path();
        assert!(cache.to_string_lossy().contains("hf"));
        assert!(cache.to_string_lossy().contains("openai"));
        assert!(cache.to_string_lossy().contains("whisper-tiny"));
    }

    #[test]
    fn test_cache_path_url_deterministic() {
        let source1 = ModelSource::Url("https://example.com/model.apr".to_string());
        let source2 = ModelSource::Url("https://example.com/model.apr".to_string());
        assert_eq!(source1.cache_path(), source2.cache_path());
    }

    #[test]
    fn test_cache_path_url_different() {
        let source1 = ModelSource::Url("https://example.com/model1.apr".to_string());
        let source2 = ModelSource::Url("https://example.com/model2.apr".to_string());
        assert_ne!(source1.cache_path(), source2.cache_path());
    }

    // ==================== RunOptions Tests ====================

    #[test]
    fn test_run_options_default() {
        let options = RunOptions::default();
        assert!(options.input.is_none());
        assert_eq!(options.output_format, "text");
        assert!(!options.force);
        assert!(!options.no_gpu);
        assert!(!options.offline);
    }

    // ==================== MD5 Hash Tests ====================

    #[test]
    fn test_md5_hash_deterministic() {
        let hash1 = md5_hash(b"test");
        let hash2 = md5_hash(b"test");
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_md5_hash_different_inputs() {
        let hash1 = md5_hash(b"test1");
        let hash2 = md5_hash(b"test2");
        assert_ne!(hash1, hash2);
    }

    // ==================== Integration Tests ====================

    #[test]
    fn test_run_model_file_not_found() {
        let result = run_model("/nonexistent/model.apr", &RunOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_run_model_with_options() {
        let options = RunOptions {
            input: Some(PathBuf::from("/tmp/test.wav")),
            prompt: None,
            max_tokens: 32,
            output_format: "json".to_string(),
            force: false,
            no_gpu: true,
            offline: false,
            benchmark: false,
            verbose: false,
            trace: false,
            trace_steps: None,
            trace_verbose: false,
            trace_output: None,
        };
        assert!(options.no_gpu);
        assert_eq!(options.output_format, "json");
    }

    // ============================================================================
    // Popperian Falsification Tests: Offline Mode (Section 9.2 Sovereign AI)
    // ============================================================================
    //
    // Per PMAT Extreme TDD: Each test defines conditions under which the claim
    // would be **proven false**.

    /// FALSIFICATION: If --offline allows HuggingFace download, the claim fails
    /// Claim: `apr run --offline hf://org/repo` rejects non-cached HF models
    #[test]
    fn offline_mode_rejects_uncached_huggingface() {
        let source = ModelSource::HuggingFace {
            org: "uncached-org".to_string(),
            repo: "nonexistent-repo".to_string(),
            file: None,
        };

        // Offline mode MUST reject non-cached HF sources
        let result = resolve_model(&source, false, true);

        assert!(result.is_err(), "FALSIFIED: Offline mode allowed HF source");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("OFFLINE MODE"),
            "FALSIFIED: Error message should mention OFFLINE MODE, got: {err}"
        );
    }

    /// FALSIFICATION: If --offline allows URL download, the claim fails
    /// Claim: `apr run --offline https://...` rejects non-cached URLs
    #[test]
    fn offline_mode_rejects_uncached_url() {
        let source = ModelSource::Url("https://example.com/model.apr".to_string());

        // Offline mode MUST reject non-cached URL sources
        let result = resolve_model(&source, false, true);

        assert!(
            result.is_err(),
            "FALSIFIED: Offline mode allowed URL source"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("OFFLINE MODE"),
            "FALSIFIED: Error message should mention OFFLINE MODE, got: {err}"
        );
    }

    /// FALSIFICATION: If --offline rejects local files, the claim fails
    /// Claim: `apr run --offline /path/to/model.apr` allows local files
    #[test]
    fn offline_mode_allows_local_files() {
        let source = ModelSource::Local(PathBuf::from("/tmp/model.apr"));

        // Offline mode MUST allow local file sources
        let result = resolve_model(&source, false, true);

        // Note: This succeeds at resolution, but may fail later if file doesn't exist
        // The key point is that offline mode doesn't reject local sources
        assert!(
            result.is_ok(),
            "FALSIFIED: Offline mode rejected local file source: {:?}",
            result
        );
    }

    /// FALSIFICATION: If default mode has offline=true, the claim fails
    /// Claim: Default RunOptions have offline=false
    #[test]
    fn default_options_are_not_offline() {
        let options = RunOptions::default();
        assert!(
            !options.offline,
            "FALSIFIED: Default options should NOT be offline"
        );
    }

    /// FALSIFICATION: If offline flag doesn't propagate, the claim fails
    /// Claim: RunOptions::offline is correctly set when specified
    #[test]
    fn offline_flag_propagates_correctly() {
        let options = RunOptions {
            offline: true,
            ..Default::default()
        };
        assert!(
            options.offline,
            "FALSIFIED: Offline flag did not propagate to options"
        );
    }

    // ==================== Sharded Model Tests (GH-127) ====================

    /// Test extract_shard_files with typical HuggingFace index.json format
    #[test]
    fn test_extract_shard_files_basic() {
        let json = r#"{
            "metadata": {"total_size": 1000000},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
                "model.layers.0.weight": "model-00001-of-00003.safetensors",
                "model.layers.1.weight": "model-00002-of-00003.safetensors",
                "model.layers.2.weight": "model-00003-of-00003.safetensors",
                "lm_head.weight": "model-00003-of-00003.safetensors"
            }
        }"#;

        let files = extract_shard_files(json);

        assert_eq!(files.len(), 3, "Should extract 3 unique shard files");
        assert!(files.contains("model-00001-of-00003.safetensors"));
        assert!(files.contains("model-00002-of-00003.safetensors"));
        assert!(files.contains("model-00003-of-00003.safetensors"));
    }

    /// Test extract_shard_files with empty weight_map
    #[test]
    fn test_extract_shard_files_empty() {
        let json = r#"{"weight_map": {}}"#;
        let files = extract_shard_files(json);
        assert!(files.is_empty(), "Empty weight_map should yield no files");
    }

    /// Test extract_shard_files with no weight_map key
    #[test]
    fn test_extract_shard_files_no_weight_map() {
        let json = r#"{"metadata": {}}"#;
        let files = extract_shard_files(json);
        assert!(files.is_empty(), "Missing weight_map should yield no files");
    }

    /// Test extract_shard_files with single shard (all tensors in one file)
    #[test]
    fn test_extract_shard_files_single_shard() {
        let json = r#"{
            "weight_map": {
                "a": "model.safetensors",
                "b": "model.safetensors",
                "c": "model.safetensors"
            }
        }"#;

        let files = extract_shard_files(json);

        assert_eq!(
            files.len(),
            1,
            "All tensors in same file should yield 1 shard"
        );
        assert!(files.contains("model.safetensors"));
    }

    /// Test extract_shard_files handles real-world Phi-4 style index
    #[test]
    fn test_extract_shard_files_phi4_style() {
        // Simplified version of microsoft/phi-4 index structure
        let json = r#"{
            "metadata": {
                "total_size": 56000000000
            },
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00006.safetensors",
                "model.layers.0.input_layernorm.weight": "model-00001-of-00006.safetensors",
                "model.layers.10.mlp.up_proj.weight": "model-00002-of-00006.safetensors",
                "model.layers.20.self_attn.v_proj.weight": "model-00003-of-00006.safetensors",
                "model.layers.30.post_attention_layernorm.weight": "model-00004-of-00006.safetensors",
                "model.layers.40.mlp.gate_proj.weight": "model-00005-of-00006.safetensors",
                "model.norm.weight": "model-00006-of-00006.safetensors",
                "lm_head.weight": "model-00006-of-00006.safetensors"
            }
        }"#;

        let files = extract_shard_files(json);

        assert_eq!(files.len(), 6, "Phi-4 style model has 6 shards");
        for i in 1..=6 {
            let expected = format!("model-{i:05}-of-00006.safetensors");
            assert!(
                files.contains(&expected),
                "Should contain shard file: {expected}"
            );
        }
    }

    /// Test that non-safetensors files are filtered out
    #[test]
    fn test_extract_shard_files_filters_non_safetensors() {
        let json = r#"{
            "weight_map": {
                "a": "model.safetensors",
                "b": "config.json",
                "c": "tokenizer.model"
            }
        }"#;

        let files = extract_shard_files(json);

        assert_eq!(files.len(), 1, "Should only include .safetensors files");
        assert!(files.contains("model.safetensors"));
        assert!(!files.contains("config.json"));
        assert!(!files.contains("tokenizer.model"));
    }
}
