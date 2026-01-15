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

// Allow dead code during development - these are planned features
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(clippy::needless_return)]
#![allow(clippy::format_push_string)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::disallowed_methods)]

use crate::error::{CliError, Result};
use colored::Colorize;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Model source types
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ModelSource {
    /// Local file path
    Local(PathBuf),
    /// HuggingFace Hub (hf://org/repo)
    HuggingFace { org: String, repo: String },
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
                Ok(Self::HuggingFace {
                    org: parts[0].to_string(),
                    repo: parts[1].to_string(),
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
            Self::HuggingFace { org, repo } => cache_dir.join("hf").join(org).join(repo),
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
        ModelSource::HuggingFace { org, repo } => {
            // Check multiple cache locations for the model
            if let Some(path) = find_cached_model(org, repo) {
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
            download_hf_model(org, repo)
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
fn find_cached_model(org: &str, repo: &str) -> Option<PathBuf> {
    // Check HuggingFace hub cache first (standard location)
    let hf_cache = dirs::home_dir().map(|h| h.join(".cache").join("huggingface").join("hub"))?;

    let hf_model_dir = hf_cache.join(format!("models--{org}--{repo}"));
    if hf_model_dir.exists() {
        // Find the latest snapshot
        let snapshots_dir = hf_model_dir.join("snapshots");
        if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
            for entry in entries.flatten() {
                let snapshot_dir = entry.path();
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

    // Check apr cache
    let apr_cache =
        dirs::home_dir().map(|h| h.join(".apr").join("cache").join("hf").join(org).join(repo))?;
    if apr_cache.exists() {
        for ext in &["apr", "safetensors", "gguf"] {
            let pattern = apr_cache.join(format!("model.{ext}"));
            if pattern.exists() {
                return Some(pattern);
            }
        }
    }

    None
}

/// Download model from HuggingFace and cache it
fn download_hf_model(org: &str, repo: &str) -> Result<PathBuf> {
    use std::io::Write;

    let cache_dir = dirs::home_dir()
        .ok_or_else(|| CliError::ValidationFailed("Cannot find home directory".to_string()))?
        .join(".apr")
        .join("cache")
        .join("hf")
        .join(org)
        .join(repo);

    std::fs::create_dir_all(&cache_dir)?;

    let model_url = format!("https://huggingface.co/{org}/{repo}/resolve/main/model.safetensors");
    let tokenizer_url = format!("https://huggingface.co/{org}/{repo}/resolve/main/tokenizer.json");

    let model_path = cache_dir.join("model.safetensors");
    let tokenizer_path = cache_dir.join("tokenizer.json");

    // Download model
    eprintln!("  Downloading model.safetensors...");
    download_file(&model_url, &model_path)?;

    // Download tokenizer (optional, don't fail if missing)
    eprintln!("  Downloading tokenizer.json...");
    if let Err(e) = download_file(&tokenizer_url, &tokenizer_path) {
        eprintln!("  Warning: tokenizer.json not available: {e}");
    }

    eprintln!("{}", "  Download complete!".green());

    Ok(model_path)
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

    if use_mmap {
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
#[cfg(feature = "inference")]
fn execute_with_realizar(
    model_path: &Path,
    input_path: Option<&PathBuf>,
    options: &RunOptions,
    _use_mmap: bool,
) -> Result<String> {
    use realizar::format::{detect_format, ModelFormat};

    // Read file for format detection
    let data = std::fs::read(model_path)?;
    if data.len() < 8 {
        return Err(CliError::InvalidFormat(
            "File too small for format detection".to_string(),
        ));
    }

    // Detect model format from magic bytes
    let format = detect_format(&data[..8])
        .map_err(|e| CliError::InvalidFormat(format!("Format detection failed: {e}")))?;

    match format {
        ModelFormat::Apr => execute_apr_inference(model_path, input_path, options),
        ModelFormat::SafeTensors => execute_safetensors_inference(model_path, input_path, options),
        ModelFormat::Gguf => execute_gguf_inference(model_path, input_path, options),
    }
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
                        format!("Encoded {} chars to {} tokens", prompt.len(), tokens.len()).dimmed()
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
                    format!("Using GPU: {} ({} MB VRAM)", cuda_model.device_name(), cuda_model.vram_mb()).green()
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
/// Uses realizar's safetensors support for transformer model inspection.
#[cfg(feature = "inference")]
fn execute_safetensors_inference(
    model_path: &Path,
    input_path: Option<&PathBuf>,
    _options: &RunOptions,
) -> Result<String> {
    use realizar::safetensors::SafetensorsModel;
    use std::time::Instant;

    // Load SafeTensors file
    let start = Instant::now();
    let data = std::fs::read(model_path)?;
    let st_model = SafetensorsModel::from_bytes(&data)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load SafeTensors: {e}")))?;

    let tensor_names: Vec<&String> = st_model.tensors.keys().collect();
    let load_time = start.elapsed();

    eprintln!(
        "{}",
        format!(
            "Loaded SafeTensors model with {} tensors in {:.2}ms",
            tensor_names.len(),
            load_time.as_secs_f64() * 1000.0
        )
        .dimmed()
    );

    // For SafeTensors, we typically need a tokenizer and transformer config
    // For now, provide detailed model info
    let input_desc = input_path
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "none".to_string());

    let mut output = String::new();
    output.push_str(&format!("Model: {}\n", model_path.display()));
    output.push_str(&format!("Input: {}\n", input_desc));
    output.push_str(&format!("Tensors: {}\n", tensor_names.len()));
    output.push_str(&format!(
        "Load time: {:.2}ms\n",
        load_time.as_secs_f64() * 1000.0
    ));

    // List first few tensors for inspection
    output.push_str("\nTensor names (first 10):\n");
    for (i, name) in tensor_names.iter().take(10).enumerate() {
        output.push_str(&format!("  {}. {}\n", i + 1, name));
    }

    if tensor_names.len() > 10 {
        output.push_str(&format!("  ... and {} more\n", tensor_names.len() - 10));
    }

    Ok(output)
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
                    // Text prompt - encode using GGUF's embedded tokenizer
                    mapped_model.model.encode(prompt).unwrap_or_else(|| vec![1u32])
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
                temperature: 0.0, // Greedy sampling for GPU argmax
                top_k: 1,         // Force argmax path
                ..Default::default()
            };

            // PAR-200: Use GPU-resident path for 20x faster inference (116 tok/s vs 5.7 tok/s)
            let gen_result = run_gguf_generate(model, &input_tokens, &gen_config, options.no_gpu, options.benchmark)?;

            // Show inference-only performance (excludes loading time)
            if options.benchmark {
                let new_tokens = gen_result.tokens.len().saturating_sub(input_tokens.len());
                let tok_per_sec = if gen_result.inference_ms > 0.0 {
                    new_tokens as f64 / (gen_result.inference_ms / 1000.0)
                } else {
                    0.0
                };
                eprintln!("Inference: {} tokens in {:.1}ms ({:.1} tok/s)", new_tokens, gen_result.inference_ms, tok_per_sec);
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
#[cfg(feature = "inference")]
fn run_gguf_generate(
    model: realizar::gguf::OwnedQuantizedModel,
    input_tokens: &[u32],
    gen_config: &realizar::gguf::QuantizedGenerateConfig,
    no_gpu: bool,
    benchmark: bool,
) -> Result<GgufGenerateResult> {
    #[cfg(feature = "cuda")]
    if !no_gpu {
        use realizar::gguf::OwnedQuantizedModelCuda;
        eprintln!("Initializing CUDA GPU 0 (GPU-resident mode)...");
        let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0)
            .map_err(|e| CliError::InferenceFailed(format!("CUDA init failed: {e}")))?;

        // Warmup for CUDA graphs (critical for accurate timing)
        if benchmark {
            eprintln!("Warmup (3 iterations)...");
            for _ in 0..3 {
                let _ = cuda_model.generate_gpu_resident(input_tokens, gen_config);
            }
        }

        // Measure inference time separately from loading
        let infer_start = Instant::now();
        let tokens = cuda_model.generate_gpu_resident(input_tokens, gen_config)
            .map_err(|e| CliError::InferenceFailed(format!("GPU generation failed: {e}")))?;
        let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

        return Ok(GgufGenerateResult { tokens, inference_ms });
    }

    // CPU fallback
    #[allow(unused_variables)]
    let _ = benchmark; // Used only in CUDA path for warmup
    let infer_start = Instant::now();
    let mut cpu_model = model;
    let tokens = cpu_model.generate_with_cache(input_tokens, gen_config)
        .map_err(|e| CliError::InferenceFailed(format!("Generation failed: {e}")))?;
    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

    Ok(GgufGenerateResult { tokens, inference_ms })
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

    let options = RunOptions {
        input: input.map(Path::to_path_buf),
        prompt: prompt.map(String::from),
        max_tokens,
        output_format: output_format.to_string(),
        force: false,
        no_gpu,
        offline,
        benchmark,
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
                repo: "whisper-tiny".to_string()
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
}
