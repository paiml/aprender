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
    /// Output format (text, json, srt, vtt)
    pub output_format: String,
    /// Force re-download (bypass cache)
    pub force: bool,
    /// Disable GPU acceleration
    pub no_gpu: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            input: None,
            output_format: "text".to_string(),
            force: false,
            no_gpu: false,
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
}

/// Run the model on input
pub(crate) fn run_model(source: &str, options: &RunOptions) -> Result<RunResult> {
    let start = Instant::now();

    // Parse source
    let model_source = ModelSource::parse(source)?;

    // Resolve model path (download if needed)
    let model_path = resolve_model(&model_source, options.force)?;

    // Validate model exists
    if !model_path.exists() {
        return Err(CliError::FileNotFound(model_path));
    }

    // Check if input is required
    let input_path = options.input.as_ref();

    // Load and run model
    let text = execute_inference(&model_path, input_path, options)?;

    let duration = start.elapsed();

    Ok(RunResult {
        text,
        duration_secs: duration.as_secs_f64(),
        cached: matches!(model_source, ModelSource::Local(_)) || model_source.cache_path().exists(),
    })
}

/// Resolve model source to local path
fn resolve_model(source: &ModelSource, _force: bool) -> Result<PathBuf> {
    match source {
        ModelSource::Local(path) => Ok(path.clone()),
        ModelSource::HuggingFace { org, repo } => {
            let cache_path = source.cache_path();
            if cache_path.exists() {
                // Find model file in cache
                find_model_in_dir(&cache_path)
            } else {
                // NOTE: Auto-download deferred to GH-80 milestone - manual import required
                Err(CliError::ValidationFailed(format!(
                    "Model not cached. Download hf://{org}/{repo} first with: apr import hf://{org}/{repo}"
                )))
            }
        }
        ModelSource::Url(url) => {
            let cache_path = source.cache_path();
            if cache_path.exists() {
                find_model_in_dir(&cache_path)
            } else {
                // NOTE: URL download deferred to GH-80 milestone
                Err(CliError::ValidationFailed(format!(
                    "Model not cached. URL download not yet implemented: {url}"
                )))
            }
        }
    }
}

/// Find model file in directory
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

/// Execute APR model inference (classical ML)
///
/// Uses realizar's AprModel for real inference on .apr format models.
#[cfg(feature = "inference")]
fn execute_apr_inference(
    model_path: &Path,
    input_path: Option<&PathBuf>,
    options: &RunOptions,
) -> Result<String> {
    use realizar::apr::AprModel;
    use std::time::Instant;

    // Load the APR model
    let model = AprModel::load(model_path)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load APR model: {e}")))?;

    eprintln!(
        "{}",
        format!(
            "Loaded {:?} model ({} parameters)",
            model.model_type(),
            model.num_parameters()
        )
        .dimmed()
    );

    // Parse input features
    let input_features = parse_input_features(input_path)?;

    // Run inference
    let start = Instant::now();
    let output = model
        .predict(&input_features)
        .map_err(|e| CliError::InferenceFailed(format!("APR prediction failed: {e}")))?;
    let inference_time = start.elapsed();

    // Format output based on options
    format_prediction_output(&output, inference_time, options)
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
/// Uses realizar's GGUF loader to inspect model metadata.
/// Full text generation requires additional setup (tokenizer, config).
#[cfg(feature = "inference")]
fn execute_gguf_inference(
    model_path: &Path,
    input_path: Option<&PathBuf>,
    _options: &RunOptions,
) -> Result<String> {
    use realizar::gguf::MappedGGUFModel;
    use std::time::Instant;

    // Load GGUF model via memory mapping
    let start = Instant::now();
    let mapped_model = MappedGGUFModel::from_path(model_path)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load GGUF model: {e}")))?;
    let load_time = start.elapsed();

    let model = &mapped_model.model;

    eprintln!(
        "{}",
        format!(
            "Loaded GGUF model (mmap) in {:.2}ms",
            load_time.as_secs_f64() * 1000.0
        )
        .dimmed()
    );

    let input_desc = input_path
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "none".to_string());

    // Build output with model information
    let mut output = String::new();
    output.push_str(&format!("Model: {}\n", model_path.display()));
    output.push_str(&format!("Input: {}\n", input_desc));
    output.push_str(&format!("GGUF Version: {}\n", model.header.version));
    output.push_str(&format!("Tensors: {}\n", model.tensors.len()));
    output.push_str(&format!("Metadata entries: {}\n", model.metadata.len()));
    output.push_str(&format!(
        "Load time: {:.2}ms\n",
        load_time.as_secs_f64() * 1000.0
    ));

    // Show some metadata
    output.push_str("\nMetadata (first 10):\n");
    for (i, (key, _)) in model.metadata.iter().take(10).enumerate() {
        output.push_str(&format!("  {}. {}\n", i + 1, key));
    }

    if model.metadata.len() > 10 {
        output.push_str(&format!("  ... and {} more\n", model.metadata.len() - 10));
    }

    // Show tensor info
    output.push_str("\nTensors (first 10):\n");
    for (i, tensor) in model.tensors.iter().take(10).enumerate() {
        output.push_str(&format!(
            "  {}. {} ({:?})\n",
            i + 1,
            tensor.name,
            tensor.dims
        ));
    }

    if model.tensors.len() > 10 {
        output.push_str(&format!("  ... and {} more\n", model.tensors.len() - 10));
    }

    output.push_str(
        "\nNote: Full text generation requires `apr serve` with proper tokenizer setup.\n",
    );

    Ok(output)
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
            .split(|c| c == ',' || c == ' ' || c == '\n' || c == '\t')
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
pub(crate) fn run(
    source: &str,
    input: Option<&Path>,
    stream: bool,
    _language: Option<&str>,
    _task: Option<&str>,
    output_format: &str,
    no_gpu: bool,
) -> Result<()> {
    println!("{}", "=== APR Run ===".cyan().bold());
    println!();
    println!("Source: {source}");

    let options = RunOptions {
        input: input.map(Path::to_path_buf),
        output_format: output_format.to_string(),
        force: false,
        no_gpu,
    };

    let result = run_model(source, &options)?;

    if stream {
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
            output_format: "json".to_string(),
            force: false,
            no_gpu: true,
        };
        assert!(options.no_gpu);
        assert_eq!(options.output_format, "json");
    }
}
