//! CLI handlers extracted from main.rs for testability (T-COV-001)
//!
//! This module contains all the CLI argument definitions and handler functions
//! that were previously in main.rs, enabling unit testing of CLI logic.

#![allow(missing_docs)]

use crate::error::{RealizarError, Result};
use clap::{Parser, Subcommand};

/// Realizar - Pure Rust ML inference engine
///
/// A lightweight, fast alternative to ollama for local model inference.
#[derive(Parser, Debug)]
#[command(name = "realizar")]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// CLI subcommands
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Run a model for inference (like `ollama run`)
    Run {
        /// Model reference (pacha://name:version, hf://org/model, or path)
        #[arg(value_name = "MODEL")]
        model: String,

        /// Optional prompt (interactive mode if omitted)
        #[arg(value_name = "PROMPT")]
        prompt: Option<String>,

        /// Maximum tokens to generate
        #[arg(short = 'n', long, default_value = "256")]
        max_tokens: usize,

        /// Sampling temperature (0.0 = deterministic)
        #[arg(short, long, default_value = "0.7")]
        temperature: f32,

        /// Output format: text, json, or stream
        #[arg(short, long, default_value = "text")]
        format: String,

        /// System prompt for chat template
        #[arg(short, long)]
        system: Option<String>,

        /// Disable chat template formatting (send raw prompt)
        #[arg(long)]
        raw: bool,

        /// Force GPU acceleration (requires CUDA feature)
        #[arg(long)]
        gpu: bool,

        /// Show verbose output (loading details, performance stats)
        #[arg(short, long)]
        verbose: bool,

        /// Enable inference tracing for debugging (e.g., --trace or --trace=attention,ffn)
        #[arg(long, value_name = "STEPS")]
        trace: Option<Option<String>>,
    },
    /// Interactive chat mode (like `ollama chat`)
    Chat {
        /// Model reference
        #[arg(value_name = "MODEL")]
        model: String,

        /// System prompt to set context
        #[arg(short, long)]
        system: Option<String>,

        /// History file for conversation persistence
        #[arg(long)]
        history: Option<String>,
    },
    /// List available models (like `ollama list`)
    List {
        /// Show remote registry models
        #[arg(short, long)]
        remote: Option<String>,

        /// Output format: table, json
        #[arg(short, long, default_value = "table")]
        format: String,
    },
    /// Pull a model from registry (like `ollama pull`)
    Pull {
        /// Model reference to pull
        #[arg(value_name = "MODEL")]
        model: String,

        /// Force re-download even if cached
        #[arg(short, long)]
        force: bool,

        /// Quantization format (q4, q8, f16)
        #[arg(short, long)]
        quantize: Option<String>,
    },
    /// Push a model to registry (like `ollama push`)
    Push {
        /// Model to push
        #[arg(value_name = "MODEL")]
        model: String,

        /// Target registry URL
        #[arg(long)]
        to: Option<String>,
    },
    /// Start the inference server (with OpenAI-compatible API)
    Serve {
        /// Host to bind to
        #[arg(short = 'H', long, default_value = "127.0.0.1")]
        host: String,

        /// Port to bind to
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Path to model file (APR, GGUF, or SafeTensors)
        #[arg(short, long)]
        model: Option<String>,

        /// Use demo model for testing
        #[arg(long)]
        demo: bool,

        /// Enable OpenAI-compatible API at /v1/*
        #[arg(long, default_value = "true")]
        openai_api: bool,

        /// Enable batch inference for M4 parity (PARITY-093)
        #[arg(long)]
        batch: bool,

        /// Force GPU acceleration (requires CUDA feature)
        #[arg(long)]
        gpu: bool,

        /// Enable inference tracing (GH-103: propagates into QuantizedGenerateConfig.trace)
        /// Per-request tracing can also be activated via X-Trace-Level header
        #[arg(long)]
        trace: bool,

        /// GH-286: Maximum context/sequence length for KV cache pre-allocation.
        /// Lower values reduce RSS (e.g., 512 for benchmarks). Default: 4096.
        #[arg(long, default_value = "4096")]
        context_length: usize,

        /// GH-286: Skip FP8 weight cache warmup. Saves ~1.5 GB RSS but
        /// reduces decode throughput ~13% on sm_89+ (Ada Lovelace).
        #[arg(long)]
        no_fp8_cache: bool,
    },
    /// Run performance benchmarks (wraps cargo bench)
    Bench {
        /// Benchmark suite to run
        #[arg(value_name = "SUITE")]
        suite: Option<String>,

        /// List available benchmark suites
        #[arg(short, long)]
        list: bool,

        /// Runtime to benchmark (realizar, llama-cpp, vllm, ollama)
        #[arg(long)]
        runtime: Option<String>,

        /// Model path or name for inference benchmarks
        #[arg(long)]
        model: Option<String>,

        /// Server URL for external runtime benchmarking
        #[arg(long)]
        url: Option<String>,

        /// Output file for JSON results (v1.1 schema)
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Run convoy test for continuous batching validation
    BenchConvoy {
        /// Runtime to benchmark
        #[arg(long)]
        runtime: Option<String>,

        /// Model path for inference
        #[arg(long)]
        model: Option<String>,

        /// Output file for JSON results
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Run saturation stress test
    BenchSaturation {
        /// Runtime to benchmark
        #[arg(long)]
        runtime: Option<String>,

        /// Model path for inference
        #[arg(long)]
        model: Option<String>,

        /// Output file for JSON results
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Compare two benchmark result files
    BenchCompare {
        /// First benchmark result file (JSON)
        #[arg(value_name = "FILE1")]
        file1: String,

        /// Second benchmark result file (JSON)
        #[arg(value_name = "FILE2")]
        file2: String,

        /// Significance threshold percentage (default: 5.0)
        #[arg(short, long, default_value = "5.0")]
        threshold: f64,
    },
    /// Detect performance regressions between baseline and current
    BenchRegression {
        /// Baseline benchmark result file (JSON)
        #[arg(value_name = "BASELINE")]
        baseline: String,

        /// Current benchmark result file (JSON)
        #[arg(value_name = "CURRENT")]
        current: String,

        /// Strict mode: fail on any regression
        #[arg(long)]
        strict: bool,
    },
    /// Visualize benchmark results (terminal output)
    Viz {
        /// Use ANSI color output
        #[arg(short, long)]
        color: bool,

        /// Number of samples to generate
        #[arg(short, long, default_value = "100")]
        samples: usize,
    },
    /// Show version and configuration info
    Info,
}

/// Configuration for the run command
#[derive(Debug, Clone)]
pub struct RunConfig {
    pub model: String,
    pub prompt: Option<String>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub format: String,
    pub system: Option<String>,
    pub raw: bool,
    pub gpu: bool,
    pub verbose: bool,
    pub trace: Option<Option<String>>,
}

/// Configuration for the serve command
#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub host: String,
    pub port: u16,
    pub model: Option<String>,
    pub demo: bool,
    pub batch: bool,
    pub gpu: bool,
    /// Enable OpenAI-compatible API at /v1/* (GH-148)
    pub openai_api: bool,
    /// GH-103: Enable inference tracing (propagates into QuantizedGenerateConfig.trace)
    pub trace: bool,
    /// GH-286: Maximum context/sequence length for KV cache pre-allocation.
    /// Lower values reduce RSS. Default: 4096.
    pub context_length: usize,
    /// GH-286: Skip FP8 weight cache warmup. Saves ~1.5 GB RSS but
    /// reduces decode throughput ~13% on sm_89+. Default: false.
    pub no_fp8_cache: bool,
}

/// Handle the serve command
#[cfg(feature = "server")]
pub async fn handle_serve(config: ServeConfig) -> Result<()> {
    if config.demo {
        super::serve_demo(&config.host, config.port, config.openai_api).await
    } else if let Some(model_path) = config.model {
        super::serve_model(
            &config.host,
            config.port,
            &model_path,
            config.batch,
            config.gpu,
            config.openai_api,
            config.trace,
            config.context_length,
            config.no_fp8_cache,
        )
        .await
    } else {
        eprintln!("Error: Either --model or --demo must be specified");
        eprintln!();
        eprintln!("Usage:");
        eprintln!("  realizar serve --demo              # Use demo model");
        eprintln!("  realizar serve --model path.gguf   # Load GGUF model");
        eprintln!("  realizar serve --model path.gguf --batch  # Enable M4 parity batch mode");
        std::process::exit(1);
    }
}

/// Handle the serve command (stub when server feature disabled)
#[cfg(not(feature = "server"))]
pub async fn handle_serve(_config: ServeConfig) -> Result<()> {
    Err(RealizarError::UnsupportedOperation {
        operation: "serve".to_string(),
        reason: "Server feature not enabled. Build with --features server".to_string(),
    })
}

/// Scan a directory for model files (.gguf, .safetensors, .apr)
fn scan_model_directory(dir: &std::path::Path) -> Vec<(String, u64)> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_file() {
                return None;
            }
            let name = path.file_name()?.to_string_lossy().to_string();
            if name.ends_with(".gguf") || name.ends_with(".safetensors") || name.ends_with(".apr") {
                let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                Some((name, size))
            } else {
                None
            }
        })
        .collect()
}

/// Print a list of models in the requested format
// serde_json::json!() uses infallible unwrap
#[allow(clippy::disallowed_methods)]
fn print_model_list(models: &[(String, u64)], format: &str) {
    match format {
        "json" => {
            let json_models: Vec<_> = models
                .iter()
                .map(|(name, size)| {
                    serde_json::json!({
                        "name": name,
                        "size_bytes": size,
                        "size_human": super::format_size(*size)
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&json_models).unwrap_or_default()
            );
        },
        _ => {
            println!("{:<40} {:>12}", "NAME", "SIZE");
            println!("{}", "-".repeat(54));
            for (name, size) in models {
                println!("{:<40} {:>12}", name, super::format_size(*size));
            }
        },
    }
}

/// Handle the list command
pub fn handle_list(remote: Option<&str>, format: &str) -> Result<()> {
    // GH-148: Headers go to stderr to avoid contaminating --format json output
    eprintln!("Available Models");
    eprintln!("================");
    eprintln!();

    if let Some(remote_url) = remote {
        eprintln!("Remote registry: {remote_url}");
        eprintln!();
        eprintln!("Note: Remote registry listing requires --features registry.");
        return Ok(());
    }

    let pacha_dir = super::home_dir().map_or_else(
        || std::path::PathBuf::from(".pacha/models"),
        |h| h.join(".pacha").join("models"),
    );

    if !pacha_dir.exists() {
        eprintln!("No models found in local registry.");
        eprintln!();
        eprintln!("Pull a model:");
        eprintln!("  realizar pull llama3:8b");
        eprintln!();
        eprintln!("Or run a local file:");
        eprintln!("  realizar run ./model.gguf \"prompt\"");
        return Ok(());
    }

    let models_found = scan_model_directory(&pacha_dir);
    if models_found.is_empty() {
        eprintln!("No models found in {}", pacha_dir.display());
    } else {
        print_model_list(&models_found, format);
    }

    Ok(())
}

/// Handle the pull command
#[allow(clippy::unused_async)]
pub async fn handle_pull(model_ref: &str, force: bool, quantize: Option<&str>) -> Result<()> {
    // GH-148: Status messages to stderr to avoid contaminating structured output
    eprintln!("Pulling model: {model_ref}");
    if force {
        eprintln!("  Force: re-downloading even if cached");
    }
    if let Some(q) = quantize {
        eprintln!("  Quantize: {q}");
    }
    eprintln!();

    if let Some(hf_path) = model_ref.strip_prefix("hf://") {
        eprintln!("Source: HuggingFace Hub");
        eprintln!("Model: {hf_path}");
        eprintln!();
        eprintln!("Enable registry support: --features registry");
        eprintln!("Or manual download:");
        eprintln!("  huggingface-cli download {hf_path}");
    } else if let Some(pacha_path) = model_ref.strip_prefix("pacha://") {
        eprintln!("Source: Pacha Registry");
        eprintln!("Model: {pacha_path}");
        eprintln!();
        eprintln!("Enable registry support: --features registry");
    } else {
        eprintln!("Source: Default registry (Pacha)");
        eprintln!("Model: {model_ref}");
        eprintln!();
        eprintln!("Enable registry support: --features registry");
        eprintln!("Or download manually and use:");
        eprintln!("  realizar run ./downloaded-model.gguf \"prompt\"");
    }

    Err(RealizarError::UnsupportedOperation {
        operation: "pull".to_string(),
        reason: "Registry feature not enabled. Build with --features registry".to_string(),
    })
}

/// Handle the push command
#[allow(clippy::unused_async)]
pub async fn handle_push(model_ref: &str, target: Option<&str>) -> Result<()> {
    // GH-148: Status messages to stderr to avoid contaminating structured output
    eprintln!("Pushing model: {model_ref}");
    if let Some(t) = target {
        eprintln!("  Target: {t}");
    } else {
        eprintln!("  Target: default Pacha registry");
    }
    eprintln!();
    eprintln!("Enable registry support: --features registry");
    Err(RealizarError::UnsupportedOperation {
        operation: "push".to_string(),
        reason: "Registry feature not enabled. Build with --features registry".to_string(),
    })
}

include!("handlers_trace.rs");
