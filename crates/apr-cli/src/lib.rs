//! apr-cli library
//!
//! This library is the foundation for the apr CLI binary.
//! Exports CLI structures for testing and reuse.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod commands;
pub mod error;
mod output;

pub use error::CliError;

#[cfg(feature = "inference")]
pub mod federation;

// Commands are crate-private, used internally by execute_command
use commands::{
    bench, canary, canary::CanaryCommands, cbtop, chat, compare_hf, convert, debug, diff, eval,
    explain, export, flow, hex, import, inspect, lint, merge, oracle, probar, profile, publish,
    pull, qa, rosetta, rosetta::RosettaCommands, run, serve, showcase, tensors, trace, tree, tui,
    tune, validate,
};

/// apr - APR Model Operations Tool
///
/// Inspect, debug, and manage .apr model files.
/// Toyota Way: Genchi Genbutsu - Go and see the actual data.
#[derive(Parser, Debug)]
#[command(name = "apr")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Box<Commands>,

    /// Output as JSON
    #[arg(long, global = true)]
    pub json: bool,

    /// Verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Quiet mode (errors only)
    #[arg(short, long, global = true)]
    pub quiet: bool,

    /// Disable network access (Sovereign AI compliance, Section 9)
    #[arg(long, global = true)]
    pub offline: bool,

    /// Skip tensor contract validation (PMAT-237: use with diagnostic tooling)
    #[arg(long, global = true)]
    pub skip_contract: bool,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Run model directly (auto-download, cache, execute)
    Run {
        /// Model source: local path, hf://org/repo, or URL
        #[arg(value_name = "SOURCE")]
        source: String,

        /// Input file (audio, text, etc.)
        #[arg(short, long)]
        input: Option<PathBuf>,

        /// Text prompt for generation (for LLM models)
        #[arg(short, long)]
        prompt: Option<String>,

        /// Maximum tokens to generate (default: 32)
        #[arg(long, default_value = "32")]
        max_tokens: usize,

        /// Enable streaming output
        #[arg(long)]
        stream: bool,

        /// Language code (for ASR models)
        #[arg(short, long)]
        language: Option<String>,

        /// Task (transcribe, translate)
        #[arg(short, long)]
        task: Option<String>,

        /// Output format (text, json, srt, vtt)
        #[arg(short = 'f', long, default_value = "text")]
        format: String,

        /// Disable GPU acceleration
        #[arg(long, conflicts_with = "gpu")]
        no_gpu: bool,

        /// Force GPU acceleration
        #[arg(long, conflicts_with = "no_gpu")]
        gpu: bool,

        /// Offline mode: block all network access (Sovereign AI compliance)
        #[arg(long)]
        offline: bool,

        /// Benchmark mode: output performance metrics (tok/s, latency)
        #[arg(long)]
        benchmark: bool,

        /// Enable inference tracing (APR-TRACE-001)
        #[arg(long)]
        trace: bool,

        /// Trace specific steps only (comma-separated)
        #[arg(long, value_delimiter = ',')]
        trace_steps: Option<Vec<String>>,

        /// Verbose tracing (show tensor values)
        #[arg(long)]
        trace_verbose: bool,

        /// Save trace output to JSON file
        #[arg(long, value_name = "FILE")]
        trace_output: Option<PathBuf>,

        /// Trace detail level (none, basic, layer, payload)
        #[arg(long, value_name = "LEVEL", default_value = "basic")]
        trace_level: String,

        /// Shorthand for --trace --trace-level payload (tensor value inspection)
        #[arg(long)]
        trace_payload: bool,

        /// Enable inline Roofline profiling (PMAT-SHOWCASE-METHODOLOGY-001)
        #[arg(long)]
        profile: bool,

        /// Apply chat template for Instruct models (GAP-UX-001)
        ///
        /// Wraps prompt in ChatML format for Qwen2, LLaMA, Mistral Instruct models.
        /// Format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
        #[arg(long)]
        chat: bool,

        /// Show verbose output (model loading, backend info)
        #[arg(short, long)]
        verbose: bool,
    },

    /// Start inference server (REST API, streaming, metrics)
    Serve {
        /// Path to model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Port to listen on
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Disable CORS
        #[arg(long)]
        no_cors: bool,

        /// Disable Prometheus metrics endpoint
        #[arg(long)]
        no_metrics: bool,

        /// Disable GPU acceleration
        #[arg(long)]
        no_gpu: bool,

        /// Force GPU acceleration (requires CUDA)
        #[arg(long)]
        gpu: bool,

        /// Enable batched GPU inference for 2X+ throughput
        #[arg(long)]
        batch: bool,

        /// Enable inference tracing (PMAT-SHOWCASE-METHODOLOGY-001)
        #[arg(long)]
        trace: bool,

        /// Trace detail level (none, basic, layer)
        #[arg(long, value_name = "LEVEL", default_value = "basic")]
        trace_level: String,

        /// Enable inline Roofline profiling (adds X-Profile headers)
        #[arg(long)]
        profile: bool,
    },

    /// Inspect model metadata, vocab, and structure
    Inspect {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Show vocabulary details
        #[arg(long)]
        vocab: bool,

        /// Show filter/security details
        #[arg(long)]
        filters: bool,

        /// Show weight statistics
        #[arg(long)]
        weights: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Simple debugging output ("drama" mode available)
    Debug {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Theatrical "drama" mode output
        #[arg(long)]
        drama: bool,

        /// Show hex dump
        #[arg(long)]
        hex: bool,

        /// Extract ASCII strings
        #[arg(long)]
        strings: bool,

        /// Limit output lines
        #[arg(long, default_value = "256")]
        limit: usize,
    },

    /// Validate model integrity and quality
    Validate {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Show 100-point quality assessment
        #[arg(long)]
        quality: bool,

        /// Strict validation (fail on warnings)
        #[arg(long)]
        strict: bool,

        /// Minimum score to pass (0-100)
        #[arg(long)]
        min_score: Option<u8>,
    },

    /// Compare two models
    Diff {
        /// First model file
        #[arg(value_name = "FILE1")]
        file1: PathBuf,

        /// Second model file
        #[arg(value_name = "FILE2")]
        file2: PathBuf,

        /// Show weight-level differences
        #[arg(long)]
        weights: bool,

        /// Compare actual tensor values with statistical analysis
        #[arg(long)]
        values: bool,

        /// Filter tensors by name pattern (for --values)
        #[arg(long)]
        filter: Option<String>,

        /// Maximum number of tensors to compare (for --values)
        #[arg(long, default_value = "10")]
        limit: usize,

        /// Account for transpose when comparing (GGUF col-major vs APR row-major)
        #[arg(long)]
        transpose_aware: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// List tensor names and shapes
    Tensors {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Show tensor statistics (mean, std, min, max)
        #[arg(long)]
        stats: bool,

        /// Filter tensors by name pattern
        #[arg(long)]
        filter: Option<String>,

        /// Limit number of tensors shown (0 = unlimited)
        #[arg(long, default_value = "0")]
        limit: usize,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Layer-by-layer trace analysis
    Trace {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Filter layers by name pattern
        #[arg(long)]
        layer: Option<String>,

        /// Compare with reference model
        #[arg(long)]
        reference: Option<PathBuf>,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Verbose output with per-layer stats
        #[arg(short, long)]
        verbose: bool,

        /// Trace payload through model
        #[arg(long)]
        payload: bool,

        /// Diff mode
        #[arg(long)]
        diff: bool,

        /// Interactive mode
        #[arg(long)]
        interactive: bool,
    },

    /// Check for best practices and conventions
    Lint {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },

    /// Explain errors, architecture, and tensors
    Explain {
        /// Explain a specific error code
        #[arg(value_name = "CODE")]
        code: Option<String>,

        /// Path to .apr model file (optional context)
        #[arg(short, long)]
        file: Option<PathBuf>,

        /// Explain a specific tensor
        #[arg(long)]
        tensor: Option<String>,
    },

    /// Manage canary tests for regression
    Canary {
        #[command(subcommand)]
        command: CanaryCommands,
    },

    /// Export model to other formats
    Export {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Output format (onnx, safetensors, gguf)
        #[arg(long, default_value = "safetensors")]
        format: String,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Apply quantization during export (int8, int4, fp16)
        #[arg(long)]
        quantize: Option<String>,
    },

    /// Import from external formats (hf://org/repo, local files, URLs)
    Import {
        /// Source: hf://org/repo, local file, or URL
        #[arg(value_name = "SOURCE")]
        source: String,

        /// Output .apr file path (default: derived from source name)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Model architecture (whisper, llama, bert, auto)
        #[arg(long, default_value = "auto")]
        arch: String,

        /// Quantization (int8, int4, fp16)
        #[arg(long)]
        quantize: Option<String>,

        /// Strict mode: reject unverified architectures and fail on validation errors
        #[arg(long)]
        strict: bool,

        /// Preserve Q4K quantization for fused kernel inference (GGUF only)
        /// Uses realizar's Q4K converter instead of dequantizing to F32
        #[arg(long)]
        preserve_q4k: bool,

        /// PMAT-232: External tokenizer.json for weights-only GGUF files.
        /// Required if the GGUF has no embedded tokenizer vocabulary.
        #[arg(long)]
        tokenizer: Option<PathBuf>,
    },

    /// Download and cache model from HuggingFace (Ollama-like UX)
    Pull {
        /// Model reference (alias, hf:// URI, or org/repo)
        #[arg(value_name = "MODEL")]
        model_ref: String,

        /// Force re-download even if cached
        #[arg(long)]
        force: bool,
    },

    /// List cached models
    #[command(name = "list", alias = "ls")]
    List,

    /// Remove model from cache
    #[command(name = "rm", alias = "remove")]
    Rm {
        /// Model reference to remove
        #[arg(value_name = "MODEL")]
        model_ref: String,
    },

    /// Convert/optimize model
    Convert {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Quantize to format (int8, int4, fp16, q4k)
        #[arg(long)]
        quantize: Option<String>,

        /// Compress output (none, zstd, zstd-max, lz4)
        #[arg(long)]
        compress: Option<String>,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Force overwrite existing files
        #[arg(short, long)]
        force: bool,
    },

    /// Merge multiple models
    Merge {
        /// Model files to merge
        #[arg(value_name = "FILES", num_args = 2..)]
        files: Vec<PathBuf>,

        /// Merge strategy (average, weighted, ties)
        #[arg(long, default_value = "average")]
        strategy: String,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Weights for weighted merge (comma-separated, e.g., "0.7,0.3")
        #[arg(long, value_delimiter = ',')]
        weights: Option<Vec<f32>>,
    },

    /// Interactive terminal UI
    Tui {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,
    },

    /// ComputeBrick pipeline monitor (cbtop)
    Cbtop {
        /// Model name (e.g., qwen2.5-coder-1.5b)
        #[arg(long)]
        model: Option<String>,

        /// Attach to running realizar process
        #[arg(long)]
        attach: Option<String>,

        /// Path to GGUF model file for real profiling
        #[arg(long, value_name = "MODEL")]
        model_path: Option<PathBuf>,

        /// Run in headless mode (no TUI, for CI/automation)
        #[arg(long)]
        headless: bool,

        /// Output JSON format (requires --headless)
        #[arg(long)]
        json: bool,

        /// Output file path (requires --headless)
        #[arg(long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// CI mode: exit with code 1 if thresholds not met
        #[arg(long)]
        ci: bool,

        /// Minimum throughput threshold in tok/s (for --ci)
        #[arg(long, value_name = "TOK_S")]
        throughput: Option<f64>,

        /// Minimum brick score threshold 0-100 (for --ci)
        #[arg(long, value_name = "SCORE")]
        brick_score: Option<u32>,

        /// Number of warmup iterations before measurement
        #[arg(long, default_value = "10")]
        warmup: usize,

        /// Number of measurement iterations
        #[arg(long, default_value = "100")]
        iterations: usize,

        /// PAR-100: Enable speculative decoding benchmark
        #[arg(long)]
        speculative: bool,

        /// PAR-100: Number of tokens to draft speculatively (default: 4)
        #[arg(long, default_value = "4")]
        speculation_k: usize,

        /// PAR-099: Path to draft model for speculative decoding
        #[arg(long, value_name = "DRAFT_MODEL")]
        draft_model: Option<PathBuf>,

        /// PAR-102: Number of concurrent requests
        #[arg(long, default_value = "1")]
        concurrent: usize,

        /// Use simulated data (for CI testing only)
        #[arg(long)]
        simulated: bool,
    },

    /// Export for probar visual testing
    Probar {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Output directory for test artifacts
        #[arg(short, long, default_value = "./probar-export")]
        output: PathBuf,

        /// Export format: json, png, or both
        #[arg(long, default_value = "both")]
        format: String,

        /// Golden reference directory for comparison
        #[arg(long)]
        golden: Option<PathBuf>,

        /// Filter layers by name pattern
        #[arg(long)]
        layer: Option<String>,
    },

    /// Compare APR model against HuggingFace source
    #[command(name = "compare-hf")]
    CompareHf {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// HuggingFace repo ID (e.g., openai/whisper-tiny)
        #[arg(long)]
        hf: String,

        /// Filter tensors by name pattern
        #[arg(long)]
        tensor: Option<String>,

        /// Comparison threshold (default: 1e-5)
        #[arg(long, default_value = "1e-5")]
        threshold: f64,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Hex dump tensor data
    Hex {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Filter tensors by name pattern
        #[arg(long)]
        tensor: Option<String>,

        /// Limit bytes to display
        #[arg(long, default_value = "64")]
        limit: usize,

        /// Show tensor statistics
        #[arg(long)]
        stats: bool,

        /// List tensor names only
        #[arg(long)]
        list: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Model architecture tree view
    Tree {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Filter by component pattern
        #[arg(long)]
        filter: Option<String>,

        /// Output format: ascii, dot, mermaid, json
        #[arg(long, default_value = "ascii")]
        format: String,

        /// Show tensor sizes
        #[arg(long)]
        sizes: bool,

        /// Maximum tree depth
        #[arg(long)]
        depth: Option<usize>,
    },

    /// Data flow visualization
    Flow {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Filter by layer pattern
        #[arg(long)]
        layer: Option<String>,

        /// Component to visualize: full, encoder, decoder, etc.
        #[arg(long, default_value = "full")]
        component: String,

        /// Verbose output with statistics
        #[arg(short, long)]
        verbose: bool,
    },

    /// Interactive chat with language model
    Chat {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Sampling temperature (0 = greedy, higher = more random)
        #[arg(long, default_value = "0.7")]
        temperature: f32,

        /// Nucleus sampling threshold
        #[arg(long, default_value = "0.9")]
        top_p: f32,

        /// Maximum tokens to generate per response
        #[arg(long, default_value = "512")]
        max_tokens: usize,

        /// System prompt to set model behavior
        #[arg(long)]
        system: Option<String>,

        /// Show inspection info (top-k probs, tokens/sec)
        #[arg(long)]
        inspect: bool,

        /// Disable GPU acceleration (use CPU)
        #[arg(long)]
        no_gpu: bool,

        /// Force GPU acceleration (requires CUDA)
        #[arg(long)]
        gpu: bool,

        /// Enable inference tracing (APR-TRACE-001)
        #[arg(long)]
        trace: bool,

        /// Trace specific steps only (comma-separated)
        #[arg(long, value_delimiter = ',')]
        trace_steps: Option<Vec<String>>,

        /// Verbose tracing
        #[arg(long)]
        trace_verbose: bool,

        /// Save trace output to JSON file
        #[arg(long, value_name = "FILE")]
        trace_output: Option<PathBuf>,

        /// Trace detail level (none, basic, layer, payload)
        #[arg(long, value_name = "LEVEL", default_value = "basic")]
        trace_level: String,

        /// Enable inline Roofline profiling (PMAT-SHOWCASE-METHODOLOGY-001)
        #[arg(long)]
        profile: bool,
    },

    /// Benchmark throughput (spec H12: >= 10 tok/s)
    Bench {
        /// Path to model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Number of warmup iterations
        #[arg(long, default_value = "3")]
        warmup: usize,

        /// Number of measurement iterations
        #[arg(long, default_value = "5")]
        iterations: usize,

        /// Max tokens to generate per iteration
        #[arg(long, default_value = "32")]
        max_tokens: usize,

        /// Test prompt
        #[arg(long)]
        prompt: Option<String>,

        /// Use realizar for fast inference (vs aprender baseline)
        #[arg(long)]
        fast: bool,

        /// Benchmark specific brick
        #[arg(long)]
        brick: Option<String>,
    },

    /// Evaluate model perplexity (spec H13: PPL <= 20)
    Eval {
        /// Path to model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Dataset: wikitext-2, lambada, or custom
        #[arg(long, default_value = "wikitext-2")]
        dataset: String,

        /// Custom text (when dataset=custom)
        #[arg(long)]
        text: Option<String>,

        /// Maximum tokens to evaluate
        #[arg(long, default_value = "512")]
        max_tokens: usize,

        /// Perplexity threshold for pass/fail
        #[arg(long, default_value = "20.0")]
        threshold: f32,
    },

    /// Deep profiling with Roofline analysis
    Profile {
        /// Path to model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Layer-by-layer granular analysis
        #[arg(long)]
        granular: bool,

        /// Output format (human, json, flamegraph)
        #[arg(long, default_value = "human")]
        format: String,

        /// Focus on specific operation
        #[arg(long)]
        focus: Option<String>,

        /// Detect naive implementations
        #[arg(long)]
        detect_naive: bool,

        /// GFLOPS threshold for naive detection
        #[arg(long, default_value = "10.0")]
        threshold: f64,

        /// Compare against HuggingFace baseline
        #[arg(long)]
        compare_hf: Option<String>,

        /// Measure energy consumption (requires RAPL)
        #[arg(long)]
        energy: bool,

        /// Compute performance grade
        #[arg(long)]
        perf_grade: bool,

        /// Show call graph
        #[arg(long)]
        callgraph: bool,

        /// Exit non-zero if naive implementation detected
        #[arg(long)]
        fail_on_naive: bool,

        /// Output file path for flamegraph SVG (GH-174, PMAT-182)
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,

        // PMAT-192: CI Assertion Mode (GH-180)
        /// Enable CI mode with assertion checks (exits 1 on failure)
        #[arg(long)]
        ci: bool,

        /// Minimum throughput in tok/s (CI assertion, exits 1 if below)
        #[arg(long)]
        assert_throughput: Option<f64>,

        /// Maximum p99 latency in ms (CI assertion, exits 1 if above)
        #[arg(long)]
        assert_p99: Option<f64>,

        /// Maximum p50 latency in ms (CI assertion, exits 1 if above)
        #[arg(long)]
        assert_p50: Option<f64>,

        /// Warmup passes before measurement (default: 3)
        #[arg(long, default_value = "3")]
        warmup: usize,

        /// Measurement passes (default: 10)
        #[arg(long, default_value = "10")]
        measure: usize,
    },

    /// Falsifiable QA checklist for model releases
    Qa {
        /// Path to model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Minimum throughput threshold in tok/s
        #[arg(long, value_name = "TPS")]
        assert_tps: Option<f64>,

        /// Minimum speedup vs Ollama
        #[arg(long, value_name = "SPEEDUP")]
        assert_speedup: Option<f64>,

        /// Minimum GPU vs CPU speedup (F-PERF-042)
        #[arg(long, value_name = "SPEEDUP")]
        assert_gpu_speedup: Option<f64>,

        /// Skip golden output test
        #[arg(long)]
        skip_golden: bool,

        /// Skip throughput benchmark
        #[arg(long)]
        skip_throughput: bool,

        /// Skip Ollama parity comparison
        #[arg(long)]
        skip_ollama: bool,

        /// Skip GPU vs CPU speedup test (F-PERF-042)
        #[arg(long)]
        skip_gpu_speedup: bool,

        /// Skip tensor contract validation (PMAT-235)
        #[arg(long)]
        skip_contract: bool,

        /// Skip cross-format parity test (F-QUAL-032)
        #[arg(long)]
        skip_format_parity: bool,

        /// SafeTensors model path for cross-format parity test (F-QUAL-032)
        #[arg(long, value_name = "PATH")]
        safetensors_path: Option<PathBuf>,

        /// Number of benchmark iterations
        #[arg(long, default_value = "10")]
        iterations: usize,

        /// Number of warmup iterations
        #[arg(long, default_value = "3")]
        warmup: usize,

        /// Maximum tokens to generate
        #[arg(long, default_value = "32")]
        max_tokens: usize,

        /// Output as JSON (for CI integration)
        #[arg(long)]
        json: bool,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// ML tuning: LoRA/QLoRA configuration and memory planning (GH-176)
    Tune {
        /// Path to model file (optional if using --model)
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,

        /// Tuning method: auto, full, lora, qlora
        #[arg(long, short = 'm', default_value = "auto")]
        method: String,

        /// LoRA rank (default: auto-selected)
        #[arg(long, short = 'r')]
        rank: Option<u32>,

        /// Available VRAM in GB
        #[arg(long, default_value = "16.0")]
        vram: f64,

        /// Only plan configuration, don't train
        #[arg(long)]
        plan: bool,

        /// Model size for planning (e.g., "7B", "1.5B")
        #[arg(long, value_name = "SIZE")]
        model: Option<String>,

        /// Freeze base model weights
        #[arg(long)]
        freeze_base: bool,

        /// Training data file (JSONL format)
        #[arg(long, value_name = "FILE")]
        train_data: Option<PathBuf>,

        /// Output as JSON (for CI integration)
        #[arg(long)]
        json: bool,
    },

    /// Qwen2.5-Coder showcase demo
    Showcase {
        /// Run all steps with auto-verification
        #[arg(long)]
        auto_verify: bool,

        /// Run specific step
        #[arg(long)]
        step: Option<String>,

        /// Model tier: tiny (0.5B), small (1.5B), medium (7B), large (32B)
        #[arg(long, default_value = "small")]
        tier: String,

        /// Model directory
        #[arg(long, default_value = "./models")]
        model_dir: PathBuf,

        /// Baselines to compare: llama-cpp,ollama
        #[arg(long, default_value = "llama-cpp,ollama")]
        baseline: String,

        /// Enable ZRAM compression
        #[arg(long)]
        zram: bool,

        /// Number of benchmark runs (spec: minimum 30)
        #[arg(long, default_value = "30")]
        runs: usize,

        /// Force GPU acceleration
        #[arg(long)]
        gpu: bool,

        /// Output results as JSON
        #[arg(long)]
        json: bool,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,

        /// Quiet mode (errors only)
        #[arg(short, long)]
        quiet: bool,
    },

    /// Model self-test: 10-stage pipeline integrity check (APR-TRACE-001)
    Check {
        /// Path to model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Disable GPU acceleration
        #[arg(long)]
        no_gpu: bool,
    },

    /// Rosetta Stone - Universal model format converter (PMAT-ROSETTA-001)
    Rosetta {
        #[command(subcommand)]
        action: RosettaCommands,
    },

    /// Publish model to HuggingFace Hub (APR-PUB-001)
    Publish {
        /// Directory containing model files to publish
        #[arg(value_name = "DIRECTORY")]
        directory: PathBuf,

        /// HuggingFace repository ID (e.g., paiml/whisper-apr-tiny)
        #[arg(value_name = "REPO_ID")]
        repo_id: String,

        /// Model display name
        #[arg(long)]
        model_name: Option<String>,

        /// License (SPDX identifier, default: mit)
        #[arg(long, default_value = "mit")]
        license: String,

        /// Pipeline tag (e.g., automatic-speech-recognition, text-generation)
        #[arg(long, default_value = "text-generation")]
        pipeline_tag: String,

        /// Library name (e.g., whisper-apr, aprender)
        #[arg(long)]
        library_name: Option<String>,

        /// Additional tags (comma-separated)
        #[arg(long, value_delimiter = ',')]
        tags: Option<Vec<String>>,

        /// Commit message
        #[arg(long)]
        message: Option<String>,

        /// Dry run (preview without uploading)
        #[arg(long)]
        dry_run: bool,
    },

    /// Model Oracle: identify family, size, constraints, and contract compliance
    ///
    /// Three modes:
    ///   apr oracle <FILE>         - Analyze local model file
    ///   apr oracle hf://org/repo  - Query HuggingFace API
    ///   apr oracle --family qwen2 - Describe contract from YAML
    Oracle {
        /// Model file path or hf:// URI
        #[arg(value_name = "SOURCE")]
        source: Option<String>,

        /// Show contract for a model family (e.g., qwen2, llama, whisper, bert)
        #[arg(long)]
        family: Option<String>,

        /// Filter to a specific size variant (e.g., 0.5b, 7b)
        #[arg(long)]
        size: Option<String>,

        /// Run full contract compliance check
        #[arg(long)]
        compliance: bool,

        /// List all tensor shapes
        #[arg(long)]
        tensors: bool,

        /// Show statistical analysis (GQA, memory, FFN, FLOPS)
        #[arg(long)]
        stats: bool,

        /// Show architecture explanations with literature references
        #[arg(long)]
        explain: bool,

        /// Show kernel compatibility report (quantization, TPS estimates)
        #[arg(long)]
        kernels: bool,

        /// Cross-validate contract against HuggingFace config.json
        #[arg(long)]
        validate: bool,

        /// Enable all analysis sections (stats + explain + kernels + validate)
        #[arg(long)]
        full: bool,
    },
}

/// PMAT-237: Extract model file paths from a command variant.
///
/// Returns paths for action commands (run, serve, bench, etc.) that should be
/// validated against the tensor contract. Returns empty vec for diagnostic
/// commands (qa, validate, inspect, debug, etc.) that must work on corrupt models.
fn extract_model_paths(command: &Commands) -> Vec<PathBuf> {
    match command {
        // === ACTION COMMANDS (gated) ===
        Commands::Run { source, .. } => {
            // Only validate local files, not hf:// or URLs
            let path = PathBuf::from(source);
            if path.exists() {
                vec![path]
            } else {
                vec![]
            }
        }
        Commands::Serve { file, .. }
        | Commands::Trace { file, .. }
        | Commands::Export { file, .. }
        | Commands::Convert { file, .. }
        | Commands::Probar { file, .. }
        | Commands::CompareHf { file, .. }
        | Commands::Chat { file, .. }
        | Commands::Bench { file, .. }
        | Commands::Eval { file, .. }
        | Commands::Profile { file, .. }
        | Commands::Check { file, .. } => vec![file.clone()],

        Commands::Merge { files, .. } => files.clone(),

        Commands::Cbtop { model_path, .. } => model_path.iter().cloned().collect(),
        Commands::Tui { file, .. } => file.iter().cloned().collect(),
        Commands::Import { source, .. } => {
            let path = PathBuf::from(source);
            if path.exists() {
                vec![path]
            } else {
                vec![]
            }
        }

        // Rosetta action subcommands
        Commands::Rosetta { action } => match action {
            RosettaCommands::Convert { source, .. }
            | RosettaCommands::Chain { source, .. }
            | RosettaCommands::Verify { source, .. } => vec![source.clone()],
            RosettaCommands::CompareInference {
                model_a, model_b, ..
            } => {
                vec![model_a.clone(), model_b.clone()]
            }
            // Diagnostic rosetta commands — exempt
            _ => vec![],
        },

        // === DIAGNOSTIC COMMANDS (exempt) ===
        // qa, validate, inspect, debug, tensors, hex, diff, lint, tree, flow,
        // explain, list, rm, pull, showcase, tune, canary, publish
        _ => vec![],
    }
}

/// PMAT-237: Validate model files against tensor contract before dispatch.
///
/// Uses `RosettaStone::validate()` to check for NaN, Inf, all-zeros, density,
/// and other contract violations. Returns `CliError::ValidationFailed` (exit 5)
/// if any violations are found.
///
/// GH-213: For sharded SafeTensors models (index.json), validates shard integrity
/// via `.apr-manifest.json` checksums instead of RosettaStone (which can't parse
/// index files). This catches truncated downloads before inference.
fn validate_model_contract(paths: &[PathBuf]) -> Result<(), CliError> {
    let rosetta = aprender::format::rosetta::RosettaStone::new();
    for path in paths {
        if !path.exists() {
            continue; // Let the subcommand handle FileNotFound
        }
        // GH-213: For sharded index.json, validate shard integrity via manifest
        if path.to_string_lossy().ends_with(".safetensors.index.json") {
            if let Some(parent) = path.parent() {
                let manifest_path = parent.join(".apr-manifest.json");
                if manifest_path.exists() {
                    validate_shard_manifest(&manifest_path, parent)?;
                }
            }
            continue; // Skip RosettaStone (index.json is not a model file)
        }
        let report = rosetta.validate(path).map_err(|e| {
            CliError::ValidationFailed(format!(
                "Contract validation failed for {}: {e}",
                path.display()
            ))
        })?;
        if !report.is_valid {
            let violation_count: usize = report.tensors.iter().map(|t| t.failures.len()).sum();
            return Err(CliError::ValidationFailed(format!(
                "PMAT-237 CONTRACT VIOLATION: {} has {} violations in {} tensors. \
                 Use 'apr qa {}' for details. Use --skip-contract to bypass.",
                path.display(),
                violation_count,
                report.failed_tensor_count,
                path.display(),
            )));
        }
    }
    Ok(())
}

/// GH-213: Validate sharded model integrity by checking file sizes against manifest.
///
/// This is an O(1)-per-file check (stat syscall only, no hashing) that catches
/// truncated downloads before they cause cryptic "tensor not found" errors.
fn validate_shard_manifest(
    manifest_path: &std::path::Path,
    cache_dir: &std::path::Path,
) -> Result<(), CliError> {
    let manifest_str = std::fs::read_to_string(manifest_path).map_err(|e| {
        CliError::ValidationFailed(format!(
            "Failed to read manifest {}: {e}",
            manifest_path.display()
        ))
    })?;
    let manifest: commands::pull::ShardManifest =
        serde_json::from_str(&manifest_str).map_err(|e| {
            CliError::ValidationFailed(format!(
                "Failed to parse manifest {}: {e}",
                manifest_path.display()
            ))
        })?;

    for (filename, checksum) in &manifest.files {
        let file_path = cache_dir.join(filename);
        if !file_path.exists() {
            return Err(CliError::ValidationFailed(format!(
                "Shard '{}' is missing. Re-run 'apr pull --force' to re-download.",
                filename
            )));
        }
        let actual_size = std::fs::metadata(&file_path)
            .map(|m| m.len())
            .map_err(|e| {
                CliError::ValidationFailed(format!("Failed to stat shard '{}': {e}", filename))
            })?;
        if actual_size != checksum.size {
            return Err(CliError::ValidationFailed(format!(
                "Shard '{}' size mismatch: expected {} bytes, got {} bytes \
                 (file may be truncated). Re-run 'apr pull --force' to re-download.",
                filename, checksum.size, actual_size
            )));
        }
    }
    Ok(())
}

/// Execute the CLI command and return the result.
#[allow(clippy::too_many_lines)]
pub fn execute_command(cli: &Cli) -> Result<(), CliError> {
    // PMAT-237: Contract gate — refuse to operate on corrupt models
    if !cli.skip_contract {
        let paths = extract_model_paths(&cli.command);
        validate_model_contract(&paths)?;
    }

    match cli.command.as_ref() {
        Commands::Check { file, no_gpu } => commands::check::run(file, *no_gpu),
        Commands::Run {
            source,
            input,
            prompt,
            max_tokens,
            stream,
            language,
            task,
            format,
            no_gpu,
            gpu,
            offline,
            benchmark,
            trace,
            trace_steps,
            trace_verbose,
            trace_output,
            trace_level,
            trace_payload,
            profile,
            chat,
            verbose,
        } => {
            // Handle --trace-payload shorthand (enables trace + sets level to payload)
            let effective_trace = *trace || *trace_payload;
            let effective_trace_level = if *trace_payload {
                "payload"
            } else {
                trace_level.as_str()
            };
            // GH-196: --gpu forces GPU, --no-gpu disables GPU.
            // When --gpu is passed, no_gpu is false (enforced by conflicts_with).
            let _ = gpu; // --gpu is the inverse of --no-gpu; no_gpu=false when --gpu is set

            // GAP-UX-001: Apply chat template if --chat flag is set
            let effective_prompt = if *chat {
                prompt
                    .as_ref()
                    .map(|p| format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", p))
            } else {
                prompt.clone()
            };

            // Use local verbose flag if set, otherwise fall back to global
            let effective_verbose = *verbose || cli.verbose;

            run::run(
                source,
                input.as_deref(),
                effective_prompt.as_deref(),
                *max_tokens,
                *stream,
                language.as_deref(),
                task.as_deref(),
                format,
                *no_gpu,
                *offline,
                *benchmark,
                effective_verbose,
                effective_trace,
                trace_steps.as_deref(),
                *trace_verbose,
                trace_output.clone(),
                effective_trace_level,
                *profile,
            )
        }

        Commands::Serve {
            file,
            port,
            host,
            no_cors,
            no_metrics,
            no_gpu,
            gpu,
            batch,
            trace,
            trace_level,
            profile,
        } => {
            // GH-152: Wire global verbose flag to server config for request/response logging
            let config = serve::ServerConfig {
                port: *port,
                host: host.clone(),
                cors: !no_cors,
                metrics: !no_metrics,
                no_gpu: *no_gpu,
                gpu: *gpu,
                batch: *batch,
                trace: *trace,
                trace_level: trace_level.clone(),
                profile: *profile,
                verbose: cli.verbose,
                ..Default::default()
            };
            serve::run(file, &config)
        }

        Commands::Inspect {
            file,
            vocab,
            filters,
            weights,
            json,
        } => inspect::run(file, *vocab, *filters, *weights, *json || cli.json),

        Commands::Debug {
            file,
            drama,
            hex,
            strings,
            limit,
        } => debug::run(file, *drama, *hex, *strings, *limit),

        Commands::Validate {
            file,
            quality,
            strict,
            min_score,
        } => validate::run(file, *quality, *strict, *min_score),

        Commands::Diff {
            file1,
            file2,
            weights,
            values,
            filter,
            limit,
            transpose_aware,
            json,
        } => diff::run(
            file1,
            file2,
            *weights,
            *values,
            filter.as_deref(),
            *limit,
            *transpose_aware,
            *json || cli.json,
        ),

        Commands::Tensors {
            file,
            stats,
            filter,
            limit,
            json,
        } => tensors::run(file, *stats, filter.as_deref(), *json || cli.json, *limit),

        Commands::Trace {
            file,
            layer,
            reference,
            json,
            verbose,
            payload,
            diff,
            interactive,
        } => trace::run(
            file,
            layer.as_deref(),
            reference.as_deref(),
            *json || cli.json,
            *verbose || cli.verbose,
            *payload,
            *diff,
            *interactive,
        ),

        Commands::Lint { file } => lint::run(file),
        Commands::Explain { code, file, tensor } => {
            explain::run(code.clone(), file.clone(), tensor.clone())
        }
        Commands::Canary { command } => canary::run(command.clone()),
        Commands::Export {
            file,
            format,
            output,
            quantize,
        } => export::run(file, format, output, quantize.as_deref()),
        Commands::Import {
            source,
            output,
            arch,
            quantize,
            strict,
            preserve_q4k,
            tokenizer,
        } => import::run(
            source,
            output.as_deref(),
            Some(arch.as_str()),
            quantize.as_deref(),
            *strict,
            *preserve_q4k,
            tokenizer.as_ref(),
        ),
        Commands::Pull { model_ref, force } => pull::run(model_ref, *force),
        Commands::List => pull::list(),
        Commands::Rm { model_ref } => pull::remove(model_ref),
        Commands::Convert {
            file,
            quantize,
            compress,
            output,
            force,
        } => convert::run(
            file,
            quantize.as_deref(),
            compress.as_deref(),
            output,
            *force,
        ),
        Commands::Merge {
            files,
            strategy,
            output,
            weights,
        } => merge::run(files, strategy, output, weights.clone()),
        Commands::Tui { file } => tui::run(file.clone()),

        Commands::Cbtop {
            model,
            attach,
            model_path,
            headless,
            json,
            output,
            ci,
            throughput,
            brick_score,
            warmup,
            iterations,
            speculative,
            speculation_k,
            draft_model,
            concurrent,
            simulated,
        } => {
            let (resolved_model, resolved_model_path) = if let Some(ref m) = model {
                let path = std::path::Path::new(m);
                let is_gguf = path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"));
                if is_gguf || path.exists() {
                    (
                        Some(
                            path.file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or(m)
                                .to_string(),
                        ),
                        Some(PathBuf::from(m)),
                    )
                } else {
                    (Some(m.clone()), model_path.clone())
                }
            } else {
                (None, model_path.clone())
            };

            cbtop::run(cbtop::CbtopConfig {
                model: resolved_model,
                attach: attach.as_deref().map(String::from),
                model_path: resolved_model_path,
                headless: *headless,
                json: *json,
                output: output.clone(),
                ci: *ci,
                throughput_threshold: *throughput,
                brick_score_threshold: *brick_score,
                warmup: *warmup,
                iterations: *iterations,
                speculative: *speculative,
                speculation_k: *speculation_k,
                draft_model_path: draft_model.clone(),
                concurrent: *concurrent,
                simulated: *simulated,
            })
        }

        Commands::Probar {
            file,
            output,
            format,
            golden,
            layer,
        } => {
            let export_format = format.parse().unwrap_or(probar::ExportFormat::Both);
            probar::run(
                file,
                output,
                export_format,
                golden.as_deref(),
                layer.as_deref(),
            )
        }

        Commands::CompareHf {
            file,
            hf,
            tensor,
            threshold,
            json,
        } => compare_hf::run(file, hf, tensor.as_deref(), *threshold, *json || cli.json),

        Commands::Hex {
            file,
            tensor,
            limit,
            stats,
            list,
            json,
        } => hex::run(
            file,
            tensor.as_deref(),
            *limit,
            *stats,
            *list,
            *json || cli.json,
        ),

        Commands::Tree {
            file,
            filter,
            format,
            sizes,
            depth,
        } => {
            let tree_format = format.parse().unwrap_or(tree::TreeFormat::Ascii);
            tree::run(file, filter.as_deref(), tree_format, *sizes, *depth)
        }

        Commands::Flow {
            file,
            layer,
            component,
            verbose,
        } => {
            let flow_component = component.parse().unwrap_or(flow::FlowComponent::Full);
            flow::run(
                file,
                layer.as_deref(),
                flow_component,
                *verbose || cli.verbose,
            )
        }

        Commands::Chat {
            file,
            temperature,
            top_p,
            max_tokens,
            system,
            inspect,
            no_gpu,
            gpu: _,
            trace,
            trace_steps,
            trace_verbose,
            trace_output,
            trace_level,
            profile,
        } => chat::run(
            file,
            *temperature,
            *top_p,
            *max_tokens,
            system.as_deref(),
            *inspect,
            *no_gpu,
            *trace,
            trace_steps.as_deref(),
            *trace_verbose,
            trace_output.clone(),
            trace_level.as_str(),
            *profile,
        ),

        Commands::Bench {
            file,
            warmup,
            iterations,
            max_tokens,
            prompt,
            fast,
            brick,
        } => bench::run(
            file,
            *warmup,
            *iterations,
            *max_tokens,
            prompt.as_deref(),
            *fast,
            brick.as_deref(),
        ),

        Commands::Eval {
            file,
            dataset,
            text,
            max_tokens,
            threshold,
        } => eval::run(
            file,
            dataset,
            text.as_deref(),
            Some(*max_tokens),
            Some(*threshold),
        ),

        Commands::Profile {
            file,
            granular,
            format,
            focus,
            detect_naive,
            threshold,
            compare_hf,
            energy,
            perf_grade,
            callgraph,
            fail_on_naive,
            output,
            ci,
            assert_throughput,
            assert_p99,
            assert_p50,
            warmup,
            measure,
        } => {
            let output_format = format.parse().unwrap_or(profile::OutputFormat::Human);

            // PMAT-192: CI mode takes precedence
            if *ci || assert_throughput.is_some() || assert_p99.is_some() || assert_p50.is_some() {
                let assertions = profile::CiAssertions {
                    min_throughput: *assert_throughput,
                    max_p99_ms: *assert_p99,
                    max_p50_ms: *assert_p50,
                    max_memory_mb: None,
                };
                match profile::run_ci(file, output_format, &assertions, *warmup, *measure) {
                    Ok(true) => Ok(()),
                    Ok(false) => {
                        // Exit with error code for CI pipeline failure
                        std::process::exit(1);
                    }
                    Err(e) => Err(e),
                }
            } else {
                // Standard profiling mode
                let profile_focus = focus
                    .as_ref()
                    .and_then(|f| f.parse().ok())
                    .unwrap_or(profile::ProfileFocus::All);
                profile::run(
                    file,
                    *granular,
                    output_format,
                    profile_focus,
                    *detect_naive,
                    *threshold,
                    compare_hf.as_deref(),
                    *energy,
                    *perf_grade,
                    *callgraph,
                    *fail_on_naive,
                    output.as_deref(),
                )
            }
        }

        Commands::Qa {
            file,
            assert_tps,
            assert_speedup,
            assert_gpu_speedup,
            skip_golden,
            skip_throughput,
            skip_ollama,
            skip_gpu_speedup,
            skip_contract,
            skip_format_parity,
            safetensors_path,
            iterations,
            warmup,
            max_tokens,
            json,
            verbose,
        } => qa::run(
            file,
            *assert_tps,
            *assert_speedup,
            *assert_gpu_speedup,
            *skip_golden,
            *skip_throughput,
            *skip_ollama,
            *skip_gpu_speedup,
            *skip_contract,
            *skip_format_parity,
            safetensors_path.clone(),
            *iterations,
            *warmup,
            *max_tokens,
            *json || cli.json,
            *verbose || cli.verbose,
        ),

        Commands::Tune {
            file,
            method,
            rank,
            vram,
            plan,
            model,
            freeze_base,
            train_data,
            json,
        } => {
            let tune_method = method.parse().unwrap_or(tune::TuneMethod::Auto);
            tune::run(
                file.as_deref(),
                tune_method,
                *rank,
                *vram,
                *plan,
                model.as_deref(),
                *freeze_base,
                train_data.as_deref(),
                *json || cli.json,
            )
        }

        Commands::Showcase {
            auto_verify,
            step,
            tier,
            model_dir,
            baseline,
            zram,
            runs,
            gpu,
            json,
            verbose,
            quiet,
        } => {
            let step = step.as_ref().and_then(|s| match s.as_str() {
                "import" => Some(showcase::ShowcaseStep::Import),
                "gguf" => Some(showcase::ShowcaseStep::GgufInference),
                "convert" => Some(showcase::ShowcaseStep::Convert),
                "apr" => Some(showcase::ShowcaseStep::AprInference),
                "bench" => Some(showcase::ShowcaseStep::Benchmark),
                "chat" => Some(showcase::ShowcaseStep::Chat),
                "visualize" => Some(showcase::ShowcaseStep::Visualize),
                "zram" => Some(showcase::ShowcaseStep::ZramDemo),
                "cuda" => Some(showcase::ShowcaseStep::CudaDemo),
                "brick" => Some(showcase::ShowcaseStep::BrickDemo),
                "all" => Some(showcase::ShowcaseStep::All),
                _ => None,
            });

            let tier = match tier.as_str() {
                "tiny" => showcase::ModelTier::Tiny,
                "small" => showcase::ModelTier::Small,
                "medium" => showcase::ModelTier::Medium,
                "large" => showcase::ModelTier::Large,
                _ => showcase::ModelTier::Small,
            };

            let baselines: Vec<showcase::Baseline> = baseline
                .split(',')
                .filter_map(|b| match b.trim() {
                    "llama-cpp" => Some(showcase::Baseline::LlamaCpp),
                    "ollama" => Some(showcase::Baseline::Ollama),
                    _ => None,
                })
                .collect();

            let export_format = if *json {
                showcase::ExportFormat::Json
            } else {
                showcase::ExportFormat::None
            };

            let config = showcase::ShowcaseConfig {
                tier,
                model: tier.model_path().to_string(),
                quant: "Q4_K_M".to_string(),
                model_dir: model_dir.clone(),
                auto_verify: *auto_verify,
                step,
                baselines,
                zram: *zram,
                bench_runs: *runs,
                export_format,
                export_path: None,
                gpu: *gpu,
                verbose: *verbose,
                quiet: *quiet,
            };

            showcase::run(&config)
        }

        Commands::Rosetta { action } => match action {
            RosettaCommands::Inspect {
                file,
                hexdump,
                json,
            } => rosetta::run_inspect(file, *hexdump, *json || cli.json),
            RosettaCommands::Convert {
                source,
                target,
                quantize,
                verify,
                json,
            } => rosetta::run_convert(
                source,
                target,
                quantize.as_deref(),
                *verify,
                *json || cli.json,
            ),
            RosettaCommands::Chain {
                source,
                formats,
                work_dir,
                json,
            } => rosetta::run_chain(source, formats, work_dir, *json || cli.json),
            RosettaCommands::Verify {
                source,
                intermediate,
                tolerance,
                json,
            } => rosetta::run_verify(source, intermediate, *tolerance, *json || cli.json),
            RosettaCommands::CompareInference {
                model_a,
                model_b,
                prompt,
                max_tokens,
                temperature,
                tolerance,
                json,
            } => rosetta::run_compare_inference(
                model_a,
                model_b,
                prompt,
                *max_tokens,
                *temperature,
                *tolerance,
                *json || cli.json,
            ),
            RosettaCommands::DiffTensors {
                model_a,
                model_b,
                mismatches_only,
                show_values,
                filter,
                json,
            } => rosetta::run_diff_tensors(
                model_a,
                model_b,
                *mismatches_only,
                *show_values,
                filter.as_deref(),
                *json || cli.json,
            ),
            RosettaCommands::Fingerprint {
                model,
                model_b,
                output,
                filter,
                verbose,
                json,
            } => rosetta::run_fingerprint(
                model,
                model_b.as_ref().map(std::path::PathBuf::as_path),
                output.as_ref().map(std::path::PathBuf::as_path),
                filter.as_deref(),
                *verbose,
                *json || cli.json,
            ),
            RosettaCommands::ValidateStats {
                model,
                reference,
                fingerprints,
                threshold,
                strict,
                json,
            } => rosetta::run_validate_stats(
                model,
                reference.as_ref().map(std::path::PathBuf::as_path),
                fingerprints.as_ref().map(std::path::PathBuf::as_path),
                *threshold,
                *strict,
                *json || cli.json,
            ),
        },

        Commands::Publish {
            directory,
            repo_id,
            model_name,
            license,
            pipeline_tag,
            library_name,
            tags,
            message,
            dry_run,
        } => publish::execute(
            directory,
            repo_id,
            model_name.as_deref(),
            license,
            pipeline_tag,
            library_name.as_deref(),
            tags.as_ref().map_or(&[], std::vec::Vec::as_slice),
            message.as_deref(),
            *dry_run,
            cli.verbose,
        ),

        Commands::Oracle {
            source,
            family,
            size,
            compliance,
            tensors,
            stats,
            explain,
            kernels,
            validate,
            full,
        } => oracle::run(
            source.as_ref(),
            family.as_ref(),
            size.as_ref(),
            *compliance,
            *tensors,
            cli.json,
            cli.verbose,
            cli.offline,
            oracle::OracleFlags {
                stats: *stats,
                explain: *explain,
                kernels: *kernels,
                validate: *validate,
                full: *full,
            },
        ),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse CLI args on a thread with 16 MB stack.
    /// Clap's parser for 34 subcommands exceeds the default test-thread
    /// stack in debug builds.
    fn parse_cli(args: Vec<&'static str>) -> Result<Cli, clap::error::Error> {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(move || Cli::try_parse_from(args))
            .expect("spawn thread")
            .join()
            .expect("join thread")
    }

    /// Test CLI parsing with clap's debug_assert
    #[test]
    fn test_cli_parsing_valid() {
        use clap::CommandFactory;
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| Cli::command().debug_assert())
            .expect("spawn")
            .join()
            .expect("join");
    }

    /// Test parsing 'apr inspect' command
    #[test]
    fn test_parse_inspect_command() {
        let args = vec!["apr", "inspect", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Inspect { file, .. } => {
                assert_eq!(file, PathBuf::from("model.apr"));
            }
            _ => panic!("Expected Inspect command"),
        }
    }

    /// Test parsing 'apr inspect' with flags
    #[test]
    fn test_parse_inspect_with_flags() {
        let args = vec!["apr", "inspect", "model.apr", "--vocab", "--json"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Inspect {
                file, vocab, json, ..
            } => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert!(vocab);
                assert!(json);
            }
            _ => panic!("Expected Inspect command"),
        }
    }

    /// Test parsing 'apr serve' command
    #[test]
    fn test_parse_serve_command() {
        let args = vec!["apr", "serve", "model.apr", "--port", "3000"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Serve { file, port, .. } => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert_eq!(port, 3000);
            }
            _ => panic!("Expected Serve command"),
        }
    }

    /// Test parsing 'apr run' command
    #[test]
    fn test_parse_run_command() {
        let args = vec![
            "apr",
            "run",
            "hf://openai/whisper-tiny",
            "--prompt",
            "Hello",
            "--max-tokens",
            "64",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run {
                source,
                prompt,
                max_tokens,
                ..
            } => {
                assert_eq!(source, "hf://openai/whisper-tiny");
                assert_eq!(prompt, Some("Hello".to_string()));
                assert_eq!(max_tokens, 64);
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// Test parsing 'apr chat' command
    #[test]
    fn test_parse_chat_command() {
        let args = vec![
            "apr",
            "chat",
            "model.gguf",
            "--temperature",
            "0.5",
            "--top-p",
            "0.95",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Chat {
                file,
                temperature,
                top_p,
                ..
            } => {
                assert_eq!(file, PathBuf::from("model.gguf"));
                assert!((temperature - 0.5).abs() < f32::EPSILON);
                assert!((top_p - 0.95).abs() < f32::EPSILON);
            }
            _ => panic!("Expected Chat command"),
        }
    }

    /// Test parsing 'apr validate' command with quality flag
    #[test]
    fn test_parse_validate_with_quality() {
        let args = vec!["apr", "validate", "model.apr", "--quality", "--strict"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Validate {
                file,
                quality,
                strict,
                ..
            } => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert!(quality);
                assert!(strict);
            }
            _ => panic!("Expected Validate command"),
        }
    }

    /// Test parsing 'apr diff' command
    #[test]
    fn test_parse_diff_command() {
        let args = vec!["apr", "diff", "model1.apr", "model2.apr", "--weights"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Diff {
                file1,
                file2,
                weights,
                ..
            } => {
                assert_eq!(file1, PathBuf::from("model1.apr"));
                assert_eq!(file2, PathBuf::from("model2.apr"));
                assert!(weights);
            }
            _ => panic!("Expected Diff command"),
        }
    }

    /// Test parsing 'apr bench' command
    #[test]
    fn test_parse_bench_command() {
        let args = vec![
            "apr",
            "bench",
            "model.gguf",
            "--warmup",
            "5",
            "--iterations",
            "10",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Bench {
                file,
                warmup,
                iterations,
                ..
            } => {
                assert_eq!(file, PathBuf::from("model.gguf"));
                assert_eq!(warmup, 5);
                assert_eq!(iterations, 10);
            }
            _ => panic!("Expected Bench command"),
        }
    }

    /// Test parsing 'apr cbtop' command with CI flags
    #[test]
    fn test_parse_cbtop_ci_mode() {
        let args = vec![
            "apr",
            "cbtop",
            "--headless",
            "--ci",
            "--throughput",
            "100.0",
            "--brick-score",
            "90",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Cbtop {
                headless,
                ci,
                throughput,
                brick_score,
                ..
            } => {
                assert!(headless);
                assert!(ci);
                assert_eq!(throughput, Some(100.0));
                assert_eq!(brick_score, Some(90));
            }
            _ => panic!("Expected Cbtop command"),
        }
    }

    /// Test parsing 'apr qa' command
    #[test]
    fn test_parse_qa_command() {
        let args = vec![
            "apr",
            "qa",
            "model.gguf",
            "--assert-tps",
            "50.0",
            "--skip-ollama",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Qa {
                file,
                assert_tps,
                skip_ollama,
                ..
            } => {
                assert_eq!(file, PathBuf::from("model.gguf"));
                assert_eq!(assert_tps, Some(50.0));
                assert!(skip_ollama);
            }
            _ => panic!("Expected Qa command"),
        }
    }

    /// Test global --verbose flag
    #[test]
    fn test_global_verbose_flag() {
        let args = vec!["apr", "--verbose", "inspect", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        assert!(cli.verbose);
    }

    /// Test global --json flag
    #[test]
    fn test_global_json_flag() {
        let args = vec!["apr", "--json", "inspect", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        assert!(cli.json);
    }

    /// Test parsing 'apr list' command (alias 'ls')
    #[test]
    fn test_parse_list_command() {
        let args = vec!["apr", "list"];
        let cli = parse_cli(args).expect("Failed to parse");
        assert!(matches!(*cli.command, Commands::List));
    }

    /// Test parsing 'apr ls' alias
    #[test]
    fn test_parse_ls_alias() {
        let args = vec!["apr", "ls"];
        let cli = parse_cli(args).expect("Failed to parse");
        assert!(matches!(*cli.command, Commands::List));
    }

    /// Test parsing 'apr rm' command (alias 'remove')
    #[test]
    fn test_parse_rm_command() {
        let args = vec!["apr", "rm", "model-name"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Rm { model_ref } => {
                assert_eq!(model_ref, "model-name");
            }
            _ => panic!("Expected Rm command"),
        }
    }

    /// Test invalid command fails parsing
    #[test]
    fn test_invalid_command() {
        let args = vec!["apr", "invalid-command"];
        let result = parse_cli(args);
        assert!(result.is_err());
    }

    /// Test missing required argument fails
    #[test]
    fn test_missing_required_arg() {
        let args = vec!["apr", "inspect"]; // Missing FILE
        let result = parse_cli(args);
        assert!(result.is_err());
    }

    /// Test parsing 'apr merge' with multiple files and weights
    #[test]
    fn test_parse_merge_command() {
        let args = vec![
            "apr",
            "merge",
            "model1.apr",
            "model2.apr",
            "--strategy",
            "weighted",
            "--weights",
            "0.7,0.3",
            "-o",
            "merged.apr",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Merge {
                files,
                strategy,
                output,
                weights,
            } => {
                assert_eq!(files.len(), 2);
                assert_eq!(strategy, "weighted");
                assert_eq!(output, PathBuf::from("merged.apr"));
                assert_eq!(weights, Some(vec![0.7, 0.3]));
            }
            _ => panic!("Expected Merge command"),
        }
    }

    /// Test parsing 'apr showcase' command
    #[test]
    fn test_parse_showcase_command() {
        let args = vec![
            "apr",
            "showcase",
            "--tier",
            "medium",
            "--gpu",
            "--auto-verify",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Showcase {
                tier,
                gpu,
                auto_verify,
                ..
            } => {
                assert_eq!(tier, "medium");
                assert!(gpu);
                assert!(auto_verify);
            }
            _ => panic!("Expected Showcase command"),
        }
    }

    /// Test parsing 'apr profile' with all options
    #[test]
    fn test_parse_profile_command() {
        let args = vec![
            "apr",
            "profile",
            "model.apr",
            "--granular",
            "--detect-naive",
            "--fail-on-naive",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Profile {
                file,
                granular,
                detect_naive,
                fail_on_naive,
                ..
            } => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert!(granular);
                assert!(detect_naive);
                assert!(fail_on_naive);
            }
            _ => panic!("Expected Profile command"),
        }
    }

    /// Test parsing 'apr profile' with CI assertions (PMAT-192, GH-180)
    #[test]
    fn test_parse_profile_ci_mode() {
        let args = vec![
            "apr",
            "profile",
            "model.gguf",
            "--ci",
            "--assert-throughput",
            "100",
            "--assert-p99",
            "50",
            "--format",
            "json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Profile {
                file,
                ci,
                assert_throughput,
                assert_p99,
                format,
                ..
            } => {
                assert_eq!(file, PathBuf::from("model.gguf"));
                assert!(ci);
                assert_eq!(assert_throughput, Some(100.0));
                assert_eq!(assert_p99, Some(50.0));
                assert_eq!(format, "json");
            }
            _ => panic!("Expected Profile command"),
        }
    }

    /// Test parsing 'apr rosetta inspect' command
    #[test]
    fn test_parse_rosetta_inspect() {
        let args = vec!["apr", "rosetta", "inspect", "model.gguf", "--json"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Rosetta { action } => match action {
                RosettaCommands::Inspect { file, json, .. } => {
                    assert_eq!(file, PathBuf::from("model.gguf"));
                    assert!(json);
                }
                _ => panic!("Expected Inspect subcommand"),
            },
            _ => panic!("Expected Rosetta command"),
        }
    }

    /// Test parsing 'apr rosetta convert' command
    #[test]
    fn test_parse_rosetta_convert() {
        let args = vec![
            "apr",
            "rosetta",
            "convert",
            "model.gguf",
            "model.safetensors",
            "--verify",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Rosetta { action } => match action {
                RosettaCommands::Convert {
                    source,
                    target,
                    verify,
                    ..
                } => {
                    assert_eq!(source, PathBuf::from("model.gguf"));
                    assert_eq!(target, PathBuf::from("model.safetensors"));
                    assert!(verify);
                }
                _ => panic!("Expected Convert subcommand"),
            },
            _ => panic!("Expected Rosetta command"),
        }
    }

    /// Test parsing 'apr rosetta chain' command
    #[test]
    fn test_parse_rosetta_chain() {
        let args = vec![
            "apr",
            "rosetta",
            "chain",
            "model.gguf",
            "safetensors",
            "apr",
            "--work-dir",
            "/tmp/rosetta",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Rosetta { action } => match action {
                RosettaCommands::Chain {
                    source,
                    formats,
                    work_dir,
                    ..
                } => {
                    assert_eq!(source, PathBuf::from("model.gguf"));
                    assert_eq!(formats, vec!["safetensors", "apr"]);
                    assert_eq!(work_dir, PathBuf::from("/tmp/rosetta"));
                }
                _ => panic!("Expected Chain subcommand"),
            },
            _ => panic!("Expected Rosetta command"),
        }
    }

    /// Test parsing 'apr rosetta verify' command
    #[test]
    fn test_parse_rosetta_verify() {
        let args = vec![
            "apr",
            "rosetta",
            "verify",
            "model.apr",
            "--intermediate",
            "gguf",
            "--tolerance",
            "1e-4",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Rosetta { action } => match action {
                RosettaCommands::Verify {
                    source,
                    intermediate,
                    tolerance,
                    ..
                } => {
                    assert_eq!(source, PathBuf::from("model.apr"));
                    assert_eq!(intermediate, "gguf");
                    assert!((tolerance - 1e-4).abs() < f32::EPSILON);
                }
                _ => panic!("Expected Verify subcommand"),
            },
            _ => panic!("Expected Rosetta command"),
        }
    }

    // =========================================================================
    // PMAT-237: Contract gate tests
    // =========================================================================

    /// Test that --skip-contract global flag is parsed
    #[test]
    fn test_parse_skip_contract_flag() {
        let args = vec!["apr", "--skip-contract", "inspect", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        assert!(cli.skip_contract);
    }

    /// Test that --skip-contract defaults to false
    #[test]
    fn test_skip_contract_default_false() {
        let args = vec!["apr", "inspect", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        assert!(!cli.skip_contract);
    }

    /// Test extract_model_paths: diagnostic commands return empty vec
    #[test]
    fn test_extract_paths_diagnostic_exempt() {
        // Diagnostic commands should return no paths (exempt from validation)
        let diagnostic_commands = vec![
            Commands::Inspect {
                file: PathBuf::from("m.apr"),
                vocab: false,
                filters: false,
                weights: false,
                json: false,
            },
            Commands::Debug {
                file: PathBuf::from("m.apr"),
                drama: false,
                hex: false,
                strings: false,
                limit: 256,
            },
            Commands::Validate {
                file: PathBuf::from("m.apr"),
                quality: false,
                strict: false,
                min_score: None,
            },
            Commands::Tensors {
                file: PathBuf::from("m.apr"),
                stats: false,
                filter: None,
                limit: 0,
                json: false,
            },
            Commands::Lint {
                file: PathBuf::from("m.apr"),
            },
            Commands::Qa {
                file: PathBuf::from("m.apr"),
                assert_tps: None,
                assert_speedup: None,
                assert_gpu_speedup: None,
                skip_golden: false,
                skip_throughput: false,
                skip_ollama: false,
                skip_gpu_speedup: false,
                skip_contract: false,
                skip_format_parity: false,
                safetensors_path: None,
                iterations: 10,
                warmup: 3,
                max_tokens: 32,
                json: false,
                verbose: false,
            },
            Commands::Hex {
                file: PathBuf::from("m.apr"),
                tensor: None,
                limit: 64,
                stats: false,
                list: false,
                json: false,
            },
            Commands::Tree {
                file: PathBuf::from("m.apr"),
                filter: None,
                format: "ascii".to_string(),
                sizes: false,
                depth: None,
            },
            Commands::Flow {
                file: PathBuf::from("m.apr"),
                layer: None,
                component: "full".to_string(),
                verbose: false,
            },
            Commands::Explain {
                code: None,
                file: None,
                tensor: None,
            },
            Commands::List,
        ];
        for cmd in &diagnostic_commands {
            let paths = extract_model_paths(cmd);
            assert!(
                paths.is_empty(),
                "Diagnostic command should be exempt: {cmd:?}"
            );
        }
    }

    /// Test extract_model_paths: action commands return file paths
    #[test]
    fn test_extract_paths_action_commands() {
        let serve_cmd = Commands::Serve {
            file: PathBuf::from("model.gguf"),
            port: 8080,
            host: "127.0.0.1".to_string(),
            no_cors: false,
            no_metrics: false,
            no_gpu: false,
            gpu: false,
            batch: false,
            trace: false,
            trace_level: "basic".to_string(),
            profile: false,
        };
        let paths = extract_model_paths(&serve_cmd);
        assert_eq!(paths, vec![PathBuf::from("model.gguf")]);

        let bench_cmd = Commands::Bench {
            file: PathBuf::from("model.apr"),
            warmup: 3,
            iterations: 5,
            max_tokens: 32,
            prompt: None,
            fast: false,
            brick: None,
        };
        let paths = extract_model_paths(&bench_cmd);
        assert_eq!(paths, vec![PathBuf::from("model.apr")]);
    }

    /// Test extract_model_paths: Run with hf:// URL returns empty
    #[test]
    fn test_extract_paths_run_hf_url() {
        let cmd = Commands::Run {
            source: "hf://org/repo".to_string(),
            input: None,
            prompt: None,
            max_tokens: 32,
            stream: false,
            language: None,
            task: None,
            format: "text".to_string(),
            no_gpu: false,
            gpu: false,
            offline: false,
            benchmark: false,
            trace: false,
            trace_steps: None,
            trace_verbose: false,
            trace_output: None,
            trace_level: "basic".to_string(),
            trace_payload: false,
            profile: false,
            chat: false,
            verbose: false,
        };
        let paths = extract_model_paths(&cmd);
        assert!(
            paths.is_empty(),
            "hf:// URLs should not be validated locally"
        );
    }

    /// Test extract_model_paths: Merge returns multiple files
    #[test]
    fn test_extract_paths_merge_multiple() {
        let cmd = Commands::Merge {
            files: vec![
                PathBuf::from("a.apr"),
                PathBuf::from("b.apr"),
                PathBuf::from("c.apr"),
            ],
            strategy: "average".to_string(),
            output: PathBuf::from("merged.apr"),
            weights: None,
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths.len(), 3);
    }

    /// Test validate_model_contract: non-existent path is skipped (Ok)
    #[test]
    fn test_validate_contract_nonexistent_skipped() {
        let paths = vec![PathBuf::from("nonexistent_model_xyz.apr")];
        let result = validate_model_contract(&paths);
        assert!(result.is_ok(), "Non-existent paths should be skipped");
    }

    /// Test validate_model_contract: empty paths is Ok
    #[test]
    fn test_validate_contract_empty_paths() {
        let result = validate_model_contract(&[]);
        assert!(result.is_ok());
    }
}
