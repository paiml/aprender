//! apr-cli library
//!
//! This library is the foundation for the apr CLI binary.
//! Exports CLI structures for testing and reuse.

use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

mod commands;
pub mod error;
mod output;

pub use error::CliError;

#[cfg(feature = "inference")]
pub mod federation;

// Commands are crate-private, used internally by execute_command
use commands::{
    bench, canary, canary::CanaryCommands, cbtop, chat, compare_hf, convert, debug, diff, eval,
    explain, export, flow, hex, import, inspect, lint, merge, oracle, probar, profile, ptx_explain,
    publish, pull, qa, rosetta, rosetta::RosettaCommands, run, serve, showcase, tensors, trace,
    tree, tui, tune, validate,
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

        /// Text prompt (positional): `apr run model.gguf "What is 2+2?"`
        #[arg(value_name = "PROMPT")]
        positional_prompt: Option<String>,

        /// Input file (audio, text, etc.)
        #[arg(short, long)]
        input: Option<PathBuf>,

        /// Text prompt for generation (for LLM models)
        #[arg(short, long)]
        prompt: Option<String>,

        /// Maximum tokens to generate (default: 32)
        #[arg(short = 'n', long, default_value = "32")]
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

        /// F-GT-001: Enforce provenance chain. Rejects pre-baked GGUF imports
        /// (only SafeTensors sources allowed). Ensures single-provenance testing.
        #[arg(long)]
        enforce_provenance: bool,
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

    /// Format-aware binary forensics (10X better than xxd)
    Hex {
        /// Path to model file (APR, GGUF, or SafeTensors)
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Filter tensors by name pattern
        #[arg(long)]
        tensor: Option<String>,

        /// Limit bytes/values to display
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

        /// Annotated file header (magic, version, tensor count, metadata)
        #[arg(long)]
        header: bool,

        /// Q4K/Q6K/Q8_0 super-block structure with field annotations
        #[arg(long)]
        blocks: bool,

        /// Value histogram + entropy + kurtosis analysis
        #[arg(long)]
        distribution: bool,

        /// Layout contract verification overlay per tensor
        #[arg(long)]
        contract: bool,

        /// Per-region byte entropy analysis
        #[arg(long)]
        entropy: bool,

        /// Raw bytes (like xxd but format-aware, with ASCII column)
        #[arg(long)]
        raw: bool,

        /// Start at byte offset (supports 0x prefix for hex)
        #[arg(long, default_value = "0")]
        offset: String,

        /// Bytes per row for raw output (default: 16)
        #[arg(long, default_value = "16")]
        width: usize,
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

        /// Compute performance grade (vs Ollama baseline)
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

        /// Number of tokens to generate per measurement pass (default: 32)
        #[arg(long, default_value = "32")]
        tokens: usize,

        /// Compare against Ollama baseline (runs ollama for comparison)
        #[arg(long)]
        ollama: bool,

        /// Disable GPU (force CPU-only profiling)
        #[arg(long)]
        no_gpu: bool,
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

        /// Skip PTX parity validation (GH-219)
        #[arg(long)]
        skip_ptx_parity: bool,

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

    /// GPU/CPU parity check (PMAT-232: genchi genbutsu — see where GPU diverges)
    Parity {
        /// Path to GGUF model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Prompt text (default: "What is 2+2?")
        #[arg(short, long, default_value = "What is 2+2?")]
        prompt: String,

        /// Assert parity (exit non-zero on divergence)
        #[arg(long)]
        assert: bool,
    },

    /// Model-to-PTX source mapping (Mieruka: make GPU kernel dispatch visible)
    #[command(name = "ptx-map")]
    PtxMap {
        /// Path to GGUF model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Filter to specific kernel (e.g., --kernel Q4KGemv)
        #[arg(long)]
        kernel: Option<String>,

        /// Reverse lookup: kernel name -> which layers/steps use it
        #[arg(long)]
        reverse: Option<String>,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Full PTX snippets and detailed analysis
        #[arg(short, long)]
        verbose: bool,

        /// Show batched prefill kernel variants instead of decode
        #[arg(long)]
        prefill: bool,
    },

    /// PTX analysis and bug detection (trueno-explain: register pressure, roofline, 15+ bug detectors)
    #[command(name = "ptx")]
    Ptx {
        /// Path to a PTX source file
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,

        /// Analyze a named kernel from trueno-gpu
        #[arg(long, short)]
        kernel: Option<String>,

        /// Strict mode (no performance whitelist)
        #[arg(long)]
        strict: bool,

        /// Show only bug analysis (skip register/memory/roofline)
        #[arg(long)]
        bugs: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Verbose output (include PTX source listing)
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

/// Dispatch `apr cbtop` — extracted to reduce cognitive complexity of `execute_command`
#[allow(clippy::too_many_arguments)]
fn dispatch_cbtop(
    model: Option<&str>,
    attach: Option<&str>,
    model_path: Option<&Path>,
    headless: bool,
    json: bool,
    output: Option<&Path>,
    ci: bool,
    throughput: Option<f64>,
    brick_score: Option<u32>,
    warmup: usize,
    iterations: usize,
    speculative: bool,
    speculation_k: usize,
    draft_model: Option<&Path>,
    concurrent: usize,
    simulated: bool,
) -> Result<(), CliError> {
    let (resolved_model, resolved_model_path) = if let Some(m) = model {
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
            (Some(m.to_string()), model_path.map(PathBuf::from))
        }
    } else {
        (None, model_path.map(PathBuf::from))
    };

    cbtop::run(cbtop::CbtopConfig {
        model: resolved_model,
        attach: attach.map(String::from),
        model_path: resolved_model_path,
        headless,
        json,
        output: output.map(PathBuf::from),
        ci,
        throughput_threshold: throughput,
        brick_score_threshold: brick_score,
        warmup,
        iterations,
        speculative,
        speculation_k,
        draft_model_path: draft_model.map(PathBuf::from),
        concurrent,
        simulated,
    })
}

/// Dispatch `apr showcase` — extracted to reduce cognitive complexity of `execute_command`
#[allow(clippy::too_many_arguments)]
fn dispatch_showcase(
    auto_verify: bool,
    step: Option<&str>,
    tier: &str,
    model_dir: &Path,
    baseline: &str,
    zram: bool,
    runs: usize,
    gpu: bool,
    json: bool,
    verbose: bool,
    quiet: bool,
) -> Result<(), CliError> {
    let step = step.and_then(|s| match s {
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

    let tier = match tier {
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

    let export_format = if json {
        showcase::ExportFormat::Json
    } else {
        showcase::ExportFormat::None
    };

    let config = showcase::ShowcaseConfig {
        tier,
        model: tier.model_path().to_string(),
        quant: "Q4_K_M".to_string(),
        model_dir: model_dir.to_path_buf(),
        auto_verify,
        step,
        baselines,
        zram,
        bench_runs: runs,
        export_format,
        export_path: None,
        gpu,
        verbose,
        quiet,
    };

    showcase::run(&config)
}

/// Dispatch `apr profile` — extracted to reduce cognitive complexity of `execute_command`
#[allow(clippy::too_many_arguments)]
fn dispatch_profile(
    file: &Path,
    granular: bool,
    format: &str,
    focus: Option<&str>,
    detect_naive: bool,
    threshold: f64,
    compare_hf: Option<&str>,
    energy: bool,
    perf_grade: bool,
    callgraph: bool,
    fail_on_naive: bool,
    output: Option<&Path>,
    ci: bool,
    assert_throughput: Option<f64>,
    assert_p99: Option<f64>,
    assert_p50: Option<f64>,
    warmup: usize,
    measure: usize,
    tokens: usize,
    ollama: bool,
    no_gpu: bool,
) -> Result<(), CliError> {
    let output_format = format.parse().unwrap_or(profile::OutputFormat::Human);

    // PMAT-192: CI mode takes precedence
    if ci || assert_throughput.is_some() || assert_p99.is_some() || assert_p50.is_some() {
        let assertions = profile::CiAssertions {
            min_throughput: assert_throughput,
            max_p99_ms: assert_p99,
            max_p50_ms: assert_p50,
            max_memory_mb: None,
        };
        match profile::run_ci(file, output_format, &assertions, warmup, measure) {
            Ok(true) => Ok(()),
            Ok(false) => {
                std::process::exit(1);
            }
            Err(e) => Err(e),
        }
    } else {
        let profile_focus = focus
            .and_then(|f| f.parse().ok())
            .unwrap_or(profile::ProfileFocus::All);
        profile::run(
            file,
            granular,
            output_format,
            profile_focus,
            detect_naive,
            threshold,
            compare_hf,
            energy,
            perf_grade,
            callgraph,
            fail_on_naive,
            output,
            tokens,
            ollama,
            no_gpu,
        )
    }
}

/// Dispatch `apr run` — extracted to reduce cognitive complexity of `execute_command`
#[allow(clippy::too_many_arguments)]
fn dispatch_run(
    source: &str,
    positional_prompt: Option<&String>,
    input: Option<&Path>,
    prompt: Option<&String>,
    max_tokens: usize,
    stream: bool,
    language: Option<&str>,
    task: Option<&str>,
    format: &str,
    no_gpu: bool,
    offline: bool,
    benchmark: bool,
    verbose: bool,
    trace: bool,
    trace_payload: bool,
    trace_steps: Option<&[String]>,
    trace_verbose: bool,
    trace_output: Option<PathBuf>,
    trace_level: &str,
    profile: bool,
    chat: bool,
) -> Result<(), CliError> {
    let effective_trace = trace || trace_payload;
    let effective_trace_level = if trace_payload {
        "payload"
    } else {
        trace_level
    };
    let merged_prompt = prompt.or(positional_prompt).cloned();
    let effective_prompt = if chat {
        merged_prompt
            .as_ref()
            .map(|p| format!("<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"))
    } else {
        merged_prompt
    };

    run::run(
        source,
        input,
        effective_prompt.as_deref(),
        max_tokens,
        stream,
        language,
        task,
        format,
        no_gpu,
        offline,
        benchmark,
        verbose,
        effective_trace,
        trace_steps,
        trace_verbose,
        trace_output,
        effective_trace_level,
        profile,
    )
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
            positional_prompt,
            input,
            prompt,
            max_tokens,
            stream,
            language,
            task,
            format,
            no_gpu,
            gpu: _,
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
        } => dispatch_run(
            source,
            positional_prompt.as_ref(),
            input.as_deref(),
            prompt.as_ref(),
            *max_tokens,
            *stream,
            language.as_deref(),
            task.as_deref(),
            format,
            *no_gpu,
            *offline,
            *benchmark,
            *verbose || cli.verbose,
            *trace,
            *trace_payload,
            trace_steps.as_deref(),
            *trace_verbose,
            trace_output.clone(),
            trace_level.as_str(),
            *profile,
            *chat,
        ),

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
            enforce_provenance,
        } => import::run(
            source,
            output.as_deref(),
            Some(arch.as_str()),
            quantize.as_deref(),
            *strict,
            *preserve_q4k,
            tokenizer.as_ref(),
            *enforce_provenance,
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
        } => dispatch_cbtop(
            model.as_deref(),
            attach.as_deref(),
            model_path.as_deref(),
            *headless,
            *json,
            output.as_deref(),
            *ci,
            *throughput,
            *brick_score,
            *warmup,
            *iterations,
            *speculative,
            *speculation_k,
            draft_model.as_deref(),
            *concurrent,
            *simulated,
        ),

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
            header,
            blocks,
            distribution,
            contract,
            entropy,
            raw,
            offset,
            width,
        } => {
            let parsed_offset = hex::parse_hex_offset(offset).map_err(CliError::InvalidFormat)?;
            hex::run(&hex::HexOptions {
                file: file.clone(),
                tensor: tensor.clone(),
                limit: *limit,
                stats: *stats,
                list: *list,
                json: *json || cli.json,
                header: *header,
                blocks: *blocks,
                distribution: *distribution,
                contract: *contract,
                entropy: *entropy,
                raw: *raw,
                offset: parsed_offset,
                width: *width,
            })
        }

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
            tokens,
            ollama,
            no_gpu,
        } => dispatch_profile(
            file,
            *granular,
            format,
            focus.as_deref(),
            *detect_naive,
            *threshold,
            compare_hf.as_deref(),
            *energy,
            *perf_grade,
            *callgraph,
            *fail_on_naive,
            output.as_deref(),
            *ci,
            *assert_throughput,
            *assert_p99,
            *assert_p50,
            *warmup,
            *measure,
            *tokens,
            *ollama,
            *no_gpu,
        ),

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
            skip_ptx_parity,
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
            *skip_ptx_parity,
            safetensors_path.clone(),
            *iterations,
            *warmup,
            *max_tokens,
            *json || cli.json,
            *verbose || cli.verbose,
        ),

        Commands::Parity {
            file,
            prompt,
            assert,
        } => commands::parity::run(file, prompt, *assert, cli.verbose),

        Commands::PtxMap {
            file,
            kernel,
            reverse,
            json,
            verbose,
            prefill,
        } => commands::ptx_map::run(
            file,
            kernel.as_deref(),
            reverse.as_deref(),
            *json || cli.json,
            *verbose || cli.verbose,
            *prefill,
        ),

        Commands::Ptx {
            file,
            kernel,
            strict,
            bugs,
            json,
            verbose,
        } => ptx_explain::run(
            file.as_deref(),
            kernel.as_deref(),
            *strict,
            *bugs,
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
        } => dispatch_showcase(
            *auto_verify,
            step.as_deref(),
            tier,
            model_dir,
            baseline,
            *zram,
            *runs,
            *gpu,
            *json,
            *verbose,
            *quiet,
        ),

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
                tokenizer,
            } => rosetta::run_convert(
                source,
                target,
                quantize.as_deref(),
                *verify,
                *json || cli.json,
                tokenizer.as_deref(),
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
                skip_ptx_parity: false,
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
                header: false,
                blocks: false,
                distribution: false,
                contract: false,
                entropy: false,
                raw: false,
                offset: "0".to_string(),
                width: 16,
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
            positional_prompt: None,
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

    // =========================================================================
    // Parse tests for all remaining command variants
    // =========================================================================

    /// Test parsing 'apr publish' command with all options
    #[test]
    fn test_parse_publish_command() {
        let args = vec![
            "apr",
            "publish",
            "/tmp/models",
            "paiml/whisper-apr-tiny",
            "--model-name",
            "Whisper Tiny",
            "--license",
            "apache-2.0",
            "--pipeline-tag",
            "automatic-speech-recognition",
            "--library-name",
            "whisper-apr",
            "--tags",
            "whisper,tiny,asr",
            "--message",
            "Initial release",
            "--dry-run",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
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
            } => {
                assert_eq!(directory, PathBuf::from("/tmp/models"));
                assert_eq!(repo_id, "paiml/whisper-apr-tiny");
                assert_eq!(model_name, Some("Whisper Tiny".to_string()));
                assert_eq!(license, "apache-2.0");
                assert_eq!(pipeline_tag, "automatic-speech-recognition");
                assert_eq!(library_name, Some("whisper-apr".to_string()));
                assert_eq!(
                    tags,
                    Some(vec![
                        "whisper".to_string(),
                        "tiny".to_string(),
                        "asr".to_string()
                    ])
                );
                assert_eq!(message, Some("Initial release".to_string()));
                assert!(dry_run);
            }
            _ => panic!("Expected Publish command"),
        }
    }

    /// Test parsing 'apr publish' with defaults
    #[test]
    fn test_parse_publish_defaults() {
        let args = vec!["apr", "publish", "./models", "org/repo"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Publish {
                license,
                pipeline_tag,
                dry_run,
                model_name,
                library_name,
                tags,
                message,
                ..
            } => {
                assert_eq!(license, "mit");
                assert_eq!(pipeline_tag, "text-generation");
                assert!(!dry_run);
                assert!(model_name.is_none());
                assert!(library_name.is_none());
                assert!(tags.is_none());
                assert!(message.is_none());
            }
            _ => panic!("Expected Publish command"),
        }
    }

    /// Test parsing 'apr eval' command with all options
    #[test]
    fn test_parse_eval_command() {
        let args = vec![
            "apr",
            "eval",
            "model.gguf",
            "--dataset",
            "lambada",
            "--text",
            "The quick brown fox",
            "--max-tokens",
            "256",
            "--threshold",
            "15.5",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Eval {
                file,
                dataset,
                text,
                max_tokens,
                threshold,
            } => {
                assert_eq!(file, PathBuf::from("model.gguf"));
                assert_eq!(dataset, "lambada");
                assert_eq!(text, Some("The quick brown fox".to_string()));
                assert_eq!(max_tokens, 256);
                assert!((threshold - 15.5).abs() < f32::EPSILON);
            }
            _ => panic!("Expected Eval command"),
        }
    }

    /// Test parsing 'apr eval' with defaults
    #[test]
    fn test_parse_eval_defaults() {
        let args = vec!["apr", "eval", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Eval {
                dataset,
                text,
                max_tokens,
                threshold,
                ..
            } => {
                assert_eq!(dataset, "wikitext-2");
                assert!(text.is_none());
                assert_eq!(max_tokens, 512);
                assert!((threshold - 20.0).abs() < f32::EPSILON);
            }
            _ => panic!("Expected Eval command"),
        }
    }

    /// Test parsing 'apr flow' command with options
    #[test]
    fn test_parse_flow_command() {
        let args = vec![
            "apr",
            "flow",
            "model.apr",
            "--layer",
            "encoder.0",
            "--component",
            "encoder",
            "-v",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Flow {
                file,
                layer,
                component,
                verbose,
            } => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert_eq!(layer, Some("encoder.0".to_string()));
                assert_eq!(component, "encoder");
                assert!(verbose);
            }
            _ => panic!("Expected Flow command"),
        }
    }

    /// Test parsing 'apr flow' with defaults
    #[test]
    fn test_parse_flow_defaults() {
        let args = vec!["apr", "flow", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Flow {
                component,
                verbose,
                layer,
                ..
            } => {
                assert_eq!(component, "full");
                assert!(!verbose);
                assert!(layer.is_none());
            }
            _ => panic!("Expected Flow command"),
        }
    }

    /// Test parsing 'apr hex' command with all options
    #[test]
    fn test_parse_hex_command() {
        let args = vec![
            "apr",
            "hex",
            "model.apr",
            "--tensor",
            "embed.weight",
            "--limit",
            "128",
            "--stats",
            "--list",
            "--json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Hex {
                file,
                tensor,
                limit,
                stats,
                list,
                json,
                ..
            } => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert_eq!(tensor, Some("embed.weight".to_string()));
                assert_eq!(limit, 128);
                assert!(stats);
                assert!(list);
                assert!(json);
            }
            _ => panic!("Expected Hex command"),
        }
    }

    /// Test parsing 'apr hex' with defaults
    #[test]
    fn test_parse_hex_defaults() {
        let args = vec!["apr", "hex", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Hex {
                limit,
                stats,
                list,
                json,
                tensor,
                ..
            } => {
                assert_eq!(limit, 64);
                assert!(!stats);
                assert!(!list);
                assert!(!json);
                assert!(tensor.is_none());
            }
            _ => panic!("Expected Hex command"),
        }
    }

    /// Test parsing 'apr tree' command with options
    #[test]
    fn test_parse_tree_command() {
        let args = vec![
            "apr",
            "tree",
            "model.apr",
            "--filter",
            "encoder",
            "--format",
            "mermaid",
            "--sizes",
            "--depth",
            "3",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Tree {
                file,
                filter,
                format,
                sizes,
                depth,
            } => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert_eq!(filter, Some("encoder".to_string()));
                assert_eq!(format, "mermaid");
                assert!(sizes);
                assert_eq!(depth, Some(3));
            }
            _ => panic!("Expected Tree command"),
        }
    }

    /// Test parsing 'apr tree' with defaults
    #[test]
    fn test_parse_tree_defaults() {
        let args = vec!["apr", "tree", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Tree {
                format,
                sizes,
                depth,
                filter,
                ..
            } => {
                assert_eq!(format, "ascii");
                assert!(!sizes);
                assert!(depth.is_none());
                assert!(filter.is_none());
            }
            _ => panic!("Expected Tree command"),
        }
    }

    /// Test parsing 'apr probar' command with options
    #[test]
    fn test_parse_probar_command() {
        let args = vec![
            "apr",
            "probar",
            "model.apr",
            "--output",
            "/tmp/probar",
            "--format",
            "json",
            "--golden",
            "/refs/golden",
            "--layer",
            "layer.0",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Probar {
                file,
                output,
                format,
                golden,
                layer,
            } => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert_eq!(output, PathBuf::from("/tmp/probar"));
                assert_eq!(format, "json");
                assert_eq!(golden, Some(PathBuf::from("/refs/golden")));
                assert_eq!(layer, Some("layer.0".to_string()));
            }
            _ => panic!("Expected Probar command"),
        }
    }

    /// Test parsing 'apr probar' with defaults
    #[test]
    fn test_parse_probar_defaults() {
        let args = vec!["apr", "probar", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Probar {
                output,
                format,
                golden,
                layer,
                ..
            } => {
                assert_eq!(output, PathBuf::from("./probar-export"));
                assert_eq!(format, "both");
                assert!(golden.is_none());
                assert!(layer.is_none());
            }
            _ => panic!("Expected Probar command"),
        }
    }

    /// Test parsing 'apr debug' command with all flags
    #[test]
    fn test_parse_debug_command() {
        let args = vec![
            "apr",
            "debug",
            "model.apr",
            "--drama",
            "--hex",
            "--strings",
            "--limit",
            "512",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Debug {
                file,
                drama,
                hex,
                strings,
                limit,
            } => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert!(drama);
                assert!(hex);
                assert!(strings);
                assert_eq!(limit, 512);
            }
            _ => panic!("Expected Debug command"),
        }
    }

    /// Test parsing 'apr debug' with defaults
    #[test]
    fn test_parse_debug_defaults() {
        let args = vec!["apr", "debug", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Debug {
                drama,
                hex,
                strings,
                limit,
                ..
            } => {
                assert!(!drama);
                assert!(!hex);
                assert!(!strings);
                assert_eq!(limit, 256);
            }
            _ => panic!("Expected Debug command"),
        }
    }

    /// Test parsing 'apr tui' command with file
    #[test]
    fn test_parse_tui_command_with_file() {
        let args = vec!["apr", "tui", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Tui { file } => {
                assert_eq!(file, Some(PathBuf::from("model.apr")));
            }
            _ => panic!("Expected Tui command"),
        }
    }

    /// Test parsing 'apr tui' without file (optional)
    #[test]
    fn test_parse_tui_command_no_file() {
        let args = vec!["apr", "tui"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Tui { file } => {
                assert!(file.is_none());
            }
            _ => panic!("Expected Tui command"),
        }
    }

    /// Test parsing 'apr import' command with all options
    #[test]
    fn test_parse_import_command() {
        let args = vec![
            "apr",
            "import",
            "hf://openai/whisper-tiny",
            "--output",
            "whisper.apr",
            "--arch",
            "whisper",
            "--quantize",
            "int8",
            "--strict",
            "--preserve-q4k",
            "--tokenizer",
            "/path/to/tokenizer.json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Import {
                source,
                output,
                arch,
                quantize,
                strict,
                preserve_q4k,
                tokenizer,
                enforce_provenance,
            } => {
                assert_eq!(source, "hf://openai/whisper-tiny");
                assert_eq!(output, Some(PathBuf::from("whisper.apr")));
                assert_eq!(arch, "whisper");
                assert_eq!(quantize, Some("int8".to_string()));
                assert!(strict);
                assert!(preserve_q4k);
                assert_eq!(tokenizer, Some(PathBuf::from("/path/to/tokenizer.json")));
                assert!(!enforce_provenance);
            }
            _ => panic!("Expected Import command"),
        }
    }

    /// Test parsing 'apr import' with defaults
    #[test]
    fn test_parse_import_defaults() {
        let args = vec!["apr", "import", "model.gguf"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Import {
                arch,
                quantize,
                strict,
                preserve_q4k,
                output,
                tokenizer,
                ..
            } => {
                assert_eq!(arch, "auto");
                assert!(quantize.is_none());
                assert!(!strict);
                assert!(!preserve_q4k);
                assert!(output.is_none());
                assert!(tokenizer.is_none());
            }
            _ => panic!("Expected Import command"),
        }
    }

    /// Test parsing 'apr export' command
    #[test]
    fn test_parse_export_command() {
        let args = vec![
            "apr",
            "export",
            "model.apr",
            "--format",
            "gguf",
            "-o",
            "model.gguf",
            "--quantize",
            "int4",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Export {
                file,
                format,
                output,
                quantize,
            } => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert_eq!(format, "gguf");
                assert_eq!(output, PathBuf::from("model.gguf"));
                assert_eq!(quantize, Some("int4".to_string()));
            }
            _ => panic!("Expected Export command"),
        }
    }

    /// Test parsing 'apr export' with defaults
    #[test]
    fn test_parse_export_defaults() {
        let args = vec!["apr", "export", "model.apr", "-o", "out.safetensors"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Export {
                format, quantize, ..
            } => {
                assert_eq!(format, "safetensors");
                assert!(quantize.is_none());
            }
            _ => panic!("Expected Export command"),
        }
    }

    /// Test parsing 'apr convert' command with all options
    #[test]
    fn test_parse_convert_command() {
        let args = vec![
            "apr",
            "convert",
            "model.apr",
            "--quantize",
            "q4k",
            "--compress",
            "zstd",
            "-o",
            "model-q4k.apr",
            "--force",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Convert {
                file,
                quantize,
                compress,
                output,
                force,
            } => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert_eq!(quantize, Some("q4k".to_string()));
                assert_eq!(compress, Some("zstd".to_string()));
                assert_eq!(output, PathBuf::from("model-q4k.apr"));
                assert!(force);
            }
            _ => panic!("Expected Convert command"),
        }
    }

    /// Test parsing 'apr convert' with defaults
    #[test]
    fn test_parse_convert_defaults() {
        let args = vec!["apr", "convert", "model.apr", "-o", "out.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Convert {
                quantize,
                compress,
                force,
                ..
            } => {
                assert!(quantize.is_none());
                assert!(compress.is_none());
                assert!(!force);
            }
            _ => panic!("Expected Convert command"),
        }
    }

    /// Test parsing 'apr oracle' command with source
    #[test]
    fn test_parse_oracle_command_with_source() {
        let args = vec![
            "apr",
            "oracle",
            "model.gguf",
            "--compliance",
            "--tensors",
            "--stats",
            "--explain",
            "--kernels",
            "--validate",
            "--full",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Oracle {
                source,
                compliance,
                tensors,
                stats,
                explain,
                kernels,
                validate,
                full,
                family,
                size,
            } => {
                assert_eq!(source, Some("model.gguf".to_string()));
                assert!(compliance);
                assert!(tensors);
                assert!(stats);
                assert!(explain);
                assert!(kernels);
                assert!(validate);
                assert!(full);
                assert!(family.is_none());
                assert!(size.is_none());
            }
            _ => panic!("Expected Oracle command"),
        }
    }

    /// Test parsing 'apr oracle' with --family flag
    #[test]
    fn test_parse_oracle_family_mode() {
        let args = vec!["apr", "oracle", "--family", "qwen2", "--size", "7b"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Oracle {
                source,
                family,
                size,
                ..
            } => {
                assert!(source.is_none());
                assert_eq!(family, Some("qwen2".to_string()));
                assert_eq!(size, Some("7b".to_string()));
            }
            _ => panic!("Expected Oracle command"),
        }
    }

    /// Test parsing 'apr oracle' with hf:// URI
    #[test]
    fn test_parse_oracle_hf_uri() {
        let args = vec!["apr", "oracle", "hf://Qwen/Qwen2.5-Coder-1.5B"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Oracle { source, .. } => {
                assert_eq!(source, Some("hf://Qwen/Qwen2.5-Coder-1.5B".to_string()));
            }
            _ => panic!("Expected Oracle command"),
        }
    }

    /// Test parsing 'apr canary create' subcommand
    #[test]
    fn test_parse_canary_create() {
        let args = vec![
            "apr",
            "canary",
            "create",
            "model.apr",
            "--input",
            "audio.wav",
            "--output",
            "canary.json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Canary { command } => match command {
                CanaryCommands::Create {
                    file,
                    input,
                    output,
                } => {
                    assert_eq!(file, PathBuf::from("model.apr"));
                    assert_eq!(input, PathBuf::from("audio.wav"));
                    assert_eq!(output, PathBuf::from("canary.json"));
                }
                _ => panic!("Expected Create subcommand"),
            },
            _ => panic!("Expected Canary command"),
        }
    }

    /// Test parsing 'apr canary check' subcommand
    #[test]
    fn test_parse_canary_check() {
        let args = vec![
            "apr",
            "canary",
            "check",
            "model.apr",
            "--canary",
            "canary.json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Canary { command } => match command {
                CanaryCommands::Check { file, canary } => {
                    assert_eq!(file, PathBuf::from("model.apr"));
                    assert_eq!(canary, PathBuf::from("canary.json"));
                }
                _ => panic!("Expected Check subcommand"),
            },
            _ => panic!("Expected Canary command"),
        }
    }

    /// Test parsing 'apr compare-hf' command
    #[test]
    fn test_parse_compare_hf_command() {
        let args = vec![
            "apr",
            "compare-hf",
            "model.apr",
            "--hf",
            "openai/whisper-tiny",
            "--tensor",
            "encoder.0",
            "--threshold",
            "1e-3",
            "--json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::CompareHf {
                file,
                hf,
                tensor,
                threshold,
                json,
            } => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert_eq!(hf, "openai/whisper-tiny");
                assert_eq!(tensor, Some("encoder.0".to_string()));
                assert!((threshold - 1e-3).abs() < f64::EPSILON);
                assert!(json);
            }
            _ => panic!("Expected CompareHf command"),
        }
    }

    /// Test parsing 'apr compare-hf' with defaults
    #[test]
    fn test_parse_compare_hf_defaults() {
        let args = vec![
            "apr",
            "compare-hf",
            "model.apr",
            "--hf",
            "openai/whisper-tiny",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::CompareHf {
                tensor,
                threshold,
                json,
                ..
            } => {
                assert!(tensor.is_none());
                assert!((threshold - 1e-5).abs() < f64::EPSILON);
                assert!(!json);
            }
            _ => panic!("Expected CompareHf command"),
        }
    }

    /// Test parsing 'apr pull' command
    #[test]
    fn test_parse_pull_command() {
        let args = vec!["apr", "pull", "hf://Qwen/Qwen2.5-Coder-1.5B", "--force"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Pull { model_ref, force } => {
                assert_eq!(model_ref, "hf://Qwen/Qwen2.5-Coder-1.5B");
                assert!(force);
            }
            _ => panic!("Expected Pull command"),
        }
    }

    /// Test parsing 'apr pull' without force
    #[test]
    fn test_parse_pull_defaults() {
        let args = vec!["apr", "pull", "qwen2.5-coder"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Pull { model_ref, force } => {
                assert_eq!(model_ref, "qwen2.5-coder");
                assert!(!force);
            }
            _ => panic!("Expected Pull command"),
        }
    }

    /// Test parsing 'apr tune' command with all options
    #[test]
    fn test_parse_tune_command() {
        let args = vec![
            "apr",
            "tune",
            "model.apr",
            "--method",
            "lora",
            "--rank",
            "16",
            "--vram",
            "24.0",
            "--plan",
            "--model",
            "7B",
            "--freeze-base",
            "--train-data",
            "data.jsonl",
            "--json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
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
                assert_eq!(file, Some(PathBuf::from("model.apr")));
                assert_eq!(method, "lora");
                assert_eq!(rank, Some(16));
                assert!((vram - 24.0).abs() < f64::EPSILON);
                assert!(plan);
                assert_eq!(model, Some("7B".to_string()));
                assert!(freeze_base);
                assert_eq!(train_data, Some(PathBuf::from("data.jsonl")));
                assert!(json);
            }
            _ => panic!("Expected Tune command"),
        }
    }

    /// Test parsing 'apr tune' with defaults (no file)
    #[test]
    fn test_parse_tune_defaults() {
        let args = vec!["apr", "tune"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
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
                assert!(file.is_none());
                assert_eq!(method, "auto");
                assert!(rank.is_none());
                assert!((vram - 16.0).abs() < f64::EPSILON);
                assert!(!plan);
                assert!(model.is_none());
                assert!(!freeze_base);
                assert!(train_data.is_none());
                assert!(!json);
            }
            _ => panic!("Expected Tune command"),
        }
    }

    /// Test parsing 'apr check' command
    #[test]
    fn test_parse_check_command() {
        let args = vec!["apr", "check", "model.apr", "--no-gpu"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Check { file, no_gpu } => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert!(no_gpu);
            }
            _ => panic!("Expected Check command"),
        }
    }

    /// Test parsing 'apr check' with defaults
    #[test]
    fn test_parse_check_defaults() {
        let args = vec!["apr", "check", "model.gguf"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Check { no_gpu, .. } => {
                assert!(!no_gpu);
            }
            _ => panic!("Expected Check command"),
        }
    }

    /// Test parsing 'apr lint' command
    #[test]
    fn test_parse_lint_command() {
        let args = vec!["apr", "lint", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Lint { file } => {
                assert_eq!(file, PathBuf::from("model.apr"));
            }
            _ => panic!("Expected Lint command"),
        }
    }

    /// Test parsing 'apr tensors' command with all options
    #[test]
    fn test_parse_tensors_command() {
        let args = vec![
            "apr",
            "tensors",
            "model.apr",
            "--stats",
            "--filter",
            "encoder",
            "--limit",
            "20",
            "--json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Tensors {
                file,
                stats,
                filter,
                limit,
                json,
            } => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert!(stats);
                assert_eq!(filter, Some("encoder".to_string()));
                assert_eq!(limit, 20);
                assert!(json);
            }
            _ => panic!("Expected Tensors command"),
        }
    }

    /// Test parsing 'apr explain' command with code
    #[test]
    fn test_parse_explain_with_code() {
        let args = vec!["apr", "explain", "E001"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Explain {
                code, file, tensor, ..
            } => {
                assert_eq!(code, Some("E001".to_string()));
                assert!(file.is_none());
                assert!(tensor.is_none());
            }
            _ => panic!("Expected Explain command"),
        }
    }

    /// Test parsing 'apr explain' with tensor and file
    #[test]
    fn test_parse_explain_with_tensor_and_file() {
        let args = vec![
            "apr",
            "explain",
            "--file",
            "model.apr",
            "--tensor",
            "embed.weight",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Explain { code, file, tensor } => {
                assert!(code.is_none());
                assert_eq!(file, Some(PathBuf::from("model.apr")));
                assert_eq!(tensor, Some("embed.weight".to_string()));
            }
            _ => panic!("Expected Explain command"),
        }
    }

    /// Test parsing 'apr trace' command with all options
    #[test]
    fn test_parse_trace_command() {
        let args = vec![
            "apr",
            "trace",
            "model.apr",
            "--layer",
            "layer.0",
            "--reference",
            "ref.apr",
            "--json",
            "-v",
            "--payload",
            "--diff",
            "--interactive",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Trace {
                file,
                layer,
                reference,
                json,
                verbose,
                payload,
                diff,
                interactive,
            } => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert_eq!(layer, Some("layer.0".to_string()));
                assert_eq!(reference, Some(PathBuf::from("ref.apr")));
                assert!(json);
                assert!(verbose);
                assert!(payload);
                assert!(diff);
                assert!(interactive);
            }
            _ => panic!("Expected Trace command"),
        }
    }

    /// Test parsing 'apr validate' with min-score
    #[test]
    fn test_parse_validate_with_min_score() {
        let args = vec!["apr", "validate", "model.apr", "--min-score", "80"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Validate {
                min_score, strict, ..
            } => {
                assert_eq!(min_score, Some(80));
                assert!(!strict);
            }
            _ => panic!("Expected Validate command"),
        }
    }

    /// Test parsing 'apr diff' with all options
    #[test]
    fn test_parse_diff_with_all_options() {
        let args = vec![
            "apr",
            "diff",
            "a.apr",
            "b.apr",
            "--values",
            "--filter",
            "embed",
            "--limit",
            "5",
            "--transpose-aware",
            "--json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Diff {
                file1,
                file2,
                values,
                filter,
                limit,
                transpose_aware,
                json,
                ..
            } => {
                assert_eq!(file1, PathBuf::from("a.apr"));
                assert_eq!(file2, PathBuf::from("b.apr"));
                assert!(values);
                assert_eq!(filter, Some("embed".to_string()));
                assert_eq!(limit, 5);
                assert!(transpose_aware);
                assert!(json);
            }
            _ => panic!("Expected Diff command"),
        }
    }

    /// Test parsing 'apr run' with --chat flag
    #[test]
    fn test_parse_run_with_chat_flag() {
        let args = vec![
            "apr",
            "run",
            "model.gguf",
            "--prompt",
            "Hello world",
            "--chat",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run {
                chat,
                prompt,
                source,
                ..
            } => {
                assert!(chat);
                assert_eq!(prompt, Some("Hello world".to_string()));
                assert_eq!(source, "model.gguf");
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// Test parsing 'apr run' with --trace-payload shorthand
    #[test]
    fn test_parse_run_with_trace_payload() {
        let args = vec![
            "apr",
            "run",
            "model.gguf",
            "--prompt",
            "test",
            "--trace-payload",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run {
                trace_payload,
                trace,
                trace_level,
                ..
            } => {
                assert!(trace_payload);
                // trace itself should default to false (trace_payload is separate flag)
                assert!(!trace);
                assert_eq!(trace_level, "basic");
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// GH-217: Positional prompt is parsed as second argument
    #[test]
    fn test_parse_run_positional_prompt() {
        let args = vec!["apr", "run", "model.gguf", "What is 2+2?"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run {
                source,
                positional_prompt,
                prompt,
                ..
            } => {
                assert_eq!(source, "model.gguf");
                assert_eq!(positional_prompt, Some("What is 2+2?".to_string()));
                assert_eq!(prompt, None);
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// GH-217: --prompt flag still works and takes precedence
    #[test]
    fn test_parse_run_flag_prompt_overrides_positional() {
        let args = vec![
            "apr",
            "run",
            "model.gguf",
            "positional text",
            "--prompt",
            "flag text",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run {
                positional_prompt,
                prompt,
                ..
            } => {
                assert_eq!(positional_prompt, Some("positional text".to_string()));
                assert_eq!(prompt, Some("flag text".to_string()));
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// GH-217: Positional prompt with -n short flag for max_tokens
    #[test]
    fn test_parse_run_positional_prompt_with_n_flag() {
        let args = vec!["apr", "run", "model.gguf", "What is 2+2?", "-n", "64"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run {
                source,
                positional_prompt,
                max_tokens,
                ..
            } => {
                assert_eq!(source, "model.gguf");
                assert_eq!(positional_prompt, Some("What is 2+2?".to_string()));
                assert_eq!(max_tokens, 64);
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// GH-217: No prompt provided (neither positional nor flag)
    #[test]
    fn test_parse_run_no_prompt() {
        let args = vec!["apr", "run", "model.gguf"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run {
                positional_prompt,
                prompt,
                ..
            } => {
                assert_eq!(positional_prompt, None);
                assert_eq!(prompt, None);
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// Test parsing 'apr run' with local verbose flag
    #[test]
    fn test_parse_run_with_local_verbose() {
        let args = vec!["apr", "run", "model.gguf", "--prompt", "hi", "-v"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run { verbose, .. } => {
                assert!(verbose);
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// Test parsing 'apr run' with all trace options
    #[test]
    fn test_parse_run_with_full_trace() {
        let args = vec![
            "apr",
            "run",
            "model.gguf",
            "--prompt",
            "test",
            "--trace",
            "--trace-steps",
            "Tokenize,Embed,Attention",
            "--trace-verbose",
            "--trace-output",
            "/tmp/trace.json",
            "--trace-level",
            "layer",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run {
                trace,
                trace_steps,
                trace_verbose,
                trace_output,
                trace_level,
                ..
            } => {
                assert!(trace);
                assert_eq!(
                    trace_steps,
                    Some(vec![
                        "Tokenize".to_string(),
                        "Embed".to_string(),
                        "Attention".to_string()
                    ])
                );
                assert!(trace_verbose);
                assert_eq!(trace_output, Some(PathBuf::from("/tmp/trace.json")));
                assert_eq!(trace_level, "layer");
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// Test parsing 'apr run' with --benchmark and --profile flags
    #[test]
    fn test_parse_run_benchmark_and_profile() {
        let args = vec![
            "apr",
            "run",
            "model.gguf",
            "--prompt",
            "test",
            "--benchmark",
            "--profile",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run {
                benchmark, profile, ..
            } => {
                assert!(benchmark);
                assert!(profile);
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// Test parsing 'apr run' with --no-gpu flag
    #[test]
    fn test_parse_run_no_gpu() {
        let args = vec!["apr", "run", "model.gguf", "--prompt", "test", "--no-gpu"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run { no_gpu, .. } => {
                assert!(no_gpu);
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// Test parsing 'apr run' with --offline flag
    #[test]
    fn test_parse_run_offline() {
        let args = vec!["apr", "run", "model.gguf", "--prompt", "test", "--offline"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run { offline, .. } => {
                assert!(offline);
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// Test parsing 'apr run' with --stream and --format options
    #[test]
    fn test_parse_run_stream_and_format() {
        let args = vec![
            "apr",
            "run",
            "model.gguf",
            "--prompt",
            "test",
            "--stream",
            "-f",
            "json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run { stream, format, .. } => {
                assert!(stream);
                assert_eq!(format, "json");
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// Test parsing 'apr run' with --input and --language for ASR
    #[test]
    fn test_parse_run_asr_options() {
        let args = vec![
            "apr",
            "run",
            "hf://openai/whisper-tiny",
            "--input",
            "audio.wav",
            "--language",
            "en",
            "--task",
            "transcribe",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run {
                input,
                language,
                task,
                ..
            } => {
                assert_eq!(input, Some(PathBuf::from("audio.wav")));
                assert_eq!(language, Some("en".to_string()));
                assert_eq!(task, Some("transcribe".to_string()));
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// Test parsing 'apr remove' alias for 'rm'
    #[test]
    fn test_parse_remove_alias() {
        let args = vec!["apr", "remove", "my-model"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Rm { model_ref } => {
                assert_eq!(model_ref, "my-model");
            }
            _ => panic!("Expected Rm command"),
        }
    }

    /// Test parsing 'apr qa' with all skip flags
    #[test]
    fn test_parse_qa_all_skip_flags() {
        let args = vec![
            "apr",
            "qa",
            "model.gguf",
            "--skip-golden",
            "--skip-throughput",
            "--skip-ollama",
            "--skip-gpu-speedup",
            "--skip-contract",
            "--skip-format-parity",
            "--safetensors-path",
            "model.safetensors",
            "--iterations",
            "20",
            "--warmup",
            "5",
            "--max-tokens",
            "64",
            "--json",
            "-v",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Qa {
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
                ..
            } => {
                assert!(skip_golden);
                assert!(skip_throughput);
                assert!(skip_ollama);
                assert!(skip_gpu_speedup);
                assert!(skip_contract);
                assert!(skip_format_parity);
                assert_eq!(safetensors_path, Some(PathBuf::from("model.safetensors")));
                assert_eq!(iterations, 20);
                assert_eq!(warmup, 5);
                assert_eq!(max_tokens, 64);
                assert!(json);
                assert!(verbose);
            }
            _ => panic!("Expected Qa command"),
        }
    }

    /// Test parsing 'apr serve' with all options
    #[test]
    fn test_parse_serve_all_options() {
        let args = vec![
            "apr",
            "serve",
            "model.apr",
            "--port",
            "9090",
            "--host",
            "0.0.0.0",
            "--no-cors",
            "--no-metrics",
            "--no-gpu",
            "--batch",
            "--trace",
            "--trace-level",
            "layer",
            "--profile",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Serve {
                port,
                host,
                no_cors,
                no_metrics,
                no_gpu,
                batch,
                trace,
                trace_level,
                profile,
                ..
            } => {
                assert_eq!(port, 9090);
                assert_eq!(host, "0.0.0.0");
                assert!(no_cors);
                assert!(no_metrics);
                assert!(no_gpu);
                assert!(batch);
                assert!(trace);
                assert_eq!(trace_level, "layer");
                assert!(profile);
            }
            _ => panic!("Expected Serve command"),
        }
    }

    /// Test parsing 'apr bench' with all options
    #[test]
    fn test_parse_bench_all_options() {
        let args = vec![
            "apr",
            "bench",
            "model.gguf",
            "--warmup",
            "10",
            "--iterations",
            "20",
            "--max-tokens",
            "64",
            "--prompt",
            "The quick brown fox",
            "--fast",
            "--brick",
            "attention",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Bench {
                warmup,
                iterations,
                max_tokens,
                prompt,
                fast,
                brick,
                ..
            } => {
                assert_eq!(warmup, 10);
                assert_eq!(iterations, 20);
                assert_eq!(max_tokens, 64);
                assert_eq!(prompt, Some("The quick brown fox".to_string()));
                assert!(fast);
                assert_eq!(brick, Some("attention".to_string()));
            }
            _ => panic!("Expected Bench command"),
        }
    }

    /// Test parsing 'apr cbtop' with speculative decoding flags
    #[test]
    fn test_parse_cbtop_speculative() {
        let args = vec![
            "apr",
            "cbtop",
            "--model-path",
            "model.gguf",
            "--speculative",
            "--speculation-k",
            "8",
            "--draft-model",
            "draft.gguf",
            "--concurrent",
            "4",
            "--simulated",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Cbtop {
                model_path,
                speculative,
                speculation_k,
                draft_model,
                concurrent,
                simulated,
                ..
            } => {
                assert_eq!(model_path, Some(PathBuf::from("model.gguf")));
                assert!(speculative);
                assert_eq!(speculation_k, 8);
                assert_eq!(draft_model, Some(PathBuf::from("draft.gguf")));
                assert_eq!(concurrent, 4);
                assert!(simulated);
            }
            _ => panic!("Expected Cbtop command"),
        }
    }

    /// Test parsing 'apr profile' with energy and perf-grade flags
    #[test]
    fn test_parse_profile_energy_perf() {
        let args = vec![
            "apr",
            "profile",
            "model.apr",
            "--energy",
            "--perf-grade",
            "--callgraph",
            "--compare-hf",
            "openai/whisper-tiny",
            "--output",
            "/tmp/flame.svg",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Profile {
                energy,
                perf_grade,
                callgraph,
                compare_hf,
                output,
                ..
            } => {
                assert!(energy);
                assert!(perf_grade);
                assert!(callgraph);
                assert_eq!(compare_hf, Some("openai/whisper-tiny".to_string()));
                assert_eq!(output, Some(PathBuf::from("/tmp/flame.svg")));
            }
            _ => panic!("Expected Profile command"),
        }
    }

    /// Test parsing 'apr chat' with all trace options
    #[test]
    fn test_parse_chat_with_trace() {
        let args = vec![
            "apr",
            "chat",
            "model.gguf",
            "--system",
            "You are a helpful assistant.",
            "--inspect",
            "--trace",
            "--trace-steps",
            "Tokenize,Decode",
            "--trace-verbose",
            "--trace-output",
            "/tmp/chat-trace.json",
            "--trace-level",
            "payload",
            "--profile",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Chat {
                system,
                inspect,
                trace,
                trace_steps,
                trace_verbose,
                trace_output,
                trace_level,
                profile,
                ..
            } => {
                assert_eq!(system, Some("You are a helpful assistant.".to_string()));
                assert!(inspect);
                assert!(trace);
                assert_eq!(
                    trace_steps,
                    Some(vec!["Tokenize".to_string(), "Decode".to_string()])
                );
                assert!(trace_verbose);
                assert_eq!(trace_output, Some(PathBuf::from("/tmp/chat-trace.json")));
                assert_eq!(trace_level, "payload");
                assert!(profile);
            }
            _ => panic!("Expected Chat command"),
        }
    }

    /// Test parsing 'apr showcase' with step and all options
    #[test]
    fn test_parse_showcase_with_step() {
        let args = vec![
            "apr", "showcase", "--step", "bench", "--tier", "tiny", "--zram", "--runs", "50",
            "--json", "-v", "-q",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Showcase {
                step,
                tier,
                zram,
                runs,
                json,
                verbose,
                quiet,
                ..
            } => {
                assert_eq!(step, Some("bench".to_string()));
                assert_eq!(tier, "tiny");
                assert!(zram);
                assert_eq!(runs, 50);
                assert!(json);
                assert!(verbose);
                assert!(quiet);
            }
            _ => panic!("Expected Showcase command"),
        }
    }

    /// Test parsing rosetta compare-inference subcommand
    #[test]
    fn test_parse_rosetta_compare_inference() {
        let args = vec![
            "apr",
            "rosetta",
            "compare-inference",
            "model_a.gguf",
            "model_b.apr",
            "--prompt",
            "What is 2+2?",
            "--max-tokens",
            "10",
            "--temperature",
            "0.5",
            "--tolerance",
            "0.05",
            "--json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Rosetta { action } => match action {
                RosettaCommands::CompareInference {
                    model_a,
                    model_b,
                    prompt,
                    max_tokens,
                    temperature,
                    tolerance,
                    json,
                } => {
                    assert_eq!(model_a, PathBuf::from("model_a.gguf"));
                    assert_eq!(model_b, PathBuf::from("model_b.apr"));
                    assert_eq!(prompt, "What is 2+2?");
                    assert_eq!(max_tokens, 10);
                    assert!((temperature - 0.5).abs() < f32::EPSILON);
                    assert!((tolerance - 0.05).abs() < f32::EPSILON);
                    assert!(json);
                }
                _ => panic!("Expected CompareInference subcommand"),
            },
            _ => panic!("Expected Rosetta command"),
        }
    }

    /// Test parsing rosetta diff-tensors subcommand
    #[test]
    fn test_parse_rosetta_diff_tensors() {
        let args = vec![
            "apr",
            "rosetta",
            "diff-tensors",
            "ref.gguf",
            "test.apr",
            "--mismatches-only",
            "--show-values",
            "5",
            "--filter",
            "lm_head",
            "--json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Rosetta { action } => match action {
                RosettaCommands::DiffTensors {
                    model_a,
                    model_b,
                    mismatches_only,
                    show_values,
                    filter,
                    json,
                } => {
                    assert_eq!(model_a, PathBuf::from("ref.gguf"));
                    assert_eq!(model_b, PathBuf::from("test.apr"));
                    assert!(mismatches_only);
                    assert_eq!(show_values, 5);
                    assert_eq!(filter, Some("lm_head".to_string()));
                    assert!(json);
                }
                _ => panic!("Expected DiffTensors subcommand"),
            },
            _ => panic!("Expected Rosetta command"),
        }
    }

    /// Test parsing rosetta fingerprint subcommand
    #[test]
    fn test_parse_rosetta_fingerprint() {
        let args = vec![
            "apr",
            "rosetta",
            "fingerprint",
            "model.gguf",
            "model2.apr",
            "--output",
            "fingerprints.json",
            "--filter",
            "encoder",
            "--verbose",
            "--json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Rosetta { action } => match action {
                RosettaCommands::Fingerprint {
                    model,
                    model_b,
                    output,
                    filter,
                    verbose,
                    json,
                } => {
                    assert_eq!(model, PathBuf::from("model.gguf"));
                    assert_eq!(model_b, Some(PathBuf::from("model2.apr")));
                    assert_eq!(output, Some(PathBuf::from("fingerprints.json")));
                    assert_eq!(filter, Some("encoder".to_string()));
                    assert!(verbose);
                    assert!(json);
                }
                _ => panic!("Expected Fingerprint subcommand"),
            },
            _ => panic!("Expected Rosetta command"),
        }
    }

    /// Test parsing rosetta validate-stats subcommand
    #[test]
    fn test_parse_rosetta_validate_stats() {
        let args = vec![
            "apr",
            "rosetta",
            "validate-stats",
            "model.apr",
            "--reference",
            "ref.gguf",
            "--fingerprints",
            "fp.json",
            "--threshold",
            "5.0",
            "--strict",
            "--json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Rosetta { action } => match action {
                RosettaCommands::ValidateStats {
                    model,
                    reference,
                    fingerprints,
                    threshold,
                    strict,
                    json,
                } => {
                    assert_eq!(model, PathBuf::from("model.apr"));
                    assert_eq!(reference, Some(PathBuf::from("ref.gguf")));
                    assert_eq!(fingerprints, Some(PathBuf::from("fp.json")));
                    assert!((threshold - 5.0).abs() < f32::EPSILON);
                    assert!(strict);
                    assert!(json);
                }
                _ => panic!("Expected ValidateStats subcommand"),
            },
            _ => panic!("Expected Rosetta command"),
        }
    }

    // =========================================================================
    // Global flag tests
    // =========================================================================

    /// Test global --offline flag
    #[test]
    fn test_global_offline_flag() {
        let args = vec!["apr", "--offline", "inspect", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        assert!(cli.offline);
    }

    /// Test global --quiet flag
    #[test]
    fn test_global_quiet_flag() {
        let args = vec!["apr", "--quiet", "inspect", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        assert!(cli.quiet);
    }

    /// Test multiple global flags combined
    #[test]
    fn test_multiple_global_flags() {
        let args = vec![
            "apr",
            "--verbose",
            "--json",
            "--offline",
            "--quiet",
            "--skip-contract",
            "inspect",
            "model.apr",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        assert!(cli.verbose);
        assert!(cli.json);
        assert!(cli.offline);
        assert!(cli.quiet);
        assert!(cli.skip_contract);
    }

    /// Test global flags default to false
    #[test]
    fn test_global_flags_default_false() {
        let args = vec!["apr", "inspect", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        assert!(!cli.verbose);
        assert!(!cli.json);
        assert!(!cli.offline);
        assert!(!cli.quiet);
        assert!(!cli.skip_contract);
    }

    // =========================================================================
    // extract_model_paths: additional command variants
    // =========================================================================

    /// Test extract_model_paths: Export returns file path
    #[test]
    fn test_extract_paths_export() {
        let cmd = Commands::Export {
            file: PathBuf::from("model.apr"),
            format: "gguf".to_string(),
            output: PathBuf::from("out.gguf"),
            quantize: None,
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.apr")]);
    }

    /// Test extract_model_paths: Convert returns file path
    #[test]
    fn test_extract_paths_convert() {
        let cmd = Commands::Convert {
            file: PathBuf::from("model.apr"),
            quantize: Some("q4k".to_string()),
            compress: None,
            output: PathBuf::from("out.apr"),
            force: false,
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.apr")]);
    }

    /// Test extract_model_paths: Check returns file path
    #[test]
    fn test_extract_paths_check() {
        let cmd = Commands::Check {
            file: PathBuf::from("model.gguf"),
            no_gpu: false,
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.gguf")]);
    }

    /// Test extract_model_paths: Trace returns file path
    #[test]
    fn test_extract_paths_trace() {
        let cmd = Commands::Trace {
            file: PathBuf::from("model.apr"),
            layer: None,
            reference: None,
            json: false,
            verbose: false,
            payload: false,
            diff: false,
            interactive: false,
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.apr")]);
    }

    /// Test extract_model_paths: Probar returns file path
    #[test]
    fn test_extract_paths_probar() {
        let cmd = Commands::Probar {
            file: PathBuf::from("model.apr"),
            output: PathBuf::from("./probar-export"),
            format: "both".to_string(),
            golden: None,
            layer: None,
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.apr")]);
    }

    /// Test extract_model_paths: CompareHf returns file path
    #[test]
    fn test_extract_paths_compare_hf() {
        let cmd = Commands::CompareHf {
            file: PathBuf::from("model.apr"),
            hf: "openai/whisper-tiny".to_string(),
            tensor: None,
            threshold: 1e-5,
            json: false,
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.apr")]);
    }

    /// Test extract_model_paths: Chat returns file path
    #[test]
    fn test_extract_paths_chat() {
        let cmd = Commands::Chat {
            file: PathBuf::from("model.gguf"),
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 512,
            system: None,
            inspect: false,
            no_gpu: false,
            gpu: false,
            trace: false,
            trace_steps: None,
            trace_verbose: false,
            trace_output: None,
            trace_level: "basic".to_string(),
            profile: false,
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.gguf")]);
    }

    /// Test extract_model_paths: Eval returns file path
    #[test]
    fn test_extract_paths_eval() {
        let cmd = Commands::Eval {
            file: PathBuf::from("model.gguf"),
            dataset: "wikitext-2".to_string(),
            text: None,
            max_tokens: 512,
            threshold: 20.0,
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.gguf")]);
    }

    /// Test extract_model_paths: Profile returns file path
    #[test]
    fn test_extract_paths_profile() {
        let cmd = Commands::Profile {
            file: PathBuf::from("model.apr"),
            granular: false,
            format: "human".to_string(),
            focus: None,
            detect_naive: false,
            threshold: 10.0,
            compare_hf: None,
            energy: false,
            perf_grade: false,
            callgraph: false,
            fail_on_naive: false,
            output: None,
            ci: false,
            assert_throughput: None,
            assert_p99: None,
            assert_p50: None,
            warmup: 3,
            measure: 10,
            tokens: 32,
            ollama: false,
            no_gpu: false,
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.apr")]);
    }

    /// Test extract_model_paths: Import with hf:// URL returns empty (non-local)
    #[test]
    fn test_extract_paths_import_hf_url() {
        let cmd = Commands::Import {
            source: "hf://openai/whisper-tiny".to_string(),
            output: Some(PathBuf::from("whisper.apr")),
            arch: "auto".to_string(),
            quantize: None,
            strict: false,
            preserve_q4k: false,
            tokenizer: None,
            enforce_provenance: false,
        };
        let paths = extract_model_paths(&cmd);
        assert!(
            paths.is_empty(),
            "hf:// URLs should not be validated locally for import"
        );
    }

    /// Test extract_model_paths: Import with non-existent local path returns empty
    #[test]
    fn test_extract_paths_import_nonexistent_local() {
        let cmd = Commands::Import {
            source: "/tmp/nonexistent_model_abc123.gguf".to_string(),
            output: None,
            arch: "auto".to_string(),
            quantize: None,
            strict: false,
            preserve_q4k: false,
            tokenizer: None,
            enforce_provenance: false,
        };
        let paths = extract_model_paths(&cmd);
        assert!(
            paths.is_empty(),
            "Non-existent local paths return empty for import"
        );
    }

    /// Test extract_model_paths: Tui with file returns file
    #[test]
    fn test_extract_paths_tui_with_file() {
        let cmd = Commands::Tui {
            file: Some(PathBuf::from("model.apr")),
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.apr")]);
    }

    /// Test extract_model_paths: Tui without file returns empty
    #[test]
    fn test_extract_paths_tui_no_file() {
        let cmd = Commands::Tui { file: None };
        let paths = extract_model_paths(&cmd);
        assert!(paths.is_empty());
    }

    /// Test extract_model_paths: Cbtop with model_path returns it
    #[test]
    fn test_extract_paths_cbtop_with_model_path() {
        let cmd = Commands::Cbtop {
            model: None,
            attach: None,
            model_path: Some(PathBuf::from("model.gguf")),
            headless: false,
            json: false,
            output: None,
            ci: false,
            throughput: None,
            brick_score: None,
            warmup: 10,
            iterations: 100,
            speculative: false,
            speculation_k: 4,
            draft_model: None,
            concurrent: 1,
            simulated: false,
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.gguf")]);
    }

    /// Test extract_model_paths: Cbtop without model_path returns empty
    #[test]
    fn test_extract_paths_cbtop_no_model_path() {
        let cmd = Commands::Cbtop {
            model: Some("qwen2.5-coder".to_string()),
            attach: None,
            model_path: None,
            headless: false,
            json: false,
            output: None,
            ci: false,
            throughput: None,
            brick_score: None,
            warmup: 10,
            iterations: 100,
            speculative: false,
            speculation_k: 4,
            draft_model: None,
            concurrent: 1,
            simulated: false,
        };
        let paths = extract_model_paths(&cmd);
        assert!(paths.is_empty());
    }

    /// Test extract_model_paths: Diff is diagnostic (exempt)
    #[test]
    fn test_extract_paths_diff_exempt() {
        let cmd = Commands::Diff {
            file1: PathBuf::from("a.apr"),
            file2: PathBuf::from("b.apr"),
            weights: false,
            values: false,
            filter: None,
            limit: 10,
            transpose_aware: false,
            json: false,
        };
        let paths = extract_model_paths(&cmd);
        assert!(paths.is_empty(), "Diff is a diagnostic command (exempt)");
    }

    /// Test extract_model_paths: Hex is diagnostic (exempt)
    #[test]
    fn test_extract_paths_hex_exempt() {
        let cmd = Commands::Hex {
            file: PathBuf::from("model.apr"),
            tensor: None,
            limit: 64,
            stats: false,
            list: false,
            json: false,
            header: false,
            blocks: false,
            distribution: false,
            contract: false,
            entropy: false,
            raw: false,
            offset: "0".to_string(),
            width: 16,
        };
        let paths = extract_model_paths(&cmd);
        assert!(paths.is_empty(), "Hex is a diagnostic command (exempt)");
    }

    /// Test extract_model_paths: Tree is diagnostic (exempt)
    #[test]
    fn test_extract_paths_tree_exempt() {
        let cmd = Commands::Tree {
            file: PathBuf::from("model.apr"),
            filter: None,
            format: "ascii".to_string(),
            sizes: false,
            depth: None,
        };
        let paths = extract_model_paths(&cmd);
        assert!(paths.is_empty(), "Tree is a diagnostic command (exempt)");
    }

    /// Test extract_model_paths: Flow is diagnostic (exempt)
    #[test]
    fn test_extract_paths_flow_exempt() {
        let cmd = Commands::Flow {
            file: PathBuf::from("model.apr"),
            layer: None,
            component: "full".to_string(),
            verbose: false,
        };
        let paths = extract_model_paths(&cmd);
        assert!(paths.is_empty(), "Flow is a diagnostic command (exempt)");
    }

    /// Test extract_model_paths: Publish is diagnostic (exempt)
    #[test]
    fn test_extract_paths_publish_exempt() {
        let cmd = Commands::Publish {
            directory: PathBuf::from("/tmp/models"),
            repo_id: "org/repo".to_string(),
            model_name: None,
            license: "mit".to_string(),
            pipeline_tag: "text-generation".to_string(),
            library_name: None,
            tags: None,
            message: None,
            dry_run: false,
        };
        let paths = extract_model_paths(&cmd);
        assert!(paths.is_empty(), "Publish is a diagnostic command (exempt)");
    }

    /// Test extract_model_paths: Tune is diagnostic (exempt)
    #[test]
    fn test_extract_paths_tune_exempt() {
        let cmd = Commands::Tune {
            file: Some(PathBuf::from("model.apr")),
            method: "auto".to_string(),
            rank: None,
            vram: 16.0,
            plan: false,
            model: None,
            freeze_base: false,
            train_data: None,
            json: false,
        };
        let paths = extract_model_paths(&cmd);
        assert!(paths.is_empty(), "Tune is a diagnostic command (exempt)");
    }

    /// Test extract_model_paths: Pull is diagnostic (exempt)
    #[test]
    fn test_extract_paths_pull_exempt() {
        let cmd = Commands::Pull {
            model_ref: "hf://org/repo".to_string(),
            force: false,
        };
        let paths = extract_model_paths(&cmd);
        assert!(paths.is_empty(), "Pull is a diagnostic command (exempt)");
    }

    /// Test extract_model_paths: Rm is diagnostic (exempt)
    #[test]
    fn test_extract_paths_rm_exempt() {
        let cmd = Commands::Rm {
            model_ref: "model-name".to_string(),
        };
        let paths = extract_model_paths(&cmd);
        assert!(paths.is_empty(), "Rm is a diagnostic command (exempt)");
    }

    /// Test extract_model_paths: Canary is diagnostic (exempt)
    #[test]
    fn test_extract_paths_canary_exempt() {
        let cmd = Commands::Canary {
            command: CanaryCommands::Check {
                file: PathBuf::from("model.apr"),
                canary: PathBuf::from("canary.json"),
            },
        };
        let paths = extract_model_paths(&cmd);
        assert!(paths.is_empty(), "Canary is a diagnostic command (exempt)");
    }

    /// Test extract_model_paths: Oracle is diagnostic (exempt)
    #[test]
    fn test_extract_paths_oracle_exempt() {
        let cmd = Commands::Oracle {
            source: Some("model.gguf".to_string()),
            family: None,
            size: None,
            compliance: false,
            tensors: false,
            stats: false,
            explain: false,
            kernels: false,
            validate: false,
            full: false,
        };
        let paths = extract_model_paths(&cmd);
        assert!(paths.is_empty(), "Oracle is a diagnostic command (exempt)");
    }

    /// Test extract_model_paths: Showcase is diagnostic (exempt)
    #[test]
    fn test_extract_paths_showcase_exempt() {
        let cmd = Commands::Showcase {
            auto_verify: false,
            step: None,
            tier: "small".to_string(),
            model_dir: PathBuf::from("./models"),
            baseline: "llama-cpp,ollama".to_string(),
            zram: false,
            runs: 30,
            gpu: false,
            json: false,
            verbose: false,
            quiet: false,
        };
        let paths = extract_model_paths(&cmd);
        assert!(
            paths.is_empty(),
            "Showcase is a diagnostic command (exempt)"
        );
    }

    /// Test extract_model_paths: Rosetta Convert returns source path
    #[test]
    fn test_extract_paths_rosetta_convert() {
        let cmd = Commands::Rosetta {
            action: RosettaCommands::Convert {
                source: PathBuf::from("model.gguf"),
                target: PathBuf::from("out.safetensors"),
                quantize: None,
                verify: false,
                json: false,
                tokenizer: None,
            },
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.gguf")]);
    }

    /// Test extract_model_paths: Rosetta Chain returns source path
    #[test]
    fn test_extract_paths_rosetta_chain() {
        let cmd = Commands::Rosetta {
            action: RosettaCommands::Chain {
                source: PathBuf::from("model.gguf"),
                formats: vec!["safetensors".to_string(), "apr".to_string()],
                work_dir: PathBuf::from("/tmp"),
                json: false,
            },
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.gguf")]);
    }

    /// Test extract_model_paths: Rosetta Verify returns source path
    #[test]
    fn test_extract_paths_rosetta_verify() {
        let cmd = Commands::Rosetta {
            action: RosettaCommands::Verify {
                source: PathBuf::from("model.apr"),
                intermediate: "safetensors".to_string(),
                tolerance: 1e-5,
                json: false,
            },
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.apr")]);
    }

    /// Test extract_model_paths: Rosetta CompareInference returns both paths
    #[test]
    fn test_extract_paths_rosetta_compare_inference() {
        let cmd = Commands::Rosetta {
            action: RosettaCommands::CompareInference {
                model_a: PathBuf::from("model_a.gguf"),
                model_b: PathBuf::from("model_b.apr"),
                prompt: "test".to_string(),
                max_tokens: 5,
                temperature: 0.0,
                tolerance: 0.1,
                json: false,
            },
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(
            paths,
            vec![PathBuf::from("model_a.gguf"), PathBuf::from("model_b.apr")]
        );
    }

    /// Test extract_model_paths: Rosetta Inspect is diagnostic (exempt)
    #[test]
    fn test_extract_paths_rosetta_inspect_exempt() {
        let cmd = Commands::Rosetta {
            action: RosettaCommands::Inspect {
                file: PathBuf::from("model.gguf"),
                hexdump: false,
                json: false,
            },
        };
        let paths = extract_model_paths(&cmd);
        assert!(
            paths.is_empty(),
            "Rosetta Inspect is a diagnostic command (exempt)"
        );
    }

    /// Test extract_model_paths: Rosetta DiffTensors is diagnostic (exempt)
    #[test]
    fn test_extract_paths_rosetta_diff_tensors_exempt() {
        let cmd = Commands::Rosetta {
            action: RosettaCommands::DiffTensors {
                model_a: PathBuf::from("a.gguf"),
                model_b: PathBuf::from("b.apr"),
                mismatches_only: false,
                show_values: 0,
                filter: None,
                json: false,
            },
        };
        let paths = extract_model_paths(&cmd);
        assert!(
            paths.is_empty(),
            "Rosetta DiffTensors is a diagnostic command (exempt)"
        );
    }

    /// Test extract_model_paths: Rosetta Fingerprint is diagnostic (exempt)
    #[test]
    fn test_extract_paths_rosetta_fingerprint_exempt() {
        let cmd = Commands::Rosetta {
            action: RosettaCommands::Fingerprint {
                model: PathBuf::from("model.gguf"),
                model_b: None,
                output: None,
                filter: None,
                verbose: false,
                json: false,
            },
        };
        let paths = extract_model_paths(&cmd);
        assert!(
            paths.is_empty(),
            "Rosetta Fingerprint is a diagnostic command (exempt)"
        );
    }

    /// Test extract_model_paths: Rosetta ValidateStats is diagnostic (exempt)
    #[test]
    fn test_extract_paths_rosetta_validate_stats_exempt() {
        let cmd = Commands::Rosetta {
            action: RosettaCommands::ValidateStats {
                model: PathBuf::from("model.apr"),
                reference: None,
                fingerprints: None,
                threshold: 3.0,
                strict: false,
                json: false,
            },
        };
        let paths = extract_model_paths(&cmd);
        assert!(
            paths.is_empty(),
            "Rosetta ValidateStats is a diagnostic command (exempt)"
        );
    }

    // =========================================================================
    // validate_model_contract: additional edge cases
    // =========================================================================

    /// Test validate_model_contract: multiple non-existent paths all skipped
    #[test]
    fn test_validate_contract_multiple_nonexistent() {
        let paths = vec![
            PathBuf::from("/tmp/nonexistent_a.apr"),
            PathBuf::from("/tmp/nonexistent_b.gguf"),
            PathBuf::from("/tmp/nonexistent_c.safetensors"),
        ];
        let result = validate_model_contract(&paths);
        assert!(result.is_ok(), "All non-existent paths should be skipped");
    }

    /// Test validate_model_contract: mix of non-existent paths
    #[test]
    fn test_validate_contract_mixed_nonexistent() {
        let paths = vec![
            PathBuf::from("/tmp/does_not_exist_xyz.apr"),
            PathBuf::from("/tmp/also_missing_123.gguf"),
        ];
        let result = validate_model_contract(&paths);
        assert!(
            result.is_ok(),
            "Mixed non-existent paths should all be skipped"
        );
    }

    // =========================================================================
    // execute_command: error path tests (file not found)
    // =========================================================================

    /// Helper: create a Cli struct with the given command and default flags
    fn make_cli(command: Commands) -> Cli {
        Cli {
            command: Box::new(command),
            json: false,
            verbose: false,
            quiet: false,
            offline: false,
            skip_contract: true, // Skip contract to test command dispatch errors
        }
    }

    /// Test execute_command: Inspect with non-existent file returns error
    #[test]
    fn test_execute_inspect_file_not_found() {
        let cli = make_cli(Commands::Inspect {
            file: PathBuf::from("/tmp/nonexistent_model_inspect_test.apr"),
            vocab: false,
            filters: false,
            weights: false,
            json: false,
        });
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Inspect should fail with non-existent file"
        );
    }

    /// Test execute_command: Debug with non-existent file returns error
    #[test]
    fn test_execute_debug_file_not_found() {
        let cli = make_cli(Commands::Debug {
            file: PathBuf::from("/tmp/nonexistent_model_debug_test.apr"),
            drama: false,
            hex: false,
            strings: false,
            limit: 256,
        });
        let result = execute_command(&cli);
        assert!(result.is_err(), "Debug should fail with non-existent file");
    }

    /// Test execute_command: Validate with non-existent file returns error
    #[test]
    fn test_execute_validate_file_not_found() {
        let cli = make_cli(Commands::Validate {
            file: PathBuf::from("/tmp/nonexistent_model_validate_test.apr"),
            quality: false,
            strict: false,
            min_score: None,
        });
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Validate should fail with non-existent file"
        );
    }

    /// Test execute_command: Diff with non-existent files returns error
    #[test]
    fn test_execute_diff_file_not_found() {
        let cli = make_cli(Commands::Diff {
            file1: PathBuf::from("/tmp/nonexistent_model_diff1.apr"),
            file2: PathBuf::from("/tmp/nonexistent_model_diff2.apr"),
            weights: false,
            values: false,
            filter: None,
            limit: 10,
            transpose_aware: false,
            json: false,
        });
        let result = execute_command(&cli);
        assert!(result.is_err(), "Diff should fail with non-existent files");
    }

    /// Test execute_command: Tensors with non-existent file returns error
    #[test]
    fn test_execute_tensors_file_not_found() {
        let cli = make_cli(Commands::Tensors {
            file: PathBuf::from("/tmp/nonexistent_model_tensors_test.apr"),
            stats: false,
            filter: None,
            limit: 0,
            json: false,
        });
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Tensors should fail with non-existent file"
        );
    }

    /// Test execute_command: Lint with non-existent file returns error
    #[test]
    fn test_execute_lint_file_not_found() {
        let cli = make_cli(Commands::Lint {
            file: PathBuf::from("/tmp/nonexistent_model_lint_test.apr"),
        });
        let result = execute_command(&cli);
        assert!(result.is_err(), "Lint should fail with non-existent file");
    }

    /// Test execute_command: Trace with non-existent file returns error
    #[test]
    fn test_execute_trace_file_not_found() {
        let cli = make_cli(Commands::Trace {
            file: PathBuf::from("/tmp/nonexistent_model_trace_test.apr"),
            layer: None,
            reference: None,
            json: false,
            verbose: false,
            payload: false,
            diff: false,
            interactive: false,
        });
        let result = execute_command(&cli);
        assert!(result.is_err(), "Trace should fail with non-existent file");
    }

    /// Test execute_command: Export with non-existent file returns error
    #[test]
    fn test_execute_export_file_not_found() {
        let cli = make_cli(Commands::Export {
            file: PathBuf::from("/tmp/nonexistent_model_export_test.apr"),
            format: "safetensors".to_string(),
            output: PathBuf::from("/tmp/out.safetensors"),
            quantize: None,
        });
        let result = execute_command(&cli);
        assert!(result.is_err(), "Export should fail with non-existent file");
    }

    /// Test execute_command: Convert with non-existent file returns error
    #[test]
    fn test_execute_convert_file_not_found() {
        let cli = make_cli(Commands::Convert {
            file: PathBuf::from("/tmp/nonexistent_model_convert_test.apr"),
            quantize: None,
            compress: None,
            output: PathBuf::from("/tmp/out.apr"),
            force: false,
        });
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Convert should fail with non-existent file"
        );
    }

    /// Test execute_command: Hex with non-existent file returns error
    #[test]
    fn test_execute_hex_file_not_found() {
        let cli = make_cli(Commands::Hex {
            file: PathBuf::from("/tmp/nonexistent_model_hex_test.apr"),
            tensor: None,
            limit: 64,
            stats: false,
            list: false,
            json: false,
            header: false,
            blocks: false,
            distribution: false,
            contract: false,
            entropy: false,
            raw: false,
            offset: String::new(),
            width: 16,
        });
        let result = execute_command(&cli);
        assert!(result.is_err(), "Hex should fail with non-existent file");
    }

    /// Test execute_command: Tree with non-existent file returns error
    #[test]
    fn test_execute_tree_file_not_found() {
        let cli = make_cli(Commands::Tree {
            file: PathBuf::from("/tmp/nonexistent_model_tree_test.apr"),
            filter: None,
            format: "ascii".to_string(),
            sizes: false,
            depth: None,
        });
        let result = execute_command(&cli);
        assert!(result.is_err(), "Tree should fail with non-existent file");
    }

    /// Test execute_command: Flow with non-existent file returns error
    #[test]
    fn test_execute_flow_file_not_found() {
        let cli = make_cli(Commands::Flow {
            file: PathBuf::from("/tmp/nonexistent_model_flow_test.apr"),
            layer: None,
            component: "full".to_string(),
            verbose: false,
        });
        let result = execute_command(&cli);
        assert!(result.is_err(), "Flow should fail with non-existent file");
    }

    /// Test execute_command: Probar with non-existent file returns error
    #[test]
    fn test_execute_probar_file_not_found() {
        let cli = make_cli(Commands::Probar {
            file: PathBuf::from("/tmp/nonexistent_model_probar_test.apr"),
            output: PathBuf::from("/tmp/probar-out"),
            format: "both".to_string(),
            golden: None,
            layer: None,
        });
        let result = execute_command(&cli);
        assert!(result.is_err(), "Probar should fail with non-existent file");
    }

    /// Test execute_command: Check with non-existent file returns error
    #[test]
    fn test_execute_check_file_not_found() {
        let cli = make_cli(Commands::Check {
            file: PathBuf::from("/tmp/nonexistent_model_check_test.apr"),
            no_gpu: true,
        });
        let result = execute_command(&cli);
        assert!(result.is_err(), "Check should fail with non-existent file");
    }

    /// Test execute_command: List succeeds (no file needed)
    #[test]
    fn test_execute_list_succeeds() {
        let cli = make_cli(Commands::List);
        // List should succeed even if cache is empty
        let result = execute_command(&cli);
        assert!(result.is_ok(), "List should succeed without arguments");
    }

    /// Test execute_command: Explain without args succeeds
    #[test]
    fn test_execute_explain_no_args() {
        let cli = make_cli(Commands::Explain {
            code: None,
            file: None,
            tensor: None,
        });
        // Explain with no args should still run (shows general help)
        let result = execute_command(&cli);
        assert!(result.is_ok(), "Explain with no args should succeed");
    }

    /// Test execute_command: Explain with code succeeds
    #[test]
    fn test_execute_explain_with_code() {
        let cli = make_cli(Commands::Explain {
            code: Some("E001".to_string()),
            file: None,
            tensor: None,
        });
        let result = execute_command(&cli);
        // Should succeed even for unknown error codes (it prints "unknown error code")
        assert!(result.is_ok(), "Explain with error code should succeed");
    }

    /// Test execute_command: Tune --plan without file succeeds
    #[test]
    fn test_execute_tune_plan_no_file() {
        let cli = make_cli(Commands::Tune {
            file: None,
            method: "auto".to_string(),
            rank: None,
            vram: 16.0,
            plan: true,
            model: Some("7B".to_string()),
            freeze_base: false,
            train_data: None,
            json: false,
        });
        let result = execute_command(&cli);
        // Tune with --plan and --model should succeed without a file
        assert!(
            result.is_ok(),
            "Tune --plan --model 7B should succeed without file"
        );
    }

    /// Test execute_command: Qa with non-existent file and all skips still succeeds
    /// because QA gates are individually skipped. With no gates enabled, it just
    /// prints summary and returns Ok.
    #[test]
    fn test_execute_qa_all_skips_succeeds() {
        let cli = make_cli(Commands::Qa {
            file: PathBuf::from("/tmp/nonexistent_model_qa_test.gguf"),
            assert_tps: None,
            assert_speedup: None,
            assert_gpu_speedup: None,
            skip_golden: true,
            skip_throughput: true,
            skip_ollama: true,
            skip_gpu_speedup: true,
            skip_contract: true,
            skip_format_parity: true,
            skip_ptx_parity: true,
            safetensors_path: None,
            iterations: 1,
            warmup: 0,
            max_tokens: 1,
            json: false,
            verbose: false,
        });
        let result = execute_command(&cli);
        assert!(
            result.is_ok(),
            "Qa with all gates skipped should succeed even with non-existent file"
        );
    }

    /// Test execute_command: Qa with non-existent file and gates enabled returns error
    #[test]
    fn test_execute_qa_with_gates_file_not_found() {
        let cli = make_cli(Commands::Qa {
            file: PathBuf::from("/tmp/nonexistent_model_qa_gates_test.gguf"),
            assert_tps: None,
            assert_speedup: None,
            assert_gpu_speedup: None,
            skip_golden: false, // Gate enabled
            skip_throughput: true,
            skip_ollama: true,
            skip_gpu_speedup: true,
            skip_contract: true,
            skip_format_parity: true,
            skip_ptx_parity: true,
            safetensors_path: None,
            iterations: 1,
            warmup: 0,
            max_tokens: 1,
            json: false,
            verbose: false,
        });
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Qa with golden gate enabled should fail with non-existent file"
        );
    }

    /// Test execute_command: Import with invalid source returns error
    #[test]
    fn test_execute_import_invalid_source() {
        let cli = make_cli(Commands::Import {
            source: "/tmp/nonexistent_model_import_test.gguf".to_string(),
            output: None,
            arch: "auto".to_string(),
            quantize: None,
            strict: false,
            preserve_q4k: false,
            tokenizer: None,
            enforce_provenance: false,
        });
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Import should fail with non-existent source file"
        );
    }

    // =========================================================================
    // --chat flag logic: effective_prompt with ChatML wrapping
    // =========================================================================

    /// Test that --chat flag wraps prompt in ChatML format (verified via parse)
    #[test]
    fn test_chat_flag_chatml_wrapping_logic() {
        // We cannot call execute_command with --chat on a non-existent model
        // without error, but we can verify the ChatML wrapping logic directly.
        let prompt = "What is the meaning of life?";
        let chat = true;

        let effective_prompt = if chat {
            Some(format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                prompt
            ))
        } else {
            Some(prompt.to_string())
        };

        assert!(effective_prompt
            .as_ref()
            .expect("prompt should exist")
            .starts_with("<|im_start|>user\n"));
        assert!(effective_prompt
            .as_ref()
            .expect("prompt should exist")
            .ends_with("<|im_start|>assistant\n"));
        assert!(effective_prompt
            .as_ref()
            .expect("prompt should exist")
            .contains("What is the meaning of life?"));
    }

    /// Test that without --chat, prompt is passed through unchanged
    #[test]
    fn test_no_chat_flag_passthrough() {
        let prompt = Some("Hello world".to_string());
        let chat = false;

        let effective_prompt = if chat {
            prompt
                .as_ref()
                .map(|p| format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", p))
        } else {
            prompt.clone()
        };

        assert_eq!(effective_prompt, Some("Hello world".to_string()));
    }

    /// Test that --chat with no prompt produces None
    #[test]
    fn test_chat_flag_no_prompt() {
        let prompt: Option<String> = None;
        let chat = true;

        let effective_prompt = if chat {
            prompt
                .as_ref()
                .map(|p| format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", p))
        } else {
            prompt.clone()
        };

        assert!(effective_prompt.is_none());
    }

    // =========================================================================
    // --trace-payload shorthand logic
    // =========================================================================

    /// Test trace-payload shorthand enables trace and sets level to payload
    #[test]
    fn test_trace_payload_shorthand_logic() {
        let trace = false;
        let trace_payload = true;
        let trace_level = "basic".to_string();

        let effective_trace = trace || trace_payload;
        let effective_trace_level = if trace_payload {
            "payload"
        } else {
            trace_level.as_str()
        };

        assert!(effective_trace);
        assert_eq!(effective_trace_level, "payload");
    }

    /// Test that without --trace-payload, trace settings are preserved
    #[test]
    fn test_no_trace_payload_preserves_settings() {
        let trace = true;
        let trace_payload = false;
        let trace_level = "layer".to_string();

        let effective_trace = trace || trace_payload;
        let effective_trace_level = if trace_payload {
            "payload"
        } else {
            trace_level.as_str()
        };

        assert!(effective_trace);
        assert_eq!(effective_trace_level, "layer");
    }

    /// Test that neither trace nor trace_payload results in no trace
    #[test]
    fn test_no_trace_no_trace_payload() {
        let trace = false;
        let trace_payload = false;
        let trace_level = "basic".to_string();

        let effective_trace = trace || trace_payload;
        let effective_trace_level = if trace_payload {
            "payload"
        } else {
            trace_level.as_str()
        };

        assert!(!effective_trace);
        assert_eq!(effective_trace_level, "basic");
    }

    // =========================================================================
    // Verbose flag inheritance (local vs global)
    // =========================================================================

    /// Test that local verbose flag overrides global false
    #[test]
    fn test_verbose_local_true_global_false() {
        let local_verbose = true;
        let global_verbose = false;
        let effective_verbose = local_verbose || global_verbose;
        assert!(effective_verbose);
    }

    /// Test that global verbose flag takes effect when local is false
    #[test]
    fn test_verbose_local_false_global_true() {
        let local_verbose = false;
        let global_verbose = true;
        let effective_verbose = local_verbose || global_verbose;
        assert!(effective_verbose);
    }

    /// Test that both verbose false means not verbose
    #[test]
    fn test_verbose_both_false() {
        let local_verbose = false;
        let global_verbose = false;
        let effective_verbose = local_verbose || global_verbose;
        assert!(!effective_verbose);
    }

    /// Test that both verbose true means verbose
    #[test]
    fn test_verbose_both_true() {
        let local_verbose = true;
        let global_verbose = true;
        let effective_verbose = local_verbose || global_verbose;
        assert!(effective_verbose);
    }

    /// Test verbose inheritance end-to-end via global flag and Run command.
    /// Note: clap with `global = true` and matching short flag `-v` means
    /// the global verbose flag propagates to both the Cli struct and the
    /// Run subcommand's local verbose field.
    #[test]
    fn test_verbose_inheritance_run_global() {
        let args = vec!["apr", "--verbose", "run", "model.gguf", "--prompt", "test"];
        let cli = parse_cli(args).expect("Failed to parse");
        assert!(cli.verbose);
        match *cli.command {
            Commands::Run { verbose, .. } => {
                // With global = true, clap propagates to both levels
                // effective_verbose = local || global = always true
                let effective = verbose || cli.verbose;
                assert!(effective);
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// Test verbose inheritance end-to-end with -v after the subcommand.
    /// Because clap uses `global = true` + `short = 'v'` on both Cli and
    /// Run, -v placed after the subcommand sets the global verbose field.
    #[test]
    fn test_verbose_inheritance_run_local() {
        let args = vec!["apr", "run", "model.gguf", "--prompt", "test", "-v"];
        let cli = parse_cli(args).expect("Failed to parse");
        // -v after the subcommand still sets the global flag due to global = true
        match *cli.command {
            Commands::Run { verbose, .. } => {
                let effective = verbose || cli.verbose;
                assert!(effective, "effective verbose should be true");
            }
            _ => panic!("Expected Run command"),
        }
    }

    // =========================================================================
    // Edge case: conflicting flags (--gpu vs --no-gpu)
    // =========================================================================

    /// Test that --gpu and --no-gpu conflict (Run command)
    #[test]
    fn test_parse_run_gpu_nogpu_conflict() {
        let args = vec![
            "apr",
            "run",
            "model.gguf",
            "--prompt",
            "test",
            "--gpu",
            "--no-gpu",
        ];
        let result = parse_cli(args);
        assert!(result.is_err(), "--gpu and --no-gpu should conflict in Run");
    }

    /// Test parsing 'apr run' with --gpu flag alone
    #[test]
    fn test_parse_run_gpu_only() {
        let args = vec!["apr", "run", "model.gguf", "--prompt", "test", "--gpu"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run { gpu, no_gpu, .. } => {
                assert!(gpu);
                assert!(!no_gpu);
            }
            _ => panic!("Expected Run command"),
        }
    }

    // =========================================================================
    // Missing required args error tests
    // =========================================================================

    /// Test that 'apr serve' without FILE fails
    #[test]
    fn test_missing_serve_file() {
        let args = vec!["apr", "serve"];
        let result = parse_cli(args);
        assert!(result.is_err(), "serve requires FILE");
    }

    /// Test that 'apr diff' with only one file fails
    #[test]
    fn test_missing_diff_second_file() {
        let args = vec!["apr", "diff", "model1.apr"];
        let result = parse_cli(args);
        assert!(result.is_err(), "diff requires two files");
    }

    /// Test that 'apr export' without output fails
    #[test]
    fn test_missing_export_output() {
        let args = vec!["apr", "export", "model.apr"];
        let result = parse_cli(args);
        assert!(result.is_err(), "export requires -o/--output");
    }

    /// Test that 'apr convert' without output fails
    #[test]
    fn test_missing_convert_output() {
        let args = vec!["apr", "convert", "model.apr"];
        let result = parse_cli(args);
        assert!(result.is_err(), "convert requires -o/--output");
    }

    /// Test that 'apr merge' with fewer than 2 files fails
    #[test]
    fn test_missing_merge_files() {
        let args = vec!["apr", "merge", "model1.apr", "-o", "out.apr"];
        let result = parse_cli(args);
        assert!(result.is_err(), "merge requires at least 2 files");
    }

    /// Test that 'apr publish' without repo_id fails
    #[test]
    fn test_missing_publish_repo_id() {
        let args = vec!["apr", "publish", "/tmp/models"];
        let result = parse_cli(args);
        assert!(result.is_err(), "publish requires REPO_ID");
    }

    /// Test that 'apr pull' without model_ref fails
    #[test]
    fn test_missing_pull_model_ref() {
        let args = vec!["apr", "pull"];
        let result = parse_cli(args);
        assert!(result.is_err(), "pull requires MODEL");
    }

    /// Test that 'apr rm' without model_ref fails
    #[test]
    fn test_missing_rm_model_ref() {
        let args = vec!["apr", "rm"];
        let result = parse_cli(args);
        assert!(result.is_err(), "rm requires MODEL");
    }

    /// Test that 'apr compare-hf' without --hf fails
    #[test]
    fn test_missing_compare_hf_hf_arg() {
        let args = vec!["apr", "compare-hf", "model.apr"];
        let result = parse_cli(args);
        assert!(result.is_err(), "compare-hf requires --hf");
    }

    /// Test that 'apr canary create' without --input fails
    #[test]
    fn test_missing_canary_create_input() {
        let args = vec![
            "apr",
            "canary",
            "create",
            "model.apr",
            "--output",
            "canary.json",
        ];
        let result = parse_cli(args);
        assert!(result.is_err(), "canary create requires --input");
    }

    /// Test that 'apr canary check' without --canary fails
    #[test]
    fn test_missing_canary_check_canary() {
        let args = vec!["apr", "canary", "check", "model.apr"];
        let result = parse_cli(args);
        assert!(result.is_err(), "canary check requires --canary");
    }

    // =========================================================================
    // execute_command: contract gate integration
    // =========================================================================

    /// Test that execute_command with skip_contract=false and non-existent paths
    /// still works because non-existent paths are skipped in validate_model_contract
    #[test]
    fn test_execute_with_contract_gate_nonexistent() {
        let cli = Cli {
            command: Box::new(Commands::Inspect {
                file: PathBuf::from("/tmp/nonexistent_contract_test.apr"),
                vocab: false,
                filters: false,
                weights: false,
                json: false,
            }),
            json: false,
            verbose: false,
            quiet: false,
            offline: false,
            skip_contract: false, // Contract enabled, but paths don't exist
        };
        // The contract gate should pass (non-existent paths are skipped),
        // but the command itself should fail (file not found)
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Should still fail from command execution, not contract"
        );
    }

    /// Test that execute_command dispatches List even with contract enabled
    #[test]
    fn test_execute_list_with_contract_enabled() {
        let cli = Cli {
            command: Box::new(Commands::List),
            json: false,
            verbose: false,
            quiet: false,
            offline: false,
            skip_contract: false, // Contract enabled
        };
        let result = execute_command(&cli);
        assert!(result.is_ok(), "List should succeed with contract enabled");
    }

    // =========================================================================
    // Rosetta command execution error paths
    // =========================================================================

    /// Test execute_command: Rosetta inspect with non-existent file returns error
    #[test]
    fn test_execute_rosetta_inspect_file_not_found() {
        let cli = make_cli(Commands::Rosetta {
            action: RosettaCommands::Inspect {
                file: PathBuf::from("/tmp/nonexistent_rosetta_inspect.gguf"),
                hexdump: false,
                json: false,
            },
        });
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Rosetta inspect should fail with non-existent file"
        );
    }

    /// Test execute_command: Rosetta convert with non-existent source returns error
    #[test]
    fn test_execute_rosetta_convert_file_not_found() {
        let cli = make_cli(Commands::Rosetta {
            action: RosettaCommands::Convert {
                source: PathBuf::from("/tmp/nonexistent_rosetta_convert.gguf"),
                target: PathBuf::from("/tmp/out.safetensors"),
                quantize: None,
                verify: false,
                json: false,
                tokenizer: None,
            },
        });
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Rosetta convert should fail with non-existent source"
        );
    }

    /// Test execute_command: Rosetta fingerprint with non-existent file returns error
    #[test]
    fn test_execute_rosetta_fingerprint_file_not_found() {
        let cli = make_cli(Commands::Rosetta {
            action: RosettaCommands::Fingerprint {
                model: PathBuf::from("/tmp/nonexistent_rosetta_fingerprint.gguf"),
                model_b: None,
                output: None,
                filter: None,
                verbose: false,
                json: false,
            },
        });
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Rosetta fingerprint should fail with non-existent file"
        );
    }

    /// Test execute_command: Bench with non-existent file returns error
    #[test]
    fn test_execute_bench_file_not_found() {
        let cli = make_cli(Commands::Bench {
            file: PathBuf::from("/tmp/nonexistent_model_bench_test.gguf"),
            warmup: 1,
            iterations: 1,
            max_tokens: 1,
            prompt: None,
            fast: false,
            brick: None,
        });
        let result = execute_command(&cli);
        assert!(result.is_err(), "Bench should fail with non-existent file");
    }

    /// Test execute_command: Eval with non-existent file returns error
    #[test]
    fn test_execute_eval_file_not_found() {
        let cli = make_cli(Commands::Eval {
            file: PathBuf::from("/tmp/nonexistent_model_eval_test.gguf"),
            dataset: "wikitext-2".to_string(),
            text: None,
            max_tokens: 32,
            threshold: 20.0,
        });
        let result = execute_command(&cli);
        assert!(result.is_err(), "Eval should fail with non-existent file");
    }

    /// Test execute_command: Profile with non-existent file returns error
    #[test]
    fn test_execute_profile_file_not_found() {
        let cli = make_cli(Commands::Profile {
            file: PathBuf::from("/tmp/nonexistent_model_profile_test.apr"),
            granular: false,
            format: "human".to_string(),
            focus: None,
            detect_naive: false,
            threshold: 10.0,
            compare_hf: None,
            energy: false,
            perf_grade: false,
            callgraph: false,
            fail_on_naive: false,
            output: None,
            ci: false,
            assert_throughput: None,
            assert_p99: None,
            assert_p50: None,
            warmup: 3,
            measure: 10,
            tokens: 32,
            ollama: false,
            no_gpu: false,
        });
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Profile should fail with non-existent file"
        );
    }

    /// Test execute_command: CompareHf with non-existent file returns error
    #[test]
    fn test_execute_compare_hf_file_not_found() {
        let cli = make_cli(Commands::CompareHf {
            file: PathBuf::from("/tmp/nonexistent_model_compare_hf_test.apr"),
            hf: "openai/whisper-tiny".to_string(),
            tensor: None,
            threshold: 1e-5,
            json: false,
        });
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "CompareHf should fail with non-existent file"
        );
    }

    /// Test execute_command: Canary check with non-existent file returns error
    #[test]
    fn test_execute_canary_check_file_not_found() {
        let cli = make_cli(Commands::Canary {
            command: CanaryCommands::Check {
                file: PathBuf::from("/tmp/nonexistent_canary_check.apr"),
                canary: PathBuf::from("/tmp/nonexistent_canary.json"),
            },
        });
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Canary check should fail with non-existent file"
        );
    }

    /// Test execute_command: Publish with non-existent directory returns error
    #[test]
    fn test_execute_publish_dir_not_found() {
        let cli = make_cli(Commands::Publish {
            directory: PathBuf::from("/tmp/nonexistent_publish_dir_test"),
            repo_id: "test/test".to_string(),
            model_name: None,
            license: "mit".to_string(),
            pipeline_tag: "text-generation".to_string(),
            library_name: None,
            tags: None,
            message: None,
            dry_run: true, // Use dry_run to avoid actual upload
        });
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Publish should fail with non-existent directory"
        );
    }

    // =========================================================================
    // Default value verification tests
    // =========================================================================

    /// Test Run command defaults
    #[test]
    fn test_parse_run_defaults() {
        let args = vec!["apr", "run", "model.gguf"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run {
                max_tokens,
                stream,
                format,
                no_gpu,
                gpu,
                offline,
                benchmark,
                trace,
                trace_payload,
                trace_verbose,
                trace_level,
                profile,
                chat,
                verbose,
                prompt,
                input,
                language,
                task,
                trace_steps,
                trace_output,
                ..
            } => {
                assert_eq!(max_tokens, 32);
                assert!(!stream);
                assert_eq!(format, "text");
                assert!(!no_gpu);
                assert!(!gpu);
                assert!(!offline);
                assert!(!benchmark);
                assert!(!trace);
                assert!(!trace_payload);
                assert!(!trace_verbose);
                assert_eq!(trace_level, "basic");
                assert!(!profile);
                assert!(!chat);
                assert!(!verbose);
                assert!(prompt.is_none());
                assert!(input.is_none());
                assert!(language.is_none());
                assert!(task.is_none());
                assert!(trace_steps.is_none());
                assert!(trace_output.is_none());
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// Test Serve command defaults
    #[test]
    fn test_parse_serve_defaults() {
        let args = vec!["apr", "serve", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Serve {
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
                ..
            } => {
                assert_eq!(port, 8080);
                assert_eq!(host, "127.0.0.1");
                assert!(!no_cors);
                assert!(!no_metrics);
                assert!(!no_gpu);
                assert!(!gpu);
                assert!(!batch);
                assert!(!trace);
                assert_eq!(trace_level, "basic");
                assert!(!profile);
            }
            _ => panic!("Expected Serve command"),
        }
    }

    /// Test Bench command defaults
    #[test]
    fn test_parse_bench_defaults() {
        let args = vec!["apr", "bench", "model.gguf"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Bench {
                warmup,
                iterations,
                max_tokens,
                prompt,
                fast,
                brick,
                ..
            } => {
                assert_eq!(warmup, 3);
                assert_eq!(iterations, 5);
                assert_eq!(max_tokens, 32);
                assert!(prompt.is_none());
                assert!(!fast);
                assert!(brick.is_none());
            }
            _ => panic!("Expected Bench command"),
        }
    }

    /// Test Cbtop command defaults
    #[test]
    fn test_parse_cbtop_defaults() {
        let args = vec!["apr", "cbtop"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
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
                assert!(model.is_none());
                assert!(attach.is_none());
                assert!(model_path.is_none());
                assert!(!headless);
                assert!(!json);
                assert!(output.is_none());
                assert!(!ci);
                assert!(throughput.is_none());
                assert!(brick_score.is_none());
                assert_eq!(warmup, 10);
                assert_eq!(iterations, 100);
                assert!(!speculative);
                assert_eq!(speculation_k, 4);
                assert!(draft_model.is_none());
                assert_eq!(concurrent, 1);
                assert!(!simulated);
            }
            _ => panic!("Expected Cbtop command"),
        }
    }

    /// Test Profile command defaults
    #[test]
    fn test_parse_profile_defaults() {
        let args = vec!["apr", "profile", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Profile {
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
                ..
            } => {
                assert!(!granular);
                assert_eq!(format, "human");
                assert!(focus.is_none());
                assert!(!detect_naive);
                assert!((threshold - 10.0).abs() < f64::EPSILON);
                assert!(compare_hf.is_none());
                assert!(!energy);
                assert!(!perf_grade);
                assert!(!callgraph);
                assert!(!fail_on_naive);
                assert!(output.is_none());
                assert!(!ci);
                assert!(assert_throughput.is_none());
                assert!(assert_p99.is_none());
                assert!(assert_p50.is_none());
                assert_eq!(warmup, 3);
                assert_eq!(measure, 10);
            }
            _ => panic!("Expected Profile command"),
        }
    }

    /// Test Qa command defaults
    #[test]
    fn test_parse_qa_defaults() {
        let args = vec!["apr", "qa", "model.gguf"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Qa {
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
                ..
            } => {
                assert!(assert_tps.is_none());
                assert!(assert_speedup.is_none());
                assert!(assert_gpu_speedup.is_none());
                assert!(!skip_golden);
                assert!(!skip_throughput);
                assert!(!skip_ollama);
                assert!(!skip_gpu_speedup);
                assert!(!skip_contract);
                assert!(!skip_format_parity);
                assert!(safetensors_path.is_none());
                assert_eq!(iterations, 10);
                assert_eq!(warmup, 3);
                assert_eq!(max_tokens, 32);
                assert!(!json);
                assert!(!verbose);
            }
            _ => panic!("Expected Qa command"),
        }
    }

    /// Test Chat command defaults
    #[test]
    fn test_parse_chat_defaults() {
        let args = vec!["apr", "chat", "model.gguf"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Chat {
                temperature,
                top_p,
                max_tokens,
                system,
                inspect,
                no_gpu,
                gpu,
                trace,
                trace_steps,
                trace_verbose,
                trace_output,
                trace_level,
                profile,
                ..
            } => {
                assert!((temperature - 0.7).abs() < f32::EPSILON);
                assert!((top_p - 0.9).abs() < f32::EPSILON);
                assert_eq!(max_tokens, 512);
                assert!(system.is_none());
                assert!(!inspect);
                assert!(!no_gpu);
                assert!(!gpu);
                assert!(!trace);
                assert!(trace_steps.is_none());
                assert!(!trace_verbose);
                assert!(trace_output.is_none());
                assert_eq!(trace_level, "basic");
                assert!(!profile);
            }
            _ => panic!("Expected Chat command"),
        }
    }
}
