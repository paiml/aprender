//! apr - APR Model Operations CLI
//!
//! Toyota Way: Genchi Genbutsu (Go and See)
//! See the actual model data, not abstractions.
//!
//! Usage:
//!   apr inspect model.apr           # Inspect model metadata
//!   apr debug model.apr             # Simple debugging output
//!   apr debug model.apr --drama     # Theatrical debugging
//!   apr validate model.apr          # Validate model integrity
//!   apr diff model1.apr model2.apr  # Compare models
//!   apr tensors model.apr           # List tensor names and shapes
//!   apr tensors model.apr --stats   # Show tensor statistics
//!   apr trace model.apr             # Layer-by-layer analysis
//!   apr probar model.apr -o ./out   # Export for probar visual testing

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;

mod commands;
mod error;
#[cfg(feature = "inference")]
pub mod federation;
mod output;

use commands::{
    bench, canary, canary::CanaryCommands, cbtop, chat, compare_hf, convert, debug, diff, eval,
    explain, export, flow, hex, import, inspect, lint, merge, probar, profile, pull, run, serve,
    showcase, tensors, trace, tree, tui, validate,
};

/// apr - APR Model Operations Tool
///
/// Inspect, debug, and manage .apr model files.
/// Toyota Way: Genchi Genbutsu - Go and see the actual data.
#[derive(Parser)]
#[command(name = "apr")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Output as JSON
    #[arg(long, global = true)]
    json: bool,

    /// Verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Quiet mode (errors only)
    #[arg(short, long, global = true)]
    quiet: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Run model directly (auto-download, cache, execute)
    Run {
        /// Model source: local path, hf://org/repo, or URL
        #[arg(value_name = "SOURCE")]
        source: String,

        /// Input file (audio, text, etc.)
        #[arg(short, long)]
        input: Option<PathBuf>,

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
        #[arg(long)]
        no_gpu: bool,

        /// Offline mode: block all network access (Sovereign AI compliance)
        /// Per Section 9.2: "apr run --offline is mandatory for production"
        #[arg(long)]
        offline: bool,
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

        /// Limit number of tensors shown
        #[arg(long, default_value = "100")]
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

        /// Output .apr file path
        #[arg(short, long)]
        output: PathBuf,

        /// Model architecture (whisper, llama, bert, auto)
        #[arg(long, default_value = "auto")]
        arch: String,

        /// Quantization (int8, int4, fp16)
        #[arg(long)]
        quantize: Option<String>,

        /// Force import even if validation fails
        #[arg(long)]
        force: bool,
    },

    /// Download GGUF model from HuggingFace (for LLM inference via realizar)
    Pull {
        /// Repository: org/repo (e.g., bartowski/Qwen2.5-Coder-32B-Instruct-GGUF)
        #[arg(value_name = "REPO")]
        repo: String,

        /// Quantization variant (Q4_K_M, Q4_K_S, Q5_K_M, Q6_K, Q8_0, F16)
        #[arg(long, default_value = "Q4_K_M")]
        quant: String,

        /// Output directory
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Force download even if file exists
        #[arg(long)]
        force: bool,
    },

    /// Convert/optimize model
    Convert {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Quantize to format (int8, int4, fp16)
        #[arg(long)]
        quantize: Option<String>,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,
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
    ///
    /// Real-time visualization of brick-level timing during inference.
    /// Per spec: docs/specifications/qwen2.5-coder-showcase-demo.md §6
    ///
    /// Headless mode for CI:
    ///   apr cbtop --headless --json --output results.json
    ///   apr cbtop --headless --ci --throughput 400 --brick-score 90
    Cbtop {
        /// Model name (e.g., qwen2.5-coder-1.5b)
        #[arg(long)]
        model: Option<String>,

        /// Attach to running realizar process
        #[arg(long)]
        attach: Option<String>,

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

    /// Hex dump tensor data (GH-122)
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

    /// Model architecture tree view (GH-122)
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

    /// Data flow visualization (GH-122)
    Flow {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Filter by layer pattern
        #[arg(long)]
        layer: Option<String>,

        /// Component to visualize: full, encoder, decoder, cross_attn, self_attn, ffn
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

        /// Benchmark specific brick (rms_norm, qkv, rope, attn, o_proj, ffn, layer)
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

    /// Deep profiling with Roofline analysis (Williams et al., 2009)
    Profile {
        /// Path to model file (.apr, .safetensors, .gguf)
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Layer-by-layer granular analysis
        #[arg(long)]
        granular: bool,

        /// Output format (human, json, flamegraph)
        #[arg(long, default_value = "human")]
        format: String,

        /// Focus on specific operation (attention, mlp, matmul, embedding)
        #[arg(long)]
        focus: Option<String>,

        /// Detect naive implementations (< threshold GFLOPS)
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

        /// Compute performance grade (Dean & Ghemawat, 2025)
        #[arg(long)]
        perf_grade: bool,

        /// Show call graph (Graham et al., 1982)
        #[arg(long)]
        callgraph: bool,

        /// Exit non-zero if naive implementation detected
        #[arg(long)]
        fail_on_naive: bool,
    },

    /// Qwen2.5-Coder showcase demo (spec: qwen2.5-coder-showcase-demo.md)
    Showcase {
        /// Run all steps with auto-verification
        #[arg(long)]
        auto_verify: bool,

        /// Run specific step: import, gguf, convert, apr, bench, visualize, all
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
}

/// Execute the CLI command and return the result.
#[allow(clippy::too_many_lines)] // CLI dispatch naturally has many command branches
fn execute_command(cli: &Cli) -> Result<(), error::CliError> {
    match &cli.command {
        Commands::Run {
            source,
            input,
            stream,
            language,
            task,
            format,
            no_gpu,
            offline,
        } => run::run(
            source,
            input.as_deref(),
            *stream,
            language.as_deref(),
            task.as_deref(),
            format,
            *no_gpu,
            *offline,
        ),

        Commands::Serve {
            file,
            port,
            host,
            no_cors,
            no_metrics,
            no_gpu,
        } => {
            let config = serve::ServerConfig {
                port: *port,
                host: host.clone(),
                cors: !no_cors,
                metrics: !no_metrics,
                no_gpu: *no_gpu,
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
            json,
        } => diff::run(file1, file2, *weights, *json || cli.json),

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
            force,
        } => import::run(
            source,
            output,
            Some(arch.as_str()),
            quantize.as_deref(),
            *force,
        ),
        Commands::Pull {
            repo,
            quant,
            output,
            force,
        } => pull::run(repo, Some(quant.as_str()), output.as_deref(), *force),
        Commands::Convert {
            file,
            quantize,
            output,
        } => convert::run(file, quantize.as_deref(), output),
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
            headless,
            json,
            output,
            ci,
            throughput,
            brick_score,
            warmup,
            iterations,
        } => cbtop::run(cbtop::CbtopConfig {
            model: model.as_deref().map(String::from),
            attach: attach.as_deref().map(String::from),
            headless: *headless,
            json: *json,
            output: output.clone(),
            ci: *ci,
            throughput_threshold: *throughput,
            brick_score_threshold: *brick_score,
            warmup: *warmup,
            iterations: *iterations,
        }),

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
        } => chat::run(
            file,
            *temperature,
            *top_p,
            *max_tokens,
            system.as_deref(),
            *inspect,
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
        } => {
            let output_format = format.parse().unwrap_or(profile::OutputFormat::Human);
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
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match execute_command(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            e.exit_code()
        }
    }
}
